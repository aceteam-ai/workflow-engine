# tests/test_attempt.py
"""
Tests for the `attempt` error boundary and scoped cancellation (#201):
`AttemptNode`, `OkNode`, `on_node_cancelled`, and `on_boundary_error`.

Section numbers below refer to the design's test plan. Parametrized over the
`algorithm` fixture (`tests/conftest.py`) except where a test is specific to
one executor or one error-handling mode.

All helper nodes take `Empty` input unless noted, so a node with no real
upstream dependency is part of the deterministic *initial* ready set
(`ValidatedWorkflow.get_initial_ready_nodes`, which iterates `self.nodes` in
list order) rather than being discovered later via `get_ready_successors`
(which iterates a hash-ordered set). Sibling execution order within a single
pass is controlled by `inner_nodes` list position: `TopologicalExecutionAlgorithm`
pops the ready dict with `popitem()` (LIFO), so a node placed *last* in
`inner_nodes` runs *first*.
"""

import asyncio
from datetime import timedelta
from typing import ClassVar, Type

import pytest
from overrides import override

from workflow_engine import (
    CancelReason,
    Data,
    DataMapping,
    Edge,
    Empty,
    ErrorClass,
    ExecutionAlgorithm,
    ExecutionContext,
    IntegerValue,
    Node,
    NodeTypeInfo,
    Params,
    Result,
    SequenceValue,
    ShouldRetry,
    ShouldYield,
    StringValue,
    ValidatedWorkflow,
    Workflow,
    WorkflowEngine,
    WorkflowException,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core.values import DataValue, ResultError, get_field_annotations
from workflow_engine.execution.parallel import (
    ErrorHandlingMode,
    ParallelExecutionAlgorithm,
)
from workflow_engine.execution.topological import TopologicalExecutionAlgorithm
from workflow_engine.nodes import (
    AttemptNode,
    ForEachNode,
    PartitionNode,
    UnwrapOrNode,
)

# ---------------------------------------------------------------------------
# Test helper nodes.
# ---------------------------------------------------------------------------


class ProbeOutput(Data):
    value: StringValue


class FailingParams(Params):
    message: StringValue
    level: StringValue = StringValue("user")  # "user" or "operator"
    error_class: StringValue = StringValue("")  # empty means "not set"
    # -1 = always fails; N >= 0 = fails the first N calls, then succeeds
    # (N=0 means "always succeeds").
    fail_count: IntegerValue = IntegerValue(-1)


class FailingNode(Node[Empty, ProbeOutput, FailingParams]):
    """Fails the first fail_count calls (or forever, if fail_count is -1)."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="AttemptTestFailing",
        description="Test helper that fails, redactably.",
        version="1.0.0",
        parameter_type=FailingParams,
    )

    calls: ClassVar[dict[str, int]] = {}

    @classmethod
    @override
    def static_input_type(cls) -> Type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> Type[ProbeOutput]:
        return ProbeOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[ProbeOutput],
        input: Empty,
    ) -> ProbeOutput:
        n = FailingNode.calls.get(self.id, 0) + 1
        FailingNode.calls[self.id] = n
        fail_count = self.params.fail_count.root
        if fail_count >= 0 and n > fail_count:
            return ProbeOutput(value=StringValue("recovered"))
        kwargs: dict = {}
        if self.params.error_class.root:
            kwargs["error_class"] = ErrorClass(self.params.error_class.root)
        message = self.params.message.root
        if self.params.level.root == "operator":
            raise WorkflowException.for_operator(message, **kwargs)
        raise WorkflowException.for_user(message, **kwargs)


class YieldOnceNode(Node[Empty, ProbeOutput, Params]):
    """Yields on the first call for a given id, succeeds on the second."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="AttemptTestYieldOnce",
        description="Test helper that yields once then succeeds.",
        version="1.0.0",
        parameter_type=Params,
    )

    calls: ClassVar[dict[str, int]] = {}

    @classmethod
    @override
    def static_input_type(cls) -> Type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> Type[ProbeOutput]:
        return ProbeOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[ProbeOutput],
        input: Empty,
    ) -> ProbeOutput:
        n = YieldOnceNode.calls.get(self.id, 0) + 1
        YieldOnceNode.calls[self.id] = n
        if n == 1:
            raise ShouldYield(f"waiting: {self.id}")
        return ProbeOutput(value=StringValue("resumed"))


class DownstreamEchoNode(Node[ProbeOutput, ProbeOutput, Params]):
    """Passes its input through, recording which ids actually ran."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="AttemptTestDownstreamEcho",
        description="Test helper that records that it ran.",
        version="1.0.0",
        parameter_type=Params,
    )

    ran: ClassVar[set[str]] = set()

    @classmethod
    @override
    def static_input_type(cls) -> Type[ProbeOutput]:
        return ProbeOutput

    @classmethod
    @override
    def static_output_type(cls) -> Type[ProbeOutput]:
        return ProbeOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[ProbeOutput],
        output_type: Type[ProbeOutput],
        input: ProbeOutput,
    ) -> ProbeOutput:
        DownstreamEchoNode.ran.add(self.id)
        return input


class FailOnceRetryParams(Params):
    fail_count: IntegerValue
    backoff_ms: IntegerValue = IntegerValue(10)


class FailOnceRetryNode(Node[Empty, ProbeOutput, FailOnceRetryParams]):
    """Raises ShouldRetry fail_count times, then succeeds."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="AttemptTestFailOnceRetry",
        description="Test helper that raises ShouldRetry N times then succeeds.",
        version="1.0.0",
        parameter_type=FailOnceRetryParams,
    )

    calls: ClassVar[dict[str, int]] = {}

    @classmethod
    @override
    def static_input_type(cls) -> Type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> Type[ProbeOutput]:
        return ProbeOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[ProbeOutput],
        input: Empty,
    ) -> ProbeOutput:
        n = FailOnceRetryNode.calls.get(self.id, 0) + 1
        FailOnceRetryNode.calls[self.id] = n
        if n <= self.params.fail_count.root:
            raise ShouldRetry.for_user(
                f"transient failure {n}",
                node=self,
                backoff=timedelta(milliseconds=self.params.backoff_ms.root),
            )
        return ProbeOutput(value=StringValue("retried ok"))


class AttemptSlowParams(Params):
    delay_ms: IntegerValue


class AttemptSlowNode(Node[Empty, ProbeOutput, AttemptSlowParams]):
    """Sleeps, then completes; records completion to prove it was not cancelled."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="AttemptTestSlow",
        description="Test helper that sleeps before completing.",
        version="1.0.0",
        parameter_type=AttemptSlowParams,
    )

    completed: ClassVar[set[str]] = set()

    @classmethod
    @override
    def static_input_type(cls) -> Type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> Type[ProbeOutput]:
        return ProbeOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[ProbeOutput],
        input: Empty,
    ) -> ProbeOutput:
        await asyncio.sleep(self.params.delay_ms.root / 1000)
        AttemptSlowNode.completed.add(self.id)
        return ProbeOutput(value=StringValue("slow ok"))


@pytest.fixture(autouse=True)
def reset_helper_state():
    FailingNode.calls = {}
    YieldOnceNode.calls = {}
    DownstreamEchoNode.ran = set()
    FailOnceRetryNode.calls = {}
    AttemptSlowNode.completed = set()
    yield


class RecordingContext(InMemoryExecutionContext):
    """Records every hook call, keyed by node id, plus one merged, ordered log."""

    def __init__(self):
        super().__init__()
        self.started: list[str] = []
        self.finished: list[str] = []
        self.errored: list[str] = []
        self.yielded: list[str] = []
        self.retried: list[str] = []
        self.expanded: list[str] = []
        self.cancelled: list[tuple[str, CancelReason, str]] = []
        self.boundary_errors: list[tuple[str, DataMapping]] = []
        self.sequence: list[str] = []

    @override
    async def on_node_start(
        self,
        *,
        node: Node,
        input_type: Type[Data],
        output_type: Type[Data],
        input: DataMapping,
    ) -> DataMapping | Workflow | None:
        self.started.append(node.id)
        self.sequence.append(f"start:{node.id}")
        return await super().on_node_start(
            node=node, input_type=input_type, output_type=output_type, input=input
        )

    @override
    async def on_node_finish(
        self,
        *,
        node: Node,
        input_type: Type[Data],
        output_type: Type[Data],
        input: DataMapping,
        output: DataMapping,
    ) -> DataMapping:
        self.finished.append(node.id)
        self.sequence.append(f"finish:{node.id}")
        return await super().on_node_finish(
            node=node,
            input_type=input_type,
            output_type=output_type,
            input=input,
            output=output,
        )

    @override
    async def on_node_error(
        self,
        *,
        node: Node,
        input_type: Type[Data],
        output_type: Type[Data],
        input: DataMapping,
        exception: WorkflowException,
    ) -> WorkflowException | DataMapping:
        self.errored.append(node.id)
        self.sequence.append(f"error:{node.id}")
        return await super().on_node_error(
            node=node,
            input_type=input_type,
            output_type=output_type,
            input=input,
            exception=exception,
        )

    @override
    async def on_node_yield(
        self,
        *,
        node: Node,
        input_type: Type[Data],
        output_type: Type[Data],
        input: DataMapping,
        exception: ShouldYield,
    ) -> None:
        self.yielded.append(node.id)
        self.sequence.append(f"yield:{node.id}")
        return await super().on_node_yield(
            node=node,
            input_type=input_type,
            output_type=output_type,
            input=input,
            exception=exception,
        )

    @override
    async def on_node_retry(
        self,
        *,
        node: Node,
        input_type: Type[Data],
        output_type: Type[Data],
        input: DataMapping,
        exception: ShouldRetry,
        attempt: int,
    ) -> None:
        self.retried.append(node.id)
        self.sequence.append(f"retry:{node.id}")
        return await super().on_node_retry(
            node=node,
            input_type=input_type,
            output_type=output_type,
            input=input,
            exception=exception,
            attempt=attempt,
        )

    @override
    async def on_node_expand(
        self,
        *,
        node: Node,
        input_type: Type[Data],
        output_type: Type[Data],
        input: DataMapping,
        workflow: ValidatedWorkflow,
    ) -> ValidatedWorkflow:
        self.expanded.append(node.id)
        self.sequence.append(f"expand:{node.id}")
        return await super().on_node_expand(
            node=node,
            input_type=input_type,
            output_type=output_type,
            input=input,
            workflow=workflow,
        )

    @override
    async def on_node_cancelled(
        self,
        *,
        node: Node,
        input_type: Type[Data],
        output_type: Type[Data],
        input: DataMapping | None,
        boundary_id: str,
        reason: CancelReason,
        cause: WorkflowException,
    ) -> None:
        self.cancelled.append((node.id, reason, boundary_id))
        self.sequence.append(f"cancel:{node.id}:{reason}")
        return await super().on_node_cancelled(
            node=node,
            input_type=input_type,
            output_type=output_type,
            input=input,
            boundary_id=boundary_id,
            reason=reason,
            cause=cause,
        )

    @override
    async def on_boundary_error(
        self,
        *,
        node: Node,
        input_type: Type[Data],
        output_type: Type[Data],
        input: DataMapping,
        error: ResultError,
        output: DataMapping,
        cause: WorkflowException,
    ) -> DataMapping:
        self.boundary_errors.append((node.id, input))
        self.sequence.append(f"boundary_error:{node.id}")
        return await super().on_boundary_error(
            node=node,
            input_type=input_type,
            output_type=output_type,
            input=input,
            error=error,
            output=output,
            cause=cause,
        )


# ---------------------------------------------------------------------------
# Helpers to build attempt(w) workflows.
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


async def _build_attempt_workflow(
    engine: WorkflowEngine,
    *,
    inner_nodes: list[Node],
    edges: list[Edge],
    output_fields: dict,
    attempt_id: str = "attempt",
) -> Workflow:
    """Build attempt(w) as a single-node workflow: {} -> {result: Result[B]}."""
    w = Workflow(
        input_node=engine.create_input_node(),
        inner_nodes=inner_nodes,
        output_node=engine.create_output_node(**output_fields),
        edges=edges,
    )
    return await engine.build_single_node_workflow(
        AttemptNode,
        node_id=attempt_id,
        params={"workflow": w},
    )


def edge(source_id: str, source_key: str, target_id: str, target_key: str) -> Edge:
    return Edge(
        source_id=source_id,
        source_key=source_key,
        target_id=target_id,
        target_key=target_key,
    )


async def _run(
    engine: WorkflowEngine, workflow: Workflow, context: InMemoryExecutionContext
):
    return await engine.execute(context=context, workflow=workflow, input={})


def as_result(value: object) -> Result:
    """Narrow a DataMapping value to Result[T] for pyright (and at runtime)."""
    assert isinstance(value, Result)
    return value


def as_sequence(value: object) -> SequenceValue:
    """Narrow a DataMapping value to SequenceValue[T] for pyright (and at runtime)."""
    assert isinstance(value, SequenceValue)
    return value


# ---------------------------------------------------------------------------
# 1. Ok arm.
# ---------------------------------------------------------------------------


class TestOkArm:
    @pytest.mark.asyncio
    async def test_ok_arm_and_hook_ids(self, algorithm: ExecutionAlgorithm):
        engine = WorkflowEngine(execution_algorithm=algorithm)
        succeed = engine.create_node(
            FailingNode,
            id="succeed",
            params=dict(message=StringValue(""), fail_count=IntegerValue(0)),
        )
        workflow = await _build_attempt_workflow(
            engine,
            inner_nodes=[succeed],
            edges=[edge("succeed", "value", "output", "final")],
            output_fields={"final": StringValue},
        )
        context = RecordingContext()
        result = await _run(engine, workflow, context)

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        result_value = as_result(result.output["result"])
        assert result_value.is_ok()

        assert {
            "attempt/input",
            "attempt/succeed",
            "attempt/ok",
            "attempt/output",
        }.issubset(set(context.started))
        assert context.boundary_errors == []
        assert context.cancelled == []


# ---------------------------------------------------------------------------
# 2. Err arm, single failure.
# ---------------------------------------------------------------------------


class TestErrArm:
    @pytest.mark.asyncio
    async def test_user_level_message_visible(self, algorithm: ExecutionAlgorithm):
        engine = WorkflowEngine(execution_algorithm=algorithm)
        failing = engine.create_node(
            FailingNode,
            id="boom",
            params=dict(
                message=StringValue("visible to user"), level=StringValue("user")
            ),
        )
        workflow = await _build_attempt_workflow(
            engine,
            inner_nodes=[failing],
            edges=[edge("boom", "value", "output", "final")],
            output_fields={"final": StringValue},
        )
        context = RecordingContext()
        result = await _run(engine, workflow, context)

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.errors.count == 0
        result_value = as_result(result.output["result"])
        assert result_value.is_err()
        error = result_value.unwrap_err()
        assert error.node_id.root == "attempt/boom"
        assert error.message.root == "visible to user"
        assert error.error_class.root == ErrorClass.SYSTEMIC
        assert len(context.boundary_errors) == 1
        assert context.boundary_errors[0][0] == "attempt"

    @pytest.mark.asyncio
    async def test_operator_level_message_redacted(self, algorithm: ExecutionAlgorithm):
        engine = WorkflowEngine(execution_algorithm=algorithm)
        failing = engine.create_node(
            FailingNode,
            id="boom",
            params=dict(
                message=StringValue("internal secret detail"),
                level=StringValue("operator"),
            ),
        )
        workflow = await _build_attempt_workflow(
            engine,
            inner_nodes=[failing],
            edges=[edge("boom", "value", "output", "final")],
            output_fields={"final": StringValue},
        )
        context = RecordingContext()
        result = await _run(engine, workflow, context)

        error = as_result(result.output["result"]).unwrap_err()
        assert error.message.root == "An internal error occurred"
        assert "internal secret detail" not in error.message.root

    @pytest.mark.asyncio
    async def test_explicit_error_class(self, algorithm: ExecutionAlgorithm):
        engine = WorkflowEngine(execution_algorithm=algorithm)
        failing = engine.create_node(
            FailingNode,
            id="boom",
            params=dict(
                message=StringValue("timed out"),
                error_class=StringValue(ErrorClass.TIMEOUT.value),
            ),
        )
        workflow = await _build_attempt_workflow(
            engine,
            inner_nodes=[failing],
            edges=[edge("boom", "value", "output", "final")],
            output_fields={"final": StringValue},
        )
        context = RecordingContext()
        result = await _run(engine, workflow, context)
        error = as_result(result.output["result"]).unwrap_err()
        assert error.error_class.root == ErrorClass.TIMEOUT

    @pytest.mark.asyncio
    async def test_continue_mode_parallel_no_run_level_error(self):
        """A boundary-contained error never reaches run-level errors, even in CONTINUE mode."""
        algorithm = ParallelExecutionAlgorithm(
            error_handling=ErrorHandlingMode.CONTINUE
        )
        engine = WorkflowEngine(execution_algorithm=algorithm)
        failing = engine.create_node(
            FailingNode, id="boom", params=dict(message=StringValue("nope"))
        )
        workflow = await _build_attempt_workflow(
            engine,
            inner_nodes=[failing],
            edges=[edge("boom", "value", "output", "final")],
            output_fields={"final": StringValue},
        )
        context = RecordingContext()
        result = await _run(engine, workflow, context)
        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.errors.count == 0
        assert as_result(result.output["result"]).is_err()


# ---------------------------------------------------------------------------
# 3 + 11. Ids stay flat; for_each(attempt(w)) end to end.
# ---------------------------------------------------------------------------


class ConditionalFailInput(Data):
    x: StringValue


class ConditionalFailNode(Node[ConditionalFailInput, ProbeOutput, Params]):
    """Fails iff its input is the literal string 'FAIL'; otherwise echoes it."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="AttemptTestConditionalFail",
        description="Test helper that fails on a marker input value.",
        version="1.0.0",
        parameter_type=Params,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[ConditionalFailInput]:
        return ConditionalFailInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[ProbeOutput]:
        return ProbeOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[ConditionalFailInput],
        output_type: Type[ProbeOutput],
        input: ConditionalFailInput,
    ) -> ProbeOutput:
        if input.x.root == "FAIL":
            raise WorkflowException.for_user(f"failing on element {input.x.root}")
        return ProbeOutput(value=input.x)


class TestForEachAttemptEndToEnd:
    @pytest.mark.asyncio
    async def test_flat_ids_and_partial_failure(self, algorithm: ExecutionAlgorithm):
        engine = WorkflowEngine(execution_algorithm=algorithm)
        cond = engine.create_node(ConditionalFailNode, id="cond")
        w = Workflow(
            input_node=engine.create_input_node(x=StringValue),
            inner_nodes=[cond],
            output_node=engine.create_output_node(final=StringValue),
            edges=[
                edge("input", "x", "cond", "x"),
                edge("cond", "value", "output", "final"),
            ],
        )
        attempted = await engine.build_single_node_workflow(
            AttemptNode, node_id="attempt", params={"workflow": w}
        )
        validated_attempted = await engine.validate(attempted)
        # attempted : {x: StringValue} -> {result: Result[StringValue]}, a
        # single output field, so ForEachNode's own element-collapsing rule
        # (mirrored by single_field_or_wrapped) yields Result[StringValue].
        result_element_type = Result[StringValue]

        for_each = engine.create_node(
            ForEachNode, id="for_each", params={"workflow": validated_attempted}
        )
        outer = Workflow(
            input_node=engine.create_input_node(items=SequenceValue[StringValue]),
            inner_nodes=[for_each],
            output_node=engine.create_output_node(
                results=SequenceValue[result_element_type]
            ),
            edges=[
                edge("input", "items", "for_each", "sequence"),
                edge("for_each", "sequence", "output", "results"),
            ],
        )

        values = SequenceValue[StringValue](
            [StringValue(v) for v in ["a", "b", "FAIL", "d", "e"]]
        )
        context = RecordingContext()
        result = await engine.execute(
            context=context, workflow=outer, input={"items": values}
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        results = as_sequence(result.output["results"])
        assert len(results) == 5
        for i, expected in enumerate(["a", "b", None, "d", "e"]):
            if expected is None:
                assert results[i].is_err()
                assert (
                    results[i]
                    .unwrap_err()
                    .node_id.root.startswith(f"for_each/element_{i}/attempt/")
                )
            else:
                assert results[i].is_ok()
                assert results[i].unwrap_ok().root == expected

        # Ids stay flat: every node id reported by hooks is a plain string,
        # namespaced purely by "/" prefixes, and the failing element's
        # boundary shows up at the expected nested-but-flat prefix.
        assert any(
            nid.startswith("for_each/element_2/attempt/") for nid in context.started
        )
        assert all("//" not in nid for nid in context.started)

        # The eliminators consume the contract without any special-casing:
        # feed the Seq[Result[T]] this run produced straight into
        # PartitionNode and UnwrapOrNode via one-node workflows.
        partition = engine.create_node(PartitionNode, id="p", element_type=StringValue)
        partition_workflow = Workflow(
            input_node=engine.create_input_node(
                sequence=SequenceValue[result_element_type]
            ),
            inner_nodes=[partition],
            output_node=engine.create_output_node(
                oks=SequenceValue[StringValue],
                ok_indices=SequenceValue[IntegerValue],
                errs=SequenceValue[DataValue[ResultError]],
                err_indices=SequenceValue[IntegerValue],
            ),
            edges=[
                edge("input", "sequence", "p", "sequence"),
                edge("p", "oks", "output", "oks"),
                edge("p", "ok_indices", "output", "ok_indices"),
                edge("p", "errs", "output", "errs"),
                edge("p", "err_indices", "output", "err_indices"),
            ],
        )
        partition_result = await engine.execute(
            context=context, workflow=partition_workflow, input={"sequence": results}
        )
        assert partition_result.status is WorkflowExecutionResultStatus.SUCCESS
        assert [v.root for v in as_sequence(partition_result.output["oks"])] == [
            "a",
            "b",
            "d",
            "e",
        ]
        assert [v.root for v in as_sequence(partition_result.output["ok_indices"])] == [
            0,
            1,
            3,
            4,
        ]
        assert [
            v.root for v in as_sequence(partition_result.output["err_indices"])
        ] == [2]

        unwrap = engine.create_node(UnwrapOrNode, id="u", element_type=StringValue)
        unwrap_workflow = Workflow(
            input_node=engine.create_input_node(
                sequence=SequenceValue[result_element_type], default=StringValue
            ),
            inner_nodes=[unwrap],
            output_node=engine.create_output_node(sequence=SequenceValue[StringValue]),
            edges=[
                edge("input", "sequence", "u", "sequence"),
                edge("input", "default", "u", "default"),
                edge("u", "sequence", "output", "sequence"),
            ],
        )
        unwrap_result = await engine.execute(
            context=context,
            workflow=unwrap_workflow,
            input={"sequence": results, "default": StringValue("MISSING")},
        )
        assert unwrap_result.status is WorkflowExecutionResultStatus.SUCCESS
        assert [v.root for v in as_sequence(unwrap_result.output["sequence"])] == [
            "a",
            "b",
            "MISSING",
            "d",
            "e",
        ]


# ---------------------------------------------------------------------------
# 4. Yield passes through.
# ---------------------------------------------------------------------------


class TestYieldPassesThrough:
    @pytest.mark.asyncio
    async def test_yield_then_resume_to_ok(self, algorithm: ExecutionAlgorithm):
        engine = WorkflowEngine(execution_algorithm=algorithm)
        yielder = engine.create_node(YieldOnceNode, id="y")
        workflow = await _build_attempt_workflow(
            engine,
            inner_nodes=[yielder],
            edges=[edge("y", "value", "output", "final")],
            output_fields={"final": StringValue},
        )
        context = RecordingContext()

        result = await _run(engine, workflow, context)
        assert result.status is WorkflowExecutionResultStatus.YIELDED
        assert any(nid.endswith("/y") for nid in result.node_yields)
        assert any(nid.endswith("/y") for nid in context.yielded)
        assert context.boundary_errors == []
        assert "result" not in result.output

        result = await _run(engine, workflow, context)
        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert as_result(result.output["result"]).is_ok()
        assert as_result(result.output["result"]).unwrap_ok().root == "resumed"


# ---------------------------------------------------------------------------
# 5. Yield wins over err, across two passes.
# ---------------------------------------------------------------------------


class TestYieldWinsOverErr:
    @pytest.mark.asyncio
    async def test_yield_wins_pass_1_then_err_pass_2(
        self, algorithm: ExecutionAlgorithm
    ):
        engine = WorkflowEngine(execution_algorithm=algorithm)
        failing = engine.create_node(
            FailingNode, id="boom", params=dict(message=StringValue("always fails"))
        )
        downstream = engine.create_node(DownstreamEchoNode, id="downstream")
        # yielder placed LAST in inner_nodes: with the topological executor's
        # LIFO ready_nodes.popitem(), it is dispatched first, so it actually
        # yields before the sibling failure is processed (required for
        # "yield wins" to have anything to win over, rather than the member
        # simply never getting a chance to run at all).
        yielder = engine.create_node(YieldOnceNode, id="y")
        w = Workflow(
            input_node=engine.create_input_node(),
            inner_nodes=[failing, downstream, yielder],
            output_node=engine.create_output_node(final=StringValue),
            edges=[
                edge("y", "value", "downstream", "value"),
                edge("downstream", "value", "output", "final"),
            ],
        )
        workflow = await engine.build_single_node_workflow(
            AttemptNode, node_id="attempt", params={"workflow": w}
        )
        context = RecordingContext()

        result = await _run(engine, workflow, context)
        assert result.status is WorkflowExecutionResultStatus.YIELDED
        assert result.errors.count == 0
        assert any(nid.endswith("/boom") for nid in context.errored)
        assert (
            "attempt/downstream",
            CancelReason.NOT_SCHEDULED,
            "attempt",
        ) in context.cancelled
        assert context.boundary_errors == []
        assert DownstreamEchoNode.ran == set()

        result = await _run(engine, workflow, context)
        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert as_result(result.output["result"]).is_err()
        assert len(context.boundary_errors) == 1

    @pytest.mark.asyncio
    async def test_yield_wins_then_resolves_ok_on_resume(
        self, algorithm: ExecutionAlgorithm
    ):
        """Variant: the error node fails only on pass 1; pass 2 resolves ok,
        documenting that replaying vs. re-running the failed member is the
        host's choice, not the engine's."""
        engine = WorkflowEngine(execution_algorithm=algorithm)
        failing = engine.create_node(
            FailingNode,
            id="boom",
            params=dict(message=StringValue("fails once"), fail_count=IntegerValue(1)),
        )
        downstream = engine.create_node(DownstreamEchoNode, id="downstream")
        yielder = engine.create_node(YieldOnceNode, id="y")
        w = Workflow(
            input_node=engine.create_input_node(),
            inner_nodes=[failing, downstream, yielder],
            output_node=engine.create_output_node(final=StringValue),
            edges=[
                edge("y", "value", "downstream", "value"),
                edge("downstream", "value", "output", "final"),
            ],
        )
        workflow = await engine.build_single_node_workflow(
            AttemptNode, node_id="attempt", params={"workflow": w}
        )
        context = RecordingContext()

        result = await _run(engine, workflow, context)
        assert result.status is WorkflowExecutionResultStatus.YIELDED

        result = await _run(engine, workflow, context)
        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert as_result(result.output["result"]).is_ok()
        assert as_result(result.output["result"]).unwrap_ok().root == "resumed"
        assert context.boundary_errors == []


# ---------------------------------------------------------------------------
# 6. Retry passes through.
# ---------------------------------------------------------------------------


class TestRetryPassesThrough:
    @pytest.mark.asyncio
    async def test_retry_then_success_is_ok(self, algorithm: ExecutionAlgorithm):
        engine = WorkflowEngine(execution_algorithm=algorithm)
        retrying = engine.create_node(
            FailOnceRetryNode,
            id="r",
            params=dict(fail_count=IntegerValue(1), backoff_ms=IntegerValue(1)),
        )
        workflow = await _build_attempt_workflow(
            engine,
            inner_nodes=[retrying],
            edges=[edge("r", "value", "output", "final")],
            output_fields={"final": StringValue},
        )
        context = RecordingContext()
        result = await _run(engine, workflow, context)

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert as_result(result.output["result"]).is_ok()
        assert len(context.retried) == 1
        assert context.boundary_errors == []

    @pytest.mark.asyncio
    async def test_retries_exhausted_becomes_err_not_run_error(self):
        for algorithm in [
            TopologicalExecutionAlgorithm(max_retries=0),
            ParallelExecutionAlgorithm(max_retries=0),
        ]:
            engine = WorkflowEngine(execution_algorithm=algorithm)
            retrying = engine.create_node(
                FailOnceRetryNode,
                id="r",
                params=dict(fail_count=IntegerValue(1), backoff_ms=IntegerValue(1)),
            )
            workflow = await _build_attempt_workflow(
                engine,
                inner_nodes=[retrying],
                edges=[edge("r", "value", "output", "final")],
                output_fields={"final": StringValue},
            )
            context = RecordingContext()
            result = await _run(engine, workflow, context)

            assert result.status is WorkflowExecutionResultStatus.SUCCESS
            assert result.errors.count == 0
            assert as_result(result.output["result"]).is_err()
            # the executor's own retry budget (0) was consulted, not the boundary
            assert context.retried == []
            FailOnceRetryNode.calls = {}


# ---------------------------------------------------------------------------
# 7. Drain, not kill (parallel only).
# ---------------------------------------------------------------------------


class TestDrainNotKill:
    @pytest.mark.asyncio
    async def test_slow_sibling_drains_before_boundary_error(self):
        algorithm = ParallelExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        slow = engine.create_node(
            AttemptSlowNode, id="slow", params=dict(delay_ms=IntegerValue(150))
        )
        failing = engine.create_node(
            FailingNode, id="boom", params=dict(message=StringValue("fast failure"))
        )
        w = Workflow(
            input_node=engine.create_input_node(),
            inner_nodes=[slow, failing],
            output_node=engine.create_output_node(final=StringValue),
            edges=[edge("slow", "value", "output", "final")],
        )
        workflow = await engine.build_single_node_workflow(
            AttemptNode, node_id="attempt", params={"workflow": w}
        )
        context = RecordingContext()
        result = await _run(engine, workflow, context)

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert as_result(result.output["result"]).is_err()
        assert "attempt/slow" in AttemptSlowNode.completed
        assert "attempt/slow" in context.finished

        finish_index = context.sequence.index("finish:attempt/slow")
        boundary_error_index = context.sequence.index("boundary_error:attempt")
        assert finish_index < boundary_error_index


# ---------------------------------------------------------------------------
# 8. Retry abandoned.
# ---------------------------------------------------------------------------


class TestRetryAbandoned:
    @pytest.mark.asyncio
    async def test_retrying_sibling_abandoned_not_redispatched(self):
        algorithm = ParallelExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        retrying = engine.create_node(
            FailOnceRetryNode,
            id="r",
            params=dict(fail_count=IntegerValue(5), backoff_ms=IntegerValue(5000)),
        )
        failing = engine.create_node(
            FailingNode,
            id="boom",
            params=dict(message=StringValue("immediate failure")),
        )
        w = Workflow(
            input_node=engine.create_input_node(),
            inner_nodes=[retrying, failing],
            output_node=engine.create_output_node(final=StringValue),
            edges=[edge("r", "value", "output", "final")],
        )
        workflow = await engine.build_single_node_workflow(
            AttemptNode, node_id="attempt", params={"workflow": w}
        )
        context = RecordingContext()
        result = await _run(engine, workflow, context)

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert as_result(result.output["result"]).is_err()
        assert FailOnceRetryNode.calls.get("attempt/r") == 1
        abandoned = [
            c
            for c in context.cancelled
            if c[0] == "attempt/r" and c[1] == CancelReason.RETRY_ABANDONED
        ]
        assert len(abandoned) == 1


# ---------------------------------------------------------------------------
# 9. Blocked members are not re-readied by a later expansion.
# ---------------------------------------------------------------------------


class TrivialIdentityInput(Data):
    v: StringValue


class TrivialIdentityOutput(Data):
    v: StringValue


class TestBlockedNotReReadied:
    @pytest.mark.asyncio
    async def test_unrelated_expansion_does_not_restart_blocked_member(self):
        algorithm = ParallelExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)

        succeed = engine.create_node(
            FailingNode,
            id="c",
            params=dict(message=StringValue(""), fail_count=IntegerValue(0)),
        )
        blocked = engine.create_node(DownstreamEchoNode, id="m")
        failing = engine.create_node(
            FailingNode, id="boom", params=dict(message=StringValue("boundary failure"))
        )
        w = Workflow(
            input_node=engine.create_input_node(),
            inner_nodes=[succeed, blocked, failing],
            output_node=engine.create_output_node(final=StringValue),
            edges=[edge("c", "value", "m", "value")],
        )
        attempt_node = engine.create_node(
            AttemptNode, id="attempt", params={"workflow": w}
        )

        identity_inner = Workflow(
            input_node=engine.create_input_node(v=StringValue),
            inner_nodes=[],
            output_node=engine.create_output_node(v=StringValue),
            edges=[edge("input", "v", "output", "v")],
        )
        for_each = engine.create_node(
            ForEachNode, id="for_each", params={"workflow": identity_inner}
        )

        outer = Workflow(
            input_node=engine.create_input_node(items=SequenceValue[StringValue]),
            inner_nodes=[attempt_node, for_each],
            output_node=engine.create_output_node(
                results=SequenceValue[StringValue], result=Result[StringValue]
            ),
            edges=[
                edge("input", "items", "for_each", "sequence"),
                edge("for_each", "sequence", "output", "results"),
                edge("attempt", "result", "output", "result"),
            ],
        )

        context = RecordingContext()
        values = SequenceValue[StringValue]([StringValue("x")])
        result = await engine.execute(
            context=context, workflow=outer, input={"items": values}
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert as_result(result.output["result"]).is_err()
        assert "for_each" in context.expanded
        # The whole point: "m" is a real, dependency-satisfied member of the
        # failed boundary, discoverable only via a full get_ready_nodes()
        # rescan (triggered here by the unrelated for_each's own expansion).
        # It must never actually start.
        assert DownstreamEchoNode.ran == set()
        assert "attempt/m" not in context.started


# ---------------------------------------------------------------------------
# 10. Boundary in boundary.
# ---------------------------------------------------------------------------


class TestNestedBoundaries:
    @pytest.mark.asyncio
    async def test_inner_fails_outer_is_ok_of_err(self, algorithm: ExecutionAlgorithm):
        engine = WorkflowEngine(execution_algorithm=algorithm)
        failing = engine.create_node(
            FailingNode, id="boom", params=dict(message=StringValue("inner failure"))
        )
        w_inner = Workflow(
            input_node=engine.create_input_node(),
            inner_nodes=[failing],
            output_node=engine.create_output_node(final=StringValue),
            edges=[edge("boom", "value", "output", "final")],
        )
        inner_attempt = engine.create_node(
            AttemptNode, id="inner_attempt", params={"workflow": w_inner}
        )
        w_outer = Workflow(
            input_node=engine.create_input_node(),
            inner_nodes=[inner_attempt],
            output_node=engine.create_output_node(final2=Result[StringValue]),
            edges=[edge("inner_attempt", "result", "output", "final2")],
        )
        outer_workflow = await engine.build_single_node_workflow(
            AttemptNode, node_id="outer_attempt", params={"workflow": w_outer}
        )
        context = RecordingContext()
        result = await _run(engine, outer_workflow, context)

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        outer_result = as_result(result.output["result"])
        assert outer_result.is_ok()
        inner_result = outer_result.unwrap_ok()
        assert isinstance(inner_result, Result)
        assert inner_result.is_err()
        assert (
            inner_result.unwrap_err().node_id.root == "outer_attempt/inner_attempt/boom"
        )
        # nested tags preserved on dump
        dumped = outer_result.model_dump(mode="json")
        assert dumped["tag"] == "ok"
        assert dumped["ok"]["tag"] == "err"

    @pytest.mark.asyncio
    async def test_outer_direct_failure_cancels_inner_members(self):
        algorithm = ParallelExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        const = engine.create_node(
            FailingNode,
            id="const",
            params=dict(message=StringValue(""), fail_count=IntegerValue(0)),
        )
        w_inner = Workflow(
            input_node=engine.create_input_node(),
            inner_nodes=[const],
            output_node=engine.create_output_node(final=StringValue),
            edges=[edge("const", "value", "output", "final")],
        )
        inner_attempt = engine.create_node(
            AttemptNode, id="inner_attempt", params={"workflow": w_inner}
        )
        outer_boom = engine.create_node(
            FailingNode,
            id="outer_boom",
            params=dict(message=StringValue("direct outer fail")),
        )
        w_outer = Workflow(
            input_node=engine.create_input_node(),
            inner_nodes=[inner_attempt, outer_boom],
            output_node=engine.create_output_node(final2=Result[StringValue]),
            edges=[edge("inner_attempt", "result", "output", "final2")],
        )
        outer_workflow = await engine.build_single_node_workflow(
            AttemptNode, node_id="outer_attempt", params={"workflow": w_outer}
        )
        context = RecordingContext()
        result = await _run(engine, outer_workflow, context)

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        outer_result = as_result(result.output["result"])
        assert outer_result.is_err()
        assert outer_result.unwrap_err().node_id.root == "outer_attempt/outer_boom"
        assert len(context.boundary_errors) == 1
        assert context.boundary_errors[0][0] == "outer_attempt"
        # some inner-boundary-prefixed member was swept as part of the
        # outer's own flush, and the inner boundary never materializes on
        # its own (that would be a second on_boundary_error call)
        assert any(
            nid.startswith("outer_attempt/inner_attempt/")
            for nid, _, _ in context.cancelled
        )


# ---------------------------------------------------------------------------
# 12. B rule.
# ---------------------------------------------------------------------------


class TestBRule:
    @pytest.mark.asyncio
    async def test_multi_field_output_wraps_in_data_value(self, engine: WorkflowEngine):
        add_like = engine.create_node(
            FailingNode,
            id="a",
            params=dict(message=StringValue(""), fail_count=IntegerValue(0)),
        )
        w = Workflow(
            input_node=engine.create_input_node(),
            inner_nodes=[add_like],
            output_node=engine.create_output_node(f1=StringValue, f2=StringValue),
            edges=[
                edge("a", "value", "output", "f1"),
                edge("a", "value", "output", "f2"),
            ],
        )
        attempted = await engine.build_single_node_workflow(
            AttemptNode, node_id="attempt", params={"workflow": w}
        )
        validated = await engine.validate(attempted)
        result_field_type = get_field_annotations(validated.output_type)["result"]
        assert issubclass(result_field_type, Result)

        context = RecordingContext()
        result = await engine.execute(context=context, workflow=attempted, input={})
        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        ok_value = as_result(result.output["result"]).unwrap_ok()
        assert isinstance(ok_value, DataValue)
        assert ok_value.root.f1.root == "recovered"
        assert ok_value.root.f2.root == "recovered"

    @pytest.mark.asyncio
    async def test_zero_field_output_round_trips(self, engine: WorkflowEngine):
        w = Workflow(
            input_node=engine.create_input_node(),
            inner_nodes=[],
            output_node=engine.create_output_node(),
            edges=[],
        )
        attempted = await engine.build_single_node_workflow(
            AttemptNode, node_id="attempt", params={"workflow": w}
        )
        context = RecordingContext()
        result = await engine.execute(context=context, workflow=attempted, input={})
        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        ok_value = as_result(result.output["result"]).unwrap_ok()
        assert isinstance(ok_value, DataValue)
        # round-trips through JSON
        dumped = as_result(result.output["result"]).model_dump(mode="json")
        assert dumped == {"tag": "ok", "ok": {}}


# ---------------------------------------------------------------------------
# 13. Reserved id collision.
# ---------------------------------------------------------------------------


class TestReservedIdCollision:
    @pytest.mark.asyncio
    async def test_inner_node_named_ok_fails_at_expansion(self, engine: WorkflowEngine):
        colliding = engine.create_node(
            FailingNode,
            id="ok",
            params=dict(message=StringValue(""), fail_count=IntegerValue(0)),
        )
        w = Workflow(
            input_node=engine.create_input_node(),
            inner_nodes=[colliding],
            output_node=engine.create_output_node(final=StringValue),
            edges=[edge("ok", "value", "output", "final")],
        )
        workflow = await engine.build_single_node_workflow(
            AttemptNode, node_id="attempt", params={"workflow": w}
        )
        context = RecordingContext()
        result = await engine.execute(context=context, workflow=workflow, input={})

        assert result.status is WorkflowExecutionResultStatus.ERROR
        assert result.errors.count == 1
        message = result.errors.messages()[0]
        assert "reserved id 'ok'" in message
