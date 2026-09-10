"""Boundary retries compose above, and never consume, executor courtesy retries."""

import asyncio
from collections import Counter
from datetime import timedelta
from typing import ClassVar

import pytest
from overrides import override
from pydantic import Field, ValidationError

from workflow_engine import (
    BooleanValue,
    Data,
    DataMapping,
    Edge,
    Empty,
    ErrorClass,
    ErrorClassValue,
    ExecutionContext,
    IntegerValue,
    Node,
    NodeTypeInfo,
    Result,
    ResultError,
    SequenceValue,
    ShouldRetry,
    ShouldYield,
    StringValue,
    Workflow,
    WorkflowEngine,
    WorkflowException,
    WorkflowExecutionResultStatus,
    WorkflowValue,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.nodes import AttemptNode, IfNode, UnwrapNode
from workflow_engine.nodes.attempt import AttemptParams


class RetryProbeData(Data):
    # A deliberate collision with the Attempt output port.
    result: StringValue = Field(title="Result", description="The original input.")


class BoundaryRetryProbeNode(Node[RetryProbeData, RetryProbeData, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Boundary Retry Probe",
        version="1.0.0",
        parameter_type=Empty,
    )

    @classmethod
    @override
    def static_input_type(cls) -> type[RetryProbeData]:
        return RetryProbeData

    @classmethod
    @override
    def static_output_type(cls) -> type[RetryProbeData]:
        return RetryProbeData

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[RetryProbeData],
        output_type: type[RetryProbeData],
        input: RetryProbeData,
    ) -> RetryProbeData:
        assert isinstance(context, RetryContext)
        context.calls[self.id] += 1
        context.inputs.append(input.result.root)
        if context.yield_at == self.id and context.calls[self.id] == 1:
            raise ShouldYield("waiting for external work")
        if context.calls[self.id] <= context.courtesy_failures:
            raise ShouldRetry.for_user(
                "transient blip",
                node=self,
                backoff=timedelta(0),
                error_class=ErrorClass.TIMEOUT,
            )
        if not any(marker in self.id for marker in context.succeed_at):
            raise WorkflowException.for_user("failed", error_class=context.error_class)
        return output_type(result=input.result)


class MeteredRetryProbeNode(BoundaryRetryProbeNode):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Metered Retry Probe",
        version="1.0.0",
        parameter_type=Empty,
        metered=True,
    )


class RetryContext(InMemoryExecutionContext):
    def __init__(self):
        super().__init__()
        self.calls: Counter[str] = Counter()
        self.inputs: list[str] = []
        self.succeed_at: tuple[str, ...] = ("try_1",)
        self.courtesy_failures = 0
        self.error_class = ErrorClass.TIMEOUT
        self.yield_at: str | None = None
        self.retry_events: list[tuple[str, int, int, bool, str]] = []
        self.courtesy_events: list[tuple[str, int]] = []
        self.boundary_errors: dict[str, DataMapping] = {}
        self.finished: dict[str, DataMapping] = {}
        self.use_cache = False

    @override
    async def on_boundary_retry(
        self,
        *,
        node: Node,
        boundary_id: str,
        error: ResultError,
        attempt: int,
        max_retries: int,
        allow_metered: bool,
    ) -> None:
        self.retry_events.append(
            (boundary_id, attempt, max_retries, allow_metered, error.node_id.root)
        )

    @override
    async def on_node_retry(
        self,
        *,
        node: Node,
        input_type: type[Data],
        output_type: type[Data],
        input: DataMapping,
        exception: ShouldRetry,
        attempt: int,
    ) -> None:
        self.courtesy_events.append((node.id, attempt))

    @override
    async def on_boundary_error(
        self,
        *,
        node: Node,
        input_type: type[Data],
        output_type: type[Data],
        input: DataMapping,
        error: ResultError,
        output: DataMapping,
        cause: WorkflowException,
    ) -> DataMapping:
        self.boundary_errors[node.id] = output
        return output

    @override
    async def on_node_finish(
        self,
        *,
        node: Node,
        input_type: type[Data],
        output_type: type[Data],
        input: DataMapping,
        output: DataMapping,
    ) -> DataMapping:
        self.finished[node.id] = output
        return output

    @override
    async def on_node_start(
        self,
        *,
        node: Node,
        input_type: type[Data],
        output_type: type[Data],
        input: DataMapping,
    ) -> DataMapping | Workflow | None:
        if self.use_cache:
            return self.boundary_errors.get(node.id, self.finished.get(node.id))
        return None


async def build_attempt(engine, *, params=None, probe=BoundaryRetryProbeNode):
    inner = await engine.build_single_node_workflow(probe, node_id="probe")
    return await engine.build_single_node_workflow(
        AttemptNode,
        node_id="attempt",
        params={"workflow": inner, **(params or {})},
    )


async def run(engine, workflow, context):
    return await engine.execute(
        context=context, workflow=workflow, input={"result": StringValue("original")}
    )


def result_value(execution) -> Result:
    assert execution.status is WorkflowExecutionResultStatus.SUCCESS
    assert execution.errors.count == 0
    value = execution.output["result"]
    assert isinstance(value, Result)
    return value


@pytest.mark.asyncio
async def test_retry_preserves_inputs_and_courtesy_budgets(algorithm):
    engine = WorkflowEngine(execution_algorithm=algorithm)
    workflow = await build_attempt(engine, params={"retries": 1})
    context = RetryContext()
    context.courtesy_failures = 2
    value = result_value(await run(engine, workflow, context))
    assert value.unwrap_ok() == StringValue("original")
    assert context.inputs == ["original"] * 6
    assert context.calls == {
        "attempt/try_0/probe": 3,
        "attempt/next/try_1/probe": 3,
    }
    assert context.courtesy_events == [
        ("attempt/try_0/probe", 1),
        ("attempt/try_0/probe", 2),
        ("attempt/next/try_1/probe", 1),
        ("attempt/next/try_1/probe", 2),
    ]
    assert context.retry_events == [("attempt", 1, 1, False, "attempt/try_0/probe")]
    assert set(context.boundary_errors) == {"attempt/try_0"}


@pytest.mark.asyncio
@pytest.mark.parametrize("error_class", list(ErrorClass))
async def test_default_retry_classes_and_exhaustion(algorithm, error_class):
    engine = WorkflowEngine(execution_algorithm=algorithm)
    workflow = await build_attempt(engine, params={"retries": 2})
    context = RetryContext()
    context.succeed_at = ()
    context.error_class = error_class
    error = result_value(await run(engine, workflow, context)).unwrap_err()
    transient = error_class in {
        ErrorClass.TIMEOUT,
        ErrorClass.UNREACHABLE,
        ErrorClass.RATE_LIMIT,
    }
    assert sum(context.calls.values()) == (3 if transient else 1)
    assert len(context.retry_events) == (2 if transient else 0)
    assert error.error_class.root == error_class
    assert error.message.root == "failed"
    expected_id = (
        "attempt/next/next/try_2/probe" if transient else "attempt/try_0/probe"
    )
    assert error.node_id.root == expected_id
    last_result = next(reversed(context.boundary_errors.values()))["result"]
    assert isinstance(last_result, Result)
    assert error == last_result.unwrap_err()


@pytest.mark.asyncio
async def test_explicit_systemic_retry_and_default_off(algorithm):
    engine = WorkflowEngine(execution_algorithm=algorithm)
    context = RetryContext()
    context.error_class = ErrorClass.SYSTEMIC
    workflow = await build_attempt(engine)
    assert result_value(await run(engine, workflow, context)).is_err()
    assert context.calls == {"attempt/probe": 1}
    assert context.retry_events == []
    context = RetryContext()
    context.error_class = ErrorClass.SYSTEMIC
    workflow = await build_attempt(
        engine, params={"retries": 1, "retry_on": ["systemic"]}
    )
    assert result_value(await run(engine, workflow, context)).is_ok()


@pytest.mark.asyncio
async def test_metered_work_requires_explicit_consent(algorithm):
    engine = WorkflowEngine(execution_algorithm=algorithm)
    with pytest.raises(WorkflowException, match="metered node 'probe'"):
        await build_attempt(engine, params={"retries": 1}, probe=MeteredRetryProbeNode)
    workflow = await build_attempt(
        engine,
        params={"retries": 1, "allow_metered": True},
        probe=MeteredRetryProbeNode,
    )
    context = RetryContext()
    assert result_value(await run(engine, workflow, context)).is_ok()
    assert context.retry_events[0][3] is True
    # Omitting retries never causes an additional metered dispatch.
    workflow = await build_attempt(engine, probe=MeteredRetryProbeNode)
    context = RetryContext()
    assert result_value(await run(engine, workflow, context)).is_err()
    assert sum(context.calls.values()) == 1


@pytest.mark.asyncio
async def test_metered_work_in_nested_branch_is_detected():
    engine = WorkflowEngine()
    metered = await engine.build_single_node_workflow(MeteredRetryProbeNode)
    branch = await engine.build_single_node_workflow(
        IfNode, params={"if_true": metered}
    )
    with pytest.raises(WorkflowException, match="allow_metered"):
        await engine.build_single_node_workflow(
            AttemptNode, params={"workflow": branch, "retries": 1}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [True, False])
async def test_yield_resume_retains_attempt_budget_and_cache_identity(
    algorithm, recover
):
    engine = WorkflowEngine(execution_algorithm=algorithm)
    workflow = await build_attempt(engine, params={"retries": 2})
    context = RetryContext()
    context.use_cache = True
    context.yield_at = "attempt/next/try_1/probe"
    context.succeed_at = ("try_2",) if recover else ()
    first = await run(engine, workflow, context)
    assert first.status is WorkflowExecutionResultStatus.YIELDED
    assert [event[1] for event in context.retry_events] == [1]
    second = result_value(await run(engine, workflow, context))
    assert second.is_ok() is recover
    assert context.calls == {
        "attempt/try_0/probe": 1,
        "attempt/next/try_1/probe": 2,
        "attempt/next/next/try_2/probe": 1,
    }
    # Resume may redeliver a hook, but its deterministic ledger key stays the same.
    assert [event[1] for event in context.retry_events] == [1, 1, 2]
    assert all(event[2] == 2 for event in context.retry_events)
    assert len(set((event[0], event[1]) for event in context.retry_events)) == 2


@pytest.mark.unit
def test_retry_policy_roundtrip_and_negative_budget():
    engine = WorkflowEngine()
    workflow = Workflow(
        input_node=engine.create_input_node(),
        inner_nodes=[],
        output_node=engine.create_output_node(),
        edges=[],
    )
    with pytest.raises(ValidationError, match="nonnegative"):
        AttemptParams(workflow=WorkflowValue(workflow), retries=IntegerValue(-1))
    policy = AttemptParams(
        workflow=WorkflowValue(workflow),
        retries=IntegerValue(2),
        retry_on=SequenceValue([ErrorClassValue(ErrorClass.TIMEOUT)]),
        allow_metered=BooleanValue(True),
    )
    assert AttemptParams.model_validate_json(policy.model_dump_json()) == policy


@pytest.mark.asyncio
async def test_exhausted_courtesy_budget_resets_for_each_boundary_try(algorithm):
    algorithm.max_retries = 1
    engine = WorkflowEngine(execution_algorithm=algorithm)
    workflow = await build_attempt(engine, params={"retries": 1})
    context = RetryContext()
    context.courtesy_failures = 10
    error = result_value(await run(engine, workflow, context)).unwrap_err()
    assert error.error_class.root is ErrorClass.TIMEOUT
    assert context.calls == {"attempt/try_0/probe": 2, "attempt/next/try_1/probe": 2}
    assert [attempt for _, attempt in context.courtesy_events] == [1, 1]
    assert len(context.retry_events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("unwrap", [True, False])
async def test_nested_boundaries_own_independent_budgets(algorithm, unwrap):
    engine = WorkflowEngine(execution_algorithm=algorithm)
    probe = await engine.build_single_node_workflow(
        BoundaryRetryProbeNode, node_id="probe"
    )
    inner = engine.create_node(
        AttemptNode, id="inner", params={"workflow": probe, "retries": 1}
    )
    input_node = engine.create_input_node(result=StringValue)
    nodes: list[Node] = [inner]
    edges = [
        Edge.from_nodes(
            source=input_node, source_key="result", target=inner, target_key="result"
        )
    ]
    if unwrap:
        unwrapped = engine.create_node(
            UnwrapNode, id="unwrap", element_type=StringValue
        )
        nodes.append(unwrapped)
        output_node = engine.create_output_node(result=StringValue)
        edges.extend(
            [
                Edge.from_nodes(
                    source=inner,
                    source_key="result",
                    target=unwrapped,
                    target_key="result",
                ),
                Edge.from_nodes(
                    source=unwrapped,
                    source_key="value",
                    target=output_node,
                    target_key="result",
                ),
            ]
        )
    else:
        output_node = engine.create_output_node(result=Result[StringValue])
        edges.append(
            Edge.from_nodes(
                source=inner,
                source_key="result",
                target=output_node,
                target_key="result",
            )
        )
    middle = Workflow(
        input_node=input_node, inner_nodes=nodes, output_node=output_node, edges=edges
    )
    workflow = await engine.build_single_node_workflow(
        AttemptNode, node_id="outer", params={"workflow": middle, "retries": 1}
    )
    context = RetryContext()
    context.succeed_at = ()
    value = result_value(await run(engine, workflow, context))
    assert sum(context.calls.values()) == (4 if unwrap else 2)
    if unwrap:
        assert value.is_err()
        assert (
            value.unwrap_err().node_id.root == "outer/next/try_1/inner/next/try_1/probe"
        )
        assert [(event[0], event[1]) for event in context.retry_events] == [
            ("outer/try_0/inner", 1),
            ("outer", 1),
            ("outer/next/try_1/inner", 1),
        ]
    else:
        inner_result = value.unwrap_ok()
        assert isinstance(inner_result, Result)
        assert inner_result.is_err()
        assert len(context.retry_events) == 1
    assert all(event[2] == 1 for event in context.retry_events)


class DrainRetryContext(RetryContext):
    def __init__(self):
        super().__init__()
        self.slow_started: dict[str, asyncio.Event] = {}
        self.slow_finished: set[str] = set()

    @override
    async def on_boundary_retry(
        self,
        *,
        node: Node,
        boundary_id: str,
        error: ResultError,
        attempt: int,
        max_retries: int,
        allow_metered: bool,
    ) -> None:
        prefix = error.node_id.root.rsplit("/", 1)[0]
        assert prefix in self.slow_finished, (
            "retry dispatched before its sibling drained"
        )
        await super().on_boundary_retry(
            node=node,
            boundary_id=boundary_id,
            error=error,
            attempt=attempt,
            max_retries=max_retries,
            allow_metered=allow_metered,
        )


class DrainRetryProbeNode(BoundaryRetryProbeNode):
    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[RetryProbeData],
        output_type: type[RetryProbeData],
        input: RetryProbeData,
    ) -> RetryProbeData:
        assert isinstance(context, DrainRetryContext)
        prefix = self.id.rsplit("/", 1)[0]
        await context.slow_started.setdefault(prefix, asyncio.Event()).wait()
        return await super().run(
            context=context, input_type=input_type, output_type=output_type, input=input
        )


class DrainRetrySiblingNode(BoundaryRetryProbeNode):
    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[RetryProbeData],
        output_type: type[RetryProbeData],
        input: RetryProbeData,
    ) -> RetryProbeData:
        assert isinstance(context, DrainRetryContext)
        prefix = self.id.rsplit("/", 1)[0]
        context.slow_started.setdefault(prefix, asyncio.Event()).set()
        await asyncio.sleep(0.02)
        context.slow_finished.add(prefix)
        return output_type(result=input.result)


@pytest.mark.asyncio
async def test_parallel_retry_drains_and_repeats_successful_inner_work():
    from workflow_engine.execution import ParallelExecutionAlgorithm

    engine = WorkflowEngine(execution_algorithm=ParallelExecutionAlgorithm())
    input_node = engine.create_input_node(result=StringValue)
    probe = engine.create_node(DrainRetryProbeNode, id="probe")
    sibling = engine.create_node(DrainRetrySiblingNode, id="sibling")
    output_node = engine.create_output_node(probe=StringValue, sibling=StringValue)
    inner = Workflow(
        input_node=input_node,
        inner_nodes=[probe, sibling],
        output_node=output_node,
        edges=[
            Edge.from_nodes(
                source=input_node, source_key="result", target=node, target_key="result"
            )
            for node in [probe, sibling]
        ]
        + [
            Edge.from_nodes(
                source=node, source_key="result", target=output_node, target_key=node.id
            )
            for node in [probe, sibling]
        ],
    )
    workflow = await engine.build_single_node_workflow(
        AttemptNode, node_id="attempt", params={"workflow": inner, "retries": 1}
    )
    context = DrainRetryContext()
    assert result_value(await run(engine, workflow, context)).is_ok()
    assert context.slow_finished == {"attempt/try_0", "attempt/next/try_1"}
    assert len(context.retry_events) == 1
