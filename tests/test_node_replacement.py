"""Single-node delegation preserves typed adapters, flat provenance and replay."""

from datetime import timedelta
from typing import ClassVar

import pytest
from overrides import override
from pydantic import Field

from tests.test_sequence_ops import edge, workflow
from workflow_engine import (
    Data,
    DataMapping,
    ErrorClass,
    IntegerValue,
    Node,
    NodeException,
    NodeTypeInfo,
    Params,
    ReplacementFrame,
    Result,
    ShouldRetry,
    ShouldYield,
    StakeholderLevel,
    StringValue,
    Workflow,
    WorkflowEngine,
    WorkflowException,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core.values import get_data_dict
from workflow_engine.nodes import AttemptNode

pytestmark = pytest.mark.integration


class ReplacementData(Data):
    value: IntegerValue = Field(title="Value", description="The value to delegate.")


class DelegationParams(Params):
    remaining: IntegerValue = Field(
        default=IntegerValue(0),
        title="Remaining",
        description="The remaining delegation hops.",
    )
    mode: StringValue = Field(
        default=StringValue("value"),
        title="Mode",
        description="The behavior of the last node.",
    )
    child_retries: IntegerValue = Field(
        default=IntegerValue(1), title="Retries", description="The child retry budget."
    )


class DelegationProbeNode(Node[ReplacementData, ReplacementData, DelegationParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Delegation probe",
        version="1.0.0",
        parameter_type=DelegationParams,
    )
    calls: ClassVar[list[str]] = []

    @classmethod
    @override
    def static_input_type(cls) -> type[ReplacementData]:
        return ReplacementData

    @classmethod
    @override
    def static_output_type(cls) -> type[ReplacementData]:
        return ReplacementData

    @override
    async def run(
        self, *, context, input_type, output_type, input
    ) -> ReplacementData | Node | Workflow:
        self.calls.append(self.id)
        if self.params.remaining.root:
            return DelegationProbeNode(
                type="DelegationProbe",
                id="author-label",
                params=self.params.model_update(
                    remaining=IntegerValue(self.params.remaining.root - 1)
                ),
            )
        if self.params.mode.root == "self":
            return self
        return ReplacementLeafNode(
            type="ReplacementLeaf",
            id="author-leaf",
            params=self.params,
            max_retries=self.params.child_retries.root,
        )


class ReplacementLeafNode(Node[ReplacementData, ReplacementData, DelegationParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Replacement leaf",
        version="1.0.0",
        parameter_type=DelegationParams,
    )
    calls: ClassVar[list[str]] = []

    @classmethod
    @override
    def static_input_type(cls) -> type[ReplacementData]:
        return ReplacementData

    @classmethod
    @override
    def static_output_type(cls) -> type[ReplacementData]:
        return ReplacementData

    @override
    async def run(
        self, *, context, input_type, output_type, input
    ) -> ReplacementData | Node | Workflow:
        self.calls.append(self.id)
        count = self.calls.count(self.id)
        mode = self.params.mode.root
        if mode == "fail":
            raise NodeException.for_user(
                "Leaf failed",
                node=self,
                name="LeafFailure",
                error_class=ErrorClass.VALIDATION,
            )
        if (mode == "yield" and count == 1) or (mode == "retry_yield" and count == 2):
            raise ShouldYield("Leaf waiting")
        if (mode == "retry" and count == 1) or (mode == "retry_yield" and count != 2):
            raise ShouldRetry(
                "Leaf retry",
                node=self,
                level=StakeholderLevel.USER,
                backoff=timedelta(0),
            )
        if mode == "expand":
            engine = WorkflowEngine()
            return workflow(
                engine,
                {"value": IntegerValue},
                {"value": IntegerValue},
                [],
                [edge("input", "value", "output", "value")],
            )
        return output_type(value=input.value)


class ReplacementContext(InMemoryExecutionContext):
    def __init__(self, frames=None, cache=None, **kwargs):
        super().__init__(**kwargs)
        self.frames = {} if frames is None else frames
        self.cache = {} if cache is None else cache
        self.events = []

    @override
    async def on_node_start(
        self, *, node, input_type, output_type, input
    ) -> DataMapping | Node | None:
        self.events.append(("start", node.id))
        if node.id in self.cache:
            return get_data_dict(output_type.model_validate_json(self.cache[node.id]))
        if node.id in self.frames:
            return ReplacementFrame.model_validate_json(self.frames[node.id]).replay(
                node=node, input_type=input_type, output_type=output_type, input=input
            )

    @override
    async def get_node_replacement_frame(self, *, node_id):
        if node_id in self.frames:
            return ReplacementFrame.model_validate_json(self.frames[node_id])

    @override
    async def on_node_replace(self, *, node, replacement, input, replacement_info):
        self.events.append(("replace", node.id, replacement.id))
        self.frames[node.id] = replacement_info.model_dump_json()

    @override
    async def on_node_replacement_checkpoint(self, *, frame):
        self.frames[frame.delegator_id] = frame.model_dump_json()

    @override
    async def on_node_finish(
        self, *, node, input_type, output_type, input, output
    ) -> DataMapping:
        self.events.append(("finish", node.id))
        self.cache[node.id] = output_type.model_validate(output).model_dump_json()
        return output

    @override
    async def on_node_error(
        self, *, node, input_type, output_type, input, exception
    ) -> WorkflowException | DataMapping:
        self.events.append(("error", node.id))
        return exception

    @override
    async def on_node_replacement_failed(self, *, node, replacement_info, exception):
        self.events.append(("failed", node.id, exception.node_id))


@pytest.fixture(autouse=True)
def reset():
    DelegationProbeNode.calls = []
    ReplacementLeafNode.calls = []


async def execute(algorithm, *, params=None, context=None):
    return await WorkflowEngine(execution_algorithm=algorithm).execute_node(
        context=context or ReplacementContext(),
        node=DelegationProbeNode,
        params=params or {},
        input={"value": 7},
    )


async def test_replacement_runs_real_child_and_finishes_caller_once(algorithm):
    context = ReplacementContext()
    result = await execute(algorithm, context=context)
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    assert result.output["value"] == IntegerValue(7)
    assert context.events == [
        ("start", "input"),
        ("finish", "input"),
        ("start", "node"),
        ("replace", "node", "node/replacement_0"),
        ("start", "node/replacement_0"),
        ("finish", "node/replacement_0"),
        ("finish", "node"),
        ("start", "output"),
        ("finish", "output"),
    ]
    frame = ReplacementFrame.model_validate_json(context.frames["node"])
    assert frame.original_label == "author-leaf"
    assert frame.status == "completed"
    assert frame.completed_slots == ("node",)


@pytest.mark.parametrize("mode", ["value", "retry", "expand"])
async def test_replacement_chain_and_workflow_completion(algorithm, mode):
    context = ReplacementContext()
    result = await execute(
        algorithm, context=context, params={"remaining": 3, "mode": mode}
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    assert result.output["value"] == IntegerValue(7)
    assert DelegationProbeNode.calls == [
        "node",
        "node/replacement_0",
        "node/replacement_1",
        "node/replacement_2",
    ]
    assert set(ReplacementLeafNode.calls) == {"node/replacement_3"}
    assert context.events.count(("finish", "node")) == 1


async def test_child_error_keeps_provenance_and_does_not_call_parent_error_hook(
    algorithm,
):
    context = ReplacementContext()
    result = await execute(algorithm, context=context, params={"mode": "fail"})
    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert set(result.errors.node_errors) == {"node/replacement_0"}
    assert ("error", "node") not in context.events
    assert ("failed", "node", "node/replacement_0") in context.events
    assert ("finish", "node") not in context.events


async def test_replacement_yield_replays_json_frame_without_redispatching_delegator(
    algorithm,
):
    context = ReplacementContext()
    first = await execute(algorithm, context=context, params={"mode": "yield"})
    assert first.status is WorkflowExecutionResultStatus.YIELDED
    assert first.node_yields == {"node/replacement_0": "Leaf waiting"}
    resumed = ReplacementContext(dict(context.frames), dict(context.cache))
    result = await execute(algorithm, context=resumed, params={"mode": "yield"})
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    assert DelegationProbeNode.calls == ["node"]
    assert ReplacementLeafNode.calls == ["node/replacement_0", "node/replacement_0"]


async def test_replacement_failure_inside_attempt(algorithm):
    engine = WorkflowEngine(execution_algorithm=algorithm)
    inner = await engine.build_single_node_workflow(
        DelegationProbeNode, params={"mode": "fail"}
    )
    result = await engine.execute_node(
        context=ReplacementContext(),
        node=AttemptNode,
        params={"workflow": inner},
        input={"value": 7},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    wrapped = result.output["result"]
    assert isinstance(wrapped, Result)
    error = wrapped.unwrap_err()
    assert error.name.root == "LeafFailure"
    assert error.node_id.root == "node/node/replacement_0"


async def test_retry_budget_survives_yield_resume(algorithm):
    context = ReplacementContext()
    first = await execute(
        algorithm, context=context, params={"mode": "retry_yield", "child_retries": 1}
    )
    assert first.status is WorkflowExecutionResultStatus.YIELDED
    resumed = ReplacementContext(dict(context.frames), dict(context.cache))
    result = await execute(
        algorithm, context=resumed, params={"mode": "retry_yield", "child_retries": 1}
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert len(ReplacementLeafNode.calls) == 3
    assert DelegationProbeNode.calls == ["node"]


@pytest.mark.parametrize("mode,remaining", [("self", 0), ("value", 4)])
async def test_replacement_self_and_hop_limits(algorithm, mode, remaining):
    algorithm.max_replacement_hops = 3
    result = await execute(algorithm, params={"mode": mode, "remaining": remaining})
    assert result.status is WorkflowExecutionResultStatus.ERROR
    error = next(iter(result.errors.node_errors.values()))[0]
    assert error is not None
    assert error.error_class is ErrorClass.VALIDATION
    assert error.name == "NodeReplacementException"
    assert not ReplacementLeafNode.calls
