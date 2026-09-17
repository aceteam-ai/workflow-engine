"""Replacement boundaries retain containment, drain and metering consent."""

import asyncio
from typing import ClassVar

import pytest
from overrides import override
from pydantic import Field

from tests.test_node_replacement import (
    DelegationProbeNode,
    ReplacementContext,
    ReplacementData,
    ReplacementLeafNode,
)
from tests.test_node_replacement_contracts import (
    ContractNode,
    IntegerRecord,
    contract_node,
)
from tests.test_sequence_ops import edge, workflow
from workflow_engine import (
    Data,
    DataMapping,
    Empty,
    ErrorClass,
    InMemoryLimitCoordinator,
    IntegerValue,
    LimitPolicy,
    Node,
    NodeException,
    NodeTypeInfo,
    RateLimitConfig,
    Result,
    StringValue,
    WorkflowEngine,
    WorkflowException,
    WorkflowExecutionResultStatus,
)
from workflow_engine.execution import ParallelExecutionAlgorithm
from workflow_engine.nodes import AttemptNode

pytestmark = pytest.mark.integration


class AttemptOutput(Data):
    result: Result[IntegerValue] = Field(
        title="Result", description="The attempted value."
    )


@pytest.fixture(autouse=True)
def reset_probes():
    DelegationProbeNode.calls = []
    ReplacementLeafNode.calls = []
    ContractNode.calls = []
    BlockingReplacementNode.started = asyncio.Event()
    BlockingReplacementNode.release = asyncio.Event()
    BlockingReplacementNode.cancelled = asyncio.Event()
    MeteredReplacementNode.calls = []
    QueuedReplacementNode.calls = []


async def attempted_target(engine):
    inner = await engine.build_single_node_workflow(
        ReplacementLeafNode, params={"mode": "fail"}
    )
    target = engine.create_node(
        AttemptNode, id="attempted-leaf", params={"workflow": inner}
    )
    parent = contract_node(engine, IntegerRecord, AttemptOutput, target=target)
    return await engine.build_single_node_workflow(ContractNode, params=parent.params)


async def test_replacement_attempt_failure_is_a_successful_delegated_result(algorithm):
    engine = WorkflowEngine(execution_algorithm=algorithm)
    graph = await attempted_target(engine)
    context = ReplacementContext()
    result = await engine.execute(context=context, workflow=graph, input={"value": 7})
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    value = result.output["result"]
    assert isinstance(value, Result)
    assert value.unwrap_err().node_id.root == "node/replacement_0/node"
    assert ("finish", "node") in context.events
    assert not [event for event in context.events if event[0] == "failed"]


class BreakCallerFinish(ReplacementContext):
    @override
    async def on_node_finish(
        self, *, node, input_type, output_type, input, output
    ) -> DataMapping:
        if node.id == "node/node":
            return {"result": StringValue("invalid Result")}
        return await super().on_node_finish(
            node=node,
            input_type=input_type,
            output_type=output_type,
            input=input,
            output=output,
        )


async def test_adapter_failure_after_inner_attempt_materialization_reaches_outer_attempt(
    algorithm,
):
    engine = WorkflowEngine(execution_algorithm=algorithm)
    inner = await attempted_target(engine)
    result = await engine.execute_node(
        context=BreakCallerFinish(),
        node=AttemptNode,
        params={"workflow": inner},
        input={"value": 7},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    value = result.output["result"]
    assert isinstance(value, Result)
    error = value.unwrap_err()
    assert error.node_id.root == "node/node"
    assert error.error_class.root is ErrorClass.VALIDATION
    assert error.name.root == "NodeReplacementException"


class BlockingReplacementNode(Node[ReplacementData, ReplacementData, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Blocking replacement", version="1.0.0", parameter_type=Empty
    )
    started: ClassVar[asyncio.Event]
    release: ClassVar[asyncio.Event]
    cancelled: ClassVar[asyncio.Event]

    @classmethod
    @override
    def static_input_type(cls) -> type[ReplacementData]:
        return ReplacementData

    @classmethod
    @override
    def static_output_type(cls) -> type[ReplacementData]:
        return ReplacementData

    @override
    async def run(self, *, context, input_type, output_type, input) -> ReplacementData:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return input


class FailAfterAdmissionNode(Node[Empty, Empty, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Fail after admission", version="1.0.0", parameter_type=Empty
    )

    @classmethod
    @override
    def static_input_type(cls) -> type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> type[Empty]:
        return Empty

    @override
    async def run(self, *, context, input_type, output_type, input) -> Empty:
        await BlockingReplacementNode.started.wait()
        raise NodeException.for_user("Sibling failed", node=self, name="SiblingFailure")


class DrainContext(ReplacementContext):
    @override
    async def on_node_error(
        self, *, node, input_type, output_type, input, exception
    ) -> WorkflowException | DataMapping:
        if node.id.endswith("boom"):
            BlockingReplacementNode.release.set()
        return await super().on_node_error(
            node=node,
            input_type=input_type,
            output_type=output_type,
            input=input,
            exception=exception,
        )


async def blocking_parent(engine):
    child = engine.create_node(BlockingReplacementNode, id="blocking")
    parent = contract_node(
        engine, IntegerRecord, IntegerRecord, target=child, id="delegate"
    )
    return parent


class QueuedReplacementNode(Node[ReplacementData, ReplacementData, Empty]):
    """A replacement whose own type carries a real, exhaustible admission pool."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Queued replacement",
        version="1.0.0",
        parameter_type=Empty,
        execution_limits=RateLimitConfig(max_concurrency=1),
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
    async def run(self, *, context, input_type, output_type, input) -> ReplacementData:
        # Reaching the body at all means admission was granted; the test
        # holds the sole slot externally, so this must never run.
        self.calls.append(self.id)
        return input


async def queued_replacement_parent(engine):
    child = engine.create_node(QueuedReplacementNode, id="blocking")
    parent = contract_node(
        engine, IntegerRecord, IntegerRecord, target=child, id="delegate"
    )
    return parent


async def test_failed_boundary_drains_child_without_publishing_pending_caller():
    engine = WorkflowEngine(execution_algorithm=ParallelExecutionAlgorithm())
    parent = await blocking_parent(engine)
    boom = engine.create_node(FailAfterAdmissionNode, id="boom")
    graph = workflow(
        engine,
        {"value": IntegerValue},
        {"value": IntegerValue},
        [parent, boom],
        [
            edge("input", "value", "delegate", "value"),
            edge("delegate", "value", "output", "value"),
        ],
    )
    context = DrainContext()
    result = await asyncio.wait_for(
        engine.execute_node(
            context=context,
            node=AttemptNode,
            params={"workflow": graph},
            input={"value": 7},
        ),
        timeout=5,
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    assert ("finish", "node/delegate/replacement_0") in context.events
    assert ("finish", "node/delegate") not in context.events
    assert ("failed", "node/delegate", "node/boom") in context.events
    wrapped = result.output["result"]
    assert isinstance(wrapped, Result)
    assert wrapped.unwrap_err().name.root == "SiblingFailure"


async def test_cancelling_execution_drains_real_replacement_task(algorithm):
    engine = WorkflowEngine(execution_algorithm=algorithm)
    parent = await blocking_parent(engine)
    graph = await engine.build_single_node_workflow(ContractNode, params=parent.params)
    context = ReplacementContext()
    task = asyncio.create_task(
        engine.execute(context=context, workflow=graph, input={"value": 7})
    )
    await asyncio.wait_for(BlockingReplacementNode.started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert BlockingReplacementNode.cancelled.is_set()
    assert ("finish", "node") not in context.events


class MeteredReplacementNode(Node[ReplacementData, ReplacementData, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Metered replacement",
        version="1.0.0",
        parameter_type=Empty,
        metered=True,
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
    async def run(self, *, context, input_type, output_type, input) -> ReplacementData:
        self.calls.append(self.id)
        raise NodeException.for_user(
            "Paid transient failure", node=self, error_class=ErrorClass.TIMEOUT
        )


@pytest.mark.parametrize("allow_metered", [False, True])
async def test_replacement_does_not_bypass_boundary_metered_retry_consent(
    algorithm, allow_metered
):
    engine = WorkflowEngine(execution_algorithm=algorithm)
    child = engine.create_node(MeteredReplacementNode, id="paid")
    parent = contract_node(engine, IntegerRecord, IntegerRecord, target=child)
    graph = await engine.build_single_node_workflow(ContractNode, params=parent.params)
    result = await engine.execute_node(
        context=ReplacementContext(),
        node=AttemptNode,
        params={"workflow": graph, "retries": 1, "allow_metered": allow_metered},
        input={"value": 7},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    value = result.output["result"]
    assert isinstance(value, Result)
    error = value.unwrap_err()
    assert error.error_class.root is (
        ErrorClass.TIMEOUT if allow_metered else ErrorClass.VALIDATION
    )
    assert len(MeteredReplacementNode.calls) == (2 if allow_metered else 0)
    if allow_metered:
        assert MeteredReplacementNode.calls == [
            "node/try_0/node/replacement_0",
            "node/next/try_1/node/replacement_0",
        ]


async def test_failed_boundary_blocks_replacement_waiting_for_quota():
    """A replacement that is still queued for admission when its boundary
    fails must be cancelled without ever running its body or leaking the
    lease it was waiting on. Admission is context-owned (see #272), so the
    queue is a real coordinator pool held externally, not a mocked limiter.
    """
    coordinator = InMemoryLimitCoordinator()
    algorithm = ParallelExecutionAlgorithm()
    engine = WorkflowEngine(execution_algorithm=algorithm)

    request = LimitPolicy().resolve(
        QueuedReplacementNode,
        "execution",
        QueuedReplacementNode.TYPE_INFO.execution_limits,
    )
    assert request is not None
    held = await coordinator.acquire((request,), invocation_id="external")

    class QueuedContext(ReplacementContext):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.cancelled_ids: list[str] = []

        @override
        async def on_node_replace(
            self, *, node, replacement, input, replacement_info
        ) -> None:
            await super().on_node_replace(
                node=node,
                replacement=replacement,
                input=input,
                replacement_info=replacement_info,
            )
            BlockingReplacementNode.started.set()

        @override
        async def on_node_cancelled(
            self, *, node, input_type, output_type, input, boundary_id, reason, cause
        ) -> None:
            self.cancelled_ids.append(node.id)

    parent = await queued_replacement_parent(engine)
    boom = engine.create_node(FailAfterAdmissionNode, id="boom")
    graph = workflow(
        engine,
        {"value": IntegerValue},
        {"value": IntegerValue},
        [parent, boom],
        [
            edge("input", "value", "delegate", "value"),
            edge("delegate", "value", "output", "value"),
        ],
    )
    context = QueuedContext(limit_coordinator=coordinator)
    try:
        result = await asyncio.wait_for(
            engine.execute_node(
                context=context,
                node=AttemptNode,
                params={"workflow": graph},
                input={"value": 7},
            ),
            timeout=5,
        )
        assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
        assert context.cancelled_ids.count("node/delegate/replacement_0") == 1
        # The queued admission attempt must never reach the node body: no
        # quota was consumed and abandoned, only ever waited-on-then-dropped.
        assert QueuedReplacementNode.calls == []
        assert (await coordinator.inspect()).waiting == ()
    finally:
        await coordinator.release(held)
