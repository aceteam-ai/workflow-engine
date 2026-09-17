"""Control signals from deferred callers resume completion, never child work."""

import asyncio
from datetime import timedelta

import pytest
from overrides import override

from tests.test_node_replacement import (
    DelegationProbeNode,
    ReplacementContext,
    ReplacementLeafNode,
)
from workflow_engine import (
    DataMapping,
    ReplacementFrame,
    Result,
    ShouldRetry,
    ShouldYield,
    StakeholderLevel,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.nodes import AttemptNode

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def reset():
    DelegationProbeNode.calls = []
    ReplacementLeafNode.calls = []


class DeferredControlContext(ReplacementContext):
    def __init__(self, signals, *, frames=None, cache=None, backoff=timedelta(0)):
        super().__init__(frames, cache)
        self.signals = list(signals)
        self.backoff = backoff
        self.retries = []
        self.yields = []
        self.retry_scheduled = asyncio.Event()

    @override
    async def on_node_finish(
        self, *, node, input_type, output_type, input, output
    ) -> DataMapping:
        if isinstance(node, DelegationProbeNode) and self.signals:
            signal = self.signals.pop(0)
            if signal == "yield":
                raise ShouldYield("Waiting for caller publication")
            raise ShouldRetry(
                "Retry caller publication",
                node=node,
                level=StakeholderLevel.OPERATOR,
                backoff=self.backoff,
            )
        return await super().on_node_finish(
            node=node,
            input_type=input_type,
            output_type=output_type,
            input=input,
            output=output,
        )

    @override
    async def on_node_retry(
        self, *, node, input_type, output_type, input, exception, attempt
    ) -> None:
        self.retries.append((node.id, exception.node_id, attempt))
        self.retry_scheduled.set()

    @override
    async def on_node_yield(
        self, *, node, input_type, output_type, input, exception
    ) -> None:
        self.yields.append(node.id)


async def execute(algorithm, context, *, retries=1, child_retries=0, attempted=False):
    engine = WorkflowEngine(execution_algorithm=algorithm)
    node = engine.create_node(
        DelegationProbeNode, id="node", params={"child_retries": child_retries}
    ).model_copy(update={"max_retries": retries})
    graph = await engine.build_single_node_workflow(
        DelegationProbeNode, params=node.params
    )
    graph = graph.model_copy(update={"inner_nodes": [node]})
    if attempted:
        return await engine.execute_node(
            context=context,
            node=AttemptNode,
            params={"workflow": graph},
            input={"value": 7},
        )
    return await engine.execute(context=context, workflow=graph, input={"value": 7})


async def test_deferred_finish_retry_uses_caller_budget_without_redispatch(algorithm):
    context = DeferredControlContext(["retry"])
    result = await execute(algorithm, context)
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    assert context.retries == [("node", "node", 1)]
    assert DelegationProbeNode.calls == ["node"]
    assert ReplacementLeafNode.calls == ["node/replacement_0"]
    assert context.events.count(("error", "node")) == 1
    assert context.events.count(("finish", "node")) == 1
    frame = ReplacementFrame.model_validate_json(context.frames["node"])
    assert frame.retries["node"].attempt == 1
    assert frame.status == "completed"


async def test_deferred_finish_yield_resume_keeps_frame_and_cached_child(algorithm):
    context = DeferredControlContext(["yield"])
    result = await execute(algorithm, context)
    assert result.status is WorkflowExecutionResultStatus.YIELDED, result.errors
    assert result.node_yields == {"node": "Waiting for caller publication"}
    assert context.yields == ["node"]
    assert ("error", "node") not in context.events
    assert (
        ReplacementFrame.model_validate_json(context.frames["node"]).status == "pending"
    )
    resumed = DeferredControlContext(
        [], frames=dict(context.frames), cache=dict(context.cache)
    )
    result = await execute(algorithm, resumed)
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    assert DelegationProbeNode.calls == ["node"]
    assert ReplacementLeafNode.calls == ["node/replacement_0"]


async def test_deferred_retry_budget_survives_yield_resume(algorithm):
    context = DeferredControlContext(["retry", "yield"])
    first = await execute(algorithm, context)
    assert first.status is WorkflowExecutionResultStatus.YIELDED, first.errors
    resumed = DeferredControlContext(
        ["retry"], frames=dict(context.frames), cache=dict(context.cache)
    )
    result = await execute(algorithm, resumed)
    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert set(result.errors.node_errors) == {"node"}
    assert not resumed.retries
    assert DelegationProbeNode.calls == ["node"]
    assert ReplacementLeafNode.calls == ["node/replacement_0"]
    assert (
        ReplacementFrame.model_validate_json(resumed.frames["node"]).status == "failed"
    )


async def test_exhausted_deferred_retry_is_contained_by_attempt(algorithm):
    context = DeferredControlContext(["retry"])
    result = await execute(
        algorithm, context, retries=0, child_retries=2, attempted=True
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    value = result.output["result"]
    assert isinstance(value, Result)
    assert value.unwrap_err().node_id.root == "node/node"
    assert not context.retries
    assert DelegationProbeNode.calls == ["node/node"]
    assert ReplacementLeafNode.calls == ["node/node/replacement_0"]


async def test_cancellation_during_deferred_backoff_preserves_pending_checkpoint(
    algorithm,
):
    context = DeferredControlContext(["retry"], backoff=timedelta(days=1))
    task = asyncio.create_task(execute(algorithm, context))
    await asyncio.wait_for(context.retry_scheduled.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    frame = ReplacementFrame.model_validate_json(context.frames["node"])
    assert frame.status == "pending"
    assert frame.retries["node"].attempt == 1
    assert frame.retries["node"].next_retry_at is not None
    assert DelegationProbeNode.calls == ["node"]
    assert ReplacementLeafNode.calls == ["node/replacement_0"]
