"""Durable frame replay and lifecycle overrides for direct delegation."""

import asyncio

import pytest
from overrides import override

from tests.test_node_replacement import (
    DelegationProbeNode,
    ReplacementContext,
    ReplacementData,
    ReplacementLeafNode,
    execute,
)
from workflow_engine import (
    DataMapping,
    ErrorClass,
    IntegerValue,
    Node,
    NodeReplacementException,
    ReplacementFrame,
    StringValue,
    WorkflowEngine,
    WorkflowException,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def reset_probes():
    DelegationProbeNode.calls = []
    ReplacementLeafNode.calls = []


async def test_original_pin_short_circuits_delegation(algorithm):
    context = ReplacementContext(cache={"node": '{"value": 19}'})
    result = await execute(algorithm, context=context)
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    assert result.output["value"] == IntegerValue(19)
    assert not DelegationProbeNode.calls and not ReplacementLeafNode.calls
    assert not context.frames


@pytest.mark.parametrize("pin", ["child", "completed_frame"])
async def test_child_pins_and_completed_frames_preserve_adaptation(algorithm, pin):
    context = ReplacementContext()
    mode = "yield" if pin == "child" else "value"
    first = await execute(algorithm, context=context, params={"mode": mode})
    assert first.status is (
        WorkflowExecutionResultStatus.YIELDED
        if pin == "child"
        else WorkflowExecutionResultStatus.SUCCESS
    )
    cache = {"node/replacement_0": '{"value": 19}'} if pin == "child" else {}
    resumed = ReplacementContext(dict(context.frames), cache)
    result = await execute(algorithm, context=resumed, params={"mode": mode})
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    assert result.output["value"] == IntegerValue(19 if pin == "child" else 7)
    assert DelegationProbeNode.calls == ["node"]
    assert ReplacementLeafNode.calls == ["node/replacement_0"]


@pytest.mark.parametrize("change", ["input", "configuration", "hop", "limit"])
async def test_resume_rejects_stale_or_tampered_checkpoint(algorithm, change):
    context = ReplacementContext()
    await execute(algorithm, context=context, params={"mode": "yield"})
    frames = dict(context.frames)
    if change == "hop":
        frame = ReplacementFrame.model_validate_json(frames["node"])
        frames["node"] = frame.model_copy(update={"hop": 1}).model_dump_json()
    if change == "limit":
        algorithm.max_replacement_hops = 12
    resumed = ReplacementContext(frames)
    result = await WorkflowEngine(execution_algorithm=algorithm).execute_node(
        context=resumed,
        node=DelegationProbeNode,
        params={"mode": "yield", "remaining": 1 if change == "configuration" else 0},
        input={"value": 8 if change == "input" else 7},
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR
    error = result.errors.node_errors["node"][0]
    assert error is not None and error.error_class is ErrorClass.VALIDATION
    assert "mismatch" in error.message
    assert ReplacementLeafNode.calls == ["node/replacement_0"]


class CrashContext(ReplacementContext):
    def __init__(self, where):
        super().__init__()
        self.where = where

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
        if self.where == "installed":
            raise asyncio.CancelledError()

    @override
    async def on_node_replacement_checkpoint(self, *, frame) -> None:
        await super().on_node_replacement_checkpoint(frame=frame)
        if (
            self.where == "adapted"
            and frame.delegator_id == "node/replacement_0"
            and frame.status == "completed"
        ):
            raise asyncio.CancelledError()


@pytest.mark.parametrize("where", ["installed", "adapted"])
async def test_crash_replay_uses_real_json_frames_without_reselecting_or_repeating_work(
    algorithm, where
):
    context = CrashContext(where)
    with pytest.raises(asyncio.CancelledError):
        await execute(algorithm, context=context, params={"remaining": 1})
    calls = list(DelegationProbeNode.calls)
    resumed = ReplacementContext(dict(context.frames))
    result = await execute(algorithm, context=resumed, params={"remaining": 1})
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    assert result.output["value"] == IntegerValue(7)
    assert DelegationProbeNode.calls.count("node") == 1
    if where == "adapted":
        assert DelegationProbeNode.calls == calls
    assert ReplacementLeafNode.calls == ["node/replacement_1"]
    assert resumed.events.count(("finish", "node")) == 1


async def test_long_chain_is_iterative_and_admission_remains_cancellable(algorithm):
    algorithm.max_replacement_hops = 350
    result = await execute(
        algorithm, context=InMemoryExecutionContext(), params={"remaining": 300}
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    assert len(DelegationProbeNode.calls) == 301
    assert ReplacementLeafNode.calls == ["node/replacement_300"]


class OverrideContext(ReplacementContext):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    @override
    async def on_node_start(
        self, *, node, input_type, output_type, input
    ) -> DataMapping | Node | None:
        if self.mode == "bad_pin" and node.id == "node":
            return {"value": StringValue("invalid")}
        return await super().on_node_start(
            node=node, input_type=input_type, output_type=output_type, input=input
        )

    @override
    async def on_node_replace(
        self, *, node, replacement, input, replacement_info
    ) -> None:
        if self.mode == "event_failure":
            raise RuntimeError("Durable event was not written")
        await super().on_node_replace(
            node=node,
            replacement=replacement,
            input=input,
            replacement_info=replacement_info,
        )

    @override
    async def on_node_finish(
        self, *, node, input_type, output_type, input, output
    ) -> DataMapping:
        await super().on_node_finish(
            node=node,
            input_type=input_type,
            output_type=output_type,
            input=input,
            output=output,
        )
        if node.id == "node" and self.mode in {"bad_finish", "recover_finish"}:
            return {"value": StringValue("invalid")}
        return output

    @override
    async def on_node_error(
        self, *, node, input_type, output_type, input, exception
    ) -> WorkflowException | DataMapping:
        await super().on_node_error(
            node=node,
            input_type=input_type,
            output_type=output_type,
            input=input,
            exception=exception,
        )
        if (self.mode == "recover_child" and node.id.endswith("replacement_0")) or (
            self.mode == "recover_finish" and node.id == "node"
        ):
            return {"value": IntegerValue(23)}
        return exception


@pytest.mark.parametrize("mode", ["bad_pin", "bad_finish", "event_failure"])
async def test_invalid_hook_outputs_and_event_failures_cannot_publish_success(
    algorithm, mode
):
    context = OverrideContext(mode)
    result = await execute(algorithm, context=context)
    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert set(result.errors.node_errors) == {"node"}
    assert context.events.count(("error", "node")) == 1
    if mode != "bad_finish":
        assert not ReplacementLeafNode.calls


@pytest.mark.parametrize("mode", ["recover_child", "recover_finish"])
async def test_recovery_output_uses_remaining_adapters_once(algorithm, mode):
    context = OverrideContext(mode)
    result = await execute(
        algorithm,
        context=context,
        params={"mode": "fail" if mode == "recover_child" else "value"},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    assert result.output["value"] == IntegerValue(23)
    assert context.events.count(("finish", "node")) == 1


async def test_chained_failure_marks_every_pending_caller_without_duplicate_error_hooks(
    algorithm,
):
    context = ReplacementContext()
    result = await execute(
        algorithm, context=context, params={"mode": "fail", "remaining": 3}
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert [event for event in context.events if event[0] == "error"] == [
        ("error", "node/replacement_3")
    ]
    assert {event[1] for event in context.events if event[0] == "failed"} == {
        "node",
        "node/replacement_0",
        "node/replacement_1",
        "node/replacement_2",
    }


async def test_replacement_retries_use_child_instance_budget(algorithm):
    algorithm.max_retries = 0
    result = await execute(algorithm, params={"mode": "retry", "child_retries": 1})
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    assert DelegationProbeNode.calls == ["node"]
    assert ReplacementLeafNode.calls == ["node/replacement_0", "node/replacement_0"]


async def test_algorithms_must_opt_in_to_replacement_outcomes():
    node = WorkflowEngine().create_node(DelegationProbeNode, id="caller")
    with pytest.raises(NodeReplacementException, match="does not support"):
        await node(
            context=ReplacementContext(),
            input_type=ReplacementData,
            output_type=ReplacementData,
            input={"value": IntegerValue(1)},
        )


@pytest.mark.parametrize(
    "selection", ["raw", "unknown", "newer_version", "invalid_workflow"]
)
async def test_replacement_registry_and_nested_workflow_validation(
    algorithm, selection
):
    from tests.test_node_replacement import DelegationParams
    from tests.test_sequence_ops import workflow
    from workflow_engine.nodes import ForEachNode

    class SelectedContext(ReplacementContext):
        @override
        async def on_node_start(
            self, *, node, input_type, output_type, input
        ) -> DataMapping | Node | None:
            if node.id != "node":
                return await super().on_node_start(
                    node=node,
                    input_type=input_type,
                    output_type=output_type,
                    input=input,
                )
            if selection == "invalid_workflow":
                engine = WorkflowEngine()
                invalid = workflow(
                    engine, {"value": IntegerValue}, {"value": IntegerValue}, [], []
                )
                return engine.create_node(
                    ForEachNode, id="unwired", params={"workflow": invalid}
                )
            return Node.model_construct(
                type="missing_type" if selection == "unknown" else "ReplacementLeaf",
                id="raw-author-label",
                version="999.0.0" if selection == "newer_version" else "1.0.0",
                params=DelegationParams(),
            )

    context = SelectedContext()
    result = await execute(algorithm, context=context)
    if selection == "raw":
        assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
        assert ReplacementLeafNode.calls == ["node/replacement_0"]
        assert not DelegationProbeNode.calls
    else:
        assert result.status is WorkflowExecutionResultStatus.ERROR
        error = result.errors.node_errors["node"][0]
        assert error is not None and error.error_class is ErrorClass.VALIDATION
        expected = {
            "unknown": "not registered",
            "newer_version": "newer",
            "invalid_workflow": "required input fields",
        }[selection]
        assert expected in error.message
        assert not ReplacementLeafNode.calls


async def test_reserved_control_record_prevents_replacement_id_reuse():
    from workflow_engine.core.replacement import NodeReplacement
    from workflow_engine.execution.boundary import BoundaryTracker
    from workflow_engine.execution.replacement import (
        ReplacementGraph,
        ReplacementTracker,
    )
    from workflow_engine.execution.retry import RetryTracker

    engine = WorkflowEngine()
    validated = await engine.validate(
        await engine.build_single_node_workflow(DelegationProbeNode)
    )
    graph = ReplacementGraph.model_validate(
        {key: getattr(validated, key) for key in type(validated).model_fields}
    )
    tracker = ReplacementTracker(256)
    tracker.occupied.add("node/replacement_0")
    context = ReplacementContext()
    child = engine.create_node(ReplacementLeafNode, id="author")
    with pytest.raises(NodeReplacementException, match="already occupied"):
        await tracker.install(
            graph,
            graph.nodes_by_id["node"],
            NodeReplacement(child, {"value": IntegerValue(7)}),
            context,
            RetryTracker(),
            BoundaryTracker(),
        )
    assert not context.frames
    assert not ReplacementLeafNode.calls
