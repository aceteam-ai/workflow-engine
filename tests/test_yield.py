# tests/test_yield.py
"""
Tests for ShouldYield / WorkflowYield execution behaviour.

Focus: verifying that execution algorithms make as much forward progress as
possible before raising WorkflowYield — i.e., nodes independent of a yielding
node still run, multiple yields are collected, and WorkflowYield is only raised
once all runnable nodes are exhausted.

Tests run against both TopologicalExecutionAlgorithm and
ParallelExecutionAlgorithm via the `algorithm` fixture.
"""

from functools import cached_property
from typing import ClassVar, Literal

import pytest

from workflow_engine import (
    Context,
    Data,
    Edge,
    ExecutionAlgorithm,
    Node,
    Params,
    ShouldYield,
    StringValue,
    Workflow,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryContext
from workflow_engine.core import NodeTypeInfo
from workflow_engine.core.io import InputNode, OutputNode
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.execution.parallel import ParallelExecutionAlgorithm
from workflow_engine.nodes import ConstantStringNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleInput(Data):
    value: StringValue


class SimpleOutput(Data):
    result: StringValue


class EchoNode(Node[SimpleInput, SimpleOutput, Params]):
    """Passes its input straight through — used to confirm a branch ran."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="YieldTestEcho",
        display_name="YieldTestEcho",
        description="Echo node for yield tests.",
        version="1.0.0",
        parameter_type=Params,
    )
    type: Literal["YieldTestEcho"] = "YieldTestEcho"  # pyright: ignore[reportIncompatibleVariableOverride]

    ran: ClassVar[set[str]] = set()

    @cached_property
    def input_type(self):
        return SimpleInput

    @cached_property
    def output_type(self):
        return SimpleOutput

    async def run(self, context: Context, input: SimpleInput) -> SimpleOutput:
        EchoNode.ran.add(self.id)
        return SimpleOutput(result=input.value)


class YieldingNode(Node[SimpleInput, SimpleOutput, Params]):
    """Always yields with a configurable message."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="YieldTestYielding",
        display_name="YieldTestYielding",
        description="Yielding node for yield tests.",
        version="1.0.0",
        parameter_type=Params,
    )
    type: Literal["YieldTestYielding"] = "YieldTestYielding"  # pyright: ignore[reportIncompatibleVariableOverride]

    message: str = "yielding"
    calls: ClassVar[dict[str, int]] = {}

    @cached_property
    def input_type(self):
        return SimpleInput

    @cached_property
    def output_type(self):
        return SimpleOutput

    async def run(self, context: Context, input: SimpleInput) -> SimpleOutput:
        YieldingNode.calls[self.id] = YieldingNode.calls.get(self.id, 0) + 1
        raise ShouldYield(self.message)


class ResumableNode(Node[SimpleInput, SimpleOutput, Params]):
    """Yields on the first call, succeeds on the second — simulating a node
    that dispatches external work and then checks for completion on resume."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="YieldTestResumable",
        display_name="YieldTestResumable",
        description="Resumable node for yield tests.",
        version="1.0.0",
        parameter_type=Params,
    )
    type: Literal["YieldTestResumable"] = "YieldTestResumable"  # pyright: ignore[reportIncompatibleVariableOverride]

    calls: ClassVar[dict[str, int]] = {}

    @cached_property
    def input_type(self):
        return SimpleInput

    @cached_property
    def output_type(self):
        return SimpleOutput

    async def run(self, context: Context, input: SimpleInput) -> SimpleOutput:
        n = ResumableNode.calls.get(self.id, 0) + 1
        ResumableNode.calls[self.id] = n
        if n == 1:
            raise ShouldYield("waiting for external work")
        return SimpleOutput(result=StringValue(f"resumed: {input.value.root}"))


@pytest.fixture(autouse=True)
def reset_echo_ran():
    EchoNode.ran = set()
    YieldingNode.calls = {}
    ResumableNode.calls = {}
    yield


@pytest.fixture(params=["topological", "parallel"])
def algorithm(request) -> ExecutionAlgorithm:
    if request.param == "topological":
        return TopologicalExecutionAlgorithm()
    else:
        return ParallelExecutionAlgorithm()


# ---------------------------------------------------------------------------
# Workflow builders
# ---------------------------------------------------------------------------


def _fan_out_workflow(
    *,
    yielding_ids: list[str],
    echo_ids: list[str],
) -> tuple[Workflow, OutputNode]:
    """
    Build a workflow where a single constant fans out to N yielding nodes and
    M echo nodes, all of whose outputs feed into a single output node.

        constant ──> yield_a ──> output.yield_a
                 ──> yield_b ──> output.yield_b
                 ──> echo_a  ──> output.echo_a
                 ──> echo_b  ──> output.echo_b
    """
    input_node = InputNode.empty()
    all_keys = {node_id: StringValue for node_id in yielding_ids + echo_ids}
    output_node = OutputNode.from_fields(**all_keys)
    constant = ConstantStringNode.from_value(id="constant", value="hello")

    inner_nodes: list[Node] = [constant]
    edges: list[Edge] = []

    for nid in yielding_ids:
        node = YieldingNode(id=nid, params=Params(), message=f"waiting: {nid}")
        inner_nodes.append(node)
        edges += [
            Edge.from_nodes(
                source=constant, source_key="value", target=node, target_key="value"
            ),
            Edge.from_nodes(
                source=node, source_key="result", target=output_node, target_key=nid
            ),
        ]

    for nid in echo_ids:
        node = EchoNode(id=nid, params=Params())
        inner_nodes.append(node)
        edges += [
            Edge.from_nodes(
                source=constant, source_key="value", target=node, target_key="value"
            ),
            Edge.from_nodes(
                source=node, source_key="result", target=output_node, target_key=nid
            ),
        ]

    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=inner_nodes,
        edges=edges,
    )
    return workflow, output_node


def _chain_workflow(*, yield_first: bool) -> Workflow:
    """
    A linear chain: constant -> node_a -> node_b -> output.

    If yield_first=True, node_a yields and node_b never runs.
    If yield_first=False, node_a echoes and node_b yields.
    """
    input_node = InputNode.empty()
    output_node = OutputNode.from_fields(result=StringValue)
    constant = ConstantStringNode.from_value(id="constant", value="hello")

    node_a: Node
    node_b: Node
    if yield_first:
        node_a = YieldingNode(id="node_a", params=Params(), message="node_a yielding")
        node_b = EchoNode(id="node_b", params=Params())
    else:
        node_a = EchoNode(id="node_a", params=Params())
        node_b = YieldingNode(id="node_b", params=Params(), message="node_b yielding")

    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[constant, node_a, node_b],
        edges=[
            Edge.from_nodes(
                source=constant, source_key="value", target=node_a, target_key="value"
            ),
            Edge.from_nodes(
                source=node_a, source_key="result", target=node_b, target_key="value"
            ),
            Edge.from_nodes(
                source=node_b,
                source_key="result",
                target=output_node,
                target_key="result",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestYieldContinuesProgress:
    @pytest.mark.asyncio
    async def test_independent_nodes_run_after_yield(
        self, algorithm: ExecutionAlgorithm
    ):
        """Nodes in independent branches still execute when another branch yields."""
        workflow, _ = _fan_out_workflow(yielding_ids=["y"], echo_ids=["e"])
        context = InMemoryContext()

        result = await algorithm.execute(context=context, workflow=workflow, input={})

        assert result.status is WorkflowExecutionResultStatus.YIELDED
        assert "e" in EchoNode.ran
        # Yielding node must run exactly once — never re-dispatched
        assert YieldingNode.calls.get("y") == 1

    @pytest.mark.asyncio
    async def test_downstream_nodes_do_not_run_after_yield(
        self, algorithm: ExecutionAlgorithm
    ):
        """A node whose only dependency yielded must not execute."""
        workflow = _chain_workflow(yield_first=True)
        context = InMemoryContext()

        result = await algorithm.execute(context=context, workflow=workflow, input={})
        assert result.status is WorkflowExecutionResultStatus.YIELDED

        assert "node_b" not in EchoNode.ran

    @pytest.mark.asyncio
    async def test_upstream_completion_runs_downstream_before_yield(
        self,
        algorithm: ExecutionAlgorithm,
    ):
        """When an earlier node succeeds, its downstream runs even if a sibling yields."""
        workflow = _chain_workflow(yield_first=False)
        context = InMemoryContext()

        result = await algorithm.execute(context=context, workflow=workflow, input={})
        assert result.status is WorkflowExecutionResultStatus.YIELDED

        # node_a (echo) ran; node_b (yield) yielded
        assert "node_a" in EchoNode.ran


class TestMultipleYields:
    @pytest.mark.asyncio
    async def test_all_yielding_nodes_collected(
        self,
        algorithm: ExecutionAlgorithm,
    ):
        """WorkflowYield contains every node that yielded, not just the first."""
        workflow, _ = _fan_out_workflow(yielding_ids=["y1", "y2", "y3"], echo_ids=[])
        context = InMemoryContext()

        result = await algorithm.execute(context=context, workflow=workflow, input={})
        assert result.status is WorkflowExecutionResultStatus.YIELDED

        assert set(result.node_yields.keys()) == {"y1", "y2", "y3"}

    @pytest.mark.asyncio
    async def test_yield_messages_preserved(self, algorithm: ExecutionAlgorithm):
        """Each yielded node's message is available in WorkflowYield.node_yields."""
        workflow, _ = _fan_out_workflow(yielding_ids=["y1", "y2"], echo_ids=[])
        context = InMemoryContext()

        result = await algorithm.execute(context=context, workflow=workflow, input={})
        assert result.status is WorkflowExecutionResultStatus.YIELDED

        assert result.node_yields["y1"] == "waiting: y1"
        assert result.node_yields["y2"] == "waiting: y2"

    @pytest.mark.asyncio
    async def test_mixed_yield_and_success(self, algorithm: ExecutionAlgorithm):
        """When some nodes yield and others succeed, both outcomes are correct."""
        workflow, _ = _fan_out_workflow(
            yielding_ids=["y1", "y2"], echo_ids=["e1", "e2"]
        )
        context = InMemoryContext()

        result = await algorithm.execute(context=context, workflow=workflow, input={})
        assert result.status is WorkflowExecutionResultStatus.YIELDED

        # Yielded nodes are captured
        assert set(result.node_yields.keys()) == {"y1", "y2"}
        # Successful nodes ran
        assert {"e1", "e2"}.issubset(EchoNode.ran)

    @pytest.mark.asyncio
    async def test_workflow_yield_not_raised_if_no_nodes_yield(
        self,
        algorithm: ExecutionAlgorithm,
    ):
        """WorkflowYield is not raised when all nodes complete normally."""
        workflow, _ = _fan_out_workflow(yielding_ids=[], echo_ids=["e1", "e2"])
        context = InMemoryContext()

        result = await algorithm.execute(context=context, workflow=workflow, input={})

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert {"e1", "e2"}.issubset(EchoNode.ran)


class TestResumption:
    def _resumable_workflow(self) -> Workflow:
        """constant -> resumable -> output"""
        input_node = InputNode.empty()
        output_node = OutputNode.from_fields(result=StringValue)
        constant = ConstantStringNode.from_value(id="constant", value="hello")
        node = ResumableNode(id="resumable", params=Params())
        return Workflow(
            input_node=input_node,
            output_node=output_node,
            inner_nodes=[constant, node],
            edges=[
                Edge.from_nodes(
                    source=constant, source_key="value", target=node, target_key="value"
                ),
                Edge.from_nodes(
                    source=node,
                    source_key="result",
                    target=output_node,
                    target_key="result",
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_first_run_yields(
        self,
        algorithm: ExecutionAlgorithm,
    ):
        """The first execution raises WorkflowYield."""
        workflow = self._resumable_workflow()
        context = InMemoryContext()

        result = await algorithm.execute(context=context, workflow=workflow, input={})
        assert result.status is WorkflowExecutionResultStatus.YIELDED

        assert "resumable" in result.node_yields
        assert ResumableNode.calls["resumable"] == 1

    @pytest.mark.asyncio
    async def test_second_run_succeeds(
        self,
        algorithm: ExecutionAlgorithm,
    ):
        """Re-running the same workflow with the same context lets the node succeed."""
        workflow = self._resumable_workflow()
        context = InMemoryContext()

        # First run — yields
        result = await algorithm.execute(context=context, workflow=workflow, input={})
        assert result.status is WorkflowExecutionResultStatus.YIELDED

        # Second run — node finds its work complete and returns a result
        result = await algorithm.execute(context=context, workflow=workflow, input={})

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.output == {"result": StringValue("resumed: hello")}
        assert ResumableNode.calls["resumable"] == 2

    @pytest.mark.asyncio
    async def test_partial_resumption(self, algorithm: ExecutionAlgorithm):
        """When only some yielded nodes are ready, the rest yield again."""
        input_node = InputNode.empty()
        output_node = OutputNode.from_fields(a=StringValue, b=StringValue)
        constant = ConstantStringNode.from_value(id="constant", value="x")
        # r_a will succeed on second call; r_b always yields
        r_a = ResumableNode(id="r_a", params=Params())
        r_b = YieldingNode(id="r_b", params=Params(), message="still waiting")
        workflow = Workflow(
            input_node=input_node,
            output_node=output_node,
            inner_nodes=[constant, r_a, r_b],
            edges=[
                Edge.from_nodes(
                    source=constant, source_key="value", target=r_a, target_key="value"
                ),
                Edge.from_nodes(
                    source=constant, source_key="value", target=r_b, target_key="value"
                ),
                Edge.from_nodes(
                    source=r_a, source_key="result", target=output_node, target_key="a"
                ),
                Edge.from_nodes(
                    source=r_b, source_key="result", target=output_node, target_key="b"
                ),
            ],
        )
        context = InMemoryContext()

        # First run — both yield
        result = await algorithm.execute(context=context, workflow=workflow, input={})
        assert result.status is WorkflowExecutionResultStatus.YIELDED
        assert set(result.node_yields.keys()) == {"r_a", "r_b"}

        # Second run — r_a succeeds, r_b yields again
        result = await algorithm.execute(context=context, workflow=workflow, input={})
        assert result.status is WorkflowExecutionResultStatus.YIELDED
        assert set(result.node_yields.keys()) == {"r_b"}
        assert ResumableNode.calls["r_a"] == 2
