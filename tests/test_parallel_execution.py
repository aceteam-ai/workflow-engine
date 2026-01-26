# tests/test_parallel_execution.py

import asyncio
from typing import ClassVar, Literal, Type

import pytest

from workflow_engine import (
    Context,
    Data,
    Edge,
    Empty,
    InputEdge,
    IntegerValue,
    Node,
    OutputEdge,
    Params,
    Workflow,
)
from workflow_engine.core.node import NodeTypeInfo
from workflow_engine.contexts import InMemoryContext
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.execution.parallel import (
    ErrorHandlingMode,
    ParallelExecutionAlgorithm,
)
from workflow_engine.nodes import (
    AddNode,
    ConstantIntegerNode,
    ConstantStringNode,
    ErrorNode,
    ForEachNode,
)


# Test node that sleeps for a configurable duration
class SlowNodeParams(Params):
    delay_ms: IntegerValue


class SlowNodeOutput(Data):
    value: IntegerValue


class SlowNode(Node[Empty, SlowNodeOutput, SlowNodeParams]):
    """A node that sleeps for a configurable duration."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="SlowNode",
        display_name="Slow Node",
        description="A node that takes time to execute",
        version="1.0.0",
        parameter_type=SlowNodeParams,
    )

    type: Literal["SlowNode"] = "SlowNode"  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def input_type(self) -> Type[Empty]:
        return Empty

    @property
    def output_type(self) -> Type[SlowNodeOutput]:
        return SlowNodeOutput

    async def run(self, context: Context, input: Empty) -> SlowNodeOutput:
        await asyncio.sleep(self.params.delay_ms.root / 1000)
        return SlowNodeOutput(value=self.params.delay_ms)

    @classmethod
    def from_delay(cls, id: str, delay_ms: int) -> "SlowNode":
        return cls(id=id, params=SlowNodeParams(delay_ms=IntegerValue(delay_ms)))


# Test node that accepts input (to create dependencies) and delays
class SlowPassthroughNodeInput(Data):
    value: IntegerValue


class SlowPassthroughNode(Node[SlowPassthroughNodeInput, SlowNodeOutput, SlowNodeParams]):
    """A node that accepts input, delays, and outputs the input value."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="SlowPassthroughNode",
        display_name="Slow Passthrough Node",
        description="A node that accepts input and delays before outputting it",
        version="1.0.0",
        parameter_type=SlowNodeParams,
    )

    type: Literal["SlowPassthroughNode"] = "SlowPassthroughNode"  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def input_type(self) -> Type[SlowPassthroughNodeInput]:
        return SlowPassthroughNodeInput

    @property
    def output_type(self) -> Type[SlowNodeOutput]:
        return SlowNodeOutput

    async def run(
        self, context: Context, input: SlowPassthroughNodeInput
    ) -> SlowNodeOutput:
        await asyncio.sleep(self.params.delay_ms.root / 1000)
        return SlowNodeOutput(value=input.value)

    @classmethod
    def from_delay(cls, id: str, delay_ms: int) -> "SlowPassthroughNode":
        return cls(id=id, params=SlowNodeParams(delay_ms=IntegerValue(delay_ms)))


@pytest.fixture
def parallel_workflow() -> Workflow:
    """Create a workflow with independent nodes that can run in parallel."""
    return Workflow(
        nodes=[
            SlowNode.from_delay(id="node_a", delay_ms=100),
            SlowNode.from_delay(id="node_b", delay_ms=100),
            SlowNode.from_delay(id="node_c", delay_ms=100),
        ],
        edges=[],
        input_edges=[],
        output_edges=[
            OutputEdge(source_id="node_a", source_key="value", output_key="a"),
            OutputEdge(source_id="node_b", source_key="value", output_key="b"),
            OutputEdge(source_id="node_c", source_key="value", output_key="c"),
        ],
    )


@pytest.mark.asyncio
async def test_parallel_execution_faster_than_sequential(
    parallel_workflow: Workflow,
):
    """Test that parallel execution is actually parallel (faster than sequential)."""
    context = InMemoryContext()

    # Sequential execution
    sequential_algo = TopologicalExecutionAlgorithm()
    start = asyncio.get_event_loop().time()
    errors, _ = await sequential_algo.execute(
        context=context,
        workflow=parallel_workflow,
        input={},
    )
    sequential_time = asyncio.get_event_loop().time() - start
    assert not errors.any()

    # Parallel execution
    parallel_algo = ParallelExecutionAlgorithm()
    start = asyncio.get_event_loop().time()
    errors, _ = await parallel_algo.execute(
        context=context,
        workflow=parallel_workflow,
        input={},
    )
    parallel_time = asyncio.get_event_loop().time() - start
    assert not errors.any()

    # Parallel should be significantly faster (3 x 100ms sequentially vs ~100ms parallel)
    # Using a generous margin to avoid flaky tests
    assert parallel_time < sequential_time * 0.6, (
        f"Parallel ({parallel_time:.3f}s) should be much faster than "
        f"sequential ({sequential_time:.3f}s)"
    )


@pytest.mark.asyncio
async def test_parallel_execution_respects_dependencies():
    """Test that dependent nodes wait for their dependencies."""
    workflow = Workflow(
        nodes=[
            a := ConstantIntegerNode.from_value(id="a", value=1),
            b := ConstantIntegerNode.from_value(id="b", value=2),
            c := AddNode(id="c"),  # depends on a and b
        ],
        edges=[
            Edge.from_nodes(source=a, source_key="value", target=c, target_key="a"),
            Edge.from_nodes(source=b, source_key="value", target=c, target_key="b"),
        ],
        input_edges=[],
        output_edges=[
            OutputEdge.from_node(source=c, source_key="sum", output_key="result"),
        ],
    )

    context = InMemoryContext()
    algorithm = ParallelExecutionAlgorithm()

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={},
    )

    assert not errors.any()
    assert output == {"result": IntegerValue(3)}


@pytest.mark.asyncio
async def test_parallel_execution_complex_dependencies():
    r"""Test parallel execution with a more complex dependency graph.

    Graph structure:
        a   b   d
         \ /    |
          c     |
           \   /
             f

    Nodes a, b, d can run in parallel (first batch).
    Node c runs after a, b complete.
    Node f runs after c, d complete.
    """
    workflow = Workflow(
        nodes=[
            a := ConstantIntegerNode.from_value(id="a", value=1),
            b := ConstantIntegerNode.from_value(id="b", value=2),
            c := AddNode(id="c"),
            d := ConstantIntegerNode.from_value(id="d", value=10),
            f := AddNode(id="f"),
        ],
        edges=[
            Edge.from_nodes(source=a, source_key="value", target=c, target_key="a"),
            Edge.from_nodes(source=b, source_key="value", target=c, target_key="b"),
            Edge.from_nodes(source=c, source_key="sum", target=f, target_key="a"),
            Edge.from_nodes(source=d, source_key="value", target=f, target_key="b"),
        ],
        input_edges=[],
        output_edges=[
            OutputEdge.from_node(source=f, source_key="sum", output_key="result"),
        ],
    )

    context = InMemoryContext()
    algorithm = ParallelExecutionAlgorithm()

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={},
    )

    assert not errors.any()
    # a + b = 3, then 3 + d(10) = 13
    assert output == {"result": IntegerValue(13)}


@pytest.mark.asyncio
async def test_parallel_execution_continue_on_error():
    """Test CONTINUE error handling mode collects all errors."""
    from workflow_engine import StringValue

    workflow = Workflow(
        nodes=[
            ok_node := ConstantStringNode.from_value(id="ok_node", value="test"),
            error1 := ErrorNode.from_name(id="error1", name="Error1"),
            error2 := ErrorNode.from_name(id="error2", name="Error2"),
        ],
        edges=[
            # Both error nodes depend on ok_node, so they run in parallel after it
            Edge.from_nodes(
                source=ok_node, source_key="value", target=error1, target_key="info"
            ),
            Edge.from_nodes(
                source=ok_node, source_key="value", target=error2, target_key="info"
            ),
        ],
        input_edges=[],
        output_edges=[
            OutputEdge.from_node(
                source=ok_node, source_key="value", output_key="value"
            ),
        ],
    )

    context = InMemoryContext()
    algorithm = ParallelExecutionAlgorithm(error_handling=ErrorHandlingMode.CONTINUE)

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={},
    )

    # Should have collected errors from both error nodes
    assert errors.any()
    assert len(errors.node_errors) == 2
    assert "error1" in errors.node_errors
    assert "error2" in errors.node_errors
    # ok_node output should still be available in partial output
    assert output == {"value": StringValue("test")}


@pytest.mark.asyncio
async def test_parallel_execution_fail_fast():
    """Test FAIL_FAST error handling mode stops on first error."""
    workflow = Workflow(
        nodes=[
            constant := ConstantStringNode.from_value(id="constant", value="test"),
            error := ErrorNode.from_name(id="error", name="TestError"),
        ],
        edges=[
            Edge.from_nodes(
                source=constant, source_key="value", target=error, target_key="info"
            ),
        ],
        input_edges=[],
        output_edges=[
            OutputEdge.from_node(
                source=constant, source_key="value", output_key="value"
            ),
        ],
    )

    context = InMemoryContext()
    algorithm = ParallelExecutionAlgorithm(error_handling=ErrorHandlingMode.FAIL_FAST)

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={},
    )

    assert errors.any()


@pytest.mark.asyncio
async def test_parallel_execution_with_node_expansion():
    """Test that node expansion works correctly with parallel execution."""
    add_workflow = Workflow(
        nodes=[AddNode(id="add")],
        edges=[],
        input_edges=[
            InputEdge(input_key="a", target_id="add", target_key="a"),
            InputEdge(input_key="b", target_id="add", target_key="b"),
        ],
        output_edges=[
            OutputEdge(source_id="add", source_key="sum", output_key="c"),
        ],
    )

    workflow = Workflow(
        nodes=[
            ForEachNode.from_workflow(id="foreach", workflow=add_workflow),
        ],
        edges=[],
        input_edges=[
            InputEdge(input_key="sequence", target_id="foreach", target_key="sequence"),
        ],
        output_edges=[
            OutputEdge(source_id="foreach", source_key="sequence", output_key="results"),
        ],
    )

    context = InMemoryContext()
    algorithm = ParallelExecutionAlgorithm()

    input_data = workflow.input_type.model_validate(
        {
            "sequence": [
                {"a": 1.0, "b": 2.0},
                {"a": 3.0, "b": 4.0},
            ]
        }
    ).to_dict()

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input=input_data,
    )

    assert not errors.any(), errors
    expected = workflow.output_type.model_validate(
        {"results": [{"c": 3.0}, {"c": 7.0}]}
    ).to_dict()
    assert output == expected


@pytest.mark.asyncio
async def test_parallel_execution_max_concurrency():
    """Test that max_concurrency limits parallel execution."""
    # Create 5 slow nodes that each take 50ms
    nodes = [SlowNode.from_delay(id=f"node_{i}", delay_ms=50) for i in range(5)]

    workflow = Workflow(
        nodes=nodes,
        edges=[],
        input_edges=[],
        output_edges=[
            OutputEdge(source_id=f"node_{i}", source_key="value", output_key=f"out_{i}")
            for i in range(5)
        ],
    )

    context = InMemoryContext()

    # With max_concurrency=2, 5 nodes taking 50ms each should take at least 150ms
    # (3 batches: 2+2+1)
    algorithm = ParallelExecutionAlgorithm(max_concurrency=2)

    start = asyncio.get_event_loop().time()
    errors, _ = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={},
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert not errors.any()
    # Should take at least 3 batches worth of time (with some tolerance)
    assert elapsed >= 0.12, f"Expected at least 120ms, got {elapsed * 1000:.0f}ms"


@pytest.mark.asyncio
async def test_parallel_execution_empty_workflow():
    """Test that parallel execution handles empty workflows."""
    workflow = Workflow(
        nodes=[],
        edges=[],
        input_edges=[],
        output_edges=[],
    )

    context = InMemoryContext()
    algorithm = ParallelExecutionAlgorithm()

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={},
    )

    assert not errors.any()
    assert output == {}


@pytest.mark.asyncio
async def test_parallel_execution_single_node():
    """Test that parallel execution works with a single node."""
    workflow = Workflow(
        nodes=[
            a := ConstantIntegerNode.from_value(id="a", value=42),
        ],
        edges=[],
        input_edges=[],
        output_edges=[
            OutputEdge.from_node(source=a, source_key="value", output_key="result"),
        ],
    )

    context = InMemoryContext()
    algorithm = ParallelExecutionAlgorithm()

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={},
    )

    assert not errors.any()
    assert output == {"result": IntegerValue(42)}


@pytest.mark.asyncio
async def test_parallel_execution_matches_sequential_output():
    """Test that parallel execution produces the same output as sequential."""
    workflow = Workflow(
        nodes=[
            a := ConstantIntegerNode.from_value(id="a", value=42),
            b := ConstantIntegerNode.from_value(id="b", value=2025),
            a_plus_b := AddNode(id="a+b"),
            a_plus_b_plus_c := AddNode(id="a+b+c"),
        ],
        edges=[
            Edge.from_nodes(
                source=a, source_key="value", target=a_plus_b, target_key="a"
            ),
            Edge.from_nodes(
                source=b, source_key="value", target=a_plus_b, target_key="b"
            ),
            Edge.from_nodes(
                source=a_plus_b, source_key="sum", target=a_plus_b_plus_c, target_key="a"
            ),
        ],
        input_edges=[
            InputEdge.from_node(
                input_key="c", target=a_plus_b_plus_c, target_key="b"
            ),
        ],
        output_edges=[
            OutputEdge.from_node(
                source=a_plus_b_plus_c, source_key="sum", output_key="sum"
            ),
        ],
    )

    context = InMemoryContext()
    input_data = {"c": IntegerValue(-256)}

    # Sequential execution
    sequential_algo = TopologicalExecutionAlgorithm()
    seq_errors, seq_output = await sequential_algo.execute(
        context=context,
        workflow=workflow,
        input=input_data,
    )

    # Parallel execution
    parallel_algo = ParallelExecutionAlgorithm()
    par_errors, par_output = await parallel_algo.execute(
        context=context,
        workflow=workflow,
        input=input_data,
    )

    assert not seq_errors.any()
    assert not par_errors.any()
    assert seq_output == par_output
    assert par_output == {"sum": 42 + 2025 + (-256)}


@pytest.mark.asyncio
async def test_parallel_execution_eager_dispatch():
    """Test that dependent nodes start immediately when dependencies complete.

    Graph:
        A (50ms) -----> C (50ms passthrough)
        B (200ms) (independent)

    With eager dispatch: A finishes at 50ms, C starts immediately and finishes at ~100ms.
    B finishes at 200ms. Total time ~200ms.

    With batch dispatch: A+B batch completes at 200ms, then C runs, finishes at 250ms.
    """
    workflow = Workflow(
        nodes=[
            a := SlowNode.from_delay(id="a", delay_ms=50),
            SlowNode.from_delay(id="b", delay_ms=200),  # Not referenced, only used by id
            c := SlowPassthroughNode.from_delay(id="c", delay_ms=50),
        ],
        edges=[
            Edge.from_nodes(source=a, source_key="value", target=c, target_key="value"),
        ],
        input_edges=[],
        output_edges=[
            OutputEdge(source_id="a", source_key="value", output_key="a"),
            OutputEdge(source_id="b", source_key="value", output_key="b"),
            OutputEdge(source_id="c", source_key="value", output_key="c"),
        ],
    )

    context = InMemoryContext()
    algorithm = ParallelExecutionAlgorithm()

    start = asyncio.get_event_loop().time()
    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={},
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert not errors.any()
    # Eager dispatch: ~200ms (B is bottleneck, C finishes at 100ms)
    # Batch dispatch would be: ~250ms (A+B at 200ms, then C at 250ms)
    # Using generous margin to avoid flaky tests
    assert elapsed < 0.23, (
        f"Expected ~200ms with eager dispatch, got {elapsed*1000:.0f}ms. "
        f"This suggests batch-based execution instead of eager dispatch."
    )
