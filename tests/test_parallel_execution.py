# tests/test_parallel_execution.py

import asyncio
from typing import ClassVar, Type

import pytest
from overrides import override

from workflow_engine import (
    Data,
    DataValue,
    Edge,
    Empty,
    ExecutionContext,
    FloatValue,
    IntegerValue,
    Node,
    NodeTypeInfo,
    Params,
    SequenceValue,
    StringValue,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.execution.parallel import (
    ErrorHandlingMode,
    ParallelExecutionAlgorithm,
)
from workflow_engine.execution.topological import TopologicalExecutionAlgorithm
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
        display_name="Slow Node",
        description="A node that takes time to execute",
        version="1.0.0",
        parameter_type=SlowNodeParams,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> Type[SlowNodeOutput]:
        return SlowNodeOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[SlowNodeOutput],
        input: Empty,
    ) -> SlowNodeOutput:
        await asyncio.sleep(self.params.delay_ms.root / 1000)
        return SlowNodeOutput(value=self.params.delay_ms)


# Test node that accepts input (to create dependencies) and delays
class SlowPassthroughNodeInput(Data):
    value: IntegerValue


class SlowPassthroughNode(
    Node[SlowPassthroughNodeInput, SlowNodeOutput, SlowNodeParams]
):
    """A node that accepts input, delays, and outputs the input value."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Slow Passthrough Node",
        description="A node that accepts input and delays before outputting it",
        version="1.0.0",
        parameter_type=SlowNodeParams,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[SlowPassthroughNodeInput]:
        return SlowPassthroughNodeInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[SlowNodeOutput]:
        return SlowNodeOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[SlowPassthroughNodeInput],
        output_type: Type[SlowNodeOutput],
        input: SlowPassthroughNodeInput,
    ) -> SlowNodeOutput:
        await asyncio.sleep(self.params.delay_ms.root / 1000)
        return SlowNodeOutput(value=input.value)


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine(execution_algorithm=ParallelExecutionAlgorithm())


@pytest.fixture
def parallel_workflow(engine: WorkflowEngine) -> Workflow:
    """A workflow with three independent SlowNodes that can run in parallel."""
    return Workflow(
        input_node=engine.create_input_node(),
        output_node=(
            output_node := engine.create_output_node(
                a=IntegerValue,
                b=IntegerValue,
                c=IntegerValue,
            )
        ),
        inner_nodes=[
            node_a := engine.create_node(
                SlowNode, id="node_a", params=dict(delay_ms=100)
            ),
            node_b := engine.create_node(
                SlowNode, id="node_b", params=dict(delay_ms=100)
            ),
            node_c := engine.create_node(
                SlowNode, id="node_c", params=dict(delay_ms=100)
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=node_a,
                source_key="value",
                target=output_node,
                target_key="a",
            ),
            Edge.from_nodes(
                source=node_b,
                source_key="value",
                target=output_node,
                target_key="b",
            ),
            Edge.from_nodes(
                source=node_c,
                source_key="value",
                target=output_node,
                target_key="c",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_parallel_execution_faster_than_sequential(
    parallel_workflow: Workflow,
):
    """Test that parallel execution is actually parallel (faster than sequential)."""
    context = InMemoryExecutionContext()

    # Sequential execution
    sequential_engine = WorkflowEngine(
        execution_algorithm=TopologicalExecutionAlgorithm()
    )
    start = asyncio.get_event_loop().time()
    sequential_result = await sequential_engine.execute(
        context=context,
        workflow=parallel_workflow,
        input={},
    )
    sequential_time = asyncio.get_event_loop().time() - start
    assert sequential_result.status is WorkflowExecutionResultStatus.SUCCESS

    # Parallel execution
    parallel_engine = WorkflowEngine(execution_algorithm=ParallelExecutionAlgorithm())
    start = asyncio.get_event_loop().time()
    parallel_result = await parallel_engine.execute(
        context=context,
        workflow=parallel_workflow,
        input={},
    )
    parallel_time = asyncio.get_event_loop().time() - start
    assert parallel_result.status is WorkflowExecutionResultStatus.SUCCESS

    # Parallel should be significantly faster (3 x 100ms sequentially vs ~100ms parallel)
    # Using a generous margin to avoid flaky tests
    assert parallel_time < sequential_time * 0.6, (
        f"Parallel ({parallel_time:.3f}s) should be much faster than "
        f"sequential ({sequential_time:.3f}s)"
    )


@pytest.mark.asyncio
async def test_parallel_execution_respects_dependencies(engine: WorkflowEngine):
    """Test that dependent nodes wait for their dependencies."""
    workflow = Workflow(
        input_node=engine.create_input_node(),
        output_node=(output_node := engine.create_output_node(result=IntegerValue)),
        inner_nodes=[
            a := engine.create_node(ConstantIntegerNode, id="a", params=dict(value=1)),
            b := engine.create_node(ConstantIntegerNode, id="b", params=dict(value=2)),
            c := engine.create_node(AddNode, id="c"),
        ],
        edges=[
            Edge.from_nodes(source=a, source_key="value", target=c, target_key="a"),
            Edge.from_nodes(source=b, source_key="value", target=c, target_key="b"),
            Edge.from_nodes(
                source=c, source_key="sum", target=output_node, target_key="result"
            ),
        ],
    )

    context = InMemoryExecutionContext()
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={},
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"result": IntegerValue(3)}


@pytest.mark.asyncio
async def test_parallel_execution_complex_dependencies(engine: WorkflowEngine):
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
        input_node=engine.create_input_node(),
        output_node=(output_node := engine.create_output_node(result=IntegerValue)),
        inner_nodes=[
            a := engine.create_node(ConstantIntegerNode, id="a", params=dict(value=1)),
            b := engine.create_node(ConstantIntegerNode, id="b", params=dict(value=2)),
            c := engine.create_node(AddNode, id="c"),
            d := engine.create_node(ConstantIntegerNode, id="d", params=dict(value=10)),
            f := engine.create_node(AddNode, id="f"),
        ],
        edges=[
            Edge.from_nodes(source=a, source_key="value", target=c, target_key="a"),
            Edge.from_nodes(source=b, source_key="value", target=c, target_key="b"),
            Edge.from_nodes(source=c, source_key="sum", target=f, target_key="a"),
            Edge.from_nodes(source=d, source_key="value", target=f, target_key="b"),
            Edge.from_nodes(
                source=f, source_key="sum", target=output_node, target_key="result"
            ),
        ],
    )

    context = InMemoryExecutionContext()
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={},
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    # a + b = 3, then 3 + d(10) = 13
    assert result.output == {"result": IntegerValue(13)}


@pytest.mark.asyncio
async def test_parallel_execution_continue_on_error():
    """Test CONTINUE error handling mode collects all errors."""
    engine = WorkflowEngine(
        execution_algorithm=ParallelExecutionAlgorithm(
            error_handling=ErrorHandlingMode.CONTINUE
        )
    )
    workflow = Workflow(
        input_node=engine.create_input_node(),
        output_node=(output_node := engine.create_output_node(value=StringValue)),
        inner_nodes=[
            ok_node := engine.create_node(
                ConstantStringNode, id="ok_node", params=dict(value="test")
            ),
            error1 := engine.create_node(
                ErrorNode, id="error1", params=dict(error_name="Error1")
            ),
            error2 := engine.create_node(
                ErrorNode, id="error2", params=dict(error_name="Error2")
            ),
        ],
        edges=[
            # Both error nodes depend on ok_node, so they run in parallel after it
            Edge.from_nodes(
                source=ok_node, source_key="value", target=error1, target_key="info"
            ),
            Edge.from_nodes(
                source=ok_node, source_key="value", target=error2, target_key="info"
            ),
            Edge.from_nodes(
                source=ok_node,
                source_key="value",
                target=output_node,
                target_key="value",
            ),
        ],
    )

    context = InMemoryExecutionContext()
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={},
    )

    # Should have collected errors from both error nodes
    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert len(result.errors.node_errors) == 2
    assert "error1" in result.errors.node_errors
    assert "error2" in result.errors.node_errors
    # ok_node output should still be available in partial output
    assert result.output == {"value": StringValue("test")}


@pytest.mark.asyncio
async def test_parallel_execution_fail_fast():
    """Test FAIL_FAST error handling mode stops on first error."""
    engine = WorkflowEngine(
        execution_algorithm=ParallelExecutionAlgorithm(
            error_handling=ErrorHandlingMode.FAIL_FAST
        )
    )
    workflow = Workflow(
        input_node=engine.create_input_node(),
        output_node=(output_node := engine.create_output_node(value=StringValue)),
        inner_nodes=[
            constant := engine.create_node(
                ConstantStringNode, id="constant", params=dict(value="test")
            ),
            error := engine.create_node(
                ErrorNode, id="error", params=dict(error_name="TestError")
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=constant, source_key="value", target=error, target_key="info"
            ),
            Edge.from_nodes(
                source=constant,
                source_key="value",
                target=output_node,
                target_key="value",
            ),
        ],
    )

    context = InMemoryExecutionContext()
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={},
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR


@pytest.mark.asyncio
async def test_parallel_execution_with_node_expansion(engine: WorkflowEngine):
    """Test that node expansion works correctly with parallel execution."""
    add_workflow = Workflow(
        input_node=(
            add_input_node := engine.create_input_node(
                a=FloatValue,
                b=FloatValue,
            )
        ),
        output_node=(add_output_node := engine.create_output_node(c=FloatValue)),
        inner_nodes=[
            add := engine.create_node(AddNode, id="add", params=dict(num_arguments=2)),
        ],
        edges=[
            Edge.from_nodes(
                source=add_input_node,
                source_key="a",
                target=add,
                target_key="a",
            ),
            Edge.from_nodes(
                source=add_input_node,
                source_key="b",
                target=add,
                target_key="b",
            ),
            Edge.from_nodes(
                source=add, source_key="sum", target=add_output_node, target_key="c"
            ),
        ],
    )
    add_workflow = await engine.validate(add_workflow)

    workflow = Workflow(
        input_node=(
            outer_input_node := engine.create_input_node(
                sequence=SequenceValue[DataValue[add_workflow.input_type]],
            )
        ),
        # add_workflow has single output (c: FloatValue), so ForEach outputs SequenceValue[FloatValue]
        output_node=(
            outer_output_node := engine.create_output_node(
                results=SequenceValue[FloatValue],
            )
        ),
        inner_nodes=[
            foreach := engine.create_node(
                ForEachNode, id="foreach", params=dict(workflow=add_workflow)
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=outer_input_node,
                source_key="sequence",
                target=foreach,
                target_key="sequence",
            ),
            Edge.from_nodes(
                source=foreach,
                source_key="sequence",
                target=outer_output_node,
                target_key="results",
            ),
        ],
    )

    context = InMemoryExecutionContext()
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={
            "sequence": [
                {"a": 1.0, "b": 2.0},
                {"a": 3.0, "b": 4.0},
            ]
        },
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS

    # Compare values directly (single output: each element is FloatValue)
    results = result.output["results"]
    assert isinstance(results, SequenceValue)
    assert len(results) == 2

    result_0 = results[0]
    assert isinstance(result_0, FloatValue)
    assert result_0 == 3.0

    result_1 = results[1]
    assert isinstance(result_1, FloatValue)
    assert result_1 == 7.0


@pytest.mark.asyncio
async def test_parallel_execution_max_concurrency():
    """Test that max_concurrency limits parallel execution."""
    # With max_concurrency=2, 5 nodes taking 50ms each should take at least 150ms
    # (3 batches: 2+2+1)
    engine = WorkflowEngine(
        execution_algorithm=ParallelExecutionAlgorithm(max_concurrency=2)
    )
    slow_nodes = [
        engine.create_node(SlowNode, id=f"node_{i}", params=dict(delay_ms=50))
        for i in range(5)
    ]

    workflow = Workflow(
        input_node=engine.create_input_node(),
        output_node=(
            output_node := engine.create_output_node(
                out_0=IntegerValue,
                out_1=IntegerValue,
                out_2=IntegerValue,
                out_3=IntegerValue,
                out_4=IntegerValue,
            )
        ),
        inner_nodes=slow_nodes,
        edges=[
            Edge.from_nodes(
                source=slow_nodes[i],
                source_key="value",
                target=output_node,
                target_key=f"out_{i}",
            )
            for i in range(5)
        ],
    )

    context = InMemoryExecutionContext()
    start = asyncio.get_event_loop().time()
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={},
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    # Should take at least 3 batches worth of time (with some tolerance)
    assert elapsed >= 0.12, f"Expected at least 120ms, got {elapsed * 1000:.0f}ms"


@pytest.mark.asyncio
async def test_parallel_execution_empty_workflow(engine: WorkflowEngine):
    """Test that parallel execution handles empty workflows."""
    workflow = Workflow(
        input_node=engine.create_input_node(),
        output_node=engine.create_output_node(),
        inner_nodes=[],
        edges=[],
    )

    context = InMemoryExecutionContext()
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {}


@pytest.mark.asyncio
async def test_parallel_execution_single_node(engine: WorkflowEngine):
    """Test that parallel execution works with a single node."""
    workflow = Workflow(
        input_node=engine.create_input_node(),
        output_node=(output_node := engine.create_output_node(result=IntegerValue)),
        inner_nodes=[
            a := engine.create_node(ConstantIntegerNode, id="a", params=dict(value=42)),
        ],
        edges=[
            Edge.from_nodes(
                source=a, source_key="value", target=output_node, target_key="result"
            ),
        ],
    )

    context = InMemoryExecutionContext()
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"result": IntegerValue(42)}


@pytest.mark.asyncio
async def test_parallel_execution_matches_sequential_output(engine: WorkflowEngine):
    """Test that parallel execution produces the same output as sequential."""
    workflow = Workflow(
        input_node=(input_node := engine.create_input_node(c=IntegerValue)),
        output_node=(output_node := engine.create_output_node(sum=IntegerValue)),
        inner_nodes=[
            a := engine.create_node(ConstantIntegerNode, id="a", params=dict(value=42)),
            b := engine.create_node(
                ConstantIntegerNode, id="b", params=dict(value=2025)
            ),
            a_plus_b := engine.create_node(AddNode, id="a+b"),
            a_plus_b_plus_c := engine.create_node(AddNode, id="a+b+c"),
        ],
        edges=[
            Edge.from_nodes(
                source=a, source_key="value", target=a_plus_b, target_key="a"
            ),
            Edge.from_nodes(
                source=b, source_key="value", target=a_plus_b, target_key="b"
            ),
            Edge.from_nodes(
                source=a_plus_b,
                source_key="sum",
                target=a_plus_b_plus_c,
                target_key="a",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="c",
                target=a_plus_b_plus_c,
                target_key="b",
            ),
            Edge.from_nodes(
                source=a_plus_b_plus_c,
                source_key="sum",
                target=output_node,
                target_key="sum",
            ),
        ],
    )

    context = InMemoryExecutionContext()
    input_data = {"c": IntegerValue(-256)}

    # Sequential execution
    sequential_engine = WorkflowEngine(
        execution_algorithm=TopologicalExecutionAlgorithm()
    )
    sequential_result = await sequential_engine.execute(
        context=context,
        workflow=workflow,
        input=input_data,
    )

    # Parallel execution (the fixture engine)
    parallel_result = await engine.execute(
        context=context,
        workflow=workflow,
        input=input_data,
    )

    assert sequential_result.status is WorkflowExecutionResultStatus.SUCCESS
    assert parallel_result.status is WorkflowExecutionResultStatus.SUCCESS
    assert (
        sequential_result.output
        == parallel_result.output
        == {"sum": 42 + 2025 + (-256)}
    )


@pytest.mark.asyncio
async def test_parallel_execution_eager_dispatch(engine: WorkflowEngine):
    """Test that dependent nodes start immediately when dependencies complete.

    Graph:
        A (50ms) -----> C (50ms passthrough)
        B (200ms) (independent)

    With eager dispatch: A finishes at 50ms, C starts immediately and finishes at ~100ms.
    B finishes at 200ms. Total time ~200ms.

    With batch dispatch: A+B batch completes at 200ms, then C runs, finishes at 250ms.
    """
    workflow = Workflow(
        input_node=engine.create_input_node(),
        output_node=(
            output_node := engine.create_output_node(
                a=IntegerValue,
                b=IntegerValue,
                c=IntegerValue,
            )
        ),
        inner_nodes=[
            a := engine.create_node(SlowNode, id="a", params=dict(delay_ms=50)),
            b := engine.create_node(SlowNode, id="b", params=dict(delay_ms=200)),
            c := engine.create_node(
                SlowPassthroughNode, id="c", params=dict(delay_ms=50)
            ),
        ],
        edges=[
            Edge.from_nodes(source=a, source_key="value", target=c, target_key="value"),
            Edge.from_nodes(
                source=a, source_key="value", target=output_node, target_key="a"
            ),
            Edge.from_nodes(
                source=b, source_key="value", target=output_node, target_key="b"
            ),
            Edge.from_nodes(
                source=c, source_key="value", target=output_node, target_key="c"
            ),
        ],
    )

    context = InMemoryExecutionContext()
    start = asyncio.get_event_loop().time()
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={},
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    # Eager dispatch: ~200ms (B is bottleneck, C finishes at 100ms)
    # Batch dispatch would be: ~250ms (A+B at 200ms, then C at 250ms)
    # Using generous margin to avoid flaky tests
    assert elapsed < 0.23, (
        f"Expected ~200ms with eager dispatch, got {elapsed * 1000:.0f}ms. "
        f"This suggests batch-based execution instead of eager dispatch."
    )
