"""Tests for workflow output type casting.

This module tests that workflow outputs are automatically cast to match the
expected output types, just like node inputs are cast to expected types.
"""

import pytest

from workflow_engine import (
    Edge,
    FloatValue,
    IntegerValue,
    JSONValue,
    SequenceValue,
    ValidationContext,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core.values import get_data_fields, resolve_path
from workflow_engine.execution.parallel import ParallelExecutionAlgorithm
from workflow_engine.execution.topological import TopologicalExecutionAlgorithm
from workflow_engine.files import JSONLinesFileValue
from workflow_engine.nodes import AddNode, ConstantIntegerNode, ConstantStringNode


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


@pytest.mark.unit
async def test_basic_output_casting(engine: WorkflowEngine):
    """Test that IntegerValue is cast to FloatValue in workflow output."""
    workflow = Workflow(
        input_node=engine.create_input_node(),
        output_node=(output_node := engine.create_output_node(result=FloatValue)),
        inner_nodes=[
            node := engine.create_node(
                ConstantIntegerNode, id="producer", params=dict(value=42)
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=node,
                source_key="value",
                target=output_node,
                target_key="result",
            )
        ],
    )

    context = InMemoryExecutionContext()
    result = await engine.execute(context=context, workflow=workflow, input={})

    assert result.status is WorkflowExecutionResultStatus.SUCCESS

    result_value = result.output["result"]
    assert isinstance(result_value, FloatValue)
    assert result_value == 42.0


@pytest.mark.unit
async def test_multiple_outputs_casting():
    """Test that multiple outputs are cast correctly in parallel."""
    engine = WorkflowEngine(execution_algorithm=TopologicalExecutionAlgorithm())

    workflow = Workflow(
        input_node=engine.create_input_node(),
        output_node=(
            output_node := engine.create_output_node(
                int_result=FloatValue,
                str_result=IntegerValue,
            )
        ),
        inner_nodes=[
            int_node := engine.create_node(
                ConstantIntegerNode, id="int_producer", params=dict(value=100)
            ),
            str_node := engine.create_node(
                ConstantStringNode, id="str_producer", params=dict(value="123")
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=int_node,
                source_key="value",
                target=output_node,
                target_key="int_result",
            ),
            Edge.from_nodes(
                source=str_node,
                source_key="value",
                target=output_node,
                target_key="str_result",
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

    int_result = result.output["int_result"]
    assert isinstance(int_result, FloatValue)
    assert int_result == 100.0

    str_result = result.output["str_result"]
    assert isinstance(str_result, IntegerValue)
    assert str_result == 123


@pytest.mark.unit
async def test_parallel_execution_algorithm():
    """Test that output casting works with ParallelExecutionAlgorithm."""
    engine = WorkflowEngine(execution_algorithm=ParallelExecutionAlgorithm())

    workflow = Workflow(
        input_node=engine.create_input_node(),
        output_node=(output_node := engine.create_output_node(result=FloatValue)),
        inner_nodes=[
            node := engine.create_node(
                ConstantIntegerNode, id="producer", params=dict(value=42)
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=node,
                source_key="value",
                target=output_node,
                target_key="result",
            )
        ],
    )

    context = InMemoryExecutionContext()
    result = await engine.execute(context=context, workflow=workflow, input={})

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert isinstance(result.output["result"], FloatValue)
    assert result.output["result"] == 42.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_complex_type_sequence_to_jsonlines():
    """Test casting complex types like SequenceValue[JSONValue] to JSONLinesFileValue."""
    json_values = [JSONValue({"name": "Alice"}), JSONValue({"name": "Bob"})]
    sequence = SequenceValue(json_values)

    context = InMemoryExecutionContext()

    assert sequence.can_cast_to(JSONLinesFileValue)

    result = await sequence.cast_to(JSONLinesFileValue, context=context)

    assert isinstance(result, JSONLinesFileValue)

    assert result.path is not None
    assert len(result.path) > 0


@pytest.mark.unit
async def test_no_casting_when_types_match():
    """Test that no casting occurs when output types already match."""
    engine = WorkflowEngine(execution_algorithm=TopologicalExecutionAlgorithm())

    workflow = Workflow(
        input_node=engine.create_input_node(),
        output_node=(output_node := engine.create_output_node(result=IntegerValue)),
        inner_nodes=[
            node := engine.create_node(
                ConstantIntegerNode, id="producer", params=dict(value=42)
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=node,
                source_key="value",
                target=output_node,
                target_key="result",
            )
        ],
    )

    context = InMemoryExecutionContext()
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={},
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert isinstance(result.output["result"], IntegerValue)
    assert result.output["result"] == 42


@pytest.mark.unit
async def test_input_casting():
    """Test that workflow inputs can be cast to expected types."""
    engine = WorkflowEngine(execution_algorithm=TopologicalExecutionAlgorithm())

    workflow = Workflow(
        input_node=(
            input_node := engine.create_input_node(
                a=FloatValue,
                b=FloatValue,
            )
        ),
        output_node=(output_node := engine.create_output_node(result=FloatValue)),
        inner_nodes=[
            add_node := engine.create_node(AddNode, id="add"),
        ],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="a",
                target=add_node,
                target_key="a",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="b",
                target=add_node,
                target_key="b",
            ),
            Edge.from_nodes(
                source=add_node,
                source_key="sum",
                target=output_node,
                target_key="result",
            ),
        ],
    )

    validated_workflow = await engine.validate(workflow)

    input_fields = get_data_fields(validated_workflow.input_type)
    assert issubclass(input_fields["a"][0], FloatValue)
    assert issubclass(input_fields["b"][0], FloatValue)

    context = InMemoryExecutionContext()
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={"a": 10.5, "b": 20.3},
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS


@pytest.mark.unit
async def test_workflow_output_type_inference(engine: WorkflowEngine):
    """Test that workflow infers output types from OutputNode."""
    workflow = Workflow(
        input_node=engine.create_input_node(),
        output_node=(output_node := engine.create_output_node(result=IntegerValue)),
        inner_nodes=[
            node := engine.create_node(
                ConstantIntegerNode,
                id="producer",
                params=dict(value=42),
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=node,
                source_key="value",
                target=output_node,
                target_key="result",
            )
        ],
    )
    context = ValidationContext()
    workflow = await workflow.validate(context=context)
    assert issubclass(
        resolve_path(data_type=workflow.output_type, path=["result"]),
        IntegerValue,
    )
