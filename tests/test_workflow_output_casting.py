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
    Workflow,
)
from workflow_engine.contexts import InMemoryContext
from workflow_engine.core.io import InputNode, OutputNode
from workflow_engine.execution.parallel import ParallelExecutionAlgorithm
from workflow_engine.execution.topological import TopologicalExecutionAlgorithm
from workflow_engine.files import JSONLinesFileValue
from workflow_engine.nodes import AddNode, ConstantIntegerNode, ConstantStringNode


@pytest.mark.unit
@pytest.mark.asyncio
async def test_basic_output_casting():
    """Test that IntegerValue is cast to FloatValue in workflow output."""
    input_node = InputNode.empty()
    output_node = OutputNode.from_fields(result=FloatValue)

    node = ConstantIntegerNode.from_value(id="producer", value=42)

    # Workflow expects FloatValue output, but node produces IntegerValue
    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[node],
        edges=[
            Edge.from_nodes(
                source=node,
                source_key="value",
                target=output_node,
                target_key="result",
            )
        ],
    )

    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()

    errors, output = await algorithm.execute(
        context=context, workflow=workflow, input={}
    )

    assert not errors.any()
    assert "result" in output
    assert isinstance(output["result"], FloatValue)
    assert output["result"] == 42.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multiple_outputs_casting():
    """Test that multiple outputs are cast correctly in parallel."""
    input_node = InputNode.empty()
    output_node = OutputNode.from_fields(
        int_result=FloatValue,
        str_result=IntegerValue,
    )

    int_node = ConstantIntegerNode.from_value(id="int_producer", value=100)
    str_node = ConstantStringNode.from_value(id="str_producer", value="123")

    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[int_node, str_node],
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

    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()

    errors, output = await algorithm.execute(
        context=context, workflow=workflow, input={}
    )

    assert not errors.any()
    assert isinstance(output["int_result"], FloatValue)
    assert output["int_result"] == 100.0
    assert isinstance(output["str_result"], IntegerValue)
    assert output["str_result"] == 123


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parallel_execution_algorithm():
    """Test that output casting works with ParallelExecutionAlgorithm."""
    input_node = InputNode.empty()
    output_node = OutputNode.from_fields(
        result=FloatValue,
    )

    node = ConstantIntegerNode.from_value(id="producer", value=42)

    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[node],
        edges=[
            Edge.from_nodes(
                source=node,
                source_key="value",
                target=output_node,
                target_key="result",
            )
        ],
    )

    context = InMemoryContext()
    algorithm = ParallelExecutionAlgorithm()

    errors, output = await algorithm.execute(
        context=context, workflow=workflow, input={}
    )

    assert not errors.any()
    assert isinstance(output["result"], FloatValue)
    assert output["result"] == 42.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_complex_type_sequence_to_jsonlines():
    """Test casting complex types like SequenceValue[JSONValue] to JSONLinesFileValue."""
    # Create a sequence of JSON values
    json_values = [JSONValue({"name": "Alice"}), JSONValue({"name": "Bob"})]
    sequence = SequenceValue(json_values)

    # Test that the sequence can be cast to JSONLinesFileValue
    context = InMemoryContext()

    # Verify the cast is possible
    assert sequence.can_cast_to(JSONLinesFileValue)

    # Perform the cast
    result = await sequence.cast_to(JSONLinesFileValue, context=context)

    assert isinstance(result, JSONLinesFileValue)

    # Verify the result has a path (may be relative or absolute depending on context)
    assert result.path is not None
    assert len(result.path) > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_casting_when_types_match():
    """Test that no casting occurs when output types already match."""
    input_node = InputNode.empty()
    output_node = OutputNode.from_fields(result=IntegerValue)

    node = ConstantIntegerNode.from_value(id="producer", value=42)

    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[node],
        edges=[
            Edge.from_nodes(
                source=node,
                source_key="value",
                target=output_node,
                target_key="result",
            )
        ],
    )

    # Output type matches (IntegerValue -> IntegerValue)
    assert workflow.output_fields["result"][0] == IntegerValue

    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()

    errors, output = await algorithm.execute(
        context=context, workflow=workflow, input={}
    )

    assert not errors.any()
    assert isinstance(output["result"], IntegerValue)
    assert output["result"] == 42


@pytest.mark.unit
@pytest.mark.asyncio
async def test_input_casting():
    """Test that workflow inputs can be cast to expected types."""
    input_node = InputNode.from_fields(
        a=FloatValue,
        b=FloatValue,
    )
    output_node = OutputNode.from_fields(
        result=FloatValue,
    )

    add_node = AddNode(id="add")

    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[add_node],
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

    # Workflow should expect FloatValue inputs
    assert workflow.input_fields["a"][0] == FloatValue
    assert workflow.input_fields["b"][0] == FloatValue

    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={"a": FloatValue(10.5), "b": FloatValue(20.3)},
    )

    assert not errors.any()


@pytest.mark.unit
def test_workflow_output_type_inference():
    """Test that workflow infers output types from OutputNode."""
    input_node = InputNode.empty()
    output_node = OutputNode.from_fields(
        result=IntegerValue,
    )

    node = ConstantIntegerNode.from_value(id="producer", value=42)

    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[node],
        edges=[
            Edge.from_nodes(
                source=node,
                source_key="value",
                target=output_node,
                target_key="result",
            )
        ],
    )

    # Should infer IntegerValue from OutputNode
    assert workflow.output_fields["result"][0] == IntegerValue
