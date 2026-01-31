"""Tests for workflow output type casting.

This module tests that workflow outputs are automatically cast to match the
expected output types, just like node inputs are cast to expected types.
"""

import pytest

from workflow_engine import (
    FloatValue,
    IntegerValue,
    JSONValue,
    OutputEdge,
    SequenceValue,
    Workflow,
)
from workflow_engine.contexts import InMemoryContext
from workflow_engine.execution.parallel import ParallelExecutionAlgorithm
from workflow_engine.execution.topological import TopologicalExecutionAlgorithm
from workflow_engine.files import JSONLinesFileValue
from workflow_engine.nodes import ConstantIntegerNode, ConstantStringNode


@pytest.mark.unit
@pytest.mark.asyncio
async def test_basic_output_casting():
    """Test that IntegerValue is cast to FloatValue in workflow output."""
    node = ConstantIntegerNode.from_value(id="producer", value=42)

    # Workflow expects FloatValue output, but node produces IntegerValue
    workflow = Workflow(
        nodes=[node],
        edges=[],
        input_edges=[],
        output_edges=[
            OutputEdge(
                source_id="producer",
                source_key="value",
                output_key="result",
                output_schema=FloatValue.to_value_schema(),
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
    int_node = ConstantIntegerNode.from_value(id="int_producer", value=100)
    str_node = ConstantStringNode.from_value(id="str_producer", value="123")

    workflow = Workflow(
        nodes=[int_node, str_node],
        edges=[],
        input_edges=[],
        output_edges=[
            OutputEdge(
                source_id="int_producer",
                source_key="value",
                output_key="int_result",
                output_schema=FloatValue.to_value_schema(),
            ),
            OutputEdge(
                source_id="str_producer",
                source_key="value",
                output_key="str_result",
                output_schema=IntegerValue.to_value_schema(),
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
async def test_partial_mode_skips_missing_outputs():
    """Test that partial mode skips missing outputs gracefully."""
    node = ConstantIntegerNode.from_value(id="producer", value=42)

    workflow = Workflow(
        nodes=[node],
        edges=[],
        input_edges=[],
        output_edges=[
            OutputEdge(
                source_id="producer",
                source_key="value",
                output_key="result",
                output_schema=FloatValue.to_value_schema(),
            )
        ],
    )

    context = InMemoryContext()

    # Call get_output with partial=True and empty node_outputs
    output = await workflow.get_output(
        context=context,
        node_outputs={},
        partial=True,
    )

    # Should return empty dict, not raise exception
    assert output == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parallel_execution_algorithm():
    """Test that output casting works with ParallelExecutionAlgorithm."""
    node = ConstantIntegerNode.from_value(id="producer", value=42)

    workflow = Workflow(
        nodes=[node],
        edges=[],
        input_edges=[],
        output_edges=[
            OutputEdge(
                source_id="producer",
                source_key="value",
                output_key="result",
                output_schema=FloatValue.to_value_schema(),
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
    node = ConstantIntegerNode.from_value(id="producer", value=42)

    workflow = Workflow(
        nodes=[node],
        edges=[],
        input_edges=[],
        output_edges=[
            OutputEdge(source_id="producer", source_key="value", output_key="result")
        ],
    )

    # Output type matches (IntegerValue -> IntegerValue)
    assert workflow.output_fields["result"] == (IntegerValue, True)

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
async def test_input_edge_with_schema():
    """Test that InputEdge can specify a schema different from target node."""
    from workflow_engine import InputEdge
    from workflow_engine.nodes import AddNode

    add_node = AddNode(id="add")

    workflow = Workflow(
        nodes=[add_node],
        edges=[],
        input_edges=[
            InputEdge(
                input_key="a",
                target_id="add",
                target_key="a",
                input_schema=FloatValue.to_value_schema(),
            ),
            InputEdge(
                input_key="b",
                target_id="add",
                target_key="b",
                input_schema=FloatValue.to_value_schema(),
            ),
        ],
        output_edges=[
            OutputEdge(source_id="add", source_key="sum", output_key="result")
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
def test_backward_compatibility_no_schema():
    """Test that edges without schemas work exactly as before."""
    node = ConstantIntegerNode.from_value(id="producer", value=42)

    workflow = Workflow(
        nodes=[node],
        edges=[],
        input_edges=[],
        output_edges=[
            OutputEdge(
                source_id="producer",
                source_key="value",
                output_key="result",
                # No output_schema specified
            )
        ],
    )

    # Should infer IntegerValue from node
    assert workflow.output_fields["result"][0] == IntegerValue
