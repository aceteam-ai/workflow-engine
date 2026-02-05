from unittest.mock import AsyncMock

import pytest

from workflow_engine import (
    Edge,
    StringMapValue,
    StringValue,
    UserException,
    ValueSchemaValue,
    Workflow,
    WorkflowErrors,
)
from workflow_engine.contexts import InMemoryContext
from workflow_engine.core.io import InputNode, OutputNode, SchemaParams
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.nodes import ConstantStringNode, ErrorNode


@pytest.fixture
def workflow():
    """Helper function to create the error workflow."""
    input_node = InputNode(id="input")
    output_node = OutputNode(
        id="output",
        params=SchemaParams(
            fields=StringMapValue[ValueSchemaValue](
                {"text": ValueSchemaValue(StringValue.to_value_schema())}
            )
        ),
    )

    constant = ConstantStringNode.from_value(id="constant", value="test")
    error = ErrorNode.from_name(id="error", name="RuntimeError")

    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[constant, error],
        edges=[
            Edge.from_nodes(
                source=constant,
                source_key="value",
                target=error,
                target_key="info",
            ),
            Edge.from_nodes(
                source=constant,
                source_key="value",
                target=output_node,
                target_key="text",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_workflow_error_handling(workflow: Workflow):
    """Test that the workflow properly handles errors and calls context callbacks."""
    context = InMemoryContext()

    # Create a mock for on_node_error while preserving the original function
    original_on_node_error = context.on_node_error
    mock_on_node_error = AsyncMock(side_effect=original_on_node_error)
    context.on_node_error = mock_on_node_error

    algorithm = TopologicalExecutionAlgorithm()

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={},
    )

    error_node = workflow.nodes_by_id["error"]
    # Verify the error was captured correctly
    assert errors == WorkflowErrors(
        workflow_errors=[],
        node_errors={error_node.id: ["RuntimeError: test"]},
    )

    # Verify the output still contains the constant value
    assert output == {"text": StringValue("test")}

    # Verify on_node_error was called with the correct arguments
    mock_on_node_error.assert_called_once()
    call_args = mock_on_node_error.call_args
    assert call_args.kwargs["node"] is error_node
    exception = call_args.kwargs["exception"]
    assert isinstance(exception, UserException)
    assert exception.message == "RuntimeError: test"
