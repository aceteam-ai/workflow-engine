from unittest.mock import AsyncMock

import pytest

from workflow_engine import (
    Edge,
    StringValue,
    Workflow,
    WorkflowEngine,
    WorkflowException,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core import WorkflowError
from workflow_engine.core.stakeholder import StakeholderLevel
from workflow_engine.nodes import ConstantStringNode, ErrorNode


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


@pytest.fixture
def workflow(engine: WorkflowEngine) -> Workflow:
    return Workflow(
        input_node=engine.create_input_node(),
        output_node=(
            output_node := engine.create_output_node(
                text=StringValue,
            )
        ),
        inner_nodes=[
            constant := engine.create_node(
                ConstantStringNode,
                id="constant",
                params=dict(value="test"),
            ),
            error := engine.create_node(
                ErrorNode,
                id="error",
                params=dict(error_name="RuntimeError"),
            ),
        ],
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
async def test_workflow_error_handling(engine: WorkflowEngine, workflow: Workflow):
    """Test that the workflow properly handles errors and calls context callbacks."""
    context = InMemoryExecutionContext()

    # Create a mock for on_node_error while preserving the original function
    original_on_node_error = context.on_node_error
    mock_on_node_error = AsyncMock(side_effect=original_on_node_error)
    context.on_node_error = mock_on_node_error

    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={},
    )

    error_node = workflow.nodes_by_id["error"]
    # Verify the error was captured correctly
    assert len(result.errors.workflow_errors) == 0
    assert set(result.errors.node_errors.keys()) == {error_node.id}
    assert len(result.errors.node_errors[error_node.id]) == 1
    error = result.errors.node_errors[error_node.id][0]
    assert isinstance(error, WorkflowError)
    assert error.message == "RuntimeError: test"
    assert error.level == StakeholderLevel.USER
    assert error.node_id == error_node.id
    assert error.cause is None

    # Verify the output still contains the constant value
    assert result.output == {"text": StringValue("test")}

    # Verify on_node_error was called with the correct arguments
    mock_on_node_error.assert_called_once()
    call_args = mock_on_node_error.call_args
    assert call_args.kwargs["node"] is error_node
    exception = call_args.kwargs["exception"]
    assert isinstance(exception, WorkflowException)
    assert exception.message == "RuntimeError: test"
