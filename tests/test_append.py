import pytest

from workflow_engine import (
    Edge,
    File,
    StringValue,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core.io import InputNode, OutputNode
from workflow_engine.files import TextFileValue
from workflow_engine.nodes import AppendToFileNode


@pytest.fixture
def workflow():
    """Helper function to create the append workflow."""
    input_node = InputNode.from_fields(
        text=StringValue,
        file=TextFileValue,
    )
    output_node = OutputNode.from_fields(
        file=TextFileValue,
    )
    append = AppendToFileNode.from_suffix(
        id="append",
        suffix="_append",
    )

    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[append],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="text",
                target=append,
                target_key="text",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="file",
                target=append,
                target_key="file",
            ),
            Edge.from_nodes(
                source=append,
                source_key="file",
                target=output_node,
                target_key="file",
            ),
        ],
    )


@pytest.mark.unit
async def test_workflow_serialization(workflow: Workflow):
    """Test that the append workflow can be serialized and deserialized correctly."""
    # Test round-trip serialization/deserialization
    workflow_json = workflow.model_dump_json()
    # Deserialize workflow, then load via engine to get typed nodes
    engine = WorkflowEngine()
    deserialized_workflow = await engine.validate(
        Workflow.model_validate_json(workflow_json)
    )
    # compare only serialized fields
    assert deserialized_workflow.model_dump() == workflow.model_dump()


@pytest.mark.asyncio
async def test_workflow_execution(workflow: Workflow):
    """Test that the workflow executes correctly and produces the expected result."""
    context = InMemoryExecutionContext()
    engine = WorkflowEngine()

    # Create input with a text file
    hello_world = "Hello, world!"
    input_file = TextFileValue(File(path="test.txt"))
    input_file = await input_file.write_text(context, text=hello_world)

    appended_text = "This text will be appended to the file."
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={
            "file": input_file,
            "text": appended_text,
        },
    )

    # Verify no errors occurred
    assert result.status is WorkflowExecutionResultStatus.SUCCESS

    # Verify the output file exists and has the correct content
    output_file = result.output["file"]
    assert isinstance(output_file, TextFileValue)
    assert output_file.path == "test_append.txt"
    output_text = await output_file.read_text(context)
    assert output_text == hello_world + appended_text
