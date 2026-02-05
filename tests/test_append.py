import pytest

from workflow_engine import Edge, File, StringValue, Workflow
from workflow_engine.contexts import InMemoryContext
from workflow_engine.core.io import InputNode, OutputNode
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.files import TextFileValue
from workflow_engine.nodes import AppendToFileNode


@pytest.fixture
def workflow():
    """Helper function to create the append workflow."""
    input_node = InputNode.from_fields(
        id="input",
        fields={"text": StringValue, "file": TextFileValue},
    )
    output_node = OutputNode.from_fields(
        id="output",
        fields={"file": TextFileValue},
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
@pytest.mark.skip(reason="Workflow serialization from from_fields() needs schema serialization fix")
def test_workflow_serialization(workflow: Workflow):
    """Test that the append workflow can be serialized and deserialized correctly."""
    # Test round-trip serialization/deserialization
    workflow_json = workflow.model_dump_json()
    deserialized_workflow = Workflow.model_validate_json(workflow_json)
    assert deserialized_workflow == workflow


@pytest.mark.asyncio
async def test_workflow_execution(workflow: Workflow):
    """Test that the workflow executes correctly and produces the expected result."""
    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()

    # Create input with a text file
    hello_world = "Hello, world!"
    input_file = TextFileValue(File(path="test.txt"))
    input_file = await input_file.write_text(context, text=hello_world)

    appended_text = StringValue("This text will be appended to the file.")
    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={
            "file": input_file,
            "text": appended_text,
        },
    )

    # Verify no errors occurred
    assert not errors.any()

    # Verify the output file exists and has the correct content
    output_file = output["file"]
    assert isinstance(output_file, TextFileValue)
    assert output_file.path == "test_append.txt"
    output_text = await output_file.read_text(context)
    assert output_text == hello_world + appended_text.root
