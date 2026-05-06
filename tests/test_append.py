import pytest

from workflow_engine import (
    Edge,
    StringValue,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.files import TextFileValue
from workflow_engine.nodes import AppendToFileNode


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


@pytest.fixture
def workflow(engine: WorkflowEngine):
    return Workflow(
        input_node=(
            input_node := engine.create_input_node(
                text=StringValue,
                file=TextFileValue,
            )
        ),
        output_node=(
            output_node := engine.create_output_node(
                file=TextFileValue,
            )
        ),
        inner_nodes=[
            append := engine.create_node(
                AppendToFileNode,
                id="append",
                params=dict(suffix="_append"),
            ),
        ],
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
async def test_workflow_serialization(engine: WorkflowEngine, workflow: Workflow):
    """Test that the append workflow can be serialized and deserialized correctly."""
    # Test round-trip serialization/deserialization
    workflow_json = workflow.model_dump_json()
    # Deserialize workflow, then load via engine to get typed nodes
    deserialized_workflow = await engine.validate(
        Workflow.model_validate_json(workflow_json)
    )
    # compare only serialized fields
    assert deserialized_workflow.model_dump(mode="json") == workflow.model_dump(
        mode="json"
    )


@pytest.mark.asyncio
async def test_workflow_execution(engine: WorkflowEngine, workflow: Workflow):
    """Test that the workflow executes correctly and produces the expected result."""
    context = InMemoryExecutionContext()

    # Create input with a text file
    hello_world = "Hello, world!"
    input_file = TextFileValue.from_path("test.txt")
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
