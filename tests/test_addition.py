import pytest

from workflow_engine import Edge, IntegerValue, StringMapValue, ValueSchemaValue, Workflow
from workflow_engine.contexts import InMemoryContext
from workflow_engine.core.io import InputNode, OutputNode, SchemaParams
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.nodes import AddNode, ConstantIntegerNode


@pytest.fixture
def workflow():
    """Helper function to create the addition workflow."""
    input_node = InputNode(
        id="input",
        params=SchemaParams(
            fields=StringMapValue[ValueSchemaValue](
                {"c": ValueSchemaValue(IntegerValue.to_value_schema())}
            )
        ),
    )
    output_node = OutputNode(
        id="output",
        params=SchemaParams(
            fields=StringMapValue[ValueSchemaValue](
                {"sum": ValueSchemaValue(IntegerValue.to_value_schema())}
            )
        ),
    )

    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[
            a := ConstantIntegerNode.from_value(id="a", value=42),
            b := ConstantIntegerNode.from_value(id="b", value=2025),
            a_plus_b := AddNode(id="a+b"),
            a_plus_b_plus_c := AddNode(id="a+b+c"),
        ],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="c",
                target=a_plus_b_plus_c,
                target_key="b",
            ),
            Edge.from_nodes(
                source=a,
                source_key="value",
                target=a_plus_b,
                target_key="a",
            ),
            Edge.from_nodes(
                source=b,
                source_key="value",
                target=a_plus_b,
                target_key="b",
            ),
            Edge.from_nodes(
                source=a_plus_b,
                source_key="sum",
                target=a_plus_b_plus_c,
                target_key="a",
            ),
            Edge.from_nodes(
                source=a_plus_b_plus_c,
                source_key="sum",
                target=output_node,
                target_key="sum",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_workflow_execution(workflow: Workflow):
    """Test that the workflow executes correctly and produces the expected result."""
    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()

    c = -256

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={"c": IntegerValue(c)},
    )
    assert not errors.any()
    assert output == {"sum": 42 + 2025 + c}
