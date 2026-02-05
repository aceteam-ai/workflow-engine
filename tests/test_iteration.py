import pytest

from workflow_engine import (
    Edge,
    FloatValue,
    SequenceValue,
    StringMapValue,
    ValueSchemaValue,
    Workflow,
)
from workflow_engine.contexts import InMemoryContext
from workflow_engine.core.io import InputNode, OutputNode, SchemaParams
from workflow_engine.core.values.schema import SequenceValueSchema
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.nodes import AddNode, ForEachNode


@pytest.fixture
def add_workflow() -> Workflow:
    """Create a workflow that adds two numbers: a + b = c."""
    input_node = InputNode(
        id="input",
        params=SchemaParams(
            fields=StringMapValue[ValueSchemaValue](
                {
                    "a": ValueSchemaValue(FloatValue.to_value_schema()),
                    "b": ValueSchemaValue(FloatValue.to_value_schema()),
                }
            )
        ),
    )
    output_node = OutputNode(
        id="output",
        params=SchemaParams(
            fields=StringMapValue[ValueSchemaValue](
                {
                    "c": ValueSchemaValue(FloatValue.to_value_schema()),
                }
            )
        ),
    )

    add = AddNode(id="add")

    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[add],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="a",
                target=add,
                target_key="a",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="b",
                target=add,
                target_key="b",
            ),
            Edge.from_nodes(
                source=add,
                source_key="sum",
                target=output_node,
                target_key="c",
            ),
        ],
    )


@pytest.fixture
def workflow(add_workflow: Workflow) -> Workflow:
    input_node = InputNode(
        id="input",
        params=SchemaParams(
            fields=StringMapValue[ValueSchemaValue](
                {
                    "sequence": ValueSchemaValue(
                        SequenceValueSchema(
                            type="array",
                            items=add_workflow.input_schema,
                        )
                    ),
                }
            )
        ),
    )
    output_node = OutputNode(
        id="output",
        params=SchemaParams(
            fields=StringMapValue[ValueSchemaValue](
                {
                    "results": ValueSchemaValue(
                        SequenceValueSchema(
                            type="array",
                            items=add_workflow.output_schema,
                        )
                    ),
                }
            )
        ),
    )

    for_each = ForEachNode.from_workflow(
        id="for_each",
        workflow=add_workflow,
    )

    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[for_each],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="sequence",
                target=for_each,
                target_key="sequence",
            ),
            Edge.from_nodes(
                source=for_each,
                source_key="sequence",
                target=output_node,
                target_key="results",
            ),
        ],
    )

    return workflow


@pytest.mark.asyncio
async def test_for_each_simple_sequence(workflow: Workflow):
    """Test that ForEachNode processes a simple sequence of addition operations."""
    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()

    input = workflow.input_type.model_validate(
        {
            "sequence": [
                {"a": 1.0, "b": 2.0},
                {"a": 3.0, "b": 4.0},
                {"a": 5.0, "b": 6.0},
            ]
        }
    ).to_dict()

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input=input,
    )

    assert not errors.any(), errors

    # Compare actual values (dynamically generated DataValue types have identity issues)
    results = output["results"]
    assert isinstance(results, SequenceValue)
    assert len(results) == 3
    assert results[0].root.c == FloatValue(3.0)
    assert results[1].root.c == FloatValue(7.0)
    assert results[2].root.c == FloatValue(11.0)


@pytest.mark.asyncio
async def test_for_each_empty(workflow: Workflow):
    """Test that ForEachNode processes an empty sequence."""
    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()

    input = workflow.input_type.model_validate(
        {
            "sequence": [],
        }
    ).to_dict()

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input=input,
    )

    assert not errors.any(), errors

    results = output["results"]
    assert isinstance(results, SequenceValue)
    assert len(results) == 0
