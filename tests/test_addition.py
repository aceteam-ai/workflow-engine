import pytest

from workflow_engine import Edge, IntegerValue, Workflow, WorkflowExecutionResultStatus
from workflow_engine.contexts import InMemoryContext
from workflow_engine.core.io import InputNode, OutputNode
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.nodes import AddNode, ConstantIntegerNode


@pytest.fixture
def workflow():
    """Helper function to create the addition workflow."""
    input_node = InputNode.from_fields(
        c=IntegerValue,
    )
    output_node = OutputNode.from_fields(
        sum=IntegerValue,
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
async def test_add_3_arguments():
    """Test AddNode with 3 arguments (a, b, c)."""
    constants = [
        ConstantIntegerNode.from_value(id=f"const_{i}", value=(i + 1) * 10)
        for i in range(3)
    ]
    add = AddNode.with_arity(id="add", arity=3)
    output_node = OutputNode.from_fields(sum=IntegerValue)

    workflow = Workflow(
        input_node=InputNode.from_fields(),
        output_node=output_node,
        inner_nodes=[*constants, add],
        edges=[
            Edge.from_nodes(
                source=constants[i], source_key="value", target=add, target_key=key
            )
            for i, key in enumerate(["a", "b", "c"])
        ]
        + [
            Edge.from_nodes(
                source=add, source_key="sum", target=output_node, target_key="sum"
            ),
        ],
    )

    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()
    result = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"sum": 10 + 20 + 30}


@pytest.mark.asyncio
async def test_add_30_arguments():
    """Test AddNode with 30 arguments (a–z, aa–ad), verifying field name generation."""
    n = 30
    # Field names: a, b, ..., z (26), aa, ab, ac, ad (4)
    expected_names = [chr(ord("a") + i) for i in range(26)] + ["aa", "ab", "ac", "ad"]
    assert len(expected_names) == n

    constants = [
        ConstantIntegerNode.from_value(id=f"const_{i}", value=i + 1) for i in range(n)
    ]
    add = AddNode.with_arity(id="add", arity=n)
    output_node = OutputNode.from_fields(sum=IntegerValue)

    workflow = Workflow(
        input_node=InputNode.from_fields(),
        output_node=output_node,
        inner_nodes=[*constants, add],
        edges=[
            Edge.from_nodes(
                source=constants[i], source_key="value", target=add, target_key=name
            )
            for i, name in enumerate(expected_names)
        ]
        + [
            Edge.from_nodes(
                source=add, source_key="sum", target=output_node, target_key="sum"
            ),
        ],
    )

    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()
    result = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"sum": sum(range(1, n + 1))}


def test_add_1000_arguments_field_names():
    """Spot-check field names for a 1000-argument AddNode without executing it."""
    add = AddNode.with_arity(id="add", arity=1000)
    fields = add.input_type.model_fields

    assert len(fields) == 1000

    # 1-letter boundaries
    assert "a" in fields  # index 0
    assert "z" in fields  # index 25

    # 2-letter boundaries
    assert "aa" in fields  # index 26
    assert "zz" in fields  # index 701

    # 3-letter start and a spot-check in the middle
    assert "aaa" in fields  # index 702
    assert "all" in fields  # index 999


@pytest.mark.asyncio
async def test_workflow_execution(workflow: Workflow):
    """Test that the workflow executes correctly and produces the expected result."""
    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()

    c = -256

    result = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={"c": IntegerValue(c)},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"sum": 42 + 2025 + c}
