import pytest

from workflow_engine import (
    Edge,
    FloatValue,
    IntegerValue,
    ValidationContext,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core.values import get_data_fields
from workflow_engine.nodes import AddNode, ConstantIntegerNode


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


@pytest.fixture
def workflow(engine: WorkflowEngine) -> Workflow:
    return Workflow(
        input_node=(
            input_node := engine.create_input_node(
                c=IntegerValue,
            )
        ),
        output_node=(
            output_node := engine.create_output_node(
                sum=IntegerValue,
            )
        ),
        inner_nodes=[
            a := engine.create_node(
                ConstantIntegerNode,
                id="a",
                params=dict(value=42),
            ),
            b := engine.create_node(
                ConstantIntegerNode,
                id="b",
                params=dict(value=2025),
            ),
            a_plus_b := engine.create_node(
                AddNode,
                id="a+b",
            ),
            a_plus_b_plus_c := engine.create_node(
                AddNode,
                id="a+b+c",
            ),
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
async def test_add_3_arguments(engine: WorkflowEngine):
    """Test AddNode with 3 arguments (a, b, c)."""
    workflow = Workflow(
        input_node=engine.create_input_node(),
        output_node=(output_node := engine.create_output_node(sum=IntegerValue)),
        inner_nodes=[
            *(
                constants := [
                    engine.create_node(
                        ConstantIntegerNode,
                        id=f"const_{i}",
                        params=dict(value=(i + 1) * 10),
                    )
                    for i in range(3)
                ]
            ),
            add := engine.create_node(
                AddNode,
                id="add",
                params=dict(num_arguments=3),
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=constants[i],
                source_key="value",
                target=add,
                target_key=key,
            )
            for i, key in enumerate(["a", "b", "c"])
        ]
        + [
            Edge.from_nodes(
                source=add,
                source_key="sum",
                target=output_node,
                target_key="sum",
            ),
        ],
    )

    context = InMemoryExecutionContext()
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"sum": 10 + 20 + 30}


@pytest.mark.asyncio
async def test_add_30_arguments(engine: WorkflowEngine):
    """Test AddNode with 30 arguments (a-z, aa-ad), verifying field name generation."""
    n = 30
    # Field names: a, b, ..., z (26), aa, ab, ac, ad (4)
    expected_names = [chr(ord("a") + i) for i in range(26)] + ["aa", "ab", "ac", "ad"]
    assert len(expected_names) == n

    workflow = Workflow(
        input_node=engine.create_input_node(),
        output_node=(output_node := engine.create_output_node(sum=IntegerValue)),
        inner_nodes=[
            *(
                constants := [
                    engine.create_node(
                        ConstantIntegerNode,
                        id=f"const_{i}",
                        params=dict(value=i + 1),
                    )
                    for i in range(n)
                ]
            ),
            add := engine.create_node(
                AddNode,
                id="add",
                params=dict(num_arguments=n),
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=constants[i],
                source_key="value",
                target=add,
                target_key=name,
            )
            for i, name in enumerate(expected_names)
        ]
        + [
            Edge.from_nodes(
                source=add,
                source_key="sum",
                target=output_node,
                target_key="sum",
            ),
        ],
    )

    context = InMemoryExecutionContext()
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"sum": sum(range(1, n + 1))}


async def test_add_1000_arguments_field_names(engine: WorkflowEngine):
    """Spot-check field names for a 1000-argument AddNode without executing it."""
    validation_context = ValidationContext()

    add = engine.create_node(
        AddNode,
        id="add",
        params=dict(num_arguments=1000),
    )
    fields = [
        (name, value_type, field_info.title)
        for name, (value_type, field_info) in get_data_fields(
            await add.input_type(validation_context)
        ).items()
    ]
    assert len(fields) == 1000

    # 1-letter boundaries
    assert fields[0] == ("a", FloatValue, "A")
    assert fields[25] == ("z", FloatValue, "Z")

    # 2-letter boundaries
    assert fields[26] == ("aa", FloatValue, "AA")
    assert fields[701] == ("zz", FloatValue, "ZZ")

    # 3-letter start and a spot-check in the middle
    assert fields[702] == ("aaa", FloatValue, "AAA")
    assert fields[999] == ("all", FloatValue, "ALL")


@pytest.mark.asyncio
async def test_workflow_execution(engine: WorkflowEngine, workflow: Workflow):
    """Test that the workflow executes correctly and produces the expected result."""
    context = InMemoryExecutionContext()

    c = -256

    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={"c": c},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"sum": 42 + 2025 + c}
