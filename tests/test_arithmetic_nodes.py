from decimal import Decimal

import pytest

from workflow_engine import (
    Edge,
    FloatValue,
    IntegerValue,
    SequenceValue,
    ValidationContext,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core.stakeholder import StakeholderLevel
from workflow_engine.core.values import get_data_fields
from workflow_engine.core.values.rounding import RoundingMode
from workflow_engine.nodes import (
    AbsoluteValueNode,
    AddNode,
    ConstantIntegerNode,
    DivideNode,
    MaximumNode,
    MinimumNode,
    MultiplyNode,
    NegateNode,
    PowerNode,
    RoundNode,
    SubtractNode,
)
from workflow_engine.nodes.arithmetic import (
    _argument_field_name,
    _divide_with_remainder,
)


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


@pytest.fixture
def context() -> InMemoryExecutionContext:
    return InMemoryExecutionContext()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subtract(engine: WorkflowEngine, context: InMemoryExecutionContext):
    result = await engine.execute_node(
        context=context,
        node=SubtractNode,
        input={"minuend": 10, "subtrahend": 3},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["difference"] == 7


@pytest.mark.unit
@pytest.mark.asyncio
async def test_divide(engine: WorkflowEngine, context: InMemoryExecutionContext):
    result = await engine.execute_node(
        context=context,
        node=DivideNode,
        input={"dividend": 10, "divisor": 3},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["quotient"] == Decimal("10") / Decimal("3")
    assert result.output["integer_quotient"] == 3
    assert result.output["remainder"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_divide_exact_quotient_with_integer_part(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    result = await engine.execute_node(
        context=context,
        node=DivideNode,
        input={"dividend": 7, "divisor": 2},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["quotient"] == 3.5
    assert result.output["integer_quotient"] == 3
    assert result.output["remainder"] == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dividend", "divisor", "mode", "expected_integer_quotient", "expected_remainder"),
    [
        (Decimal("10"), Decimal("3"), "down", Decimal("3"), Decimal("1")),
        (Decimal("-10"), Decimal("3"), "down", Decimal("-4"), Decimal("2")),
        (Decimal("10"), Decimal("3"), "up", Decimal("4"), Decimal("-2")),
        (Decimal("7"), Decimal("2"), "toward_zero", Decimal("3"), Decimal("1")),
    ],
)
def test_divide_with_remainder_rounding_modes(
    dividend: Decimal,
    divisor: Decimal,
    mode: str,
    expected_integer_quotient: Decimal,
    expected_remainder: Decimal,
):
    quotient, integer_quotient, remainder = _divide_with_remainder(
        dividend,
        divisor,
        mode=RoundingMode(mode),
    )
    assert quotient == dividend / divisor
    assert integer_quotient == expected_integer_quotient
    assert remainder == expected_remainder


@pytest.mark.unit
@pytest.mark.asyncio
async def test_divide_by_zero(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    result = await engine.execute_node(
        context=context,
        node=DivideNode,
        input={"dividend": 1, "divisor": 0},
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert "node" in result.errors.node_errors
    error = result.errors.node_errors["node"][0]
    assert error is not None
    assert error.level is StakeholderLevel.USER
    assert "divide by zero" in error.message.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_power(engine: WorkflowEngine, context: InMemoryExecutionContext):
    result = await engine.execute_node(
        context=context,
        node=PowerNode,
        input={"base": 2, "exponent": 10},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["power"] == 1024


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multiply_scalar(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    result = await engine.execute_node(
        context=context,
        node=MultiplyNode,
        input={"values": 6},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"product": 6}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multiply_sequence(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    result = await engine.execute_node(
        context=context,
        node=MultiplyNode,
        input={"values": [2, 3, 4]},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"product": 24}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multiply_empty_sequence(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    result = await engine.execute_node(
        context=context,
        node=MultiplyNode,
        input={"values": []},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"product": 1}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_min_and_max_sequence(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    workflow = Workflow(
        input_node=(
            input_node := engine.create_input_node(values=SequenceValue[FloatValue])
        ),
        output_node=(
            output_node := engine.create_output_node(
                minimum=FloatValue,
                maximum=FloatValue,
            )
        ),
        inner_nodes=[
            min_node := engine.create_node(MinimumNode, id="min"),
            max_node := engine.create_node(MaximumNode, id="max"),
        ],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="values",
                target=min_node,
                target_key="values",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="values",
                target=max_node,
                target_key="values",
            ),
            Edge.from_nodes(
                source=min_node,
                source_key="minimum",
                target=output_node,
                target_key="minimum",
            ),
            Edge.from_nodes(
                source=max_node,
                source_key="maximum",
                target=output_node,
                target_key="maximum",
            ),
        ],
    )
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={"values": [3, 1, 4, 1, 5]},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"minimum": 1, "maximum": 5}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_min_empty_sequence(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    result = await engine.execute_node(
        context=context,
        node=MinimumNode,
        input={"values": []},
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert "node" in result.errors.node_errors


@pytest.mark.unit
@pytest.mark.asyncio
async def test_negate_and_abs(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    negated = await engine.execute_node(
        context=context,
        node=NegateNode,
        input={"a": 5},
    )
    assert negated.status is WorkflowExecutionResultStatus.SUCCESS
    assert negated.output["negated"] == -5

    absolute = await engine.execute_node(
        context=context,
        node=AbsoluteValueNode,
        input={"a": -5},
    )
    assert absolute.status is WorkflowExecutionResultStatus.SUCCESS
    assert absolute.output["absolute"] == 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_round_half_even(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    result = await engine.execute_node(
        context=context,
        node=RoundNode,
        params={"digits": 0, "rounding_mode": "half_even"},
        input={"a": 2.5},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["rounded"] == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_round_half_away_from_zero(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    result = await engine.execute_node(
        context=context,
        node=RoundNode,
        params={"digits": 0, "rounding_mode": "half_away_from_zero"},
        input={"a": 2.5},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["rounded"] == 3


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "mode", "expected"),
    [
        (Decimal("23.7"), "down", Decimal("23")),
        (Decimal("-23.2"), "down", Decimal("-24")),
        (Decimal("23.2"), "up", Decimal("24")),
        (Decimal("-23.7"), "up", Decimal("-23")),
        (Decimal("23.7"), "toward_zero", Decimal("23")),
        (Decimal("-23.7"), "toward_zero", Decimal("-23")),
        (Decimal("23.2"), "away_from_zero", Decimal("24")),
        (Decimal("-23.2"), "away_from_zero", Decimal("-24")),
        (Decimal("23.5"), "half_up", Decimal("24")),
        (Decimal("-23.5"), "half_up", Decimal("-23")),
        (Decimal("23.5"), "half_down", Decimal("23")),
        (Decimal("-23.5"), "half_down", Decimal("-24")),
        (Decimal("23.5"), "half_toward_zero", Decimal("23")),
        (Decimal("-23.5"), "half_toward_zero", Decimal("-23")),
        (Decimal("23.5"), "half_away_from_zero", Decimal("24")),
        (Decimal("-23.5"), "half_away_from_zero", Decimal("-24")),
        (Decimal("2.5"), "half_even", Decimal("2")),
        (Decimal("3.5"), "half_even", Decimal("4")),
        (Decimal("2.5"), "half_odd", Decimal("3")),
        (Decimal("3.5"), "half_odd", Decimal("3")),
    ],
)
def test_round_decimal_wikipedia_modes(value: Decimal, mode: str, expected: Decimal):
    assert RoundingMode(mode).round(value, digits=0) == expected


@pytest.mark.unit
@pytest.mark.asyncio
async def test_round_decimal_fractions_exact(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    """Round preserves Decimal exactness (0.1 + 0.2 stays 0.3 at one decimal place)."""
    result = await engine.execute_node(
        context=context,
        node=RoundNode,
        params={"digits": 1, "rounding_mode": "half_away_from_zero"},
        input={"a": Decimal("0.1") + Decimal("0.2")},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"rounded": 0.3}


@pytest.fixture
def chained_add_workflow(engine: WorkflowEngine) -> Workflow:
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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_add_three_arguments(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    result = await engine.execute_node(
        context=context,
        node=AddNode,
        params={"num_arguments": 3},
        input={"a": 10, "b": 20, "c": 30},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"sum": 60}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_add_thirty_arguments(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    n = 30
    names = [_argument_field_name(i) for i in range(n)]
    result = await engine.execute_node(
        context=context,
        node=AddNode,
        params={"num_arguments": n},
        input={name: i + 1 for i, name in enumerate(names)},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"sum": sum(range(1, n + 1))}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_add_thousand_argument_field_names(engine: WorkflowEngine):
    add = engine.create_node(AddNode, id="add", params={"num_arguments": 1000})
    validation_context = ValidationContext()
    fields = [
        (name, value_type, field_info.title)
        for name, (value_type, field_info) in get_data_fields(
            await add.input_type(validation_context)
        ).items()
    ]
    assert len(fields) == 1000
    assert fields[0] == ("a", FloatValue, "A")
    assert fields[25] == ("z", FloatValue, "Z")
    assert fields[26] == ("aa", FloatValue, "AA")
    assert fields[701] == ("zz", FloatValue, "ZZ")
    assert fields[702] == ("aaa", FloatValue, "AAA")
    assert fields[999] == ("all", FloatValue, "ALL")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_add_exact_decimal_fractions(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    """AddNode sums decimal fractions exactly (0.1 + 0.2 == 0.3)."""
    result = await engine.execute_node(
        context=context,
        node=AddNode,
        input={"a": 0.1, "b": 0.2},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["sum"] == 0.3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chained_add_workflow(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
    chained_add_workflow: Workflow,
):
    result = await engine.execute(
        context=context,
        workflow=chained_add_workflow,
        input={"c": -256},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"sum": 42 + 2025 - 256}
