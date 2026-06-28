from decimal import Decimal

import pytest

from workflow_engine import (
    Edge,
    FloatValue,
    SequenceValue,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core.stakeholder import StakeholderLevel
from workflow_engine.nodes import (
    AbsoluteValueNode,
    DivideNode,
    MaximumNode,
    MinimumNode,
    MultiplyNode,
    NegateNode,
    PowerNode,
    RoundNode,
    SubtractNode,
)
from workflow_engine.nodes.arithmetic import _divide_with_remainder, _round_decimal


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


@pytest.fixture
def context() -> InMemoryExecutionContext:
    return InMemoryExecutionContext()


async def _run_unary(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
    *,
    node_cls,
    node_id: str,
    input_value: float | int,
    output_key: str,
    params: dict | None = None,
) -> dict:
    workflow = Workflow(
        input_node=(input_node := engine.create_input_node(a=FloatValue)),
        output_node=(
            output_node := engine.create_output_node(**{output_key: FloatValue})
        ),
        inner_nodes=[
            node := engine.create_node(node_cls, id=node_id, params=params or {}),
        ],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="a",
                target=node,
                target_key="a",
            ),
            Edge.from_nodes(
                source=node,
                source_key=output_key,
                target=output_node,
                target_key=output_key,
            ),
        ],
    )
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={"a": input_value},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output is not None
    return result.output


async def _run_binary(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
    *,
    node_cls,
    node_id: str,
    a: float | int,
    b: float | int,
    output_key: str | tuple[str, ...],
    input_keys: tuple[str, str] = ("a", "b"),
    params: dict | None = None,
) -> dict:
    left_key, right_key = input_keys
    output_keys = (output_key,) if isinstance(output_key, str) else output_key
    workflow = Workflow(
        input_node=(
            input_node := engine.create_input_node(
                **{left_key: FloatValue, right_key: FloatValue},
            )
        ),
        output_node=(
            output_node := engine.create_output_node(
                **{key: FloatValue for key in output_keys},
            )
        ),
        inner_nodes=[
            node := engine.create_node(node_cls, id=node_id, params=params or {}),
        ],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key=left_key,
                target=node,
                target_key=left_key,
            ),
            Edge.from_nodes(
                source=input_node,
                source_key=right_key,
                target=node,
                target_key=right_key,
            ),
            *(
                Edge.from_nodes(
                    source=node,
                    source_key=key,
                    target=output_node,
                    target_key=key,
                )
                for key in output_keys
            ),
        ],
    )
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={left_key: a, right_key: b},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output is not None
    return result.output


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subtract(engine: WorkflowEngine, context: InMemoryExecutionContext):
    output = await _run_binary(
        engine,
        context,
        node_cls=SubtractNode,
        node_id="sub",
        a=10,
        b=3,
        output_key="difference",
        input_keys=("minuend", "subtrahend"),
    )
    assert output["difference"] == 7


@pytest.mark.unit
@pytest.mark.asyncio
async def test_divide(engine: WorkflowEngine, context: InMemoryExecutionContext):
    output = await _run_binary(
        engine,
        context,
        node_cls=DivideNode,
        node_id="div",
        a=10,
        b=3,
        output_key=("quotient", "integer_quotient", "remainder"),
        input_keys=("dividend", "divisor"),
    )
    assert output["quotient"] == Decimal("10") / Decimal("3")
    assert output["integer_quotient"] == 3
    assert output["remainder"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_divide_exact_quotient_with_integer_part(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    output = await _run_binary(
        engine,
        context,
        node_cls=DivideNode,
        node_id="div",
        a=7,
        b=2,
        output_key=("quotient", "integer_quotient", "remainder"),
        input_keys=("dividend", "divisor"),
    )
    assert output["quotient"] == 3.5
    assert output["integer_quotient"] == 3
    assert output["remainder"] == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dividend", "divisor", "mode", "expected_integer_quotient", "expected_remainder"),
    [
        (Decimal("10"), Decimal("3"), "floor", Decimal("3"), Decimal("1")),
        (Decimal("-10"), Decimal("3"), "floor", Decimal("-4"), Decimal("2")),
        (Decimal("10"), Decimal("3"), "ceiling", Decimal("4"), Decimal("-2")),
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
        mode=mode,
    )
    assert quotient == dividend / divisor
    assert integer_quotient == expected_integer_quotient
    assert remainder == expected_remainder


@pytest.mark.unit
@pytest.mark.asyncio
async def test_divide_by_zero(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    workflow = Workflow(
        input_node=(
            input_node := engine.create_input_node(
                dividend=FloatValue,
                divisor=FloatValue,
            )
        ),
        output_node=(output_node := engine.create_output_node(quotient=FloatValue)),
        inner_nodes=[divide := engine.create_node(DivideNode, id="div")],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="dividend",
                target=divide,
                target_key="dividend",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="divisor",
                target=divide,
                target_key="divisor",
            ),
            Edge.from_nodes(
                source=divide,
                source_key="quotient",
                target=output_node,
                target_key="quotient",
            ),
        ],
    )
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={"dividend": 1, "divisor": 0},
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert "div" in result.errors.node_errors
    error = result.errors.node_errors["div"][0]
    assert error.level is StakeholderLevel.USER
    assert "divide by zero" in error.message.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_power(engine: WorkflowEngine, context: InMemoryExecutionContext):
    output = await _run_binary(
        engine,
        context,
        node_cls=PowerNode,
        node_id="pow",
        a=2,
        b=10,
        output_key="power",
        input_keys=("base", "exponent"),
    )
    assert output["power"] == 1024


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multiply_scalar(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    workflow = Workflow(
        input_node=(input_node := engine.create_input_node(values=FloatValue)),
        output_node=(output_node := engine.create_output_node(product=FloatValue)),
        inner_nodes=[multiply := engine.create_node(MultiplyNode, id="mul")],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="values",
                target=multiply,
                target_key="values",
            ),
            Edge.from_nodes(
                source=multiply,
                source_key="product",
                target=output_node,
                target_key="product",
            ),
        ],
    )
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={"values": 6},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"product": 6}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multiply_sequence(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    workflow = Workflow(
        input_node=(
            input_node := engine.create_input_node(values=SequenceValue[FloatValue])
        ),
        output_node=(output_node := engine.create_output_node(product=FloatValue)),
        inner_nodes=[multiply := engine.create_node(MultiplyNode, id="mul")],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="values",
                target=multiply,
                target_key="values",
            ),
            Edge.from_nodes(
                source=multiply,
                source_key="product",
                target=output_node,
                target_key="product",
            ),
        ],
    )
    result = await engine.execute(
        context=context,
        workflow=workflow,
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
    workflow = Workflow(
        input_node=(
            input_node := engine.create_input_node(values=SequenceValue[FloatValue])
        ),
        output_node=(output_node := engine.create_output_node(product=FloatValue)),
        inner_nodes=[multiply := engine.create_node(MultiplyNode, id="mul")],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="values",
                target=multiply,
                target_key="values",
            ),
            Edge.from_nodes(
                source=multiply,
                source_key="product",
                target=output_node,
                target_key="product",
            ),
        ],
    )
    result = await engine.execute(
        context=context,
        workflow=workflow,
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
    workflow = Workflow(
        input_node=(
            input_node := engine.create_input_node(values=SequenceValue[FloatValue])
        ),
        output_node=(output_node := engine.create_output_node(minimum=FloatValue)),
        inner_nodes=[min_node := engine.create_node(MinimumNode, id="min")],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="values",
                target=min_node,
                target_key="values",
            ),
            Edge.from_nodes(
                source=min_node,
                source_key="minimum",
                target=output_node,
                target_key="minimum",
            ),
        ],
    )
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={"values": []},
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert "min" in result.errors.node_errors


@pytest.mark.unit
@pytest.mark.asyncio
async def test_negate_and_abs(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    negated = await _run_unary(
        engine,
        context,
        node_cls=NegateNode,
        node_id="neg",
        input_value=5,
        output_key="negated",
    )
    assert negated["negated"] == -5

    absolute = await _run_unary(
        engine,
        context,
        node_cls=AbsoluteValueNode,
        node_id="abs",
        input_value=-5,
        output_key="absolute",
    )
    assert absolute["absolute"] == 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_round_half_even(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    output = await _run_unary(
        engine,
        context,
        node_cls=RoundNode,
        node_id="round",
        input_value=2.5,
        output_key="rounded",
        params={"digits": 0, "rounding_mode": "half_even"},
    )
    assert output["rounded"] == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_round_half_away_from_zero(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    output = await _run_unary(
        engine,
        context,
        node_cls=RoundNode,
        node_id="round",
        input_value=2.5,
        output_key="rounded",
        params={"digits": 0, "rounding_mode": "half_away_from_zero"},
    )
    assert output["rounded"] == 3


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "mode", "expected"),
    [
        (Decimal("23.7"), "floor", Decimal("23")),
        (Decimal("-23.2"), "floor", Decimal("-24")),
        (Decimal("23.2"), "ceiling", Decimal("24")),
        (Decimal("-23.7"), "ceiling", Decimal("-23")),
        (Decimal("23.7"), "toward_zero", Decimal("23")),
        (Decimal("-23.7"), "toward_zero", Decimal("-23")),
        (Decimal("23.2"), "away_from_zero", Decimal("24")),
        (Decimal("-23.2"), "away_from_zero", Decimal("-24")),
        (Decimal("23.5"), "half_toward_positive_infinity", Decimal("24")),
        (Decimal("-23.5"), "half_toward_positive_infinity", Decimal("-23")),
        (Decimal("23.5"), "half_toward_negative_infinity", Decimal("23")),
        (Decimal("-23.5"), "half_toward_negative_infinity", Decimal("-24")),
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
    assert _round_decimal(value, digits=0, mode=mode) == expected


@pytest.mark.unit
@pytest.mark.asyncio
async def test_round_decimal_fractions_exact(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    """Round preserves Decimal exactness (0.1 + 0.2 stays 0.3 at one decimal place)."""
    workflow = Workflow(
        input_node=(input_node := engine.create_input_node(a=FloatValue)),
        output_node=(output_node := engine.create_output_node(rounded=FloatValue)),
        inner_nodes=[
            round_node := engine.create_node(
                RoundNode,
                id="round",
                params={"digits": 1, "rounding_mode": "half_away_from_zero"},
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="a",
                target=round_node,
                target_key="a",
            ),
            Edge.from_nodes(
                source=round_node,
                source_key="rounded",
                target=output_node,
                target_key="rounded",
            ),
        ],
    )
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={"a": Decimal("0.1") + Decimal("0.2")},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"rounded": 0.3}
