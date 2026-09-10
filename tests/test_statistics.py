"""Numeric conventions, Decimal behavior, portable schemas, and sequence length."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from workflow_engine import (
    ErrorClass,
    ExecutionAlgorithm,
    FloatValue,
    IntegerValue,
    Result,
    SequenceValue,
    StringValue,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.nodes import (
    AttemptNode,
    LengthNode,
    MedianNode,
    ModeNode,
    PercentileNode,
    QuantileNode,
    RangeNode,
    StandardDeviationNode,
    VarianceNode,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def engine(algorithm: ExecutionAlgorithm) -> WorkflowEngine:
    return WorkflowEngine(execution_algorithm=algorithm)


async def execute_roundtrip(engine, cls, *, values, params=None):
    graph = await engine.build_single_node_workflow(cls, params=params)
    restored = Workflow.model_validate_json(graph.model_dump_json())
    result = await engine.execute(
        context=InMemoryExecutionContext(), workflow=restored, input={"values": values}
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    value = result.output["value"]
    assert isinstance(value, FloatValue)
    return value.root


@pytest.mark.parametrize(
    "values,expected",
    [([5, -3, 0], 0), ([-3, 5], 1), ([7], 7), ([4, 1, 3, 2], Decimal("2.5"))],
)
async def test_median_roundtrip(engine, values, expected):
    assert await execute_roundtrip(engine, MedianNode, values=values) == expected


@pytest.mark.parametrize(
    "values,expected",
    [([3, 1, 3, 1], 3), ([1, 3, 3, 1], 1), ([4, 7, 1], 4), ([1, 2, 2, 3], 2), ([8], 8)],
)
async def test_mode_ties_keep_first_encountered(engine, values, expected):
    assert await execute_roundtrip(engine, ModeNode, values=values) == expected


@pytest.mark.parametrize(
    "values,expected", [([-10, -3, -7], 7), ([8], 0), ([0.1, 0.3], Decimal("0.2"))]
)
async def test_range(engine, values, expected):
    assert await execute_roundtrip(engine, RangeNode, values=values) == expected


@pytest.mark.parametrize(
    "cls,values,population,expected",
    [
        (VarianceNode, [1, 3], False, 2),
        (VarianceNode, [1, 3], True, 1),
        (VarianceNode, [7], True, 0),
        (StandardDeviationNode, [2, 4, 4, 4, 5, 5, 7, 9], True, 2),
        (StandardDeviationNode, [0, 2], False, Decimal(2).sqrt()),
        (StandardDeviationNode, [7], True, 0),
    ],
)
async def test_population_and_sample_statistics(
    engine, cls, values, population, expected
):
    assert (
        await execute_roundtrip(
            engine, cls, values=values, params={"population": population}
        )
        == expected
    )


@pytest.mark.parametrize(
    "cls,expected", [(VarianceNode, 2), (StandardDeviationNode, Decimal(2).sqrt())]
)
async def test_population_defaults_to_sample(engine, cls, expected):
    assert await execute_roundtrip(engine, cls, values=[1, 3]) == expected
    raw = cls.TYPE_INFO.parameter_schema.model_dump(mode="json")
    assert raw["properties"]["population"]["default"] is False


@pytest.mark.parametrize("cls", [VarianceNode, StandardDeviationNode])
async def test_sample_requires_two_values(engine, cls):
    inner = await engine.build_single_node_workflow(cls)
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=AttemptNode,
        params={"workflow": inner},
        input={"values": [7]},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    error_value = result.output["result"]
    assert isinstance(error_value, Result)
    error = error_value.unwrap_err()
    assert error.error_class.root is ErrorClass.VALIDATION
    assert "at least two" in error.message.root


@pytest.mark.parametrize(
    "cls,params",
    [
        (MedianNode, {}),
        (ModeNode, {}),
        (VarianceNode, {}),
        (StandardDeviationNode, {}),
        (RangeNode, {}),
        (PercentileNode, {"q": 50}),
        (QuantileNode, {"q": 0.5}),
    ],
)
async def test_empty_statistics_are_user_visible_validation_errors(engine, cls, params):
    inner = await engine.build_single_node_workflow(cls, params=params)
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=AttemptNode,
        params={"workflow": inner},
        input={"values": []},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    captured = result.output["result"]
    assert isinstance(captured, Result)
    error = captured.unwrap_err()
    assert error.error_class.root is ErrorClass.VALIDATION
    assert "empty sequence" in error.message.root


@pytest.mark.parametrize(
    "cls,params,expected",
    [
        (MedianNode, {}, "0.15"),
        (ModeNode, {}, "0.1"),
        (VarianceNode, {}, "0.005"),
        (RangeNode, {}, "0.1"),
        (QuantileNode, {"q": 0.25}, "0.125"),
        (PercentileNode, {"q": 25}, "0.125"),
    ],
)
async def test_decimal_values_avoid_binary_float_roundtrips(
    engine, cls, params, expected
):
    result = await execute_roundtrip(
        engine, cls, values=[Decimal("0.1"), Decimal("0.2")], params=params
    )
    assert result == Decimal(expected)


@pytest.mark.parametrize(
    "method,expected",
    [
        ("linear", Decimal("7.5")),
        ("lower", 0),
        ("higher", 10),
        ("midpoint", 5),
        ("nearest", 10),
    ],
)
async def test_quantile_interpolation_methods(engine, method, expected):
    values = [30, 0, 20, 10]
    assert (
        await execute_roundtrip(
            engine,
            QuantileNode,
            values=values,
            params={"q": 0.25, "interpolation": method},
        )
        == expected
    )
    assert (
        await execute_roundtrip(
            engine,
            PercentileNode,
            values=values,
            params={"q": 25, "interpolation": method},
        )
        == expected
    )


@pytest.mark.parametrize("method", ["linear", "lower", "higher", "midpoint", "nearest"])
async def test_quantile_endpoints_and_singleton(engine, method):
    assert (
        await execute_roundtrip(
            engine,
            QuantileNode,
            values=[10, 0, 20],
            params={"q": 0, "interpolation": method},
        )
        == 0
    )
    assert (
        await execute_roundtrip(
            engine,
            QuantileNode,
            values=[10, 0, 20],
            params={"q": 1, "interpolation": method},
        )
        == 20
    )
    assert (
        await execute_roundtrip(
            engine,
            QuantileNode,
            values=[7],
            params={"q": 0.35, "interpolation": method},
        )
        == 7
    )


async def test_nearest_interpolation_uses_even_index_on_ties(engine):
    assert (
        await execute_roundtrip(
            engine,
            QuantileNode,
            values=[0, 10],
            params={"q": 0.5, "interpolation": "nearest"},
        )
        == 0
    )
    assert (
        await execute_roundtrip(
            engine,
            QuantileNode,
            values=[0, 10, 20, 30],
            params={"q": 0.5, "interpolation": "nearest"},
        )
        == 20
    )


@pytest.mark.parametrize(
    "cls,q",
    [
        (PercentileNode, -1),
        (PercentileNode, 101),
        (QuantileNode, -0.1),
        (QuantileNode, 1.1),
    ],
)
async def test_quantile_rejects_out_of_range_parameter(engine, cls, q):
    with pytest.raises(ValidationError, match="between"):
        engine.create_node(cls, id="statistic", params={"q": q})


@pytest.mark.parametrize("cls,q", [(PercentileNode, 50), (QuantileNode, 0.5)])
async def test_quantile_explicit_default_and_unknown_interpolation(engine, cls, q):
    raw = cls.TYPE_INFO.parameter_schema.model_dump(mode="json")
    assert raw["properties"]["interpolation"]["default"] == "linear"
    with pytest.raises(ValidationError, match="interpolation"):
        engine.create_node(
            cls, id="statistic", params={"q": q, "interpolation": "unknown"}
        )
    with pytest.raises(ValidationError, match="q"):
        engine.create_node(cls, id="statistic", params={})


async def test_integer_source_casts_to_statistics_input(engine):
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=MedianNode,
        input_fields={"values": SequenceValue[IntegerValue]},
        input={"values": [1, 2]},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["value"].root == Decimal("1.5")


@pytest.mark.parametrize(
    "value_type,items,expected",
    [(IntegerValue, [], 0), (IntegerValue, [1, 2, 3], 3), (StringValue, ["a", "b"], 2)],
)
async def test_generic_length_roundtrip(engine, value_type, items, expected):
    graph = await engine.build_single_node_workflow(
        LengthNode, params={"element_schema": value_type.to_value_schema()}
    )
    result = await engine.execute(
        context=InMemoryExecutionContext(),
        workflow=Workflow.model_validate_json(graph.model_dump_json()),
        input={"sequence": items},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["length"].root == expected


async def test_length_counts_result_errors_as_elements(engine):
    items = [
        {"tag": "ok", "ok": 1},
        {
            "tag": "err",
            "err": {
                "name": "Missing",
                "message": "missing",
                "node_id": "origin",
                "error_class": "validation",
            },
        },
    ]
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=LengthNode,
        params={"element_schema": Result[IntegerValue].to_value_schema()},
        input={"sequence": items},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["length"].root == 2
