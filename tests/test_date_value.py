from datetime import datetime, timezone

import pytest

from workflow_engine.contexts.in_memory import InMemoryExecutionContext
from workflow_engine.core import DateValue, FloatValue, IntegerValue, StringValue


@pytest.fixture
def context():
    return InMemoryExecutionContext()


@pytest.mark.unit
def test_date_value_json_roundtrip():
    value = DateValue("2026-07-01T07:17:05+00:00")
    assert value.root == datetime(2026, 7, 1, 7, 17, 5, tzinfo=timezone.utc)
    assert value.model_dump_json() == '"2026-07-01T07:17:05+00:00"'
    assert DateValue.model_validate_json(value.model_dump_json()) == value
    assert str(value) == "2026-07-01T07:17:05+00:00"


@pytest.mark.unit
async def test_cast_integer_to_date(context):
    value = await IntegerValue(1_719_834_000).cast_to(DateValue, context=context)
    assert isinstance(value, DateValue)
    assert value.root == datetime.fromtimestamp(1_719_834_000, tz=timezone.utc)


@pytest.mark.unit
async def test_cast_float_to_date(context):
    value = await FloatValue(1_719_834_000.5).cast_to(DateValue, context=context)
    assert isinstance(value, DateValue)
    assert value.root == datetime.fromtimestamp(1_719_834_000.5, tz=timezone.utc)


@pytest.mark.unit
async def test_cast_iso_string_to_date(context):
    value = await StringValue("2026-07-01T07:17:05Z").cast_to(
        DateValue, context=context
    )
    assert isinstance(value, DateValue)
    assert value.root == datetime(2026, 7, 1, 7, 17, 5, tzinfo=timezone.utc)


@pytest.mark.unit
async def test_cast_slack_timestamp_string_to_date_fails(context):
    with pytest.raises(ValueError, match="numeric string"):
        await StringValue("1719834567.123456").cast_to(DateValue, context=context)


@pytest.mark.unit
async def test_cast_date_to_string(context):
    value = await DateValue("2026-07-01T07:17:05Z").cast_to(
        StringValue, context=context
    )
    assert isinstance(value, StringValue)
    assert value.root == "2026-07-01T07:17:05+00:00"


@pytest.mark.unit
async def test_cast_date_to_float(context):
    from decimal import Decimal

    date = DateValue(1_719_834_000.5)
    assert date.timestamp() == Decimal(str(1_719_834_000.5))
    float_val = await date.cast_to(FloatValue, context=context)
    assert isinstance(float_val, FloatValue)
    assert float_val == date.timestamp()
