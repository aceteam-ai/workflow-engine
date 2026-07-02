from datetime import datetime, timedelta, timezone

import pytest

from workflow_engine.contexts.in_memory import InMemoryExecutionContext
from workflow_engine.core import (
    DateValue,
    FloatValue,
    IntegerValue,
    StringValue,
)


@pytest.fixture
def context():
    return InMemoryExecutionContext()


@pytest.mark.unit
def test_date_value_from_iso_string():
    value = DateValue("2026-07-01T07:17:05Z")
    assert value.root == datetime(2026, 7, 1, 7, 17, 5, tzinfo=timezone.utc)


@pytest.mark.unit
def test_date_value_from_datetime():
    dt = datetime(2026, 7, 1, 7, 17, 5, tzinfo=timezone.utc)
    value = DateValue(dt)
    assert value.root == dt


@pytest.mark.unit
def test_date_value_from_unix_seconds():
    value = DateValue(1_719_834_000)
    assert value.root == datetime.fromtimestamp(1_719_834_000, tz=timezone.utc)


@pytest.mark.unit
def test_date_value_normalizes_to_utc():
    dt = datetime(2026, 7, 1, 3, 17, 5, tzinfo=timezone(timedelta(hours=-4)))
    value = DateValue(dt)
    assert value.root == dt.astimezone(timezone.utc)


@pytest.mark.unit
def test_date_value_rejects_naive_datetime():
    with pytest.raises(ValueError, match="Naive datetimes"):
        DateValue(datetime(2026, 7, 1, 7, 17, 5))


@pytest.mark.unit
def test_date_value_rejects_date_only_string():
    with pytest.raises(ValueError, match="Naive datetimes"):
        DateValue("2026-07-01")


@pytest.mark.unit
def test_date_value_rejects_slack_style_timestamp_string():
    with pytest.raises(ValueError, match="numeric string"):
        DateValue("1719834567.123456")


@pytest.mark.unit
def test_date_value_json_roundtrip():
    value = DateValue("2026-07-01T07:17:05+00:00")
    json_str = value.model_dump_json()
    assert json_str == '"2026-07-01T07:17:05+00:00"'
    assert DateValue.model_validate_json(json_str) == value


@pytest.mark.unit
def test_date_value_str_is_isoformat():
    value = DateValue("2026-07-01T07:17:05Z")
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
