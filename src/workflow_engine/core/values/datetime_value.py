from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated

from pydantic import BeforeValidator, PlainSerializer

from .primitives import FloatValue, IntegerValue, StringValue
from .value import Value

if TYPE_CHECKING:
    from ..context import ExecutionContext


def _to_utc_datetime(value: datetime | Decimal | int | float | str) -> datetime:
    if value is None:
        raise ValueError("Expected datetime")
    if isinstance(value, bool):
        raise TypeError("bool is not a valid datetime")
    if isinstance(value, (datetime, str)):
        dt = value if isinstance(value, datetime) else datetime.fromisoformat(value)
        # Naive datetimes are UTC, not local time — values cross machines and timezones.
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(value, Decimal):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, int):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, float):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    raise TypeError(f"Cannot convert {type(value).__name__} to datetime")


def _serialize_datetime_for_json(value: datetime) -> str:
    return value.isoformat()


# JSON has no datetime type. Pydantic emits `type: string` with `format: date-time`
# (from the JSON Schema Validation spec) because the wire representation is ISO 8601.
_UtcDateTimeRoot = Annotated[
    datetime,
    BeforeValidator(_to_utc_datetime),
    PlainSerializer(
        _serialize_datetime_for_json,
        return_type=str,
        when_used="json",
    ),
]


class DateValue(Value[_UtcDateTimeRoot]):
    """A timezone-aware UTC instant serialized as ISO 8601."""

    # Pyright reads Annotated[datetime, BeforeValidator(...)] as "constructor takes
    # datetime only", but BeforeValidator(_to_utc_datetime) coerces int/float/Decimal/
    # ISO strings at runtime. Widening _UtcDateTimeRoot's Annotated inner type would
    # fix __init__ typing but also widen .root to the union — we want .root to stay
    # datetime.
    if TYPE_CHECKING:

        def __init__(
            self,
            root: datetime | Decimal | int | float | str,
            /,
        ) -> None: ...

    def __str__(self) -> str:
        return self.root.isoformat()

    def timestamp(self) -> Decimal:
        return Decimal(self.root.timestamp())


@IntegerValue.register_cast_to(DateValue)
def cast_integer_to_date(
    value: IntegerValue,
    context: ExecutionContext,
) -> DateValue:
    return DateValue(value.root)


@FloatValue.register_cast_to(DateValue)
def cast_float_to_date(
    value: FloatValue,
    context: ExecutionContext,
) -> DateValue:
    return DateValue(value.root)


@StringValue.register_cast_to(DateValue)
def cast_string_to_date(
    value: StringValue,
    context: ExecutionContext,
) -> DateValue:
    return DateValue(value.root)


@DateValue.register_cast_to(StringValue)
def cast_date_to_string(
    value: DateValue,
    context: ExecutionContext,
) -> StringValue:
    return StringValue(value.root.isoformat())


@DateValue.register_cast_to(FloatValue)
def cast_date_to_float(
    value: DateValue,
    context: ExecutionContext,
) -> FloatValue:
    return FloatValue(value.timestamp())


__all__ = ("DateValue",)
