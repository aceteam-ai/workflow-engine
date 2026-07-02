from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated

from pydantic import BeforeValidator, ConfigDict, PlainSerializer

from .datetime_utils import parse_iso8601_datetime, to_utc_datetime
from .primitives import FloatValue, IntegerValue, StringValue
from .value import Value

if TYPE_CHECKING:
    from ..context import ExecutionContext


def _serialize_datetime_for_json(value: datetime) -> str:
    return value.isoformat()


_UtcDateTimeRoot = Annotated[
    datetime,
    BeforeValidator(to_utc_datetime),
    PlainSerializer(
        _serialize_datetime_for_json,
        return_type=str,
        when_used="json",
    ),
]


class DateValue(Value[_UtcDateTimeRoot]):
    """A timezone-aware UTC instant serialized as ISO 8601."""

    model_config = Value.model_config | ConfigDict(
        json_schema_extra={"format": "date-time"},
    )

    if TYPE_CHECKING:

        def __init__(
            self,
            root: datetime | int | float | str,
            /,
        ) -> None: ...

    def __str__(self) -> str:
        return self.root.isoformat()


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
    return DateValue(float(value.root))


@StringValue.register_cast_to(DateValue)
def cast_string_to_date(
    value: StringValue,
    context: ExecutionContext,
) -> DateValue:
    return DateValue(parse_iso8601_datetime(value.root))


@DateValue.register_cast_to(StringValue)
def cast_date_to_string(
    value: DateValue,
    context: ExecutionContext,
) -> StringValue:
    return StringValue(value.root.isoformat())


__all__ = ["DateValue"]
