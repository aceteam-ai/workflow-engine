# workflow_engine/core/values/primitives.py

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated

from pydantic import (
    BeforeValidator,
    GetJsonSchemaHandler,
    PlainSerializer,
    ValidationError,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from ...utils.iter import only
from .decimal_utils import to_decimal
from .value import Value

if TYPE_CHECKING:
    from ..context import ExecutionContext


def _serialize_decimal_for_json(value: Decimal) -> float:
    return float(value)


_NUMERIC_SCHEMA_ALIASES = {
    "ge": "minimum",
    "le": "maximum",
    "gt": "exclusiveMinimum",
    "lt": "exclusiveMaximum",
    "multiple_of": "multipleOf",
}


class _FlattenDecimalJsonSchema:
    """Pydantic emits Decimal as anyOf[number, string]; we want plain type: number."""

    def __get_pydantic_json_schema__(
        self,
        core_schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        json_schema = handler(core_schema)
        if "anyOf" not in json_schema:
            return json_schema
        number_schema = only(
            item for item in json_schema["anyOf"] if item.get("type") == "number"
        )
        result = dict(number_schema)
        for key, value in json_schema.items():
            if key == "anyOf":
                continue
            out_key = _NUMERIC_SCHEMA_ALIASES.get(key, key)
            if out_key == "type" and "type" in result:
                continue
            result[out_key] = value
        return result


_DecimalRoot = Annotated[
    Decimal,
    BeforeValidator(to_decimal),
    PlainSerializer(_serialize_decimal_for_json, return_type=float, when_used="json"),
    _FlattenDecimalJsonSchema(),
]


class BooleanValue(Value[bool]):
    pass


class FloatValue(Value[_DecimalRoot]):
    # Pyright reads Annotated[Decimal, BeforeValidator(...)] as "constructor takes
    # Decimal only", but BeforeValidator(to_decimal) coerces int/float at runtime.
    # Widening _DecimalRoot's Annotated inner type would fix __init__ typing but
    # also widen .root to the union — we want .root to stay Decimal. str is
    # deliberately omitted here so callers don't normalize FloatValue("3.14");
    # strings still coerce via model_validate / StringValue casts when needed.
    if TYPE_CHECKING:

        def __init__(self, root: int | float | Decimal, /) -> None: ...

    def is_integer(self) -> bool:
        return self.root == self.root.to_integral_value()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Value):
            return self.root == other.root
        if isinstance(other, bool):
            return False
        if isinstance(other, (int, float, Decimal)):
            try:
                return self.root == FloatValue(other).root
            except ValidationError:
                return False
        return False

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        json_schema = handler(core_schema)
        extras = cls.model_config.get("json_schema_extra")
        if isinstance(extras, Mapping):
            json_schema = {**json_schema, **extras}
        return json_schema


class IntegerValue(Value[int]):
    def __index__(self) -> int:
        return self.root.__index__()


class NullValue(Value[None]):
    pass


class StringValue(Value[str]):
    def __len__(self) -> int:
        return len(self.root)

    def __contains__(self, substring: str | StringValue) -> bool:
        if isinstance(substring, StringValue):
            substring = substring.root
        return substring in self.root


@IntegerValue.register_cast_to(FloatValue)
def cast_integer_to_float(
    value: IntegerValue,
    context: "ExecutionContext",
) -> FloatValue:
    return FloatValue(value.root)


@FloatValue.register_cast_to(IntegerValue)
def cast_float_to_integer(
    value: FloatValue,
    context: "ExecutionContext",
) -> IntegerValue:
    """
    Convert a float to an integer only if the float is already an integer.
    Otherwise, raise a ValueError.
    """
    if value.is_integer():
        return IntegerValue(int(value.root))
    else:
        raise ValueError(f"Cannot convert {value} to {IntegerValue}")


@Value.register_cast_to(StringValue)
def cast_value_to_string(
    value: Value,
    context: "ExecutionContext",
) -> StringValue:
    return StringValue(str(value.root))


@StringValue.register_cast_to(BooleanValue)
def cast_string_to_boolean(
    value: StringValue,
    context: "ExecutionContext",
) -> BooleanValue:
    return BooleanValue.model_validate_json(value.root)


@StringValue.register_cast_to(IntegerValue)
def cast_string_to_integer(
    value: StringValue,
    context: "ExecutionContext",
) -> IntegerValue:
    return IntegerValue.model_validate_json(value.root)


@StringValue.register_cast_to(FloatValue)
def cast_string_to_float(
    value: StringValue,
    context: "ExecutionContext",
) -> FloatValue:
    return FloatValue.model_validate_json(value.root)


__all__ = [
    "BooleanValue",
    "FloatValue",
    "IntegerValue",
    "NullValue",
    "StringValue",
]
