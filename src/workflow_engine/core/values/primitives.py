# workflow_engine/core/values/primitives.py

from __future__ import annotations

from typing import TYPE_CHECKING

from .value import Value

if TYPE_CHECKING:
    from ..context import ExecutionContext


class BooleanValue(Value[bool]):
    pass


class FloatValue(Value[float]):
    def is_integer(self) -> bool:
        return self.root.is_integer()


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
    value: IntegerValue, context: "ExecutionContext"
) -> FloatValue:
    return FloatValue(float(value.root))


@FloatValue.register_cast_to(IntegerValue)
def cast_float_to_integer(
    value: FloatValue, context: "ExecutionContext"
) -> IntegerValue:
    """
    Convert a float to an integer only if the float is already an integer.
    Otherwise, raise a ValueError.
    """
    if value.root.is_integer():
        return IntegerValue(int(value.root))
    else:
        raise ValueError(f"Cannot convert {value} to {IntegerValue}")


@Value.register_cast_to(StringValue)
def cast_value_to_string(value: Value, context: "ExecutionContext") -> StringValue:
    return StringValue(str(value.root))


@StringValue.register_cast_to(BooleanValue)
def cast_string_to_boolean(
    value: StringValue, context: "ExecutionContext"
) -> BooleanValue:
    return BooleanValue.model_validate_json(value.root)


@StringValue.register_cast_to(IntegerValue)
def cast_string_to_integer(
    value: StringValue, context: "ExecutionContext"
) -> IntegerValue:
    return IntegerValue.model_validate_json(value.root)


@StringValue.register_cast_to(FloatValue)
def cast_string_to_float(value: StringValue, context: "ExecutionContext") -> FloatValue:
    return FloatValue.model_validate_json(value.root)


__all__ = [
    "BooleanValue",
    "FloatValue",
    "IntegerValue",
    "NullValue",
    "StringValue",
]
