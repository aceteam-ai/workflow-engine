# workflow_engine/core/values/json.py

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Type

from .mapping import StringMapValue
from .primitives import BooleanValue, FloatValue, IntegerValue, NullValue
from .sequence import SequenceValue
from .value import Caster, Value, get_origin_and_args

if TYPE_CHECKING:
    from ..context import Context


type JSON = Mapping[str, JSON] | Sequence[JSON] | None | bool | int | float | str


class JSONValue(Value[JSON]):
    pass


@Value.register_cast_to(JSONValue)
def cast_any_to_json(value: Value, context: "Context") -> JSONValue:
    return JSONValue(value.model_dump())


@JSONValue.register_cast_to(NullValue)
def cast_json_to_null(value: JSONValue, context: "Context") -> NullValue:
    if value.root is None:
        return NullValue(None)
    raise ValueError(f"Expected null, got {type(value.root)}")


@JSONValue.register_cast_to(BooleanValue)
def cast_json_to_boolean(value: JSONValue, context: "Context") -> BooleanValue:
    if isinstance(value.root, bool):
        return BooleanValue(value.root)
    raise ValueError(f"Expected bool, got {type(value.root)}")


@JSONValue.register_cast_to(IntegerValue)
def cast_json_to_integer(value: JSONValue, context: "Context") -> IntegerValue:
    # Note: bool is a subclass of int in Python, so we must exclude it
    if isinstance(value.root, int) and not isinstance(value.root, bool):
        return IntegerValue(value.root)
    raise ValueError(f"Expected int, got {type(value.root)}")


@JSONValue.register_cast_to(FloatValue)
def cast_json_to_float(value: JSONValue, context: "Context") -> FloatValue:
    # Accept both int and float, but not bool
    if isinstance(value.root, (int, float)) and not isinstance(value.root, bool):
        return FloatValue(float(value.root))
    raise ValueError(f"Expected float or int, got {type(value.root)}")


@JSONValue.register_generic_cast_to(SequenceValue)
def cast_json_to_sequence(
    source_type: Type[JSONValue],
    target_type: Type[SequenceValue],
) -> Caster[JSONValue, SequenceValue] | None:
    assert issubclass(source_type, JSONValue)

    target_origin, _ = get_origin_and_args(target_type)
    assert target_origin is SequenceValue

    def _cast(value: JSONValue, context: "Context") -> SequenceValue:
        if not isinstance(value.root, Sequence) or isinstance(value.root, str):
            raise ValueError(
                f"Expected sequence, got {type(value.root)} (strings are not valid sequences for this cast)"
            )
        return target_type(value.root)  # type: ignore

    return _cast


@JSONValue.register_generic_cast_to(StringMapValue)
def cast_json_to_string_map(
    source_type: Type[JSONValue],
    target_type: Type[StringMapValue],
) -> Caster[JSONValue, StringMapValue] | None:
    assert issubclass(source_type, JSONValue)

    target_origin, _ = get_origin_and_args(target_type)
    assert target_origin is StringMapValue

    def _cast(value: JSONValue, context: "Context") -> StringMapValue:
        if not isinstance(value.root, Mapping):
            raise ValueError(f"Expected mapping, got {type(value.root)}")
        return target_type(value.root)  # type: ignore

    return _cast


__all__ = [
    "JSON",
    "JSONValue",
]
