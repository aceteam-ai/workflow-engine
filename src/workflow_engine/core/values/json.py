# workflow_engine/core/values/json.py

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import TYPE_CHECKING

from .mapping import StringMapValue
from .primitives import BooleanValue, FloatValue, IntegerValue, NullValue
from .sequence import SequenceValue
from .value import Caster, Value, get_origin_and_args

if TYPE_CHECKING:
    from ..context import ExecutionContext


type JSON = Mapping[str, JSON] | Sequence[JSON] | None | bool | int | Decimal | str

# Wider than JSON: accepts float literals at construction time (e.g. from
# json.loads or Python code). Stored .root values remain JSON (Decimal, not float).
type JSONInput = (
    Mapping[str, JSONInput]
    | Sequence[JSONInput]
    | None
    | bool
    | int
    | float
    | Decimal
    | str
)


class JSONValue(Value[JSON]):
    # Pyright treats Value[JSON] as "constructor takes JSON only", but JSON
    # deliberately omits float (numbers round-trip as Decimal). Call sites still
    # pass float payloads — JSONValue(3.14), nested dicts with 1.5, etc. Widening
    # the JSON alias would fix __init__ typing but also widen .root; this stub
    # accepts JSONInput at construction while .root stays JSON.
    if TYPE_CHECKING:

        def __init__(self, root: JSONInput, /) -> None: ...


@Value.register_cast_to(JSONValue)
def cast_any_to_json(value: Value, context: "ExecutionContext") -> JSONValue:
    return JSONValue(value.model_dump(mode="json"))


@JSONValue.register_cast_to(NullValue)
def cast_json_to_null(value: JSONValue, context: "ExecutionContext") -> NullValue:
    if value.root is None:
        return NullValue(None)
    raise ValueError(f"Expected null, got {type(value.root)}")


@JSONValue.register_cast_to(BooleanValue)
def cast_json_to_boolean(value: JSONValue, context: "ExecutionContext") -> BooleanValue:
    if isinstance(value.root, bool):
        return BooleanValue(value.root)
    raise ValueError(f"Expected bool, got {type(value.root)}")


@JSONValue.register_cast_to(IntegerValue)
def cast_json_to_integer(value: JSONValue, context: "ExecutionContext") -> IntegerValue:
    # Note: bool is a subclass of int in Python, so we must exclude it
    if isinstance(value.root, int) and not isinstance(value.root, bool):
        return IntegerValue(value.root)
    raise ValueError(f"Expected int, got {type(value.root)}")


@JSONValue.register_cast_to(FloatValue)
def cast_json_to_float(value: JSONValue, context: "ExecutionContext") -> FloatValue:
    root = value.root
    if isinstance(root, bool):
        raise ValueError(f"Expected float or int, got {type(root)}")
    if isinstance(root, (int, Decimal)):
        return FloatValue(root)
    raise ValueError(f"Expected float or int, got {type(root)}")


@JSONValue.register_generic_cast_to(SequenceValue)
def cast_json_to_sequence(
    source_type: type[JSONValue],
    target_type: type[SequenceValue],
) -> Caster[JSONValue, SequenceValue] | None:
    assert issubclass(source_type, JSONValue)

    target_origin, _ = get_origin_and_args(target_type)
    assert issubclass(target_origin, SequenceValue)

    def _cast(value: JSONValue, context: "ExecutionContext") -> SequenceValue:
        if not isinstance(value.root, Sequence) or isinstance(value.root, str):
            raise ValueError(
                f"Expected sequence, got {type(value.root)} (strings are not valid sequences for this cast)"
            )
        return target_type(value.root)  # type: ignore

    return _cast


@JSONValue.register_generic_cast_to(StringMapValue)
def cast_json_to_string_map(
    source_type: type[JSONValue],
    target_type: type[StringMapValue],
) -> Caster[JSONValue, StringMapValue] | None:
    assert issubclass(source_type, JSONValue)

    target_origin, _ = get_origin_and_args(target_type)
    assert issubclass(target_origin, StringMapValue)

    def _cast(value: JSONValue, context: "ExecutionContext") -> StringMapValue:
        if not isinstance(value.root, Mapping):
            raise ValueError(f"Expected mapping, got {type(value.root)}")
        return target_type(value.root)  # type: ignore

    return _cast


__all__ = [
    "JSON",
    "JSONInput",
    "JSONValue",
]
