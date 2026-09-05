"""
Tests that every Value/Data type satisfies the round-trip property:
    X.to_value_schema().to_value_cls() == X

This ensures converting a Value type to a schema and back yields the same type.
"""

from typing import cast

import pytest
from pydantic.fields import FieldInfo

from workflow_engine import (
    BooleanValue,
    Data,
    FileValue,
    FloatValue,
    IntegerValue,
    JSONValue,
    NullValue,
    Result,
    SequenceValue,
    StringMapValue,
    StringValue,
    UnionValue,
    ValueSchemaValue,
    ValueType,
    WorkflowValue,
)
from workflow_engine.core.values.data import (
    DataValue,
    build_data_type,
    get_data_dict,
    get_data_schema,
)
from workflow_engine.core.values.result import ErrorClass, ErrorClassValue, ResultError
from workflow_engine.core.values.schema import validate_value_schema
from workflow_engine.core.values.union import resolve_union_type
from workflow_engine.core.values.value import Value, ValueRegistry, get_origin_and_args
from workflow_engine.files import (
    JSONFileValue,
    JSONLinesFileValue,
    PDFFileValue,
    TextFileValue,
)

# Ensure node types are registered (needed for WorkflowValue schema)
from workflow_engine.nodes import AddNode  # noqa: F401


def _value_type_roundtrip(value_cls: ValueType) -> ValueType:
    """Value type → schema → Value type."""
    return value_cls.to_value_schema().to_value_cls()


# --- Primitive and built-in Value types ---


VALUE_TYPES = [
    BooleanValue,
    FileValue,
    FloatValue,
    IntegerValue,
    JSONValue,
    NullValue,
    StringValue,
    ValueSchemaValue,
    WorkflowValue,
]

FILE_VALUE_TYPES = [
    JSONFileValue,
    JSONLinesFileValue,
    PDFFileValue,
    TextFileValue,
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "value_cls", [pytest.param(t, id=t.__name__) for t in VALUE_TYPES]
)
def test_value_type_roundtrip(value_cls: ValueType):
    """Value type → schema → to_value_cls() returns the same Value type."""
    result = _value_type_roundtrip(value_cls)
    assert result is value_cls, f"Expected {value_cls!r}, got {result!r}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "value_cls", [pytest.param(t, id=t.__name__) for t in FILE_VALUE_TYPES]
)
def test_file_value_type_roundtrip(value_cls: ValueType):
    """FileValue subclass → schema → to_value_cls() returns the same type."""
    result = _value_type_roundtrip(value_cls)
    assert result is value_cls, f"Expected {value_cls!r}, got {result!r}"


# --- Generic Value types (SequenceValue[T], StringMapValue[T]) ---


@pytest.mark.unit
@pytest.mark.parametrize(
    "item_type",
    [
        pytest.param(BooleanValue, id="SequenceValue[BooleanValue]"),
        pytest.param(FloatValue, id="SequenceValue[FloatValue]"),
        pytest.param(IntegerValue, id="SequenceValue[IntegerValue]"),
        pytest.param(NullValue, id="SequenceValue[NullValue]"),
        pytest.param(StringValue, id="SequenceValue[StringValue]"),
    ],
)
def test_sequence_value_type_roundtrip(item_type: ValueType):
    """SequenceValue[T] → schema → to_value_cls() returns the same type."""
    value_cls = SequenceValue[item_type]
    result = _value_type_roundtrip(value_cls)
    assert result == value_cls, f"Expected {value_cls!r}, got {result!r}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "item_type",
    [
        pytest.param(BooleanValue, id="StringMapValue[BooleanValue]"),
        pytest.param(FloatValue, id="StringMapValue[FloatValue]"),
        pytest.param(IntegerValue, id="StringMapValue[IntegerValue]"),
        pytest.param(NullValue, id="StringMapValue[NullValue]"),
        pytest.param(StringValue, id="StringMapValue[StringValue]"),
    ],
)
def test_string_map_value_type_roundtrip(item_type: ValueType):
    """StringMapValue[T] → schema → to_value_cls() returns the same type."""
    value_cls = StringMapValue[item_type]
    result = _value_type_roundtrip(value_cls)
    assert result == value_cls, f"Expected {value_cls!r}, got {result!r}"


@pytest.mark.unit
def test_union_value_type_roundtrip():
    """UnionValue[A, B] → schema → to_value_cls() returns the same type."""
    value_cls = resolve_union_type(UnionValue[FloatValue, SequenceValue[FloatValue]])
    result = _value_type_roundtrip(value_cls)
    assert result == value_cls, f"Expected {value_cls!r}, got {result!r}"


# --- Data types ---


class FooBarData(Data):
    foo: StringValue
    bar: IntegerValue


@pytest.mark.unit
def test_data_type_roundtrip():
    """Data subclass → schema → to_value_cls() returns DataValue with same field structure."""
    schema = get_data_schema(FooBarData)
    result = schema.to_value_cls()
    # DataValueSchema.build_data_cls creates a new Data class; get the inner type
    origin, args = get_origin_and_args(result)
    assert origin is DataValue
    inner_data_cls = args[0]
    assert inner_data_cls.model_fields.keys() == FooBarData.model_fields.keys()


# --- Schema without $defs (round-trip using extra_defs from registry) ---


def _registry_defs():
    """Build defs from registered Value types for resolving $refs without $defs."""
    ValueRegistry.DEFAULT.build()
    return {
        name: value_cls.to_value_schema()
        for name, value_cls in ValueRegistry.DEFAULT.all_value_classes()
    }


def _roundtrip_without_defs(value_cls: ValueType, registry_defs: dict) -> ValueType:
    """Value type → schema (with $defs stripped) → to_value_cls(extra_defs) → Value type."""
    from workflow_engine.core.values import validate_value_schema

    schema = value_cls.model_json_schema()
    schema_without_defs = {k: v for k, v in schema.items() if k != "$defs"}
    parsed = validate_value_schema(schema_without_defs)
    return parsed.to_value_cls(registry_defs)


@pytest.mark.unit
@pytest.mark.parametrize(
    "value_cls", [pytest.param(t, id=t.__name__) for t in VALUE_TYPES]
)
def test_value_type_roundtrip_without_defs(value_cls: ValueType):
    """Value type round-trips without $defs when registry types are passed as extra_defs."""
    result = _roundtrip_without_defs(value_cls, _registry_defs())
    assert result is value_cls, f"Expected {value_cls!r}, got {result!r}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "value_cls", [pytest.param(t, id=t.__name__) for t in FILE_VALUE_TYPES]
)
def test_file_value_type_roundtrip_without_defs(value_cls: ValueType):
    """FileValue subclass round-trips without $defs when registry types are passed as extra_defs."""
    result = _roundtrip_without_defs(value_cls, _registry_defs())
    assert result is value_cls, f"Expected {value_cls!r}, got {result!r}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "item_type",
    [
        pytest.param(BooleanValue, id="SequenceValue[BooleanValue]"),
        pytest.param(FloatValue, id="SequenceValue[FloatValue]"),
        pytest.param(IntegerValue, id="SequenceValue[IntegerValue]"),
        pytest.param(JSONValue, id="SequenceValue[JSONValue]"),
        pytest.param(NullValue, id="SequenceValue[NullValue]"),
        pytest.param(StringValue, id="SequenceValue[StringValue]"),
    ],
)
def test_sequence_value_type_roundtrip_without_defs(item_type: ValueType):
    """SequenceValue[T] round-trips without $defs when registry types are passed as extra_defs."""
    value_cls = SequenceValue[item_type]
    result = _roundtrip_without_defs(value_cls, _registry_defs())
    assert result == value_cls, f"Expected {value_cls!r}, got {result!r}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "item_type",
    [
        pytest.param(BooleanValue, id="StringMapValue[BooleanValue]"),
        pytest.param(FloatValue, id="StringMapValue[FloatValue]"),
        pytest.param(IntegerValue, id="StringMapValue[IntegerValue]"),
        pytest.param(JSONValue, id="StringMapValue[JSONValue]"),
        pytest.param(NullValue, id="StringMapValue[NullValue]"),
        pytest.param(StringValue, id="StringMapValue[StringValue]"),
    ],
)
def test_string_map_value_type_roundtrip_without_defs(item_type: ValueType):
    """StringMapValue[T] round-trips without $defs when registry types are passed as extra_defs."""
    value_cls = StringMapValue[item_type]
    result = _roundtrip_without_defs(value_cls, _registry_defs())
    assert result == value_cls, f"Expected {value_cls!r}, got {result!r}"


@pytest.mark.unit
def test_data_type_roundtrip_without_defs():
    """Data subclass round-trips without $defs when registry types are passed as extra_defs."""
    schema = FooBarData.model_json_schema()
    schema_without_defs = {k: v for k, v in schema.items() if k != "$defs"}
    from workflow_engine.core.values import validate_value_schema

    parsed = validate_value_schema(schema_without_defs)
    result = parsed.to_value_cls(_registry_defs())
    origin, args = get_origin_and_args(result)
    assert origin is DataValue
    inner_data_cls = args[0]
    assert inner_data_cls.model_fields.keys() == FooBarData.model_fields.keys()


@pytest.mark.unit
@pytest.mark.xfail(
    raises=KeyError,
    reason="Deeply nested generics use Pydantic auto-generated def IDs (e.g. SequenceValue_StringMapValue_IntegerValue__) not in the registry meaning that they cannot be resolved without $defs",
)
def test_nested_value_type_roundtrip_without_defs():
    """StringMapValue[SequenceValue[StringMapValue[IntegerValue]]] round-trips without $defs."""
    value_cls = StringMapValue[SequenceValue[StringMapValue[IntegerValue]]]
    result = _roundtrip_without_defs(value_cls, _registry_defs())
    assert result == value_cls, f"Expected {value_cls!r}, got {result!r}"


# --- Constrained subclass round-trips ---


@pytest.mark.unit
def test_constrained_float_roundtrip():
    """FloatValue with numeric constraints round-trips without losing the constraints."""
    from workflow_engine.core.values.schema import (
        _NUMERIC_FIELD_MAP,
        _build_constrained_cls,
    )

    original = _build_constrained_cls(
        FloatValue, _NUMERIC_FIELD_MAP, {"minimum": 0.0, "maximum": 1.0}
    )
    result = _value_type_roundtrip(original)
    assert (
        result.model_fields["root"].metadata == original.model_fields["root"].metadata
    )


@pytest.mark.unit
def test_constrained_integer_roundtrip():
    """IntegerValue with numeric constraints round-trips without losing the constraints."""
    from workflow_engine.core.values.schema import (
        _NUMERIC_FIELD_MAP,
        _build_constrained_cls,
    )

    original = _build_constrained_cls(
        IntegerValue, _NUMERIC_FIELD_MAP, {"minimum": 1, "maximum": 100}
    )
    result = _value_type_roundtrip(original)
    assert (
        result.model_fields["root"].metadata == original.model_fields["root"].metadata
    )


@pytest.mark.unit
def test_constrained_string_roundtrip():
    """StringValue with length constraints round-trips without losing the constraints."""
    from workflow_engine.core.values.schema import (
        _STRING_FIELD_MAP,
        _build_constrained_cls,
    )

    original = _build_constrained_cls(
        StringValue, _STRING_FIELD_MAP, {"minLength": 1, "maxLength": 50}
    )
    result = _value_type_roundtrip(original)
    assert (
        result.model_fields["root"].metadata == original.model_fields["root"].metadata
    )


# --- Nested generic types ---


@pytest.mark.unit
def test_nested_value_type_roundtrip():
    """StringMapValue[SequenceValue[StringMapValue[IntegerValue]]] round-trips."""
    value_cls = StringMapValue[SequenceValue[StringMapValue[IntegerValue]]]
    result = _value_type_roundtrip(value_cls)
    assert result == value_cls, f"Expected {value_cls!r}, got {result!r}"


# --- Generic containers of Result[T] (#215) ---
#
# Result[T] overrides to_value_schema() so it round-trips through its own
# schema variant (ResultValueSchema), which is not the same shape as its raw
# Pydantic model_json_schema() (a discriminated union). SequenceValue and
# StringMapValue must delegate to the item type's own to_value_schema()
# rather than trusting model_json_schema() to describe the item, or nesting
# Result[T] inside them fails to round-trip.


@pytest.mark.unit
def test_sequence_of_result_value_type_roundtrip():
    """SequenceValue[Result[T]] round-trips (the for_each(attempt(w)) shape)."""
    value_cls = SequenceValue[Result[IntegerValue]]
    result = _value_type_roundtrip(value_cls)
    assert result == value_cls, f"Expected {value_cls!r}, got {result!r}"


@pytest.mark.unit
def test_string_map_of_result_value_type_roundtrip():
    """StringMapValue[Result[T]] round-trips."""
    value_cls = StringMapValue[Result[IntegerValue]]
    result = _value_type_roundtrip(value_cls)
    assert result == value_cls, f"Expected {value_cls!r}, got {result!r}"


@pytest.mark.unit
def test_doubly_nested_result_in_sequence_roundtrip():
    """SequenceValue[Result[SequenceValue[T]]] round-trips at two levels of nesting."""
    value_cls = SequenceValue[Result[SequenceValue[IntegerValue]]]
    result = _value_type_roundtrip(value_cls)
    assert result == value_cls, f"Expected {value_cls!r}, got {result!r}"


@pytest.mark.unit
def test_doubly_nested_result_in_string_map_roundtrip():
    """StringMapValue[Result[StringMapValue[T]]] round-trips at two levels of nesting."""
    value_cls = StringMapValue[Result[StringMapValue[IntegerValue]]]
    result = _value_type_roundtrip(value_cls)
    assert result == value_cls, f"Expected {value_cls!r}, got {result!r}"


@pytest.mark.unit
def test_string_map_of_value_roundtrip():
    """
    StringMapValue[Value] (the fully-open map, produced when a schema's
    additionalProperties is bare True) round-trips via additionalProperties:
    True rather than trying to delegate to an unparameterized Value.
    """
    from workflow_engine.core.values.value import Value

    value_cls = StringMapValue[Value]
    result = _value_type_roundtrip(value_cls)
    assert result == value_cls, f"Expected {value_cls!r}, got {result!r}"


# --- Constrained generic containers round-trip without losing constraints ---


@pytest.mark.unit
def test_constrained_sequence_roundtrip():
    """A length-constrained SequenceValue[T] round-trips without losing the constraint."""
    from workflow_engine.core.values.schema import _build_constrained_sequence_cls

    original = _build_constrained_sequence_cls(
        IntegerValue, {"minItems": 1, "maxItems": 5}
    )
    result = _value_type_roundtrip(original)
    assert (
        result.model_fields["root"].metadata == original.model_fields["root"].metadata
    )


@pytest.mark.unit
def test_constrained_string_map_roundtrip():
    """A size-constrained StringMapValue[T] round-trips without losing the constraint."""
    from workflow_engine.core.values.schema import _build_constrained_map_cls

    original = _build_constrained_map_cls(
        StringValue, {"minProperties": 1, "maxProperties": 5}
    )
    result = _value_type_roundtrip(original)
    assert (
        result.model_fields["root"].metadata == original.model_fields["root"].metadata
    )


# --- Raw model_json_schema() -> validate_value_schema() -> to_value_cls() (#220) ---
#
# Before #220, Result[T]'s raw Pydantic schema (model_json_schema()) was a
# discriminated union of internal models, not the published ok/err wire
# shape. A caller that fed that raw schema straight to validate_value_schema()
# and then to_value_cls() -- rather than going through to_value_schema() /
# get_data_schema(), which already special-cased Result -- hit
# NotImplementedError. The __get_pydantic_json_schema__ hook on Result makes
# this path work too.


@pytest.mark.unit
def test_sequence_of_result_raw_schema_roundtrip():
    """
    SequenceValue[Result[T]].model_json_schema() -> validate_value_schema()
    -> to_value_cls() used to raise NotImplementedError; it must now return
    SequenceValue[Result[T]].
    """
    value_cls = SequenceValue[Result[FloatValue]]
    schema = validate_value_schema(value_cls.model_json_schema())
    assert schema.to_value_cls() is value_cls


# --- DataValue[D].to_value_schema().build_value_cls() round-trips (#220) ---
#
# A Data class field is rebuilt from its schema via build_value_cls(), not
# to_value_cls() (build_data_cls() always constructs a fresh Data subclass,
# per its own docstring, so the outer Data/DataValue types are never equal
# to the originals -- only their fields are). This checks that a Result[T]
# field, at various nesting depths, keeps its ok/err identity across that
# round-trip: the *values* inside the rebuilt field, once revalidated, equal
# the originals.


def _result_error(name: str) -> ResultError:
    return ResultError(
        error_class=ErrorClassValue(ErrorClass.SYSTEMIC),
        name=StringValue(name),
        message=StringValue(f"{name} failed"),
        node_id=StringValue("node-1"),
    )


def _data_field_value_roundtrip(value_type: ValueType, value: Value) -> None:
    """
    Build a one-field Data class with a field of *value_type*, wrap it in
    DataValue, and round-trip *value* through
    DataValue[D].to_value_schema().build_value_cls(): dump -> validate on the
    rebuilt class -> compare the field's value (not the outer Data class,
    which build_value_cls() always rebuilds as a distinct, non-equal class)
    to the original.
    """
    D = build_data_type(
        name="RoundTripField",
        fields={
            "value": (
                value_type,
                FieldInfo(title="Value", description="The field under test."),
            )
        },
    )
    original = DataValue[D](D.model_validate({"value": value}))
    rebuilt_cls = DataValue[D].to_value_schema().build_value_cls()
    dumped = original.model_dump(mode="json")
    revalidated = rebuilt_cls.model_validate(dumped)
    original_value = get_data_dict(cast(Data, original.root))["value"]
    revalidated_value = get_data_dict(cast(Data, revalidated.root))["value"]
    assert revalidated_value == original_value, (
        f"Expected {original_value!r}, got {revalidated_value!r}"
    )


@pytest.mark.unit
def test_data_field_result_of_null_roundtrips():
    _data_field_value_roundtrip(
        Result[NullValue], Result[NullValue].ok(NullValue(None))
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(
            Result[Result[IntegerValue]].ok(Result[IntegerValue].ok(IntegerValue(5))),
            id="ok-of-ok",
        ),
        pytest.param(
            Result[Result[IntegerValue]].ok(
                Result[IntegerValue].err(_result_error("Inner"))
            ),
            id="ok-of-err",
        ),
        pytest.param(
            Result[Result[IntegerValue]].err(_result_error("Outer")),
            id="err",
        ),
    ],
)
def test_data_field_nested_result_roundtrips(value: Result):
    _data_field_value_roundtrip(Result[Result[IntegerValue]], value)


@pytest.mark.unit
def test_data_field_string_map_of_result_of_sequence_roundtrips():
    value_type = StringMapValue[Result[SequenceValue[StringValue]]]
    value = value_type(
        {
            "a": Result[SequenceValue[StringValue]].ok(
                SequenceValue[StringValue]([StringValue("x")])
            ),
            "b": Result[SequenceValue[StringValue]].err(_result_error("Fetch")),
        }
    )
    _data_field_value_roundtrip(value_type, value)


class _Inner(Data):
    x: IntegerValue


@pytest.mark.unit
def test_data_field_result_of_data_value_roundtrips():
    """
    Result[DataValue[_Inner]] round-trips. build_data_cls() rebuilds
    DataValue's inner Data class as a fresh, non-equal class (same as the
    outer Data class in the other cases here), so this compares the ok
    arm's own field, one level further in, rather than the DataValue
    wrapper itself.
    """
    value_type = Result[DataValue[_Inner]]
    value = value_type.ok(DataValue[_Inner](_Inner(x=IntegerValue(3))))
    D = build_data_type(
        name="RoundTripField",
        fields={
            "value": (
                value_type,
                FieldInfo(title="Value", description="The field under test."),
            )
        },
    )
    original = DataValue[D](D.model_validate({"value": value}))
    rebuilt_cls = DataValue[D].to_value_schema().build_value_cls()
    dumped = original.model_dump(mode="json")
    revalidated = rebuilt_cls.model_validate(dumped)
    original_result = get_data_dict(cast(Data, original.root))["value"]
    revalidated_result = get_data_dict(cast(Data, revalidated.root))["value"]
    assert isinstance(original_result, Result)
    assert isinstance(revalidated_result, Result)
    assert revalidated_result.is_ok() == original_result.is_ok()
    original_inner = get_data_dict(cast(Data, original_result.unwrap_ok().root))["x"]
    revalidated_inner = get_data_dict(cast(Data, revalidated_result.unwrap_ok().root))[
        "x"
    ]
    assert revalidated_inner == original_inner


@pytest.mark.unit
def test_data_field_sequence_of_result_roundtrips():
    value_type = SequenceValue[Result[IntegerValue]]
    value = value_type(
        [
            Result[IntegerValue].ok(IntegerValue(1)),
            Result[IntegerValue].err(_result_error("Item")),
        ]
    )
    _data_field_value_roundtrip(value_type, value)


@pytest.mark.unit
def test_data_field_result_of_string_map_roundtrips():
    value_type = Result[StringMapValue[IntegerValue]]
    value = value_type.ok(StringMapValue[IntegerValue]({"a": IntegerValue(1)}))
    _data_field_value_roundtrip(value_type, value)
