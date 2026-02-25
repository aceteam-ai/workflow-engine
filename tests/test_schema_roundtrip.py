"""
Tests that every Value/Data type satisfies the round-trip property:
    X.to_value_schema().to_value_cls() == X

This ensures converting a Value type to a schema and back yields the same type.
"""

import pytest

from workflow_engine import (
    BooleanValue,
    Data,
    FileValue,
    FloatValue,
    IntegerValue,
    JSONValue,
    NullValue,
    SequenceValue,
    StringMapValue,
    StringValue,
    ValueSchemaValue,
    ValueType,
    WorkflowValue,
)
from workflow_engine.core.values.data import DataValue
from workflow_engine.core.values.value import ValueRegistry, get_origin_and_args
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


# --- Data types ---


class FooBarData(Data):
    foo: StringValue
    bar: IntegerValue


@pytest.mark.unit
def test_data_type_roundtrip():
    """Data subclass → schema → to_value_cls() returns DataValue with same field structure."""
    schema = FooBarData.to_value_schema()
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
        _build_constrained_cls,
        _NUMERIC_FIELD_MAP,
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
