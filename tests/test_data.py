# tests/test_data.py
from collections.abc import Mapping
from typing import Type

import pytest
from pydantic import ValidationError

from workflow_engine import (
    BooleanValue,
    Data,
    FloatValue,
    IntegerValue,
    StringMapValue,
    StringValue,
    ValueType,
)
from workflow_engine.core.values import (
    DataValue,
    build_data_type,
    get_data_fields,
    has_path,
    resolve_path,
    validate_value_schema,
)
from workflow_engine.core.values.value import get_origin_and_args


@pytest.fixture
def ExampleData() -> Type[Data]:
    """Test data class."""

    class ExampleData(Data):
        name: StringValue
        age: IntegerValue
        active: BooleanValue = None  # type: ignore

    return ExampleData


@pytest.fixture
def example_fields() -> Mapping[str, tuple[ValueType, bool]]:
    """Test fields."""

    return {
        "name": (StringValue, True),
        "age": (IntegerValue, True),
        "active": (BooleanValue, False),
    }


@pytest.mark.unit
def test_get_data_fields(
    ExampleData: Type[Data],
    example_fields: Mapping[str, tuple[ValueType, bool]],
):
    """Test that get_data_fields returns the correct fields."""

    assert get_data_fields(ExampleData) == example_fields


@pytest.mark.unit
def test_build_data_type(
    ExampleData: Type[Data],
    example_fields: Mapping[str, tuple[ValueType, bool]],
):
    """Test that build_data_type returns the correct class."""

    cls = build_data_type("ExampleData", example_fields)

    # build_data_type preserves field ordering (used by InputNode/OutputNode.from_fields)
    assert list(cls.model_fields.keys()) == ["name", "age", "active"]

    # can't exactly just test equality, instead we need to test that both
    # classes behave identically in instantiation, serialization, etc.
    assert cls.__name__ == ExampleData.__name__

    # Test that the class can be instantiated
    example_data = {
        "name": "John",
        "age": 30,
        "active": True,
    }

    # Test that each class behaves the same way when instantiating the data
    for kls in (cls, ExampleData):
        instance = kls(name="John", age=30, active=True)  # type: ignore
        deserialized = kls.model_validate(example_data)

        assert instance == deserialized
        assert instance.model_dump() == deserialized.model_dump() == example_data

        # missing optional field
        kls(name="John", age=30)  # type: ignore

        # missing required field
        with pytest.raises(ValidationError):
            kls(name="John", active=True)  # type: ignore

        # bad type
        with pytest.raises(ValidationError):
            kls(name="John", age="thirty", active=True)  # type: ignore

        # extra field
        with pytest.raises(ValidationError):
            kls(name="John", age=30, active=True, extra=1)  # type: ignore


@pytest.mark.unit
def test_resolve_path_from_json_schema_simple():
    """resolve_path works with types from JSON Schema → ValueSchema → to_value_cls()."""
    json_schema = {
        "title": "FooBarData",
        "type": "object",
        "properties": {
            "foo": {"x-value-type": "StringValue"},
            "bar": {"x-value-type": "IntegerValue"},
        },
        "required": ["foo", "bar"],
    }
    schema = validate_value_schema(json_schema)
    value_cls = schema.to_value_cls()
    assert issubclass(value_cls, DataValue)

    _, (data_type,) = get_origin_and_args(value_cls)

    assert resolve_path(data_type=value_cls, path=["foo"]) == StringValue
    assert resolve_path(data_type=value_cls, path=["bar"]) == IntegerValue
    assert has_path(data_type=data_type, path=["foo"])
    assert has_path(data_type=data_type, path=["bar"])
    assert not has_path(data_type=data_type, path=["baz"])
    with pytest.raises(ValueError, match="does not have field quux"):
        resolve_path(data_type=value_cls, path=["quux"])


@pytest.mark.unit
def test_resolve_path_from_json_schema_nested():
    """resolve_path traverses nested DataValue from schema-derived types."""
    json_schema = {
        "title": "OuterData",
        "type": "object",
        "properties": {
            "inner": {
                "title": "InnerData",
                "type": "object",
                "properties": {
                    "nested": {"x-value-type": "IntegerValue"},
                },
                "required": ["nested"],
            },
        },
        "required": ["inner"],
    }
    schema = validate_value_schema(json_schema)
    value_cls = schema.to_value_cls()

    assert resolve_path(data_type=value_cls, path=["inner", "nested"]) == IntegerValue
    with pytest.raises(ValueError, match="does not have field missing"):
        resolve_path(data_type=value_cls, path=["inner", "missing"])


@pytest.mark.unit
def test_resolve_path_from_json_schema_string_map():
    """resolve_path traverses StringMapValue from schema-derived types."""
    json_schema = {
        "title": "WithMap",
        "type": "object",
        "properties": {
            "metadata": {
                "type": "object",
                "additionalProperties": {"x-value-type": "IntegerValue"},
            },
        },
        "required": ["metadata"],
    }
    schema = validate_value_schema(json_schema)
    value_cls = schema.to_value_cls()

    assert (
        resolve_path(data_type=value_cls, path=["metadata"])
        == StringMapValue[IntegerValue]
    )
    assert (
        resolve_path(data_type=value_cls, path=["metadata", "any_key"]) == IntegerValue
    )


@pytest.mark.unit
def test_has_path_from_json_schema():
    """has_path works on types from JSON Schema → ValueSchema → to_value_cls()."""
    json_schema = {
        "title": "FooBarData",
        "type": "object",
        "properties": {
            "foo": {"x-value-type": "StringValue"},
            "bar": {"x-value-type": "IntegerValue"},
        },
        "required": ["foo", "bar"],
    }
    schema = validate_value_schema(json_schema)
    value_cls = schema.to_value_cls()
    _, (data_type,) = get_origin_and_args(value_cls)

    assert has_path(data_type=data_type, path=["foo"]) is True
    assert has_path(data_type=data_type, path=["bar"]) is True
    assert has_path(data_type=data_type, path=["quux"]) is False


@pytest.mark.unit
@pytest.mark.parametrize("path_factory", (list, tuple))
def test_resolve_path_from_schema_flat(path_factory):
    """Resolve paths on a type from JSON Schema → ValueSchema → to_value_cls()."""
    json_schema = {
        "title": "FlatData",
        "type": "object",
        "properties": {
            "name": {"x-value-type": "StringValue"},
            "count": {"x-value-type": "IntegerValue"},
        },
        "required": ["name"],
    }
    schema = validate_value_schema(json_schema)
    value_cls = schema.to_value_cls()
    assert issubclass(value_cls, DataValue)

    _, (data_type,) = get_origin_and_args(value_cls)

    assert resolve_path(data_type=value_cls, path=path_factory(["name"])) == StringValue
    assert resolve_path(data_type=value_cls, path=path_factory(["count"])) == IntegerValue
    assert has_path(data_type=data_type, path=path_factory(["name"])) is True
    assert has_path(data_type=data_type, path=path_factory(["count"])) is True
    assert has_path(data_type=data_type, path=path_factory(["missing"])) is False

    with pytest.raises(ValueError, match="does not have field missing"):
        resolve_path(data_type=value_cls, path=path_factory(["missing"]))


# ---------------------------------------------------------------------------
# resolve_path and has_path — types from JSON Schema → ValueSchema → to_value_cls
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("path_factory", (list, tuple))
def test_resolve_path_data_from_json_schema(path_factory):
    """resolve_path works on DataValue from JSON Schema with properties."""
    json_schema = {
        "title": "FooBarData",
        "type": "object",
        "properties": {
            "foo": {"x-value-type": "StringValue"},
            "bar": {"x-value-type": "IntegerValue"},
        },
        "required": ["foo", "bar"],
    }
    schema = validate_value_schema(json_schema)
    value_cls = schema.to_value_cls()
    assert issubclass(value_cls, DataValue)

    _, (data_type,) = get_origin_and_args(value_cls)

    # Single-level paths
    assert resolve_path(data_type=data_type, path=path_factory(("foo",))) == StringValue
    assert resolve_path(data_type=data_type, path=path_factory(("bar",))) == IntegerValue

    # Empty path returns DataValue[data_type]
    result = resolve_path(data_type=data_type, path=path_factory(()))
    assert issubclass(result, DataValue)

    # Invalid path raises
    with pytest.raises(ValueError, match="does not have field qux"):
        resolve_path(data_type=data_type, path=path_factory(("qux",)))

    # has_path
    assert has_path(data_type=data_type, path=path_factory(("foo",))) is True
    assert has_path(data_type=data_type, path=path_factory(("bar",))) is True
    assert has_path(data_type=data_type, path=path_factory(("qux",))) is False


@pytest.mark.unit
def test_resolve_path_nested_data_from_json_schema():
    """resolve_path traverses nested DataValue (object with object property)."""
    json_schema = {
        "title": "OuterData",
        "type": "object",
        "properties": {
            "inner": {
                "title": "InnerData",
                "type": "object",
                "properties": {
                    "nested": {"x-value-type": "IntegerValue"},
                },
                "required": ["nested"],
            },
        },
        "required": ["inner"],
    }
    schema = validate_value_schema(json_schema)
    value_cls = schema.to_value_cls()
    _, (data_type,) = get_origin_and_args(value_cls)

    assert resolve_path(data_type=data_type, path=("inner", "nested")) == IntegerValue
    assert has_path(data_type=data_type, path=("inner", "nested")) is True
    assert has_path(data_type=data_type, path=("inner", "missing")) is False


@pytest.mark.unit
def test_resolve_path_string_map_value_from_json_schema():
    """resolve_path traverses StringMapValue (additionalProperties)."""
    json_schema = {
        "title": "WithMap",
        "type": "object",
        "properties": {
            "tags": {
                "type": "object",
                "additionalProperties": {"x-value-type": "StringValue"},
            },
        },
        "required": ["tags"],
    }
    schema = validate_value_schema(json_schema)
    value_cls = schema.to_value_cls()
    _, (data_type,) = get_origin_and_args(value_cls)

    # Path into StringMapValue: any key gives StringValue
    assert resolve_path(data_type=data_type, path=("tags", "any_key")) == StringValue
    assert has_path(data_type=data_type, path=("tags", "any_key")) is True


# ---------------------------------------------------------------------------
# resolve_path and has_path (types from JSON Schema → ValueSchema → to_value_cls)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_path_flat_data_from_schema():
    """resolve_path resolves single-segment paths on DataValue from JSON Schema."""
    json_schema = {
        "title": "FlatData",
        "type": "object",
        "properties": {
            "foo": {"x-value-type": "StringValue"},
            "bar": {"x-value-type": "IntegerValue"},
        },
        "required": ["foo"],
    }
    schema = validate_value_schema(json_schema)
    value_cls = schema.to_value_cls()
    assert issubclass(value_cls, DataValue)

    assert resolve_path(data_type=value_cls, path=["foo"]) == StringValue
    assert resolve_path(data_type=value_cls, path=["bar"]) == IntegerValue


@pytest.mark.unit
def test_resolve_path_through_string_map():
    """resolve_path through StringMapValue: path [metadata, any_key] → FloatValue."""
    json_schema = {
        "title": "DataWithMap",
        "type": "object",
        "properties": {
            "id": {"x-value-type": "StringValue"},
            "metadata": {
                "type": "object",
                "additionalProperties": {"x-value-type": "FloatValue"},
            },
        },
        "required": ["id", "metadata"],
    }
    schema = validate_value_schema(json_schema)
    value_cls = schema.to_value_cls()
    assert resolve_path(data_type=value_cls, path=["metadata", "score"]) == FloatValue


@pytest.mark.unit
def test_resolve_path_nested_data_from_schema():
    """resolve_path resolves multi-segment paths through nested DataValue."""
    json_schema = {
        "title": "OuterData",
        "type": "object",
        "properties": {
            "inner": {
                "title": "InnerData",
                "type": "object",
                "properties": {
                    "nested": {"x-value-type": "IntegerValue"},
                },
                "required": ["nested"],
            },
        },
        "required": ["inner"],
    }
    schema = validate_value_schema(json_schema)
    value_cls = schema.to_value_cls()

    assert issubclass(resolve_path(data_type=value_cls, path=["inner"]), DataValue)
    resolved = resolve_path(data_type=value_cls, path=["inner", "nested"])
    assert resolved == IntegerValue


@pytest.mark.unit
def test_resolve_path_string_map_from_schema():
    """resolve_path traverses StringMapValue; any key yields the value type."""
    json_schema = {
        "type": "object",
        "additionalProperties": {"x-value-type": "FloatValue"},
    }
    schema = validate_value_schema(json_schema)
    _ = schema.to_value_cls()

    # StringMapValue needs a Data root - use a wrapper object
    wrapper_schema = {
        "title": "WrapperData",
        "type": "object",
        "properties": {
            "items": json_schema,
        },
        "required": ["items"],
    }
    wrapper = validate_value_schema(wrapper_schema).to_value_cls()
    resolved = resolve_path(data_type=wrapper, path=["items", "any_key"])

    assert resolved == FloatValue


@pytest.mark.unit
def test_resolve_path_invalid_field_raises():
    """resolve_path raises ValueError for non-existent field."""
    json_schema = {
        "title": "FlatData",
        "type": "object",
        "properties": {
            "foo": {"x-value-type": "StringValue"},
        },
        "required": ["foo"],
    }
    schema = validate_value_schema(json_schema)
    value_cls = schema.to_value_cls()

    with pytest.raises(ValueError, match="does not have field nosuch"):
        resolve_path(data_type=value_cls, path=["nosuch"])


@pytest.mark.unit
def test_has_path_from_schema():
    """has_path returns True/False for types from JSON Schema."""
    json_schema = {
        "title": "FlatData",
        "type": "object",
        "properties": {
            "foo": {"x-value-type": "StringValue"},
            "bar": {"x-value-type": "IntegerValue"},
        },
        "required": ["foo"],
    }
    schema = validate_value_schema(json_schema)
    value_cls = schema.to_value_cls()
    _origin, (data_type,) = get_origin_and_args(value_cls)

    assert has_path(data_type=data_type, path=["foo"]) is True
    assert has_path(data_type=data_type, path=["bar"]) is True
    assert has_path(data_type=data_type, path=["nosuch"]) is False
