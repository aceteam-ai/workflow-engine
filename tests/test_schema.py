# tests/test_schema.py

"""
Tests two sets of functionalities:
1. that .to_value_cls() is the inverse of .to_value_schema()
2. that we can manually write JSON Schemas that will turn into the correct Value
   classes when .to_value_cls() is called on them.
3. that the aliasing system (e.g. { "title": "StringValue" } -> StringValue)
   works for all non-generic Value classes.

In this file we follow the convention of using T for the expected type and U for
the type returned by .to_value_cls().
"""

import pytest

from workflow_engine import (
    BooleanValue,
    Data,
    Empty,
    FileValue,
    FloatValue,
    IntegerValue,
    NullValue,
    SequenceValue,
    StringMapValue,
    StringValue,
    ValueSchemaValue,
    WorkflowValue,
)
from workflow_engine.core.values import validate_value_schema
from workflow_engine.core.values.schema import (
    BooleanValueSchema,
    BaseValueSchema,
    FloatValueSchema,
    IntegerValueSchema,
    NullValueSchema,
    ReferenceValueSchema,
    SequenceValueSchema,
    StringMapValueSchema,
    StringValueSchema,
)
from workflow_engine.files import (
    JSONFileValue,
    JSONLinesFileValue,
    PDFFileValue,
    TextFileValue,
)

# ensure that these node types are registered for the workflow tests
from workflow_engine.nodes import (
    AddNode,  # noqa: F401
    ConstantIntegerNode,  # noqa: F401
)


@pytest.mark.unit
def test_boolean_schema_roundtrip():
    T = BooleanValue
    schema = T.to_value_schema()
    assert isinstance(schema, BooleanValueSchema)
    U = schema.to_value_cls()
    assert U == T
    assert U.to_value_schema() == schema

    t1 = T(True)
    u1 = U.model_validate(t1)
    assert u1 == t1

    u2 = U(False)
    t2 = T.model_validate(u2)
    assert t2 == u2


@pytest.mark.unit
def test_boolean_schema_manual():
    T = BooleanValue
    json_schema = {
        "type": "boolean",
    }
    schema = validate_value_schema(json_schema)
    assert isinstance(schema, BooleanValueSchema)
    U = schema.to_value_cls()
    assert U == T

    t1 = T(False)
    u1 = U.model_validate(t1)
    assert u1 == t1

    u2 = U(True)
    t2 = T.model_validate(u2)
    assert t2 == u2


@pytest.mark.unit
def test_boolean_schema_aliasing():
    T = BooleanValue
    json_schema = {
        "title": "BooleanValue",
    }
    schema = validate_value_schema(json_schema)
    assert isinstance(schema, BaseValueSchema)
    U = schema.to_value_cls()
    assert U == BooleanValue

    t1 = T(True)
    u1 = U.model_validate(t1)
    assert u1 == t1

    u2 = U(False)
    t2 = T.model_validate(u2)
    assert t2 == u2


@pytest.mark.unit
def test_integer_schema_roundtrip():
    T = IntegerValue
    schema = T.to_value_schema()
    assert isinstance(schema, IntegerValueSchema)
    U = schema.to_value_cls()
    assert U == T
    assert U.to_value_schema() == schema

    t1 = T(42)
    u1 = U.model_validate(t1)
    assert u1 == t1

    u2 = U(2520)
    t2 = T.model_validate(u2)
    assert t2 == u2


@pytest.mark.unit
def test_integer_schema_manual():
    T = IntegerValue
    json_schema = {
        "type": "integer",
    }
    schema = validate_value_schema(json_schema)
    assert isinstance(schema, IntegerValueSchema)
    U = schema.to_value_cls()
    assert U == T

    t1 = T(-42)
    u1 = U.model_validate(t1)
    assert u1 == t1

    u2 = U(2520)
    t2 = T.model_validate(u2)
    assert t2 == u2


@pytest.mark.unit
def test_integer_schema_aliasing():
    T = IntegerValue
    json_schema = {
        "title": "IntegerValue",
    }
    schema = validate_value_schema(json_schema)
    assert isinstance(schema, BaseValueSchema)
    U = schema.to_value_cls()
    assert U == T

    t1 = T(2048)
    u1 = U.model_validate(t1)
    assert u1 == t1

    u2 = U(2520)
    t2 = T.model_validate(u2)
    assert t2 == u2


@pytest.mark.unit
def test_float_schema_roundtrip():
    T = FloatValue
    schema = T.to_value_schema()
    assert isinstance(schema, FloatValueSchema)
    U = schema.to_value_cls()
    assert U == T
    assert U.to_value_schema() == schema

    t1 = T(3.14159)
    u1 = U.model_validate(t1)
    assert u1 == t1

    u2 = U(2.71828)
    t2 = T.model_validate(u2)
    assert t2 == u2


@pytest.mark.unit
def test_float_schema_manual():
    T = FloatValue
    json_schema = {
        "type": "number",
    }
    schema = validate_value_schema(json_schema)
    assert isinstance(schema, FloatValueSchema)
    U = schema.to_value_cls()
    assert U == T

    t1 = T(2.71828)
    u1 = U.model_validate(t1)
    assert u1 == t1

    u2 = U(3.14159)
    t2 = T.model_validate(u2)
    assert t2 == u2


@pytest.mark.unit
def test_float_schema_aliasing():
    T = FloatValue
    json_schema = {
        "title": "FloatValue",
    }
    schema = validate_value_schema(json_schema)
    assert isinstance(schema, BaseValueSchema)
    U = schema.to_value_cls()
    assert U == FloatValue

    t1 = T(1.41421)
    u1 = U.model_validate(t1)
    assert u1 == t1

    u2 = U(3.14159)
    t2 = T.model_validate(u2)
    assert t2 == u2


@pytest.mark.unit
def test_null_schema_roundtrip():
    T = NullValue
    schema = T.to_value_schema()
    assert isinstance(schema, NullValueSchema)
    U = schema.to_value_cls()
    assert U == T
    assert U.to_value_schema() == schema

    t1 = T(None)
    u1 = U.model_validate(t1)
    assert u1 == t1

    u2 = U(None)
    t2 = T.model_validate(u2)
    assert t2 == u2


@pytest.mark.unit
def test_null_schema_manual():
    T = NullValue
    json_schema = {
        "type": "null",
    }
    schema = validate_value_schema(json_schema)
    assert isinstance(schema, NullValueSchema)
    U = schema.to_value_cls()
    assert U == T

    t1 = T(None)
    u1 = U.model_validate(t1)
    assert u1 == t1

    u2 = U(None)
    t2 = T.model_validate(u2)
    assert t2 == u2


@pytest.mark.unit
def test_null_schema_aliasing():
    T = NullValue
    json_schema = {
        "title": "NullValue",
    }
    schema = validate_value_schema(json_schema)
    assert isinstance(schema, BaseValueSchema)
    U = schema.to_value_cls()
    assert U == T

    t1 = T(None)
    u1 = U.model_validate(t1)
    assert u1 == t1

    u2 = U(None)
    t2 = T.model_validate(u2)
    assert t2 == u2


@pytest.mark.unit
def test_string_schema_roundtrip():
    T = StringValue
    schema = T.to_value_schema()
    assert isinstance(schema, StringValueSchema)
    U = schema.to_value_cls()
    assert U == StringValue
    assert U.to_value_schema() == schema

    t1 = T("hello wengine")
    u1 = U.model_validate(t1)
    assert u1 == t1

    u2 = U("hi wengine")
    t2 = T.model_validate(u2)
    assert t2 == u2


@pytest.mark.unit
def test_string_schema_manual():
    T = StringValue
    json_schema = {
        "type": "string",
    }
    schema = validate_value_schema(json_schema)
    assert isinstance(schema, StringValueSchema)
    U = schema.to_value_cls()
    assert U == T

    t1 = T("salutations wengine")
    u1 = U.model_validate(t1)
    assert u1 == t1

    u2 = U("good morning wengine")
    t2 = T.model_validate(u2)
    assert t2 == u2


@pytest.mark.unit
def test_string_schema_aliasing():
    T = StringValue
    json_schema = {
        "title": "StringValue",
    }
    schema = validate_value_schema(json_schema)
    assert isinstance(schema, BaseValueSchema)
    U = schema.to_value_cls()
    assert U == StringValue

    t1 = T("hey wengine")
    u1 = U.model_validate(t1)
    assert u1 == t1

    u2 = U("good afternoon wengine")
    t2 = T.model_validate(u2)
    assert t2 == u2


@pytest.mark.unit
def test_sequence_schema_roundtrip():
    for ItemType, ItemSchema in (
        (BooleanValue, BooleanValueSchema),
        (FloatValue, FloatValueSchema),
        (IntegerValue, IntegerValueSchema),
        (NullValue, NullValueSchema),
        (StringValue, StringValueSchema),
    ):
        T = SequenceValue[ItemType]
        schema = T.to_value_schema()
        assert isinstance(schema, SequenceValueSchema)
        assert isinstance(schema.items, ReferenceValueSchema)
        assert isinstance(schema.defs[schema.items.id], ItemSchema)
        U = schema.to_value_cls()
        assert U == T
        assert U.to_value_schema() == schema


@pytest.mark.unit
def test_sequence_schema_manual():
    for type, ItemType, ItemSchema in (
        ("boolean", BooleanValue, BooleanValueSchema),
        ("number", FloatValue, FloatValueSchema),
        ("integer", IntegerValue, IntegerValueSchema),
        ("null", NullValue, NullValueSchema),
        ("string", StringValue, StringValueSchema),
    ):
        T = SequenceValue[ItemType]
        json_schema = {
            "type": "array",
            "items": {"type": type},
        }
        schema = validate_value_schema(json_schema)
        assert isinstance(schema, SequenceValueSchema)
        assert isinstance(schema.items, ItemSchema)
        U = schema.to_value_cls()
        assert U == T


@pytest.mark.unit
def test_sequence_schema_aliasing():
    for ItemType in (
        BooleanValue,
        FloatValue,
        IntegerValue,
        NullValue,
        StringValue,
    ):
        T = SequenceValue[ItemType]
        json_schema = {
            "type": "array",
            "items": {"title": ItemType.__name__},
        }
        schema = validate_value_schema(json_schema)
        assert isinstance(schema, SequenceValueSchema)
        assert isinstance(schema.items, BaseValueSchema)
        U = schema.to_value_cls()
        assert U == T


@pytest.mark.unit
def test_string_map_schema_roundtrip():
    for ItemType, ItemSchema in (
        (BooleanValue, BooleanValueSchema),
        (FloatValue, FloatValueSchema),
        (IntegerValue, IntegerValueSchema),
        (NullValue, NullValueSchema),
        (StringValue, StringValueSchema),
    ):
        T = StringMapValue[ItemType]
        schema = T.to_value_schema()
        assert isinstance(schema, StringMapValueSchema)
        assert isinstance(schema.additionalProperties, ReferenceValueSchema)
        assert isinstance(schema.defs[schema.additionalProperties.id], ItemSchema)
        U = schema.to_value_cls()
        assert U == T
        assert U.to_value_schema() == schema


@pytest.mark.unit
def test_string_map_schema_manual():
    for type, ItemType, ItemSchema in (
        ("boolean", BooleanValue, BooleanValueSchema),
        ("number", FloatValue, FloatValueSchema),
        ("integer", IntegerValue, IntegerValueSchema),
        ("null", NullValue, NullValueSchema),
        ("string", StringValue, StringValueSchema),
    ):
        T = StringMapValue[ItemType]
        json_schema = {
            "type": "object",
            "additionalProperties": {"type": type},
        }
        schema = validate_value_schema(json_schema)
        assert isinstance(schema, StringMapValueSchema)
        assert isinstance(schema.additionalProperties, ItemSchema)
        U = schema.to_value_cls()
        assert U == T


@pytest.mark.unit
def test_string_map_schema_aliasing():
    for ItemType in (
        BooleanValue,
        FloatValue,
        IntegerValue,
        NullValue,
        StringValue,
    ):
        T = StringMapValue[ItemType]
        json_schema = {
            "type": "object",
            "additionalProperties": {"title": ItemType.__name__},
        }
        schema = validate_value_schema(json_schema)
        assert isinstance(schema, StringMapValueSchema)
        assert isinstance(schema.additionalProperties, BaseValueSchema)
        U = schema.to_value_cls()
        assert U == T


@pytest.mark.unit
def test_super_recursive_schema_roundtrip():
    for T in (
        StringMapValue[SequenceValue[StringMapValue[StringValue]]],
        SequenceValue[StringMapValue[SequenceValue[NullValue]]],
        StringMapValue[StringMapValue[StringMapValue[IntegerValue]]],
        SequenceValue[SequenceValue[SequenceValue[BooleanValue]]],
    ):
        schema = T.to_value_schema()
        U = schema.to_value_cls()
        assert U == T
        assert U.to_value_schema() == schema


@pytest.mark.unit
def test_super_recursive_schema_manual():
    T = StringMapValue[SequenceValue[StringMapValue[StringValue]]]
    json_schema = {
        "type": "object",
        "additionalProperties": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": {"title": "StringValue"},
            },
        },
    }
    schema = validate_value_schema(json_schema)
    assert isinstance(schema, StringMapValueSchema)
    assert isinstance(schema.additionalProperties, SequenceValueSchema)
    assert isinstance(schema.additionalProperties.items, StringMapValueSchema)
    assert isinstance(
        schema.additionalProperties.items.additionalProperties, BaseValueSchema
    )
    U = schema.to_value_cls()
    assert U == T


@pytest.mark.unit
def test_empty_schema_roundtrip():
    T = Empty
    schema = T.to_value_schema()
    U = schema.to_value_cls()

    # for Empty, to_value_cls returns a new class not equal to the original
    # but it can serialize and deserialize instances of the original class
    t1 = T()
    u1 = U.model_validate(t1.model_dump())
    # equality check fails because they are technically different classes,
    # but they have the exact same fields
    assert u1.root.__dict__ == t1.__dict__

    u2 = U.model_validate({})
    t2 = T.model_validate(u2.model_dump())
    # equality check fails because they are technically different classes,
    # but they have the exact same fields
    assert t2.__dict__ == u2.root.__dict__


# defined outside of test_data_schema_roundtrip to get a proper class name
class FooBarData(Data):
    foo: StringValue
    bar: IntegerValue


@pytest.mark.unit
def test_data_schema_roundtrip():
    T = FooBarData
    schema = T.to_value_schema()
    U = schema.to_value_cls()

    # it can serialize and deserialize instances of the original class
    t1 = T(
        foo=StringValue("foo"),
        bar=IntegerValue(1),
    )
    u1 = U.model_validate(t1.model_dump())
    # equality check fails because they are technically different classes,
    # but they have the exact same fields
    assert u1.root.foo == t1.foo
    assert u1.root.bar == t1.bar
    assert u1.root.__dict__ == t1.__dict__

    u2 = U.model_validate(
        {
            "foo": "bar",
            "bar": 2,
        }
    )
    t2 = T.model_validate(u2.model_dump())
    # equality check fails because they are technically different classes,
    # but they have the exact same fields
    assert t2.foo == u2.root.foo
    assert t2.bar == u2.root.bar


@pytest.mark.unit
def test_data_schema_manual():
    T = FooBarData
    json_schema = {
        "title": "FooBarData",
        "type": "object",
        "properties": {
            "foo": {"type": "string"},
            "bar": {"type": "integer"},
        },
        "required": ["foo", "bar"],
    }
    schema = validate_value_schema(json_schema)
    U = schema.to_value_cls()

    # it can serialize and deserialize instances of the original class
    t1 = T(
        foo=StringValue("bar"),
        bar=IntegerValue(12),
    )
    u1 = U.model_validate(t1.model_dump())
    # equality check fails because they are technically different classes,
    # but they have the exact same fields
    assert u1.root.foo == t1.foo
    assert u1.root.bar == t1.bar

    u2 = U.model_validate(
        {
            "foo": "baz",
            "bar": 24,
        }
    )
    t2 = T.model_validate(u2.model_dump())
    # equality check fails because they are technically different classes,
    # but they have the exact same fields
    assert u2.root.foo == t2.foo
    assert u2.root.bar == t2.bar


@pytest.mark.unit
def test_data_schema_aliasing():
    T = FooBarData
    json_schema = {
        "title": "FooBarData",
        "type": "object",
        "properties": {
            "foo": {"title": "StringValue"},
            "bar": {"title": "IntegerValue"},
        },
        "required": ["foo", "bar"],
    }
    schema = validate_value_schema(json_schema)
    U = schema.to_value_cls()

    # it can serialize and deserialize instances of the original class
    t1 = T(
        foo=StringValue("foobar"),
        bar=IntegerValue(123),
    )
    u1 = U.model_validate(t1.model_dump())
    # equality check fails because they are technically different classes,
    # but they have the exact same fields
    assert u1.root.foo == t1.foo
    assert u1.root.bar == t1.bar

    u2 = U.model_validate(
        {
            "foo": "baz",
            "bar": 24,
        }
    )
    t2 = T.model_validate(u2.model_dump())
    assert u2.root.foo == t2.foo
    assert u2.root.bar == t2.bar


@pytest.mark.unit
def test_file_schema_roundtrip():
    for T in (
        FileValue,
        JSONFileValue,
        JSONLinesFileValue,
        PDFFileValue,
        TextFileValue,
    ):
        schema = T.to_value_schema()
        U = schema.to_value_cls()
        assert U == T
        assert U.to_value_schema() == schema

        t1 = T.from_path("foo", foo="bar", bar="baz")
        u1 = U.model_validate(t1.model_dump())
        assert u1 == t1

        u2 = U.model_validate({"path": "bar", "metadata": {"baz": 3}})
        t2 = T.model_validate(u2.model_dump())
        assert t2.path == u2.root.path
        assert t2.metadata == u2.root.metadata


@pytest.mark.unit
def test_file_schema_aliasing():
    for T in (
        FileValue,
        JSONFileValue,
        JSONLinesFileValue,
        PDFFileValue,
        TextFileValue,
    ):
        json_schema = {"title": T.__name__}
        schema = validate_value_schema(json_schema)
        U = schema.to_value_cls()
        assert U == T

        t1 = T.from_path("bar", bar="baz", baz="foo")
        u1 = U.model_validate(t1.model_dump())
        assert u1.root.path == t1.path
        assert u1.root.metadata == t1.metadata

        u2 = U.model_validate({"path": "bar", "metadata": {"baz": 3}})
        t2 = T.model_validate(u2.model_dump())
        assert t2.path == u2.root.path
        assert t2.metadata == u2.root.metadata


@pytest.mark.unit
def test_workflow_schema_roundtrip():
    T = WorkflowValue
    schema = T.to_value_schema()
    U = schema.to_value_cls()
    assert U == T
    assert U.to_value_schema() == schema

    with open("examples/addition.json", "r") as f:
        workflow_json = f.read().strip()
    t1 = T.model_validate_json(workflow_json)
    u1 = U.model_validate_json(workflow_json)
    assert u1 == t1


@pytest.mark.unit
def test_workflow_schema_aliasing():
    T = WorkflowValue
    json_schema = {"title": T.__name__}
    schema = validate_value_schema(json_schema)
    U = schema.to_value_cls()
    assert U == T

    with open("examples/addition.json", "r") as f:
        workflow_json = f.read().strip()

    t1 = T.model_validate_json(workflow_json)
    u1 = U.model_validate_json(workflow_json)
    assert u1 == t1


@pytest.mark.unit
def test_value_schema_value_roundtrip():
    T = ValueSchemaValue
    schema = T.to_value_schema()
    U = schema.to_value_cls()
    assert U == T
    assert U.to_value_schema() == schema

    t1 = T(StringValueSchema(type="string", title="StringValue"))
    u1 = U.model_validate(t1.model_dump())
    assert u1 == t1

    u2 = U(IntegerValueSchema(type="integer", title="IntegerValue"))
    t2 = T.model_validate(u2.model_dump())
    assert t2 == u2


@pytest.mark.unit
def test_value_schema_value_manual():
    T = ValueSchemaValue
    json_schema = {"title": T.__name__}
    schema = validate_value_schema(json_schema)
    U = schema.to_value_cls()
    assert U == T

    t1 = T(BooleanValueSchema(type="boolean", title="BooleanValue"))
    u1 = U.model_validate(t1.model_dump())
    assert u1 == t1

    u2 = U(NullValueSchema(type="null", title="NullValue"))
    t2 = T.model_validate(u2.model_dump())
    assert t2 == u2


@pytest.mark.unit
def test_value_schema_value_aliasing():
    T = ValueSchemaValue
    json_schema = {"title": T.__name__}
    schema = validate_value_schema(json_schema)
    U = schema.to_value_cls()
    assert U == T

    t1 = T(FloatValueSchema(type="number", title="FloatValue"))
    u1 = U.model_validate(t1.model_dump())
    assert u1 == t1

    u2 = U(
        SequenceValueSchema(
            type="array",
            items=StringValueSchema(type="string", title="StringValue"),
        )
    )
    t2 = T.model_validate(u2.model_dump())
    assert t2 == u2
