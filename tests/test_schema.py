# tests/test_schema.py

"""
Tests two sets of functionalities:
1. that .to_value_cls() is the inverse of .to_value_schema()
2. that we can manually write JSON Schemas that will turn into the correct Value
   classes when .to_value_cls() is called on them.
3. that the aliasing system (e.g. { "x-value-type": "StringValue" } -> StringValue)
   works for all non-generic Value classes. { "title": "..." } also works as a
   backwards-compatible fallback.

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
from workflow_engine.core.values import get_data_schema, validate_value_schema
from workflow_engine.core.values.schema import (
    BooleanValueSchema,
    BaseValueSchema,
    FieldSchemaMappingValue,
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
        "x-value-type": "BooleanValue",
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
def test_integer_schema_with_extra_fields():
    """Constrained integer schemas produce subclasses that enforce the constraints."""
    json_schema = {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
    }
    schema = validate_value_schema(json_schema)
    assert isinstance(schema, IntegerValueSchema)
    U = schema.to_value_cls()

    assert issubclass(U, IntegerValue)
    assert U.model_json_schema()["minimum"] == 0
    assert U.model_json_schema()["maximum"] == 100

    assert U.model_validate(50).root == 50
    with pytest.raises(Exception):
        U.model_validate(-1)
    with pytest.raises(Exception):
        U.model_validate(101)


@pytest.mark.unit
def test_float_schema_with_extra_fields():
    """Constrained float schemas produce subclasses that enforce the constraints."""
    json_schema = {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
    }
    schema = validate_value_schema(json_schema)
    assert isinstance(schema, FloatValueSchema)
    U = schema.to_value_cls()

    assert issubclass(U, FloatValue)
    assert U.model_json_schema()["minimum"] == 0.0
    assert U.model_json_schema()["maximum"] == 1.0

    assert U.model_validate(0.5).root == 0.5
    with pytest.raises(Exception):
        U.model_validate(-0.1)
    with pytest.raises(Exception):
        U.model_validate(1.1)


@pytest.mark.unit
def test_string_schema_with_extra_fields():
    """Constrained string schemas produce subclasses that enforce the constraints."""
    json_schema = {
        "type": "string",
        "minLength": 2,
        "maxLength": 5,
    }
    schema = validate_value_schema(json_schema)
    assert isinstance(schema, StringValueSchema)
    U = schema.to_value_cls()

    assert issubclass(U, StringValue)
    assert U.model_json_schema()["minLength"] == 2
    assert U.model_json_schema()["maxLength"] == 5

    assert U.model_validate("hi").root == "hi"
    with pytest.raises(Exception):
        U.model_validate("x")
    with pytest.raises(Exception):
        U.model_validate("toolong")


@pytest.mark.unit
def test_unknown_extras_preserved_in_constrained_cls():
    """Extras not in the field map are carried through as json_schema_extra."""
    json_schema = {
        "type": "number",
        "minimum": 0.0,
        "x-foo": "bar",
    }
    schema = validate_value_schema(json_schema)
    U = schema.to_value_cls()

    assert issubclass(U, FloatValue)
    assert U.model_json_schema()["minimum"] == 0.0
    assert U.model_json_schema()["x-foo"] == "bar"
    with pytest.raises(Exception):
        U.model_validate(-0.1)


@pytest.mark.unit
def test_schema_from_pydantic_model_with_field_constraints():
    """Schemas extracted from Pydantic Field constraints produce enforcing subclasses."""
    from pydantic import BaseModel, Field

    class ModelWithConstraints(BaseModel):
        count: int = Field(ge=0, le=100)
        score: float = Field(ge=0.0, le=1.0)

    pydantic_schema = ModelWithConstraints.model_json_schema()
    count_schema = pydantic_schema["properties"]["count"]
    score_schema = pydantic_schema["properties"]["score"]

    count_cls = validate_value_schema(count_schema).to_value_cls()
    assert issubclass(count_cls, IntegerValue)
    count_cls.model_validate(50)
    with pytest.raises(Exception):
        count_cls.model_validate(101)

    score_cls = validate_value_schema(score_schema).to_value_cls()
    assert issubclass(score_cls, FloatValue)
    score_cls.model_validate(0.5)
    with pytest.raises(Exception):
        score_cls.model_validate(1.5)


@pytest.mark.unit
def test_integer_schema_aliasing():
    T = IntegerValue
    json_schema = {
        "x-value-type": "IntegerValue",
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
        "x-value-type": "FloatValue",
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
        "x-value-type": "NullValue",
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
        "x-value-type": "StringValue",
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
            "items": {"x-value-type": ItemType.__name__},
        }
        schema = validate_value_schema(json_schema)
        assert isinstance(schema, SequenceValueSchema)
        assert isinstance(schema.items, BaseValueSchema)
        U = schema.to_value_cls()
        assert U == T


@pytest.mark.unit
def test_constrained_sequence_is_castable():
    """Constrained SequenceValue subclasses must remain compatible with the casting system."""
    import asyncio
    from workflow_engine.contexts import InMemoryContext

    json_schema = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 2,
        "maxItems": 4,
    }
    U = validate_value_schema(json_schema).to_value_cls()

    assert issubclass(U, SequenceValue)

    # Constraint is enforced at validation time
    assert U.model_validate([1.0, 2.0]) == [1.0, 2.0]
    with pytest.raises(Exception):
        U.model_validate([1.0])  # too short

    # A plain SequenceValue[FloatValue] can be cast to the constrained subclass
    source = SequenceValue[FloatValue].model_validate([1.0, 2.0, 3.0])
    context = InMemoryContext()
    result = asyncio.run(source.cast_to(U, context=context))
    assert isinstance(result, U)

    # A constrained instance can be cast to a plain SequenceValue[FloatValue]
    constrained = U.model_validate([1.0, 2.0])
    result2 = asyncio.run(
        constrained.cast_to(SequenceValue[FloatValue], context=context)
    )
    assert isinstance(result2, SequenceValue)


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
            "additionalProperties": {"x-value-type": ItemType.__name__},
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
                "additionalProperties": {"x-value-type": "StringValue"},
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
    schema = get_data_schema(T)
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
    schema = get_data_schema(T)
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
            "foo": {"x-value-type": "StringValue"},
            "bar": {"x-value-type": "IntegerValue"},
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
def test_data_schema_with_multiple_constrained_float_properties():
    """Two FloatValue properties with different constraints must not collide in $defs."""
    json_schema = {
        "title": "BoundedData",
        "type": "object",
        "properties": {
            "probability": {"type": "number", "minimum": 0, "maximum": 1},
            "temperature": {"type": "number", "minimum": -273.15},
        },
        "required": ["probability", "temperature"],
    }
    schema = validate_value_schema(json_schema)
    U = schema.to_value_cls()

    # Both fields enforce their own distinct constraints
    instance = U.model_validate({"probability": 0.5, "temperature": 20.0})
    assert instance.root.probability.root == 0.5
    assert instance.root.temperature.root == 20.0

    with pytest.raises(Exception):
        U.model_validate({"probability": 1.5, "temperature": 20.0})

    with pytest.raises(Exception):
        U.model_validate({"probability": 0.5, "temperature": -300.0})

    # Schema must have two distinct $defs entries, not one clobbering the other
    full_schema = U.model_json_schema()
    defs = full_schema.get("$defs", {})
    float_defs = [v for v in defs.values() if v.get("type") == "number"]
    assert len(float_defs) == 2


@pytest.mark.unit
def test_field_schema_mapping_preserves_extras():
    """FieldSchemaMappingValue preserves schema extras (both known and unknown) on property schemas."""
    from workflow_engine.core.values.schema import ValueSchemaValue

    score_schema = FloatValueSchema(
        **{"type": "number", "minimum": 0.0, "maximum": 1.0, "x-foo": "bar"}
    )
    mapping = FieldSchemaMappingValue({"score": ValueSchemaValue(score_schema)})
    data_schema = mapping.to_data_schema("TestData")
    U = data_schema.to_value_cls()

    # Known constraints are enforced
    assert U.model_validate({"score": 0.5}).root.score.root == 0.5
    with pytest.raises(Exception):
        U.model_validate({"score": 1.5})
    with pytest.raises(Exception):
        U.model_validate({"score": -0.1})

    # Unknown extras survive into the property's schema
    score_field_schema = U.model_json_schema()["$defs"]
    score_def = next(
        v for v in score_field_schema.values() if v.get("type") == "number"
    )
    assert score_def["x-foo"] == "bar"


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
        json_schema = {"x-value-type": T.__name__}
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
    json_schema = {"x-value-type": T.__name__}
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
    json_schema = {"x-value-type": T.__name__}
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
    json_schema = {"x-value-type": T.__name__}
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
