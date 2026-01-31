from typing import Literal

import pytest

from workflow_engine import Context
from workflow_engine.contexts.in_memory import InMemoryContext
from workflow_engine.core import (
    FloatValue,
    IntegerValue,
    SequenceValue,
    StringMapValue,
    StringValue,
    Value,
)
from workflow_engine.core.values.value import (
    get_origin_and_args,
    get_value_type_key,
)


@pytest.fixture
def context():
    """Create a test context for value casting operations."""
    return InMemoryContext()


@pytest.mark.unit
def test_basic_value_creation():
    """Test basic Value creation and properties."""
    # Test StringValue
    str_val = StringValue("hello")
    assert str_val == "hello"
    assert isinstance(str_val, StringValue)
    assert isinstance(str_val, Value)

    # Test IntegerValue
    int_val = IntegerValue(42)
    assert int_val == 42
    assert isinstance(int_val, IntegerValue)
    assert isinstance(int_val, Value)

    # Test FloatValue
    float_val = FloatValue(3.14)
    assert float_val == 3.14
    assert isinstance(float_val, FloatValue)
    assert isinstance(float_val, Value)


@pytest.mark.unit
def test_value_equality_and_hash():
    """Test Value equality and hash functionality."""
    val1 = StringValue("test")
    val2 = StringValue("test")
    val3 = StringValue("different")

    assert val1 == val2
    assert val1 != val3
    assert hash(val1) == hash(val2)
    assert hash(val1) != hash(val3)

    # Test with different types
    int_val = IntegerValue(42)
    assert val1 != int_val
    assert hash(val1) != hash(int_val)


@pytest.mark.unit
def test_value_serialization():
    """Test Value serialization and deserialization."""
    # Test StringValue
    str_val = StringValue("hello world")
    json_str = str_val.model_dump_json()
    deserialized = StringValue.model_validate_json(json_str)
    assert deserialized == str_val

    # Test IntegerValue
    int_val = IntegerValue(123)
    json_str = int_val.model_dump_json()
    deserialized = IntegerValue.model_validate_json(json_str)
    assert deserialized == int_val

    # Test FloatValue
    float_val = FloatValue(3.14159)
    json_str = float_val.model_dump_json()
    deserialized = FloatValue.model_validate_json(json_str)
    assert deserialized == float_val


@pytest.mark.unit
async def test_basic_casting(context):
    """Test basic casting between Value types."""
    # Integer to String
    int_val = IntegerValue(42)
    str_val = await int_val.cast_to(StringValue, context=context)
    assert isinstance(str_val, StringValue)
    assert str_val == "42"

    # String to Integer
    str_val = StringValue("123")
    int_val = await str_val.cast_to(IntegerValue, context=context)
    assert isinstance(int_val, IntegerValue)
    assert int_val == 123

    # String to Float
    str_val = StringValue("3.14")
    float_val = await str_val.cast_to(FloatValue, context=context)
    assert isinstance(float_val, FloatValue)
    assert float_val == 3.14

    # Integer to Float
    int_val = IntegerValue(42)
    float_val = await int_val.cast_to(FloatValue, context=context)
    assert isinstance(float_val, FloatValue)
    assert float_val == 42.0


@pytest.mark.unit
async def test_sequence_value(context):
    """Test SequenceValue functionality."""
    # Create sequence of integers
    int_sequence = SequenceValue[IntegerValue](
        [IntegerValue(1), IntegerValue(2), IntegerValue(3)]
    )
    assert len(int_sequence) == 3
    assert all(isinstance(x, IntegerValue) for x in int_sequence)

    # Cast to sequence of strings
    str_sequence = await int_sequence.cast_to(
        SequenceValue[StringValue], context=context
    )
    assert isinstance(str_sequence, SequenceValue)
    assert len(str_sequence) == 3
    assert all(isinstance(x, StringValue) for x in str_sequence)
    assert str_sequence == ["1", "2", "3"]

    # Cast back to sequence of integers
    int_sequence_again = await str_sequence.cast_to(
        SequenceValue[IntegerValue], context=context
    )
    assert int_sequence_again == int_sequence


@pytest.mark.unit
async def test_string_map_value(context):
    """Test StringMapValue functionality."""
    # Create map of integers
    int_map = StringMapValue[IntegerValue](
        {
            "a": IntegerValue(1),
            "b": IntegerValue(2),
            "c": IntegerValue(3),
        }
    )
    assert len(int_map) == 3
    assert all(isinstance(v, IntegerValue) for v in int_map.values())

    # Cast to map of strings
    str_map = await int_map.cast_to(StringMapValue[StringValue], context=context)
    assert isinstance(str_map, StringMapValue)
    assert len(str_map) == 3
    assert all(isinstance(v, StringValue) for v in str_map.values())
    assert str_map == {
        "a": "1",
        "b": "2",
        "c": "3",
    }

    # Cast back to map of integers
    int_map_again = await str_map.cast_to(StringMapValue[IntegerValue], context=context)
    assert int_map_again == int_map


@pytest.mark.unit
async def test_cast_from_class_method(context):
    """Test the cast_from class method."""
    # Test StringValue.cast_from
    int_val = IntegerValue(42)
    str_val = await StringValue.cast_from(int_val, context=context)
    assert isinstance(str_val, StringValue)
    assert str_val == "42"

    # Test IntegerValue.cast_from
    str_val = StringValue("123")
    int_val = await IntegerValue.cast_from(str_val, context=context)
    assert isinstance(int_val, IntegerValue)
    assert int_val == 123

    # Test FloatValue.cast_from
    int_val = IntegerValue(42)
    float_val = await FloatValue.cast_from(int_val, context=context)
    assert isinstance(float_val, FloatValue)
    assert float_val == 42.0


@pytest.mark.unit
async def test_cast_cache(context):
    """Test that casting results are cached."""
    int_val = IntegerValue(42)

    # First cast should compute the result
    str_val1 = await int_val.cast_to(StringValue, context=context)
    assert str_val1 == "42"

    # Second cast should use cache
    str_val2 = await int_val.cast_to(StringValue, context=context)
    assert str_val2 is str_val1  # Should be the same object from cache


@pytest.mark.unit
async def test_invalid_casting(context):
    """Test that invalid casting raises appropriate errors."""
    int_val = IntegerValue(42)

    # Test casting to a type that doesn't have a registered caster
    class CustomValue(Value[str]):
        pass

    # This should fail because there's no registered caster from IntegerValue to CustomValue
    with pytest.raises(ValueError, match="Cannot convert"):
        await int_val.cast_to(CustomValue, context=context)


@pytest.mark.unit
async def test_self_casting(context):
    """Test that casting to the same type returns the same object."""
    int_val = IntegerValue(42)
    same_val = await int_val.cast_to(IntegerValue, context=context)
    assert same_val is int_val


@pytest.mark.unit
def test_get_origin_and_args():
    """Test the get_origin_and_args utility function."""
    # Test with Value types
    origin, args = get_origin_and_args(StringValue)
    assert origin == StringValue
    assert args == ()

    origin, args = get_origin_and_args(SequenceValue[IntegerValue])
    assert origin == SequenceValue
    assert args == (IntegerValue,)


@pytest.mark.unit
def test_get_value_type_key():
    """Test the get_value_type_key utility function."""
    # Test with simple types - IntegerValue is a RootModel[int], so it gets the int type
    result = get_value_type_key(IntegerValue)
    assert result == ("IntegerValue", ())

    # Test with generic types
    result = get_value_type_key(SequenceValue[IntegerValue])
    assert result == ("SequenceValue", (("IntegerValue", ()),))


@pytest.mark.unit
def test_value_frozen_behavior():
    """Test that Value objects are frozen (immutable)."""
    int_val = IntegerValue(42)

    # Should not be able to modify the root attribute
    with pytest.raises(Exception):  # Pydantic will raise an error
        int_val.root = 100


@pytest.mark.unit
async def test_complex_nested_casting(context):
    """Test complex nested casting scenarios."""
    # Create a complex nested structure
    nested_sequence = SequenceValue[SequenceValue[IntegerValue]](
        [
            SequenceValue[IntegerValue]([IntegerValue(1), IntegerValue(2)]),
            SequenceValue[IntegerValue]([IntegerValue(3), IntegerValue(4)]),
        ]
    )

    # Cast to nested sequence of strings
    str_nested = await nested_sequence.cast_to(
        SequenceValue[SequenceValue[StringValue]], context=context
    )
    assert str_nested == SequenceValue[SequenceValue[StringValue]](
        [
            SequenceValue[StringValue]([StringValue("1"), StringValue("2")]),
            SequenceValue[StringValue]([StringValue("3"), StringValue("4")]),
        ]
    )


@pytest.mark.unit
async def test_cast_registration(context):
    """Test that cast functions can be registered and work correctly."""

    class QuestionValue(Value[str]):
        pass

    class AnswerValue(Value[Literal[42]]):
        pass

    @QuestionValue.register_cast_to(AnswerValue)
    def cast_question_to_answer(value: QuestionValue, context: Context) -> AnswerValue:
        return AnswerValue(42)

    # Try to register the same cast again (before any casting operations)
    with pytest.raises(
        AssertionError,
        match="Type caster from QuestionValue to AnswerValue already registered",
    ):

        @QuestionValue.register_cast_to(AnswerValue)
        def cast_question_to_answer(
            value: QuestionValue, context: Context
        ) -> AnswerValue:
            return AnswerValue(42)

    # Now test that casting works
    assert QuestionValue.can_cast_to(AnswerValue)
    question = QuestionValue("the universe")
    answer = await question.cast_to(AnswerValue, context=context)
    assert answer == 42

    # Try to register the same cast again (after casting operations)
    with pytest.raises(
        RuntimeError,
        match="Cannot add casters for QuestionValue after it has been used to cast values",
    ):

        @QuestionValue.register_cast_to(AnswerValue)
        def cast_question_to_answer(
            value: QuestionValue, context: Context
        ) -> AnswerValue:
            return AnswerValue(42)


@pytest.mark.unit
def test_json_parsing_edge_cases():
    """Test JSON parsing edge cases."""
    # Test with invalid JSON
    with pytest.raises(Exception):  # Pydantic will raise a validation error
        IntegerValue.model_validate_json("invalid json")

    # Test with empty string
    with pytest.raises(Exception):
        StringValue.model_validate_json("")

    # Test with null
    with pytest.raises(Exception):
        IntegerValue.model_validate_json("null")


@pytest.mark.unit
def test_sequence_value_json():
    """Test SequenceValue JSON serialization and deserialization."""
    # Test with simple sequence
    sequence_json = "[1, 2, 3]"
    sequence_val = SequenceValue[IntegerValue].model_validate_json(sequence_json)
    assert len(sequence_val) == 3
    assert sequence_val == [1, 2, 3]

    # Test serialization back to JSON
    json_str = sequence_val.model_dump_json()
    assert json_str == "[1,2,3]"


@pytest.mark.unit
def test_string_map_value_json():
    """Test StringMapValue JSON serialization and deserialization."""
    # Test with simple map
    map_json = '{"a": 1, "b": 2, "c": 3}'
    map_val = StringMapValue[IntegerValue].model_validate_json(map_json)
    assert len(map_val) == 3
    assert map_val == {"a": 1, "b": 2, "c": 3}

    # Test serialization back to JSON
    json_str = map_val.model_dump_json()
    assert json_str == '{"a":1,"b":2,"c":3}'


@pytest.mark.unit
async def test_async_caster_registration_and_usage(context):
    """Test async caster registration, coroutine detection, and error handling."""
    import asyncio

    class TestValue(Value[str]):
        pass

    # Register an async caster
    @TestValue.register_cast_to(StringValue)
    async def cast_test_to_string(value: TestValue, context: Context) -> StringValue:
        await asyncio.sleep(0.001)  # Simulate async work
        return StringValue(f"converted: {value.root}")

    test_value = TestValue("hello world")

    # The caster function should be a coroutine function
    cast_fn = TestValue.get_caster(StringValue)
    assert cast_fn is not None
    result = cast_fn(test_value, context)
    assert hasattr(result, "__await__")
    assert asyncio.iscoroutine(result)
    await result  # silence the warning of an unawaited awaitable

    # Test async casting
    result = await test_value.cast_to(StringValue, context=context)
    assert isinstance(result, StringValue)
    assert result == "converted: hello world"

    # Test classmethod usage
    result_from = await StringValue.cast_from(test_value, context=context)
    assert isinstance(result_from, StringValue)
    assert result_from == "converted: hello world"


@pytest.mark.unit
async def test_json_value_to_null(context):
    """Test JSONValue to NullValue casting."""
    from workflow_engine.core import JSONValue, NullValue

    # Successful cast
    json_null = JSONValue(None)
    null_val = await json_null.cast_to(NullValue, context=context)
    assert isinstance(null_val, NullValue)
    assert null_val.root is None

    # Failed casts
    with pytest.raises(ValueError, match="Expected null"):
        await JSONValue(42).cast_to(NullValue, context=context)

    with pytest.raises(ValueError, match="Expected null"):
        await JSONValue("null").cast_to(NullValue, context=context)


@pytest.mark.unit
async def test_json_value_to_boolean(context):
    """Test JSONValue to BooleanValue casting."""
    from workflow_engine.core import BooleanValue, JSONValue

    # Successful casts
    json_true = JSONValue(True)
    bool_val = await json_true.cast_to(BooleanValue, context=context)
    assert isinstance(bool_val, BooleanValue)
    assert bool_val.root is True

    json_false = JSONValue(False)
    bool_val = await json_false.cast_to(BooleanValue, context=context)
    assert isinstance(bool_val, BooleanValue)
    assert bool_val.root is False

    # Failed casts - integers should not cast to boolean
    with pytest.raises(ValueError, match="Expected bool"):
        await JSONValue(1).cast_to(BooleanValue, context=context)

    with pytest.raises(ValueError, match="Expected bool"):
        await JSONValue(0).cast_to(BooleanValue, context=context)

    with pytest.raises(ValueError, match="Expected bool"):
        await JSONValue("true").cast_to(BooleanValue, context=context)


@pytest.mark.unit
async def test_json_value_to_integer(context):
    """Test JSONValue to IntegerValue casting."""
    from workflow_engine.core import IntegerValue, JSONValue

    # Successful casts
    json_int = JSONValue(42)
    int_val = await json_int.cast_to(IntegerValue, context=context)
    assert isinstance(int_val, IntegerValue)
    assert int_val.root == 42

    json_negative = JSONValue(-10)
    int_val = await json_negative.cast_to(IntegerValue, context=context)
    assert isinstance(int_val, IntegerValue)
    assert int_val.root == -10

    # Failed casts - bool should not cast to int (even though bool is subclass of int)
    with pytest.raises(ValueError, match="Expected int"):
        await JSONValue(True).cast_to(IntegerValue, context=context)

    with pytest.raises(ValueError, match="Expected int"):
        await JSONValue(False).cast_to(IntegerValue, context=context)

    # Float should not cast to int
    with pytest.raises(ValueError, match="Expected int"):
        await JSONValue(3.14).cast_to(IntegerValue, context=context)

    # String should not cast to int
    with pytest.raises(ValueError, match="Expected int"):
        await JSONValue("42").cast_to(IntegerValue, context=context)


@pytest.mark.unit
async def test_json_value_to_float(context):
    """Test JSONValue to FloatValue casting."""
    from workflow_engine.core import FloatValue, JSONValue

    # Successful cast from float
    json_float = JSONValue(3.14)
    float_val = await json_float.cast_to(FloatValue, context=context)
    assert isinstance(float_val, FloatValue)
    assert float_val.root == 3.14

    # Successful cast from int (int should be accepted for float)
    json_int = JSONValue(42)
    float_val = await json_int.cast_to(FloatValue, context=context)
    assert isinstance(float_val, FloatValue)
    assert float_val.root == 42.0

    # Failed casts - bool should not cast to float
    with pytest.raises(ValueError, match="Expected float or int"):
        await JSONValue(True).cast_to(FloatValue, context=context)

    with pytest.raises(ValueError, match="Expected float or int"):
        await JSONValue(False).cast_to(FloatValue, context=context)

    # String should not cast to float
    with pytest.raises(ValueError, match="Expected float or int"):
        await JSONValue("3.14").cast_to(FloatValue, context=context)


@pytest.mark.unit
async def test_json_value_to_sequence(context):
    """Test JSONValue to SequenceValue casting."""
    from workflow_engine.core import JSONValue, SequenceValue

    # Successful cast with JSONValue elements
    json_list = JSONValue([1, 2, 3])
    seq_val = await json_list.cast_to(SequenceValue[JSONValue], context=context)
    assert isinstance(seq_val, SequenceValue)
    assert len(seq_val) == 3
    assert [x.root for x in seq_val] == [1, 2, 3]

    # Successful cast with nested conversion to IntegerValue
    int_seq = await json_list.cast_to(SequenceValue[IntegerValue], context=context)
    assert isinstance(int_seq, SequenceValue)
    assert len(int_seq) == 3
    assert all(isinstance(x, IntegerValue) for x in int_seq)
    assert [x.root for x in int_seq] == [1, 2, 3]

    # Successful cast with empty list
    json_empty = JSONValue([])
    empty_seq = await json_empty.cast_to(SequenceValue[JSONValue], context=context)
    assert isinstance(empty_seq, SequenceValue)
    assert len(empty_seq) == 0

    # Failed casts - string should not be treated as sequence
    with pytest.raises(ValueError, match="Expected sequence.*strings are not valid"):
        await JSONValue("hello").cast_to(SequenceValue[JSONValue], context=context)

    # Failed casts - mapping should not cast to sequence
    with pytest.raises(ValueError, match="Expected sequence"):
        await JSONValue({"a": 1}).cast_to(SequenceValue[JSONValue], context=context)

    # Failed casts - primitive should not cast to sequence
    with pytest.raises(ValueError, match="Expected sequence"):
        await JSONValue(42).cast_to(SequenceValue[JSONValue], context=context)


@pytest.mark.unit
async def test_json_value_to_string_map(context):
    """Test JSONValue to StringMapValue casting."""
    from workflow_engine.core import JSONValue, StringMapValue

    # Successful cast with JSONValue values
    json_dict = JSONValue({"a": 1, "b": 2, "c": 3})
    map_val = await json_dict.cast_to(StringMapValue[JSONValue], context=context)
    assert isinstance(map_val, StringMapValue)
    assert len(map_val) == 3
    assert {k: v.root for k, v in map_val.items()} == {"a": 1, "b": 2, "c": 3}

    # Successful cast with nested conversion to IntegerValue
    int_map = await json_dict.cast_to(StringMapValue[IntegerValue], context=context)
    assert isinstance(int_map, StringMapValue)
    assert len(int_map) == 3
    assert all(isinstance(v, IntegerValue) for v in int_map.values())
    assert {k: v.root for k, v in int_map.items()} == {"a": 1, "b": 2, "c": 3}

    # Successful cast with empty dict
    json_empty = JSONValue({})
    empty_map = await json_empty.cast_to(StringMapValue[JSONValue], context=context)
    assert isinstance(empty_map, StringMapValue)
    assert len(empty_map) == 0

    # Failed casts - list should not cast to mapping
    with pytest.raises(ValueError, match="Expected mapping"):
        await JSONValue([1, 2, 3]).cast_to(StringMapValue[JSONValue], context=context)

    # Failed casts - primitive should not cast to mapping
    with pytest.raises(ValueError, match="Expected mapping"):
        await JSONValue(42).cast_to(StringMapValue[JSONValue], context=context)


@pytest.mark.unit
async def test_json_value_complex_nested_casting(context):
    """Test complex nested JSON structures."""
    from workflow_engine.core import FloatValue, JSONValue, SequenceValue, StringMapValue

    # Nested sequence of maps
    json_data = JSONValue([{"x": 1.5, "y": 2.0}, {"x": 3.0, "y": 4.5}])
    result = await json_data.cast_to(
        SequenceValue[StringMapValue[FloatValue]], context=context
    )
    assert isinstance(result, SequenceValue)
    assert len(result) == 2
    assert all(isinstance(item, StringMapValue) for item in result)
    first_map = result[0]
    assert {k: v.root for k, v in first_map.items()} == {"x": 1.5, "y": 2.0}

    # Map of sequences
    json_data2 = JSONValue({"a": [1, 2, 3], "b": [4, 5, 6]})
    result2 = await json_data2.cast_to(
        StringMapValue[SequenceValue[IntegerValue]], context=context
    )
    assert isinstance(result2, StringMapValue)
    assert len(result2) == 2
    assert all(isinstance(v, SequenceValue) for v in result2.values())
    a_seq = result2["a"]
    assert [x.root for x in a_seq] == [1, 2, 3]


@pytest.mark.unit
async def test_json_value_type_checking_strictness(context):
    """Test that type checking is strict and doesn't allow invalid conversions."""
    from workflow_engine.core import BooleanValue, IntegerValue, JSONValue

    # Python's bool is a subclass of int, but we should reject bool -> int
    json_bool = JSONValue(True)
    with pytest.raises(ValueError, match="Expected int"):
        await json_bool.cast_to(IntegerValue, context=context)

    # And reject int -> bool
    json_int = JSONValue(1)
    with pytest.raises(ValueError, match="Expected bool"):
        await json_int.cast_to(BooleanValue, context=context)


if __name__ == "__main__":
    pytest.main([__file__])
