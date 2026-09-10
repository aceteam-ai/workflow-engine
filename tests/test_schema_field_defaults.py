"""Record schemas distinguish missing defaults from explicit null values."""

import pytest
from pydantic import ValidationError

from workflow_engine import IntegerValue, NullValue
from workflow_engine.core.values.schema import validate_value_schema

pytestmark = pytest.mark.unit


def test_untitled_record_has_a_usable_generated_name():
    cls = validate_value_schema(
        {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
    ).to_value_cls()
    assert cls.__name__ == "DataValue[Data]"
    assert cls.model_validate({"count": 4}).root.count == IntegerValue(4)


def test_non_required_property_without_default_fails_with_explanation():
    schema = validate_value_schema(
        {"type": "object", "properties": {"count": {"type": "integer"}}}
    )
    with pytest.raises(
        ValueError, match="Non-required properties need an explicit default"
    ):
        schema.to_value_cls()


def test_required_null_property_does_not_acquire_an_implicit_default():
    cls = validate_value_schema(
        {
            "type": "object",
            "properties": {"empty": {"type": "null"}},
            "required": ["empty"],
        }
    ).to_value_cls()
    with pytest.raises(ValidationError):
        cls.model_validate({})
    assert cls.model_validate({"empty": None}).root.empty == NullValue(None)


def test_explicit_null_default_is_preserved():
    cls = validate_value_schema(
        {"type": "object", "properties": {"empty": {"type": "null", "default": None}}}
    ).to_value_cls()
    assert cls.model_validate({}).root.empty == NullValue(None)
