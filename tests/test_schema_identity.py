"""Explicit Value identities survive references; display titles never select types."""

import pytest
from pydantic import Field

from workflow_engine import (
    Data,
    DataValue,
    IntegerValue,
    JSONValue,
    Result,
    SequenceValue,
    StringValue,
)
from workflow_engine.core.values.data import get_data_fields, get_data_schema
from workflow_engine.core.values.result import ErrorClassValue
from workflow_engine.core.values.schema import ValueSchemaValue, validate_value_schema
from workflow_engine.core.values.value import ValueRegistry
from workflow_engine.core.workflow import WorkflowValue
from workflow_engine.nodes.iteration import ForEachParams

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ({"type": "string", "title": "IntegerValue"}, StringValue),
        (
            {"type": "array", "items": {"type": "string"}, "title": "JSONValue"},
            SequenceValue[StringValue],
        ),
    ],
)
def test_colliding_title_rebuilds_declared_structure(document, expected):
    assert validate_value_schema(document).to_value_cls() is expected


def test_colliding_record_title_stays_a_record():
    cls = validate_value_schema(
        {
            "type": "object",
            "title": "StringValue",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
    ).to_value_cls()
    assert issubclass(cls, DataValue)
    assert cls.model_validate({"count": 4}).root.count == IntegerValue(4)


class IdentityRecord(Data):
    payload: JSONValue = Field(title="IntegerValue", description="The JSON payload.")
    classification: ErrorClassValue = Field(
        title="StringValue", description="The error classification."
    )
    result: Result[IntegerValue] = Field(
        title="JSONValue", description="The tagged result."
    )


def test_nested_definitions_identify_values_independently_of_field_titles():
    raw = IdentityRecord.model_json_schema()
    assert raw["$defs"]["JSONValue"]["x-value-type"] == "JSONValue"
    assert raw["$defs"]["ErrorClassValue"]["x-value-type"] == "ErrorClassValue"
    # The wrapper's marker must not leak into the enum it references.
    assert "x-value-type" not in raw["$defs"]["ErrorClass"]
    rebuilt = validate_value_schema(raw).to_value_cls()
    data_cls = rebuilt.model_fields["root"].annotation
    assert data_cls is not None and issubclass(data_cls, Data)
    fields = get_data_fields(data_cls)
    assert fields["payload"][0] is JSONValue
    assert fields["classification"][0] is ErrorClassValue
    assert fields["result"][0] is Result[IntegerValue]


@pytest.mark.parametrize("cls", [WorkflowValue, ValueSchemaValue])
def test_recursive_value_schemas_keep_references_and_round_trip(cls):
    raw = cls.model_json_schema()
    assert raw["x-value-type"] == cls.__name__
    assert "$defs" in raw
    assert validate_value_schema(raw).to_value_cls() is cls


def test_nested_workflow_definition_retains_identity():
    raw = ForEachParams.model_json_schema()
    assert raw["$defs"]["WorkflowValue"]["x-value-type"] == "WorkflowValue"
    rebuilt = get_data_schema(ForEachParams).to_value_cls()
    data_cls = rebuilt.model_fields["root"].annotation
    assert data_cls is not None and issubclass(data_cls, Data)
    assert get_data_fields(data_cls)["workflow"][0] is WorkflowValue


def test_every_registered_value_publishes_its_explicit_identity():
    for name, cls in ValueRegistry.DEFAULT.all_value_classes():
        assert cls.model_json_schema()["x-value-type"] == name
