# tests/test_union_value.py
"""Tests for UnionValue / OptionalValue."""

from __future__ import annotations

import pytest
from pydantic import Field

from workflow_engine.core import (
    Data,
    Edge,
    FloatValue,
    IntegerValue,
    NullValue,
    SequenceValue,
    StringValue,
    UnionValue,
)
from workflow_engine.core.values import (
    OptionalValue,
    get_data_fields,
    get_field_annotations,
    resolve_path,
)
from workflow_engine.core.values.union import resolve_union_type, union_value_type

OptionalInteger = OptionalValue[IntegerValue]
OptionalString = OptionalValue[StringValue]
NumericInput = UnionValue[FloatValue, SequenceValue[FloatValue]]


class MessageItem(Data):
    sender_id: OptionalInteger
    text: OptionalString = Field(
        title="Text",
        description="The message body.",
    )


@pytest.mark.unit
def test_optional_value_construction_explicit_members():
    item = MessageItem(
        sender_id=NullValue(None),
        text=StringValue("hello"),
    )
    assert isinstance(item.sender_id, NullValue)
    assert item.text.root == "hello"


@pytest.mark.unit
def test_optional_value_construction_with_integer_member():
    item = MessageItem(
        sender_id=IntegerValue(42),
        text=StringValue("hello"),
    )
    assert item.sender_id.root == 42
    assert item.text.root == "hello"


@pytest.mark.unit
def test_optional_value_construction_null_members():
    item = MessageItem(
        sender_id=NullValue(None),
        text=NullValue(None),
    )
    assert isinstance(item.sender_id, NullValue)
    assert isinstance(item.text, NullValue)


@pytest.mark.unit
def test_optional_value_model_validate():
    item = MessageItem.model_validate({"sender_id": 7, "text": None})
    assert isinstance(item.sender_id, IntegerValue)
    assert item.sender_id.root == 7
    assert isinstance(item.text, NullValue)


@pytest.mark.unit
def test_resolve_union_type_optional():
    union_type = resolve_union_type(OptionalInteger)
    assert get_field_annotations(MessageItem)["sender_id"] == union_type
    assert union_type == union_value_type(IntegerValue, NullValue)


@pytest.mark.unit
def test_get_data_fields_resolves_union_value():
    fields = get_data_fields(MessageItem)
    sender_type, _ = fields["sender_id"]
    assert sender_type == union_value_type(IntegerValue, NullValue)


@pytest.mark.unit
def test_union_value_edge_validation():
    class Source(Data):
        out: IntegerValue = Field(title="Out", description="The output.")

    class Target(Data):
        inp: OptionalInteger = Field(title="In", description="The input.")

    edge = Edge(
        source_id="src",
        source_key="out",
        target_id="tgt",
        target_key="inp",
    )
    edge.validate_types(source_type=Source, target_type=Target)


@pytest.mark.unit
def test_union_value_resolve_path():
    union_type = resolve_path(data_type=MessageItem, path=["sender_id"])
    assert union_type == union_value_type(IntegerValue, NullValue)


@pytest.mark.unit
def test_union_value_non_optional():
    class Input(Data):
        values: NumericInput = Field(
            title="Values",
            description="The values.",
        )

    scalar = Input.model_validate({"values": 2.5})
    assert isinstance(scalar.values, FloatValue)
    sequence = Input.model_validate({"values": [1.0, 2.0]})
    assert isinstance(sequence.values, SequenceValue)

    union_type = resolve_union_type(NumericInput)
    assert union_type == union_value_type(FloatValue, SequenceValue[FloatValue])


@pytest.mark.unit
def test_resolve_union_type_from_public_union_value():
    union_type = resolve_union_type(UnionValue[IntegerValue, NullValue])
    assert union_type == union_value_type(IntegerValue, NullValue)


@pytest.mark.unit
def test_invalid_data_field_annotation_raises():
    with pytest.raises(TypeError, match="must be a Value type"):

        class Bad(Data):
            value: int  # type: ignore[assignment]
