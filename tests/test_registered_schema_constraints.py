"""Registry identity, intrinsic constraints and additional constraints compose."""

from collections.abc import Sequence
from typing import Annotated, ClassVar

import pytest
from pydantic import ConfigDict, Field, ValidationError, field_validator

from workflow_engine import (
    BooleanValue,
    Data,
    DataValue,
    FileValue,
    FloatValue,
    IntegerValue,
    JSONValue,
    NullValue,
    SequenceValue,
    StringMapValue,
    StringValue,
    Value,
    ValueSchemaValue,
    WorkflowValue,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core.values.data import get_data_schema
from workflow_engine.core.values.result import ErrorClassValue, Result
from workflow_engine.core.values.schema import validate_value_schema
from workflow_engine.files import (
    JSONFileValue,
    JSONLinesFileValue,
    PDFFileValue,
    TextFileValue,
)

pytestmark = pytest.mark.unit


# Match the explicit built-in/file lists in test_schema_roundtrip, adding Result.
IDENTITY_VALUE_TYPES = [
    BooleanValue,
    FileValue,
    FloatValue,
    IntegerValue,
    JSONValue,
    NullValue,
    StringValue,
    ValueSchemaValue,
    WorkflowValue,
    JSONFileValue,
    JSONLinesFileValue,
    PDFFileValue,
    TextFileValue,
    ErrorClassValue,
    Result[IntegerValue],
]


class ResourceTicketIdValue(Value[str]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra={"x-resource-type": "ticket"}
    )


class TicketLabelValue(Value[str]):
    pass


@ResourceTicketIdValue.register_cast_to(TicketLabelValue)
def ticket_label(value: ResourceTicketIdValue, context) -> TicketLabelValue:
    return TicketLabelValue(f"Ticket {value.root}")


class SchemaPercentValue(Value[Annotated[int, Field(ge=0, le=100)]]):
    pass


class SchemaTicketSequence(SequenceValue[ResourceTicketIdValue]):
    root: Annotated[Sequence[ResourceTicketIdValue], Field(min_length=1)]


class SchemaTicketMap(StringMapValue[ResourceTicketIdValue]):
    pass


class SchemaPrefixedValue(Value[Annotated[str, Field(pattern="^T-")]]):
    pass


class SchemaNormalizedValue(Value[str]):
    @field_validator("root")
    @classmethod
    def normalize(cls, value: str) -> str:
        return value.upper()


def publish_resource_metadata(schema):
    schema["x-resource-type"] = "ticket"


def publish_resource_model_metadata(schema, model):
    schema["x-resource-type"] = "ticket"
    schema["x-model"] = model.__name__


class SchemaCallableMetadataValue(Value[str]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra=publish_resource_metadata
    )


class SchemaCallableModelMetadataValue(Value[str]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra=publish_resource_model_metadata
    )


class TicketRecord(Data):
    ticket: ResourceTicketIdValue = Field(
        title="Ticket", description="The ticket to process."
    )


class TicketEnvelope(Data):
    data: DataValue[TicketRecord] = Field(
        title="Data", description="The ticket record."
    )


def rebuild(cls: type[Value], **constraints) -> type[Value]:
    return validate_value_schema(
        {**cls.to_value_schema().model_dump(mode="json"), **constraints}
    ).to_value_cls()


@pytest.mark.parametrize(
    "cls",
    [ResourceTicketIdValue, SchemaPercentValue, SchemaTicketSequence, SchemaTicketMap],
)
def test_registered_metadata_and_intrinsic_constraints_keep_exact_identity(cls):
    assert rebuild(cls) is cls


async def test_nested_metadata_type_keeps_its_cast():
    cls = get_data_schema(TicketEnvelope).to_value_cls()
    value = cls.model_validate({"data": {"ticket": "T-42"}}).root.data.root.ticket
    assert type(value) is ResourceTicketIdValue
    assert type(value).can_cast_to(TicketLabelValue)
    assert await value.cast_to(
        TicketLabelValue, context=InMemoryExecutionContext()
    ) == TicketLabelValue("Ticket T-42")


async def test_additional_metadata_preserves_registered_casts_and_both_annotations():
    cls = rebuild(ResourceTicketIdValue, **{"x-picker": "recent"})
    assert issubclass(cls, ResourceTicketIdValue)
    assert cls.can_cast_to(TicketLabelValue)
    assert await cls("T-42").cast_to(
        TicketLabelValue, context=InMemoryExecutionContext()
    ) == TicketLabelValue("Ticket T-42")
    wire = cls.to_value_schema().model_dump(mode="json")
    assert wire["x-value-type"] == "ResourceTicketIdValue"
    assert wire["x-resource-type"] == "ticket"
    assert wire["x-picker"] == "recent"


@pytest.mark.parametrize("constraints", [{"maximum": 50}, {"maximum": 200}])
def test_additional_numeric_bounds_preserve_the_registered_type_and_intrinsic_bounds(
    constraints,
):
    cls = rebuild(SchemaPercentValue, **constraints)
    assert issubclass(cls, SchemaPercentValue)
    cls.model_validate(50)
    for value in (-1, 101, 201):
        with pytest.raises(ValidationError):
            cls.model_validate(value)
    if constraints["maximum"] == 50:
        with pytest.raises(ValidationError):
            cls.model_validate(51)
    wire = cls.to_value_schema().model_dump(mode="json")
    assert wire["x-value-type"] == "SchemaPercentValue"
    assert wire["minimum"] == 0
    assert wire["maximum"] == constraints["maximum"]
    again = validate_value_schema(wire).to_value_cls()
    assert issubclass(again, SchemaPercentValue)
    with pytest.raises(ValidationError):
        again.model_validate(101)


def test_constraint_on_compact_identity_uses_the_registered_type():
    cls = validate_value_schema(
        {"x-value-type": "IntegerValue", "minimum": 3}
    ).to_value_cls()
    assert issubclass(cls, IntegerValue)
    assert cls.model_validate(3).root == 3
    with pytest.raises(ValidationError):
        cls.model_validate(2)
    wire = cls.to_value_schema().model_dump(mode="json")
    assert wire["minimum"] == 3
    assert wire["x-value-type"] == "IntegerValue"


@pytest.mark.parametrize(
    "constraints", [{"pattern": "^T-"}, {"enum": ["T-42"]}, {"minLength": 4}]
)
def test_string_constraint_fields_cannot_bypass_registered_identity(constraints):
    cls = rebuild(ResourceTicketIdValue, **constraints)
    assert issubclass(cls, ResourceTicketIdValue)
    assert cls.can_cast_to(TicketLabelValue)
    cls.model_validate("T-42")
    with pytest.raises(ValidationError):
        cls.model_validate("bad")
    again = validate_value_schema(cls.to_value_schema()).to_value_cls()
    with pytest.raises(ValidationError):
        again.model_validate("bad")


def test_both_inherited_and_added_patterns_are_enforced_and_published():
    cls = rebuild(SchemaPrefixedValue, pattern=r"\d$")
    cls.model_validate("T-42")
    for value in ("T-x", "42"):
        with pytest.raises(ValidationError):
            cls.model_validate(value)
    wire = cls.to_value_schema().model_dump(mode="json")
    assert wire["pattern"] == r"\d$"
    assert {"pattern": "^T-"} in wire["allOf"]
    again = validate_value_schema(wire).to_value_cls()
    with pytest.raises(ValidationError):
        again.model_validate("42")


def test_additional_constraints_run_after_inherited_normalization():
    cls = rebuild(SchemaNormalizedValue, pattern="^[A-Z]+$")
    assert cls.model_validate("abc").root == "ABC"


def test_constrained_float_keeps_decimal_conversion_and_json_number_serializer():
    cls = rebuild(FloatValue, minimum=1.5)
    value = cls.model_validate(2.5)
    assert value.model_dump(mode="json") == 2.5
    with pytest.raises(ValidationError):
        cls.model_validate(True)


@pytest.mark.parametrize(
    ("cls", "good", "bad", "constraint"),
    [
        (SchemaTicketSequence, ["T-1"], ["T-1", "T-2"], {"maxItems": 1}),
        (SchemaTicketMap, {"a": "T-1"}, {"a": "T-1", "b": "T-2"}, {"maxProperties": 1}),
    ],
)
def test_registered_containers_keep_identity_and_enforce_size(
    cls, good, bad, constraint
):
    rebuilt = rebuild(cls, **constraint)
    assert issubclass(rebuilt, cls)
    rebuilt.model_validate(good)
    with pytest.raises(ValidationError):
        rebuilt.model_validate(bad)
    if cls is SchemaTicketSequence:
        with pytest.raises(ValidationError):
            rebuilt.model_validate([])


def test_identity_aliases_fold_and_conflicting_aliases_fail():
    schema = validate_value_schema(
        {
            "type": "integer",
            "x-value-type": "IntegerValue",
            "value_type": "IntegerValue",
        }
    )
    assert "x-value-type" not in (schema.model_extra or {})
    assert schema.to_value_cls() is IntegerValue
    with pytest.raises(ValueError):
        validate_value_schema(
            {"x-value-type": "IntegerValue", "value_type": "StringValue"}
        )


@pytest.mark.parametrize(
    "cls",
    [
        *IDENTITY_VALUE_TYPES,
        SequenceValue[IntegerValue],
        StringMapValue[IntegerValue],
    ],
)
def test_wire_identity_never_leaks_into_extra_fields(cls):
    assert "x-value-type" not in (cls.to_value_schema().model_extra or {})


def test_legacy_recursive_schema_names_the_cycle_and_restamping_restores_identity():
    raw = {
        "$ref": "#/$defs/LegacyJSONValue",
        "$defs": {
            "LegacyJSONValue": {"title": "JSONValue", "$ref": "#/$defs/JSON"},
            "JSON": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"$ref": "#/$defs/JSON"}},
                ]
            },
        },
    }
    with pytest.raises(
        ValueError, match="Cyclic schema reference to definition 'JSON'"
    ):
        validate_value_schema(raw).to_value_cls()
    raw["$defs"]["LegacyJSONValue"]["x-value-type"] = "JSONValue"
    assert validate_value_schema(raw).to_value_cls() is JSONValue


def test_cycle_guard_resets_after_a_failure_and_does_not_reject_shared_siblings():
    with pytest.raises(ValueError, match="definition 'loop'"):
        validate_value_schema(
            {"$ref": "#/$defs/loop", "$defs": {"loop": {"$ref": "#/$defs/loop"}}}
        ).to_value_cls()
    schema = validate_value_schema(
        {
            "type": "object",
            "properties": {
                "a": {"$ref": "#/$defs/leaf"},
                "b": {"$ref": "#/$defs/leaf"},
            },
            "required": ["a", "b"],
            "$defs": {"leaf": {"type": "integer"}},
        }
    )
    assert schema.to_value_cls().model_validate({"a": 1, "b": 2}).model_dump(
        mode="json"
    ) == {"a": 1, "b": 2}


def test_successive_pattern_constraints_survive_rebuilding():
    from workflow_engine.core.values.schema import (
        _STRING_FIELD_MAP,
        _build_constrained_cls,
    )

    first = _build_constrained_cls(StringValue, _STRING_FIELD_MAP, {"pattern": "^a"})
    second = _build_constrained_cls(first, _STRING_FIELD_MAP, {"pattern": "b$"})
    for cls in (second, second.to_value_schema().to_value_cls()):
        assert cls.model_validate("ab").root == "ab"
        for value in ("ax", "xb"):
            with pytest.raises(ValidationError):
                cls.model_validate(value)


def test_successive_multiple_of_constraints_survive_rebuilding():
    from workflow_engine.core.values.schema import (
        _NUMERIC_FIELD_MAP,
        _build_constrained_cls,
    )

    first = _build_constrained_cls(IntegerValue, _NUMERIC_FIELD_MAP, {"multipleOf": 2})
    second = _build_constrained_cls(first, _NUMERIC_FIELD_MAP, {"multipleOf": 3})
    for cls in (second, second.to_value_schema().to_value_cls()):
        cls.model_validate(6)
        for value in (2, 3, 4):
            with pytest.raises(ValidationError):
                cls.model_validate(value)


def test_decimal_multiple_of_is_exact_and_survives_rebuilding():
    from decimal import Decimal

    constrained = rebuild(FloatValue, multipleOf=0.1)
    for cls in (constrained, constrained.to_value_schema().to_value_cls()):
        assert cls.model_validate(Decimal("0.3")).root == Decimal("0.3")
        with pytest.raises(ValidationError):
            cls.model_validate(Decimal("0.31"))


def test_unsupported_conjunction_is_rejected_instead_of_weakened():
    with pytest.raises(ValueError, match="Unsupported allOf constraint keywords: not"):
        rebuild(StringValue, allOf=[{"not": {"const": "forbidden"}}])


def test_plain_string_identity_does_not_drop_declared_pattern():
    cls = validate_value_schema(
        {"x-value-type": "StringValue", "type": "string", "pattern": "^a"}
    ).to_value_cls()
    assert issubclass(cls, StringValue)
    cls.model_validate("abc")
    with pytest.raises(ValidationError):
        cls.model_validate("bad")


@pytest.mark.parametrize("cls", IDENTITY_VALUE_TYPES)
def test_value_definitions_are_stamped_through_two_record_levels(cls):
    from workflow_engine.core.values.data import build_data_type

    inner = build_data_type(
        name="StampedInner",
        fields={"value": (cls, Field(title="Value", description="The nested value."))},
    )
    outer = build_data_type(
        name="StampedOuter",
        fields={
            "inner": (
                DataValue[inner],
                Field(title="Inner", description="The nested record."),
            )
        },
    )
    wire = get_data_schema(outer).model_dump(mode="json")
    definition = next(
        value
        for value in wire["$defs"].values()
        if value.get("x-value-type") == cls.__name__
    )
    if cls is Result[IntegerValue]:
        for value in definition["err"]["properties"].values():
            assert value["x-value-type"] in {"StringValue", "ErrorClassValue"}


def test_non_record_titles_are_not_needed_for_nested_registered_identity():
    from workflow_engine.core.values.data import build_data_type, get_field_annotations

    types = {
        "json_data": JSONValue,
        "file": FileValue,
        "classification": ErrorClassValue,
    }
    record = build_data_type(
        name="UntitledLeaves",
        fields={
            name: (cls, Field(title="Value", description="The nested value."))
            for name, cls in types.items()
        },
    )
    raw = get_data_schema(record).model_dump(mode="json")

    def remove_titles(node):
        if isinstance(node, list):
            for value in node:
                remove_titles(value)
        elif isinstance(node, dict):
            if "properties" not in node:
                node.pop("title", None)
            for value in node.values():
                remove_titles(value)

    remove_titles(raw)
    rebuilt = validate_value_schema(raw).to_value_cls()
    rebuilt_record = rebuilt.model_fields["root"].annotation
    assert rebuilt_record is not None and issubclass(rebuilt_record, Data)
    assert get_field_annotations(rebuilt_record) == types


@pytest.mark.parametrize(
    "cls", [SchemaCallableMetadataValue, SchemaCallableModelMetadataValue]
)
def test_constrained_subclass_preserves_callable_schema_metadata(cls):
    assert rebuild(cls) is cls
    constrained = rebuild(cls, maxLength=4)
    wire = constrained.to_value_schema().model_dump(mode="json")
    assert wire["x-resource-type"] == "ticket"
    assert wire["x-value-type"] == cls.__name__
    if cls is SchemaCallableModelMetadataValue:
        assert wire["x-model"] == cls.__name__
    for target in (constrained, validate_value_schema(wire).to_value_cls()):
        target.model_validate("T-42")
        with pytest.raises(ValidationError):
            target.model_validate("T-1234")


def test_explicit_identity_overrides_display_and_structural_type():
    cls = rebuild(ResourceTicketIdValue, type="integer", title="NotATicket")
    assert cls is ResourceTicketIdValue
    assert cls.model_validate("T-42").root == "T-42"
