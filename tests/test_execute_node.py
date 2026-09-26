from typing import ClassVar

import pytest
from overrides import override
from pydantic import Field, ValidationError
from pydantic.fields import FieldInfo

from workflow_engine import (
    BooleanValue,
    Data,
    Empty,
    ExecutionContext,
    FieldSchemaMappingValue,
    FloatValue,
    IntegerValue,
    JSONValue,
    Node,
    NodeTypeInfo,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core.stakeholder import StakeholderLevel
from workflow_engine.nodes import AddNode, DivideNode, SubtractNode


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


@pytest.fixture
def context() -> InMemoryExecutionContext:
    return InMemoryExecutionContext()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_node_infers_fields(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    result = await engine.execute_node(
        context=context,
        node=SubtractNode,
        input={"minuend": 10, "subtrahend": 3},
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"difference": 7}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_node_resolves_dynamic_input_fields(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    result = await engine.execute_node(
        context=context,
        node=AddNode,
        params={"num_arguments": 3},
        input={"a": 1, "b": 2, "c": 3},
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"sum": 6}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_node_surfaces_node_errors(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    result = await engine.execute_node(
        context=context,
        node=DivideNode,
        input={"dividend": 1, "divisor": 0},
    )

    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert "node" in result.errors.node_errors
    messages = result.errors.messages()
    assert any("divide by zero" in message.lower() for message in messages)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_single_node_workflow_accepts_explicit_fields(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    workflow = await engine.build_single_node_workflow(
        SubtractNode,
        input_fields={"minuend": FloatValue, "subtrahend": FloatValue},
        output_fields={"difference": FloatValue},
    )
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={"minuend": 5, "subtrahend": 2},
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"difference": 3}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_node_respects_user_visible_error_level(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    result = await engine.execute_node(
        context=context,
        node=DivideNode,
        input={"dividend": 1, "divisor": 0},
    )

    error = result.errors.node_errors["node"][0]
    assert error is not None
    assert error.level is StakeholderLevel.USER


class _OptionalPortsInput(Data):
    keys: IntegerValue = Field(title="Keys", description="A required input.")
    offset: IntegerValue = Field(
        default=IntegerValue(3), title="Offset", description="Has a default."
    )
    extra: JSONValue = Field(
        default_factory=lambda: JSONValue(None),
        title="Extra",
        description="Has a default factory.",
    )


class _OptionalPortsOutput(Data):
    total: IntegerValue = Field(title="Total", description="keys plus offset.")
    extra_was_null: BooleanValue = Field(
        title="Extra was null", description="Whether extra was left as null."
    )


class _OptionalPortsNode(Node[_OptionalPortsInput, _OptionalPortsOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Optional Ports Probe", version="1.0.0", parameter_type=Empty
    )

    @classmethod
    @override
    def static_input_type(cls) -> type[_OptionalPortsInput]:
        return _OptionalPortsInput

    @classmethod
    @override
    def static_output_type(cls) -> type[_OptionalPortsOutput]:
        return _OptionalPortsOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[_OptionalPortsInput],
        output_type: type[_OptionalPortsOutput],
        input: _OptionalPortsInput,
    ) -> _OptionalPortsOutput:
        return output_type(
            total=IntegerValue(input.keys.root + input.offset.root),
            extra_was_null=BooleanValue(input.extra.root is None),
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_node_applies_defaults_for_omitted_optional_ports(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    result = await engine.execute_node(
        context=context,
        node=_OptionalPortsNode,
        input={"keys": 4},
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS, (
        result.errors.messages()
    )
    assert result.output == {"total": 7, "extra_was_null": True}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_node_supplied_optional_ports_override_defaults(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    result = await engine.execute_node(
        context=context,
        node=_OptionalPortsNode,
        input={"keys": 4, "offset": 10, "extra": {"year": 2020}},
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS, (
        result.errors.messages()
    )
    assert result.output == {"total": 14, "extra_was_null": False}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_node_still_rejects_omitted_required_port(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    with pytest.raises(ValidationError, match="keys"):
        await engine.execute_node(
            context=context,
            node=_OptionalPortsNode,
            input={"offset": 1},
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_single_node_workflow_input_schema_marks_only_required_ports(
    engine: WorkflowEngine,
):
    workflow = await engine.build_single_node_workflow(_OptionalPortsNode)
    validated = await engine.validate(workflow)

    fields = validated.input_type.model_fields
    assert fields["keys"].is_required()
    assert not fields["offset"].is_required()
    assert not fields["extra"].is_required()

    schema = workflow.input_node.params.fields.to_data_schema("InputData")
    assert list(schema.required) == ["keys"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_single_node_workflow_defaults_survive_serialization_roundtrip(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    workflow = await engine.build_single_node_workflow(_OptionalPortsNode)
    restored = Workflow.model_validate(workflow.model_dump(mode="json"))

    result = await engine.execute(context=context, workflow=restored, input={"keys": 1})

    assert result.status is WorkflowExecutionResultStatus.SUCCESS, (
        result.errors.messages()
    )
    assert result.output == {"total": 4, "extra_was_null": True}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_single_node_workflow_accepts_explicit_optional_fields(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    workflow = await engine.build_single_node_workflow(
        _OptionalPortsNode,
        input_fields={
            "keys": IntegerValue,
            "offset": (IntegerValue, FieldInfo(default=IntegerValue(100))),
            "extra": (JSONValue, FieldInfo(default_factory=lambda: JSONValue(None))),
        },
    )
    result = await engine.execute(context=context, workflow=workflow, input={"keys": 1})

    assert result.status is WorkflowExecutionResultStatus.SUCCESS, (
        result.errors.messages()
    )
    assert result.output == {"total": 101, "extra_was_null": True}


@pytest.mark.unit
def test_field_schema_mapping_records_defaults_as_json():
    fields = FieldSchemaMappingValue.from_fields(
        keys=IntegerValue,
        required_pair=(IntegerValue, FieldInfo(description="No default.")),
        offset=(IntegerValue, FieldInfo(default=IntegerValue(3))),
        extra=(JSONValue, FieldInfo(default_factory=lambda: JSONValue({"a": [1]}))),
    )
    schema = fields.to_data_schema("InputData")

    assert list(schema.required) == ["keys", "required_pair"]
    for name in ("keys", "required_pair"):
        assert "default" not in schema.properties[name].model_fields_set
    assert "default" in schema.properties["offset"].model_fields_set
    assert schema.properties["offset"].default == 3
    assert schema.properties["extra"].default == {"a": [1]}


@pytest.mark.unit
def test_field_schema_mapping_keeps_data_dependent_factory_required():
    fields = FieldSchemaMappingValue.from_fields(
        keys=IntegerValue,
        derived=(
            IntegerValue,
            FieldInfo(default_factory=lambda data: IntegerValue(data["keys"].root)),
        ),
    )
    schema = fields.to_data_schema("InputData")

    assert list(schema.required) == ["keys", "derived"]
