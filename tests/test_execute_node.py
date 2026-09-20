from typing import ClassVar, cast

import pytest
from pydantic import ConfigDict, Field, ValidationError

from workflow_engine import (
    FloatValue,
    JSONValue,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core import Data, Empty, Node, NodeTypeInfo
from workflow_engine.core.stakeholder import StakeholderLevel
from workflow_engine.nodes import AddNode, DivideNode, SubtractNode

factory_calls = 0


def make_optional_value(validated_data: dict[str, JSONValue]) -> JSONValue:
    global factory_calls
    factory_calls += 1
    return JSONValue({"key": validated_data["required"].root, "call": factory_calls})


class DefaultsInput(Data):
    required: JSONValue
    literal: JSONValue = Field(default=JSONValue("literal"))
    generated: JSONValue = Field(default_factory=make_optional_value)


class DefaultsOutput(Data):
    result: JSONValue


class DefaultsNode(Node[DefaultsInput, DefaultsOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Defaults",
        description="Exercise inferred input defaults.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @classmethod
    def static_input_type(cls) -> type[DefaultsInput]:
        return DefaultsInput

    @classmethod
    def static_output_type(cls) -> type[DefaultsOutput]:
        return DefaultsOutput

    async def run(self, *, context, input_type, output_type, input):
        return output_type(result=JSONValue(input.model_dump(mode="json")))


class ValidatedDefaultInput(Data):
    model_config = ConfigDict(validate_default=True)
    optional: JSONValue = Field(default_factory=lambda: cast(JSONValue, None))


class ValidatedDefaultNode(Node[ValidatedDefaultInput, DefaultsOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Validated Default",
        description="Exercise source Data model configuration.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @classmethod
    def static_input_type(cls) -> type[ValidatedDefaultInput]:
        return ValidatedDefaultInput

    @classmethod
    def static_output_type(cls) -> type[DefaultsOutput]:
        return DefaultsOutput

    async def run(self, *, context, input_type, output_type, input):
        return output_type(result=input.optional)


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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inferred_defaults_survive_repeated_execution_and_json_reload(
    engine: WorkflowEngine,
):
    global factory_calls
    factory_calls = 0
    workflow = await engine.build_single_node_workflow(DefaultsNode)
    assert factory_calls == 0
    await engine.validate(workflow)
    assert factory_calls == 0
    restored = type(workflow).model_validate_json(workflow.model_dump_json())
    assert factory_calls == 0

    first = await engine.execute(
        context=InMemoryExecutionContext(), workflow=workflow, input={"required": 1}
    )
    second = await engine.execute(
        context=InMemoryExecutionContext(), workflow=restored, input={"required": 2}
    )
    assert first.status is WorkflowExecutionResultStatus.SUCCESS
    assert second.status is WorkflowExecutionResultStatus.SUCCESS
    assert first.output == {
        "result": {
            "required": 1,
            "literal": "literal",
            "generated": {"key": 1, "call": 1},
        }
    }
    assert second.output == {
        "result": {
            "required": 2,
            "literal": "literal",
            "generated": {"key": 2, "call": 2},
        }
    }
    assert factory_calls == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_node_default_override_and_explicit_fields(
    engine: WorkflowEngine,
):
    global factory_calls
    factory_calls = 0
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=DefaultsNode,
        input={"required": 3, "generated": {"supplied": True}},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {
        "result": {
            "required": 3,
            "literal": "literal",
            "generated": {"supplied": True},
        }
    }
    assert factory_calls == 0

    with pytest.raises(ValidationError, match="literal"):
        await engine.execute_node(
            context=InMemoryExecutionContext(),
            node=DefaultsNode,
            input={"required": 3},
            input_fields={
                "required": JSONValue,
                "literal": JSONValue,
                "generated": JSONValue,
            },
        )
    with pytest.raises(ValidationError, match="required"):
        await engine.execute_node(
            context=InMemoryExecutionContext(),
            node=DefaultsNode,
            input={"generated": None},
        )

    subset = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=DefaultsNode,
        input={"required": 4},
        input_fields={"required": JSONValue},
    )
    assert subset.status is WorkflowExecutionResultStatus.SUCCESS
    assert subset.output == {
        "result": {
            "required": 4,
            "literal": "literal",
            "generated": {"key": 4, "call": 1},
        }
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inferred_defaults_follow_namespaced_source_and_model_config(
    engine: WorkflowEngine,
):
    workflow = await engine.build_single_node_workflow(ValidatedDefaultNode)
    assert workflow.input_node.source_node_id == "node"
    namespaced = workflow.with_namespace("outer")
    assert namespaced.input_node.source_node_id == "outer/node"
    assert namespaced.output_node.source_node_id is None
    restored = type(namespaced).model_validate_json(namespaced.model_dump_json())
    result = await engine.execute(
        context=InMemoryExecutionContext(), workflow=restored, input={}
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"result": None}

    stale = restored.model_update(
        input_node=restored.input_node.model_update(source_node_id="missing")
    )
    with pytest.raises(ValueError, match="Source node 'missing'"):
        await engine.validate(stale)
