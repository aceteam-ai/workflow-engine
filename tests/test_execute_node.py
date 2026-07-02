import pytest

from workflow_engine import (
    FloatValue,
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
