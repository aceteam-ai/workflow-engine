import pytest

from workflow_engine import (
    BooleanValue,
    Edge,
    IntegerValue,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.nodes import AddNode, ConstantIntegerNode, IfElseNode


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


@pytest.fixture
def add_one_workflow(engine: WorkflowEngine) -> Workflow:
    """Create a workflow that adds one to a number."""
    return Workflow(
        input_node=(
            input_node := engine.create_input_node(
                start=IntegerValue,
            )
        ),
        output_node=(
            output_node := engine.create_output_node(
                result=IntegerValue,
            )
        ),
        inner_nodes=[
            one := engine.create_node(
                ConstantIntegerNode,
                id="one",
                params=dict(value=1),
            ),
            add_one := engine.create_node(
                AddNode,
                id="add_one",
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=one,
                source_key="value",
                target=add_one,
                target_key="b",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="start",
                target=add_one,
                target_key="a",
            ),
            Edge.from_nodes(
                source=add_one,
                source_key="sum",
                target=output_node,
                target_key="result",
            ),
        ],
    )


@pytest.fixture
def subtract_one_workflow(engine: WorkflowEngine) -> Workflow:
    """Create a workflow that subtracts one from a number."""
    return Workflow(
        input_node=(
            input_node := engine.create_input_node(
                start=IntegerValue,
            )
        ),
        output_node=(
            output_node := engine.create_output_node(
                result=IntegerValue,
            )
        ),
        inner_nodes=[
            negative_one := engine.create_node(
                ConstantIntegerNode,
                id="negative_one",
                params=dict(value=-1),
            ),
            subtract_one := engine.create_node(
                AddNode,
                id="subtract_one",
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=negative_one,
                source_key="value",
                target=subtract_one,
                target_key="b",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="start",
                target=subtract_one,
                target_key="a",
            ),
            Edge.from_nodes(
                source=subtract_one,
                source_key="sum",
                target=output_node,
                target_key="result",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_conditional_workflow(
    engine: WorkflowEngine,
    add_one_workflow: Workflow,
    subtract_one_workflow: Workflow,
):
    """Test that the workflow outputs start+1 when condition is True, and
    start-1 when condition is False."""
    context = InMemoryExecutionContext()

    start_value = 42

    workflow = Workflow(
        input_node=(
            input_node := engine.create_input_node(
                start=IntegerValue,
                condition=BooleanValue,
            )
        ),
        output_node=(
            output_node := engine.create_output_node(
                result=IntegerValue,
            )
        ),
        inner_nodes=[
            conditional := engine.create_node(
                IfElseNode,
                id="conditional",
                params=dict(
                    if_true=add_one_workflow,
                    if_false=subtract_one_workflow,
                ),
            )
        ],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="start",
                target=conditional,
                target_key="start",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="condition",
                target=conditional,
                target_key="condition",
            ),
            Edge.from_nodes(
                source=conditional,
                source_key="result",
                target=output_node,
                target_key="result",
            ),
        ],
    )

    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={
            "start": start_value,
            "condition": False,
        },
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"result": start_value - 1}

    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={
            "start": start_value,
            "condition": True,
        },
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"result": start_value + 1}


@pytest.mark.asyncio
async def test_conditional_workflow_twice_series(
    engine: WorkflowEngine,
    add_one_workflow: Workflow,
    subtract_one_workflow: Workflow,
):
    """Test that the workflow behaves correctly when condition is called twice
    in series, once with True and once with False."""
    context = InMemoryExecutionContext()

    start_value = 42

    workflow = Workflow(
        input_node=(
            input_node := engine.create_input_node(
                start=IntegerValue,
                condition_1=BooleanValue,
                condition_2=BooleanValue,
            )
        ),
        output_node=(
            output_node := engine.create_output_node(
                result=IntegerValue,
            )
        ),
        inner_nodes=[
            conditional_1 := engine.create_node(
                IfElseNode,
                id="conditional_1",
                params=dict(
                    if_true=add_one_workflow,
                    if_false=subtract_one_workflow,
                ),
            ),
            conditional_2 := engine.create_node(
                IfElseNode,
                id="conditional_2",
                params=dict(
                    if_true=add_one_workflow,
                    if_false=subtract_one_workflow,
                ),
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=conditional_1,
                source_key="result",
                target=conditional_2,
                target_key="start",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="start",
                target=conditional_1,
                target_key="start",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="condition_1",
                target=conditional_1,
                target_key="condition",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="condition_2",
                target=conditional_2,
                target_key="condition",
            ),
            Edge.from_nodes(
                source=conditional_2,
                source_key="result",
                target=output_node,
                target_key="result",
            ),
        ],
    )

    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={
            "start": start_value,
            "condition_1": True,
            "condition_2": False,
        },
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"result": start_value}

    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={
            "start": start_value,
            "condition_1": False,
            "condition_2": True,
        },
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"result": start_value}

    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={
            "start": start_value,
            "condition_1": True,
            "condition_2": True,
        },
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"result": start_value + 2}

    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={
            "start": start_value,
            "condition_1": False,
            "condition_2": False,
        },
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"result": start_value - 2}
