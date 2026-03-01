import pytest

from workflow_engine import (
    BooleanValue,
    Edge,
    InputNode,
    IntegerValue,
    OutputNode,
    Workflow,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryContext
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.nodes import AddNode, ConstantIntegerNode, IfElseNode


@pytest.fixture
def add_one_workflow() -> Workflow:
    """Create a workflow that adds one to a number."""
    input_node = InputNode.from_fields(
        start=IntegerValue,
    )
    output_node = OutputNode.from_fields(
        result=IntegerValue,
    )

    one = ConstantIntegerNode.from_value(id="one", value=1)
    add_one = AddNode(id="add_one")

    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[one, add_one],
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
def subtract_one_workflow() -> Workflow:
    """Create a workflow that subtracts one from a number."""
    input_node = InputNode.from_fields(
        start=IntegerValue,
    )
    output_node = OutputNode.from_fields(
        result=IntegerValue,
    )

    negative_one = ConstantIntegerNode.from_value(id="negative_one", value=-1)
    subtract_one = AddNode(id="subtract_one")

    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[negative_one, subtract_one],
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
    add_one_workflow: Workflow,
    subtract_one_workflow: Workflow,
):
    """Test that the workflow outputs start+1 when condition is True, and
    start-1 when condition is False."""
    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()

    start_value = 42

    input_node = InputNode.from_fields(
        start=IntegerValue,
        condition=BooleanValue,
    )
    output_node = OutputNode.from_fields(
        result=IntegerValue,
    )

    conditional = IfElseNode.from_workflows(
        id="conditional",
        if_true=add_one_workflow,
        if_false=subtract_one_workflow,
    )

    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[conditional],
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

    result = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={
            "start": IntegerValue(start_value),
            "condition": BooleanValue(False),
        },
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"result": start_value - 1}

    result = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={
            "start": IntegerValue(start_value),
            "condition": BooleanValue(True),
        },
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"result": start_value + 1}


@pytest.mark.asyncio
async def test_conditional_workflow_twice_series(
    add_one_workflow: Workflow,
    subtract_one_workflow: Workflow,
):
    """Test that the workflow behaves correctly when condition is called twice
    in series, once with True and once with False."""
    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()

    start_value = 42

    input_node = InputNode.from_fields(
        start=IntegerValue,
        condition_1=BooleanValue,
        condition_2=BooleanValue,
    )
    output_node = OutputNode.from_fields(
        result=IntegerValue,
    )

    conditional_1 = IfElseNode.from_workflows(
        id="conditional_1",
        if_true=add_one_workflow,
        if_false=subtract_one_workflow,
    )
    conditional_2 = IfElseNode.from_workflows(
        id="conditional_2",
        if_true=add_one_workflow,
        if_false=subtract_one_workflow,
    )

    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[conditional_1, conditional_2],
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

    result = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={
            "start": IntegerValue(start_value),
            "condition_1": BooleanValue(True),
            "condition_2": BooleanValue(False),
        },
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"result": start_value}

    result = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={
            "start": IntegerValue(start_value),
            "condition_1": BooleanValue(False),
            "condition_2": BooleanValue(True),
        },
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"result": start_value}

    result = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={
            "start": IntegerValue(start_value),
            "condition_1": BooleanValue(True),
            "condition_2": BooleanValue(True),
        },
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"result": start_value + 2}

    result = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={
            "start": IntegerValue(start_value),
            "condition_1": BooleanValue(False),
            "condition_2": BooleanValue(False),
        },
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {"result": start_value - 2}
