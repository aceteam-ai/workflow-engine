"""
Tests for ForEachNode covering all 4 input/output combinations:

1. Single input, single output   - scalar sequence, direct wiring
2. Single input, multi output   - scalar sequence, GatherDataNode only
3. Multi input, single output   - Data sequence, ExpandDataNode only
4. Multi input, multi output    - Data sequence, both adapters
"""

import pytest

from workflow_engine import (
    DataValue,
    Edge,
    Empty,
    FloatValue,
    SequenceValue,
    ValidationContext,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core import ValidatedWorkflow
from workflow_engine.core.values import (
    get_field_annotations,
    get_only_field,
)
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.nodes import AddNode, ForEachNode


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine(execution_algorithm=TopologicalExecutionAlgorithm())


def _input_element_type(workflow: ValidatedWorkflow):
    """Mirror ForEachNode's element type logic for input."""
    n = len(get_field_annotations(workflow.input_type))
    if n == 1:
        return get_only_field(workflow.input_type)[1]
    return DataValue[workflow.input_type]


def _output_element_type(workflow: ValidatedWorkflow):
    """Mirror ForEachNode's element type logic for output."""
    n = len(get_field_annotations(workflow.output_type))
    if n == 0:
        return None
    if n == 1:
        return get_only_field(workflow.output_type)[1]
    return DataValue[workflow.output_type]


def _workflow_with_foreach(
    engine: WorkflowEngine,
    inner_workflow: ValidatedWorkflow,
) -> Workflow:
    """Wrap an inner workflow in a ForEach with outer input/output nodes."""
    input_elem = _input_element_type(inner_workflow)
    output_elem = _output_element_type(inner_workflow)
    if output_elem is None:
        raise ValueError("_workflow_with_foreach does not support 0-output workflows")
    return Workflow(
        input_node=(
            input_node := engine.create_input_node(
                sequence=SequenceValue[input_elem],
            )
        ),
        output_node=(
            output_node := engine.create_output_node(
                results=SequenceValue[output_elem],
            )
        ),
        inner_nodes=[
            for_each := engine.create_node(
                ForEachNode,
                id="for_each",
                params=dict(workflow=inner_workflow),
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="sequence",
                target=for_each,
                target_key="sequence",
            ),
            Edge.from_nodes(
                source=for_each,
                source_key="sequence",
                target=output_node,
                target_key="results",
            ),
        ],
    )


################################################################################
# Fixtures: inner workflows for each of the 4 combinations
################################################################################


@pytest.fixture
def single_in_single_out_workflow(engine: WorkflowEngine) -> Workflow:
    """1 input (x), 1 output (y). Doubles the input: y = 2x."""
    return Workflow(
        input_node=(input_node := engine.create_input_node(x=FloatValue)),
        output_node=(output_node := engine.create_output_node(y=FloatValue)),
        inner_nodes=[
            add := engine.create_node(AddNode, id="add"),
        ],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="x",
                target=add,
                target_key="a",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="x",
                target=add,
                target_key="b",
            ),
            Edge.from_nodes(
                source=add,
                source_key="sum",
                target=output_node,
                target_key="y",
            ),
        ],
    )


@pytest.fixture
def single_in_multi_out_workflow(engine: WorkflowEngine) -> Workflow:
    """1 input (x), 2 outputs (first, second). Duplicates: first=x, second=x."""
    return Workflow(
        input_node=(input_node := engine.create_input_node(x=FloatValue)),
        output_node=(
            output_node := engine.create_output_node(
                first=FloatValue,
                second=FloatValue,
            )
        ),
        inner_nodes=[],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="x",
                target=output_node,
                target_key="first",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="x",
                target=output_node,
                target_key="second",
            ),
        ],
    )


@pytest.fixture
def multi_in_single_out_workflow(engine: WorkflowEngine) -> Workflow:
    """2 inputs (a, b), 1 output (c). Adds: c = a + b."""
    return Workflow(
        input_node=(
            input_node := engine.create_input_node(
                a=FloatValue,
                b=FloatValue,
            )
        ),
        output_node=(output_node := engine.create_output_node(c=FloatValue)),
        inner_nodes=[
            add := engine.create_node(AddNode, id="add"),
        ],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="a",
                target=add,
                target_key="a",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="b",
                target=add,
                target_key="b",
            ),
            Edge.from_nodes(
                source=add,
                source_key="sum",
                target=output_node,
                target_key="c",
            ),
        ],
    )


@pytest.fixture
def multi_in_multi_out_workflow(engine: WorkflowEngine) -> Workflow:
    """2 inputs (a, b), 2 outputs (sum, sum_copy). Both outputs are a + b."""
    return Workflow(
        input_node=(
            input_node := engine.create_input_node(
                a=FloatValue,
                b=FloatValue,
            )
        ),
        output_node=(
            output_node := engine.create_output_node(
                sum=FloatValue,
                sum_copy=FloatValue,
            )
        ),
        inner_nodes=[
            add := engine.create_node(AddNode, id="add"),
        ],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="a",
                target=add,
                target_key="a",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="b",
                target=add,
                target_key="b",
            ),
            Edge.from_nodes(
                source=add,
                source_key="sum",
                target=output_node,
                target_key="sum",
            ),
            Edge.from_nodes(
                source=add,
                source_key="sum",
                target=output_node,
                target_key="sum_copy",
            ),
        ],
    )


################################################################################
# Tests
################################################################################


@pytest.mark.asyncio
async def test_single_in_single_out(
    engine: WorkflowEngine,
    single_in_single_out_workflow: Workflow,
):
    """ForEach with 1 input field, 1 output field: scalar sequence, no adapters."""
    context = InMemoryExecutionContext()
    workflow = _workflow_with_foreach(
        engine, await engine.validate(single_in_single_out_workflow)
    )
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={"sequence": [1.0, 2.0, 3.0]},
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    results = result.output["results"]
    assert isinstance(results, SequenceValue)
    assert len(results) == 3
    # Single output: each element is FloatValue directly
    assert results[0].root == 2.0  # 2 * 1
    assert results[1].root == 4.0  # 2 * 2
    assert results[2].root == 6.0  # 2 * 3


@pytest.mark.asyncio
async def test_single_in_multi_out(
    engine: WorkflowEngine,
    single_in_multi_out_workflow: Workflow,
):
    """ForEach with 1 input field, 2 output fields: scalar in, GatherDataNode."""
    context = InMemoryExecutionContext()
    workflow = _workflow_with_foreach(
        engine, await engine.validate(single_in_multi_out_workflow)
    )
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={"sequence": [5.0, 10.0]},
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    results = result.output["results"]
    assert isinstance(results, SequenceValue)
    assert len(results) == 2
    # Multi output: each element is DataValue[Data] with first, second fields
    assert results[0].root.first.root == 5.0
    assert results[0].root.second.root == 5.0
    assert results[1].root.first.root == 10.0
    assert results[1].root.second.root == 10.0


@pytest.mark.asyncio
async def test_multi_in_single_out(
    engine: WorkflowEngine,
    multi_in_single_out_workflow: Workflow,
):
    """ForEach with 2 input fields, 1 output field: ExpandDataNode, no Gather."""
    context = InMemoryExecutionContext()
    workflow = _workflow_with_foreach(
        engine, await engine.validate(multi_in_single_out_workflow)
    )

    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={
            "sequence": [
                {"a": 1.0, "b": 2.0},
                {"a": 3.0, "b": 4.0},
                {"a": 5.0, "b": 6.0},
            ]
        },
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    results = result.output["results"]
    assert isinstance(results, SequenceValue)
    assert len(results) == 3
    # Multi input, single output: each element is FloatValue directly (no GatherData)
    assert results[0].root == 3.0
    assert results[1].root == 7.0
    assert results[2].root == 11.0


@pytest.mark.asyncio
async def test_multi_in_multi_out(
    engine: WorkflowEngine,
    multi_in_multi_out_workflow: Workflow,
):
    """ForEach with 2 input fields, 2 output fields: both ExpandData and GatherData."""
    context = InMemoryExecutionContext()
    workflow = _workflow_with_foreach(
        engine, await engine.validate(multi_in_multi_out_workflow)
    )
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={
            "sequence": [
                {"a": 1.0, "b": 2.0},
                {"a": 10.0, "b": 20.0},
            ]
        },
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    results = result.output["results"]
    assert isinstance(results, SequenceValue)
    assert len(results) == 2
    assert results[0].root.sum.root == 3.0
    assert results[0].root.sum_copy.root == 3.0
    assert results[1].root.sum.root == 30.0
    assert results[1].root.sum_copy.root == 30.0


@pytest.fixture
def zero_output_single_in_workflow(engine: WorkflowEngine) -> Workflow:
    """1 input (x), 0 outputs. Consumes input for side effect (add), produces nothing."""
    return Workflow(
        input_node=(input_node := engine.create_input_node(x=FloatValue)),
        output_node=engine.create_output_node(),
        inner_nodes=[
            add := engine.create_node(AddNode, id="add"),
        ],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="x",
                target=add,
                target_key="a",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="x",
                target=add,
                target_key="b",
            ),
        ],
    )


@pytest.fixture
def zero_output_multi_in_workflow(engine: WorkflowEngine) -> Workflow:
    """2 inputs (a, b), 0 outputs. Consumes input for side effect (add), produces nothing."""
    return Workflow(
        input_node=(
            input_node := engine.create_input_node(
                a=FloatValue,
                b=FloatValue,
            )
        ),
        output_node=engine.create_output_node(),
        inner_nodes=[
            add := engine.create_node(AddNode, id="add"),
        ],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="a",
                target=add,
                target_key="a",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="b",
                target=add,
                target_key="b",
            ),
        ],
    )


async def _test_zero_output_workflow(
    engine: WorkflowEngine,
    workflow: Workflow,
    input_data: dict,
) -> None:
    """Shared logic for zero-output tests."""
    for_each = engine.create_node(
        ForEachNode,
        id="for_each",
        params=dict(workflow=workflow),
    )
    validation_context = ValidationContext()
    assert await for_each.output_type(validation_context) is Empty

    workflow = await engine.validate(workflow)

    input_elem = _input_element_type(workflow)
    wf = Workflow(
        input_node=(
            input_node := engine.create_input_node(
                sequence=SequenceValue[input_elem],
            )
        ),
        output_node=engine.create_output_node(),
        inner_nodes=[for_each],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="sequence",
                target=for_each,
                target_key="sequence",
            ),
        ],
    )

    context = InMemoryExecutionContext()
    result = await engine.execute(
        context=context,
        workflow=wf,
        input=input_data,
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output == {}


@pytest.mark.asyncio
async def test_zero_output_single_in(
    engine: WorkflowEngine,
    zero_output_single_in_workflow: Workflow,
):
    """ForEach: 1 input, 0 outputs. Outputs nothing (Empty)."""
    await _test_zero_output_workflow(
        engine,
        zero_output_single_in_workflow,
        {"sequence": [1.0, 2.0, 3.0]},
    )


@pytest.mark.asyncio
async def test_zero_output_multi_in(
    engine: WorkflowEngine,
    zero_output_multi_in_workflow: Workflow,
):
    """ForEach: 2 inputs, 0 outputs. Outputs nothing (Empty)."""
    await _test_zero_output_workflow(
        engine,
        zero_output_multi_in_workflow,
        {
            "sequence": [
                {"a": 1.0, "b": 2.0},
                {"a": 3.0, "b": 4.0},
            ]
        },
    )


@pytest.mark.asyncio
async def test_zero_output_multi_in_empty(
    engine: WorkflowEngine,
    zero_output_multi_in_workflow: Workflow,
):
    """ForEach: 2 inputs, 0 outputs. Outputs nothing (Empty)."""
    await _test_zero_output_workflow(
        engine,
        zero_output_multi_in_workflow,
        {"sequence": []},
    )


@pytest.mark.asyncio
async def test_for_each_empty(
    engine: WorkflowEngine,
    multi_in_single_out_workflow: Workflow,
):
    """Empty sequence produces empty results."""
    context = InMemoryExecutionContext()
    workflow = _workflow_with_foreach(
        engine, await engine.validate(multi_in_single_out_workflow)
    )
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={"sequence": []},
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    results = result.output.get("results", [])
    assert isinstance(results, SequenceValue)
    assert len(results) == 0
