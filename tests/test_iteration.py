"""
Tests for ForEachNode covering all 4 input/output combinations:

1. Single input, single output   - scalar sequence, direct wiring
2. Single input, multi output   - scalar sequence, GatherDataNode only
3. Multi input, single output   - Data sequence, ExpandDataNode only
4. Multi input, multi output    - Data sequence, both adapters
"""

import pytest

from workflow_engine import (
    Edge,
    Empty,
    FloatValue,
    SequenceValue,
    Workflow,
    DataValue,
)
from workflow_engine.core.values import get_data_dict, get_field_annotations, get_only_field
from workflow_engine.contexts import InMemoryContext
from workflow_engine.core.io import InputNode, OutputNode
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.nodes import AddNode, ForEachNode


def _input_element_type(workflow: Workflow):
    """Mirror ForEachNode's element type logic for input."""
    n = len(get_field_annotations(workflow.input_type))
    if n == 1:
        return get_only_field(workflow.input_type)[1]
    return DataValue[workflow.input_type]


def _output_element_type(workflow: Workflow):
    """Mirror ForEachNode's element type logic for output."""
    n = len(get_field_annotations(workflow.output_type))
    if n == 0:
        return None
    if n == 1:
        return get_only_field(workflow.output_type)[1]
    return DataValue[workflow.output_type]


def _workflow_with_foreach(inner_workflow: Workflow) -> Workflow:
    """Wrap an inner workflow in a ForEach with outer input/output nodes."""
    for_each = ForEachNode.from_workflow(id="for_each", workflow=inner_workflow)
    input_elem = _input_element_type(inner_workflow)
    output_elem = _output_element_type(inner_workflow)
    if output_elem is None:
        raise ValueError("_workflow_with_foreach does not support 0-output workflows")
    input_node = InputNode.from_fields(
        sequence=SequenceValue[input_elem],
    )
    output_node = OutputNode.from_fields(
        results=SequenceValue[output_elem],
    )
    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[for_each],
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
def single_in_single_out_workflow() -> Workflow:
    """1 input (x), 1 output (y). Doubles the input: y = 2x."""
    input_node = InputNode.from_fields(x=FloatValue)
    output_node = OutputNode.from_fields(y=FloatValue)
    add = AddNode(id="add")
    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[add],
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
def single_in_multi_out_workflow() -> Workflow:
    """1 input (x), 2 outputs (first, second). Duplicates: first=x, second=x."""
    input_node = InputNode.from_fields(x=FloatValue)
    output_node = OutputNode.from_fields(
        first=FloatValue,
        second=FloatValue,
    )
    return Workflow(
        input_node=input_node,
        output_node=output_node,
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
def multi_in_single_out_workflow() -> Workflow:
    """2 inputs (a, b), 1 output (c). Adds: c = a + b."""
    input_node = InputNode.from_fields(
        a=FloatValue,
        b=FloatValue,
    )
    output_node = OutputNode.from_fields(c=FloatValue)
    add = AddNode(id="add")
    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[add],
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
def multi_in_multi_out_workflow() -> Workflow:
    """2 inputs (a, b), 2 outputs (sum, sum_copy). Both outputs are a + b."""
    input_node = InputNode.from_fields(
        a=FloatValue,
        b=FloatValue,
    )
    output_node = OutputNode.from_fields(
        sum=FloatValue,
        sum_copy=FloatValue,
    )
    add = AddNode(id="add")
    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[add],
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
async def test_single_in_single_out(single_in_single_out_workflow: Workflow):
    """ForEach with 1 input field, 1 output field: scalar sequence, no adapters."""
    workflow = _workflow_with_foreach(single_in_single_out_workflow)
    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()

    input_data = get_data_dict(workflow.input_type.model_validate(
        {"sequence": [1.0, 2.0, 3.0]}
    ))

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input=input_data,
    )

    assert not errors.any(), errors
    results = output["results"]
    assert isinstance(results, SequenceValue)
    assert len(results) == 3
    # Single output: each element is FloatValue directly
    assert results[0].root == 2.0  # 2 * 1
    assert results[1].root == 4.0  # 2 * 2
    assert results[2].root == 6.0  # 2 * 3


@pytest.mark.asyncio
async def test_single_in_multi_out(single_in_multi_out_workflow: Workflow):
    """ForEach with 1 input field, 2 output fields: scalar in, GatherDataNode."""
    workflow = _workflow_with_foreach(single_in_multi_out_workflow)
    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()

    input_data = get_data_dict(workflow.input_type.model_validate({"sequence": [5.0, 10.0]}))

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input=input_data,
    )

    assert not errors.any(), errors
    results = output["results"]
    assert isinstance(results, SequenceValue)
    assert len(results) == 2
    # Multi output: each element is DataValue[Data] with first, second fields
    assert results[0].root.first.root == 5.0
    assert results[0].root.second.root == 5.0
    assert results[1].root.first.root == 10.0
    assert results[1].root.second.root == 10.0


@pytest.mark.asyncio
async def test_multi_in_single_out(multi_in_single_out_workflow: Workflow):
    """ForEach with 2 input fields, 1 output field: ExpandDataNode, no Gather."""
    workflow = _workflow_with_foreach(multi_in_single_out_workflow)
    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()

    input_data = get_data_dict(workflow.input_type.model_validate(
        {
            "sequence": [
                {"a": 1.0, "b": 2.0},
                {"a": 3.0, "b": 4.0},
                {"a": 5.0, "b": 6.0},
            ]
        }
    ))

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input=input_data,
    )

    assert not errors.any(), errors
    results = output["results"]
    assert isinstance(results, SequenceValue)
    assert len(results) == 3
    # Multi input, single output: each element is FloatValue directly (no GatherData)
    assert results[0].root == 3.0
    assert results[1].root == 7.0
    assert results[2].root == 11.0


@pytest.mark.asyncio
async def test_multi_in_multi_out(multi_in_multi_out_workflow: Workflow):
    """ForEach with 2 input fields, 2 output fields: both ExpandData and GatherData."""
    workflow = _workflow_with_foreach(multi_in_multi_out_workflow)
    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()

    input_data = get_data_dict(workflow.input_type.model_validate(
        {
            "sequence": [
                {"a": 1.0, "b": 2.0},
                {"a": 10.0, "b": 20.0},
            ]
        }
    ))

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input=input_data,
    )

    assert not errors.any(), errors
    results = output["results"]
    assert isinstance(results, SequenceValue)
    assert len(results) == 2
    assert results[0].root.sum.root == 3.0
    assert results[0].root.sum_copy.root == 3.0
    assert results[1].root.sum.root == 30.0
    assert results[1].root.sum_copy.root == 30.0


@pytest.fixture
def zero_output_single_in_workflow() -> Workflow:
    """1 input (x), 0 outputs. Consumes input for side effect (add), produces nothing."""
    input_node = InputNode.from_fields(x=FloatValue)
    output_node = OutputNode.empty()
    add = AddNode(id="add")
    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[add],
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
def zero_output_multi_in_workflow() -> Workflow:
    """2 inputs (a, b), 0 outputs. Consumes input for side effect (add), produces nothing."""
    input_node = InputNode.from_fields(
        a=FloatValue,
        b=FloatValue,
    )
    output_node = OutputNode.empty()
    add = AddNode(id="add")
    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[add],
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


async def _test_zero_output_workflow(workflow: Workflow, input_data: dict) -> None:
    """Shared logic for zero-output tests."""
    for_each = ForEachNode.from_workflow(
        id="for_each",
        workflow=workflow,
    )
    assert for_each.output_type is Empty

    input_elem = _input_element_type(workflow)
    input_node = InputNode.from_fields(
        sequence=SequenceValue[input_elem],
    )
    output_node = OutputNode.empty()
    wf = Workflow(
        input_node=input_node,
        output_node=output_node,
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

    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()
    input_dict = get_data_dict(wf.input_type.model_validate(input_data))
    errors, output = await algorithm.execute(
        context=context,
        workflow=wf,
        input=input_dict,
    )

    assert not errors.any(), errors
    assert output == {}


@pytest.mark.asyncio
async def test_zero_output_single_in(zero_output_single_in_workflow: Workflow):
    """ForEach: 1 input, 0 outputs. Outputs nothing (Empty)."""
    await _test_zero_output_workflow(
        zero_output_single_in_workflow,
        {"sequence": [1.0, 2.0, 3.0]},
    )


@pytest.mark.asyncio
async def test_zero_output_multi_in(zero_output_multi_in_workflow: Workflow):
    """ForEach: 2 inputs, 0 outputs. Outputs nothing (Empty)."""
    await _test_zero_output_workflow(
        zero_output_multi_in_workflow,
        {
            "sequence": [
                {"a": 1.0, "b": 2.0},
                {"a": 3.0, "b": 4.0},
            ]
        },
    )


@pytest.mark.asyncio
async def test_for_each_empty(multi_in_single_out_workflow: Workflow):
    """Empty sequence produces empty results."""
    workflow = _workflow_with_foreach(multi_in_single_out_workflow)
    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()

    input_data = get_data_dict(workflow.input_type.model_validate({"sequence": []}))

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input=input_data,
    )

    assert not errors.any(), errors
    results = output["results"]
    assert isinstance(results, SequenceValue)
    assert len(results) == 0
