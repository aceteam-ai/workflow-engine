"""
Tests for the hints annotation channel (#203).

The contract under test: a host that ignores every hint still computes the
same result. This is checked directly by running a fan-out workflow twice,
once with a ``max_concurrency`` hint attached to its ``ForEach`` node and once
with ``Workflow.without_hints()`` applied, and asserting identical outputs
under both execution algorithms.
"""

import pytest

from workflow_engine import (
    Edge,
    FloatValue,
    Hints,
    Node,
    SequenceValue,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core import ExecutionAlgorithm
from workflow_engine.core.node import NodeRegistry
from workflow_engine.nodes import AddNode, ForEachNode


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


@pytest.fixture
def parameterized_engine(algorithm: ExecutionAlgorithm) -> WorkflowEngine:
    """An engine using each of the bundled execution algorithms in turn."""
    return WorkflowEngine(execution_algorithm=algorithm)


def _double_workflow(engine: WorkflowEngine) -> Workflow:
    """Inner workflow for the fan-out below: y = 2x."""
    return Workflow(
        input_node=(input_node := engine.create_input_node(x=FloatValue)),
        output_node=(output_node := engine.create_output_node(y=FloatValue)),
        inner_nodes=[add := engine.create_node(AddNode, id="add")],
        edges=[
            Edge.from_nodes(
                source=input_node, source_key="x", target=add, target_key="a"
            ),
            Edge.from_nodes(
                source=input_node, source_key="x", target=add, target_key="b"
            ),
            Edge.from_nodes(
                source=add, source_key="sum", target=output_node, target_key="y"
            ),
        ],
    )


def _fan_out_workflow(
    engine: WorkflowEngine, *, hints: Hints | None = None
) -> Workflow:
    """A ForEach over the doubling workflow, with an optional hints-bearing node."""
    inner = _double_workflow(engine)
    for_each_kwargs = {} if hints is None else {"hints": hints}
    return Workflow(
        input_node=(
            input_node := engine.create_input_node(sequence=SequenceValue[FloatValue])
        ),
        output_node=(
            output_node := engine.create_output_node(results=SequenceValue[FloatValue])
        ),
        inner_nodes=[
            for_each := engine.create_node(
                ForEachNode,
                id="for_each",
                params=dict(workflow=inner),
                **for_each_kwargs,
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


# --------------------------------------------------------------------------
# Hints as a value: defaults, round-tripping, forward compatibility
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_hints_default_is_empty():
    hints = Hints()
    assert hints.max_concurrency is None


@pytest.mark.unit
def test_node_default_hints_is_empty(engine: WorkflowEngine):
    node = engine.create_node(AddNode, id="add")
    assert node.hints == Hints()
    assert node.hints.max_concurrency is None


@pytest.mark.unit
def test_node_hints_round_trip_json(engine: WorkflowEngine):
    node = engine.create_node(AddNode, id="add", hints=Hints(max_concurrency=4))

    dumped = node.model_dump_json()
    reloaded = type(node).model_validate_json(dumped)

    assert reloaded.hints.max_concurrency == 4


@pytest.mark.unit
def test_hints_reject_non_positive_max_concurrency():
    with pytest.raises(ValueError):
        Hints(max_concurrency=0)


@pytest.mark.unit
def test_hints_unknown_key_round_trips():
    """
    A hint's entire point is that not understanding it is safe: unknown keys
    must survive a dump/validate round trip rather than being rejected.
    """
    hints = Hints.model_validate({"max_concurrency": 2, "future_hint": "clamp-me"})

    dumped = hints.model_dump()

    assert dumped["max_concurrency"] == 2
    assert dumped["future_hint"] == "clamp-me"


@pytest.mark.unit
def test_node_registry_load_preserves_hints(engine: WorkflowEngine):
    """
    NodeRegistry.load() dumps an untyped node and re-validates it into the
    concrete class; hints (including a key unknown to this engine) must
    survive that path, since this is exactly what happens when loading a
    graph produced by a newer host.
    """
    untyped = Node.model_validate(
        {
            "type": "Add",
            "id": "add",
            "hints": {"max_concurrency": 3, "future_hint": "value"},
        }
    )

    loaded = NodeRegistry.DEFAULT.load(untyped)

    assert loaded.hints.max_concurrency == 3
    assert loaded.hints.model_dump()["future_hint"] == "value"


@pytest.mark.unit
def test_with_namespace_preserves_hints(engine: WorkflowEngine):
    node = engine.create_node(AddNode, id="add", hints=Hints(max_concurrency=2))

    namespaced = node.with_namespace("outer")

    assert namespaced.id == "outer/add"
    assert namespaced.hints.max_concurrency == 2


@pytest.mark.unit
def test_without_hints_erases_node_hints(engine: WorkflowEngine):
    node = engine.create_node(AddNode, id="add", hints=Hints(max_concurrency=2))

    stripped = node.without_hints()

    assert stripped.hints == Hints()
    assert stripped.hints.max_concurrency is None
    # Nothing else about the node changes.
    assert stripped.id == node.id
    assert stripped.type == node.type


@pytest.mark.unit
def test_workflow_without_hints_strips_every_node(engine: WorkflowEngine):
    workflow = _fan_out_workflow(engine, hints=Hints(max_concurrency=1))
    assert workflow.nodes_by_id["for_each"].hints.max_concurrency == 1

    stripped = workflow.without_hints()

    for node in stripped.nodes:
        assert node.hints == Hints()


# --------------------------------------------------------------------------
# The contract: ignoring every hint does not change the result.
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ignoring_hints_does_not_change_result(
    parameterized_engine: WorkflowEngine,
):
    """
    A fan-out with a max_concurrency hint on its ForEach node must compute
    the exact same output as the same graph with every hint stripped, under
    both bundled execution algorithms. Nothing in this engine reads
    ``Node.hints`` today, so this test is the executable form of "a host
    that ignores every annotation still computes the same result."
    """
    hinted = _fan_out_workflow(parameterized_engine, hints=Hints(max_concurrency=1))
    unhinted = hinted.without_hints()

    input_data = {"sequence": [1.0, 2.0, 3.0, 4.0, 5.0]}

    hinted_result = await parameterized_engine.execute(
        context=InMemoryExecutionContext(),
        workflow=hinted,
        input=input_data,
    )
    unhinted_result = await parameterized_engine.execute(
        context=InMemoryExecutionContext(),
        workflow=unhinted,
        input=input_data,
    )

    assert hinted_result.status is WorkflowExecutionResultStatus.SUCCESS
    assert unhinted_result.status is WorkflowExecutionResultStatus.SUCCESS
    assert hinted_result.output == unhinted_result.output
    assert hinted_result.output == {"results": [2.0, 4.0, 6.0, 8.0, 10.0]}


def test_without_hints_does_not_reach_into_a_nested_workflow(
    engine: WorkflowEngine,
):
    """
    ``Node.without_hints()`` clears a node's own hints and nothing else. A
    ForEach carries an entire workflow in ``params.workflow``, and hints on
    the nodes inside that nested workflow survive stripping.

    This does not break the hints contract: no execution code reads
    ``Node.hints`` at any depth, so a surviving nested hint still cannot
    change a result. It does mean ``without_hints()`` is shallower than
    "erase every node's hints" suggests, and that the contract test's
    "stripped twin" is only stripped at the top level.

    Pinned here so the limitation is a known, deliberate boundary rather
    than an assumption someone later relies on. See the follow-up issue for
    making it recurse.
    """
    inner = _double_workflow(engine)
    hinted_inner = inner.model_update(
        inner_nodes=[
            node.model_update(hints=Hints(max_concurrency=7))
            for node in inner.inner_nodes
        ],
    )
    assert any(node.hints.max_concurrency == 7 for node in hinted_inner.inner_nodes), (
        "fixture should have hinted at least one inner node"
    )

    outer = Workflow(
        input_node=(
            input_node := engine.create_input_node(sequence=SequenceValue[FloatValue])
        ),
        output_node=(
            output_node := engine.create_output_node(results=SequenceValue[FloatValue])
        ),
        inner_nodes=[
            for_each := engine.create_node(
                ForEachNode,
                id="for_each",
                params=dict(workflow=hinted_inner),
                hints=Hints(max_concurrency=1),
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

    stripped = outer.without_hints()

    # The top level is stripped, as documented.
    for node in stripped.nodes:
        assert node.hints == Hints()

    # The nested workflow is not.
    nested = stripped.nodes_by_id["for_each"].params.workflow.root
    assert any(node.hints.max_concurrency == 7 for node in nested.inner_nodes), (
        "expected nested hints to survive; if this now fails, without_hints() "
        "recurses and this test should be replaced with the positive assertion"
    )
