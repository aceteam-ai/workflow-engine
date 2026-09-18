# tests/test_unvalidated_workflow_guard.py
"""
Regression coverage for #282: a plain `Workflow` that reaches an
`ExecutionAlgorithm` must fail loudly at the executor boundary, and the
error-path partial-output handler must return real partial results, not an
empty mapping that would pass a weaker assertion.
"""

import pytest

from workflow_engine import (
    Edge,
    ExecutionAlgorithm,
    StringValue,
    Workflow,
    WorkflowEngine,
    WorkflowException,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.execution.replacement import ReplacementGraph
from workflow_engine.nodes import ConstantStringNode, ErrorNode


def _partial_success_workflow(engine: WorkflowEngine) -> Workflow:
    """
    A two-node workflow where one node succeeds and wires to the output,
    and a sibling node fails independently. On the error path this proves
    the partial-output handler returns the surviving node's real output,
    not an empty mapping.
    """
    return Workflow(
        input_node=engine.create_input_node(),
        output_node=(output_node := engine.create_output_node(value=StringValue)),
        inner_nodes=[
            constant := engine.create_node(
                ConstantStringNode, id="constant", params=dict(value="survives")
            ),
            error := engine.create_node(
                ErrorNode, id="error", params=dict(error_name="BoomError")
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=constant,
                source_key="value",
                target=error,
                target_key="info",
            ),
            Edge.from_nodes(
                source=constant,
                source_key="value",
                target=output_node,
                target_key="value",
            ),
        ],
    )


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_path_returns_real_partial_output(
    algorithm: ExecutionAlgorithm,
    engine: WorkflowEngine,
):
    """
    The bug this guards against turned the error-path partial-output
    handler into an AttributeError. This asserts the handler not only
    avoids crashing but returns the surviving node's actual output.
    """
    workflow = _partial_success_workflow(engine)
    engine = WorkflowEngine(execution_algorithm=algorithm)

    result = await engine.execute(
        context=InMemoryExecutionContext(),
        workflow=workflow,
        input={},
    )

    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert result.output == {"value": "survives"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_rejects_unvalidated_workflow(
    algorithm: ExecutionAlgorithm,
    engine: WorkflowEngine,
):
    """
    An ExecutionAlgorithm's `execute()` is typed to take a
    `ValidatedWorkflow`. Calling it directly with a plain `Workflow`
    (bypassing `WorkflowEngine.execute`'s call to `validate()`) must fail
    with a clear, specific diagnostic at the boundary, not an incidental
    AttributeError deep inside scheduler internals.
    """
    plain_workflow = await engine.build_single_node_workflow(
        ConstantStringNode, params=dict(value="unused")
    )
    assert type(plain_workflow) is Workflow

    with pytest.raises(WorkflowException) as excinfo:
        await algorithm.execute(
            context=InMemoryExecutionContext(),
            workflow=plain_workflow,
            input={},
        )

    message = str(excinfo.value)
    assert "ValidatedWorkflow" in message
    assert "Workflow" in message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_from_validated_returns_the_replacement_graph_type():
    """
    ``ReplacementGraph.from_validated`` must return a ``ReplacementGraph``.

    This is a contract test, not a regression test for #282. The failure it
    describes cannot be reproduced in this repository: the downgrade that
    motivated ``from_validated`` only appears once an embedding application has
    defined its own node types, and it was found by running a downstream
    application's suite against the engine. The test is here so that a future
    change back to ``model_validate`` has to argue with an explicit assertion
    rather than silently reintroduce the bug.
    """
    engine = WorkflowEngine()
    workflow = await engine.build_single_node_workflow(
        ConstantStringNode, params={"value": StringValue("x")}
    )
    validated = await engine.validate(workflow)

    graph = ReplacementGraph.from_validated(validated)

    assert type(graph) is ReplacementGraph
    assert graph.nodes == validated.nodes
