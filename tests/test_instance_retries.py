"""Per-instance retry policy survives graph copying and beats fallback budgets."""

import pytest
from pydantic import ValidationError

from tests.test_retry import CustomRetryNode, RetryableNode
from workflow_engine import (
    Edge,
    Node,
    StringValue,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.nodes import ForEachNode


def workflow_for_node(engine: WorkflowEngine, node: Node) -> Workflow:
    return Workflow(
        input_node=engine.create_input_node(value=StringValue),
        inner_nodes=[node],
        output_node=engine.create_output_node(result=StringValue),
        edges=[
            Edge(
                source_id="input",
                source_key="value",
                target_id=node.id,
                target_key="value",
            ),
            Edge(
                source_id=node.id,
                source_key="result",
                target_id="output",
                target_key="result",
            ),
        ],
    )


@pytest.fixture(autouse=True)
def reset_counts():
    CustomRetryNode._attempt_counts.clear()
    RetryableNode._attempt_counts.clear()


@pytest.mark.parametrize(
    "node_cls,instance_budget,run_budget,fail_count,expected_calls,succeeds",
    [
        (CustomRetryNode, 0, 3, 1, 1, False),
        (CustomRetryNode, 2, 0, 3, 3, False),
        (CustomRetryNode, 6, 0, 6, 7, True),
        (CustomRetryNode, None, 0, 4, 5, True),
        (RetryableNode, None, 2, 3, 3, False),
        (RetryableNode, None, 3, 2, 3, True),
    ],
)
async def test_instance_type_run_precedence(
    algorithm,
    node_cls,
    instance_budget,
    run_budget,
    fail_count,
    expected_calls,
    succeeds,
):
    algorithm.max_retries = run_budget
    engine = WorkflowEngine(execution_algorithm=algorithm)
    node = engine.create_node(
        node_cls,
        id="probe",
        params={"fail_count": fail_count},
        max_retries=instance_budget,
    )
    workflow = workflow_for_node(engine, node)
    copied = Workflow.model_validate_json(workflow.model_dump_json())
    result = await engine.execute(
        context=InMemoryExecutionContext(), workflow=copied, input={"value": "input"}
    )
    assert node_cls._attempt_counts["probe"] == expected_calls
    assert (result.status is WorkflowExecutionResultStatus.SUCCESS) is succeeds


async def test_instance_retry_budget_survives_foreach_expansion(algorithm):
    algorithm.max_retries = 0
    engine = WorkflowEngine(execution_algorithm=algorithm)
    node = engine.create_node(
        CustomRetryNode, id="probe", params={"fail_count": 1}, max_retries=1
    )
    inner = workflow_for_node(engine, node)
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=ForEachNode,
        params={"workflow": inner},
        input={"sequence": [StringValue("a"), StringValue("b")]},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert len(CustomRetryNode._attempt_counts) == 2
    assert set(CustomRetryNode._attempt_counts.values()) == {2}


def test_optional_budget_wire_shape_and_hint_erasure():
    engine = WorkflowEngine()
    node = engine.create_node(RetryableNode, id="probe", params={"fail_count": 0})
    assert "max_retries" not in node.model_dump(mode="json")
    assert "max_retries" not in node.model_update(max_retries=None).model_dump(
        mode="json"
    )
    disabled = node.model_update(max_retries=0)
    assert disabled.model_dump(mode="json")["max_retries"] == 0
    assert disabled.without_hints().max_retries == 0
    with pytest.raises(ValidationError):
        node.model_update(max_retries=-1)
