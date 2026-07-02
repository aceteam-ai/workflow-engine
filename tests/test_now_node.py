from datetime import datetime, timezone

import pytest

from workflow_engine import (
    DateValue,
    Edge,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.nodes import NowNode


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


@pytest.mark.asyncio
async def test_now_node(engine: WorkflowEngine):
    before = datetime.now(timezone.utc)
    workflow = Workflow(
        input_node=engine.create_input_node(),
        output_node=(output_node := engine.create_output_node(now=DateValue)),
        inner_nodes=[now := engine.create_node(NowNode, id="now")],
        edges=[
            Edge.from_nodes(
                source=now,
                source_key="now",
                target=output_node,
                target_key="now",
            ),
        ],
    )
    result = await engine.execute(
        context=InMemoryExecutionContext(),
        workflow=workflow,
        input={},
    )
    after = datetime.now(timezone.utc)

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    now_value = result.output["now"]
    assert isinstance(now_value, DateValue)
    assert before <= now_value.root <= after
