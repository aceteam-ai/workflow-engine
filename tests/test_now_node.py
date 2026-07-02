from datetime import datetime, timezone

import pytest

from workflow_engine import (
    DateValue,
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
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=NowNode,
        input={},
    )
    after = datetime.now(timezone.utc)

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    now_value = result.output["now"]
    assert isinstance(now_value, DateValue)
    assert before <= now_value.root <= after
