"""Keep the published example graphs valid against this engine's schema."""

from pathlib import Path

import pytest

import workflow_engine.nodes  # noqa: F401  Register built-in node and value types.
from workflow_engine import Workflow, WorkflowEngine

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["addition", "append", "error"])
async def test_bundled_example_validates(name: str) -> None:
    workflow = Workflow.model_validate_json((EXAMPLES / f"{name}.json").read_text())
    await WorkflowEngine().validate(workflow)
