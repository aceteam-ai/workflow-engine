# tests/conftest.py
"""Shared test fixtures and utilities.

Note: defining a Node subclass here causes it to auto-register on the default
node registry, so anything declared here is available to every test file
without needing to be re-imported.
"""

from typing import ClassVar, Type

import pytest
from overrides import override

from workflow_engine import (
    Data,
    Empty,
    ExecutionAlgorithm,
    ExecutionContext,
    Node,
    NodeTypeInfo,
    Params,
    ShouldYield,
    StringValue,
)
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.execution.parallel import ParallelExecutionAlgorithm

# Fixture modules registered project-wide. Add new fixture files here rather
# than letting this conftest sprawl.
pytest_plugins = [
    "tests.fixtures.filesystem",
]


@pytest.fixture(params=["topological", "parallel"])
def algorithm(request) -> ExecutionAlgorithm:
    """Parameterized over the two built-in execution algorithms."""
    if request.param == "topological":
        return TopologicalExecutionAlgorithm()
    return ParallelExecutionAlgorithm()


class SimpleInput(Data):
    value: StringValue


class SimpleOutput(Data):
    result: StringValue


class YieldingNode(Node[Empty, SimpleOutput, Params]):
    """Test helper: always raises ShouldYield with a configurable message."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Yielding",
        description="Test helper that always yields.",
        version="1.0.0",
        parameter_type=Params,
    )

    message: str = "waiting for approval"

    # Per-id invocation counter. Reset between tests by ``reset_test_node_state``.
    calls: ClassVar[dict[str, int]] = {}

    @classmethod
    @override
    def static_input_type(cls) -> Type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> Type[SimpleOutput]:
        return SimpleOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[SimpleOutput],
        input: Empty,
    ) -> SimpleOutput:
        YieldingNode.calls[self.id] = YieldingNode.calls.get(self.id, 0) + 1
        raise ShouldYield(self.message)


@pytest.fixture(autouse=True)
def reset_test_node_state():
    """Reset class-level state on shared test nodes before each test."""
    YieldingNode.calls = {}
    yield
