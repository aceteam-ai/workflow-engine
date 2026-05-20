# tests/conftest.py
"""Shared test fixtures and utilities.

Note: defining a Node subclass here causes it to auto-register on the default
node registry, so anything declared here is available to every test file
without needing to be re-imported.
"""

from pathlib import Path
from typing import Callable, ClassVar, Type

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


@pytest.fixture
def confine_is_file_to(monkeypatch) -> Callable[[Path], None]:
    """Confine `Path.is_file()` to a subtree, hiding everything outside it.

    CLI helpers that walk up the filesystem looking for marker files
    (`engine.yaml`, `pyproject.toml`, ...) ascend all the way to the root, so
    a stray file in a real parent directory (e.g. `/tmp` or a CI home dir) can
    make "not found" assertions flaky. Call the returned function with a root
    (typically `tmp_path`): paths at or below it delegate to the real
    `is_file`, everything else reports `False`.
    """
    real_is_file = Path.is_file

    def confine(root: Path) -> None:
        resolved_root = root.resolve()

        def fake(self: Path) -> bool:
            resolved = self.resolve()
            if resolved != resolved_root and resolved_root not in resolved.parents:
                return False
            return real_is_file(self)

        monkeypatch.setattr(Path, "is_file", fake)

    return confine


@pytest.fixture(autouse=True)
def reset_test_node_state():
    """Reset class-level state on shared test nodes before each test."""
    YieldingNode.calls = {}
    yield
