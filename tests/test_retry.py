# tests/test_retry.py
"""Tests for retry behavior in workflow execution."""

from datetime import timedelta
from typing import ClassVar, Type, override
from unittest.mock import AsyncMock

import pytest

from workflow_engine import (
    Data,
    Edge,
    ExecutionContext,
    Node,
    NodeTypeInfo,
    Params,
    ShouldRetry,
    StringValue,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.execution.retry import NodeRetryState, RetryTracker
from workflow_engine.execution.topological import TopologicalExecutionAlgorithm
from workflow_engine.nodes import ConstantStringNode


class RetryableInput(Data):
    value: StringValue


class RetryableOutput(Data):
    result: StringValue


class RetryableParams(Params):
    fail_count: int
    """Number of times to fail before succeeding."""


class RetryableNode(Node[RetryableInput, RetryableOutput, RetryableParams]):
    """A node that fails a configurable number of times before succeeding."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Retryable",
        description="A node that fails a configurable number of times.",
        version="0.4.0",
        parameter_type=RetryableParams,
    )

    _attempt_counts: ClassVar[dict[str, int]] = {}

    @classmethod
    @override
    def static_input_type(cls) -> Type[RetryableInput]:
        return RetryableInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[RetryableOutput]:
        return RetryableOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[RetryableInput],
        output_type: Type[RetryableOutput],
        input: RetryableInput,
    ) -> RetryableOutput:
        if self.id not in RetryableNode._attempt_counts:
            RetryableNode._attempt_counts[self.id] = 0
        RetryableNode._attempt_counts[self.id] += 1

        if RetryableNode._attempt_counts[self.id] <= self.params.fail_count:
            raise ShouldRetry.for_user(
                f"Temporary failure (attempt {RetryableNode._attempt_counts[self.id]})",
                node=self,
                backoff=timedelta(milliseconds=10),
            )

        return RetryableOutput(result=StringValue(f"Success: {input.value.root}"))


class RetryableNode2(Node[RetryableInput, RetryableOutput, RetryableParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Retryable2",
        description="Another retryable node.",
        version="0.4.0",
        parameter_type=RetryableParams,
    )
    _attempt_counts: ClassVar[dict[str, int]] = {}

    @classmethod
    @override
    def static_input_type(cls) -> Type[RetryableInput]:
        return RetryableInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[RetryableOutput]:
        return RetryableOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[RetryableInput],
        output_type: Type[RetryableOutput],
        input: RetryableInput,
    ) -> RetryableOutput:
        if self.id not in RetryableNode2._attempt_counts:
            RetryableNode2._attempt_counts[self.id] = 0
        RetryableNode2._attempt_counts[self.id] += 1

        if RetryableNode2._attempt_counts[self.id] <= self.params.fail_count:
            raise ShouldRetry.for_user(
                f"Temporary failure (attempt {RetryableNode2._attempt_counts[self.id]})",
                node=self,
                backoff=timedelta(milliseconds=10),
            )
        return RetryableOutput(result=StringValue(f"Node2: {input.value.root}"))


class CustomRetryNode(Node[RetryableInput, RetryableOutput, RetryableParams]):
    """A node with custom max_retries configured in TYPE_INFO."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Custom Retry",
        description="A node with custom max_retries.",
        version="0.4.0",
        parameter_type=RetryableParams,
        max_retries=5,
    )

    _attempt_counts: ClassVar[dict[str, int]] = {}

    @classmethod
    @override
    def static_input_type(cls) -> Type[RetryableInput]:
        return RetryableInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[RetryableOutput]:
        return RetryableOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[RetryableInput],
        output_type: Type[RetryableOutput],
        input: RetryableInput,
    ) -> RetryableOutput:
        if self.id not in CustomRetryNode._attempt_counts:
            CustomRetryNode._attempt_counts[self.id] = 0
        CustomRetryNode._attempt_counts[self.id] += 1

        if CustomRetryNode._attempt_counts[self.id] <= self.params.fail_count:
            raise ShouldRetry.for_user(
                f"Temporary failure (attempt {CustomRetryNode._attempt_counts[self.id]})",
                node=self,
                backoff=timedelta(milliseconds=10),
            )

        return RetryableOutput(result=StringValue(f"Success: {input.value.root}"))


@pytest.fixture(autouse=True)
def reset_attempt_counts():
    """Reset attempt counts before each test."""
    RetryableNode._attempt_counts = {}
    RetryableNode2._attempt_counts = {}
    CustomRetryNode._attempt_counts = {}
    yield


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine(
        execution_algorithm=TopologicalExecutionAlgorithm(max_retries=3)
    )


# Unit tests for RetryTracker
class TestRetryTracker:
    @pytest.fixture
    def node(self, engine: WorkflowEngine) -> Node:
        return engine.create_node(RetryableNode, id="node1", params=dict(fail_count=0))

    @pytest.mark.unit
    def test_initial_state(self):
        """Test that RetryTracker starts with empty state."""
        tracker = RetryTracker(default_max_retries=3)
        assert tracker.states == {}
        assert tracker.default_max_retries == 3

    @pytest.mark.unit
    def test_should_retry_within_limit(self, node: Node):
        """Test that should_retry returns True when under max retries."""
        tracker = RetryTracker(default_max_retries=3)
        assert tracker.should_retry(node.id, None) is True

        tracker.record_retry(
            node.id,
            ShouldRetry.for_user(
                "error",
                node=node,
                backoff=timedelta(seconds=1),
            ),
        )
        assert tracker.should_retry(node.id, None) is True

        tracker.record_retry(
            node.id,
            ShouldRetry.for_user(
                "error",
                node=node,
                backoff=timedelta(seconds=1),
            ),
        )
        assert tracker.should_retry(node.id, None) is True

    @pytest.mark.unit
    def test_should_retry_at_limit(self, node: Node):
        """Test that should_retry returns False at max retries."""
        tracker = RetryTracker(default_max_retries=2)

        tracker.record_retry(
            node.id,
            ShouldRetry.for_user(
                "error",
                node=node,
                backoff=timedelta(seconds=1),
            ),
        )
        tracker.record_retry(
            node.id,
            ShouldRetry.for_user(
                "error",
                node=node,
                backoff=timedelta(seconds=1),
            ),
        )

        assert tracker.should_retry(node.id, None) is False

    @pytest.mark.unit
    def test_node_specific_max_retries(self, node: Node):
        """Test that node-specific max_retries overrides default."""
        tracker = RetryTracker(default_max_retries=2)

        tracker.record_retry(
            node.id,
            ShouldRetry.for_user(
                "error",
                node=node,
                backoff=timedelta(seconds=1),
            ),
        )
        tracker.record_retry(
            node.id,
            ShouldRetry.for_user(
                "error",
                node=node,
                backoff=timedelta(seconds=1),
            ),
        )

        assert tracker.should_retry(node.id, None) is False
        assert tracker.should_retry(node.id, 5) is True


class TestNodeRetryState:
    @pytest.mark.unit
    def test_initial_state(self):
        state = NodeRetryState(node_id="node1")
        assert state.node_id == "node1"
        assert state.attempt == 0
        assert state.next_retry_at is None
        assert state.last_error is None
        assert state.is_ready() is True

    @pytest.mark.unit
    def test_schedule_retry(self):
        state = NodeRetryState(node_id="node1")
        state.schedule_retry(timedelta(seconds=10))

        assert state.attempt == 1
        assert state.next_retry_at is not None
        assert state.is_ready() is False

    @pytest.mark.unit
    def test_time_until_ready(self):
        state = NodeRetryState(node_id="node1")

        assert state.time_until_ready() == timedelta(0)

        state.schedule_retry(timedelta(seconds=10))
        time_remaining = state.time_until_ready()
        assert time_remaining > timedelta(0)
        assert time_remaining <= timedelta(seconds=10)


class TestRetryIntegration:
    @pytest.mark.asyncio
    async def test_retry_succeeds_within_limit(self, engine: WorkflowEngine):
        """Test that a node succeeds after retrying within the limit."""
        workflow = Workflow(
            input_node=engine.create_input_node(),
            output_node=(output_node := engine.create_output_node(result=StringValue)),
            inner_nodes=[
                constant := engine.create_node(
                    ConstantStringNode, id="constant", params=dict(value="input")
                ),
                retryable := engine.create_node(
                    RetryableNode, id="retryable", params=dict(fail_count=2)
                ),
            ],
            edges=[
                Edge.from_nodes(
                    source=constant,
                    source_key="value",
                    target=retryable,
                    target_key="value",
                ),
                Edge.from_nodes(
                    source=retryable,
                    source_key="result",
                    target=output_node,
                    target_key="result",
                ),
            ],
        )

        context = InMemoryExecutionContext()
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.output == {"result": StringValue("Success: input")}
        assert RetryableNode._attempt_counts["retryable"] == 3

    @pytest.mark.asyncio
    async def test_retry_fails_at_limit(self, engine: WorkflowEngine):
        """Test that a node fails after exhausting retries."""
        workflow = Workflow(
            input_node=engine.create_input_node(),
            output_node=(output_node := engine.create_output_node(result=StringValue)),
            inner_nodes=[
                constant := engine.create_node(
                    ConstantStringNode, id="constant", params=dict(value="input")
                ),
                retryable := engine.create_node(
                    RetryableNode, id="retryable", params=dict(fail_count=5)
                ),
            ],
            edges=[
                Edge.from_nodes(
                    source=constant,
                    source_key="value",
                    target=retryable,
                    target_key="value",
                ),
                Edge.from_nodes(
                    source=retryable,
                    source_key="result",
                    target=output_node,
                    target_key="result",
                ),
            ],
        )

        context = InMemoryExecutionContext()
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.ERROR
        assert "retryable" in result.errors.node_errors
        assert RetryableNode._attempt_counts["retryable"] == 4

    @pytest.mark.asyncio
    async def test_on_node_retry_hook_called(self, engine: WorkflowEngine):
        """Test that the on_node_retry hook is called."""
        workflow = Workflow(
            input_node=engine.create_input_node(),
            output_node=(output_node := engine.create_output_node(result=StringValue)),
            inner_nodes=[
                constant := engine.create_node(
                    ConstantStringNode, id="constant", params=dict(value="input")
                ),
                retryable := engine.create_node(
                    RetryableNode, id="retryable", params=dict(fail_count=2)
                ),
            ],
            edges=[
                Edge.from_nodes(
                    source=constant,
                    source_key="value",
                    target=retryable,
                    target_key="value",
                ),
                Edge.from_nodes(
                    source=retryable,
                    source_key="result",
                    target=output_node,
                    target_key="result",
                ),
            ],
        )

        context = InMemoryExecutionContext()
        mock_on_node_retry = AsyncMock()
        context.on_node_retry = mock_on_node_retry
        _ = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert mock_on_node_retry.call_count == 2

        first_call = mock_on_node_retry.call_args_list[0]
        assert first_call.kwargs["node"].id == "retryable"
        assert isinstance(first_call.kwargs["exception"], ShouldRetry)
        assert first_call.kwargs["attempt"] == 1

        second_call = mock_on_node_retry.call_args_list[1]
        assert second_call.kwargs["attempt"] == 2

    @pytest.mark.asyncio
    async def test_node_type_max_retries_override(self):
        """Test that NodeTypeInfo.max_retries overrides algorithm default."""
        # Algorithm default is 2, but CustomRetryNode.TYPE_INFO.max_retries is 5
        engine = WorkflowEngine(
            execution_algorithm=TopologicalExecutionAlgorithm(max_retries=2)
        )

        workflow = Workflow(
            input_node=engine.create_input_node(),
            output_node=(output_node := engine.create_output_node(result=StringValue)),
            inner_nodes=[
                constant := engine.create_node(
                    ConstantStringNode, id="constant", params=dict(value="input")
                ),
                custom := engine.create_node(
                    CustomRetryNode, id="custom", params=dict(fail_count=4)
                ),
            ],
            edges=[
                Edge.from_nodes(
                    source=constant,
                    source_key="value",
                    target=custom,
                    target_key="value",
                ),
                Edge.from_nodes(
                    source=custom,
                    source_key="result",
                    target=output_node,
                    target_key="result",
                ),
            ],
        )

        context = InMemoryExecutionContext()
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.output == {"result": StringValue("Success: input")}
        assert CustomRetryNode._attempt_counts["custom"] == 5

    @pytest.mark.asyncio
    async def test_retry_with_rate_limiting(self):
        """Test that retry and rate limiting work together correctly."""
        from workflow_engine.execution import RateLimitConfig, RateLimitRegistry

        rate_limits = RateLimitRegistry()
        rate_limits.configure("Retryable", RateLimitConfig(max_concurrency=1))
        engine = WorkflowEngine(
            execution_algorithm=TopologicalExecutionAlgorithm(
                max_retries=3,
                rate_limits=rate_limits,
            )
        )

        workflow = Workflow(
            input_node=engine.create_input_node(),
            output_node=(output_node := engine.create_output_node(result=StringValue)),
            inner_nodes=[
                constant := engine.create_node(
                    ConstantStringNode, id="constant", params=dict(value="input")
                ),
                retryable := engine.create_node(
                    RetryableNode, id="retryable", params=dict(fail_count=1)
                ),
            ],
            edges=[
                Edge.from_nodes(
                    source=constant,
                    source_key="value",
                    target=retryable,
                    target_key="value",
                ),
                Edge.from_nodes(
                    source=retryable,
                    source_key="result",
                    target=output_node,
                    target_key="result",
                ),
            ],
        )

        context = InMemoryExecutionContext()
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.output == {"result": StringValue("Success: input")}
        assert RetryableNode._attempt_counts["retryable"] == 2

        limiter = rate_limits.get_limiter("Retryable")
        assert limiter is not None
        assert limiter._semaphore is not None
        assert limiter._semaphore._value == 1

    @pytest.mark.asyncio
    async def test_multiple_retryable_nodes_in_sequence(self, engine: WorkflowEngine):
        """Test workflow with multiple nodes that can retry in sequence."""
        workflow = Workflow(
            input_node=engine.create_input_node(),
            output_node=(
                output_node := engine.create_output_node(final_result=StringValue)
            ),
            inner_nodes=[
                constant := engine.create_node(
                    ConstantStringNode, id="constant", params=dict(value="start")
                ),
                node1 := engine.create_node(
                    RetryableNode, id="node1", params=dict(fail_count=1)
                ),
                node2 := engine.create_node(
                    RetryableNode2, id="node2", params=dict(fail_count=1)
                ),
            ],
            edges=[
                Edge.from_nodes(
                    source=constant,
                    source_key="value",
                    target=node1,
                    target_key="value",
                ),
                Edge.from_nodes(
                    source=node1,
                    source_key="result",
                    target=node2,
                    target_key="value",
                ),
                Edge.from_nodes(
                    source=node2,
                    source_key="result",
                    target=output_node,
                    target_key="final_result",
                ),
            ],
        )

        context = InMemoryExecutionContext()
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.output == {"final_result": StringValue("Node2: Success: start")}
        assert RetryableNode._attempt_counts["node1"] == 2
        assert RetryableNode2._attempt_counts["node2"] == 2
