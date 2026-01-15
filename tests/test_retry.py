# tests/test_retry.py
"""Tests for retry behavior in workflow execution."""

from datetime import timedelta
from typing import ClassVar, Literal
from unittest.mock import AsyncMock

import pytest

from workflow_engine import (
    Context,
    Data,
    Edge,
    Node,
    OutputEdge,
    Params,
    ShouldRetry,
    StringValue,
    Workflow,
)
from workflow_engine.core import NodeTypeInfo
from workflow_engine.contexts import InMemoryContext
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.execution.retry import NodeRetryState, RetryTracker


# Test node that raises ShouldRetry a configurable number of times
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
        name="Retryable",
        display_name="Retryable",
        description="A node that fails a configurable number of times.",
        version="0.4.0",
        parameter_type=RetryableParams,
    )

    type: Literal["Retryable"] = "Retryable"  # pyright: ignore[reportIncompatibleVariableOverride]
    _attempt_counts: ClassVar[dict[str, int]] = {}

    @property
    def input_type(self):
        return RetryableInput

    @property
    def output_type(self):
        return RetryableOutput

    async def run(self, context: Context, input: RetryableInput) -> RetryableOutput:
        # Track attempts per node id
        if self.id not in RetryableNode._attempt_counts:
            RetryableNode._attempt_counts[self.id] = 0
        RetryableNode._attempt_counts[self.id] += 1

        if RetryableNode._attempt_counts[self.id] <= self.params.fail_count:
            raise ShouldRetry(
                message=f"Temporary failure (attempt {RetryableNode._attempt_counts[self.id]})",
                backoff=timedelta(milliseconds=10),  # Short backoff for tests
            )

        return RetryableOutput(result=StringValue(f"Success: {input.value.root}"))

    @classmethod
    def from_fail_count(cls, id: str, fail_count: int) -> "RetryableNode":
        return cls(id=id, params=RetryableParams(fail_count=fail_count))


# Node with custom max_retries in TYPE_INFO
class CustomRetryNode(Node[RetryableInput, RetryableOutput, RetryableParams]):
    """A node with custom max_retries configured in TYPE_INFO."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="CustomRetry",
        display_name="Custom Retry",
        description="A node with custom max_retries.",
        version="0.4.0",
        parameter_type=RetryableParams,
        max_retries=5,
    )

    type: Literal["CustomRetry"] = "CustomRetry"  # pyright: ignore[reportIncompatibleVariableOverride]
    _attempt_counts: ClassVar[dict[str, int]] = {}

    @property
    def input_type(self):
        return RetryableInput

    @property
    def output_type(self):
        return RetryableOutput

    async def run(self, context: Context, input: RetryableInput) -> RetryableOutput:
        if self.id not in CustomRetryNode._attempt_counts:
            CustomRetryNode._attempt_counts[self.id] = 0
        CustomRetryNode._attempt_counts[self.id] += 1

        if CustomRetryNode._attempt_counts[self.id] <= self.params.fail_count:
            raise ShouldRetry(
                message=f"Temporary failure (attempt {CustomRetryNode._attempt_counts[self.id]})",
                backoff=timedelta(milliseconds=10),
            )

        return RetryableOutput(result=StringValue(f"Success: {input.value.root}"))

    @classmethod
    def from_fail_count(cls, id: str, fail_count: int) -> "CustomRetryNode":
        return cls(id=id, params=RetryableParams(fail_count=fail_count))


@pytest.fixture(autouse=True)
def reset_attempt_counts():
    """Reset attempt counts before each test."""
    RetryableNode._attempt_counts = {}
    CustomRetryNode._attempt_counts = {}
    yield


# Unit tests for ShouldRetry exception
class TestShouldRetry:
    @pytest.mark.unit
    def test_should_retry_default_backoff(self):
        """Test that ShouldRetry has default backoff of 1 second."""
        exc = ShouldRetry("test error")
        assert exc.message == "test error"
        assert exc.backoff == timedelta(seconds=1)

    @pytest.mark.unit
    def test_should_retry_custom_backoff(self):
        """Test that ShouldRetry accepts custom backoff."""
        exc = ShouldRetry("test error", backoff=timedelta(seconds=30))
        assert exc.message == "test error"
        assert exc.backoff == timedelta(seconds=30)

    @pytest.mark.unit
    def test_should_retry_inherits_from_user_exception(self):
        """Test that ShouldRetry inherits from UserException."""
        from workflow_engine import UserException

        exc = ShouldRetry("test error")
        assert isinstance(exc, UserException)


# Unit tests for RetryTracker
class TestRetryTracker:
    @pytest.mark.unit
    def test_initial_state(self):
        """Test that RetryTracker starts with empty state."""
        tracker = RetryTracker(default_max_retries=3)
        assert tracker.states == {}
        assert tracker.default_max_retries == 3

    @pytest.mark.unit
    def test_should_retry_within_limit(self):
        """Test that should_retry returns True when under max retries."""
        tracker = RetryTracker(default_max_retries=3)
        assert tracker.should_retry("node1", None) is True

        # Record first retry
        tracker.record_retry("node1", ShouldRetry("error", timedelta(seconds=1)))
        assert tracker.should_retry("node1", None) is True

        # Record second retry
        tracker.record_retry("node1", ShouldRetry("error", timedelta(seconds=1)))
        assert tracker.should_retry("node1", None) is True

    @pytest.mark.unit
    def test_should_retry_at_limit(self):
        """Test that should_retry returns False at max retries."""
        tracker = RetryTracker(default_max_retries=2)

        tracker.record_retry("node1", ShouldRetry("error", timedelta(seconds=1)))
        tracker.record_retry("node1", ShouldRetry("error", timedelta(seconds=1)))

        assert tracker.should_retry("node1", None) is False

    @pytest.mark.unit
    def test_node_specific_max_retries(self):
        """Test that node-specific max_retries overrides default."""
        tracker = RetryTracker(default_max_retries=2)

        tracker.record_retry("node1", ShouldRetry("error", timedelta(seconds=1)))
        tracker.record_retry("node1", ShouldRetry("error", timedelta(seconds=1)))

        # With default (2), should not retry
        assert tracker.should_retry("node1", None) is False

        # With node-specific override (5), should retry
        assert tracker.should_retry("node1", 5) is True


# Unit tests for NodeRetryState
class TestNodeRetryState:
    @pytest.mark.unit
    def test_initial_state(self):
        """Test that NodeRetryState starts correctly."""
        state = NodeRetryState(node_id="node1")
        assert state.node_id == "node1"
        assert state.attempt == 0
        assert state.next_retry_at is None
        assert state.last_error is None
        assert state.is_ready() is True

    @pytest.mark.unit
    def test_schedule_retry(self):
        """Test that schedule_retry updates state correctly."""
        state = NodeRetryState(node_id="node1")
        state.schedule_retry(timedelta(seconds=10))

        assert state.attempt == 1
        assert state.next_retry_at is not None
        assert state.is_ready() is False

    @pytest.mark.unit
    def test_time_until_ready(self):
        """Test that time_until_ready returns correct value."""
        state = NodeRetryState(node_id="node1")

        # Initially ready, so no wait time
        assert state.time_until_ready() == timedelta(0)

        # After scheduling, has wait time
        state.schedule_retry(timedelta(seconds=10))
        time_remaining = state.time_until_ready()
        assert time_remaining > timedelta(0)
        assert time_remaining <= timedelta(seconds=10)


# Integration tests for retry behavior
class TestRetryIntegration:
    @pytest.mark.asyncio
    async def test_retry_succeeds_within_limit(self):
        """Test that a node succeeds after retrying within the limit."""
        from workflow_engine.nodes import ConstantStringNode

        workflow = Workflow(
            nodes=[
                constant := ConstantStringNode.from_value(id="constant", value="input"),
                retryable := RetryableNode.from_fail_count(id="retryable", fail_count=2),
            ],
            edges=[
                Edge.from_nodes(
                    source=constant,
                    source_key="value",
                    target=retryable,
                    target_key="value",
                ),
            ],
            input_edges=[],
            output_edges=[
                OutputEdge.from_node(
                    source=retryable,
                    source_key="result",
                    output_key="result",
                ),
            ],
        )

        context = InMemoryContext()
        algorithm = TopologicalExecutionAlgorithm(max_retries=3)

        errors, output = await algorithm.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert errors.any() is False
        assert output == {"result": StringValue("Success: input")}
        assert RetryableNode._attempt_counts["retryable"] == 3  # 2 failures + 1 success

    @pytest.mark.asyncio
    async def test_retry_fails_at_limit(self):
        """Test that a node fails after exhausting retries."""
        from workflow_engine.nodes import ConstantStringNode

        workflow = Workflow(
            nodes=[
                constant := ConstantStringNode.from_value(id="constant", value="input"),
                retryable := RetryableNode.from_fail_count(id="retryable", fail_count=5),
            ],
            edges=[
                Edge.from_nodes(
                    source=constant,
                    source_key="value",
                    target=retryable,
                    target_key="value",
                ),
            ],
            input_edges=[],
            output_edges=[
                OutputEdge.from_node(
                    source=retryable,
                    source_key="result",
                    output_key="result",
                ),
            ],
        )

        context = InMemoryContext()
        algorithm = TopologicalExecutionAlgorithm(max_retries=3)

        errors, output = await algorithm.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert errors.any() is True
        assert "retryable" in errors.node_errors
        # Should have tried 3 times (max_retries) + initial attempt
        assert RetryableNode._attempt_counts["retryable"] == 4

    @pytest.mark.asyncio
    async def test_on_node_retry_hook_called(self):
        """Test that the on_node_retry hook is called."""
        from workflow_engine.nodes import ConstantStringNode

        workflow = Workflow(
            nodes=[
                constant := ConstantStringNode.from_value(id="constant", value="input"),
                retryable := RetryableNode.from_fail_count(id="retryable", fail_count=2),
            ],
            edges=[
                Edge.from_nodes(
                    source=constant,
                    source_key="value",
                    target=retryable,
                    target_key="value",
                ),
            ],
            input_edges=[],
            output_edges=[
                OutputEdge.from_node(
                    source=retryable,
                    source_key="result",
                    output_key="result",
                ),
            ],
        )

        context = InMemoryContext()
        mock_on_node_retry = AsyncMock()
        context.on_node_retry = mock_on_node_retry

        algorithm = TopologicalExecutionAlgorithm(max_retries=3)

        await algorithm.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        # Should have been called twice (for 2 retries)
        assert mock_on_node_retry.call_count == 2

        # Check the first call arguments
        first_call = mock_on_node_retry.call_args_list[0]
        assert first_call.kwargs["node"].id == "retryable"
        assert isinstance(first_call.kwargs["exception"], ShouldRetry)
        assert first_call.kwargs["attempt"] == 1

        # Check the second call
        second_call = mock_on_node_retry.call_args_list[1]
        assert second_call.kwargs["attempt"] == 2

    @pytest.mark.asyncio
    async def test_node_type_max_retries_override(self):
        """Test that NodeTypeInfo.max_retries overrides algorithm default."""
        from workflow_engine.nodes import ConstantStringNode

        workflow = Workflow(
            nodes=[
                constant := ConstantStringNode.from_value(id="constant", value="input"),
                custom := CustomRetryNode.from_fail_count(id="custom", fail_count=4),
            ],
            edges=[
                Edge.from_nodes(
                    source=constant,
                    source_key="value",
                    target=custom,
                    target_key="value",
                ),
            ],
            input_edges=[],
            output_edges=[
                OutputEdge.from_node(
                    source=custom,
                    source_key="result",
                    output_key="result",
                ),
            ],
        )

        context = InMemoryContext()
        # Algorithm default is 2, but CustomRetryNode.TYPE_INFO.max_retries is 5
        algorithm = TopologicalExecutionAlgorithm(max_retries=2)

        errors, output = await algorithm.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        # Should succeed because node-specific max_retries (5) > fail_count (4)
        assert errors.any() is False
        assert output == {"result": StringValue("Success: input")}
        assert CustomRetryNode._attempt_counts["custom"] == 5  # 4 failures + 1 success

    @pytest.mark.asyncio
    async def test_retry_with_rate_limiting(self):
        """Test that retry and rate limiting work together correctly."""
        from workflow_engine.nodes import ConstantStringNode
        from workflow_engine.execution import RateLimitConfig, RateLimitRegistry

        workflow = Workflow(
            nodes=[
                constant := ConstantStringNode.from_value(id="constant", value="input"),
                retryable := RetryableNode.from_fail_count(id="retryable", fail_count=1),
            ],
            edges=[
                Edge.from_nodes(
                    source=constant,
                    source_key="value",
                    target=retryable,
                    target_key="value",
                ),
            ],
            input_edges=[],
            output_edges=[
                OutputEdge.from_node(
                    source=retryable,
                    source_key="result",
                    output_key="result",
                ),
            ],
        )

        context = InMemoryContext()

        # Configure rate limiting for the retryable node
        rate_limits = RateLimitRegistry()
        rate_limits.configure("Retryable", RateLimitConfig(max_concurrency=1))

        algorithm = TopologicalExecutionAlgorithm(max_retries=3, rate_limits=rate_limits)

        errors, output = await algorithm.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        # Should succeed after 1 retry
        assert errors.any() is False
        assert output == {"result": StringValue("Success: input")}
        assert RetryableNode._attempt_counts["retryable"] == 2  # 1 failure + 1 success

        # Verify rate limiter was properly released (can acquire again)
        limiter = rate_limits.get_limiter("Retryable")
        assert limiter is not None
        assert limiter._semaphore._value == 1  # Back to max value

    @pytest.mark.asyncio
    async def test_multiple_retryable_nodes_in_sequence(self):
        """Test workflow with multiple nodes that can retry in sequence."""
        from workflow_engine.nodes import ConstantStringNode

        # Create a second retryable node type to track separately
        class RetryableNode2(Node[RetryableInput, RetryableOutput, RetryableParams]):
            TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
                name="Retryable2",
                display_name="Retryable2",
                description="Another retryable node.",
                version="0.4.0",
                parameter_type=RetryableParams,
            )
            type: Literal["Retryable2"] = "Retryable2"  # pyright: ignore[reportIncompatibleVariableOverride]
            _attempt_counts: ClassVar[dict[str, int]] = {}

            @property
            def input_type(self):
                return RetryableInput

            @property
            def output_type(self):
                return RetryableOutput

            async def run(self, context: Context, input: RetryableInput) -> RetryableOutput:
                if self.id not in RetryableNode2._attempt_counts:
                    RetryableNode2._attempt_counts[self.id] = 0
                RetryableNode2._attempt_counts[self.id] += 1

                if RetryableNode2._attempt_counts[self.id] <= self.params.fail_count:
                    raise ShouldRetry(
                        message=f"Temporary failure (attempt {RetryableNode2._attempt_counts[self.id]})",
                        backoff=timedelta(milliseconds=10),
                    )
                return RetryableOutput(result=StringValue(f"Node2: {input.value.root}"))

            @classmethod
            def from_fail_count(cls, id: str, fail_count: int) -> "RetryableNode2":
                return cls(id=id, params=RetryableParams(fail_count=fail_count))

        # Reset counts
        RetryableNode2._attempt_counts = {}

        workflow = Workflow(
            nodes=[
                constant := ConstantStringNode.from_value(id="constant", value="start"),
                node1 := RetryableNode.from_fail_count(id="node1", fail_count=1),
                node2 := RetryableNode2.from_fail_count(id="node2", fail_count=1),
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
            ],
            input_edges=[],
            output_edges=[
                OutputEdge.from_node(
                    source=node2,
                    source_key="result",
                    output_key="final_result",
                ),
            ],
        )

        context = InMemoryContext()
        algorithm = TopologicalExecutionAlgorithm(max_retries=3)

        errors, output = await algorithm.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert errors.any() is False
        assert output == {"final_result": StringValue("Node2: Success: start")}
        # Both nodes should have retried once
        assert RetryableNode._attempt_counts["node1"] == 2
        assert RetryableNode2._attempt_counts["node2"] == 2
