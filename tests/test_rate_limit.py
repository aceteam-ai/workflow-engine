# tests/test_rate_limit.py
"""Tests for rate limiting in workflow execution."""

import asyncio
import time
from datetime import timedelta

import pytest
from pydantic import ValidationError

from workflow_engine import (
    Edge,
    OutputEdge,
    StringValue,
    Workflow,
)
from workflow_engine.contexts import InMemoryContext
from workflow_engine.execution import (
    RateLimitConfig,
    RateLimiter,
    RateLimitRegistry,
    TopologicalExecutionAlgorithm,
)
from workflow_engine.nodes import ConstantStringNode


# Unit tests for RateLimitConfig
class TestRateLimitConfig:
    @pytest.mark.unit
    def test_default_config(self):
        """Test that RateLimitConfig has correct defaults."""
        config = RateLimitConfig()
        assert config.max_concurrency is None
        assert config.requests_per_window is None
        assert config.window_duration == timedelta(seconds=60)

    @pytest.mark.unit
    def test_custom_config(self):
        """Test that RateLimitConfig accepts custom values."""
        config = RateLimitConfig(
            max_concurrency=5,
            requests_per_window=100,
            window_duration=timedelta(seconds=30),
        )
        assert config.max_concurrency == 5
        assert config.requests_per_window == 100
        assert config.window_duration == timedelta(seconds=30)

    @pytest.mark.unit
    def test_config_is_frozen(self):
        """Test that RateLimitConfig is immutable."""
        config = RateLimitConfig(max_concurrency=5)
        with pytest.raises(ValidationError):
            config.max_concurrency = 10  # type: ignore


# Unit tests for RateLimiter
class TestRateLimiter:
    @pytest.mark.unit
    def test_no_limits(self):
        """Test that RateLimiter with no limits doesn't block."""
        config = RateLimitConfig()
        limiter = RateLimiter(config)
        assert limiter._semaphore is None

    @pytest.mark.unit
    def test_concurrency_limit_creates_semaphore(self):
        """Test that concurrency limit creates a semaphore."""
        config = RateLimitConfig(max_concurrency=5)
        limiter = RateLimiter(config)
        assert limiter._semaphore is not None
        assert limiter._semaphore._value == 5  # type: ignore

    @pytest.mark.asyncio
    async def test_acquire_and_release(self):
        """Test basic acquire/release behavior."""
        config = RateLimitConfig(max_concurrency=2)
        limiter = RateLimiter(config)

        # Should be able to acquire twice
        await limiter.acquire()
        await limiter.acquire()

        # Release both
        limiter.release()
        limiter.release()

    @pytest.mark.asyncio
    async def test_concurrency_blocks_at_limit(self):
        """Test that concurrency limit blocks when at limit."""
        config = RateLimitConfig(max_concurrency=1)
        limiter = RateLimiter(config)

        await limiter.acquire()

        # Try to acquire again - should not complete immediately
        acquired = asyncio.Event()

        async def try_acquire():
            await limiter.acquire()
            acquired.set()

        task = asyncio.create_task(try_acquire())

        # Give time for the acquire to potentially complete
        await asyncio.sleep(0.01)
        assert not acquired.is_set(), "Should be blocked at concurrency limit"

        # Release and verify the acquire completes
        limiter.release()
        await asyncio.sleep(0.01)
        assert acquired.is_set(), "Should complete after release"

        # Cleanup
        limiter.release()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_rate_limit_request_tracking(self):
        """Test that request rate limit tracks requests."""
        config = RateLimitConfig(
            requests_per_window=3,
            window_duration=timedelta(seconds=1),
        )
        limiter = RateLimiter(config)

        # Should be able to make 3 requests quickly
        start = time.monotonic()
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should complete quickly (within tolerance)
        assert elapsed < 0.1, f"First 3 requests should be fast, took {elapsed}s"


# Unit tests for RateLimitRegistry
class TestRateLimitRegistry:
    @pytest.mark.unit
    def test_initial_state(self):
        """Test that RateLimitRegistry starts empty."""
        registry = RateLimitRegistry()
        assert registry.get_limiter("SomeNode") is None
        assert registry.get_config("SomeNode") is None

    @pytest.mark.unit
    def test_configure_and_retrieve(self):
        """Test that configured limiters can be retrieved."""
        registry = RateLimitRegistry()
        config = RateLimitConfig(max_concurrency=5)

        registry.configure("SomeNode", config)

        assert registry.get_config("SomeNode") == config
        assert registry.get_limiter("SomeNode") is not None

    @pytest.mark.unit
    def test_multiple_node_types(self):
        """Test that different node types have separate limiters."""
        registry = RateLimitRegistry()

        registry.configure("NodeA", RateLimitConfig(max_concurrency=5))
        registry.configure("NodeB", RateLimitConfig(max_concurrency=10))

        limiter_a = registry.get_limiter("NodeA")
        limiter_b = registry.get_limiter("NodeB")

        assert limiter_a is not None
        assert limiter_b is not None
        assert limiter_a is not limiter_b
        assert limiter_a._semaphore._value == 5  # type: ignore
        assert limiter_b._semaphore._value == 10  # type: ignore


# Integration tests for rate limiting with execution
class TestRateLimitIntegration:
    @pytest.mark.asyncio
    async def test_execution_with_rate_limit_registry(self):
        """Test that execution algorithm uses rate limit registry."""
        workflow = Workflow(
            nodes=[
                constant := ConstantStringNode.from_value(id="constant", value="test"),
            ],
            edges=[],
            input_edges=[],
            output_edges=[
                OutputEdge.from_node(
                    source=constant,
                    source_key="value",
                    output_key="result",
                ),
            ],
        )

        context = InMemoryContext()
        rate_limits = RateLimitRegistry()
        rate_limits.configure("ConstantString", RateLimitConfig(max_concurrency=1))

        algorithm = TopologicalExecutionAlgorithm(rate_limits=rate_limits)

        errors, output = await algorithm.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert errors.any() is False
        assert output == {"result": StringValue("test")}

    @pytest.mark.asyncio
    async def test_execution_without_rate_limits(self):
        """Test that execution works without rate limits configured."""
        workflow = Workflow(
            nodes=[
                constant := ConstantStringNode.from_value(id="constant", value="test"),
            ],
            edges=[],
            input_edges=[],
            output_edges=[
                OutputEdge.from_node(
                    source=constant,
                    source_key="value",
                    output_key="result",
                ),
            ],
        )

        context = InMemoryContext()
        algorithm = TopologicalExecutionAlgorithm()

        errors, output = await algorithm.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert errors.any() is False
        assert output == {"result": StringValue("test")}

    @pytest.mark.asyncio
    async def test_rate_limiter_released_on_error(self):
        """Test that rate limiter is released even when node fails."""
        from workflow_engine.nodes import ErrorNode

        workflow = Workflow(
            nodes=[
                constant := ConstantStringNode.from_value(id="constant", value="test"),
                error := ErrorNode.from_name(id="error", name="TestError"),
            ],
            edges=[
                Edge.from_nodes(
                    source=constant,
                    source_key="value",
                    target=error,
                    target_key="info",
                ),
            ],
            input_edges=[],
            output_edges=[
                OutputEdge.from_node(
                    source=constant,
                    source_key="value",
                    output_key="result",
                ),
            ],
        )

        context = InMemoryContext()
        rate_limits = RateLimitRegistry()
        config = RateLimitConfig(max_concurrency=1)
        rate_limits.configure("Error", config)

        algorithm = TopologicalExecutionAlgorithm(rate_limits=rate_limits)

        errors, _ = await algorithm.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert errors.any() is True

        # Verify the limiter was released (semaphore should be at max value)
        limiter = rate_limits.get_limiter("Error")
        assert limiter is not None
        assert limiter._semaphore._value == 1  # type: ignore
