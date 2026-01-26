# workflow_engine/execution/rate_limit.py
"""
Rate limiting for workflow node execution.

Supports both concurrency limits and request rate limits per node type.
"""

import asyncio
import time
from datetime import timedelta

from ..utils.immutable import ImmutableBaseModel


class RateLimitConfig(ImmutableBaseModel):
    """Configuration for rate limiting a node type."""

    max_concurrency: int | None = None
    """Maximum concurrent executions of this node type. None = unlimited."""

    requests_per_window: int | None = None
    """Maximum requests within the time window. None = unlimited."""

    window_duration: timedelta = timedelta(seconds=60)
    """Time window for rate limiting (default: 60 seconds)."""


class RateLimiter:
    """
    Rate limiter that supports both concurrency limits and request rate limits.

    Thread-safe for use with asyncio.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._semaphore: asyncio.Semaphore | None = None
        self._request_times: list[float] = []
        self._lock = asyncio.Lock()

        if config.max_concurrency is not None:
            self._semaphore = asyncio.Semaphore(config.max_concurrency)

    async def acquire(self) -> None:
        """
        Acquire permission to execute.

        May block if rate limited or at max concurrency.
        """
        # Check request rate limit first
        if self.config.requests_per_window is not None:
            await self._wait_for_rate_limit()

        # Acquire concurrency semaphore
        if self._semaphore is not None:
            await self._semaphore.acquire()

    def release(self) -> None:
        """Release after execution completes."""
        if self._semaphore is not None:
            self._semaphore.release()

    async def _wait_for_rate_limit(self) -> None:
        """Wait if we've exceeded the request rate limit."""
        async with self._lock:
            window_seconds = self.config.window_duration.total_seconds()
            window_start = time.monotonic() - window_seconds

            # Clean old request times
            self._request_times = [t for t in self._request_times if t > window_start]

            # Check if we're at the limit
            if (
                self.config.requests_per_window is not None
                and len(self._request_times) >= self.config.requests_per_window
            ):
                # Need to wait until oldest request falls outside window
                oldest = self._request_times[0]
                wait_time = oldest - window_start
                if wait_time > 0:
                    # Release lock while sleeping
                    self._lock.release()
                    try:
                        await asyncio.sleep(wait_time)
                    finally:
                        await self._lock.acquire()
                    # Re-clean after waiting
                    window_start = time.monotonic() - window_seconds
                    self._request_times = [
                        t for t in self._request_times if t > window_start
                    ]

            # Record this request
            self._request_times.append(time.monotonic())


class RateLimitRegistry:
    """Registry of rate limiters by node type."""

    def __init__(self) -> None:
        self._configs: dict[str, RateLimitConfig] = {}
        self._limiters: dict[str, RateLimiter] = {}

    def configure(self, node_type: str, config: RateLimitConfig) -> None:
        """Configure rate limits for a node type."""
        self._configs[node_type] = config
        self._limiters[node_type] = RateLimiter(config)

    def get_limiter(self, node_type: str) -> RateLimiter | None:
        """Get the rate limiter for a node type, or None if not configured."""
        return self._limiters.get(node_type)

    def get_config(self, node_type: str) -> RateLimitConfig | None:
        """Get the rate limit config for a node type, or None if not configured."""
        return self._configs.get(node_type)


__all__ = [
    "RateLimitConfig",
    "RateLimiter",
    "RateLimitRegistry",
]
