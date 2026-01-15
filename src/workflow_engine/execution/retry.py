# workflow_engine/execution/retry.py
"""
Retry tracking for workflow node execution.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..core.error import ShouldRetry


@dataclass
class NodeRetryState:
    """Tracks retry state for a single node during execution."""

    node_id: str
    attempt: int = 0
    next_retry_at: datetime | None = None
    last_error: ShouldRetry | None = None

    def schedule_retry(self, backoff: timedelta) -> None:
        """Schedule the next retry attempt."""
        self.attempt += 1
        self.next_retry_at = datetime.now() + backoff

    def is_ready(self) -> bool:
        """Check if the node is ready to retry (backoff expired)."""
        if self.next_retry_at is None:
            return True
        return datetime.now() >= self.next_retry_at

    def time_until_ready(self) -> timedelta:
        """Time remaining until retry is allowed."""
        if self.next_retry_at is None:
            return timedelta(0)
        remaining = self.next_retry_at - datetime.now()
        return max(remaining, timedelta(0))


@dataclass
class RetryTracker:
    """Tracks retry state for all nodes during workflow execution."""

    default_max_retries: int = 3
    states: dict[str, NodeRetryState] = field(default_factory=dict)

    def get_state(self, node_id: str) -> NodeRetryState:
        """Get or create retry state for a node."""
        if node_id not in self.states:
            self.states[node_id] = NodeRetryState(node_id=node_id)
        return self.states[node_id]

    def should_retry(self, node_id: str, node_max_retries: int | None) -> bool:
        """
        Check if the node has retries remaining.

        node_id: the node to check
        node_max_retries: the node-specific max retries, or None to use default
        """
        max_retries = (
            node_max_retries if node_max_retries is not None else self.default_max_retries
        )
        state = self.get_state(node_id)
        return state.attempt < max_retries

    def record_retry(self, node_id: str, error: ShouldRetry) -> None:
        """Record a retry attempt for a node."""
        state = self.get_state(node_id)
        state.last_error = error
        state.schedule_retry(error.backoff)

    def get_pending_retries(self) -> list[str]:
        """Get nodes that are waiting for retry (in backoff)."""
        return [
            node_id
            for node_id, state in self.states.items()
            if state.next_retry_at is not None and not state.is_ready()
        ]

    def min_wait_time(self) -> timedelta | None:
        """Minimum time to wait for any pending retry to become ready."""
        pending = [
            state.time_until_ready()
            for state in self.states.values()
            if state.next_retry_at is not None and not state.is_ready()
        ]
        return min(pending) if pending else None


__all__ = [
    "NodeRetryState",
    "RetryTracker",
]
