"""Local-process durable admission using short SQLite transactions.

The database is an operator-controlled local file, never a distributed-filesystem
lock service. Waiting coroutines consume no database connection or I/O thread.
"""

from __future__ import annotations

import asyncio
import math
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Self

from overrides import override

from ..core.limits import (
    InMemoryLimitCoordinator,
    LimitError,
    T,
    _complete_cleanup,
    _State,
)


class SQLiteLimitCoordinator(InMemoryLimitCoordinator):
    """Persist request history, FIFO tickets, policy revisions, and fenced leases."""

    def __init__(
        self,
        path: str | Path,
        *,
        lease_duration: float = 30,
        poll_interval: float = 0.1,
        busy_timeout: float = 5,
        clock: Callable[[], float] = time.time,
    ):
        if (
            not math.isfinite(lease_duration)
            or not math.isfinite(poll_interval)
            or lease_duration <= 0
            or not 0 < poll_interval <= lease_duration / 3
        ):
            raise ValueError(
                "Lease duration must be positive; polling must be at most one third of it."
            )
        if not math.isfinite(busy_timeout) or busy_timeout <= 0:
            raise ValueError("SQLite busy timeout must be positive.")
        super().__init__(clock=clock)
        self.path = Path(path).absolute()
        self.lease_duration = lease_duration
        self.poll_interval = poll_interval
        self.busy_timeout = busy_timeout
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="wengine-limits"
        )
        self._closed = False

    def _transaction(self, operation: Callable[[_State, float], T]) -> T:
        try:
            with (
                closing(
                    sqlite3.connect(self.path, timeout=self.busy_timeout)
                ) as connection,
                connection,
            ):
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS wengine_limit_state "
                    "(id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL, state TEXT NOT NULL)"
                )
                row = connection.execute(
                    "SELECT version, state FROM wengine_limit_state WHERE id = 1"
                ).fetchone()
                if row is not None and row[0] != 1:
                    raise LimitError.for_operator(
                        "Unsupported SQLite limit-store schema version."
                    )
                state = (
                    _State.model_validate_json(row[1]) if row is not None else _State()
                )
                now = max(self._clock(), state.clock_floor)
                state.clock_floor = now
                state.expire(now)
                result = operation(state, now)
                connection.execute(
                    "INSERT INTO wengine_limit_state (id, version, state) VALUES (1, 1, ?) "
                    "ON CONFLICT(id) DO UPDATE SET state = excluded.state",
                    (state.model_dump_json(),),
                )
                return result
        except LimitError:
            raise
        except Exception as error:
            raise LimitError.for_operator(
                "SQLite admission storage is unavailable or invalid."
            ) from error

    @override
    async def _mutate(self, operation: Callable[[_State, float], T]) -> T:
        if self._closed:
            raise LimitError.for_operator("SQLite admission coordinator is closed.")
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, self._transaction, operation)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # A transaction cannot be cancelled halfway through its commit.
            # Wait for its outcome so acquire's cancellation cleanup can remove
            # a grant that raced with task cancellation.
            try:
                await _complete_cleanup(future)
            except Exception:
                pass
            raise

    async def close(self) -> None:
        """Close only after contexts using this shared coordinator have settled."""
        self._closed = True
        await asyncio.to_thread(self._executor.shutdown, wait=True)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()
