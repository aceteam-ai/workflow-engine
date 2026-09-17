"""Atomic quota admission shared by contexts, algorithms, and event loops."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from datetime import timedelta
from typing import Protocol, TypeVar

from pydantic import BaseModel, Field

from ..utils.model import ImmutableBaseModel
from .error import WorkflowException


class RateLimitConfig(ImmutableBaseModel):
    max_concurrency: int | None = Field(default=None, gt=0)
    requests_per_window: int | None = Field(default=None, gt=0)
    window_duration: timedelta = Field(default=timedelta(seconds=60), gt=timedelta(0))


class LimitRequest(ImmutableBaseModel):
    """A host-resolved pool and its complete, versioned policy."""

    pool: str
    config: RateLimitConfig
    revision: str = "1"


class LimitPolicy(ImmutableBaseModel):
    """Complete overrides keyed by ``module.qualname:region``; never graph hints."""

    overrides: Mapping[str, RateLimitConfig] = Field(default_factory=dict)
    pools: Mapping[str, str] = Field(default_factory=dict)
    namespace: str = "default"
    revision: str = "1"

    @staticmethod
    def scope(node_type: type, region: str = "execution") -> str:
        return f"{node_type.__module__}.{node_type.__qualname__}:{region}"

    def resolve(
        self, node_type: type, region: str, default: RateLimitConfig | None
    ) -> LimitRequest | None:
        scope = self.scope(node_type, region)
        config = self.overrides.get(scope, default)
        if config is None:
            return None
        return LimitRequest(
            pool=f"{self.namespace}:{self.pools.get(scope, scope)}",
            config=config,
            revision=self.revision,
        )


class LimitLease(ImmutableBaseModel):
    id: str
    invocation_id: str
    requests: tuple[LimitRequest, ...]
    generation: int
    expires_at: float | None = None


class LimitWaitInfo(ImmutableBaseModel):
    invocation_id: str
    requests: tuple[LimitRequest, ...]
    reason: str
    retry_after: float | None = None


class LimitSnapshot(ImmutableBaseModel):
    active: tuple[LimitLease, ...]
    waiting: tuple[str, ...]
    request_counts: Mapping[str, int]


class LimitError(WorkflowException):
    """Policy conflicts, unavailable coordination, or lost admission leases."""


WaitHook = Callable[[LimitWaitInfo], Awaitable[None]]


class LimitCoordinator(Protocol):
    lease_duration: float | None

    async def acquire(
        self,
        requests: tuple[LimitRequest, ...],
        *,
        invocation_id: str,
        on_wait: WaitHook | None = None,
    ) -> LimitLease: ...

    async def release(self, lease: LimitLease) -> None: ...

    async def renew(self, lease: LimitLease) -> LimitLease: ...

    async def cancel_wait(self, invocation_id: str) -> None: ...

    async def inspect(self) -> LimitSnapshot: ...

    async def retire_pool(self, pool: str) -> None: ...


class _Pool(BaseModel):
    config: RateLimitConfig
    revision: str
    requests: list[float] = Field(default_factory=list)


class _Ticket(BaseModel):
    requests: tuple[LimitRequest, ...]
    expires_at: float | None = None


class _State(BaseModel):
    pools: dict[str, _Pool] = Field(default_factory=dict)
    waiting: dict[str, _Ticket] = Field(default_factory=dict)
    leases: dict[str, LimitLease] = Field(default_factory=dict)
    generation: int = 0
    clock_floor: float = 0

    def expire(self, now: float) -> None:
        for pool in self.pools.values():
            cutoff = now - pool.config.window_duration.total_seconds()
            pool.requests[:] = [t for t in pool.requests if t > cutoff]
        self.leases = {
            key: lease
            for key, lease in self.leases.items()
            if lease.expires_at is None or lease.expires_at > now
        }
        self.waiting = {
            key: ticket
            for key, ticket in self.waiting.items()
            if ticket.expires_at is None or ticket.expires_at > now
        }

    def available(self, requests: tuple[LimitRequest, ...]) -> bool:
        for request in requests:
            pool = self.pools[request.pool]
            config = pool.config
            active = sum(
                any(r.pool == request.pool for r in lease.requests)
                for lease in self.leases.values()
            )
            if config.max_concurrency is not None and active >= config.max_concurrency:
                return False
            if (
                config.requests_per_window is not None
                and len(pool.requests) >= config.requests_per_window
            ):
                return False
        return True


T = TypeVar("T")


async def _complete_cleanup(operation: Awaitable[T]) -> T:
    """Finish cleanup before propagating repeated cancellation to the caller.

    Shield alone keeps the cleanup task alive but lets its caller leave early,
    which can retire a worker pool before its pending lease release finishes.
    """
    cleanup = asyncio.ensure_future(operation)
    interrupted = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            interrupted = True
    result = cleanup.result()
    if interrupted:
        raise asyncio.CancelledError
    return result


class InMemoryLimitCoordinator:
    """Thread-safe state with loop-local notifications; no restart durability."""

    lease_duration: float | None = None
    poll_interval: float | None = None

    def __init__(self, *, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._state = _State()
        self._lock = threading.Lock()
        self._listeners_lock = threading.Lock()
        self._listeners: set[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = set()

    async def _mutate(self, operation: Callable[[_State, float], T]) -> T:
        with self._lock:
            now = max(self._clock(), self._state.clock_floor)
            self._state.clock_floor = now
            self._state.expire(now)
            return operation(self._state, now)

    def notify_waiters(self) -> None:
        """Wake loop-local waiters, including after an injected clock advances."""
        with self._listeners_lock:
            for loop, event in self._listeners:
                try:
                    loop.call_soon_threadsafe(event.set)
                except RuntimeError:
                    pass  # The owning loop has already shut down.

    def _try_acquire(
        self, state: _State, now: float, requests: tuple[LimitRequest, ...], owner: str
    ) -> LimitLease | LimitWaitInfo:
        if owner in state.leases:
            lease = state.leases[owner]
            if lease.requests != requests:
                raise LimitError.for_operator(
                    "An invocation changed its admission bundle."
                )
            return lease
        ticket = state.waiting.get(owner)
        if ticket is not None and ticket.requests != requests:
            raise LimitError.for_operator(
                "An invocation changed its queued admission bundle."
            )
        # Validate the complete bundle before installing any new policy. A
        # rejected request must not latch an ungranted pool's revision/config.
        for request in requests:
            pool = state.pools.get(request.pool)
            if pool is not None and (
                pool.config != request.config or pool.revision != request.revision
            ):
                raise LimitError.for_operator(
                    f"Conflicting limit policies for pool '{request.pool}'."
                )
        for request in requests:
            if request.pool not in state.pools:
                state.pools[request.pool] = _Pool(
                    config=request.config, revision=request.revision
                )
        state.waiting[owner] = _Ticket(
            requests=requests,
            expires_at=(now + 2 * self.lease_duration) if self.lease_duration else None,
        )
        available = state.available(requests)
        keys = {request.pool for request in requests}
        # Preserve arrival order among eligible bundles. A blocked provider must
        # not reserve the shared execution pool and starve unrelated providers.
        for prior_owner, prior in state.waiting.items():
            if prior_owner == owner:
                break
            if keys.intersection(r.pool for r in prior.requests) and state.available(
                prior.requests
            ):
                available = False
                break
        if available:
            del state.waiting[owner]
            state.generation += 1
            lease = LimitLease(
                id=uuid.uuid4().hex,
                invocation_id=owner,
                requests=requests,
                generation=state.generation,
                expires_at=(now + self.lease_duration) if self.lease_duration else None,
            )
            state.leases[owner] = lease
            for request in requests:
                pool = state.pools[request.pool]
                if pool.config.requests_per_window is not None:
                    pool.requests.append(now)
            return lease
        deadlines = (
            [
                pool.requests[0] + pool.config.window_duration.total_seconds()
                for pool in state.pools.values()
                if pool.requests
            ]
            + [
                lease.expires_at
                for lease in state.leases.values()
                if lease.expires_at is not None
            ]
            + [
                ticket.expires_at
                for ticket in state.waiting.values()
                if ticket.expires_at is not None
            ]
        )
        delay = max(0, min(deadlines) - now) if deadlines else None
        return LimitWaitInfo(
            invocation_id=owner,
            requests=requests,
            reason="capacity_or_queue",
            retry_after=delay,
        )

    async def acquire(
        self,
        requests: tuple[LimitRequest, ...],
        *,
        invocation_id: str,
        on_wait: WaitHook | None = None,
    ) -> LimitLease:
        requests = tuple(sorted(requests, key=lambda request: request.pool))
        if len({request.pool for request in requests}) != len(requests):
            raise LimitError.for_operator("An admission bundle repeats a pool.")
        if not requests:
            return LimitLease(
                id=uuid.uuid4().hex,
                invocation_id=invocation_id,
                requests=(),
                generation=0,
            )
        listener = (asyncio.get_running_loop(), asyncio.Event())
        with self._listeners_lock:
            self._listeners.add(listener)
        reported = False
        try:
            while True:
                listener[1].clear()
                result = await self._mutate(
                    lambda state, now: self._try_acquire(
                        state, now, requests, invocation_id
                    )
                )
                if isinstance(result, LimitLease):
                    self.notify_waiters()
                    return result
                if not reported and on_wait is not None:
                    reported = True
                    await on_wait(result)
                delay = result.retry_after
                if self.poll_interval is not None:
                    delay = (
                        min(delay, self.poll_interval)
                        if delay is not None
                        else self.poll_interval
                    )
                try:
                    await asyncio.wait_for(listener[1].wait(), timeout=delay)
                except TimeoutError:
                    pass
        except BaseException:
            await _complete_cleanup(self.cancel_wait(invocation_id))
            raise
        finally:
            with self._listeners_lock:
                self._listeners.discard(listener)

    async def cancel_wait(self, invocation_id: str) -> None:
        def cancel(state: _State, now: float) -> None:
            state.waiting.pop(invocation_id, None)
            state.leases.pop(invocation_id, None)

        await self._mutate(cancel)
        self.notify_waiters()

    async def release(self, lease: LimitLease) -> None:
        if not lease.requests:
            return

        def release(state: _State, now: float) -> None:
            current = state.leases.get(lease.invocation_id)
            if current is not None and current.id == lease.id:
                del state.leases[lease.invocation_id]

        await self._mutate(release)
        self.notify_waiters()

    async def renew(self, lease: LimitLease) -> LimitLease:
        duration = self.lease_duration
        if not lease.requests or duration is None:
            return lease

        def renew(state: _State, now: float) -> LimitLease:
            current = state.leases.get(lease.invocation_id)
            if current is None or current.id != lease.id:
                raise LimitError.for_operator("Admission lease was lost or expired.")
            current = current.model_copy(update={"expires_at": now + duration})
            state.leases[lease.invocation_id] = current
            return current

        return await self._mutate(renew)

    async def inspect(self) -> LimitSnapshot:
        return await self._mutate(
            lambda state, now: LimitSnapshot(
                active=tuple(state.leases.values()),
                waiting=tuple(state.waiting),
                request_counts={
                    key: len(pool.requests) for key, pool in state.pools.items()
                },
            )
        )

    async def retire_pool(self, pool: str) -> None:
        """Remove an unused run-local capacity pool without erasing rate history."""

        def retire(state: _State, now: float) -> None:
            active = any(
                any(r.pool == pool for r in lease.requests)
                for lease in state.leases.values()
            )
            queued = any(
                any(r.pool == pool for r in ticket.requests)
                for ticket in state.waiting.values()
            )
            if active or queued or (pool in state.pools and state.pools[pool].requests):
                raise LimitError.for_operator(
                    "Cannot retire a pool with active admission state."
                )
            state.pools.pop(pool, None)

        await self._mutate(retire)
