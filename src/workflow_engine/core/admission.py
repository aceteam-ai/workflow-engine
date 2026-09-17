"""Context-owned lifecycle and per-run bookkeeping for quota admission."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .error import ErrorClass
from .limits import (
    InMemoryLimitCoordinator,
    LimitCoordinator,
    LimitError,
    LimitLease,
    LimitPolicy,
    LimitRequest,
    LimitWaitInfo,
    RateLimitConfig,
)

if TYPE_CHECKING:
    from .context import ExecutionContext
    from .node import Node


@dataclass
class RunAdmission:
    id: str
    coordinator: LimitCoordinator
    capacity: LimitRequest | None
    legacy: Mapping[str, RateLimitConfig]
    admitted: set[str] = field(default_factory=set)
    attempts: dict[str, int] = field(default_factory=dict)
    blocked: Callable[[str], bool] = lambda _: False


@dataclass
class _Invocation:
    id: str
    ordinal: int = 0


@dataclass
class _Held:
    task: asyncio.Task
    lease: LimitLease


class ContextAdmission:
    def __init__(
        self,
        context: ExecutionContext,
        *,
        coordinator: LimitCoordinator | None,
        policy: LimitPolicy | None,
        run_id: str,
    ):
        self.context = context
        self.explicit_coordinator = coordinator is not None
        self.coordinator = coordinator or InMemoryLimitCoordinator()
        self.policy = policy or LimitPolicy()
        self.run_id = run_id
        self.run: ContextVar[RunAdmission | None] = ContextVar(
            "admission_run", default=None
        )
        self.invocation: ContextVar[_Invocation | None] = ContextVar(
            "admission_invocation", default=None
        )
        self.held: ContextVar[tuple[_Held, ...]] = ContextVar(
            "admission_held", default=()
        )

    @asynccontextmanager
    async def execution_scope(
        self,
        *,
        max_concurrency: int | None = None,
        legacy: Mapping[str, RateLimitConfig] | None = None,
        legacy_coordinator: LimitCoordinator | None = None,
    ) -> AsyncIterator[RunAdmission]:
        run_id = uuid.uuid4().hex
        capacity = None
        if max_concurrency is not None:
            capacity = LimitRequest(
                pool=f"execution:{self.run_id}:{run_id}",
                config=RateLimitConfig(max_concurrency=max_concurrency),
            )
        coordinator = self.coordinator
        if legacy and legacy_coordinator is not None and not self.explicit_coordinator:
            coordinator = legacy_coordinator
        run = RunAdmission(run_id, coordinator, capacity, legacy or {})
        token = self.run.set(run)
        try:
            yield run
        finally:
            self.run.reset(token)
            if capacity is not None:
                await asyncio.shield(coordinator.retire_pool(capacity.pool))

    @asynccontextmanager
    async def limit(self, node: Node, region: str, *, whole_node: bool = False):
        run = self.run.get()
        coordinator = run.coordinator if run is not None else self.coordinator
        default = node.TYPE_INFO.execution_limits
        if not whole_node:
            if region not in node.TYPE_INFO.limit_regions:
                raise LimitError.for_builder(
                    f"Node '{node.id}' has no declared limit region '{region}'.",
                    error_class=ErrorClass.VALIDATION,
                )
            default = node.TYPE_INFO.limit_regions[region]
        scope = self.policy.scope(type(node), region)
        if whole_node and run is not None and node.type in run.legacy:
            legacy = run.legacy[node.type]
            if (
                scope in self.policy.overrides
                and self.policy.overrides[scope] != legacy
            ):
                raise LimitError.for_builder(
                    f"Conflicting context and legacy limits for '{node.type}'.",
                    error_class=ErrorClass.VALIDATION,
                )
            default = legacy
        request = self.policy.resolve(type(node), region, default)
        requests = [request] if request is not None else []
        if whole_node and run is not None and run.capacity is not None:
            requests.append(run.capacity)
        task = asyncio.current_task()
        assert task is not None
        held = tuple(entry for entry in self.held.get() if entry.task is task)
        if not whole_node and request is not None:
            for entry in held:
                existing = next(
                    (r for r in entry.lease.requests if r.pool == request.pool), None
                )
                if existing is not None:
                    if existing != request:
                        raise LimitError.for_builder(
                            "Cannot change a held pool's policy."
                        )
                    yield entry.lease
                    return
            keys = [
                r.pool
                for entry in held
                for r in entry.lease.requests
                if not r.pool.startswith("execution:")
            ]
            if keys and request.pool <= max(keys):
                raise LimitError.for_builder(
                    "Nested limit regions must acquire distinct pools in canonical key order.",
                    error_class=ErrorClass.VALIDATION,
                )
        invocation = self.invocation.get()
        token = None
        if whole_node or invocation is None:
            attempt = 0
            if run is not None:
                attempt = run.attempts.get(node.id, 0)
                run.attempts[node.id] = attempt + 1
            invocation = _Invocation(
                f"{self.run_id}/{run.id if run else uuid.uuid4().hex}/{node.id}/{attempt}"
            )
            token = self.invocation.set(invocation)
        ordinal = invocation.ordinal
        invocation.ordinal += 1
        invocation_id = f"{invocation.id}/{region}/{ordinal}"

        async def on_wait(info: LimitWaitInfo) -> None:
            if whole_node:
                await self.context.on_node_waiting_for_limit(
                    node=node, invocation_id=invocation_id, wait_info=info
                )
            else:
                await self.context.on_limit_wait(
                    node=node, region=region, wait_info=info
                )

        lease: LimitLease | None = None
        held_token = None
        renewal: asyncio.Task | None = None
        lost: list[Exception] = []

        async def renew() -> None:
            assert coordinator.lease_duration is not None and lease is not None
            try:
                while True:
                    await asyncio.sleep(coordinator.lease_duration / 3)
                    await coordinator.renew(lease)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                lost.append(error)
                task.cancel()

        try:
            lease = await coordinator.acquire(
                tuple(requests), invocation_id=invocation_id, on_wait=on_wait
            )
            if whole_node and run is not None:
                if run.blocked(node.id):
                    raise asyncio.CancelledError
                run.admitted.add(node.id)
            held_token = self.held.set((*held, _Held(task, lease)))
            if lease.expires_at is not None:
                renewal = asyncio.create_task(renew())
            if whole_node:
                await self.context.on_node_admitted(
                    node=node, invocation_id=invocation_id, lease=lease
                )
            else:
                await self.context.on_limit_acquired(
                    node=node, region=region, lease=lease
                )
            yield lease
        except asyncio.CancelledError:
            if lost:
                raise LimitError.for_operator(
                    "Admission lease renewal failed; execution stopped."
                ) from lost[0]
            raise
        finally:
            if renewal is not None:
                renewal.cancel()
                await asyncio.gather(renewal, return_exceptions=True)
            if held_token is not None:
                self.held.reset(held_token)
            if token is not None:
                self.invocation.reset(token)
            if lease is not None:
                await asyncio.shield(coordinator.release(lease))
                if not whole_node:
                    await self.context.on_limit_released(
                        node=node, region=region, lease=lease
                    )
