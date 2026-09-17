"""Durable quota state, cross-process recovery, and cancellation races."""

import asyncio
import json
import subprocess
import sys
import threading
from datetime import timedelta

import pytest

from workflow_engine import (
    InMemoryLimitCoordinator,
    LimitError,
    LimitRequest,
    RateLimitConfig,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.limits import SQLiteLimitCoordinator

from .test_context_limits import Clock, LimitedProbeNode, ProbeContext


def request(*, rate=2, concurrency=1):
    return LimitRequest(
        pool="provider",
        config=RateLimitConfig(
            max_concurrency=concurrency,
            requests_per_window=rate,
            window_duration=timedelta(seconds=60),
        ),
    )


async def test_independent_sqlite_coordinators_share_atomic_limits(tmp_path):
    path = tmp_path / "limits.db"
    clock = Clock()
    async with (
        SQLiteLimitCoordinator(path, clock=clock) as one,
        SQLiteLimitCoordinator(path, clock=clock) as two,
    ):
        first = await one.acquire((request(),), invocation_id="first")
        waiting = asyncio.Event()

        async def on_wait(info):
            waiting.set()

        pending = asyncio.create_task(
            two.acquire((request(),), invocation_id="second", on_wait=on_wait)
        )
        await waiting.wait()
        assert (await one.inspect()).request_counts == {"provider": 1}
        assert len((await one.inspect()).active) == 1
        await one.release(first)
        second = await asyncio.wait_for(pending, 2)
        assert second.generation > first.generation
        assert (await two.inspect()).request_counts == {"provider": 2}
        await two.release(second)
    async with SQLiteLimitCoordinator(path, clock=clock) as reopened:
        snapshot = await reopened.inspect()
        assert snapshot.active == () and snapshot.request_counts == {"provider": 2}
        waiting = asyncio.Event()

        async def on_wait(info):
            waiting.set()

        pending = asyncio.create_task(
            reopened.acquire((request(),), invocation_id="third", on_wait=on_wait)
        )
        await waiting.wait()
        clock.now += 60
        reopened.notify_waiters()
        third = await asyncio.wait_for(pending, 2)
        await reopened.release(third)


async def test_clock_rollback_does_not_refund_quota(tmp_path):
    clock = Clock()
    async with SQLiteLimitCoordinator(
        tmp_path / "limits.db", clock=clock
    ) as coordinator:
        lease = await coordinator.acquire((request(rate=1),), invocation_id="first")
        await coordinator.release(lease)
        clock.now = 20
        assert (await coordinator.inspect()).request_counts == {"provider": 1}
        clock.now = 160
        assert (await coordinator.inspect()).request_counts == {"provider": 0}


async def test_crashed_owner_expires_with_new_fencing_generation(tmp_path):
    path = tmp_path / "limits.db"
    program = """
import asyncio, sys
from workflow_engine import LimitRequest, RateLimitConfig
from workflow_engine.limits import SQLiteLimitCoordinator
async def main():
    async with SQLiteLimitCoordinator(sys.argv[1], clock=lambda: 100.0, lease_duration=10) as coordinator:
        lease = await coordinator.acquire((LimitRequest(pool="provider", config=RateLimitConfig(max_concurrency=1)),), invocation_id="crashed")
        print(lease.model_dump_json(), flush=True)
        await asyncio.Event().wait()
asyncio.run(main())
"""
    process = subprocess.Popen(
        [sys.executable, "-c", program, str(path)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert process.stdout is not None
        payload = await asyncio.wait_for(asyncio.to_thread(process.stdout.readline), 10)
        first = json.loads(payload)
    finally:
        process.kill()
        await asyncio.to_thread(process.wait)
        if process.stdout is not None:
            process.stdout.close()
    clock = Clock()
    async with SQLiteLimitCoordinator(
        path, clock=clock, lease_duration=10
    ) as coordinator:
        assert len((await coordinator.inspect()).active) == 1
        clock.now = 111
        second = await coordinator.acquire(
            (LimitRequest(pool="provider", config=RateLimitConfig(max_concurrency=1)),),
            invocation_id="recovered",
        )
        assert second.generation > first["generation"]
        assert second.id != first["id"]
        await coordinator.release(second)


async def test_cancel_racing_with_database_grant_releases_committed_lease(tmp_path):
    async with SQLiteLimitCoordinator(tmp_path / "limits.db") as coordinator:
        committed = threading.Event()
        finish = threading.Event()
        original = coordinator._transaction

        def transaction(operation):
            result = original(operation)
            if not committed.is_set():
                committed.set()
                finish.wait(timeout=10)
            return result

        coordinator._transaction = transaction
        pending = asyncio.create_task(
            coordinator.acquire((request(),), invocation_id="cancelled")
        )
        assert await asyncio.to_thread(committed.wait, 5)
        pending.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        snapshot = await coordinator.inspect()
        assert snapshot.active == () and snapshot.waiting == ()
        # A committed grant counts against rate history, even when cancelled.
        assert snapshot.request_counts == {"provider": 1}


async def test_policy_conflict_and_backend_failure_are_closed(tmp_path):
    async with SQLiteLimitCoordinator(tmp_path / "limits.db") as coordinator:
        lease = await coordinator.acquire((request(),), invocation_id="one")
        with pytest.raises(LimitError, match="Conflicting"):
            await coordinator.acquire((request(rate=10),), invocation_id="two")
        await coordinator.release(lease)
    with pytest.raises(LimitError, match="closed"):
        await coordinator.acquire((request(),), invocation_id="later")
    async with SQLiteLimitCoordinator(
        tmp_path / "missing" / "limits.db"
    ) as unavailable:
        with pytest.raises(LimitError, match="unavailable"):
            await unavailable.acquire((request(),), invocation_id="failure")


async def test_context_renews_long_running_lease_and_cancellation_releases(tmp_path):
    clock = Clock()
    async with SQLiteLimitCoordinator(
        tmp_path / "limits.db", clock=clock, lease_duration=0.3, poll_interval=0.01
    ) as coordinator:
        renewed = asyncio.Event()
        original = coordinator.renew

        async def renew(lease):
            result = await original(lease)
            renewed.set()
            return result

        coordinator.renew = renew
        context = ProbeContext(limit_coordinator=coordinator)
        context.started["node"] = asyncio.Event()
        context.release["node"] = asyncio.Event()
        pending = asyncio.create_task(
            WorkflowEngine().execute_node(
                context=context, node=LimitedProbeNode, input={}
            )
        )
        await context.started["node"].wait()
        clock.now += 0.1
        await asyncio.wait_for(renewed.wait(), 2)
        assert (await coordinator.inspect()).active[0].expires_at == pytest.approx(
            100.4
        )
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert (await coordinator.inspect()).active == ()


async def test_lease_loss_stops_node_and_returns_operator_failure(tmp_path):
    async with SQLiteLimitCoordinator(
        tmp_path / "limits.db", lease_duration=0.3, poll_interval=0.01
    ) as coordinator:

        async def renew(lease):
            raise LimitError.for_operator("lost ownership")

        coordinator.renew = renew
        context = ProbeContext(limit_coordinator=coordinator)
        context.release["node"] = asyncio.Event()
        result = await asyncio.wait_for(
            WorkflowEngine().execute_node(
                context=context, node=LimitedProbeNode, input={}
            ),
            2,
        )
        assert result.status is WorkflowExecutionResultStatus.ERROR
        assert (await coordinator.inspect()).active == ()


@pytest.mark.parametrize("backend", [InMemoryLimitCoordinator, SQLiteLimitCoordinator])
async def test_wait_hook_failure_cleans_ticket(tmp_path, backend):
    coordinator = (
        backend(tmp_path / "limits.db")
        if backend is SQLiteLimitCoordinator
        else backend()
    )
    lease = await coordinator.acquire((request(),), invocation_id="first")

    async def on_wait(info):
        raise RuntimeError("observer failed")

    try:
        with pytest.raises(RuntimeError, match="observer"):
            await coordinator.acquire(
                (request(),), invocation_id="second", on_wait=on_wait
            )
        assert (await coordinator.inspect()).waiting == ()
        await coordinator.release(lease)
    finally:
        if isinstance(coordinator, SQLiteLimitCoordinator):
            await coordinator.close()
