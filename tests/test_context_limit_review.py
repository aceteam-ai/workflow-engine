"""Review regressions for rejected bundles, ownership, and cancellation cleanup."""

import asyncio
from typing import ClassVar

import pytest
from overrides import override

from workflow_engine import (
    Empty,
    InMemoryLimitCoordinator,
    LimitError,
    LimitPolicy,
    LimitRequest,
    Node,
    NodeTypeInfo,
    RateLimitConfig,
)
from workflow_engine.contexts import InMemoryExecutionContext

pytestmark = pytest.mark.unit


class ReviewRegionNode(Node[Empty, Empty, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Review regions",
        version="1.0.0",
        parameter_type=Empty,
        limit_regions={
            "high": RateLimitConfig(max_concurrency=1),
            "low": RateLimitConfig(max_concurrency=1),
        },
    )


async def test_rejected_bundle_does_not_install_ungranted_pool_policy():
    coordinator = InMemoryLimitCoordinator()
    held = await coordinator.acquire(
        (LimitRequest(pool="z", config=RateLimitConfig(max_concurrency=1)),),
        invocation_id="held",
    )
    with pytest.raises(LimitError, match="Conflicting"):
        await coordinator.acquire(
            (
                LimitRequest(pool="a", config=RateLimitConfig(max_concurrency=1)),
                LimitRequest(pool="z", config=RateLimitConfig(max_concurrency=2)),
            ),
            invocation_id="rejected",
        )
    assert "a" not in (await coordinator.inspect()).request_counts
    accepted = await coordinator.acquire(
        (
            LimitRequest(
                pool="a", config=RateLimitConfig(max_concurrency=2), revision="2"
            ),
        ),
        invocation_id="accepted",
    )
    await coordinator.release(accepted)
    await coordinator.release(held)


async def test_user_execution_namespace_does_not_disable_nested_order_check():
    policy = LimitPolicy(
        namespace="execution",
        pools={
            LimitPolicy.scope(ReviewRegionNode, "high"): "z",
            LimitPolicy.scope(ReviewRegionNode, "low"): "a",
        },
    )
    context = InMemoryExecutionContext(limit_policy=policy)
    node = ReviewRegionNode(type="ReviewRegion", id="review")
    async with context.limit(node=node, region="high"):
        with pytest.raises(LimitError, match="canonical"):
            async with context.limit(node=node, region="low"):
                pass


async def test_second_cancellation_cannot_skip_renewal_and_lease_cleanup():
    renewing, cleaning, allow_cleanup = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    class Coordinator(InMemoryLimitCoordinator):
        lease_duration: float | None = 0.003

        async def renew(self, lease):
            renewing.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await allow_cleanup.wait()
            return lease

    coordinator = Coordinator(clock=lambda: 100.0)
    context = InMemoryExecutionContext(limit_coordinator=coordinator)
    node = ReviewRegionNode(type="ReviewRegion", id="review")

    async def admitted():
        async with context.limit(node=node, region="high"):
            await asyncio.Event().wait()

    task = asyncio.create_task(admitted())
    await asyncio.wait_for(renewing.wait(), 1)
    task.cancel()
    await asyncio.wait_for(cleaning.wait(), 1)
    task.cancel()
    # Let the second cancellation propagate before completing cleanup.
    await asyncio.sleep(0)
    allow_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await coordinator.inspect()).active == ()


async def test_child_task_does_not_reenter_parent_tasks_region_lease():
    waiting = asyncio.Event()

    class Context(InMemoryExecutionContext):
        @override
        async def on_limit_wait(self, *, node, region, wait_info):
            waiting.set()

    coordinator = InMemoryLimitCoordinator()
    context = Context(limit_coordinator=coordinator)
    node = ReviewRegionNode(type="ReviewRegion", id="review")

    async def child():
        async with context.limit(node=node, region="high") as lease:
            return lease.id

    async with context.limit(node=node, region="high") as outer:
        task = asyncio.create_task(child())
        await asyncio.wait_for(waiting.wait(), 1)
        snapshot = await coordinator.inspect()
        assert len(snapshot.active) == 1 and len(snapshot.waiting) == 1
    assert await asyncio.wait_for(task, 1) != outer.id
    assert (await coordinator.inspect()).active == ()


async def test_expired_lease_cannot_release_or_renew_a_successor_with_same_owner():
    now = [100.0]
    coordinator = InMemoryLimitCoordinator(clock=lambda: now[0])
    coordinator.lease_duration = 1.0
    request = LimitRequest(pool="shared", config=RateLimitConfig(max_concurrency=1))
    old = await coordinator.acquire((request,), invocation_id="owner")
    now[0] += 2
    new = await coordinator.acquire((request,), invocation_id="owner")
    await coordinator.release(old)
    assert (await coordinator.inspect()).active == (new,)
    with pytest.raises(LimitError, match="lost or expired"):
        await coordinator.renew(old)
    await coordinator.release(new)


async def test_repeated_cancellation_waits_for_release_before_retiring_worker_pool():
    admitted, releasing, allow_release = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    class Coordinator(InMemoryLimitCoordinator):
        async def release(self, lease):
            releasing.set()
            await allow_release.wait()
            await super().release(lease)

    coordinator = Coordinator()
    context = InMemoryExecutionContext(limit_coordinator=coordinator)
    node = ReviewRegionNode(type="ReviewRegion", id="review")

    async def run():
        async with context.execution_scope(max_concurrency=1):
            async with context.admit_node(node):
                admitted.set()
                await asyncio.Event().wait()

    task = asyncio.create_task(run())
    await asyncio.wait_for(admitted.wait(), 1)
    task.cancel()
    await asyncio.wait_for(releasing.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    allow_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    snapshot = await coordinator.inspect()
    assert snapshot.active == () and snapshot.waiting == ()
    assert snapshot.request_counts == {}
