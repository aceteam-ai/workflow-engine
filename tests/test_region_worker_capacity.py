"""Region admission returns only worker capacity, retaining provider ownership."""

import asyncio
from typing import ClassVar

import pytest
from overrides import override

from workflow_engine import (
    Empty,
    ExecutionContext,
    InMemoryLimitCoordinator,
    LimitPolicy,
    NodeTypeInfo,
    RateLimitConfig,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.execution import ParallelExecutionAlgorithm
from workflow_engine.limits import SQLiteLimitCoordinator

from .test_context_limits import (
    LimitedProbeNode,
    OtherLimitedProbeNode,
    ProbeContext,
    probe_workflow,
)


@pytest.fixture(params=["memory", "sqlite"])
async def coordinator(request, tmp_path):
    if request.param == "memory":
        yield InMemoryLimitCoordinator()
    else:
        async with SQLiteLimitCoordinator(tmp_path / "limits.db") as coordinator:
            yield coordinator


class RegionContext(ProbeContext):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.region_waiting = asyncio.Event()
        self.fail_wait_hook = False
        self.restored = False
        self.region_events = []

    @override
    async def on_limit_wait(self, *, node, region, wait_info):
        self.region_waiting.set()
        if self.fail_wait_hook:
            raise RuntimeError("region observer failed")

    @override
    async def on_limit_acquired(self, *, node, region, lease):
        self.region_events.append(("acquired", region))

    @override
    async def on_limit_released(self, *, node, region, lease):
        self.region_events.append(("released", region))


class WaitingRegionProbeNode(LimitedProbeNode):
    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[Empty],
        output_type: type[Empty],
        input: Empty,
    ) -> Empty:
        assert isinstance(context, RegionContext)
        try:
            async with context.limit(node=self, region="provider"):
                pass
        except RuntimeError:
            if not context.fail_wait_hook:
                raise
        run = context._admission.run.get()
        assert run is not None and run.capacity is not None
        snapshot = await run.coordinator.inspect()
        capacity_leases = [
            lease
            for lease in snapshot.active
            if any(request.pool == run.capacity.pool for request in lease.requests)
        ]
        assert len(capacity_leases) == 1
        assert f"/{self.id}/" in capacity_leases[0].invocation_id
        context.restored = True
        return input


async def test_region_wait_frees_worker_but_retains_whole_node_provider_quota(
    coordinator,
):
    policy = LimitPolicy()
    region = policy.resolve(
        WaitingRegionProbeNode,
        "provider",
        WaitingRegionProbeNode.TYPE_INFO.limit_regions["provider"],
    )
    whole = policy.resolve(
        WaitingRegionProbeNode,
        "execution",
        WaitingRegionProbeNode.TYPE_INFO.execution_limits,
    )
    assert region is not None and whole is not None
    external = await coordinator.acquire((region,), invocation_id="external")
    engine = WorkflowEngine(
        execution_algorithm=ParallelExecutionAlgorithm(max_concurrency=1)
    )
    context = RegionContext(limit_coordinator=coordinator)
    context.started["other"] = asyncio.Event()
    graph = await probe_workflow(
        engine,
        [
            engine.create_node(WaitingRegionProbeNode, id="region"),
            engine.create_node(OtherLimitedProbeNode, id="other"),
        ],
    )
    task = asyncio.create_task(
        engine.execute(context=context, workflow=graph, input={})
    )
    await context.region_waiting.wait()
    await asyncio.wait_for(context.started["other"].wait(), 2)
    snapshot = await coordinator.inspect()
    assert any(
        any(request.pool == whole.pool for request in lease.requests)
        for lease in snapshot.active
    )
    assert not context.restored
    await coordinator.release(external)
    assert (
        await asyncio.wait_for(task, 2)
    ).status is WorkflowExecutionResultStatus.SUCCESS
    assert context.restored
    assert (await coordinator.inspect()).active == ()


async def test_caught_region_wait_error_restores_worker_before_continuing(coordinator):
    region = LimitPolicy().resolve(
        WaitingRegionProbeNode,
        "provider",
        WaitingRegionProbeNode.TYPE_INFO.limit_regions["provider"],
    )
    assert region is not None
    external = await coordinator.acquire((region,), invocation_id="external")
    engine = WorkflowEngine(
        execution_algorithm=ParallelExecutionAlgorithm(max_concurrency=1)
    )
    context = RegionContext(limit_coordinator=coordinator)
    context.fail_wait_hook = True
    result = await asyncio.wait_for(
        engine.execute_node(context=context, node=WaitingRegionProbeNode, input={}), 2
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert context.restored
    assert context.region_events == []
    assert len((await coordinator.inspect()).active) == 1
    await coordinator.release(external)


async def test_cancellation_during_region_wait_releases_parent_without_waiting_for_worker(
    coordinator,
):
    region = LimitPolicy().resolve(
        WaitingRegionProbeNode,
        "provider",
        WaitingRegionProbeNode.TYPE_INFO.limit_regions["provider"],
    )
    assert region is not None
    external = await coordinator.acquire((region,), invocation_id="external")
    engine = WorkflowEngine(
        execution_algorithm=ParallelExecutionAlgorithm(max_concurrency=1)
    )
    context = RegionContext(limit_coordinator=coordinator)
    task = asyncio.create_task(
        engine.execute_node(context=context, node=WaitingRegionProbeNode, input={})
    )
    await context.region_waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    snapshot = await coordinator.inspect()
    assert snapshot.waiting == ()
    assert [lease.id for lease in snapshot.active] == [external.id]
    await coordinator.release(external)


class NestedRegionProbeNode(LimitedProbeNode):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Nested Region Probe",
        version="1.0.0",
        parameter_type=Empty,
        execution_limits=RateLimitConfig(max_concurrency=1),
        limit_regions={
            "provider": RateLimitConfig(max_concurrency=1),
            "secondary": RateLimitConfig(max_concurrency=1),
        },
    )


async def test_nested_regions_return_capacity_to_each_parent_in_order(coordinator):
    context = RegionContext(limit_coordinator=coordinator)
    node = WorkflowEngine().create_node(NestedRegionProbeNode, id="nested")
    async with context.execution_scope(max_concurrency=1) as run:
        assert run.capacity is not None
        capacity_pool = run.capacity.pool

        async def owner():
            leases = (await coordinator.inspect()).active
            return [
                lease.id
                for lease in leases
                if any(r.pool == capacity_pool for r in lease.requests)
            ]

        async with context.admit_node(node) as whole:
            assert await owner() == [whole.id]
            async with context.limit(node=node, region="provider") as outer:
                assert await owner() == [outer.id]
                async with context.limit(node=node, region="secondary") as inner:
                    assert await owner() == [inner.id]
                assert await owner() == [outer.id]
            assert await owner() == [whole.id]
    assert (await coordinator.inspect()).active == ()


async def test_cancellation_interrupts_worker_restoration_after_region_error(
    coordinator,
):
    context = RegionContext(limit_coordinator=coordinator)
    node = WorkflowEngine().create_node(WaitingRegionProbeNode, id="restore")
    region = LimitPolicy().resolve(
        WaitingRegionProbeNode,
        "provider",
        WaitingRegionProbeNode.TYPE_INFO.limit_regions["provider"],
    )
    assert region is not None
    external = await coordinator.acquire((region,), invocation_id="external-provider")
    restoring = asyncio.Event()
    workers = []
    acquire = coordinator.acquire

    async def observe_acquire(requests, *, invocation_id, on_wait=None):
        if invocation_id.endswith("/restore-worker"):
            restoring.set()
        return await acquire(requests, invocation_id=invocation_id, on_wait=on_wait)

    coordinator.acquire = observe_acquire
    async with context.execution_scope(max_concurrency=1) as run:
        assert run.capacity is not None
        capacity = run.capacity

        async def fail_after_worker_is_taken(*, node, region, wait_info):
            workers.append(
                await coordinator.acquire((capacity,), invocation_id="external-worker")
            )
            raise RuntimeError("observer failed")

        context.on_limit_wait = fail_after_worker_is_taken

        async def invoke():
            async with context.admit_node(node):
                async with context.limit(node=node, region="provider"):
                    pass

        task = asyncio.create_task(invoke())
        await asyncio.wait_for(restoring.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert (await coordinator.inspect()).waiting == ()
        await coordinator.release(workers[0])
    await coordinator.release(external)
    assert (await coordinator.inspect()).active == ()
