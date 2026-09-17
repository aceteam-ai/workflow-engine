"""Deterministic admission and scheduler regressions; no timing races."""

import asyncio
from datetime import timedelta
from typing import ClassVar

import pytest
from overrides import override
from pydantic import ValidationError

from workflow_engine import (
    Empty,
    ErrorClass,
    ExecutionContext,
    InMemoryLimitCoordinator,
    LimitError,
    LimitPolicy,
    LimitRequest,
    Node,
    NodeTypeInfo,
    RateLimitConfig,
    ShouldRetry,
    ShouldYield,
    Workflow,
    WorkflowEngine,
    WorkflowException,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.execution import (
    ParallelExecutionAlgorithm,
    TopologicalExecutionAlgorithm,
)
from workflow_engine.nodes import AttemptNode


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_concurrency": 0},
        {"max_concurrency": -1},
        {"requests_per_window": 0},
        {"window_duration": timedelta(0)},
    ],
)
def test_limit_config_requires_positive_values(kwargs):
    with pytest.raises(ValidationError):
        RateLimitConfig(**kwargs)


async def test_atomic_grants_do_not_charge_queued_work_and_recheck_wakeups():
    clock = Clock()
    coordinator = InMemoryLimitCoordinator(clock=clock)
    request = LimitRequest(
        pool="api",
        config=RateLimitConfig(
            max_concurrency=1,
            requests_per_window=2,
            window_duration=timedelta(seconds=10),
        ),
    )
    first = await coordinator.acquire((request,), invocation_id="first")
    waited = asyncio.Event()

    async def on_wait(info):
        waited.set()

    second = asyncio.create_task(
        coordinator.acquire((request,), invocation_id="second", on_wait=on_wait)
    )
    await waited.wait()
    snapshot = await coordinator.inspect()
    assert snapshot.request_counts == {"api": 1}
    assert len(snapshot.active) == 1
    assert snapshot.waiting == ("second",)
    await coordinator.release(first)
    second_lease = await asyncio.wait_for(second, 1)
    await coordinator.release(second_lease)
    waited.clear()
    third = asyncio.create_task(
        coordinator.acquire((request,), invocation_id="third", on_wait=on_wait)
    )
    await waited.wait()
    assert not third.done()
    assert (await coordinator.inspect()).active == ()
    clock.now += 10
    coordinator.notify_waiters()
    lease = await asyncio.wait_for(third, 1)
    assert (await coordinator.inspect()).request_counts == {"api": 1}
    await coordinator.release(lease)


async def test_fifo_cancellation_and_atomic_bundle_skip_blocked_provider():
    coordinator = InMemoryLimitCoordinator()
    scarce = LimitRequest(pool="scarce", config=RateLimitConfig(max_concurrency=1))
    workers = LimitRequest(pool="workers", config=RateLimitConfig(max_concurrency=1))
    held = await coordinator.acquire((scarce,), invocation_id="held")
    waiting = asyncio.Event()

    async def on_wait(info):
        waiting.set()

    blocked = asyncio.create_task(
        coordinator.acquire((scarce, workers), invocation_id="blocked", on_wait=on_wait)
    )
    await waiting.wait()
    other = await asyncio.wait_for(
        coordinator.acquire((workers,), invocation_id="other"), 1
    )
    assert {lease.invocation_id for lease in (await coordinator.inspect()).active} == {
        "held",
        "other",
    }
    blocked.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocked
    assert (await coordinator.inspect()).waiting == ()
    await coordinator.release(other)
    await coordinator.release(held)


async def test_shared_coordinator_notifies_a_different_thread_event_loop():
    coordinator = InMemoryLimitCoordinator()
    request = LimitRequest(pool="threaded", config=RateLimitConfig(max_concurrency=1))
    first = await coordinator.acquire((request,), invocation_id="main")
    loop = asyncio.get_running_loop()
    waiting = asyncio.Event()

    async def threaded():
        async def on_wait(info):
            loop.call_soon_threadsafe(waiting.set)

        lease = await coordinator.acquire(
            (request,), invocation_id="thread", on_wait=on_wait
        )
        await coordinator.release(lease)
        return lease.generation

    other = asyncio.create_task(asyncio.to_thread(asyncio.run, threaded()))
    await asyncio.wait_for(waiting.wait(), 2)
    await coordinator.release(first)
    assert await asyncio.wait_for(other, 2) > first.generation
    assert (await coordinator.inspect()).active == ()


class LimitedProbeNode(Node[Empty, Empty, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Limited Probe",
        version="1.0.0",
        parameter_type=Empty,
        execution_limits=RateLimitConfig(max_concurrency=1),
        limit_regions={"provider": RateLimitConfig(max_concurrency=1)},
    )

    @classmethod
    @override
    def static_input_type(cls) -> type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> type[Empty]:
        return Empty

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[Empty],
        output_type: type[Empty],
        input: Empty,
    ) -> Empty:
        assert isinstance(context, ProbeContext)
        context.calls[self.id] = context.calls.get(self.id, 0) + 1
        context.started.setdefault(self.id, asyncio.Event()).set()
        if context.mode == "retry" and context.calls[self.id] == 1:
            raise ShouldRetry.for_user("again", node=self, backoff=timedelta(0))
        if context.mode == "yield":
            raise ShouldYield("later")
        if context.mode == "fail":
            raise WorkflowException.for_user(
                "failed", error_class=ErrorClass.VALIDATION
            )
        if context.mode == "region":
            async with context.limit(node=self, region="provider"):
                return input
        if self.id in context.release:
            await context.release[self.id].wait()
        return input


class OtherLimitedProbeNode(LimitedProbeNode):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Other Limited Probe",
        version="1.0.0",
        parameter_type=Empty,
    )


class ProbeContext(InMemoryExecutionContext):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.started = {}
        self.waiting = asyncio.Event()
        self.release = {}
        self.calls = {}
        self.mode = "ok"
        self.events = []
        self.cache = False

    @override
    async def on_node_start(self, *, node, input_type, output_type, input):
        if self.cache:
            return {}
        return None

    @override
    async def on_node_waiting_for_limit(self, *, node, invocation_id, wait_info):
        self.events.append(("waiting", node.id))
        self.waiting.set()

    @override
    async def on_node_admitted(self, *, node, invocation_id, lease):
        self.events.append(("admitted", node.id))


async def probe_workflow(engine, nodes):
    return Workflow(
        input_node=engine.create_input_node(),
        inner_nodes=nodes,
        output_node=engine.create_output_node(),
        edges=[],
    )


async def test_shared_context_admission_coordinates_distinct_algorithms():
    coordinator = InMemoryLimitCoordinator()
    first_engine = WorkflowEngine(execution_algorithm=TopologicalExecutionAlgorithm())
    second_engine = WorkflowEngine(execution_algorithm=ParallelExecutionAlgorithm())
    first_context = ProbeContext(limit_coordinator=coordinator)
    second_context = ProbeContext(limit_coordinator=coordinator)
    first_context.release["one"] = asyncio.Event()
    first_context.started["one"] = asyncio.Event()
    one = asyncio.create_task(
        first_engine.execute_node(
            context=first_context, node=LimitedProbeNode, node_id="one", input={}
        )
    )
    await first_context.started["one"].wait()
    two = asyncio.create_task(
        second_engine.execute_node(
            context=second_context, node=LimitedProbeNode, node_id="two", input={}
        )
    )
    await second_context.waiting.wait()
    assert second_context.calls == {}
    first_context.release["one"].set()
    results = await asyncio.wait_for(asyncio.gather(one, two), 2)
    assert all(
        result.status is WorkflowExecutionResultStatus.SUCCESS for result in results
    )
    assert second_context.calls == {"two": 1}
    assert (await coordinator.inspect()).active == ()


async def test_waiting_provider_does_not_hold_parallel_worker_capacity():
    coordinator = InMemoryLimitCoordinator()
    policy = LimitPolicy()
    request = policy.resolve(
        LimitedProbeNode, "execution", LimitedProbeNode.TYPE_INFO.execution_limits
    )
    assert request is not None
    held = await coordinator.acquire((request,), invocation_id="external")
    engine = WorkflowEngine(
        execution_algorithm=ParallelExecutionAlgorithm(max_concurrency=1)
    )
    context = ProbeContext(limit_coordinator=coordinator)
    context.started["other"] = asyncio.Event()
    graph = await probe_workflow(
        engine,
        [
            engine.create_node(LimitedProbeNode, id="blocked"),
            engine.create_node(OtherLimitedProbeNode, id="other"),
        ],
    )
    task = asyncio.create_task(
        engine.execute(context=context, workflow=graph, input={})
    )
    await context.waiting.wait()
    await asyncio.wait_for(context.started["other"].wait(), 1)
    assert "blocked" not in context.calls
    await coordinator.release(held)
    assert (
        await asyncio.wait_for(task, 2)
    ).status is WorkflowExecutionResultStatus.SUCCESS
    assert all(
        not pool.startswith("execution:")
        for pool in (await coordinator.inspect()).request_counts
    )


@pytest.mark.parametrize("mode", ["retry", "yield", "fail", "region"])
async def test_release_on_terminal_signals_and_retry(algorithm, mode):
    coordinator = InMemoryLimitCoordinator()
    context = ProbeContext(limit_coordinator=coordinator)
    context.mode = mode
    engine = WorkflowEngine(execution_algorithm=algorithm)
    result = await engine.execute_node(context=context, node=LimitedProbeNode, input={})
    assert (
        result.status
        is {
            "retry": WorkflowExecutionResultStatus.SUCCESS,
            "region": WorkflowExecutionResultStatus.SUCCESS,
            "yield": WorkflowExecutionResultStatus.YIELDED,
            "fail": WorkflowExecutionResultStatus.ERROR,
        }[mode]
    )
    assert (await coordinator.inspect()).active == ()
    assert sum(context.calls.values()) == (2 if mode == "retry" else 1)


async def test_cache_hit_skips_admission_and_operator_can_override_to_unlimited():
    coordinator = InMemoryLimitCoordinator()
    engine = WorkflowEngine()
    context = ProbeContext(limit_coordinator=coordinator)
    context.cache = True
    result = await engine.execute_node(context=context, node=LimitedProbeNode, input={})
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert context.events == [] and context.calls == {}
    assert (await coordinator.inspect()).request_counts == {}
    policy = LimitPolicy(
        overrides={LimitPolicy.scope(LimitedProbeNode): RateLimitConfig()}
    )
    context = ProbeContext(limit_coordinator=coordinator, limit_policy=policy)
    assert (
        await engine.execute_node(context=context, node=LimitedProbeNode, input={})
    ).status is WorkflowExecutionResultStatus.SUCCESS


async def test_boundary_cancels_queued_member_but_drains_admitted_members():
    coordinator = InMemoryLimitCoordinator()
    engine = WorkflowEngine(execution_algorithm=ParallelExecutionAlgorithm())
    context = ProbeContext(limit_coordinator=coordinator)
    context.mode = "fail"
    request = LimitPolicy().resolve(
        LimitedProbeNode, "execution", LimitedProbeNode.TYPE_INFO.execution_limits
    )
    assert request is not None
    held = await coordinator.acquire((request,), invocation_id="external")
    inner = await probe_workflow(
        engine,
        [
            engine.create_node(LimitedProbeNode, id="queued"),
            engine.create_node(OtherLimitedProbeNode, id="failure"),
        ],
    )
    result = await asyncio.wait_for(
        engine.execute_node(
            context=context, node=AttemptNode, params={"workflow": inner}, input={}
        ),
        2,
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert "node/queued" not in context.calls
    assert (await coordinator.inspect()).waiting == ()
    await coordinator.release(held)


async def test_region_reentry_shares_lease_and_undeclared_region_fails():
    coordinator = InMemoryLimitCoordinator()
    context = ProbeContext(limit_coordinator=coordinator)
    node = WorkflowEngine().create_node(LimitedProbeNode, id="limited")
    async with context.limit(node=node, region="provider") as outer:
        async with context.limit(node=node, region="provider") as inner:
            assert inner.id == outer.id
            assert len((await coordinator.inspect()).active) == 1
    assert (await coordinator.inspect()).active == ()
    with pytest.raises(LimitError, match="no declared"):
        async with context.limit(node=node, region="unknown"):
            pass
