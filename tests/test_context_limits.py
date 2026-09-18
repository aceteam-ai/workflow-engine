"""Deterministic admission and scheduler regressions; no timing races."""

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import ClassVar

import pytest
from overrides import override
from pydantic import ValidationError

from workflow_engine import (
    DataMapping,
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
    ValidatedWorkflow,
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
        # Independent of `events` (which some tests assert is empty): every
        # id for which on_node_start actually fired, and every terminal hook
        # (on_node_finish / on_node_expand / on_node_error) that fired per
        # node id, in call order. Used to assert the start/terminal pairing
        # invariant without disturbing existing `events`-based assertions.
        self.node_starts: list[str] = []
        self.terminal_hooks: dict[str, list[str]] = {}

    def _record_terminal(self, node_id: str, hook: str) -> None:
        self.terminal_hooks.setdefault(node_id, []).append(hook)

    @override
    async def on_node_start(self, *, node, input_type, output_type, input):
        self.node_starts.append(node.id)
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

    @override
    async def on_node_finish(
        self, *, node, input_type, output_type, input, output
    ) -> DataMapping:
        self._record_terminal(node.id, "finish")
        return output

    @override
    async def on_node_expand(
        self, *, node, input_type, output_type, input, workflow
    ) -> ValidatedWorkflow:
        self._record_terminal(node.id, "expand")
        return workflow

    @override
    async def on_node_error(
        self, *, node, input_type, output_type, input, exception
    ) -> WorkflowException:
        self._record_terminal(node.id, "error")
        return exception


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


async def test_cancelled_admission_still_reaches_exactly_one_terminal_hook():
    """Issue #277: a node cancelled while queued for admission must still

    reach a terminal hook, and cancellation must still propagate.

    Set up a fan-out scope (an AttemptNode boundary) with a concurrency
    limit of 1 that is already fully consumed externally, so two members
    queue for admission (more than the limit allows). A third member fails,
    which fails the boundary and cancels the two still-queued members.
    Before the fix, those two members fire on_node_start but never reach any
    terminal hook, which is exactly the leak a host keys resource release
    off of.
    """
    coordinator = InMemoryLimitCoordinator()
    engine = WorkflowEngine(execution_algorithm=ParallelExecutionAlgorithm())
    context = ProbeContext(limit_coordinator=coordinator)
    context.mode = "fail"
    request = LimitPolicy().resolve(
        LimitedProbeNode, "execution", LimitedProbeNode.TYPE_INFO.execution_limits
    )
    assert request is not None
    # Consume the only slot externally so both LimitedProbeNode members
    # below queue for admission instead of running.
    held = await coordinator.acquire((request,), invocation_id="external")
    inner = await probe_workflow(
        engine,
        [
            engine.create_node(LimitedProbeNode, id="queued1"),
            engine.create_node(LimitedProbeNode, id="queued2"),
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

    # Neither queued member ever actually ran.
    assert "node/queued1" not in context.calls
    assert "node/queued2" not in context.calls

    # Every node that fired on_node_start (this includes the boundary's own
    # nodes, keyed by their flat ids) must have reached exactly one terminal
    # hook. Today's code leaves the cancelled queued members with zero.
    assert context.node_starts, "sanity: on_node_start should have fired at all"
    for node_id in context.node_starts:
        hooks = context.terminal_hooks.get(node_id, [])
        assert len(hooks) == 1, (
            f"node {node_id!r} fired on_node_start but reached "
            f"{len(hooks)} terminal hooks ({hooks!r}), expected exactly 1"
        )

    # The cancelled members specifically must have been terminated via
    # on_node_error (the "normal" terminal hook for an in-flight node that
    # gets cancelled), not silently dropped.
    assert context.terminal_hooks.get("node/queued1") == ["error"]
    assert context.terminal_hooks.get("node/queued2") == ["error"]

    await coordinator.release(held)


async def test_admission_cancellation_still_propagates_past_a_swallowing_hook():
    """Issue #277: on_node_error's normal "silence the error by returning an

    output" contract must not apply to a cancellation. If it did, a boundary
    cancellation would just vanish into a fabricated successful output
    instead of unwinding the cancelled task, which the issue calls out as
    worse than the leak the fix addresses.

    Drives ``Node.__call__`` directly (the same dispatch every execution
    algorithm uses) against a context whose ``admit_node`` always cancels,
    and whose ``on_node_error`` tries to swallow the error by returning a
    replacement output. The call must still raise ``CancelledError``.
    """

    class CancelOnAdmitContext(InMemoryExecutionContext):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.node_starts: list[str] = []
            self.terminal_hooks: dict[str, list[str]] = {}

        @override
        async def on_node_start(self, *, node, input_type, output_type, input):
            self.node_starts.append(node.id)
            return None

        @override
        def admit_node(self, node):
            @asynccontextmanager
            async def _cancel_immediately():
                raise asyncio.CancelledError
                yield  # pragma: no cover - unreachable, satisfies the CM protocol

            return _cancel_immediately()

        @override
        async def on_node_finish(
            self, *, node, input_type, output_type, input, output
        ) -> DataMapping:
            self.terminal_hooks.setdefault(node.id, []).append("finish")
            return output

        @override
        async def on_node_error(
            self, *, node, input_type, output_type, input, exception
        ) -> WorkflowException | DataMapping:
            self.terminal_hooks.setdefault(node.id, []).append("error")
            # Try to silence the error by returning a replacement output.
            # This must be ignored for a cancellation.
            return {}

    engine = WorkflowEngine()
    context = CancelOnAdmitContext()
    node = engine.create_node(LimitedProbeNode, id="n1")

    with pytest.raises(asyncio.CancelledError):
        await node(context=context, input_type=Empty, output_type=Empty, input={})

    assert context.node_starts == ["n1"]
    assert context.terminal_hooks == {"n1": ["error"]}


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
