# Context-owned execution limits (#183)

Status: proposed design; this document does not change runtime behavior.

## Problem and decision

Both executors currently own `RateLimitRegistry` and acquire a limiter before
calling the node. Separate algorithm instances normally create separate
registries, so their quotas do not coordinate. The limiter combines an
in-process asyncio semaphore with an in-memory list of request timestamps.
Waiting precedes the node lifecycle hooks, so hosts cannot reliably distinguish
queued work from executing work through those hooks.

Move admission policy and shared quota state behind `ExecutionContext`. Keep
one public configuration vocabulary for whole-node defaults, operator overrides,
and explicitly limited regions inside a node. A coordinator grants rate and
concurrency capacity atomically. Awaiting admission suspends a coroutine;
workers do not poll or sleep while holding execution slots.

This is an enforcement service, not a new error boundary or a graph hint.
The distinctions in [discussion #198](https://github.com/aceteam-ai/workflow-engine/discussions/198)
and [the hints contract](../../schema/hints.md) remain in force. Host storage,
credential, tenant, and machine identifiers never become portable hints.

## Proposed public surface

Retain the existing `RateLimitConfig` shape, add positive-value validation, and
move its definition to `core/limits.py`; keep its old import as an alias:

```python
class RateLimitConfig(ImmutableBaseModel):
    max_concurrency: int | None = None
    requests_per_window: int | None = None
    window_duration: timedelta = timedelta(seconds=60)

class NodeTypeInfo(ImmutableBaseModel):
    execution_limits: RateLimitConfig | None = None
    limit_regions: Mapping[str, RateLimitConfig] = Field(default_factory=dict)

class ExecutionContext:
    def __init__(self, *, limit_coordinator=None, limit_policy=None, ...): ...

    def limit(self, *, node: Node, region: str) -> AsyncContextManager[LimitLease]: ...
```

`execution_limits` covers one call of `node.run`, not a composite node's entire
expanded subtree. `limit_regions` names narrower regions, for example
`provider_request`; authors write `async with context.limit(node=self,
region="provider_request"):` around just that external call. A missing region is
a builder-visible configuration error rather than silently unlimited work.
Pure built-ins default to no limits. `None` means unlimited for each dimension;
configured counts and durations must be positive, and a window is relevant only
when its request count is set.

An operator supplies a `LimitPolicy` with complete per-scope overrides and an
optional mapping of several scopes to one physical quota pool. A configured
override replaces the entire author default; an explicitly unlimited override
is distinguishable from no override. Hosts may tighten or relax limits. The
resolved policy and revision are observable before admission, so changing an
operator limit cannot silently mean a different policy to two workers.

Scopes identify `(resolved implementation, region)`, with `execution` reserved
for the whole-node scope. The default implementation identity is module plus
qualified class name, independent of graph node ID, mounted alias, and node
version. Hosts map this to an authenticated tenant/provider pool when needed.
A graph-provided string cannot select another tenant's quota. Pool identity and
backend configuration belong to the context; author region names are portable.

The coordinator protocol accepts an authenticated pool key, effective policy
revision, and an invocation identity, and returns a lease or an awaitable wait
notification. It supports acquire, release, renew, cancel-wait, and inspection.
Acquire is idempotent for the same invocation. Public node code uses the context
manager rather than manipulating coordinator counters.

## Atomic admission and durability

A grant transaction must find capacity in **both** dimensions before recording
a request timestamp and issuing a concurrency lease. It must recheck the current
policy and window after waking. Do not consume a rate token while waiting for a
concurrency slot, or hold a concurrency slot while waiting for a rate window.
Normal release returns concurrency capacity; a dispatched request still counts
against the rate window after success, error, or cancellation.

Use FIFO tickets within one pool. Coordinator notification or the next known
window/lease deadline wakes waiters; waking is permission to recheck, not a grant.
Cancellation removes the ticket. A cancellation racing with a grant releases
its lease exactly once. Acquisition of a required bundle of pools is atomic in
canonical key order, so a node cannot hold half its admission bundle indefinitely.
The initial implementation supports bundles within a single coordinator backend.

Provide two explicit backend modes:

- `InMemoryLimitCoordinator`, for contexts deliberately sharing one coordinator
  in a process. It coordinates separate algorithms and event loops using
  thread-safe state and loop-specific wakeups. It does not claim restart or
  cross-process durability.
- `SQLiteLimitCoordinator`, the first durable local backend, using one
  operator-configured database for policy revisions, wait tickets, request
  timestamps, and leases. Short transactions run through a bounded I/O executor;
  no transaction or worker thread remains open while waiting for capacity.
  This coordinates multiple local processes and survives coordinator restarts.
  Cross-machine deployments supply an implementation of the same atomic protocol;
  a shared filesystem is not advertised as a distributed database.

Leases carry an opaque ID, owner invocation, expiry, and monotonically increasing
fencing generation. Long-running work renews its lease; crash recovery reclaims
expired leases. The durable backend owns the clock. Persist a nondecreasing
logical wall-clock floor so a backwards clock adjustment delays capacity rather
than granting extra requests. Test clock behavior with an injected clock.

Expiry alone cannot guarantee that an external provider stopped an old request.
After lease loss the context must stop local dispatch and report an operator-level
failure; hosts requiring hard concurrency bounds on remote jobs must propagate
fencing/lease identity to a resource that rejects stale owners. Document this
limit explicitly rather than promising exactly-once external execution.
Backend unavailability fails closed before new dispatch, with a classified
operator-level error; it does not silently fall back to process-local limits.

## Lifecycle and scheduler integration

Preserve `on_node_start` as the existing cache/override hook. A returned cached
mapping performs no `run` and consumes no execution admission. Before actually
calling `run`, the invocation awaits context admission, then emits
`on_node_admitted(node, invocation_id, lease)`; while queued it emits
`on_node_waiting_for_limit(node, invocation_id, wait_info)` once per wait episode.
Wait information contains a reason, scope, policy revision, queue ticket, and
optional earliest retry time, not backend credentials. Region context managers
emit equivalent `on_limit_wait`/`on_limit_acquired`/`on_limit_released` hooks with
the region name. These hooks are observational, with no alternate output channel.

The invocation identity includes the host's run ID, flat node ID, dispatch
attempt, and region acquisition ordinal. A host mints dispatch identities and
persists them when resuming external work; serialized graph IDs alone are not
unique across runs. Replayed ledger events upsert this identity. Admission wait
is a live invocation state, not `ShouldRetry`, `ShouldYield`, or a new
`WorkflowExecutionResultStatus`; transient waiting consumes no retry budget.

A parallel executor must not acquire its local running-worker permit before
resource admission. Represent the run-wide `max_concurrency` capacity as a local
execution pool in the same context admission bundle, so waiting for one scarce
provider does not reserve all execution slots. Once admitted, nodes still run
through the existing flat scheduler. The sequential executor naturally has one
active invocation; its wait suspends the coroutine rather than an OS thread.

Release whole-node leases in `finally` on success, expansion/replacement,
`ShouldRetry`, `ShouldYield`, exception, and task cancellation. Backoff holds no
concurrency lease. Every actual courtesy retry reacquires; an Attempt boundary
retry has its own child IDs and admission events. Rate-limit waiting never
creates a Result err unless admission itself fails.

A failed boundary prevents queued-but-not-admitted members from acquiring and
reports their normal cancellation disposition. Members admitted before failure
retain the existing drain contract. Atomic admission therefore defines the
transition from queued to in-flight. Resuming a yielded node must not reacquire
an old lease under a new identity while its remote work is still running;
external-job contexts either renew/recover the recorded lease or reconcile its
terminal state before dispatch.

Nested regions use distinct pools by default. Reentering an already-held pool
shares its lease and does not count another request; count another request by
exiting and entering again. Reject undeclared cycles or attempts to upgrade an
existing lease to a larger bundle while holding it. This avoids whole-node and
inner-region self-deadlock.

## Compatibility and rollout

1. Add validated config models, coordinator protocol, fake clock, and in-memory
   backend. Default contexts with no configured limits preserve current behavior.
2. Add SQLite persistence and process-crash tests before describing limits as
   durable. Publish the storage schema/version and recovery procedure.
3. Add context admission hooks and region manager, then move both executors'
   existing rate-limit call sites to one invocation admission path. Remove their
   duplicate acquire/release paths in that same change.
4. For one deprecation window, translate an algorithm's legacy `rate_limits`
   argument into a context policy adapter. Reject simultaneous conflicting
   context and legacy policies rather than enforcing both. Preserve old import
   paths. Legacy adapters are explicitly process-local unless backed by the new
   shared coordinator.
5. Exercise two algorithms and two resumed contexts sharing the same durable pool
   before retiring the legacy argument. Do not change hint erasure, Result shape,
   flat node IDs, or the two existing retry-accounting layers.

## Required verification

Use deterministic barriers/events and an injected clock, not sleep races.
Cover author default versus complete operator override; aliases sharing one
pool; tenant separation; atomic rate+concurrency grants; wakeup rechecking;
FIFO cancellation; zero admission for cache hits; all release paths; failed
boundary queue cancellation versus admitted-member drain; courtesy and boundary
retry budgets; region reentry; and unrelated providers making progress while a
pool is saturated.

Run concurrent algorithm instances against the same coordinator and separate
processes against the same SQLite database. Kill a lease owner, recover the
coordinator, and verify request history, expiry/fencing, policy revision, and
wait-ticket cleanup. Simulate lost notifications, grant/cancel races, lease
renewal failure, backend outage, and clock rollback. Verify flat-ID ledger and
yield/resume behavior with a persistent context. Measure transaction volume and
peak queued-task memory on a wide fan-out before enabling the durable backend
by default in any host.
