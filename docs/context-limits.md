# Context-owned execution limits

Limits apply around `Node.run`, after `on_node_start` has had an opportunity to
return a cached output. A node author supplies defaults on `NodeTypeInfo`:

```python
TYPE_INFO = NodeTypeInfo.from_parameter_type(
    display_name="Provider request",
    version="1.0.0",
    parameter_type=RequestParams,
    execution_limits=RateLimitConfig(max_concurrency=4),
    limit_regions={
        "provider": RateLimitConfig(requests_per_window=60),
    },
)
```

Counts and window durations must be positive. Omitted counts mean unlimited.
To restrict a named section of code, use
`async with context.limit(node=self, region="provider"):` inside `run`.
Undeclared regions are errors. Nested regions mapped to an already-held pool
share its lease without charging another request. Distinct nested pools must
be acquired in canonical pool-key order to prevent circular waits.

Operators pass a `LimitPolicy` to the execution context. An override replaces
the complete author default, including an explicit unlimited `RateLimitConfig()`:

```python
coordinator = InMemoryLimitCoordinator()
policy = LimitPolicy(
    namespace="tenant-a",
    overrides={
        LimitPolicy.scope(ProviderNode): RateLimitConfig(max_concurrency=2),
    },
)
context = InMemoryExecutionContext(
    limit_coordinator=coordinator,
    limit_policy=policy,
)
```

Share the same coordinator among contexts to enforce quotas across concurrent
algorithms and event loops in one process. Independent coordinators have
independent quotas. This backend does not persist state across process restarts.
Contexts default to their own coordinator when none is supplied.

Scopes use the implementation's module and qualified class name plus region,
not a graph ID, mounted alias, or node version. `LimitPolicy.pools` can map several
scopes to one host-owned pool. Their effective configurations and policy revision
must agree; conflicting configurations fail closed. Policy keys, namespaces,
and backend configuration belong to the host, never to graph hints. Dynamic
policy replacement is not an implicit side effect of acquiring a lease.

Rate and concurrency capacity are granted in one transaction. Queued work
consumes neither. Rate history remains charged after dispatch even when the
node fails or is cancelled. Waiters recheck availability after every notification
or rate-window deadline. Among eligible bundles sharing a pool, older waiters
run first; a bundle blocked on a separate provider does not reserve the
executor's shared worker capacity.

`ParallelExecutionAlgorithm.max_concurrency` participates as another pool in
the same atomic admission bundle. Tasks awaiting initial admission do not occupy
that capacity. They suspend their coroutine rather than occupying a worker
thread. Entering a narrower region temporarily returns only scheduler capacity;
the region and worker capacity are then admitted together. On region exit the
worker capacity transfers back to the parent lease atomically. The node retains
its whole-node provider quota and rate history throughout. A failed Attempt
cancels queued members and
drains admitted members, retaining the existing boundary semantics.

The context distinguishes admission from the existing cache/start hook:

- `on_node_waiting_for_limit(node, invocation_id, wait_info)` runs once when
  whole-node admission must wait.
- `on_node_admitted(node, invocation_id, lease)` runs after the grant and before
  `run`, including nodes with no configured quota.
- Region managers report `on_limit_wait`, `on_limit_acquired`, and
  `on_limit_released` with the region name.

These hooks are observational and do not replace node output. Wait information
contains the resolved policy/revision, invocation identity, and an optional retry
delay. `inspect()` on a coordinator returns active leases, queued invocation IDs,
and request counts for host diagnostics.

Every courtesy retry reacquires capacity. Backoff and yielded nodes retain no
concurrency lease. Boundary retries keep their existing distinct child IDs and
metered consent; quota admission never changes either retry budget or turns a
wait into `ShouldRetry`. Cache hits consume no admission. Invocation identities
include context run identity, execution pass, flat node ID, dispatch count, and
region ordinal. Hosts resuming remote work must reconcile its existing external
job before permitting a new dispatch; the coordinator cannot stop an external
service's old request merely because a local coroutine has yielded.

For compatibility, executor `rate_limits=RateLimitRegistry(...)` configuration
is translated into context admission; it no longer wraps node execution in the
old limiter. A shared legacy registry has a shared in-process coordinator unless
the context supplies an explicit one. Conflicting legacy and context overrides
are rejected. The old `execution.rate_limit.RateLimitConfig` import remains an
alias of the validated core model. New integrations should configure their
context directly.

## Durable local coordination

`SQLiteLimitCoordinator` persists quota state in an operator-selected local file:

```python
from workflow_engine.limits import SQLiteLimitCoordinator

async with SQLiteLimitCoordinator("/var/lib/my-host/limits.db") as coordinator:
    context = InMemoryExecutionContext(limit_coordinator=coordinator)
    await engine.execute(context=context, workflow=workflow, input=input_data)
```

Create the parent directory first and share the path among local worker
processes. Contexts in a process should generally share one coordinator object.
The implementation uses a bounded one-thread I/O executor for short SQLite
transactions. Database connections close after each transaction; a queued node
holds no connection or I/O thread while awaiting capacity. Same-object release
notifications wake waiters immediately. Independent coordinators/processes use
an asynchronous refresh timer (default 100 ms) to discover remote changes and
recover lost notifications. No thread busy-waits or remains open in a transaction.

The store has one versioned `wengine_limit_state` row containing policy revisions,
request timestamps, pending tickets, active leases, the fencing generation, and
a nondecreasing clock floor. Each update runs under `BEGIN IMMEDIATE` and commits
atomically. This initial representation favors a small auditable transaction
boundary; transaction work grows with retained admission state. Hosts should
measure their expected fan-out and database contention before adopting it for
large installations. A shared/network filesystem is not supported as a
cross-machine coordinator; distributed hosts implement the same coordinator
protocol against their own atomic storage service.

Leases expire after 30 seconds by default. Contexts renew active leases every
third of that duration. Wait tickets have a bounded lifetime and are refreshed
while their owner waits, so a crashed waiter does not block the queue forever.
A coordinator restart preserves charged rate history. Expired leases release
concurrency capacity and a later grant receives a larger fencing generation.
The durable clock floor prevents a clock rollback from refunding requests.
The refresh interval must be positive and at most one third of the lease duration.

Storage failures, incompatible schema versions, conflicting policies, and lease
loss fail closed. If renewal fails, the context cancels local execution and
reports an operator error. Remote operations may outlive that cancellation or
lease expiry: hard remote concurrency requires the provider to enforce the lease
identity/fencing generation or a host to reconcile the outstanding operation.
This backend does not claim exactly-once remote execution. Closing the shared
coordinator is the host's responsibility after its contexts have settled.

For recovery, retain the database together with the host run ledger; do not clear
it to work around a policy mismatch or temporary outage, since that would erase
rate history. Resume with the same pool keys and effective configuration. The
first implementation rejects mixed live policy revisions; coordinated policy
migration and a normalized high-volume storage layout are separate host/backend
work, not implicit behavior of `acquire`.
