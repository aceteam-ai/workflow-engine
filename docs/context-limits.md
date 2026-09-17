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
thread. Once admitted, a node retains its whole-node quota across its execution,
including narrower region waits. A failed Attempt cancels queued members and
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
