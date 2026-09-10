# Attempt retries

`Attempt` gains an optional nonnegative `retries` budget (default zero), a
`retry_on` list (default timeout, unreachable, rate_limit), and an explicit
`allow_metered` flag (default false). These are functional parameters and
survive removing hints. Validation rejects a positive budget over any declared
metered node unless the flag is true, including nodes in nested workflow params.
Node authors declare metered execution with `NodeTypeInfo.metered`; authors of
opaque dynamic or externally dispatched work must mark their owning node when
that work may incur a charge. The engine cannot infer external billing from
Python code or a host callback. Unknown failures remain systemic and do not
retry by default; authors can include that class explicitly.

Retries compose ordinary single-shot boundaries and a runtime continuation
node. The first child is `attempt/try_0`; later children are
`attempt/next/try_1`, `attempt/next/next/try_2`, and so on. `retries=0` preserves
existing IDs and behavior. Each child keeps the existing drain, yield, and
innermost-boundary semantics. Only a materialized err with a matching class
advances the continuation. No executor or RetryTracker reset is needed: every
try receives its own courtesy budget under a distinct node ID. Already
successful work inside an earlier try is dispatched again under the new ID.
An enclosing workflow's work outside the Attempt is not replayed.

This costs one continuation and one single-shot boundary per try. It avoids
mutable graph rollback, reusing cache keys for newly charged work, and a second
scheduler inside a node. A nested Attempt owns its own independent budget;
its err is ordinary data until explicitly unwrapped by an enclosing workflow.
The original inputs travel on typed edges to every continuation, so a workflow
with an input named `result` remains valid.

`on_boundary_error` reports each failed single-shot child. `on_boundary_retry`
reports the original boundary ID, the continuation node, the next attempt
number (one-based), total retry budget, metered consent, and the materialized
error before dispatch. These hooks are separate from `on_node_retry`; the
latter continues to count only ShouldRetry courtesy attempts. Final exhaustion
passes through the last ResultError unchanged, including its attempt-specific
root node ID. Successful or exhausted continuations have normal node finish
hooks. Hosts may use those and child boundary hooks for caching.

Budgets are part of the serialized graph, and consumed attempts are encoded in
continuation params and deterministic IDs. A yielded child has no materialized
Result, so it cannot advance to another try. On resume hosts must retain the
normal node-output and boundary-result cache; re-executing from an empty context
is a new execution, as with any workflow. Replayed retry hook events share the
same `(boundary_id, attempt)` identity: persistent ledgers must upsert that key
within their run, rather than count hook delivery as spend. Actual dispatches
have distinct attempt IDs. The budget itself never increases during replay.
