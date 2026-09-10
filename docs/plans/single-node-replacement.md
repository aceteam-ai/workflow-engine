# Single-node replacement (#174)

Status: proposed design; this document does not change runtime behavior.

## Problem and decision

A delegating node currently returns a complete `Workflow`, even when its body is
one other node with compatible input and output fields. Both executors then
schedule the wrapper's input and output nodes. Allow `Node.run()` to return a
`Node` directly, validate the delegation contract, and schedule the returned node
as visible work in the existing flat execution.

The caller remains the logical output slot used by its existing downstream
edges. Its replacement gets a distinct child ID. A small runtime continuation
connects their completion; it is scheduler metadata, not another execution
algorithm or a synthetic executable input/output pair. This follows the flat-ID,
ledger, and pinning requirements in
[discussion #198](https://github.com/aceteam-ai/workflow-engine/discussions/198).

Do not initially promise execution on the same worker. Removing wrapper nodes
already removes their scheduling and lifecycle overhead. An optional worker
trampoline can follow measurement, subject to the same admission, cancellation,
and fairness rules.

## Public author and context APIs

Extend the existing return annotation without changing call arguments:

```python
async def run(
    self,
    *,
    context: ExecutionContext,
    input_type: type[Input],
    output_type: type[Output],
    input: Input,
) -> Output | Workflow | Node:
    return ChosenImplementation(id="implementation", params=self.params.target)
```

The returned node is immutable configuration. The engine does not call its
`run()` from inside the caller's `run()`. Authors needing input remapping or
constant injection still return a Workflow; direct replacement uses matching
field names and the caller's already-cast input.

Extend `on_node_start`'s existing override union to include `Node`, so a host can
supply a recorded replacement as it already supplies cached output or an
expanded workflow. Both sources go through the same validation. Add an
observational `on_node_replace(node, replacement, input, replacement_info)` hook,
returning `None`. It runs after validation and normalization but before dispatch.
`replacement_info` contains the logical caller ID, immediate delegator ID,
normalized replacement ID, and hop number. It contains no executable callback.

A context needing crash-safe replay durably records this event before returning.
If the hook fails, no replacement is dispatched. The default context retains
ordinary in-memory behavior; the core does not claim a durable ledger on its own.

## Type compatibility and value adaptation

Resolve the replacement's dynamic input and output schemas through the same
`ValidationContext` and registries as ordinary workflow nodes. Its concrete type,
version, params, and nested workflow params must pass their usual validation.
A raw unresolved `Node` is dispatched through the registry before checking types.

Given caller `A -> B` and replacement `C -> D`, require:

1. Every required input field of C exists in A. Additional defaulted C fields
   may use their defaults; nullable fields without defaults remain required.
2. For every field supplied from A to C, A's declared Value type is assignable
   to C's declared Value type using the existing engine compatibility rules.
   Caller fields that C does not accept are projected away.
3. Every required output field of B exists in D, with a Value type assignable
   to B's declared type. Missing defaulted B fields use B's defaults; extra
   D fields are projected away.

Apply these checks before invoking the replacement. An incompatible declaration
raises a builder-visible `NodeReplacementException`, classified `VALIDATION`,
on the delegating node, naming the field and the incompatible contracts. It is
an ordinary failure that an enclosing Attempt may contain. Register this engine
error name in the public error contract as part of implementation.

At runtime, preserve both adaptation stages: incoming values have already been
cast to A before the caller runs; project/cast those values to C and instantiate
C to materialize defaults and validators. After the replacement's finish hook,
project/cast D to B and instantiate B before publishing the caller's output.
Apply the same final contract check after the caller's finish hook, whose
existing output override remains supported. Cache overrides pass these checks
too. A failing concrete value cast is classified as validation at the relevant
adapter, retaining the original cause.

Do not simply retarget existing edges to the replacement's declared types.
Casting is not assumed to be transitive: a valid A-to-C-to-D-to-B delegation
must preserve those intermediate casts even when a downstream type would accept
D directly. Tagged Result values remain tagged; replacement never adds an ok
wrapper, unwraps an err, or invents a missing output value.

## Flat IDs and scheduler representation

For logical node `choose`, the first replacement runs as `choose/replacement_0`.
Further replacements in that same chain use `choose/replacement_1`, and so on.
Normalize the returned instance with `model_copy(update={"id": assigned_id})`;
its supplied ID is a local author label recorded in replacement provenance, not
an alternate runtime address. Params, registered type/version, functional node
configuration, and hints are preserved. Do not inherit the caller's hint pins or
retry override onto a different implementation implicitly.

Check the assigned ID against all existing flat nodes and control records before
installation. A collision is a validation failure; never choose a different ID
based on dispatch order. If `choose/replacement_1` returns a Workflow, ordinary
expansion places its children beneath `choose/replacement_1/...`. Each delegated
node, expanded descendant, and original caller remains individually addressable
in the host ledger. The checkpoint format must record the replacement relation
rather than infer it by stripping path components.

Introduce an internal immutable `ReplacementFrame` and a shared tracker used by
both executors. A frame records the logical output slot, each delegator and its
typed input/output contracts, the active replacement ID, hop count, and completed
adaptations. Installing a frame marks the caller pending, installs the real
replacement node in the runtime flat node/type maps, and seeds its already
validated input. The caller is excluded from ordinary ready-node discovery
until the frame reaches a terminal state.

On success, the tracker applies output adapters from the innermost delegator
outward, publishing each completed logical slot once. Only then does normal
successor discovery release that slot's outgoing edges. This is a completion
relation, not a dataflow edge or an executable adapter node. Expanded replacement
workflows complete through their existing designated output node. Detached
subgraph work keeps the existing flat scheduler's semantics; replacement does
not introduce an implicit error boundary or a new all-descendants barrier.

The tracker must support replacement from inside an already expanded workflow
and nested chains owned by different logical callers. No mutable frame is stored
on a reusable Node or shared ExecutionAlgorithm instance.

## Lifecycle, failures, retries, and limits

Each actual replacement invocation gets the ordinary start, finish, error,
retry, and yield hooks under its normalized ID. A delegator's start is followed
by its replacement event and, on successful completion, exactly one finish hook
with its adapted output. Existing Workflow-returning nodes keep their current
hook behavior; filling their existing deferred-finish TODO is a separate change.

A terminal child failure is processed once at the failing child, retaining its
error name, class, and flat node ID. The tracker marks pending delegators failed
with a reference to that failure; it does not call their `on_node_error` hooks
again, which would add implicit recovery policies or violate the current
exception node-ID check. Add an observational
`on_node_replacement_failed(node, replacement_info, exception)` hook for this
ledger transition. If a child error hook supplies a valid output, ordinary
successful adaptation follows instead. Output-adapter or delegator-finish errors
belong to that delegator and use its normal error hook once.

`ShouldRetry` retries the same active replacement ID with its own effective
instance/type/executor budget. It does not rerun the delegator, increment the
replacement hop, or reset a retry budget. `ShouldYield` leaves the frame pending
and reports the actual child ID in `node_yields`. Neither signal is converted to
a Result err by replacement. Exhausted courtesy retries and actual failures
continue through the nearest existing Attempt boundary, whose separate retries
create fresh attempt namespaces as usual.

Boundary lookup includes all normalized replacement IDs. A failed boundary
blocks queued replacements and their pending adapters; already admitted work
obeys the existing drain contract. A successful in-flight child cannot publish a
caller output after its enclosing boundary has failed. Cancellation and terminal
failure clear frames and pending inputs without pretending they succeeded.

Release the delegator's execution quota before admitting the replacement under
its own type and region policy. This avoids holding a scarce slot while waiting
for the delegated implementation. The proposed
[context-owned limits](context-owned-limits.md) apply to every actual invocation;
replacement never bypasses metering or grants permission for boundary retries.

Use an iterative tracker, not recursive Python calls. Reject returning `self`
with a validation error and add an explicit executor option
`max_replacement_hops=256`, recorded in run configuration. The count persists
across yields and resumes. Exceeding it is a validation failure at the current
delegator. Repeated equivalent configs need not indicate a loop, so fingerprints
serve replay validation rather than speculative cycle detection. Each installed
hop returns control to scheduler admission/fairness, even if a later optimization
can reuse a worker.

## Checkpoint, pinning, and serialization

A checkpoint records the immutable normalized replacement node, immediate parent
and logical caller IDs, hop, input fingerprint, resolved schema fingerprints,
node versions, completed slots, and active retry state. Serialize nodes and
Values with the canonical registry-backed representation; do not pickle Python
type objects or introduce a public edge variant. Host references obey normal
export resolution rules, and ignoring hints does not change delegation results.

On replay, a context supplies the recorded node from the delegator's
`on_node_start` hook, validates it against the recorded input and schema/version
fingerprints, and installs the same frame and IDs before consulting child pins
or cached results. A mismatch is a resume validation failure, not silent fresh
execution. Durable hosts must persist the replacement event before child
dispatch so a crash cannot lose the selected implementation after its external
side effect. As with ordinary nodes, this does not guarantee exactly-once remote
work without host idempotency support.

A valid cached final output or pin on the original caller short-circuits the
entire chain. A child pin requires the recorded replacement relation to be
replayed first; its adapted output still passes every remaining caller contract.
A live frame may be reconstructed from journal events or checkpoint state, but
not guessed from node-ID prefixes. Persist the consumed hop and retry counters
so repeated yield/resume cannot create a fresh budget.

## Compatibility, rollout, and verification

1. Add the Node return union, replacement validation/outcome, shared frame
   tracker, lifecycle hooks, and a versioned checkpoint event definition. Keep
   ordinary output and Workflow returns unchanged.
2. Integrate the same tracker with both executors, including boundary blocking,
   descendant expansion completion, and cancellation. Land registry-backed
   round-trip tests with this stage before allowing host replay.
3. Add a delegating-node authoring example and host ledger/pinning example.
   Existing custom algorithms must opt into the new outcome protocol; publish
   that interface change and a clear unsupported-outcome error. Existing graphs
   and ordinary Node subclasses need no migration or node-version bump.
4. Benchmark direct delegation against a one-node Workflow wrapper. Consider a
   same-worker trampoline only if it preserves observable admission and fairness.

Test both executors with exact hook traces and deterministic barriers. Cover
required/defaulted/nullable inputs and outputs; field projection; enum/custom
Value identity; nontransitive casts; tagged Result preservation; dynamic schemas;
invalid nested workflow params; cache and hook overrides; ID normalization and
collisions; replacement chains; replacement followed by expansion; nested
Attempt success, failure, cancellation, and drain; and distinct type/instance
retry budgets without delegator redispatch.

Persist and reload real JSON checkpoints after replacement installation, after
a child yields, and after an inner adapter completes. Assert stable child IDs,
caller and child pins, no repeated completion hooks within one execution pass,
remaining retry/hop exhaustion, stale-input/schema rejection, and cache-hit
short-circuiting. Use a long chain to prove bounded Python stack usage and
cancellation responsiveness. Compare lifecycle counts and allocation/scheduling
overhead with the wrapper baseline rather than asserting unmeasured speedups.
