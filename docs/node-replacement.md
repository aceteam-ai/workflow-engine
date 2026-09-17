# Returning a replacement node

A node can return a `Node` when another implementation accepts its input fields
and supplies its output fields. Both built-in execution algorithms dispatch that
node as ordinary visible work. There are no synthetic input/output adapter nodes
and no recursive call to the child's `run()`.

```python
async def run(self, *, context, input_type, output_type, input):
    return context.validation_context.node_registry.create_node(
        ChosenImplementation,
        id="implementation",
        params={"setting": self.params.setting},
    )
```

Use a `Workflow` instead when delegation needs input renaming, constant injection,
or multiple connected nodes. Returning `self` is rejected.

## Contracts and IDs

For a caller `A -> B` delegating to `C -> D`, every required C field must exist in
A and be assignable to C's field type. Every required B field must exist in D and
be assignable to B's field type. Extra fields are projected away and defaulted
fields may be absent. Nullable fields without defaults remain required.

The engine preserves every cast stage: upstream values are first cast to A,
then projected/cast into an instantiated C. After the child's finish hook, D is
validated and projected/cast into B. The caller's finish-hook override is also
validated as B. This matters for custom casts: direct upstream-to-C or
D-to-downstream conversion is not equivalent in general. Registered Value
identity and tagged `Result` values are preserved. Cached caller outputs are
validated against the caller's contract before and after its finish hook.

For caller `choose`, successive replacements run as `choose/replacement_0`,
`choose/replacement_1`, and so on. Their supplied IDs are recorded as author
labels; they are not runtime addresses. Functional configuration, parameters,
version, hints, and the replacement's own retry override are preserved. Caller
hints and retry overrides are not inherited. ID collisions fail validation.

The original caller remains the output slot used by its existing edges. A
run-local completion tracker publishes each intermediate caller once after
applying its output adapter. Replacement inside an expanded workflow works in
that workflow's namespace. A replacement returning a Workflow expands beneath
its assigned ID and completes through the workflow's designated output node.
Detached work retains ordinary workflow scheduling semantics; replacement adds
no implicit all-descendants barrier.

## Lifecycle and failure

A successful direct delegation produces these hooks:

1. Caller `on_node_start`.
2. Caller `on_node_replace`, with validated, normalized replacement and frame.
3. Child start and finish hooks under its assigned ID.
4. Caller finish hook with its adapted output.

The replacement event is observational: returning it cannot change the child.
A failing event hook prevents dispatch. Ordinary Workflow-returning nodes retain
their existing lifecycle; this feature does not add their deferred finish hook.

A child failure uses its own error hook and retains its error name, class, and
node ID. Pending callers receive `on_node_replacement_failed`, not duplicate
error/recovery hooks. A valid child error-hook output instead completes the
normal adapters. An adapter or caller-finish failure belongs to that caller and
uses its error hook once. Incompatible declarations, invalid cast values,
checkpoint mismatches, self replacement, collisions and exhausted hop limits
raise the builder-visible `NodeReplacementException` with class `validation`.

`ShouldRetry` repeats the actual child with its own effective retry budget; it
does not repeat the selector or consume another replacement hop. `ShouldYield`
reports the child ID and leaves the relation pending. Existing Attempt boundaries
contain failures, hold on yields, and drain in-flight children normally. After a
boundary fails, its pending callers cannot publish successful outputs. Attempt
retry consent also applies to newly revealed replacement nodes: a metered target
cannot bypass `allow_metered` by being absent from the authored graph.

A control signal from the caller's deferred output adaptation or finish hook
belongs to that caller. `ShouldYield` reports its logical ID and keeps the frame
pending. `ShouldRetry` uses the caller's remaining retry budget and resumes only
its adaptation and finish hook after backoff; neither the selector nor the child
body runs again. That retry state is checkpointed with the frame. On a new
execution pass, hosts replay the relation and cache completed child outputs as
usual. Exhausted completion retries remain attributable to the caller and flow
through its enclosing Attempt boundary.

Both algorithms accept `max_replacement_hops=256`. It must be positive. Hops are
iterative, and the scheduler regains control between admissions. There is no
same-worker or thread-local execution guarantee.

## Durable replay and pins

`ReplacementFrame` is a versioned, JSON-serializable immutable checkpoint. It
contains the normalized Node, caller relation and author label, hop/limit,
registry-backed input/output schemas, input/configuration/schema fingerprints,
retry attempts and next eligible times, completed slots and final output JSON.
No Python type object or callback is serialized.

Core supplies these optional context hooks:

- `on_node_replace(node, replacement, input, replacement_info)` runs after
  validation and before dispatch. Durably record the frame before returning.
- `on_node_replacement_checkpoint(frame)` records retry and completion changes.
- `get_node_replacement_frame(node_id)` supplies the saved frame when a recorded
  replacement is replayed from `on_node_start`.
- `on_node_replacement_failed(node, replacement_info, exception)` observes the
  terminal ledger transition while preserving the actual failure's provenance.

A host can implement its start hook with this pattern:

```python
async def on_node_start(self, *, node, input_type, output_type, input):
    frame_json = await self.ledger.get_replacement(node.id)
    if frame_json is None:
        return None
    frame = ReplacementFrame.model_validate_json(frame_json)
    return frame.replay(
        node=node, input_type=input_type, output_type=output_type, input=input
    )
```

`frame.replay` validates the caller configuration, inputs, and schemas. It returns
the recorded Node for a pending frame, or typed output for a completed frame.
The tracker validates the installed child contracts, hop and limit again before
dispatch. A mismatch or terminal failed frame is rejected. Save frames using both
replacement-event and checkpoint hooks, and return them from
`get_node_replacement_frame`; replaying a Node alone is insufficient to restore
retry history.

A valid original-caller pin bypasses the whole chain. To use a child pin, replay
the recorded relation first, then return cached output for the normalized child
ID. Its output still passes through every remaining caller contract. Completed
intermediate frames can supply their outputs without repeating selection or
child work after a crash. Journal records must be scoped to one immutable run:
ordinary workflow inputs, cached outputs, and executor configuration must not
be silently reused for a different run.

The engine owns frame/event shape and validation, not a durable store. Hosts own
atomic persistence before external dispatch, general node-output caching,
unchanged executor settings on resume, stale-run exclusion, and remote-work
idempotency. Hooks may be delivered again across execution passes; upsert by run
and delegator ID. This does not guarantee exactly-once external side effects.

## Custom execution algorithms

`Node.__call__` accepts `allow_replacement=True` and may then return the
`NodeReplacement` outcome from `workflow_engine.core.replacement`. Algorithms
must install and complete its relation, admit the child normally, and retain
both contract adapters. A custom algorithm that does not opt in receives an
explicit unsupported-replacement error. Ordinary Data and Workflow outcomes
retain their existing protocol.

The built-in shared tracker is in `execution/replacement.py`; it is runtime
implementation machinery, not a second executor or a public edge type.

## Benchmark

Run `PYTHONPATH=src python scripts/benchmark_node_replacement.py` to compare a
pure identity replacement with the same leaf inside a one-node Workflow. It
reports median public-engine execution time and hook counts for both schedulers,
after warmup, with no external I/O or durable checkpoint writes.

A local Python 3.12 run with 100 measured iterations and 10 warmup iterations
produced the following observations (milliseconds):

| Scheduler | Direct | Workflow wrapper | Direct start/finish hooks | Wrapper start/finish hooks |
| --- | ---: | ---: | ---: | ---: |
| Topological | 2.246 | 1.740 | 4 / 4 | 6 / 5 |
| Parallel | 2.468 | 1.954 | 4 / 4 | 6 / 5 |

Direct replacement removes two scheduled wrapper nodes and adds the caller's
deferred finish hook. In this small local workload, contract checks and frame
serialization cost more than the removed scheduling work. These measurements
do not establish a latency improvement or predict remote-worker overhead.
