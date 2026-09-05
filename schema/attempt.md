# `attempt` wire shape and boundary semantics

`Attempt` (`AttemptNode`) runs an inner workflow inside an error boundary,
producing a [`Result[T]`](result.md) instead of letting a member's failure
propagate to the run. See discussion
[#198](https://github.com/aceteam-ai/workflow-engine/discussions/198) for the
motivation and #201 for the design.

This document covers the node's wire shape, the flat id table an expansion
produces, the reserved `ok` id, the boundary semantics, and the two
`ExecutionContext` hooks a host uses to observe a boundary. See
[`docs/execution.md`](../docs/execution.md#error-boundaries) for the
execution-algorithm-level walkthrough and
[`docs/contexts.md`](../docs/contexts.md#node-level-hooks) for every hook
signature.

## Node type

| | |
| --- | --- |
| Wire type name | `"Attempt"` |
| Params | `{ "workflow": <inline Workflow value> }` |
| Input | the inner workflow's own input type, `A` |
| Output | `{ "result": Result[B] }` |

```json
{
  "type": "Attempt",
  "id": "attempt",
  "params": {
    "workflow": { "input_node": ..., "inner_nodes": [...], "output_node": ..., "edges": [...] }
  }
}
```

This is the same shape `ForEach` uses for its own `workflow` param: an inline,
serialized `Workflow` value, not a reference. Wrapping an `Attempt` node in
the standard input/output nodes (as `WorkflowEngine.build_single_node_workflow`
does) gives a workflow of type `A -> Result[B]`.

### The `B` rule

`B` is derived from the inner workflow's own output type, the same rule
`ForEach` uses to collapse its per-element output:

- Exactly one output field: `B` is that field's own value type.
- Zero or more than one output field: `B` is `DataValue[InnerOutputType]`,
  the whole output record wrapped as a single value.

This means `Result[T]`'s own nesting rule applies here too: if the inner
workflow's single output field is itself a `Result[X]` (e.g. a nested
`attempt`), `B` is `Result[X]` directly, and the outer `attempt`'s own
`Result[B]` becomes `Result[Result[X]]` with both tags preserved
independently.

### `for_each(attempt(w))`

There is no dedicated combinator; compose the two node types directly:

```python
attempted = await engine.build_single_node_workflow(
    AttemptNode, node_id="attempt", params={"workflow": w},
)
# attempted : A -> {result: Result[B]}
for_each = engine.create_node(ForEachNode, id="for_each", params={"workflow": attempted})
# for_each : Seq[A] -> Seq[Result[B]]
```

`ForEach`'s own output-collapsing rule sees `attempted`'s single output
field (`result: Result[B]`) and produces `Seq[Result[B]]` directly, the
standard shape the eliminators (`partition`, `unwrap_or`, `all_ok`,
`first_error`, see [`docs/nodes.md`](../docs/nodes.md)) consume.

## Flat ids

`AttemptNode.run` builds a `Workflow` whose `input_node` is the inner
workflow's own input node, verbatim; whose `inner_nodes` are the inner
workflow's own inner nodes plus one synthetic `Ok` node with the reserved id
`ok`; and whose `output_node` is a new output node with a single `result`
field. The inner workflow's own output node is dropped; every edge that
targeted it is retargeted to `ok` instead, and `ok`'s own output feeds the
new output node.

The executor expands this the same way it expands any composite node:
namespaced under the `Attempt` node's own id, with the inner and output
nodes kept as ordinary flat nodes rather than a nested subgraph. For
`for_each(attempt(w))` where `w` has one inner node `subworkflow`, the
resulting ids are:

```
for_each                                   ForEachNode (expands, then removed)
for_each/expand
for_each/gather
for_each/element_16/attempt                AttemptNode (expands, then removed) = boundary id
for_each/element_16/attempt/input          w's own input node, pass-through
for_each/element_16/attempt/subworkflow    w's own inner node, verbatim
for_each/element_16/attempt/ok             the ok arm
for_each/element_16/attempt/output         the boundary's own output node; err is materialized here
```

Boundary membership is a plain string-prefix test on these flat ids:
`node_id == boundary_id or node_id.startswith(boundary_id + "/")`. This is
the same rule `Workflow` enforces at validation time to prevent id
collisions between a composite node and its own future expansion, so ids
stay flat and downstream ledger, resume, and pin machinery that keys off of
them needs no changes to support `attempt`.

### The reserved `ok` id

Because the synthetic `Ok` node is always inserted at id `ok` inside the
boundary, an inner workflow with its own node literally named `ok` collides.
Rename it; `AttemptNode` rejects the collision at expansion time with a
builder-level error naming the offending node.

## Boundary semantics

- **Not a third error-handling mode.** Boundary containment is orthogonal
  to the parallel executor's `FAIL_FAST` / `CONTINUE` modes: a
  boundary-contained error never reaches run-level `errors` in either mode.
- **Yield wins.** If any member of the boundary has yielded in the current
  execution pass, the boundary does not materialize `err` in that pass. New
  scheduling inside the boundary still stops, in-flight members drain to
  their own normal completion, the run returns `YIELDED`, and the boundary
  re-expands and is re-evaluated from scratch on the resume pass.
- **Drain, not kill.** A member already dispatched when its boundary fails
  runs to its own completion; nothing is force-cancelled.
- **Innermost catches.** A failure fails the innermost boundary enclosing
  it. A nested `attempt`'s `err` is an ordinary value to its enclosing
  boundary; an enclosing boundary's own direct failure blocks and eventually
  sweeps everything nested under it.

## The err arm

`materialize_error` builds `{"result": Result[B].err(error)}` where `error`
is a [`ResultError`](result.md#resulterror-the-err-arm) built from the
failing exception:

| `ResultError` field | source |
| --- | --- |
| `node_id` | the failing member's own flat id |
| `name` | the class name of the root cause (the exception's `__cause__` chain walked to its end) |
| `message` | the exception's own message if it is already `USER` level; otherwise `"An internal error occurred"` |
| `error_class` | the exception's own `error_class` if set, otherwise `"systemic"` |

`message` redaction matters here specifically because a materialized `err`
value flows straight into user-visible workflow output, unlike
`WorkflowErrors`, which every viewer filters through `WorkflowError.filter`
before it is rendered.

## Hooks

Both hooks have working default implementations; a host that does not
override them is unaffected.

### `on_node_cancelled`

Fires once per pass for each member of a failed boundary that will not run
this pass.

```python
async def on_node_cancelled(
    self, *, node, input_type, output_type, input, boundary_id, reason, cause,
) -> None: ...
```

| Field | Meaning |
| --- | --- |
| `boundary_id` | the flat id of the boundary the member belongs to |
| `reason` | `"not_scheduled"`: the boundary failed before this member was dispatched, `input` is `None`. `"retry_abandoned"`: the member was in `ShouldRetry` backoff and is not re-dispatched, `input` is its input. |
| `cause` | the exception that failed the boundary; `cause.node_id` is the sibling that actually failed, not necessarily `node.id` |

Never fires for a member that was in flight (it gets its normal
`on_node_finish` / `on_node_error` / `on_node_yield` instead), for a yielded
member, or for the boundary's own output node. In a held pass (yield wins),
this disposition is per pass: a member reported cancelled here may still run
on a later, resumed pass.

### `on_boundary_error`

Fires when a boundary materializes its `err` arm, after every member has
settled and none yielded this pass.

```python
async def on_boundary_error(
    self, *, node, input_type, output_type, input, error, output, cause,
) -> DataMapping:
    return output
```

`node` is the boundary node itself (e.g. the `AttemptNode`); `input` is the
input it was expanded with; `output` is the mapping about to be written as
the output of the boundary's own output node, returnable (possibly
replaced) the same way `on_node_finish` works. A host that persists `output`
against `node.id` may short-circuit the whole boundary from `on_node_start`
on a later pass: that is safe specifically because nothing inside the
boundary is left suspended when this hook fires.
