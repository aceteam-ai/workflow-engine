# Bounded generation with Unfold

`Unfold` generates a sequence from a runtime seed. Its inline step workflow must
have exactly this signature:

```text
{seed: S} -> {items: Sequence[T], next: S, done: Boolean}
```

The node accepts `{seed: S}` and returns `{sequence: Sequence[T]}`. The seed and
next-cursor schemas must match. A cursor can be any portable Value, including a
record wrapped in `DataValue`. The item type can itself be `Result[T]`; Unfold
never interprets, skips, or replaces an error-tagged item.

The parameters are `workflow`, required positive `max_iterations`, and optional
`truncate` (explicit default `false`). Each page's items are included in order,
including those from the page that sets `done=true`. Empty pages contribute no
items and still count as iterations. A completed first page is a one-step run,
and may produce an empty result.

Completion is checked before the limit: a final page that sets `done=true` on
the last permitted iteration succeeds. Reaching the limit with `done=false`
raises a user-visible validation error, without fetching another page. With
`truncate=true`, the node instead returns all items generated within the limit.
The limit counts logical pages; existing executor retry accounting is separate.
Unfold adds no retries of its own. Ordinary step failures and yields retain
their normal meaning; use an explicit `Attempt` boundary to collect a failure.

## Flat expansion and checkpoints

An Unfold invocation expands one inline step and a continuation. The continuation
receives `items`, `next`, and `done` on ordinary edges. It either returns the
page's items, reports exhaustion, or expands the next step with a decremented
budget. No step runs inside a hidden nested execution or a Python loop.

For a node named `pages`, a step node named `fetch` has ids:

```text
pages/step/fetch
pages/next/step/fetch
pages/next/next/step/fetch
```

`UnfoldNext` is the portable continuation node. `UnfoldJoin` is a pure helper
that concatenates one page with the sequence produced by later pages. Tail joins
keep accumulated values on typed, checkpointed edges instead of storing mutable
state or embedding runtime items in parameters. They also include the final page
without a special placeholder. The graph grows only as pages become available;
its depth and number of step copies are bounded by `max_iterations`.

All helper types are public package entry points, and their params are portable.
Generated ids depend on the step workflow and iteration position. Reloading the
original graph and resuming with a context's saved node outputs replays completed
pages under the same ids and fetches only missing pages. Step completion follows
ordinary workflow data dependencies; detached effects are subject to the same
ordering rules as other workflow expansions.
