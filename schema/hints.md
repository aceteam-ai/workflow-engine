# The hints channel

This is the published contract for the hints annotation channel (#203),
described in [discussion #198](https://github.com/aceteam-ai/workflow-engine/discussions/198).
It covers only the hints channel; it is not the full interchange schema
(#206), which is a separate, later contract.

## What a hint is

Discussion #198 draws three categories of information that can appear
alongside a workflow graph:

| Category | Test | Home |
| --- | --- | --- |
| Semantics | changes the output value or its type | the engine, always |
| Environment reference | resolves only inside a host (a stored id, a specific machine) | in the graph, must be resolvable or erasable at export |
| Hint | a host may honor, clamp, or ignore it without changing the result | this channel |

A hint is host-facing information that is safe to not understand. **A host
that ignores every hint on every node still computes the exact same result.**
That property holds by construction in this engine: nothing under
`execution/` or `nodes/` reads `Node.hints`. A hint never changes a node's
input or output type, so it can never change what a workflow computes,
only how a host chooses to run it.

The motivating case is a concurrency bound. A fan-out authored to protect a
single-GPU machine should travel with that intent so a recipient host can
clamp its own scheduling accordingly, rather than defaulting to unbounded
parallelism and losing information only the original author had.

## Wire shape

Every node carries a `hints` object:

```json
{
  "type": "ForEach",
  "id": "for_each",
  "params": { "workflow": { "...": "..." } },
  "hints": { "max_concurrency": 4 }
}
```

`hints` defaults to `{}` when omitted, which is equivalent to every hint
being absent.

### `max_concurrency`

| Field | Type | Description |
| --- | --- | --- |
| `max_concurrency` | `integer \| null`, `>= 1` | A suggested upper bound on how many of this node's parallel branches (for example, the items of a `ForEach`) a host should run at once. |

A host may clamp `max_concurrency` lower to protect its own resources, run
fewer branches at a time than the number given, or ignore it entirely and
run unbounded. The workflow's result does not depend on how many branches
were in flight at once. This engine's own `ParallelExecutionAlgorithm`
already leaves the interleaving of independent nodes unspecified, and
separately exposes its own run-wide `max_concurrency` knob; a per-node hint
only ever narrows that existing interleaving, it does not introduce a new
source of nondeterminism.

### Unknown keys

The `hints` object accepts and preserves keys it does not recognize. A
hint's entire point is that not understanding it is safe, so an engine that
predates a newer hint must still round-trip it rather than rejecting the
graph or silently dropping the key. A key that only resolves inside one
host is an environment reference, not a hint, and does not belong in this
channel regardless of whether it happens to be preserved.

## What deliberately does not live here: the node pin

Point 6 of the host-requirements discussion on the epic (#199) asks for two
annotations: a concurrency bound and a **node pin**, a reference used by a
host to route a fan-out's work to a specific machine. Only the concurrency
bound is a hint. The node pin is an environment reference, and the two are
not the same channel:

- A hint has host-independent meaning. Any host, including one that has
  never seen the annotation before, can safely keep `max_concurrency: 4`
  attached to a node it doesn't otherwise recognize and either honor it or
  ignore it. The annotation means the same thing everywhere.
- A node pin has meaning only inside the host that minted it. A machine
  identifier is not portable; a different host, or the same host after the
  machine is decommissioned, cannot interpret it and must not silently keep
  it as if it still pointed somewhere real.

That difference in lifecycle is the discriminator, not "who enforces it"
(both are host-enforced). It is why the table in #198 calls for an
environment reference to be resolvable or erasable at export: a node pin
must be inlined, re-pointed, or stripped when a graph leaves its
originating host, while a hint is safe to carry forward untouched forever.
Building that export-time resolution path is out of scope here; it belongs
with the interchange schema (#206).

## What is deliberately not here

This document covers the hints channel only. `attempt`, the eliminator
vocabulary, and the full interchange schema are separate, later contracts.
See #199 for the overall sequencing.
