# Contextual Type Resolution Redesign

**Status:** Planning  
**First phase:** [#66 — Asynchronous and contextual node types via build contexts](https://github.com/aceteam-ai/workflow-engine/issues/66)

This document tracks the redesign of node input/output type resolution to support generic nodes, contextual resolution, and partial workflows.

---

## Problem Statement

We need generic nodes whose output types are determined by:

- **Parameters** — e.g., field name, element count
- **Edges** — types flowing in from upstream nodes
- **Context** — build-time environment (auth, ValueRegistry, etc.)

Today, types are `@cached_property` on the node, derived only from params. This does not support:

- Output types that depend on input types (e.g., GetField)
- Resolution that requires Context or ValueRegistry (e.g., types from a database)
- Arbitrary transformations of input type parameters to output types

---

## Design Constraints

| Constraint | Implication |
|------------|-------------|
| **Partial workflows during editing** | Input types must be knowable before any edges exist; used to validate "can I connect this edge?" |
| **Contextual resolution** | Types may depend on Context and ValueRegistry; `@cached_property` on nodes is inappropriate |
| **Types passed at runtime** | Nodes use `output_type` in `run()` to construct outputs; types must be passed in at invocation |
| **Library for operators** | External packages define their own nodes; breaking API changes are acceptable (RC phase) |
| **Interactive editing performance** | User-facing type resolution runs on every node update; must support incremental / scoped resolution |
| **Type identity** | Resolved types must be stable; caching lives at caller/resolver level, not on the node |

---

## Architecture (Target State)

### 1. Build Context vs Execution Context vs Validation Context

- **Build context** — Read-only resources available when resolving workflow types (auth, ValueRegistry, workflow graph).
- **Validation context** — Build context + edges. Used for type resolution and edge validation. Nodes never see edges during execution.
- **Execution context** — Read/write resources during workflow execution (file I/O, lifecycle hooks). No edges, no type-resolution concerns.
- Edges are passed to the resolver separately; nodes are not given edges at execution time.
- See [#66](https://github.com/aceteam-ai/workflow-engine/issues/66).

### 2. Type Resolution is External and Contextual

- No `input_type` / `output_type` properties on the node.
- Types come from a **resolver** that takes a **resolution context** (build context + workflow + edges).
- Node provides async `compute_input_type(ctx)` and `compute_output_type(ctx)` that the resolver invokes.
- **Resolution runs before execution.** Must be async — it may depend on external lookups (e.g., database).

### 3. Types Passed at Execution

- Engine resolves types before invoking a node.
- Types passed into `__call__` and `run`:

```python
async def __call__(
    self,
    context: Context,
    input: DataMapping,
    *,
    input_type: type[Data],
    output_type: type[Data],
) -> DataMapping | Workflow:

async def run(
    self,
    context: Context,
    input: Input_contra,
    *,
    input_type: type[Input_contra],
    output_type: type[Output_co],
) -> Output_co | Workflow:
```

### 4. Unresolved Types

- Nodes with edge-dependent output may have **unresolved** fields when inputs are not connected.
- When a node expects arbitrary types for an edge-bound input field, its input type should include `UnresolvedType` for that field.
- `UnresolvedType` is a **privileged subclass of ValueType** representing "type not yet determinable."
- Use in partial data types: `build_data_type("Output", {"known": IntegerValue, "unknown": UnresolvedType})`.
- **Cast semantics** — both directions allow: `X.can_cast_to(UnresolvedType)` → True; `UnresolvedType.can_cast_to(X)` → True.

### 5. Node Examples (no special handling)

- **InputNode** — Strongly typed output; controls how a UI renders a form for workflow inputs.
- **OutputNode** — Declares what fields it expects; types come from edges (or Unresolved when unconnected).
- **ForEachNode** — Forwards a sequenced version of its inner workflow's input/output. The resolution context must support resolving nested workflows on the spot.

### 6. Type Validation on the Engine

- Type validation may live on the **engine** (rather than on Edge or Workflow).
- Pass around the **subset of engine fields** needed for validation (e.g., registries, resolver) — not the entire engine. This keeps validation/editing contexts lightweight and avoids coupling them to execution-specific pieces.

### 7. Resolution Performance

- **Incremental resolution** — When a node changes, invalidate that node and descendants; resolve only affected nodes.
- **Scoped API** — `resolver.resolve_node(node_id)` for "what are types for this node?" without full DAG propagation.
- **Two modes** — Quick (params-only, placeholders) for editing; full (with propagation) for validate/execute.

---

## Phases

### Phase 1: Contextualization of Type Inference (current)

**Issue:** [#66 — Asynchronous and contextual node types via build contexts](https://github.com/aceteam-ai/workflow-engine/issues/66)

- Introduce **build context** as a concept distinct from execution Context.
- Move display name, input type, output type into build context.
- Turn `input_type`, `output_type` into functions that take build context.
- Support ValueRegistry access for multi-tenancy (different users, different Value types).

### Phase 2: External Resolution + Pass Types at Runtime

- Introduce type **resolver** and **resolution context** (build context + workflow + edges).
- Add `compute_input_type(ctx)` and `compute_output_type(ctx)` to Node.
- Remove `@cached_property` input_type/output_type; engine resolves and passes types at invocation.
- Update `__call__` and `run` signatures to accept `input_type` and `output_type`.
- Migrate all built-in nodes and consumers. Types are used everywhere: edge validation, workflow schema, execution, context hooks, form rendering, etc.

### Phase 3: Unresolved Types + Partial Workflows

- Introduce `UnresolvedType` sentinel.
- Implement semantics for `can_cast_to`, `resolve_path` with UnresolvedType.
- Update `compute_output_type` in generic nodes to return partial types (mix of resolved and unresolved fields) when edges are missing.
- Document edge validation behavior: optimistic during editing, strict at validate/execute.

### Phase 4: Resolution Performance + Incremental invalidation

- Implement incremental resolution: invalidate on change, resolve only affected nodes.
- Expose `resolve_node(node_id)` API for scoped resolution.
- Optional: two-tier resolution (quick vs full) for interactive editing.

---

## Resolved Decisions

- **Context hooks** — Will need updated signatures to receive `output_type` (and `input_type` where relevant).
- **UnresolvedType** — A privileged subclass of ValueType, used in field positions for unresolved types.

---

## Open Questions

- [ ] **ValueRegistry** — Role in resolution is unclear. It is not used dynamically today. One example: workflow deserialization uses a value registry to define certain base value types for type resolution. Exact shape of how/when it is passed remains TBD.
- [ ] **Build context shape** — Fields, lifetime, how ValueRegistry (if any) is passed.

---

## Implementation Note

Method signatures and API details are deliberately left vague. Specifics will be driven by implementation-time nitpicks that are not predictable until work begins.

---

## Food for Thought: Ways Resolution Can Fail

When resolution fails, we need structured errors so downstream can show something meaningful to users. Brainstorm of failure modes:

**External / async**

| Failure | Cause | Example |
|---------|-------|---------|
| External lookup fails | DB/API unavailable, timeout, auth | Node fetches schema from DB |
| Not found | Resource missing | "Function X does not exist" |
| Timeout | Call too slow | UI needs fast resolution during editing |

**Node / compute**

| Failure | Cause | Example |
|---------|-------|---------|
| `compute_*` raises | Bug or wrong assumption | Node expects field upstream type doesn't have |
| Missing dependency | Node needs something ctx doesn't provide | Auth, ValueRegistry lookup |
| Invalid type construction | `build_data_type` or similar fails | Bad args, malformed schema |

**Graph / structure**

| Failure | Cause | Example |
|---------|-------|---------|
| Broken edge reference | Edge targets missing node | Node removed but edge still references it |
| Path doesn't exist | `resolve_path` on non-existent field | Edge references field type doesn't have |
| Invalid workflow state | Mid-edit, inconsistent structure | Duplicate IDs, orphan edges |

**Registry / schema**

| Failure | Cause | Example |
|---------|-------|---------|
| Type not registered | Custom Value or Node used but not registered | Schema references unknown type |
| Schema load fails | `ValueRegistry.load_value(schema)` fails | Malformed or incompatible schema |

**Nested / recursion**

| Failure | Cause | Example |
|---------|-------|---------|
| Inner workflow fails | ForEachNode's inner workflow | Failure propagates from nested resolution |
| Depth / recursion | Very deep nesting | Stack overflow or resource limit |

**Error shape** — Consider structured exceptions (e.g. `ResolutionError`) with `node_id`, `kind`, `message`, `cause`. Possibly categorize: transient (retry?), configuration, data, logic. Library structures the error; downstream maps to UI.

---

## References

- [Issue #66](https://github.com/aceteam-ai/workflow-engine/issues/66) — Asynchronous and contextual node types via build contexts
- [Architecture](../architecture.md) — Module structure, design decisions
