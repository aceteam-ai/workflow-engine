# Design proposal: node resource estimates

Status: design only; no API or scheduler changes are implemented by this document.
Addresses [#67](https://github.com/aceteam-ai/workflow-engine/issues/67).

## Decision

Publish advisory resource estimates separately from execution requirements and
host allocations. First make estimates inspectable; subsequently offer an opt-in
host scheduling policy that prioritizes long dependency paths and admits work
against resource reservations. Keep the existing default executors unchanged.

A forecast is an estimate, not an exact workflow duration. Even if node durations
were known, limited resources introduce scheduling choices; a longest dependency
path alone does not solve the resource-constrained scheduling problem. Dynamic
expansion, retries, yields, and differing worker hardware add further uncertainty.

## Three distinct contracts

| Information | Classification | Consequence |
| --- | --- | --- |
| Expected duration, CPU demand, peak RAM/VRAM | Advisory estimate | A host may ignore or replace it without changing successful output values |
| A kernel requires CUDA or another execution capability | Execution requirement | An incapable host must report inability to execute; it cannot silently substitute another implementation |
| A particular worker, machine id, or allocation lease | Environment reference | Resolved by the host; must be resolved or erased according to export rules |

This follows the existing [hints contract](../../schema/hints.md). A positive VRAM
estimate is not proof that a GPU is mandatory, and a zero or absent estimate is
not proof that execution needs no memory. Exact machine references do not belong
inside resource hints. Resource estimates describe the worker doing the actual
computation, which may differ from the process orchestrating a dispatch node.

## Proposed public shape

Add a frozen `ResourceEstimate` model with these optional fields. Every omitted
field means unknown; explicit zero is a real estimate of zero. Values must be
nonnegative integers, and absent fields are omitted when serialized.

| Field | Unit | Meaning |
| --- | --- | --- |
| `duration_ms` | Milliseconds | Expected wall time on the author's reference execution profile |
| `cpu_millicores` | Thousandths of one CPU core | Expected CPU demand while executing |
| `memory_bytes` | Bytes | Expected peak resident memory at the execution worker |
| `accelerator_memory_bytes` | Bytes | Expected peak device memory at the execution worker |

Proposed attachment points:

```python
NodeTypeInfo.resource_estimate: ResourceEstimate | None = None
Hints.resources: ResourceEstimate | None = None

def Node.estimate_resources(
    self, *, input: DataMapping | None
) -> ResourceEstimate | None:
    ...
```

The type-level estimate is the default for that implementation version. The
optional method refines it from params and already available input metadata. It
must not execute work, read files, contact a provider, or mutate the node. The
base implementation returns the type-level estimate. A per-node `hints.resources`
value expresses author intent for that particular graph invocation.

Resolve each known field in this order: host calibration/override, instance
hint, input-aware method, type-level default, unknown. Estimates do not accumulate
by adding these sources. The host records which source supplied each resolved
field in its run ledger; hardware profile ids and observations stay there.

Example portable annotation:

```json
{
  "hints": {
    "max_concurrency": 2,
    "resources": {
      "duration_ms": 18000,
      "memory_bytes": 2147483648,
      "accelerator_memory_bytes": 6442450944
    }
  }
}
```

The diagram, editor, and forecast API can consume this without enabling resource
scheduling. Stripping hints recursively must preserve all executable params and
the final values, including in nested workflows.

## Requirements are a separate later API

The proposed requirement surface is a frozen `ExecutionRequirements` with
`capabilities: tuple[str, ...]`, `accelerator_count: int`, and
`minimum_memory_bytes: int`, attached to `NodeTypeInfo`. Capabilities use a
documented vocabulary such as `accelerator.cuda`; unknown required capabilities
mean the host cannot confirm execution support. They are not ignored hints.
Capability negotiation must also distinguish compatible node implementations
and their pinned versions, rather than changing an implementation implicitly.

Do not ship mandatory GPU routing as a side effect of accepting advisory
estimates. The requirement vocabulary, version constraints, and worker resolver
need a concrete host consumer before implementation. The initial estimate API
can land independently, and existing node validation/execution remains intact.

## Scheduling integration

A later opt-in policy receives ready nodes, resolved estimates, requirements,
and a host-owned resource snapshot:

```python
class SchedulingPolicy(Protocol):
    def select(
        self,
        *,
        ready: Sequence[ReadyNode],
        available: AvailableResources,
    ) -> Sequence[str]: ...
```

`ReadyNode` contains the flat node id, resolved estimate, and requirements.
`AvailableResources` identifies host-managed workers/devices and remaining
capacity; it is execution state, not portable graph data. The host atomically
reserves resources for selected ids before dispatch. A policy cannot bypass
ordinary dependency readiness, current retry/backoff rules, or failure scopes.

Start with a deterministic list policy: rank ready nodes by estimated remaining
critical-path duration, use flat id as a tie-breaker, and admit fitting nodes.
Compute priorities over the graph currently known and recompute affected paths
after expansion. Missing duration uses an explicit host fallback, never an
implicit zero. Consider every terminal path, including detached work, because
workflow execution may still need to wait for it after the output is available.

Reservations must distinguish local tasks from externally dispatched jobs.
`ShouldYield` can mean a remote job is still running: releasing its worker lease
merely because the orchestration coroutine yielded would oversubscribe the GPU.
The host releases that lease on confirmed job completion/cancellation, not solely
on the executor's yield hook. Retries must not double-reserve a still-live job.
Cancellation and lease expiry require idempotent host bookkeeping.

Estimates may underpredict actual usage. They guide admission, while hard host
quotas remain authoritative. If one ready node exceeds available total capacity,
report an unsatisfied resource request rather than waiting forever. A host may
calibrate or clamp estimates; it may not relax a genuine execution requirement.

## Forecast and observations

Return a forecast with the current graph digest, estimated elapsed duration,
coverage (nodes with known durations), unknown/unexpanded regions, and the host
capacity assumptions. Report a critical-path lower bound and, when requested,
the makespan of the selected scheduling simulation. Do not label either an exact
completion time or an optimal schedule. Unfold-like unknown iteration counts
remain unknown until execution or an explicit host forecasting assumption.

Collect elapsed time and peak resource observations per implementation version,
worker profile, and execution attempt. Calibrate in host storage rather than
rewriting a user's graph. Do not place input payloads, machine ids, or provider
credentials in portable estimates. Cost authorization and retry spend budgets
remain separate from performance estimates.

## Compatibility and implementation phases

1. Add the frozen estimate model, optional type metadata, and nested resource
   hints. Defaults remain absent. Publish schema and inspector output only.
2. Add the pure input-aware method and a forecast utility over a validated graph.
   Host observations/calibration are separate adapters. No scheduling changes.
3. Implement the policy protocol and one opt-in resource admission adapter with
   lease lifecycle integration. The current topological and parallel behavior
   stays the default. Update the hints documentation when an opt-in executor can
   actually honor resource hints.
4. Introduce capability requirements only with a worker resolver and explicit
   unsupported-capability behavior. Evaluate packing and priority policies from
   observed workloads before adding more optimization strategies.

Optional metadata is additive; graph value types and node params do not change.
Older hosts retain unknown hints and may ignore them. Unknown mandatory
requirements cannot be treated as optional when that later protocol is added.
No migration should convert existing concurrency hints into resource limits.

## Required validation

- Schema round trips preserve unknown versus zero, units, defaults, and nested
  hints; negative estimates fail validation.
- Hinted and stripped graphs have identical output values under both default
  executors, including nested traversal and generation.
- Forecast fixtures cover serial chains, independent branches, unknown costs,
  detached terminal work, heterogeneous workers, and expansion invalidation.
- Admission never spends the same reservation twice; property tests maintain
  per-worker and per-device capacity accounting.
- Yielded remote jobs retain leases; completion, cancellation, retry, and resume
  release or reuse them exactly once. Oversized jobs produce a clear outcome.
- Injected estimation failures degrade to unknown and a diagnostic; they do not
  execute work or replace the node's semantic result.
