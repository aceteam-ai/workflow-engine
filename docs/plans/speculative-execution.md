# Design proposal: isolated speculative previews

Status: design only; no speculative executor is implemented by this document.
Addresses [#87](https://github.com/aceteam-ai/workflow-engine/issues/87).

## Decision

Treat an editable output revision as input to an isolated preview execution.
Evaluate only eligible descendants, present results as tentative, and keep their
outputs and lifecycle events separate from the authoritative workflow run.
Confirming an edit uses normal execution against an immutable accepted snapshot.
Initial implementation must not promote preview results into the final cache.

This supports the motivating interaction: while a person edits text, pure
formatting, parsing, or numeric transformations can update a preview. Execution
stops at a node that may dispatch a billable job, write a file, send a message,
read changing external state, or otherwise has an unknown effect contract.

## Eligibility is executable metadata

Proposed additions to `NodeTypeInfo`:

```python
class EffectKind(StrEnum):
    UNKNOWN = "unknown"
    PURE = "pure"
    READ_ONLY = "read_only"
    EFFECTFUL = "effectful"

NodeTypeInfo.effects: EffectKind = EffectKind.UNKNOWN
NodeTypeInfo.deterministic: bool = False
```

`PURE` means evaluation changes no external state and depends only on immutable
inputs, params, implementation version, and captured execution configuration.
`deterministic` is a separate assertion that repeated evaluation under that same
snapshot produces the same values. Numeric precision/rounding configuration is
part of that snapshot. Clock access and unseeded randomness do not qualify.
Neither an idempotency key nor a compensation handler makes an effect pure.

Phase one requires an audited implementation with `effects=PURE`,
`deterministic=true`, and no metered work. Reuse the metered declaration from
[#205](https://github.com/aceteam-ai/workflow-engine/issues/205) when available;
its absence is not evidence that a preview is free. Until a complete declaration
exists, use an explicit audited allowlist of free built-ins. `READ_ONLY` remains
ineligible initially: database reads, mutable files, and remote lookups can cost
money or observe changing state even without a write.

These assertions cannot live in `Hints`. Ignoring a resource/concurrency hint
preserves the result; treating an unknown side-effect contract as permission to
duplicate execution could change the outside world. Estimates may prioritize
eligible preview work, but they never authorize it. Host machine and session ids
are execution references, not portable annotations.

## Proposed host-facing API

```python
class PreviewBudget:
    max_node_starts: int
    max_elapsed_ms: int

class PreviewRequest:
    workflow_digest: str
    source_node_id: str
    revision_id: str
    tentative_output: DataMapping
    budget: PreviewBudget

async def WorkflowEngine.preview(
    self,
    *,
    workflow: ValidatedWorkflow,
    request: PreviewRequest,
    context: PreviewExecutionContext,
) -> PreviewResult: ...
```

Budgets are required and finite; the host keeps previews disabled unless the
editor enables them. A revision id identifies one immutable snapshot. The host
checks that the requested source is editable in the active session and validates
the complete tentative output against that source's resolved output type.
Input/output shapes, including Result tags, remain the ordinary workflow shapes.

`PreviewResult` carries `revision_id`, the preview execution id, tentative node
outputs, nodes where evaluation stopped, and preview errors/limit status. It is
not a successful `WorkflowExecutionResult` for the authoritative run. The editor
shows which revision produced a result and never presents an old revision as
current merely because it finished last.

The context supplies already validated authoritative ancestor outputs from the
same workflow/input snapshot. The preview scheduler seeds its local output map
with those ancestors plus the edited source output. It evaluates the downstream
slice when all required inputs are present and stops at ineligible nodes. The
source's own implementation is not rerun just to inject a person's draft.
No tentative value is inserted into the authoritative context's output map.

## Revisions, cancellation, and cache isolation

Assign a distinct execution namespace to every preview, while retaining original
flat node ids inside its graph. Ledger and cache keys include that namespace and
revision, not node id alone. The current `on_node_start` cache pattern can reuse
outputs solely by node id within one immutable execution; sharing that cache
across edits would silently return stale results.

When a newer revision arrives, mark the older preview superseded before starting
replacement work. Cancel queued work and request cancellation of running tasks.
Every result/event passes a current-revision check before display or storage as
the active preview. Cancellation may not stop native or remote computation
immediately; keep its budget/reservation charged until it actually settles.
Use host-side debounce/coalescing and a bounded number of live revisions to
prevent a keystroke stream from creating unbounded computation.

`PreviewExecutionContext` provides separate preview callbacks and storage. It
must not forward preview completion to production `on_node_finish`,
`on_workflow_finish`, export, notification, or artifact-publication handlers.
Its write operations refuse external writes. Phase one permits only immutable
input data; a FileValue path by itself is not an immutable content snapshot.
Defensive context restrictions complement audited metadata; they do not sandbox
arbitrary Python node code that bypasses the context.

Known node outputs may be reused only when implementation version, params,
validated inputs, and captured configuration match. Phase one needs no shared
preview cache at all. A later content-addressed cache must include complete input
fingerprints, including immutable file content identities where supported.
Neither node id nor serialized file path is sufficient proof of equivalence.

## Expansion and failures

A container is eligible only when the actual work it can dispatch is eligible.
For built-in inline containers, inspect embedded workflows before starting and
check every newly expanded node again before dispatch. A trusted outer ForEach,
Fold, Unfold, conditional, or Attempt cannot confer purity on an unknown inner
node. Custom expanding nodes require a separately audited expansion contract.
Keep expansion flat so existing node provenance and checkpoint inspection remain
useful within the preview namespace.

A conditional follows the branch selected by the tentative input; this proposal
does not execute both branches automatically. Existing Attempt and Result
semantics remain intact within a preview. A preview error is visible as a preview
error or Result value, never as a final failure of the authoritative run.
ShouldYield and ShouldRetry retain their meaning; their records, retry budgets,
and any resumed preview context are isolated from canonical execution. A retry
must also fit the remaining preview dispatch/time budget.

The preview stops gracefully at an ineligible downstream node and reports that
frontier. This is not a workflow type error or an instruction to run the node by
another route. A newer preview revision cannot resume an older revision's yielded
external job under the same cache key.

## Accepting an edit

Accepting a revision produces an immutable authoritative snapshot and normal
execution. Start a fresh execution namespace for the affected computation;
reuse only validated unchanged ancestor outputs. Do not resume with a cache that
still contains descendant outputs from an earlier accepted or preview revision.
A host may offer the same user-facing run id while versioning its execution
attempts internally, but this versioning must be explicit in the ledger.

Do not promote preview outputs initially, even for pure nodes. This avoids
coupling a first preview implementation to cache promotion, artifact ownership,
and input fingerprinting. A future promotion feature would require complete
identity checks, immutable artifacts, compatible captured configuration, and
atomic publication; confirmation alone does not establish those conditions.

Metered previews are also deferred. They require a separate explicit spend
authorization, dispatch cap, and attributable cost ledger. A user's approval to
run the final workflow is not approval to multiply charges on each edit.

## Compatibility and implementation phases

1. Add optional conservative metadata and audit a small set of free deterministic
   built-ins. Existing types default to UNKNOWN/false and remain executable
   normally. Document author obligations and test their claimed purity.
2. Implement the isolated preview request/context/result API for static DAG
   slices, with required budgets and revision filtering. Keep all ordinary
   execution APIs and caches unchanged. Ship an editor integration behind an
   explicit host feature setting.
3. Add audited flat expansion, conditional branch handling, and preview-only
   yield/resume after the static path is proven. Apply eligibility to every
   generated node. Reuse resource estimates only for admission/prioritization.
4. Consider content-addressed preview caching, safe promotion, immutable file
   inputs, and explicit metered authorization as separate follow-ups.

Graphs remain portable: the graph's params, values, and hints need no new preview
flags. Revision ids, preview namespaces, authorizations, and UI debounce settings
belong to host execution/session state. Older hosts that lack previews continue
to execute the final workflow normally. Effect declarations are tied to a node
implementation version; updating a plugin requires reevaluating its eligibility.

## Required validation

- Unknown, effectful, read-only, nondeterministic, and metered implementations
  never dispatch in phase-one previews; a pure outer wrapper cannot bypass this.
- Rapid revisions finishing out of order display only the current revision.
  Cancelled/superseded results cannot overwrite current or canonical outputs.
- Preview contexts never invoke production publication hooks or write outputs
  into canonical cache/storage, including on errors and boundary materialization.
- Reused ancestor outputs must match the complete accepted snapshot. Reusing a
  node id with changed input must not reuse its previous result.
- Static slices stop at ineligible nodes, validate tentative types, preserve
  Result tags, and report missing-input and budget frontiers clearly.
- Expansion tests introduce an effectful child only at runtime and verify it
  is rejected before execution. Test nested containers and conditional branches.
- Preview yield/resume and retries use separate namespaces/budgets; stale jobs
  cannot be rebound to a new revision.
- Confirming an edit follows ordinary authoritative execution with no preview
  promotion, stale descendant cache entries, or automatic metered duplication.
