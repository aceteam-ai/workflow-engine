# Composable output provenance (#147)

Status: proposed design for [#147](https://github.com/aceteam-ai/workflow-engine/issues/147), written for review before implementation dispatch. The issue has already chosen the non-breaking union return contract. This document settles the remaining ownership, composition, caching, and compatibility boundaries; it changes no runtime code.

## Decision

Let a node return its existing output `Data`, an `AnnotatedOutput[Data]`, or its existing expansion `Workflow`. The built-in executors record provenance for each materialized output, synthesize conservative declared-input/parameter dependencies when authors supply none, and retain the graph bindings needed to trace final outputs. `Value` and `Data` wire payloads stay unchanged; provenance is an execution-result sidecar.

The guarantee is total coverage of emitted output fields relative to declared dependencies, not discovery of every external influence. The engine cannot infer an HTTP request, model call, environment read, clock, random seed, custom cast side effect, or host cache's historical inputs. Author annotations can identify external sources. Records distinguish inferred coverage, author claims, and unavailable history so an empty source list is not mistaken for verified independence.

## Code audit and integration boundary

The execution wrapper is [`Node.__call__`](../../src/workflow_engine/core/node.py), not `Node.execute`. It casts inputs, consults `on_node_start`, calls `run`, and invokes `on_node_finish` or `on_node_expand`. `on_node_error` can also replace a failure with output. Both [topological](../../src/workflow_engine/execution/topological.py) and [parallel](../../src/workflow_engine/execution/parallel.py) algorithms store output mappings directly. A boundary can materialize an error output outside this normal wrapper.

[`ValidatedWorkflow.expand_node`](../../src/workflow_engine/core/workflow.py) removes the expanding node and rewires namespaced subgraph I/O. The final original workflow alone is therefore insufficient to reconstruct what executed. [`WorkflowExecutionResult`](../../src/workflow_engine/core/execution.py) currently exposes output/errors/yields; hosts may construct it directly and return it from workflow hooks. Existing context methods can replace both individual outputs and the final result. All these paths must participate before provenance is advertised as total.

No initialization or teardown depends on a context subclass calling `super()`. Each algorithm creates an invocation-local collector, and the shared execution wrapper returns facts to that collector. This avoids context-global mutable state and leakage between concurrent invocations.

## Public models and return API

Proposed immutable models live in `core/provenance.py` and are exported from `workflow_engine`:

```python
class AnnotatedOutput[O](ImmutableBaseModel):
    output: O
    annotations: tuple[NodeOutputAnnotation, ...] = ()
    binding: ProvenanceBinding | None = None  # Optional cache attestation.

# Existing implementations returning a narrower Output remain valid overrides.
# The wrapper is for completed data; Workflow expansion remains a separate arm.
run_return_type = Output | AnnotatedOutput[Output] | Workflow
```

`ProvenanceBinding` identifies the node type/version and canonical input/parameter fingerprints for a cache attestation; live `run()` returns do not need to supply it. It may also reference an original invocation retained in a compatible sidecar. `NodeOutputAnnotation` has `output_path`, `sources`, and `coverage` (`complete` or `partial`, default `partial`). `complete` asserts that its sources cover the entire named output region; `partial` adds citations without excluding other dependencies. The engine cannot prove an author's completeness claim. `Source` has a discriminated `root`, `path`, `verbatim=False`, and optional `confidence` in [0, 1]. Confidence is author-supplied evidence, not a calibrated engine probability.

Core source roots are `InputRoot(input)`, `ParamRoot(param)`, `UrlRoot(url)`, `ModelRoot(provider, model, revision?)`, `ApiRoot(service, operation, resource?)`, `DbRoot(connection, relation, key?)`, and `FileRoot(locator)`. Locators identify sources without embedding credentials, file contents, query text, or full database rows. The engine records identifiers; it never dereferences an external root. A namespaced `ExternalRoot(kind, attributes)` envelope allows additional host roots without editing a closed engine union.

Paths describe the JSON representation of values, never their Python `root` wrappers. A component is a string field/key, a nonnegative integer index (excluding Boolean values), or a discriminated object. V1 supports `SpanComponent(type="span", start, end, unit="unicode_codepoints")`, with half-open intervals and `0 <= start <= end`. A terminal span applies to a string. `ExtensionComponent(type="extension", name="acme.bbox", data={...})` preserves domain-specific components through older readers. An optional resolver adapter can interpret that namespaced component; without one, the component survives but precision is unknown. No import-time mutation of a Pydantic union is required.

The head of `output_path` must name an actual output field. The engine validates supported paths and bounds against the finalized output, and input/parameter paths against the invocation's values. External source paths receive structural validation only, because validation must not fetch them. Unknown local fields, negative indices, out-of-bounds spans, invalid confidence, or a malformed core component fail as a node contract error. Unknown extension semantics are retained as opaque, not rejected as a malformed core path. Core paths into `Result` use its actual `ok`/`err` shape.

## Defaulting and honest coverage

For every field actually emitted by an invocation, the default annotation is:

`[output_field] <- every declared input field + every declared parameter field`

Each source has an empty relative path, `verbatim=false`, and no confidence. Coverage is `inferred_declared_dependencies`, and external-source completeness is `unknown`. Declared fields are used even if the author probably read only a subset. A node with no declared input or parameter has an empty inferred dependency set; this does not assert the absence of external reads.

An author-complete annotation replaces the fallback only within its specified region. Author-partial annotations are additive. A complete annotation for one span cannot erase the fallback for the rest of that string; an annotation for one element cannot claim the entire array. V1 keeps a whole-field fallback plus specific regions and resolves coverage at query time. If complete region coverage cannot be established, include the coarse fallback. Overlapping precise annotations are unioned in deterministic order. An explicitly empty, complete annotation is an author's assertion of independence and remains labeled as such.

The collector should intern each invocation's default source set, referring to it from output fields instead of copying every input/param pair for every field. It stores dependency edges, not fully expanded transitive closures.

## Lifecycle and public compatibility

Introduce a shared internal `Node._execute_with_provenance(...) -> NodeExecutionOutcome`. An outcome contains the final data mapping or validated expansion plus annotations, actual input-binding/cast facts, and origin (`run`, `cache`, or `recovery`). Both built-in schedulers consume it. Public `Node.__call__` keeps its current `DataMapping | ValidatedWorkflow` contract by delegating to the same helper and returning the payload. Direct callers can use the engine's existing `execute_node()` to receive the provenance sidecar.

`Node.run` is widened to the union above. Existing hooks keep their argument lists. Their output-returning types may additionally accept `AnnotatedOutput[DataMapping]`, allowing a host to return precise cached/replacement annotations without changing any existing override. No annotation state is initialized inside hooks.

Finalization order is: obtain run/cache output; unwrap and validate annotations; call the existing finish hook with the bare mapping; then finalize against the mapping actually returned. If a bare hook result changes any output field, discard that field's precision and synthesize a fallback with `origin="hook_replacement"` and unknown transformation history. Unchanged fields retain precision after supported path validation. A hook returning its own wrapper can supply replacement annotations. A recovery mapping from `on_node_error` uses the same finalizer even though it bypasses `on_node_finish` today. Do not add an extra finish-hook call merely to record provenance.

Retain a before-hook snapshot/digest of the output's serialized value so an in-place mutation cannot accidentally retain stale annotations. This is internal comparison evidence, not stored output content in the provenance document. Built-in boundary materialization must use the same finalization machinery after `on_boundary_error`; its error region also cites a synthetic failure source identifying the originating failed invocation. Failures, yields, cancellations, and abandoned retries have no invented successful output annotations.

`WorkflowExecutionResult` gains optional `provenance: ExecutionProvenance | None = None`, omitted when absent so existing result constructors and old serialized results remain readable. Constructors accept the new field optionally. The built-in algorithms attach their finalized sidecar after workflow finish/error/yield hooks. If a hook changes final output, use the same invalidation/default policy at workflow scope. A hook-supplied sidecar is accepted only after envelope and binding checks. Existing result fields must be preserved when updating the result.

No observer hook is required in v1; callers read or persist `result.provenance`. A later streaming observer can consume the collector without becoming its owner. Custom execution algorithms keep their existing interface; until adapted they may return `provenance=None`, which means unavailable, not a complete empty graph. The built-in executor guarantee must not be claimed for an unadapted custom algorithm.

## Sidecar, caches, and resume

`ExecutionProvenance` has `schema_version`, an invocation/pass ID, immutable invocation records, expansion/edge bindings, workflow-output bindings, and diagnostics. Records use `(pass_id, node_id, dispatch_ordinal)` identifiers, not just node ID. They retain node type/version, local source references, annotation origin/coverage, and opaque binding fingerprints for compatibility checks. Repeated retries or expanded nodes cannot overwrite an earlier attempt merely because the flat ID is reused.

A node cache can return `AnnotatedOutput` with local annotations for the same node contract. The engine revalidates paths against current inputs/params/output. Precise historical annotations require a matching node-version/input/param fingerprint supplied with the cache envelope; a mismatched or absent fingerprint degrades to the conservative current declared-dependency fallback with history marked unverified. The engine cannot establish cache-key correctness from output equality alone.

A whole-workflow cache may return an existing `WorkflowExecutionResult` with a compatible sidecar and workflow/input binding fingerprint. The engine can retain its original invocation IDs and mark reuse. A legacy cached result without history gets only coarse final-output dependencies on workflow inputs and declared node params, plus an unavailable-history diagnostic; it cannot fabricate completed internal invocations. A yielded result includes records for completed outputs only. Resume links reused records through an explicit compatible cache envelope; it never merges different passes by node ID alone.

The first implementation uses versioned sidecars and no durable global provenance store. Hosts decide retention and access control. Raw values, prompts, secrets, and exception messages are not copied into lineage metadata. Provenance locators and parameter names can themselves be sensitive; publication uses the host's normal execution-result access policy.

## Edge-aware composition

Expose a pure, offline resolver on the sidecar:

```python
trace: LineageTrace = provenance.trace(node_id="render", output_path=("text",))
final_trace: LineageTrace = provenance.trace_output(("answer",))
```

The default selects the materialized invocation for that output in this result; callers can select an explicit invocation ID for a retained historical record. `LineageTrace` returns terminal source roots, the derivation edges used, and any completeness/precision diagnostics. It does not contact external sources or need a live execution context.

For an `InputRoot("x")`, consult the invocation's captured incoming edge. Prepend its complete `source_key_path` to the source-relative path and continue at the upstream invocation. Do not split or join path strings: existing deep edges can select nested data/map fields. Workflow input bindings terminate in a `WorkflowInputRoot`. A genuinely materialized unwired input default terminates in an engine-recorded `DefaultRoot`; an absent binding with no recorded default becomes an explicit unavailable source. Params terminate at the owning invocation's `ParamRoot` unless they were created during expansion, as described below.

Path projection depends on the transformation. A coarse whole-field dependency only says that the whole input may influence the output; an output suffix must not be copied blindly onto every input source. Exact subregion translation is allowed only across author-complete, verbatim, compatible span mappings and actual identity bindings. Otherwise widen to the cited source region and mark lost precision. Never infer verbatim copying from a type cast or equal-looking values. Preserve per-hop confidence; do not multiply or average confidence into an unsupported end-to-end probability.

Input casts and output-boundary casts can change representation and can call host services. Capture each actual cast as a transformation in the binding record. V1 treats non-identity casts conservatively: retain upstream dependencies, clear verbatim/span precision across that step, and mark external influences unknown. A future caster annotation API can refine them. This prevents the graph from claiming exact text ancestry merely because an edge was type-compatible.

## Expansion and boundary composition

When a node returns a `Workflow`, capture an `ExpansionFrame` before `expand_node` removes it: original invocation, its input/param dependencies, returned graph's namespaced I/O, and all edge rewrites. Apply the same namespace mapping to annotation references. Inner output nodes and outer output bindings can then be followed without reconstructing the expansion from current code.

Graph edges alone are insufficient for data-dependent expansion. `ForEach`, for example, can turn an input element into a generated constant node's parameter. That constant has no ordinary incoming edge back to the original sequence. V1 therefore conservatively connects generated parameter roots and expansion outputs to the expanding invocation's declared inputs/params, while retaining any inner external citations. The outer field-level fallback is included until the expander provides an explicit complete mapping in a future refinement. This gives sound coarse ancestry rather than treating generated constants as independent external facts. The initial wrapper does not annotate returned workflows; precise expansion bindings are a separate API extension after this baseline works.

For an error boundary's err output, record the boundary's declared dependencies and a `FailureRoot` pointing at the originating failed invocation, including its recorded input/param dependencies. Do not pretend the cancelled sibling nodes produced data. For ok outputs follow the actual inner output bindings. Nested boundaries, retries, and resumed passes use captured invocation/frame IDs so containment and namespace prefixes do not collapse distinct histories.

## Determinism, budgets, and rollout

Serialize records and sources in stable graph/path order rather than task completion order; retain distinct derivation edges even when terminal roots deduplicate. Resolver traversal memoizes invocation/path states, reports malformed cycles in imported sidecars, and uses explicit record/path/depth budgets. Reaching a capture or query budget yields a visible `incomplete` diagnostic with a coarse dependency fallback where possible, never a silently complete partial trace. Proposed configurable defaults are 100,000 invocation records and 1,000,000 source references per capture, 100,000 visited derivation edges per query, path length 64, depth 256, and 64 KiB per extension payload. A representative fan-out fixture must verify bounded memory and the incomplete-report behavior before enabling capture by default.

1. Add frozen models, the output wrapper, path validation, versioned sidecar serialization, and the shared internal outcome/finalizer. Integrate both schedulers, bare outputs, cache/recovery/finish replacement paths, workflow-result attachment, and boundary outputs together. Existing nodes remain unchanged. Do not ship a feature flag labeled total before these paths are covered.
2. Add captured edge/expansion bindings and the offline resolver, including casts, generated params, final-output projection, partial results, and compatible cache reuse. Export the sidecar schema through #206's generator when available. This slice delivers the issue's end-to-end payoff; local annotation lists alone do not close #147.
3. Add an opt-in string templating/extraction test node to demonstrate spans and an external root, then migrate one real precision node in its owning package. Publish host guidance for cache envelopes and custom executor adoption. Namespaced extensions remain losslessly readable before any domain adapter is installed.

Acceptance requires both executors to produce equivalent normalized lineage for unchanged nodes, a diamond graph, deep source paths, a constant, unwired materialized defaults, and a fan-out expansion. Tests must cover author complete/partial regions and uncited output fields; invalid paths/spans; opaque extensions; coarse projection through casts; hook mutation/replacement; node and workflow cache hits with/without/mismatched history; error recovery; boundary ok/err, cancellation, retry, and yield/resume; concurrent context reuse without leakage; imported-cycle detection; deterministic serialization; and explicit budget exhaustion. Include typing fixtures proving existing `run`, hook, and custom algorithm subclasses still type-check. All existing output-value tests, Ruff lint/format, Pyright, and full pytest must pass at implementation time.

The dispatch gate is review of this sidecar/ownership contract and its proposed capture limits. Existing node implementations do not need to adopt the wrapper. Classification, PII propagation, arbitrary annotation categories, precise expansion APIs, distributed provenance storage, and inferred external-source discovery remain outside v1.
