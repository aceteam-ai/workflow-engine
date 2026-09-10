# Resource validation without execution (#98)

Status: proposed design for [#98](https://github.com/aceteam-ai/workflow-engine/issues/98), ready for review before implementation dispatch. The resource names and provider examples below are illustrative; the engine does not own an agent database. This document changes no runtime behavior.

## Decision and scope

Add an optional asynchronous resource preflight that reads parameter declarations from the selected node registry and asks a host-owned resolver about concrete resource IDs. It accepts an incomplete stored graph, so a missing or null `agent_id` can produce a field diagnostic without constructing the node or running its dynamic type methods. A caller can use this alongside structural graph validation; neither check implies the other has passed.

The engine owns declaration traversal, paths, batching, and the report format. The host owns resource kinds, authentication, authorization, existence checks, and service access. There is no node-name allowlist and no engine dependency on any host's models. Calling the existing `validate()` or `execute()` does not implicitly perform resource preflight.

## Existing code and dependencies

At the audited base (`e2a3ef5`), [`Workflow.validate`](../../src/workflow_engine/core/workflow.py) calls `NodeRegistry.load`, infers node I/O types, and validates edges. [`NodeRegistry.load`](../../src/workflow_engine/core/node.py) performs migration and strict concrete-node construction together. Those operations are unsuitable as prerequisites for inspecting an incomplete parameter set. [`ValidationContext`](../../src/workflow_engine/core/context.py) already carries isolated node/value registries and is the appropriate capability boundary; file access and execution lifecycle hooks belong to `ExecutionContext` and are unnecessary here.

[#97](https://github.com/aceteam-ai/workflow-engine/issues/97) and [#224](https://github.com/aceteam-ai/workflow-engine/issues/224) preserve custom Value identity and `x-resource-type` through schema resolution. This proposal consumes the registered node's published `TYPE_INFO.parameter_schema`; it does not infer resource meaning from a title or attempt to infer it from a structurally rebuilt string class. Required input-edge validation (#95) remains a separate structural concern.

## Public API

Proposed entry points, exported from `workflow_engine`:

```python
async def validate_resources(
    workflow: Workflow | Mapping[str, Any],
    context: ValidationContext,
    *,
    options: ResourceValidationOptions | None = None,
) -> ResourceValidationReport: ...

# Convenience wrapper using the engine's normal validation-context factory.
class WorkflowEngine:
    async def validate_resources(
        self,
        workflow: Workflow | Mapping[str, Any],
        *,
        context: ValidationContext | None = None,
        options: ResourceValidationOptions | None = None,
    ) -> ResourceValidationReport: ...
```

`ValidationContext.__init__` gains optional `resource_resolver: ResourceResolver | None = None`. Existing subclasses and constructor calls remain valid. An explicit context controls its own registries, as with structural validation; the engine must not silently mix its registry with a caller's context. Execution callers pass `execution_context.validation_context` when they want the same host capabilities.

The resolver is a protocol with one method:

```python
class ResourceResolver(Protocol):
    async def check_resources(
        self, requests: Sequence[ResourceRequest]
    ) -> Mapping[str, ResourceCheck]: ...
```

Each immutable request has `request_id`, `resource_type`, and `resource_id`. IDs are nonempty strings in v1 and retain their exact spelling; the engine does not trim, cast to integers, or resolve URLs. Requests never carry the whole workflow, unrelated params, node inputs, or credentials. The provider authenticates through its own configuration.

`ResourceCheck.status` is one of `available`, `missing`, `forbidden`, `unavailable`, or `unsupported`. Optional `message` must be safe for the caller. The provider may report `missing` for inaccessible resources when its disclosure policy requires that. Exceptions other than cancellation become an `unavailable` check for the affected batch with a generic message; raw exception text is not copied into reports. Missing/extra request IDs in a response are provider protocol errors, surfaced as incomplete validation rather than assumed success.

## Parameter declaration contract

A resource ID schema declares `x-resource-type: "agent"`, on a registered Value type or a parameter field. A referencing property's local metadata overrides the target definition's annotation. The marker must be a nonempty string. For an array/map of IDs, annotate its item/value schema; a marker on an object or collection itself is an unsupported declaration in v1.

The effective parent schema's `required` list determines whether a resource field is required. An optional Boolean `x-resource-required` overrides that choice when resource requirements differ from serialization requiredness. This extension is consumed only by resource preflight. Its default is deliberately explicit:

| Parameter state | Result |
| --- | --- |
| Required resource absent or null | `missing_resource`; no provider call |
| Optional resource absent or null | Omitted; no issue and no provider call |
| Absent with an explicit JSON schema default | Inspect that literal default using the same rules |
| Present empty string, whitespace-only string, or non-string ID | `invalid_resource_id`; no provider call |
| Validly shaped, present ID | Ask the resolver |
| Required parent object/collection absent with resource descendants | `missing_resource` at the parent path |
| Optional parent absent/null | Omit the subtree |

Preflight never invokes default factories, `Value.cast_to`, dynamic I/O methods, `Node.run`, context lifecycle hooks, or resource picker code. Factories needed to materialize a resource reference are reported as unsupported coverage when detectable from the declared parameter type; hosts can instead supply an explicit saved value. Ordinary non-resource parameter errors are outside this report, except where an invalid container prevents resource traversal.

Traverse records, concrete array items, map values, local `$defs`/`$ref`, and the resource-bearing branch of nullable schemas. References are resolved against their lexical definition scope; do not strip `$defs`. A discriminator can select a tagged branch from the supplied data. When remaining union branches disagree about resource meaning, return `ambiguous_resource_schema`; do not probe every branch or pick one arbitrarily. Constraint-only `allOf` metadata is combined when annotations agree; conflicting declarations are reported. Remote `$ref` fetching and general JSON Schema evaluation are out of scope.

Traversal follows actual finite parameter data. Guard schema-reference cycles on the current traversal stack while allowing recursive records that consume another data segment; shared sibling definitions are valid. Unknown extension keywords remain metadata. Unknown or unsupported resource-bearing schema constructs make coverage incomplete instead of disappearing from the result.

## Incomplete graphs, versions, and nested workflows

The raw entry point reads only the document's node envelopes (`input_node`, `inner_nodes`, `output_node`). It checks the shape of that envelope, node IDs/types, and parameter mappings independently, and continues past invalid nodes to inspect others. It does not instantiate a `Workflow`, so an unrelated missing edge or cycle cannot hide a missing resource. Unknown node types, duplicate IDs, and invalid envelopes have explicit diagnostics and make coverage incomplete; `node_path` disambiguates even duplicate/missing IDs.

V1 handles current versions and `"latest"` (resolved through the supplied registry). An older/newer or malformed version returns `unsupported_node_version` for that node. It does not apply today's private migration helper to partial data: a migration can require missing fields and can change parameter paths, making diagnostics point at a different document. Hosts can use the existing migration/save flow and then preflight the resulting current document. Factoring a supported partial migration API is a separate follow-up, not a prerequisite for solving the current editor case.

Inspect inline `WorkflowValue` params recursively using their declared identity, with the same raw-node procedure. Report their containing node/parameter path rather than fabricating runtime namespace IDs. Each distinct inline use has its own location, even when the stored definitions are equal. A runtime-only subgraph cannot be expanded without execution; report that limitation in coverage when an expander declares unresolved workflow content. V1 does not promise discovery of resources hidden in arbitrary node code. Declaration completeness remains the node author's contract.

## Report and determinism

Frozen report models have this proposed shape:

```json
{
  "issues": [{
    "node_id": "send",
    "node_path": ["inner_nodes", 0],
    "param_path": ["agent_id"],
    "resource_type": "agent",
    "code": "missing_resource",
    "message": "Choose an agent."
  }],
  "checked_references": 0,
  "complete": true
}
```

Paths are arrays of string keys and nonnegative integer indices, never strings that require escaping dots or slashes. `node_path` is relative to the submitted document; `param_path` is relative to that node's params. Envelope-level issues can have `node_id: null` and an empty `param_path`. Issue `code` is a closed, engine-owned vocabulary; message text is display-only. Resource IDs are not repeated in user-facing issues.

`complete` means every declared reference in the supported document scope received a definitive check or a local missing/invalid-ID diagnostic. Missing/forbidden IDs make a complete report unsuccessful. An unavailable/missing provider, unknown declaration, unsupported version, ambiguous branch, invalid envelope, or traversal/batch budget exhaustion makes it incomplete. The derived `valid` property is `complete and not issues`; an empty but incomplete report must never enable execution accidentally. A fully traversed graph with no resource declarations is complete and valid without a provider. Execution can still fail because resources or authorization change after this snapshot.

Deduplicate requests by `(resource_type, resource_id)` only within one invocation and one resolver, then fan each result back to all original paths. Preserve all usage diagnostics and sort by document traversal order plus parameter path, independent of request completion order. No global cache crosses contexts, tenants, or validation calls.

`ResourceValidationOptions` proposes bounded defaults: batch size 100, at most 4 batches in flight, 10 seconds per batch, 10,000 distinct references, and nesting depth 64. Exceeding a budget produces `validation_limit` and `complete=false`; already collected diagnostics remain available. Cancellation cancels pending checks and propagates normally. Hosts can set smaller/larger limits explicitly; callers receive no silent truncation.

## Rollout and acceptance plan

1. Add immutable request/check/report models, resolver protocol, optional context capability, and declaration traversal with a fake provider. Publish the exact marker, status, and completeness semantics. Keep execution APIs unchanged.
2. Add both entry points, raw envelope traversal, inline-workflow traversal, request deduplication, and bounded batching. Export the report schema with the portable schemas from #206 when available. Resolve current/latest versions via each supplied registry.
3. A host adds a real resolver and resource Value declarations, then compares preflight reports with its existing editor checks. Replace that host's hardcoded node-name checks only after its missing/null/forbidden cases match. The engine release alone does not remove host workarounds.

Required tests:

- A raw incomplete node with a required null ID reports `missing_resource`; unrelated strict-param and dynamic-I/O failures cannot prevent the report. Spy hooks prove no node execution, casts, factories, or file operations occur.
- Field/type markers through `$defs`, #97 custom metadata, required/optional/default/null behavior, nested records, collection elements, map keys containing punctuation, inline workflows, recursive/shared definitions, tagged unions, and ambiguous unsupported unions.
- Current/latest version success, unsupported old/future versions, missing/duplicate node IDs, unknown node types, isolated registries, and useful diagnostics for valid siblings of a malformed node.
- Provider available/missing/forbidden/unavailable/unsupported responses, malformed response IDs, absent provider, deduplication with multiple locations, deadlines, cancellation, and all budget exits. Serial and concurrent checks produce identical ordered reports.
- No resource declarations passes without a provider; incomplete coverage never computes `valid=true`; resource IDs and backend exceptions do not leak through diagnostic serialization.
- Existing graph validation/execution tests and context-subclass typing fixtures remain unchanged; Ruff, format, Pyright, and full pytest pass at implementation time.

The review decision is whether this raw-document and coverage contract is the right initial boundary. Once accepted, implementation can proceed in the three slices above without choosing host-specific resource models.
