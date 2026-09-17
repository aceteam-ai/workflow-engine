# Resource preflight

`validate_resources()` inspects declared resource parameters without executing a workflow, constructing concrete nodes, inferring dynamic I/O, evaluating default factories, or casting Values. It accepts a `Workflow` or an incomplete raw document, so editors can report a missing resource before the graph is executable.

Declare a resource kind on a Value type or a parameter field:

```python
from typing import ClassVar
from pydantic import ConfigDict
from workflow_engine import Value

class AgentId(Value[str]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra={"x-resource-type": "agent"}
    )
```

Use that type in a node's `Params`. Alternatively, set `json_schema_extra={"x-resource-type": "agent"}` on the parameter's `Field`. Resource IDs are strings; annotate item/value schemas for arrays and maps. The field's local declaration overrides a referenced type's declaration. Titles do not identify resource kinds.

Supply a resolver through the validation context. The host implements resource existence and access checks; the engine has no agent database or resource-kind allowlist. This small example uses an in-memory set:

```python
from collections.abc import Mapping, Sequence
from workflow_engine import (
    ResourceCheck, ResourceRequest, ValidationContext, validate_resources,
)

class KnownResources:
    def __init__(self, available: set[tuple[str, str]]):
        self.available = available

    async def check_resources(
        self, requests: Sequence[ResourceRequest]
    ) -> Mapping[str, ResourceCheck]:
        return {
            request.request_id: ResourceCheck(
                status="available"
                if (request.resource_type, request.resource_id) in self.available
                else "missing"
            )
            for request in requests
        }

context = ValidationContext(
    resource_resolver=KnownResources({("agent", "agent-123")}),
)
# raw_document is the editor's current workflow document.
report = await validate_resources(raw_document, context)
for issue in report.issues:
    print(issue.node_id, issue.param_path, issue.code)
```

Use `node_registry=` on the context for host-defined or isolated node sets. `engine.validate_resources(document)` uses the engine's normal validation-context factory. An explicit `context=` controls its own registry and resolver; it is not combined with a different engine registry. From execution code, pass `execution_context.validation_context`.

## Interpret the report

- `issues` contains immutable per-node diagnostics, with a node ID, document-relative `node_path`, parameter-relative `param_path`, resource kind, stable code, and display message. Paths retain string keys and integer indices, including keys containing dots or slashes.
- `checked_references` counts distinct IDs with definitive provider responses (`available`, `missing`, or `forbidden`). Local missing/invalid IDs and unavailable checks are not counted.
- `complete` means all declared references in the supported document scope were inspected. Missing or forbidden resources can still produce a complete report.
- `valid` is `complete and not issues`. It says nothing about graph structure, required input wiring, or whether a resource will still exist when execution starts.

Provider statuses are `available`, `missing`, `forbidden`, `unavailable`, and `unsupported`. Providers must return exactly one response for every request ID. Invalid response IDs or statuses produce `provider_protocol_error`; provider exceptions and timeouts produce `resource_unavailable`. These failures make coverage incomplete. Cancellation propagates and pending provider tasks are cancelled.

A provider's optional message must be safe for the caller. Backend exception text and resource ID values are not copied into engine-generated diagnostics. Hosts may return `missing` instead of `forbidden` when their access policy must conceal existence. A graph with no resource declarations needs no provider and produces a valid report.

## Requiredness and traversal

A schema's `required` list controls required resource fields. Boolean `x-resource-required` overrides it for preflight. Missing or null required IDs produce `missing_resource`; absent/null optional IDs are omitted. An explicit schema default is used only when the field is absent, never when it is explicitly null. Root union branches contribute resource-required overrides and defaults before omission is decided; a literal object default can select a tagged branch. Empty/whitespace-only/non-string IDs produce `invalid_resource_id`; other strings retain their original spelling. A required missing parent with resource descendants is reported at that parent path.

The engine traverses records, arrays, maps, local `$defs`/`$ref`, nullable resource schemas, and discriminated unions whose tag selects a branch. Agreeing resource branches and constraint-only `allOf` declarations are supported. Ambiguous branches, unsupported resource-bearing constructs, invalid containers, unavailable definitions, and cyclic reference chains make coverage incomplete. This is not a general JSON Schema validator. Optional absent subtrees do not require provider checks. Detectable resource default factories require an explicitly saved value and produce `unsupported_default` instead of running the factory.

Inline `WorkflowValue` parameters are inspected recursively, using document paths instead of invented runtime node IDs. `ValueSchemaValue` parameters describe data and are treated as metadata, not as resource instances. Runtime-generated subgraphs and resources hidden in arbitrary node code cannot be discovered without execution.

Raw node envelopes are inspected independently. Unknown types, duplicate/missing IDs, malformed envelopes, and unsupported node versions do not suppress diagnostics for other nodes. Current versions and `latest` are supported. Migrate/save older documents through the host's existing migration flow before preflight; partial migrations and dynamic expansion are deliberately not performed.

## Limits and schemas

`ResourceValidationOptions` configures batch size (100), maximum concurrent batches (4), per-batch timeout (10 seconds), distinct references (10,000), traversal depth (64), and total traversal visits (100,000, including raw mapping-key inspection before sorting). Reference/visit exhaustion stops further traversal while preserving collected issues and checking already collected requests. Depth exhaustion stops that subtree. All limits produce `validation_limit` and incomplete coverage. Schema inspection also bounds recursion and distinct schema nodes; shared definitions are visited once per inspection instead of unfolding their reference graph repeatedly.

Requests deduplicate by resource kind and ID within one call, then each outcome is reported at every usage path. There is no cross-call/context cache. Issues follow node document order and sorted parameter paths regardless of provider completion order.

`ResourceValidationReport.model_json_schema()` exports a standalone report schema. Integration into #206's portable schema generator will follow when that generator is available on main. Hosts still need to implement their resolver and compare results against existing editor checks before removing node-specific workarounds.

See the [design and acceptance plan](plans/resource-validation.md) for the full contract and rollout boundaries.
