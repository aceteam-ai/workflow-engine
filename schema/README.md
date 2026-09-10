# Portable graph interchange

This directory publishes the graph, value and annotation contract from the same
repository as the engine. The generated schemas use [JSON Schema Draft
2020-12](https://json-schema.org/draft/2020-12/json-schema-core) and contain their
own shared definitions. Validation needs no network access. Use files from the
same release or commit as the receiving engine; `main` describes development
head, and the schemas' `$id` URLs identify their repository locations.

| Document | What it validates |
| -------- | ----------------- |
| [workflow.schema.json](workflow.schema.json) | A serialized graph: input/output nodes, inner nodes, edges, typed I/O declarations, hints, and current built-in node parameters. |
| [value-type.schema.json](value-type.schema.json) | **Type metadata**, such as `{"type":"integer","x-value-type":"IntegerValue"}`. This is the schema of the engine's schema language. |
| [values.schema.json](values.schema.json) | **Serialized instances** of registered Value types. Select `#/$defs/IntegerValue`, `#/$defs/FloatValue`, etc. for a specific type; the document root is their union and includes arbitrary JSON. |
| [result.schema.json](result.schema.json) | A tagged Result **instance**, with an unconstrained ok payload and the fixed structured err payload. |
| [hints.schema.json](hints.schema.json) | Erasable annotations, including unknown hint keys. |

The Python source is authoritative for generation. The portable profile adds
explicit export requirements that Python's more permissive loader does not
impose: node versions must be resolved, metadata must describe a recognized
structural form or use explicit registry identity, and a malformed Result
metadata object cannot pass as an open map. Python-only coercions are not part
of the JSON wire format.

## Graph shape

A graph is one object with required `input_node`, `inner_nodes`, `output_node`
and `edges` fields. Nodes have `type`, `id`, a resolved semantic `version`,
`params` (required when the node has required parameters), and optional
`hints`. Edges identify a `source_id`, a
`source_key` (a nonempty string or nonempty array of path segments), a
`target_id`, and a nonempty `target_key` string.

The node namespace is open. Current built-in node names and versions select
their generated parameter contracts. A differently named extension node, or a
node at a different version, is checked against the common envelope; its
provider must supply its parameter contract and executable implementation.
Input/output parameter declarations remain typed at every level. Schema
validation is not evidence that an extension is installed or portable.

`params.workflow` on `ForEach`, `Attempt`, and other workflow-taking built-ins
contains an **inline graph**, with the same contract recursively. A stored
workflow ID is not an inline graph. Node metadata's `parameter_schema` is not
copied into every graph node: parameters hold values. In this publication,
parameter definitions such as `ForEachParams` reference the shared
`WorkflowValue`/`Workflow` definitions exactly once. The recursive
`ValueSchemaValue` family is also shared, not repeatedly expanded. The runtime's
existing `NodeTypeInfo.parameter_schema` API continues to publish its existing
self-contained form; these documents are a deliberate compact projection.

See [examples/foreach-workflow.json](examples/foreach-workflow.json) for a
runnable inline fan-out and a concurrency hint. Executing it with
`{"sequence":[1,2,3]}` returns the same sequence.

## Type descriptions and values

`x-value-type` identifies a registered Value type. `title` and `description`
are presentation metadata; a string schema titled `IntegerValue` is still a
string. A marker can stand alone as registry shorthand. Otherwise the structural
forms include primitive `type` values, `items` for sequences,
`additionalProperties` for maps, `properties`/`required` for records, `anyOf`
for unions, and the dedicated `ok`/`err` Result metadata. Generic parameters
are described by their nested schemas; a receiver must not parse Python class
names out of `x-value-type` to infer them.

`$ref` resolves against local `$defs`, which must travel with the document.
Definitions can also contain plain Pydantic models and the schema language
itself, so not every definition is a reconstructible Value. The receiving
engine checks reference resolution and type construction. Inlining every
reference is impossible for recursive Workflow and ValueSchema models.

A record property listed in `required` must be supplied. A non-required property
needs an explicit, valid `default` because a Data field always contains a Value.
A `null` type or a union containing it permits a null value; it does not by
itself make a field optional. Untitled records receive a generated name.

Value instances carry their type's JSON representation, without an outer
`{"root":...}` wrapper. For example, IntegerValue is an integer, FloatValue is
a JSON number, sequences are arrays, maps and Data records are objects, and file
values describe the file reference expected by a receiving context. A
FileValue schema can validate the shape of that reference but cannot ensure
the file exists or is accessible. ModelValue requires an agreed model contract;
a Python model's implementation is not transferred by naming it.

### Result has two distinct schemas

A Result value is exactly one of:

```json
{"tag":"ok","ok":7}
```

```json
{"tag":"err","err":{"error_class":"timeout","name":"FetchTimeout","message":"Try later.","node_id":"fetch"}}
```

Both arms, a missing payload, and an unknown tag are invalid. `ok: null` is valid
when the declared payload type permits null. Nested Results retain every tag.
The generic `result.schema.json` validates the outer tag and the fixed error
shape; validation against the declared `T` is additionally required. An outer
ok payload is unrestricted in that generic schema, so it cannot detect a
malformed nested Result without knowing `T`.

`Result[T].to_value_schema()` describes the *type* using `ok` and `err` schema
members. Those names are engine metadata, not standard JSON Schema assertions.
Passing that document directly to a generic JSON Schema validator does **not**
validate Result instances. The instance publication translates it into
`oneOf` branches with `tag`, required payloads and no extra keys. The metadata
publication describes the original engine protocol unchanged. The example
[result-workflow.json](examples/result-workflow.json) accepts and returns a
`Result[IntegerValue]`, including its err arm. [result.md](result.md) specifies
the full error vocabulary and type metadata.

## Annotations and export erasability

A host may honor, clamp or ignore every hint without changing the result value
or its type. `hints.max_concurrency` is a positive integer when present. Unknown
keys survive copying and are equally safe to ignore. Execution-affecting
parameters belong in `params`, never in hints. See [hints.md](hints.md).

Host-specific execution nodes must be **erasable at export into pure public
nodes**. A host exporting a stored-workflow fan-out must resolve its stored
version, inline that graph, and emit the public `ForEach` form. Inlining freezes
the workflow version at copy time. Resolve `latest` to an exact node version as
well. Preserve a source version ID only as inert provenance, for example in a
namespaced node extension field; execution must not depend on resolving it.

A machine or storage reference that affects execution is an environment
reference, not a hint. The exporter must resolve it, replace it with a portable
reference the recipient can resolve, or erase it while preserving behavior.
A pure scheduling preference can become an erasable hint. Reject an export
whose remaining dependencies cannot be resolved by its recipient; silently
replacing unavailable outputs or dropping a node is not erasure.

These are obligations on a host's exporter. This publication does not claim to
convert arbitrary unknown host nodes automatically. Extensions with public,
agreed semantics can remain when the recipient has their implementation and
version; host-private semantics must first be translated into public ones.

## Validate and execute

For structural validation with the Python development dependencies:

```python
import json
from pathlib import Path
from jsonschema import Draft202012Validator

contract = json.loads(Path("schema/workflow.schema.json").read_text())
graph = json.loads(Path("schema/examples/foreach-workflow.json").read_text())
Draft202012Validator(contract).validate(graph)
```

A receiver must then load the graph through its node/value registries and run
engine validation before execution. That checks installed implementations and
versions, node parameters, DAG structure, node IDs, edge endpoints, and port types. Input values are validated during
execution. Missing required input edges are currently detected during execution
rather than rejected by graph validation (see [#95](https://github.com/aceteam-ai/workflow-engine/issues/95)).
Resource availability, credentials and host environment references also remain
the recipient's responsibility. Schema and graph validation therefore do not
prove that every node can execute successfully.

## Regeneration and drift checks

```bash
uv run python scripts/generate_interchange_schema.py
uv run python scripts/generate_interchange_schema.py --check
```

The generator discovers public node classes from checked-out source exports,
so newly added nodes do not depend on an already reinstalled entry-point
catalogue. Shared definitions are generated together and reduced to those each
document actually references. [Pydantic's schema generation
API](https://docs.pydantic.dev/latest/concepts/json_schema/) supplies
the model contracts; the script records the export projections explicitly.
CI compares the generated JSON byte for byte. Tests validate the schemas with
a standard Draft 2020-12 validator, check local references, reject malformed
examples, and load and execute the committed workflows under both executors.
