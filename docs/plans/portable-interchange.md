# Portable interchange (#206)

The design was recorded before implementation: distinguish engine type metadata
from schemas validating serialized instances, share recursive definitions, and
make erasability and runtime validation obligations explicit.

## Publication

Use generated Draft 2020-12 schemas for the workflow envelope, engine ValueSchema
metadata, registered Value instances, Result instances and hints. Result's
existing `ok`/`err` metadata is not a standard tagged-object validator; the
instance schema explicitly translates it into tagged `oneOf` arms. This leaves
the runtime protocol unchanged and prevents a consumer from mistaking metadata
validation for value validation.

Generate all model definitions together, then bundle each document's reachable
definitions locally. ForEach parameters refer to one shared Workflow definition
and one shared ValueSchema family. Retain the runtime parameter_schema API's
self-contained documents rather than forcing its callers to learn a new
external-reference protocol. The shared graph document also contains current
built-in node parameter contracts, selected by node name and version. Extension
names remain open, with receiver-side implementation/parameter checks mandatory.
Source exports supply the node catalogue, independent of installed entry points.

Require resolved versions in portable exports, map serializer-only aliases to
their wire names, and reject incomplete metadata roots that Python's permissive
catch-all can parse but cannot reconstruct. Definitions may include plain
Pydantic/schema-language models, not just Value types. All reference and type
resolution still belongs to the receiving engine.

## Portability

Structural validation establishes JSON shape. The receiving engine resolves
node types/versions, checks parameter contracts, graph structure and edges, and
verifies available resources. A schema cannot establish arbitrary host or type
semantics.

Host export must replace stored-workflow nodes with public equivalents, resolve
and inline workflow versions, preserve source IDs only as inert provenance, and
reject unresolved execution dependencies. Hints may be ignored without changing
the result, including unknown keys. No claim is made that the core engine can
export arbitrary unknown host nodes automatically.

## Verification

Commit canonical schemas and runnable examples. Check regeneration in CI, use an
independent JSON Schema validator for positive and negative wire fixtures, and
execute round-tripped examples under both scheduling algorithms. Keep JSON Schema
validation a development dependency rather than adding a runtime dependency.
