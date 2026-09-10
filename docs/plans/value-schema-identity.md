# Explicit Value identity and residual constraints

This implements the shared design for [#224](https://github.com/aceteam-ai/workflow-engine/issues/224#issuecomment-5613768057) and [#97](https://github.com/aceteam-ai/workflow-engine/issues/97). The identity, resolver, and compatibility changes form one contract: generated schemas must identify registered types explicitly before removing title lookup, and the resolver must retain that identity when schemas contain constraints or host metadata.

## Publication and identity

`Value.__get_pydantic_json_schema__` adds `x-value-type` to the handler's result without unfolding references or consulting the registry. This covers nested `$defs`, avoids freezing the lazy registry while node classes are still being defined, and retains recursive definitions for JSON, workflows, and value schemas. Float's specialized hook calls the base hook before publishing its JSON number shape. Result keeps its existing custom schema hook.

Only `x-value-type` (or its Python spelling `value_type`) selects a registered class. A title is presentation metadata, and even an inconsistent structural `type` cannot override explicit identity. Equal identity aliases fold into one field; conflicting aliases fail validation. An unknown identity continues to fall back to structural rebuilding; this change does not introduce strict registry-name validation.

## Resolve identity, then constraints

The resolver always attempts registry lookup. It compares the supplied schema's constraint metadata with a cached, normalized view of the registered class's own published schema. The comparison includes unknown extras and the declared string `pattern` and `enum` fields, following a root `$ref` when an enum/model wrapper puts those constraints in a definition. Titles, descriptions, defaults, structural fields, and identity are not residual constraints.

When the supplied constraints are intrinsic to the class, resolution returns that exact class. For example, a registered ticket ID publishing `x-resource-type: ticket` retains its custom cast table, and a percentage class declaring bounds 0–100 keeps its exact identity. Comparing intrinsic metadata avoids rebuilding a class merely because its own schema contains extras.

Residual constraints create an unregistered subclass of the identified class. Additional validation runs after inherited root validation without replacing the root annotation; replacing it would discard annotation metadata, custom validators, or intrinsic bounds. This keeps custom casts, serializers, normalization, and bounds intact. Extra metadata is published alongside inherited metadata, including callable `json_schema_extra` definitions.

Numeric bounds and multiples, string lengths/patterns/enums, and sequence/map sizes use the existing supported constraint vocabulary. Decimal validation handles numeric additions so fractional `multipleOf` is exact and does not attempt `Decimal % float`. Validators return the original validated root rather than the adapter's converted value.

Successive constraints are intersections. Each generated validator has its own name so a later subclass cannot shadow an earlier constraint validator. If an added keyword replaces a different inherited keyword in the emitted schema, the inherited restriction is retained in `allOf`. Rebuilding enforces these constraint-only conjunctions recursively; unsupported clauses fail with their keyword names instead of silently weakening validation. This is deliberately not a general JSON Schema composition implementation. Unknown top-level extras remain publication metadata under the existing contract.

## Recursive legacy schemas and compatibility

Unstamped recursive schemas previously relied on title lookup for early identity recovery. After removing title lookup, a repeated reference must raise a named `ValueError`, not recurse until Python overflows. A context-local stack tracks the referenced definition object and name during reconstruction, resetting on success or failure. Definition identity avoids false positives for shadowed names, and stack lifetime permits shared sibling references.

Regenerate stored schemas that relied on titles before upgrading, preserving their new explicit markers and recursive `$defs`. No legacy title compatibility switch is provided. No default-field or untitled-record behavior is part of this shared design.

## Verification

Regressions cover exact identity with intrinsic metadata/bounds, nested resource casts, stronger and weaker added bounds, patterns/enums, inherited normalization, Decimal serialization, constrained registered containers, alias conflicts, recursive failure/restamping, and two record levels of nested Value definitions. Chained patterns and multiples are checked before and after serializing/rebuilding the generated subclass. Callable schema metadata is covered in both supported callback forms.

The metadata audit compares all 46 public built-in `TYPE_INFO` declarations against the pre-change main branch. Only 35 added `x-value-type` keys differ. Mutation checks independently remove the base stamp, Float delegation, unconditional registry lookup, alias folding, and cycle guard, and restore title fallback; each mutation fails its targeted regression. Full pytest, Ruff lint/format, and Pyright are required before integration.
