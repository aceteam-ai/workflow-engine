# Sequence algebra

Sequence operators preserve order and treat `Result[T]` elements as ordinary
values. They never skip failed elements, substitute placeholders, or accept an
error-handling policy. Resolve a `Result` using the explicit result eliminators
when a later operation needs its payload.

## Pure shape operations

| Node | Inputs | Output | Parameters |
| --- | --- | --- | --- |
| `FlattenSequence` | `sequence: Seq[Seq[T]]` | `sequence: Seq[T]` | `element_schema` for `T` |
| `ChunkSequence` | `sequence: Seq[T]` | `sequence: Seq[Seq[T]]` | `element_schema`, positive `size` |
| `Zip` | `first: Seq[A]`, `second: Seq[B]` | `sequence: Seq[{first: A, second: B}]` | `first_schema`, `second_schema` |
| `Entries` | `mapping: Map[T]` | `sequence: Seq[{key: String, value: T}]` | `element_schema` for mapping values |
| `SelectSequence` | `sequence: Seq[T]`, `decisions: Seq[Boolean]` | `sequence: Seq[T]` | `element_schema` |
| `GroupSequence` | `sequence: Seq[T]`, `keys: Seq[String]` | `mapping: Map[Seq[T]]` | `element_schema` |

The schemas are ordinary portable value schemas, including `x-value-type`.
In Python, supply `IntegerValue.to_value_schema()` for integer elements;
in JSON, supply its serialized schema object. Types are kept in params so a
saved graph can reconstruct its exact I/O types without Python-only attributes.
Records in the table are `DataValue` elements and maps are `StringMapValue`.

`FlattenSequence` removes exactly one nesting level. Empty inner lists contribute
nothing. `ChunkSequence` emits a short final chunk when needed and emits no
chunks for an empty input. A nonpositive size is rejected during validation.

`Zip` is strict: unequal lengths fail instead of discarding or shifting an
unpaired item. `SelectSequence` and `GroupSequence` similarly require one
decision/key per item. These shape errors use the ordinary failure channel and
can be caught by an enclosing `Attempt`. No operation interprets a `Result` tag.

`Entries` sorts keys lexicographically. Grouping preserves input order within
each group, including repeated keys and empty-string keys. Empty grouping
produces an empty mapping; empty `Entries` produces an empty sequence.

## Traverse and closure capture

`ForEach` is the traversal operator. It maps an inline workflow over `sequence`,
returning `sequence` in input order. A single per-item input is scalar; multiple
per-item fields form a `DataValue` record. The same collapse rule applies to
outputs; a workflow with no output fields runs for its side effects.

`constant_inputs` is an optional list of workflow input names supplied once on
separate top-level ports. With a step accepting `{item, config}`, setting
`constant_inputs: ["config"]` gives the traversal `{sequence: Seq[Item], config}`.
Each item receives the same `config` through ordinary expansion edges. Input
names must exist and be unique; `sequence` is reserved, and at least one field
must remain per-item. Omitting the list preserves existing graph behavior.

The inline workflow and names survive JSON round trips. Generated element ids
remain `element_0`, `element_1`, etc.; broadcast values do not affect those ids.
`ForEach(w)` uses ordinary failure propagation. `ForEach(Attempt(w))` collects
`Result` elements without dropping their positions.
