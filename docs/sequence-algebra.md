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
