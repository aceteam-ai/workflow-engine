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

## Workflow operations

| Node | Inputs | Output | Inline workflow signature |
| --- | --- | --- | --- |
| `Fold` | `seed: A`, `sequence: Seq[T]` | `acc: A` | `{acc: A, item: T} -> {acc: A}` |
| `Filter` | `sequence: Seq[T]` | `sequence: Seq[T]` | `T -> Boolean` |
| `GroupBy` | `sequence: Seq[T]` | `mapping: Map[Seq[T]]` | `T -> String` |

Each takes `params.workflow`. For fold, the accumulator field must be named
`acc`, its input and output schemas must match, and the step must return only
that field. Other input fields form the item, using the traversal single-field
collapse rule. The input seed is returned unchanged on an empty sequence.
A fold never assumes associativity or evaluates a combining tree.

`Filter` and `GroupBy` require exactly one output field of the indicated type;
the field name is arbitrary. Multi-field items remain records. The predicate
or key workflow runs once per item, including items that themselves have a
`Result` type. A returned `Result[Boolean]` or `Result[String]` is not a decision:
use `Unwrap`, `UnwrapOrValue`, or other explicit routing inside the workflow to
produce the required boolean or key. A predicate/key failure uses ordinary
failure propagation; wrapping the operation in `Attempt` catches that failure.

`GroupBy -> Entries -> ForEach` exposes dynamic groups as an ordinary sequence
for further processing. Set `Entries.element_schema` to `SequenceValue[T]`'s
schema in this composition.

### Expansion and replay

All workflow operations emit flat graph expansions into the existing executor.
They do not run hidden sub-executions or loop over effectful workflows inside
one node. The host sees, checkpoints, and can replay each inner node normally.

Fold emits `expand` and a chain of `step_0`, `step_1`, etc. Each `FoldStep`
adapter takes the previous accumulator and its item before expanding the inline
workflow; its inner ids become `<fold-id>/step_<i>/<inner-id>`. Thus even an
independent source node inside a later step is not discovered before its
predecessor produces the accumulator. The accumulator is persisted through the
ordinary step output node. Generated ids depend only on input length and the
inline workflow, making replay deterministic after serialization.

Sequencing follows the accumulator's data dependencies. Detached side effects
that do not feed the accumulator may outlive their step's output and overlap
later steps, just as detached work in any ordinary workflow can outlive its
output. To sequence an effect, its completion value must feed the accumulator's
output path. This rule also applies inside any workflows a step expands. Fold
does not add a scope-wide completion barrier or change executor failure policy.

Filter and grouping emit `traverse` plus `combine`. Their inner ids become
`<operation-id>/traverse/element_<i>/<inner-id>`. The pure combine node preserves
alignment and runs after all decisions are available. Completed item work can
be loaded from a host context's cache after a yield; missing work resumes under
the same ids. Concurrency hints retain the normal portable annotation contract.

## Generating a sequence

[`Unfold`](unfold.md) complements traversal and fold: it repeatedly invokes a
cursor step until completion, with a required finite iteration budget and flat,
checkpointed expansion. It returns the concatenated page items as a sequence
that the operators above can consume.
