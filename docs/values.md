# Value Type System

Values are type-safe, immutable wrappers around data. They are the currency of data flow between nodes.

## Value Schemas and Type Resolution

Value types serialize to JSON Schema via `to_value_schema()` (which uses Pydantic’s `model_json_schema()`). These schemas describe the structure of values for validation and type resolution.

### How schema resolution works

1. **Title-based lookup**: Each Value type can register itself in a `ValueRegistry` by name (e.g. `"IntegerValue"`, `"JSONValue"`). When a schema has a `title` that matches a registered type, it resolves to that type immediately.

2. **$defs and $ref**: Nested and recursive types use JSON Schema `$defs` and `$ref`. For example, `SequenceValue[JSONValue]` produces a schema where the array’s `items` is a `$ref` to `#/$defs/JSONValue`. The registry can supply these defs via `extra_defs` so references resolve without embedding `$defs` in the schema.

3. **Composite def IDs**: For types nested beyond one level (e.g. `StringMapValue[SequenceValue[StringMapValue[IntegerValue]]]`), Pydantic generates composite def IDs such as `SequenceValue_StringMapValue_IntegerValue__`. These IDs are internal to that schema and do **not** correspond to any registry entry.

### Limitation: deeply nested generics require $defs

For types with **one level of nesting** (e.g. `SequenceValue[JSONValue]`, `StringMapValue[IntegerValue]`), you can omit `$defs` from the schema and still resolve correctly by passing the registry’s types as `extra_defs`, since the referenced type (e.g. `JSONValue`) is registered.

For **two or more levels of nesting**, resolution fails without `$defs`. The composite def IDs (like `SequenceValue_StringMapValue_IntegerValue__`) are schema-specific; they cannot be reconstructed from the registry alone. The registry only knows base types (`SequenceValue`, `StringMapValue`, `IntegerValue`), not parameterized combinations. If you strip `$defs`, those references cannot be resolved.

**Takeaway**: Schemas with deeply nested generics must include `$defs` for full round-trip type resolution. This is a limitation of how Pydantic generates JSON Schema for recursive generics.

## Primitive Values

| Type           | Wraps   | Notes                                                |
| -------------- | ------- | ---------------------------------------------------- |
| `BooleanValue` | `bool`  |                                                      |
| `FloatValue`   | `float` | Has `is_integer()` method                            |
| `IntegerValue` | `int`   | Implements `__index__()` for use as sequence indices |
| `NullValue`    | `None`  |                                                      |
| `StringValue`  | `str`   | Supports `len()` and `in` operator                   |

### Usage

```python
from workflow_engine import IntegerValue, StringValue, FloatValue

x = IntegerValue(42)
y = FloatValue(3.14)
name = StringValue("hello")

# Access the underlying Python value
print(x.root)  # 42
print(len(name))  # 5
```

## Collection Values

### SequenceValue[T]

A generic sequence of values. `T` must be a `Value` subtype.

```python
from workflow_engine import SequenceValue, IntegerValue

seq = SequenceValue[IntegerValue](root=[IntegerValue(1), IntegerValue(2), IntegerValue(3)])
print(len(seq))     # 3
print(seq[0])       # IntegerValue(1)

for item in seq:
    print(item)
```

### StringMapValue[V]

A string-keyed mapping of values. `V` must be a `Value` subtype.

```python
from workflow_engine import StringMapValue, StringValue

mapping = StringMapValue[StringValue](root={"key": StringValue("value")})
print(mapping["key"])       # StringValue("value")
print("key" in mapping)     # True

for key, value in mapping.items():
    print(f"{key}: {value}")
```

### DataValue[D]

Wraps a `Data` object (typed container of Value fields) as a single Value.

```python
from workflow_engine import Data, DataValue, StringValue, IntegerValue

class Person(Data):
    name: StringValue
    age: IntegerValue

person = Person(name=StringValue("Alice"), age=IntegerValue(30))
wrapped = DataValue[Person](root=person)
```

## Structured Values

### JSONValue

Wraps arbitrary JSON-compatible data (dicts, lists, strings, numbers, booleans, null).

```python
from workflow_engine import JSONValue

data = JSONValue(root={"key": [1, 2, 3], "nested": {"a": True}})
```

## File Values

File values reference files managed by the execution `Context`.

| Type                 | MIME Type           | Key Methods                                   |
| -------------------- | ------------------- | --------------------------------------------- |
| `FileValue`          | (base class)        | `read()`, `write()`, `copy_from_local_file()` |
| `TextFileValue`      | `text/plain`        | `read_text()`, `write_text()`                 |
| `JSONFileValue`      | `application/json`  | `read_data()`, `write_data()`                 |
| `JSONLinesFileValue` | `application/jsonl` | `read_data()`, `write_data()`                 |
| `PDFFileValue`       | `application/pdf`   |                                               |

## Type Casting

Values can be automatically cast between compatible types. Casting is async and uses a registered `Caster` system.

### Checking Cast Compatibility

```python
from workflow_engine import IntegerValue, FloatValue

# Static check (no value needed)
can_cast = IntegerValue.can_cast_to(FloatValue)  # True

# Perform the cast
value = IntegerValue(42)
result = await value.cast_to(FloatValue)  # FloatValue(42.0)
```

### Available Casts

**Primitive conversions:**

| From           | To             | Condition              |
| -------------- | -------------- | ---------------------- |
| `IntegerValue` | `FloatValue`   | Always                 |
| `FloatValue`   | `IntegerValue` | Only if `is_integer()` |
| Any `Value`    | `StringValue`  | Always (via `str()`)   |
| `StringValue`  | `BooleanValue` | Via JSON parsing       |
| `StringValue`  | `IntegerValue` | Via JSON parsing       |
| `StringValue`  | `FloatValue`   | Via JSON parsing       |

**JSON conversions:**

| From        | To               | Condition                    |
| ----------- | ---------------- | ---------------------------- |
| Any `Value` | `JSONValue`      | Always (via `model_dump()`)  |
| `JSONValue` | `NullValue`      | If value is `null`           |
| `JSONValue` | `BooleanValue`   | If value is `bool`           |
| `JSONValue` | `IntegerValue`   | If value is `int`            |
| `JSONValue` | `FloatValue`     | If value is `float` or `int` |
| `JSONValue` | `SequenceValue`  | If value is `list`           |
| `JSONValue` | `StringMapValue` | If value is `dict`           |

**File conversions:**

| From                 | To                                                         |
| -------------------- | ---------------------------------------------------------- |
| `TextFileValue`      | `StringValue`                                              |
| `StringValue`        | `TextFileValue`                                            |
| `JSONFileValue`      | Primitives, `SequenceValue`, `StringMapValue`, `JSONValue` |
| Any `Value`          | `JSONFileValue`                                            |
| `JSONLinesFileValue` | `SequenceValue[T]`                                         |
| `SequenceValue`      | `JSONLinesFileValue`                                       |

**Collection conversions:**

| From                | To                  | Condition                     |
| ------------------- | ------------------- | ----------------------------- |
| `SequenceValue[S]`  | `SequenceValue[T]`  | If `S` can cast to `T`        |
| `StringMapValue[S]` | `StringMapValue[T]` | If `S` can cast to `T`        |
| `DataValue[S]`      | `DataValue[T]`      | Field-by-field casting        |
| `DataValue[D]`      | `StringMapValue[V]` | If all fields can cast to `V` |
| `StringMapValue[V]` | `DataValue[D]`      | Runtime field matching        |

The full casting graph is visualized in the repository: [typecast_graph.svg](typecast_graph.svg).

## Result Values

`Result[T]` is a tagged ok/err value: exactly one of an `ok` payload (`T`) or
an `err` payload (a structured `ResultError`) is set, and which one is
recorded explicitly by a `tag` field. Unlike `UnionValue[T, ErrValue]`, a
`Result[T]` is always an instance of the single `Result` class — the tag
travels with the value instead of being inferred from which member type
validated. That is what makes `Result[Result[T]]` representable: each level
keeps its own tag, so nesting never collapses or loses a level on the way to
the wire and back.

```python
from workflow_engine import ErrorClass, ErrorClassValue, FloatValue, Result, ResultError, StringValue

ok = Result[FloatValue].ok(FloatValue(3.14))
ok.is_ok()       # True
ok.unwrap_ok()   # FloatValue(3.14)

error = ResultError(
    error_class=ErrorClassValue(ErrorClass.TIMEOUT),
    name=StringValue("FetchTimeout"),
    message=StringValue("The upstream service did not respond in time."),
    node_id=StringValue("fetch-1"),
)
err = Result[FloatValue].err(error)
err.is_err()        # True
err.unwrap_err()     # ResultError(...)
```

### The err arm: `ResultError`

| Field         | Type              | Notes                                                     |
| ------------- | ----------------- | ---------------------------------------------------------- |
| `error_class` | `ErrorClassValue` | Closed vocabulary: `timeout`, `unreachable`, `rate_limit`, `validation`, `permission`, `systemic`. |
| `name`        | `StringValue`     | Short, machine-readable name of the error.                 |
| `message`     | `StringValue`     | User-facing description of what went wrong.                |
| `node_id`     | `StringValue`     | The id of the node that produced the error (provenance).   |

`error_class` is a closed vocabulary rather than a free-form string so that
callers (retry policies, circuit breakers, run ledgers) can key off one field
instead of string-matching `message`. It is aligned with the workflow-engine
error classification used elsewhere in the engine.

### The published wire shape

See [`schema/result.md`](../schema/result.md) for the full published contract.
In short, `Result[T]` serializes as a tagged object with exactly the keys relevant to
its arm — `tag` plus `ok`, or `tag` plus `err`, never both:

```json
{"tag": "ok", "ok": <T's serialized value>}
```

```json
{
  "tag": "err",
  "err": {
    "error_class": "timeout",
    "name": "FetchTimeout",
    "message": "The upstream service did not respond in time.",
    "node_id": "fetch-1"
  }
}
```

The root is a discriminated union of two variant shapes (`tag: "ok"` with
`ok`, or `tag: "err"` with `err`), not a single object with two optional
sibling fields defaulting to `null`. That matters for round-tripping: a
sibling-fields shape would use `null` as the "this arm is absent" sentinel,
which is indistinguishable from a populated `ok` payload that is itself
`null` (e.g. `Result[NullValue]`). Routing on `tag` first means the payload
is only ever validated against its own type. Nesting preserves both levels'
tags:

```json
{
  "tag": "ok",
  "ok": {"tag": "err", "err": {"...": "..."}}
}
```

The value-type schema (`Result[T].to_value_schema()`) mirrors this shape —
`x-value-type` set to e.g. `"Result[FloatValue]"`, plus `ok` (the schema for
`T`) and `err` (the fixed `ResultError` schema) — rather than the generic
`properties`/`required` shape a plain `Data` record would produce. This keeps
`Result` round-tripping through its own dedicated schema variant instead of
being silently rebuilt as an ordinary 3-field record, which would lose the
ok/err distinction.

### Casting

`Result[S]` casts to `Result[T]` when `S` can cast to `T`: the `ok` arm casts
its payload, the `err` arm passes the `ResultError` through unchanged.

### Gather-side typing

A `SequenceValue[Result[T]]` needs no special handling from `GatherSequenceNode`
or `ExpandSequenceNode`: both already require one value per index (see
`nodes/data.py`), so an element that failed is present at its index tagged
`err`, never absent or shifting the indices after it.

## Union Values

`UnionValue[A, B, ...]` accepts any of several member types. Validated and cast values are always an instance of one member (`FloatValue`, `SequenceValue[FloatValue]`, …), never a wrapper object.

Assign a module-level type alias (pyright requires this for multi-member unions):

```python
from decimal import Decimal

from workflow_engine import Data, FloatValue, SequenceValue, UnionValue
from pydantic import Field

NumericValues = UnionValue[FloatValue, SequenceValue[FloatValue]]

class SumInput(Data):
    values: NumericValues = Field(
        title="Values",
        description="A scalar or sequence of numbers to sum.",
    )

def as_decimals(value: FloatValue | SequenceValue[FloatValue]) -> list[Decimal]:
    if isinstance(value, FloatValue):
        return [value.root]
    return [item.root for item in value.root]
```

At construction time, pass explicit member instances (`FloatValue(1.5)`, `NullValue(None)`, …). Pydantic still coerces raw Python values when deserializing (`model_validate`, JSON). Use `isinstance` on members in node code — not on `UnionValue` itself.

### Optional fields: `OptionalValue`

For optional fields (`T | NullValue`), use `OptionalValue[T]` — shorthand for `UnionValue[T, NullValue]`:

```python
from workflow_engine import Data, IntegerValue, NullValue, OptionalValue, StringValue

OptionalInteger = OptionalValue[IntegerValue]

class MessageItem(Data):
    sender_id: OptionalInteger
    text: OptionalValue[StringValue]

MessageItem(sender_id=NullValue(None), text=StringValue("hello"))
MessageItem(sender_id=IntegerValue(42), text=StringValue("hi"))
MessageItem(sender_id=NullValue(None), text=NullValue(None))
```

Both `UnionValue` and `OptionalValue` support call syntax too: `UnionValue(FloatValue, ...)`, `OptionalValue(IntegerValue)`.

## Creating Custom Values

To create a custom Value type:

```python
from workflow_engine import Value

class UrlValue(Value[str]):
    """A URL string value."""
    pass
```

To add casting support, register a `Caster`:

```python
from workflow_engine.core.values.value import Caster

class UrlToStringCaster(Caster[UrlValue, StringValue]):
    @classmethod
    def source_type(cls):
        return UrlValue

    @classmethod
    def target_type(cls):
        return StringValue

    @classmethod
    def can_cast(cls, source_type, target_type):
        return True

    @classmethod
    async def cast(cls, source, target_type, context):
        return StringValue(source.root)
```
