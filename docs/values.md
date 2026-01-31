# Value Type System

Values are type-safe, immutable wrappers around data. They are the currency of data flow between nodes.

## Primitive Values

| Type | Wraps | Notes |
|------|-------|-------|
| `BooleanValue` | `bool` | |
| `FloatValue` | `float` | Has `is_integer()` method |
| `IntegerValue` | `int` | Implements `__index__()` for use as sequence indices |
| `NullValue` | `None` | |
| `StringValue` | `str` | Supports `len()` and `in` operator |

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

| Type | MIME Type | Key Methods |
|------|-----------|-------------|
| `FileValue` | (base class) | `read()`, `write()`, `copy_from_local_file()` |
| `TextFileValue` | `text/plain` | `read_text()`, `write_text()` |
| `JSONFileValue` | `application/json` | `read_data()`, `write_data()` |
| `JSONLinesFileValue` | `application/jsonl` | `read_data()`, `write_data()` |
| `PDFFileValue` | `application/pdf` | |

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

| From | To | Condition |
|------|----|-----------|
| `IntegerValue` | `FloatValue` | Always |
| `FloatValue` | `IntegerValue` | Only if `is_integer()` |
| Any `Value` | `StringValue` | Always (via `str()`) |
| `StringValue` | `BooleanValue` | Via JSON parsing |
| `StringValue` | `IntegerValue` | Via JSON parsing |
| `StringValue` | `FloatValue` | Via JSON parsing |

**JSON conversions:**

| From | To | Condition |
|------|----|-----------|
| Any `Value` | `JSONValue` | Always (via `model_dump()`) |
| `JSONValue` | `NullValue` | If value is `null` |
| `JSONValue` | `BooleanValue` | If value is `bool` |
| `JSONValue` | `IntegerValue` | If value is `int` |
| `JSONValue` | `FloatValue` | If value is `float` or `int` |
| `JSONValue` | `SequenceValue` | If value is `list` |
| `JSONValue` | `StringMapValue` | If value is `dict` |

**File conversions:**

| From | To |
|------|----|
| `TextFileValue` | `StringValue` |
| `StringValue` | `TextFileValue` |
| `JSONFileValue` | Primitives, `SequenceValue`, `StringMapValue`, `JSONValue` |
| Any `Value` | `JSONFileValue` |
| `JSONLinesFileValue` | `SequenceValue[T]` |
| `SequenceValue` | `JSONLinesFileValue` |

**Collection conversions:**

| From | To | Condition |
|------|----|-----------|
| `SequenceValue[S]` | `SequenceValue[T]` | If `S` can cast to `T` |
| `StringMapValue[S]` | `StringMapValue[T]` | If `S` can cast to `T` |
| `DataValue[S]` | `DataValue[T]` | Field-by-field casting |
| `DataValue[D]` | `StringMapValue[V]` | If all fields can cast to `V` |
| `StringMapValue[V]` | `DataValue[D]` | Runtime field matching |

The full casting graph is visualized in the repository: [typecast_graph.svg](typecast_graph.svg).

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
