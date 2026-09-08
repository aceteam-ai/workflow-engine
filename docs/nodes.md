# Built-in Nodes

All built-in nodes are in `src/workflow_engine/nodes/`. Import them via `import workflow_engine.nodes` to register them for deserialization.

## Arithmetic

### Add

Adds two numbers.

| Field            | Type         |
| ---------------- | ------------ |
| **Input** `a`    | `FloatValue` |
| **Input** `b`    | `FloatValue` |
| **Output** `sum` | `FloatValue` |

```json
{ "type": "Add", "id": "add1", "params": {} }
```

### Sum

Sums a sequence of numbers.

| Field              | Type                        |
| ------------------ | --------------------------- |
| **Input** `values` | `SequenceValue[FloatValue]` |
| **Output** `sum`   | `FloatValue`                |

### Factorization

Factorizes an integer into its prime factors.

| Field                | Type                          |
| -------------------- | ----------------------------- |
| **Input** `value`    | `IntegerValue`                |
| **Output** `factors` | `SequenceValue[IntegerValue]` |

## Comparison

Each comparison node takes two `FloatValue` inputs `a` and `b` and outputs a `BooleanValue` `result`. `IntegerValue` sources cast to `FloatValue` automatically.

| Node               | `result` is true when |
| ------------------ | --------------------- |
| `Equal`            | `a == b`              |
| `NotEqual`         | `a != b`              |
| `GreaterThan`      | `a > b`               |
| `GreaterThanEqual` | `a >= b`              |
| `LessThan`         | `a < b`               |
| `LessThanEqual`    | `a <= b`              |

```json
{ "type": "GreaterThan", "id": "gt1", "params": {} }
```

`Equal` and `NotEqual` compare with `math.isclose`. By default both tolerances are `0`, so the comparison is **exact**. To absorb floating-point rounding, set `rel_tol` (relative tolerance) and/or `abs_tol` (absolute tolerance); use `abs_tol` when comparing values near zero, where a relative tolerance is too strict.

```json
{ "type": "Equal", "id": "eq1", "params": { "rel_tol": 1e-6, "abs_tol": 1e-9 } }
```

## Logic

### And

Outputs true only when all inputs are true. Variadic like `Add`: the `num_arguments` parameter (default `2`, minimum `2`) controls how many boolean inputs (`a`, `b`, `c`, …) appear.

| Field                         | Type           |
| ----------------------------- | -------------- |
| **Parameter** `num_arguments` | `IntegerValue` |
| **Input** `a`, `b`, …         | `BooleanValue` |
| **Output** `result`           | `BooleanValue` |

```json
{ "type": "And", "id": "and1", "params": { "num_arguments": 3 } }
```

### Or

Outputs true when at least one input is true. Variadic like `And` via `num_arguments`.

| Field                         | Type           |
| ----------------------------- | -------------- |
| **Parameter** `num_arguments` | `IntegerValue` |
| **Input** `a`, `b`, …         | `BooleanValue` |
| **Output** `result`           | `BooleanValue` |

### Not

Returns the opposite of the input value.

| Field               | Type           |
| ------------------- | -------------- |
| **Input** `a`       | `BooleanValue` |
| **Output** `result` | `BooleanValue` |

## Constants

### ConstantBoolean

Outputs a constant boolean value.

| Field                 | Type           |
| --------------------- | -------------- |
| **Parameter** `value` | `BooleanValue` |
| **Output** `value`    | `BooleanValue` |

### ConstantInteger

Outputs a constant integer value.

| Field                 | Type           |
| --------------------- | -------------- |
| **Parameter** `value` | `IntegerValue` |
| **Output** `value`    | `IntegerValue` |

### ConstantString

Outputs a constant string value.

| Field                 | Type          |
| --------------------- | ------------- |
| **Parameter** `value` | `StringValue` |
| **Output** `value`    | `StringValue` |

## Conditional

### If

Executes a sub-workflow if the condition is true. Output is always `Empty` (since the sub-workflow may not execute).

| Field                    | Type                                        |
| ------------------------ | ------------------------------------------- |
| **Input** `condition`    | `BooleanValue`                              |
| **Input** _(additional)_ | Fields from `if_true` workflow's input type |
| **Parameter** `if_true`  | `WorkflowValue`                             |
| **Output**               | `Empty`                                     |

### IfElse

Executes one of two sub-workflows based on a condition. Output type is the intersection of both sub-workflow output types.

| Field                    | Type                                  |
| ------------------------ | ------------------------------------- |
| **Input** `condition`    | `BooleanValue`                        |
| **Input** _(additional)_ | Fields from sub-workflow input types  |
| **Parameter** `if_true`  | `WorkflowValue`                       |
| **Parameter** `if_false` | `WorkflowValue`                       |
| **Output**               | Intersection of both workflow outputs |

### MatchErrorClass

Runs one of several sub-workflows depending on `error_class`, the closed, engine-owned vocabulary carried by a `Result[T]` err arm (`timeout`, `unreachable`, `rate_limit`, `validation`, `permission`, `systemic`; see `core/error.py` and `docs/values.md`). `branches` must have exactly one entry per value `ErrorClass` currently defines: a missing or unrecognized key fails graph validation, naming the node and the offending value(s), rather than silently falling through to some default branch.

| Field                     | Type                                                          |
| ------------------------- | -------------------------------------------------------------- |
| **Input** `error_class`   | `ErrorClassValue`                                              |
| **Input** _(additional)_  | Fields common to every branch's input type                     |
| **Parameter** `branches`  | `StringMapValue[WorkflowValue]`, keyed by `ErrorClass` value    |
| **Output**                | Intersection of every branch's output                          |

**Why this node is exhaustive and other conditionals in this engine are not.** `error_class` is a closed vocabulary versioned with the engine itself, not a per-node error type versioned independently by whoever wrote that node. That is the one property that makes "does this conditional cover every case" a question graph validation can answer honestly. A conditional keyed on anything else in this engine cannot make the same promise, because nothing here can guarantee the branch author enumerated every value some other, independently-versioned vocabulary might ever take.

**The consequence, stated up front so it is not a surprise later.** `ErrorClass` is expected to grow: today's six values are not forever. When a seventh value is added to `ErrorClass`, every stored graph containing a `MatchErrorClass` node that branches on the current six will fail graph validation the next time it is loaded, until a human adds a branch for the new value (or removes the node). This is deliberate, not a bug to work around, and it is the reason this node exists at all: the alternative is a stored graph that silently routes the new class down whichever branch happens to look like a fallback, forever, with nobody told. A loud failure that names the node and the missing value is strictly better than a quiet misroute that surfaces as a support ticket months later. Before adding a new `ErrorClass` value, expect to update every graph that matches on it; that cost is the entire point of this check, not an accident of how it was implemented.

## Iteration

### ForEach

Executes a sub-workflow for each item in an input sequence. Dynamically expands into `ExpandSequence` -> N copies of the sub-workflow -> `GatherSequence`.

| Field                    | Type                                             |
| ------------------------ | ------------------------------------------------ |
| **Input** `sequence`     | `SequenceValue[DataValue[workflow.input_type]]`  |
| **Parameter** `workflow` | `WorkflowValue`                                  |
| **Output** `sequence`    | `SequenceValue[DataValue[workflow.output_type]]` |

## Data Manipulation

These nodes are primarily used internally by composite nodes (ForEach, If, IfElse) but can be used directly.

### ExpandSequence / GatherSequence

Splits a sequence into individual elements (`element_0`, `element_1`, ...) or collects them back.

| Field                  | Type           |
| ---------------------- | -------------- |
| **Parameter** `length` | `IntegerValue` |

### ExpandMapping / GatherMapping

Splits a string-keyed mapping into individual fields or collects them back.

| Field                | Type                         |
| -------------------- | ---------------------------- |
| **Parameter** `keys` | `SequenceValue[StringValue]` |

### ExpandData / GatherData

Splits a `DataValue` into its component fields or wraps fields into a `DataValue`.

## Text

### AppendToFile

Appends text to a file, with an optional suffix.

| Field                  | Type            |
| ---------------------- | --------------- |
| **Input** `file`       | `TextFileValue` |
| **Input** `text`       | `StringValue`   |
| **Parameter** `suffix` | `StringValue`   |
| **Output** `file`      | `TextFileValue` |

## Error

### Error

Always raises a `WorkflowException`. Useful for testing error handling or for explicit failure conditions.

| Field                      | Type          |
| -------------------------- | ------------- |
| **Input** `info`           | `StringValue` |
| **Parameter** `error_name` | `StringValue` |

## Date and Time

### Now

Outputs the current UTC date and time.

| Field           | Type        |
| --------------- | ----------- |
| **Output** `now` | `DateValue` |

```json
{ "type": "Now", "id": "now1", "params": {} }
```
