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

## Shell

### Bash

Runs a shell command and captures its output. The command is a Jinja template
rendered with the node's `arguments`, so values from upstream nodes can be
substituted in (e.g. `echo Hello {{ name }}`).

> **Warning:** This node executes arbitrary shell commands in the same
> environment as the engine itself — it is effectively remote code execution.
> Only enable it for trusted, locally-operated engines, and never expose it on an
> engine that runs untrusted workflows or accepts workflows over the network.

| Field                          | Type                          |
| ------------------------------ | ----------------------------- |
| **Input** `arguments`          | `StringMapValue[StringValue]` |
| **Parameter** `command`        | `StringValue`                 |
| **Parameter** `combine_output` | `BooleanValue`                |
| **Output** `stdout`            | `TextFileValue`               |
| **Output** `stderr`            | `TextFileValue`               |
| **Output** `exit_code`         | `IntegerValue`                |

When `combine_output` is `true`, standard error is merged into standard output
and the node produces a single `output` (`TextFileValue`) field instead of the
separate `stdout`/`stderr` files (alongside `exit_code`).

## Error

### Error

Always raises a `WorkflowException`. Useful for testing error handling or for explicit failure conditions.

| Field                      | Type          |
| -------------------------- | ------------- |
| **Input** `info`           | `StringValue` |
| **Parameter** `error_name` | `StringValue` |
