# Built-in Nodes

All built-in nodes are in `src/workflow_engine/nodes/`. Import them via `import workflow_engine.nodes` to register them for deserialization.

## Arithmetic

### Add

Adds two numbers.

| Field | Type |
|-------|------|
| **Input** `a` | `FloatValue` |
| **Input** `b` | `FloatValue` |
| **Output** `sum` | `FloatValue` |

```json
{"type": "Add", "id": "add1", "params": {}}
```

### Sum

Sums a sequence of numbers.

| Field | Type |
|-------|------|
| **Input** `values` | `SequenceValue[FloatValue]` |
| **Output** `sum` | `FloatValue` |

### Factorization

Factorizes an integer into its prime factors.

| Field | Type |
|-------|------|
| **Input** `value` | `IntegerValue` |
| **Output** `factors` | `SequenceValue[IntegerValue]` |

## Constants

### ConstantBoolean

Outputs a constant boolean value.

| Field | Type |
|-------|------|
| **Parameter** `value` | `BooleanValue` |
| **Output** `value` | `BooleanValue` |

### ConstantInteger

Outputs a constant integer value.

| Field | Type |
|-------|------|
| **Parameter** `value` | `IntegerValue` |
| **Output** `value` | `IntegerValue` |

### ConstantString

Outputs a constant string value.

| Field | Type |
|-------|------|
| **Parameter** `value` | `StringValue` |
| **Output** `value` | `StringValue` |

## Conditional

### If

Executes a sub-workflow if the condition is true. Output is always `Empty` (since the sub-workflow may not execute).

| Field | Type |
|-------|------|
| **Input** `condition` | `BooleanValue` |
| **Input** *(additional)* | Fields from `if_true` workflow's input type |
| **Parameter** `if_true` | `WorkflowValue` |
| **Output** | `Empty` |

### IfElse

Executes one of two sub-workflows based on a condition. Output type is the intersection of both sub-workflow output types.

| Field | Type |
|-------|------|
| **Input** `condition` | `BooleanValue` |
| **Input** *(additional)* | Fields from sub-workflow input types |
| **Parameter** `if_true` | `WorkflowValue` |
| **Parameter** `if_false` | `WorkflowValue` |
| **Output** | Intersection of both workflow outputs |

## Iteration

### ForEach

Executes a sub-workflow for each item in an input sequence. Dynamically expands into `ExpandSequence` -> N copies of the sub-workflow -> `GatherSequence`.

| Field | Type |
|-------|------|
| **Input** `sequence` | `SequenceValue[DataValue[workflow.input_type]]` |
| **Parameter** `workflow` | `WorkflowValue` |
| **Output** `sequence` | `SequenceValue[DataValue[workflow.output_type]]` |

## Data Manipulation

These nodes are primarily used internally by composite nodes (ForEach, If, IfElse) but can be used directly.

### ExpandSequence / GatherSequence

Splits a sequence into individual elements (`element_0`, `element_1`, ...) or collects them back.

| Field | Type |
|-------|------|
| **Parameter** `length` | `IntegerValue` |

### ExpandMapping / GatherMapping

Splits a string-keyed mapping into individual fields or collects them back.

| Field | Type |
|-------|------|
| **Parameter** `keys` | `SequenceValue[StringValue]` |

### ExpandData / GatherData

Splits a `DataValue` into its component fields or wraps fields into a `DataValue`.

## Text

### AppendToFile

Appends text to a file, with an optional suffix.

| Field | Type |
|-------|------|
| **Input** `file` | `TextFileValue` |
| **Input** `text` | `StringValue` |
| **Parameter** `suffix` | `StringValue` |
| **Output** `file` | `TextFileValue` |

## Error

### Error

Always raises a `UserException`. Useful for testing error handling or for explicit failure conditions.

| Field | Type |
|-------|------|
| **Input** `info` | `StringValue` |
| **Parameter** `error_name` | `StringValue` |
