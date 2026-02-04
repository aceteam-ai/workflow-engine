# Matrix Determinant Example: Deep Field Access

This example showcases the **deep field access** feature of the workflow engine by calculating the determinant of a 2×2 matrix.

## What It Demonstrates

For a 2×2 matrix:
```
[[a, b],
 [c, d]]
```

The determinant is: `a×d - b×c`

### Deep Field Access in Action

The workflow uses **path expressions** like `("matrix", 0, 0)` to extract individual elements from a nested `SequenceValue[SequenceValue[FloatValue]]`:

```python
# Extract top-left element (a)
Edge(
    source_id="matrix_input",
    source_key=("matrix", 0, 0),  # Deep path!
    target_id="a_times_d",
    target_key="a"
)

# Extract bottom-right element (d)
Edge(
    source_id="matrix_input",
    source_key=("matrix", 1, 1),  # Deep path!
    target_id="a_times_d",
    target_key="b"
)
```

### Key Benefits

1. **No Intermediate Extraction Nodes**: Extract nested values directly in edge definitions
2. **Type-Safe**: Path validation happens at workflow construction time
3. **Clean & Readable**: The workflow structure clearly shows data flow
4. **Backwards Compatible**: Simple string keys still work exactly as before

## Workflow Structure

```
Input: matrix (2×2 nested SequenceValue)
   ↓
MatrixInputNode (pass-through)
   ├─[matrix, 0, 0]→ MultiplyNode (a×d)
   ├─[matrix, 1, 1]→     ↓
   ├─[matrix, 0, 1]→ MultiplyNode (b×c)
   └─[matrix, 1, 0]→     ↓
                    SubtractNode (a×d - b×c)
                         ↓
                    Output: determinant
```

## Running the Example

```bash
# Run the test
uv run pytest tests/test_matrix_determinant.py -v

# Example output for matrix [[3, 8], [4, 6]]:
# determinant = 3×6 - 8×4 = 18 - 32 = -14
```

## JSON Representation

The deep paths are serialized as JSON arrays:

```json
{
  "source_id": "matrix_input",
  "source_key": ["matrix", 0, 0],
  "target_id": "a_times_d",
  "target_key": "a"
}
```

See [`matrix_determinant.json`](./matrix_determinant.json) for the complete workflow definition.

## Deep Field Access Syntax

### Supported Path Types

- **String segment**: Access Data fields or StringMapValue keys
  ```python
  "field_name"  # Simple field access (backwards compatible)
  ("data", "field")  # Nested field access
  ```

- **Integer segment**: Index into SequenceValue
  ```python
  ("items", 0)  # Access first element of a sequence
  ("matrix", 1, 0)  # Access matrix[1][0]
  ```

### Type Safety

Path validation ensures:
- Fields exist on Data types
- Indices are only applied to SequenceValue
- String keys are only applied to Data or StringMapValue
- Final type is compatible with target
