# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aceteam Workflow Engine is a Python library for building and executing graph-based workflows. It uses Pydantic for validation, NetworkX for DAG operations, and asyncio for concurrent execution.

## Common Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run specific test markers
uv run pytest -m unit          # Unit tests only
uv run pytest -m integration   # Integration tests only

# Run a single test file or function
uv run pytest tests/test_addition.py
uv run pytest tests/test_value.py::TestValue::test_cast

# Linting and formatting
uv run ruff check .            # Check for lint errors
uv run ruff format .           # Auto-format code

# Type checking
uv run pyright
```

## Architecture

### Core Concepts

- **Workflow**: A DAG of nodes with typed data flow between them
- **Node**: A unit of computation with typed inputs, outputs, and parameters
- **Edge**: Connects a node output field to another node's input field with type validation
- **Value**: Type-safe immutable wrapper around data (IntegerValue, StringValue, FileValue, etc.)
- **Data**: Immutable Pydantic model containing only Value fields
- **Context**: Execution environment providing file I/O and lifecycle hooks
- **ExecutionAlgorithm**: Strategy for scheduling node execution (currently topological sort)

### Module Structure

```
src/workflow_engine/
├── core/           # Base classes: Node, Workflow, Edge, Context, Value
│   └── values/     # Value type system (primitives, file, json, sequence, mapping)
├── nodes/          # Built-in node implementations (arithmetic, conditional, iteration)
├── contexts/       # Storage backends (LocalContext, InMemoryContext)
├── execution/      # Execution strategies (TopologicalExecutionAlgorithm)
└── utils/          # Helpers (immutable base models, semver)
```

### Key Patterns

**Node Definition**: Nodes use a discriminator pattern with `type: Literal["NodeName"]` for polymorphic serialization:
```python
class MyNode(Node[MyInput, MyOutput, MyParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="MyNode",
        display_name="My Node",
        description="...",
        version="1.0.0",  # Semantic versioning required
        parameter_type=MyParams,
    )
    type: Literal["MyNode"] = "MyNode"

    @property
    def input_type(self) -> Type[MyInput]:
        return MyInput

    @property
    def output_type(self) -> Type[MyOutput]:
        return MyOutput

    async def run(self, context: Context, input: MyInput) -> MyOutput:
        # Implementation
        pass
```

**Immutability**: All core objects are frozen Pydantic models. Use `model_copy()` for updates.

**Async Execution**: Node `run()` methods and Context hooks are all async.

**Value Casting**: Type conversion between Values is async via `can_cast_to()` and `cast_to()` methods.

**Node Registration**: Nodes auto-register via `__init_subclass__` when they define a `type` discriminator field.

### Execution Flow

1. Load/build a `Workflow` (validates DAG structure, no cycles, types match)
2. Create a `Context` (LocalContext for files, InMemoryContext for testing)
3. Create an `ExecutionAlgorithm` (TopologicalExecutionAlgorithm)
4. Call `algorithm.execute(context, workflow, input_data)`
5. Handle `WorkflowErrors` and output data

### Error Handling

- `UserException`: User-visible errors with messages
- `NodeException`: Errors during node execution (includes node ID)
- `NodeExpansionException`: Errors during dynamic node replacement
- `WorkflowErrors`: Accumulates errors by node
