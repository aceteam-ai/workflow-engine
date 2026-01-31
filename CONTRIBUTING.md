# Contributing to Aceteam Workflow Engine

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

### Getting Started

```bash
# Clone the repository
git clone https://github.com/aceteam-ai/workflow-engine.git
cd workflow-engine

# Install all dependencies (including dev)
uv sync

# Verify your setup
uv run pytest
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting.

```bash
# Check for lint errors
uv run ruff check .

# Auto-format code
uv run ruff format .
```

Ruff configuration is in `pyproject.toml`. Key rules:
- Line length: default (88 characters)
- Import sorting handled by Ruff

## Type Checking

We use [Pyright](https://github.com/microsoft/pyright) for static type analysis.

```bash
uv run pyright
```

All public APIs should have type annotations. The codebase makes heavy use of generics (especially in Value types and Nodes).

## Testing

We use [pytest](https://docs.pytest.org/) with [pytest-asyncio](https://pytest-asyncio.readthedocs.io/) for async tests.

```bash
# Run all tests
uv run pytest

# Run only unit tests
uv run pytest -m unit

# Run only integration tests
uv run pytest -m integration

# Run a specific test file
uv run pytest tests/test_addition.py

# Run a specific test
uv run pytest tests/test_value.py::TestValue::test_cast

# Run with verbose output
uv run pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Use `pytest.mark.unit` or `pytest.mark.integration` markers
- Use `InMemoryContext` for unit tests (no filesystem side effects)
- All node `run()` methods are async, so use `async def test_*` with pytest-asyncio

## Creating a New Node

Nodes follow a specific pattern. Here's a step-by-step guide:

### 1. Define Input/Output Data Types

```python
from workflow_engine import Data, StringValue, IntegerValue

class MyInput(Data):
    text: StringValue

class MyOutput(Data):
    length: IntegerValue
```

### 2. Define Parameters (or use Empty)

```python
from workflow_engine import Empty

# Use Empty if no parameters needed
# Or define a custom params class:
class MyParams(Data):
    max_length: IntegerValue
```

### 3. Define the Node

```python
from typing import ClassVar, Literal, Type
from workflow_engine import Node, Context
from workflow_engine.core.node import NodeTypeInfo

class MyNode(Node[MyInput, MyOutput, MyParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="MyNode",
        display_name="My Node",
        description="Computes the length of a string",
        version="1.0.0",
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
        length = len(input.text)
        return MyOutput(length=IntegerValue(length))
```

### 4. Register the Node

Nodes auto-register when they define a `type` discriminator field with a `Literal` type. Make sure your module is imported (add it to `src/workflow_engine/nodes/__init__.py`).

### 5. Write Tests

```python
import pytest
from workflow_engine import IntegerValue, StringValue
from workflow_engine.contexts import InMemoryContext

@pytest.mark.unit
async def test_my_node():
    node = MyNode(type="MyNode", id="test", params=MyParams(max_length=IntegerValue(100)))
    context = InMemoryContext()
    result = await node.run(context, MyInput(text=StringValue("hello")))
    assert result.length == IntegerValue(5)
```

## Node Versioning

All nodes use [semantic versioning](https://semver.org/):

- **Patch** (0.4.0 -> 0.4.1): Bug fixes, no schema changes
- **Minor** (0.4.0 -> 0.5.0): New optional fields, backward-compatible changes
- **Major** (0.4.0 -> 1.0.0): Breaking schema changes

When making breaking changes, write a migration:

```python
from workflow_engine import Migration, migration_registry

@migration_registry.register
class MyNode_1_0_0_to_2_0_0(Migration):
    node_type = "MyNode"
    from_version = "1.0.0"
    to_version = "2.0.0"

    def migrate(self, data):
        result = dict(data)
        params = dict(result.get("params", {}))
        params["new_field"] = params.pop("old_field", "default")
        result["params"] = params
        return result
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure all checks pass:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run pyright
   uv run pytest
   ```
4. Open a PR against `main`
5. PRs require review before merging

## Key Design Principles

- **Immutability**: All core objects are frozen Pydantic models. Use `model_copy()` for updates.
- **Async-first**: Node execution and value casting are all async.
- **Type safety**: Data flow between nodes is validated at workflow construction time.
- **Simplicity**: Prefer simple, focused implementations over abstractions.
