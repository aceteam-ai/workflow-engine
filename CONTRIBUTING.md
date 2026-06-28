# Contributing to AceTeam Workflow Engine

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

See [docs/authoring-nodes.md](docs/authoring-nodes.md) for the step-by-step
guide to writing, registering, and testing a node. When a change breaks a node's
schema, bump its version and add a migration — see
[docs/MIGRATIONS.md](docs/MIGRATIONS.md).

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
