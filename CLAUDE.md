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

- **Workflow**: A DAG of nodes with typed data flow between them. Contains an `input_node`, `inner_nodes`, `output_node`, and `edges`.
- **InputNode**: Special node that defines the workflow's input schema. Created with `InputNode.from_fields()`.
- **OutputNode**: Special node that defines the workflow's output schema. Created with `OutputNode.from_fields()`.
- **Node**: A unit of computation with typed inputs, outputs, and parameters
- **Edge**: Connects a node output field to another node's input field with type validation. All edges (including to/from input/output nodes) use the same format.
- **Value**: Type-safe immutable wrapper around data (IntegerValue, StringValue, FileValue, etc.)
- **Data**: Immutable Pydantic model containing only Value fields
- **Context**: Execution environment providing file I/O and lifecycle hooks
- **ExecutionAlgorithm**: Strategy for scheduling node execution (currently topological sort)

### Module Structure

```text
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

    @cached_property
    def input_type(self):
        return MyInput

    @cached_property
    def output_type(self):
        return MyOutput

    async def run(self, context: Context, input: MyInput) -> MyOutput:
        # Implementation
        pass
```

**Field Titles and Descriptions**: Every field in `Data`, `Params`, and output model classes must have a `title` and `description` via `Field()`. These are surfaced to end users, so write them for a non-technical audience:

- `title`: A short, human-readable label (title case, e.g. `"Error Name"`)
- `description`: A noun phrase starting with "The", followed by full sentences only if more context is needed (e.g. `"The name of the error to raise."`)

```python
class MyInput(Data):
    value: IntegerValue = Field(title="Value", description="The input value.")
    items: SequenceValue[StringValue] = Field(title="Items", description="The list of items to process.")
```

**Immutability**: All core objects are frozen Pydantic models. Use `model_copy()` for updates.

**Async Execution**: Node `run()` methods and Context hooks are all async.

**Value Casting**: Type conversion between Values is async via `can_cast_to()` and `cast_to()` methods.

**Node Registration**: Nodes auto-register via `__init_subclass__` when they define a `type` discriminator field.

### Coding Best Practices

**Prefer Explicit Methods Over Dunder Method Overrides**

Avoid clever dunder method overrides (`__contains__`, `__getitem__`, etc.) in favor of explicit named methods:

```python
# ❌ Avoid: Dunder methods are hard to trace with IDE tools
class Registry:
    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __getitem__(self, name: str) -> Item:
        return self._items[name]

# Usage: hard to Ctrl+Click or find references
if "foo" in registry:  # Where does this go? IDE can't tell
    item = registry["foo"]  # Where does this go?

# ✅ Prefer: Explicit methods are IDE-friendly
class Registry:
    def has_name(self, name: str) -> bool:
        return name in self._items

    def get_item(self, name: str) -> Item:
        return self._items[name]

# Usage: IDE can jump to definition, autocomplete works
if registry.has_name("foo"):  # Ctrl+Click jumps to has_name()
    item = registry.get_item("foo")  # Ctrl+Click jumps to get_item()
```

**Rationale:**

- Explicit methods show up in IDE autocomplete
- "Go to Definition" / Ctrl+Click works properly
- Easier to search for all usages in codebase
- More discoverable for new developers
- Self-documenting code

**Exceptions where dunder methods are appropriate:**

- Core data structures (custom collections, numerical types)
- Protocol implementations (context managers, iterators)
- Pydantic model internals (`__init_subclass__`, `model_validator`)

## Release Process

To cut a new release:

1. **Update `CHANGELOG.md`** — study the commits since the last release tag (`git log <last-tag>..HEAD`), then add a new section at the top (after the header) with the new version, date, and a human-readable summary of what changed. Group entries under `### Added`, `### Changed`, and/or `### Fixed` as appropriate. Follow the existing format.

2. **Run the release script** — must be on a clean, up-to-date `main` branch. The script bumps the version in `pyproject.toml` and `src/workflow_engine/__init__.py`, then runs the full check suite (tests, ruff check, ruff format, pyright) before committing, tagging, pushing, and creating a GitHub release:

```bash
./release.sh --rc      # bump release candidate: 2.0.0rc4 -> 2.0.0rc5
./release.sh           # bump patch: 2.0.0 -> 2.0.1
./release.sh --minor   # bump minor: 2.0.0 -> 2.1.0
./release.sh --major   # bump major: 2.0.0 -> 3.0.0
./release.sh 2.0.0     # explicit version
```

The script prompts for confirmation before making any changes.

3. **PyPI publish** — triggered automatically by GitHub Actions when the tag is pushed. For a manual deploy (emergency or CI unavailable), use `./deploy.sh`.

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
