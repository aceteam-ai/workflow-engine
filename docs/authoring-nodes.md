# Authoring a Node

This guide shows how to write a new node. For the catalog of nodes that ship
with the engine, see [nodes.md](nodes.md). For schema-change migrations, see
[MIGRATIONS.md](MIGRATIONS.md).

A node is a `Node[Input, Output, Params]` subclass. `Input` and `Output` are
`Data` subclasses describing the typed fields that flow in and out; `Params` is
a `Params` subclass (or `Empty`) for configuration that is fixed at workflow
build time, independent of the inputs.

## 1. Define the input and output data types

Every field needs a `title` and `description` (surfaced to end users — write
them for a non-technical audience).

```python
from pydantic import Field
from workflow_engine import Data, IntegerValue, StringValue


class MyInput(Data):
    text: StringValue = Field(title="Text", description="The text to measure.")


class MyOutput(Data):
    length: IntegerValue = Field(
        title="Length", description="The number of characters in the text."
    )
```

## 2. Define parameters (or use `Empty`)

Use `Empty` when the node has no parameters, or subclass `Params`:

```python
from pydantic import Field
from workflow_engine import IntegerValue, Params


class MyParams(Params):
    max_length: IntegerValue = Field(
        title="Max Length", description="The maximum allowed length."
    )
```

## 3. Define the node

Declare the type arguments, a `TYPE_INFO`, the static input/output types, and an
async `run`. Do **not** declare a `type` field — the discriminator is derived
automatically from the class name (`MyNode` → `"My"`; the `Node` suffix is
stripped).

```python
from typing import ClassVar, Type

from overrides import override
from workflow_engine import (
    Data,
    Empty,
    ExecutionContext,
    IntegerValue,
    Node,
    NodeTypeInfo,
)


class MyNode(Node[MyInput, MyOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="My Node",
        description="Computes the length of a string.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[MyInput]:
        return MyInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[MyOutput]:
        return MyOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[MyInput],
        output_type: Type[MyOutput],
        input: MyInput,
    ) -> MyOutput:
        return MyOutput(length=IntegerValue(len(input.text.root)))
```

Notes:

- `run` is `async` and all arguments are **keyword-only**. `input_type` /
  `output_type` are the resolved types for this invocation (relevant for nodes
  with dynamic types).
- `NodeTypeInfo.from_parameter_type` takes `display_name`, `version`, and
  `parameter_type` (required), plus optional `description` and `max_retries`.
  There is no `name=` argument.
- Access parameters inside `run` via `self.params`.
- `version` is a semantic version — see [Node versioning](#node-versioning).

## Static vs. dynamic types

If the input or output schema is fixed, override the `static_*` classmethods as
above. If the schema depends on the node's parameters (e.g. `Add` accepts
`num_arguments` operands), leave the `static_*` method returning `None` (the
default) and override the async `dynamic_input_type` / `dynamic_output_type`
instead, building the type at validation time with `build_data_type`. See
`src/workflow_engine/nodes/arithmetic.py` (`AddNode`) for a worked example.

## 4. Register the node

Nodes auto-register via `__init_subclass__` when the class is defined — every
concrete `Node` subclass registers itself on `NodeRegistry.DEFAULT` under its
derived type name. You only need to make sure the module is imported; add it to
`src/workflow_engine/nodes/__init__.py` for built-in nodes.

## 5. Write a test

```python
import pytest
from workflow_engine import IntegerValue, StringValue
from workflow_engine.contexts import InMemoryContext


@pytest.mark.unit
async def test_my_node():
    node = MyNode(id="test")
    context = InMemoryContext()
    result = await node.run(
        context=context,
        input_type=MyInput,
        output_type=MyOutput,
        input=MyInput(text=StringValue("hello")),
    )
    assert result.length == IntegerValue(5)
```

## Node versioning

All nodes use [semantic versioning](https://semver.org/):

- **Patch** (`0.4.0` → `0.4.1`): bug fixes, no schema changes
- **Minor** (`0.4.0` → `0.5.0`): new optional fields, backward-compatible changes
- **Major** (`0.4.0` → `1.0.0`): breaking schema changes

When you make a breaking change, write a migration — see
[MIGRATIONS.md](MIGRATIONS.md).
