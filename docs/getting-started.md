# Getting Started

## Installation

```bash
pip install aceteam-workflow-engine
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add aceteam-workflow-engine
```

## Core Concepts

The workflow engine has four key concepts:

- **Workflow**: A directed acyclic graph (DAG) of nodes
- **Node**: A unit of computation with typed inputs, outputs, and parameters
- **Edge**: A connection between a node's output field and another node's input field
- **Context**: The execution environment that provides file I/O and lifecycle hooks

## Your First Workflow

### Running a JSON Workflow

The simplest way to start is loading and running an existing workflow:

```python
import asyncio
from workflow_engine import IntegerValue, Workflow
import workflow_engine.nodes  # Register built-in nodes
from workflow_engine.contexts import InMemoryContext
from workflow_engine.execution import TopologicalExecutionAlgorithm

# Create execution components
context = InMemoryContext()
algorithm = TopologicalExecutionAlgorithm()

# Load a workflow from JSON
with open("examples/addition.json") as f:
    workflow = Workflow.model_validate_json(f.read())

# Execute the workflow
errors, output = asyncio.run(algorithm.execute(
    context=context,
    workflow=workflow,
    input={"c": IntegerValue(-256)},
))

print(output)  # {'sum': IntegerValue(1811)}
```

### Building a Workflow in Code

You can also construct workflows programmatically:

```python
import asyncio
from workflow_engine import (
    Edge, IntegerValue, Workflow, WorkflowErrors,
)
from workflow_engine.nodes.arithmetic import AddNode, SumNode
from workflow_engine.nodes.constant import ConstantIntegerNode
from workflow_engine.contexts import InMemoryContext
from workflow_engine.execution import TopologicalExecutionAlgorithm

# Define nodes
const_a = ConstantIntegerNode(
    type="ConstantInteger",
    id="a",
    params={"value": IntegerValue(10)},
)
const_b = ConstantIntegerNode(
    type="ConstantInteger",
    id="b",
    params={"value": IntegerValue(20)},
)
add = AddNode(type="Add", id="add", params={})

# Define edges (connect outputs to inputs)
edges = [
    Edge(source="a", target="add", source_field="value", target_field="a"),
    Edge(source="b", target="add", source_field="value", target_field="b"),
]

# Build the workflow
workflow = Workflow(
    nodes=[const_a, const_b, add],
    edges=edges,
    output={"result": {"node": "add", "field": "sum"}},
)

# Execute
context = InMemoryContext()
algorithm = TopologicalExecutionAlgorithm()
errors, output = asyncio.run(algorithm.execute(
    context=context,
    workflow=workflow,
    input={},
))

print(output)  # {'result': FloatValue(30.0)}
```

## Choosing an Execution Algorithm

The engine provides two execution strategies:

### TopologicalExecutionAlgorithm

Executes nodes sequentially in dependency order. Simple and predictable.

```python
from workflow_engine.execution import TopologicalExecutionAlgorithm

algorithm = TopologicalExecutionAlgorithm()
```

### ParallelExecutionAlgorithm

Executes independent nodes concurrently using asyncio. Better for workflows with I/O-bound nodes.

```python
from workflow_engine.execution import ParallelExecutionAlgorithm

algorithm = ParallelExecutionAlgorithm(
    max_concurrency=4,  # Limit concurrent nodes
)
```

See [Execution](execution.md) for details on retry, rate limiting, and error handling modes.

## Choosing a Context

### InMemoryContext

Stores files in memory. Good for testing and ephemeral workflows.

```python
from workflow_engine.contexts import InMemoryContext
context = InMemoryContext()
```

### LocalContext

Persists files to the local filesystem. Supports caching and resumption.

```python
from workflow_engine.contexts import LocalContext
context = LocalContext(base_dir="./output")
```

See [Contexts](contexts.md) for details on lifecycle hooks and custom contexts.

## Error Handling

Workflow execution returns a tuple of `(errors, output)`:

```python
errors, output = await algorithm.execute(context=context, workflow=workflow, input={})

if errors:
    for node_id, error in errors.items():
        print(f"Node {node_id} failed: {error}")
else:
    print("Success:", output)
```

## Next Steps

- [Architecture](architecture.md) - Understand the module structure and design
- [Nodes](nodes.md) - Browse built-in node types
- [Values](values.md) - Learn about the type system
- [Execution](execution.md) - Configure retry, rate limiting, and parallelism
- [Contexts](contexts.md) - Customize storage and lifecycle hooks
