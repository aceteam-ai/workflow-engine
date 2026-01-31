# Contexts

A `Context` provides the execution environment for workflows. It handles file I/O and exposes lifecycle hooks for monitoring and caching.

## Built-in Contexts

### InMemoryContext

Stores files in a Python dictionary. No persistence, no side effects.

```python
from workflow_engine.contexts import InMemoryContext

context = InMemoryContext()
```

Best for: unit tests, ephemeral workflows, CI/CD pipelines.

### LocalContext

Stores files on the local filesystem with full lifecycle tracking.

```python
from workflow_engine.contexts import LocalContext

context = LocalContext(
    base_dir="./output",   # Base directory (default: "./local")
    run_id="my-run-001",   # Unique run ID (auto-generated if None)
)
```

**Directory structure created by LocalContext:**

```
output/
  my-run-001/
    workflow.json        # Serialized workflow definition
    input.json           # Workflow input data
    output.json          # Final output (on success)
    error.json           # Errors + partial output (on failure)
    files/               # File value storage
    input/               # Per-node input snapshots (node_id.json)
    output/              # Per-node output snapshots (node_id.json)
```

**Caching**: If `output/<node_id>.json` already exists, LocalContext returns the cached result and skips re-execution. This enables resumption of partially completed workflows.

Best for: production use, debugging, workflow resumption.

## Lifecycle Hooks

The `Context` base class defines async hooks called at each stage of execution. Override them in a custom context to add monitoring, caching, or transformation logic.

### Workflow-Level Hooks

```python
class MyContext(Context):
    async def on_workflow_start(self, workflow, input):
        """Called before workflow execution begins.

        Return a (WorkflowErrors, DataMapping) tuple to skip execution
        and use cached results. Return None to proceed normally.
        """
        return None

    async def on_workflow_finish(self, workflow, input, output):
        """Called after successful workflow execution.

        Can modify and return the output DataMapping.
        """
        return output

    async def on_workflow_error(self, workflow, input, errors, output):
        """Called when workflow execution produces errors.

        Can modify errors and partial output.
        Returns (WorkflowErrors, DataMapping).
        """
        return errors, output
```

### Node-Level Hooks

```python
class MyContext(Context):
    async def on_node_start(self, node, input):
        """Called before a node executes.

        Return a DataMapping to skip execution and use cached results.
        Return None to proceed normally.
        """
        return None

    async def on_node_finish(self, node, input, output):
        """Called after successful node execution.

        Can modify and return the output DataMapping.
        """
        return output

    async def on_node_error(self, node, input, exception):
        """Called when a node raises an exception.

        Can modify or replace the exception.
        Return the (possibly modified) exception.
        """
        return exception

    async def on_node_retry(self, node, input, retry_state):
        """Called when a node is scheduled for retry.

        retry_state contains attempt count, next retry time, etc.
        """
        pass
```

## Custom Context Example

```python
import logging
from workflow_engine import Context

logger = logging.getLogger(__name__)

class LoggingContext(Context):
    """A context that logs all lifecycle events."""

    def __init__(self):
        self._storage: dict[str, bytes] = {}

    async def read(self, file):
        return self._storage.get(file.path, b"")

    async def write(self, file, content):
        self._storage[file.path] = content

    async def on_node_start(self, node, input):
        logger.info(f"Starting node {node.id} ({node.type})")
        return None

    async def on_node_finish(self, node, input, output):
        logger.info(f"Finished node {node.id}: {list(output.keys())}")
        return output

    async def on_node_error(self, node, input, exception):
        logger.error(f"Node {node.id} failed: {exception}")
        return exception
```
