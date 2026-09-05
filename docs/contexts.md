# Contexts

A `Context` provides the execution environment for workflows. It handles file I/O and exposes lifecycle hooks for monitoring and caching.

## Built-in Contexts

### InMemoryExecutionContext

Stores files in a Python dictionary. No persistence, no side effects.

```python
from workflow_engine.contexts import InMemoryExecutionContext

context = InMemoryExecutionContext()
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

`ExecutionContext` defines async hooks called at each stage of execution. All are keyword-only and every node-level hook receives `node`, `input_type`, and `output_type` in addition to the parameters shown below; override any of them in a custom context to add monitoring, caching, or transformation logic. Each has a working default (shown in the signature), so a host only needs to override the ones it cares about.

### Workflow-Level Hooks

```python
class MyContext(ExecutionContext):
    async def on_workflow_start(self, *, workflow, input) -> WorkflowExecutionResult | None:
        """Called before workflow execution begins.

        Return a WorkflowExecutionResult to skip execution and use a cached
        result. Return None to proceed normally.
        """
        return None

    async def on_workflow_finish(self, *, workflow, input, output) -> WorkflowExecutionResult:
        """Called after successful workflow execution (no errors, no yields)."""
        return WorkflowExecutionResult.success(output=output)

    async def on_workflow_error(
        self, *, workflow, input, errors, partial_output, node_yields
    ) -> WorkflowExecutionResult:
        """Called when workflow execution produces errors.

        node_yields carries the per-node yield messages for any nodes that
        also yielded during this run, even though errors take precedence.
        """
        return WorkflowExecutionResult.error(
            errors=errors, partial_output=partial_output, node_yields=node_yields
        )

    async def on_workflow_yield(
        self, *, workflow, input, partial_output, node_yields
    ) -> WorkflowExecutionResult:
        """Called when the workflow yields (one or more nodes raised ShouldYield
        and nothing failed at the run level)."""
        return WorkflowExecutionResult.yielded(
            partial_output=partial_output, node_yields=node_yields
        )
```

### Node-Level Hooks

```python
class MyContext(ExecutionContext):
    async def on_node_start(self, *, node, input_type, output_type, input) -> DataMapping | Workflow | None:
        """Called before a node executes.

        Return a DataMapping (or a Workflow, to short-circuit straight to an
        expansion) to skip execution and use a cached result. Return None to
        proceed normally.
        """
        return None

    async def on_node_finish(self, *, node, input_type, output_type, input, output) -> DataMapping:
        """Called after a node returns a DataMapping (not a Workflow).

        Can modify and return the output.
        """
        return output

    async def on_node_expand(self, *, node, input_type, output_type, input, workflow) -> ValidatedWorkflow:
        """Called after a node returns a Workflow (i.e. it expands into a
        subgraph). Can modify and return the workflow. Does NOT get
        on_node_finish for this node; see TestOnNodeFinish::test_not_called_for_expanding_nodes.
        """
        return workflow

    async def on_node_error(self, *, node, input_type, output_type, input, exception) -> WorkflowException | DataMapping:
        """Called when a node raises an exception.

        Return a different exception to replace it, or a DataMapping to
        absorb the error and continue as if the node had succeeded.
        """
        return exception

    async def on_node_yield(self, *, node, input_type, output_type, input, exception) -> None:
        """Called when a node raises ShouldYield. exception.message describes
        what the node is waiting for."""
        pass

    async def on_node_retry(self, *, node, input_type, output_type, input, exception, attempt) -> None:
        """Called when a node is scheduled for retry after raising ShouldRetry.
        attempt is 1 for the first retry, 2 for the second, etc."""
        pass

    async def on_node_cancelled(
        self, *, node, input_type, output_type, input, boundary_id, reason, cause
    ) -> None:
        """Called once per pass for a member of a failed error boundary
        (`attempt`, see the Error Boundaries section of docs/execution.md)
        that will not run this pass. reason is a CancelReason
        (NOT_SCHEDULED or RETRY_ABANDONED)."""
        pass

    async def on_boundary_error(
        self, *, node, input_type, output_type, input, error, output, cause
    ) -> DataMapping:
        """Called when an error boundary materializes its err arm. Mirrors
        on_node_finish: return the output (possibly replaced)."""
        return output
```

## Custom Context Example

```python
import logging
from workflow_engine import ExecutionContext

logger = logging.getLogger(__name__)

class LoggingContext(ExecutionContext):
    """A context that logs all lifecycle events."""

    def __init__(self):
        super().__init__()
        self._storage: dict[str, bytes] = {}

    async def read(self, file):
        return self._storage.get(file.path, b"")

    async def write(self, file, content):
        self._storage[file.path] = content
        return file

    async def on_node_start(self, *, node, input_type, output_type, input):
        logger.info(f"Starting node {node.id} ({node.type})")
        return None

    async def on_node_finish(self, *, node, input_type, output_type, input, output):
        logger.info(f"Finished node {node.id}: {list(output.keys())}")
        return output

    async def on_node_error(self, *, node, input_type, output_type, input, exception):
        logger.error(f"Node {node.id} failed: {exception}")
        return exception
```
