# Execution

Execution algorithms determine how workflow nodes are scheduled and run.

## TopologicalExecutionAlgorithm

Executes nodes sequentially in topological (dependency) order. Each node runs to completion before the next one starts.

```python
from workflow_engine.execution import TopologicalExecutionAlgorithm

algorithm = TopologicalExecutionAlgorithm()
errors, output = await algorithm.execute(context=context, workflow=workflow, input=data)
```

Best for: simple workflows, debugging, deterministic execution.

## ParallelExecutionAlgorithm

Executes independent nodes concurrently using asyncio. Nodes are dispatched eagerly as soon as their dependencies are satisfied.

```python
from workflow_engine.execution import ParallelExecutionAlgorithm

algorithm = ParallelExecutionAlgorithm(
    max_concurrency=4,
)
```

### Error Handling Modes

```python
from workflow_engine.execution.parallel import ErrorHandlingMode

# Stop on first error (default)
algorithm = ParallelExecutionAlgorithm(
    error_handling=ErrorHandlingMode.FAIL_FAST,
)

# Continue executing, collect all errors
algorithm = ParallelExecutionAlgorithm(
    error_handling=ErrorHandlingMode.CONTINUE,
)
```

- **FAIL_FAST**: Cancels all running tasks when any node fails. Returns immediately with the error.
- **CONTINUE**: Keeps running nodes that don't depend on the failed node. Returns all errors and any partial output.

### Concurrency Limit

```python
# Unlimited concurrency (default)
algorithm = ParallelExecutionAlgorithm(max_concurrency=None)

# Limit to 8 concurrent nodes
algorithm = ParallelExecutionAlgorithm(max_concurrency=8)
```

## Retry

Both algorithms support automatic retry for transient failures. Nodes signal retryable failures by raising `ShouldRetry`:

```python
from workflow_engine import ShouldRetry
from datetime import timedelta

class MyNode(Node[MyInput, MyOutput, Empty]):
    async def run(self, *, context, input_type, output_type, input):
        try:
            return await call_external_api(input)
        except RateLimitError:
            raise ShouldRetry(
                message="Rate limited by API",
                backoff=timedelta(seconds=30),
            )
```

### Configuration

```python
# Set default max retries (applies to all nodes)
algorithm = TopologicalExecutionAlgorithm(max_retries=5)

# Or with parallel execution
algorithm = ParallelExecutionAlgorithm(max_retries=5)
```

The retry system uses exponential backoff based on the `backoff` value in `ShouldRetry`. The `RetryTracker` manages retry state across all nodes during execution.

## Rate Limiting

Rate limiting controls how frequently nodes of a given type can execute. This is useful for nodes that call external APIs with rate limits.

```python
from datetime import timedelta
from workflow_engine.execution.rate_limit import RateLimitConfig, RateLimitRegistry

# Create a registry
registry = RateLimitRegistry()

# Limit "ApiCall" nodes to 2 concurrent, 10 per minute
registry.configure("ApiCall", RateLimitConfig(
    max_concurrency=2,
    requests_per_window=10,
    window_duration=timedelta(minutes=1),
))

# Limit "ImageGen" nodes to 1 concurrent
registry.configure("ImageGen", RateLimitConfig(
    max_concurrency=1,
))

# Pass to either algorithm
algorithm = ParallelExecutionAlgorithm(rate_limits=registry)
# or
algorithm = TopologicalExecutionAlgorithm(rate_limits=registry)
```

### RateLimitConfig Options

| Parameter             | Type          | Default    | Description                                         |
| --------------------- | ------------- | ---------- | --------------------------------------------------- |
| `max_concurrency`     | `int \| None` | `None`     | Maximum concurrent executions (None = unlimited)    |
| `requests_per_window` | `int \| None` | `None`     | Maximum requests per time window (None = unlimited) |
| `window_duration`     | `timedelta`   | 60 seconds | Time window for request rate limiting               |

## Node Expansion

Some nodes (like `ForEach`, `If`, `IfElse`) are composite: they expand into sub-workflows at execution time. The execution algorithm handles this transparently:

1. When a composite node is encountered, its `expand()` method is called
2. The returned sub-workflow replaces the composite node in the execution graph
3. Execution continues with the expanded graph

This expansion happens dynamically during execution, not at workflow construction time.

## Error Boundaries

`attempt` (`AttemptNode`, see [`schema/attempt.md`](../schema/attempt.md)) runs an inner workflow inside an error boundary, producing `Result[B]` instead of letting a failure inside it propagate to the run. It is an expanding node like `ForEach`, marked by the core-level `ErrorBoundaryNode` ABC. Both execution algorithms register a boundary at the single site where an `ErrorBoundaryNode` expands, keyed by the node's flat id; membership of every other node is a `/`-prefix test on that id, so ids stay flat and downstream ledger, resume, and pin machinery keeps working unmodified. Nesting is supported: a failure fails the innermost enclosing boundary, and an outer boundary's own failure blocks (and, once nothing under it is in flight or yielded, sweeps) everything inside it.

### Semantics

- **Contained, not classified.** Boundary containment is orthogonal to `ErrorHandlingMode`: a boundary-contained error never reaches run-level `errors`, in `FAIL_FAST` or `CONTINUE` mode.
- **Yield wins.** If any member of a boundary has yielded in the current pass, the boundary does not materialize `err` in that pass. New scheduling inside the boundary still stops (fail-fast within the boundary: not-yet-dispatched members are reported cancelled, not run), in-flight members drain, the run returns `YIELDED`, and the boundary is re-evaluated from scratch on the resume pass. This is a deliberate divergence from run-level precedence (where, in `CONTINUE` mode, an unboundaried error takes precedence over a yield) because the boundary's whole purpose is to keep the run alive for a later resume, and a partially-committed `err` while a member is still suspended would either strand that member's remote work or invite an arm flip on the next pass.
- **Drain, not kill.** A member already dispatched when its boundary fails runs to its own completion (success, failure, or yield) and gets its normal terminal hook (`on_node_finish` / `on_node_error` / `on_node_yield`). Nothing calls `task.cancel()` on a boundary member.
- **Retries pass through.** A member's own `ShouldRetry` handling is unaffected by being inside a boundary. If retries are exhausted, that becomes the boundary's failure like any other. A member still in backoff when its boundary fails is not re-dispatched; it is reported cancelled instead (see below).

### Hooks

Two `ExecutionContext` hooks exist for boundaries, both with safe default implementations so a host that does not override them is unaffected:

- `on_node_cancelled(node, input_type, output_type, input, boundary_id, reason, cause)`: fires once per pass for each member of a failed boundary that will not run this pass. `reason` is a `CancelReason`: `NOT_SCHEDULED` (the boundary failed before this member was dispatched; `input` is `None`) or `RETRY_ABANDONED` (the member was in `ShouldRetry` backoff; `input` is its input). Never fires for a member that was in flight (it gets its normal terminal hook instead), nor for a yielded member, nor for the boundary's own output node.
- `on_boundary_error(node, input_type, output_type, input, error, output, cause)`: fires when a boundary materializes its `err` arm, after every member has settled and none yielded this pass. `node` is the boundary node itself (e.g. the `AttemptNode`); `output` is the mapping about to be written as the output of its output node, returnable (possibly replaced) like `on_node_finish`. A host that persists `output` against `node.id` may safely short-circuit the whole boundary from `on_node_start` on a later pass, since by construction nothing inside is suspended when this hook fires.
