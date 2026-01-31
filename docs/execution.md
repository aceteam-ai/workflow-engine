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
    async def run(self, context, input):
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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_concurrency` | `int \| None` | `None` | Maximum concurrent executions (None = unlimited) |
| `requests_per_window` | `int \| None` | `None` | Maximum requests per time window (None = unlimited) |
| `window_duration` | `timedelta` | 60 seconds | Time window for request rate limiting |

## Node Expansion

Some nodes (like `ForEach`, `If`, `IfElse`) are composite: they expand into sub-workflows at execution time. The execution algorithm handles this transparently:

1. When a composite node is encountered, its `expand()` method is called
2. The returned sub-workflow replaces the composite node in the execution graph
3. Execution continues with the expanded graph

This expansion happens dynamically during execution, not at workflow construction time.
