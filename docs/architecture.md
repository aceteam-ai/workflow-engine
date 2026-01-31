# Architecture

## Module Structure

```
src/workflow_engine/
├── __init__.py            # Public API exports and version
├── core/                  # Core abstractions
│   ├── context.py         # Context base class (file I/O, lifecycle hooks)
│   ├── data.py            # Data base class (typed field containers)
│   ├── edge.py            # Edge, InputEdge, OutputEdge definitions
│   ├── execution.py       # ExecutionAlgorithm base class
│   ├── file.py            # File reference type
│   ├── node.py            # Node base class, NodeTypeInfo, node registry
│   ├── workflow.py        # Workflow definition and validation
│   ├── values/            # Value type system
│   │   ├── value.py       # Value base class, Caster registry
│   │   ├── primitives.py  # BooleanValue, FloatValue, IntegerValue, NullValue, StringValue
│   │   ├── json.py        # JSONValue
│   │   ├── file.py        # FileValue base
│   │   ├── sequence.py    # SequenceValue[T]
│   │   ├── mapping.py     # StringMapValue[V]
│   │   ├── data.py        # DataValue[D]
│   │   └── files/         # Typed file values (text, json, pdf)
│   └── migration/         # Node versioning and migration
│       ├── migration.py   # Migration base class
│       ├── registry.py    # MigrationRegistry
│       ├── runner.py      # MigrationRunner
│       └── workflow_migration.py  # Workflow-level migration utilities
├── nodes/                 # Built-in node implementations
│   ├── arithmetic.py      # AddNode, SumNode, FactorizationNode
│   ├── conditional.py     # IfNode, IfElseNode
│   ├── constant.py        # ConstantBooleanNode, ConstantIntegerNode, ConstantStringNode
│   ├── data.py            # Gather/Expand nodes for sequences, mappings, data
│   ├── error.py           # ErrorNode
│   ├── iteration.py       # ForEachNode
│   └── text.py            # AppendToFileNode
├── contexts/              # Context implementations
│   ├── in_memory.py       # InMemoryContext
│   └── local.py           # LocalContext
├── execution/             # Execution algorithm implementations
│   ├── topological.py     # TopologicalExecutionAlgorithm
│   ├── parallel.py        # ParallelExecutionAlgorithm
│   ├── retry.py           # RetryTracker, NodeRetryState
│   └── rate_limit.py      # RateLimitConfig, RateLimiter, RateLimitRegistry
└── utils/                 # Helpers
    ├── immutable.py       # Frozen Pydantic base models
    └── semver.py          # Semantic version comparison
```

## Design Decisions

### Immutability

All core objects (`Node`, `Workflow`, `Edge`, `Data`, `Value`) are frozen Pydantic models. This ensures:
- Thread safety during parallel execution
- Predictable behavior (no hidden mutation)
- Clean serialization/deserialization

To "modify" an object, use `model_copy(update={...})`.

### Async-First

All execution-related operations are async:
- `Node.run()` - Node computation
- `Value.cast_to()` - Type conversion between values
- `Context` lifecycle hooks
- `ExecutionAlgorithm.execute()`

This enables non-blocking I/O operations (API calls, file operations) within nodes without blocking the event loop.

### Type-Safe Data Flow

Edges connect specific output fields of one node to input fields of another. At workflow construction time, the engine validates:
- Referenced nodes and fields exist
- The graph is acyclic (DAG)
- Type compatibility between connected fields (with automatic casting where possible)

### Node Registration

Nodes auto-register via `__init_subclass__` when they define a `type` field with a `Literal` type annotation. This enables polymorphic deserialization: a `Workflow` JSON containing `{"type": "Add", ...}` automatically deserializes to an `AddNode` instance.

### Discriminated Unions

Node serialization uses Pydantic's discriminated union pattern. Each node class has a `type: Literal["NodeName"]` field that acts as the discriminator, enabling efficient deserialization without trying every possible node type.

## Execution Flow

1. **Load**: Parse workflow JSON into a `Workflow` object (validates structure, detects cycles, checks types)
2. **Prepare**: Create a `Context` and `ExecutionAlgorithm`
3. **Execute**: The algorithm processes nodes according to its strategy:
   - Resolves input edges (collects outputs from upstream nodes)
   - Casts values to match expected input types
   - Calls `node.run(context, input)`
   - Stores outputs for downstream nodes
   - Handles node expansion (composite nodes like `ForEach` replace themselves with sub-workflows)
4. **Collect**: Resolves output edges to build the final result
5. **Return**: Returns `(WorkflowErrors, DataMapping)` tuple

## Error Propagation

Errors during execution are collected per-node in a `WorkflowErrors` object. The behavior depends on the execution algorithm:
- **Topological**: Stops on first error
- **Parallel (FAIL_FAST)**: Cancels remaining tasks on first error
- **Parallel (CONTINUE)**: Runs all possible nodes, collects all errors

Nodes downstream of a failed node are skipped. The engine returns both the errors and any partial output that was successfully computed.
