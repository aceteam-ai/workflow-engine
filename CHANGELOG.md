# Changelog

All notable changes to this project will be documented in this file.

This project uses [PEP 440](https://peps.python.org/pep-0440/) versioning with release candidates (rcN) for pre-release versions.

## [2.0.0rc3] - 2026-02-23

### Added
- `ShouldYield` / `WorkflowYield` control-flow signals for nodes that dispatch external work and cannot return a value in the current execution (#77)
- `on_node_yield(node, input, exception)` context hook, called once per yielded node before `WorkflowYield` is raised (#77)
- `on_node_expand` context hook called when a node's `run()` returns a `Workflow` instead of a `DataMapping` (#74)
- Constrained Value subclasses: `build_value_cls()` on primitive schema classes now enforces JSON Schema keywords (`minimum`, `maxLength`, `minItems`, etc.) at runtime via Pydantic `Field` constraints (#76)
- `x-value-type` JSON Schema extension field for stable Value type detection; `title` retained as a backwards-compatible fallback (#76)

### Fixed
- `$defs` key collisions when multiple constrained variants of the same base Value type appear in a single `DataValue` schema (#76)
- Bug in `ParallelExecutionAlgorithm` where yielded nodes were missing from the ready-node exclusion filter, which would have caused an infinite re-dispatch loop (#77)

## [2.0.0rc2] - 2026-02-13

### Added
- Adaptive `ForEachNode` wiring based on inner workflow I/O (#72)
- `FieldSchemaMappingValue` convenience data type for string-field schema mappings (#71)
- First-class support for CSV, DocX, and Markdown file types (#69)
- Iterable APIs and typed `extend()` method for registries (#64)
- Comprehensive tests for round-trip type (de)serialization and failure cases (#73)

### Changed
- Schemas now allow extra fields (#70)
- Updated `cryptography` dependency from 46.0.3 to 46.0.5

## [2.0.0rc1] - 2025-02-05

### Added
- `WorkflowEngine` class with node and workflow deserialization capabilities
- Execution capabilities to `WorkflowEngine`

### Changed
- **BREAKING**: Migrated workflow format from `InputEdge`/`OutputEdge` to `InputNode`/`OutputNode`
- Moved node/edge validation from `Workflow` to `WorkflowEngine`
- Updated examples and documentation for new `InputNode`/`OutputNode` format

## [1.2.0rc1] - 2025-02-01

### Added
- Generic `ModelValue[M]` type for arbitrary Pydantic models
- `ExtractionResultValue` type
- Comprehensive tests for `NodeRegistry` implementations
- Coding best practices to CLAUDE.md about explicit methods vs dunder overrides

### Changed
- Refactored `ValueRegistry` with builder pattern and comprehensive tests
- Updated codebase to use new `ValueRegistry` API

### Fixed
- Pytest warning by renaming `TestValue` to `ExampleValue`

## [1.1.0rc1] - 2025-01-30

### Added
- Workflow migration utilities for safely loading and migrating workflows (`load_workflow_with_migration`, `clean_edges_after_migration`)
- Typecasts from `JSONValue` to primitive types (`NullValue`, `BooleanValue`, `IntegerValue`, `FloatValue`)
- Optional `input_type` and `output_type` fields on `Workflow` for explicit type declarations
- Logging for previously unlogged exception handlers
- Project documentation: getting started guide, architecture overview, node reference, value types, execution, contexts

### Fixed
- `NodeExpansionException` property setter bug
- Unlogged exceptions in execution error handlers now produce proper log output

### Changed
- Version bump from `1.0.0rc2` to `1.1.0rc1`
- Documentation URL updated to point to GitHub docs

## [1.0.0rc2] - 2025-01-15

### Changed
- Converted optional dependencies (`pyright`, `pytest`, `ruff`, etc.) into dev dependencies
- Updated to Python 3.12+ minimum
- Made type checking non-blocking in CI
- Improved release script

## [1.0.0rc1] - 2025-01-10

### Added
- PyPI publishing via GitHub Actions
- Node retry system with exponential backoff (`ShouldRetry`, `RetryTracker`)
- Rate limiting per node type (`RateLimitConfig`, `RateLimiter`, `RateLimitRegistry`)
- Node versioning and migration system (`Migration`, `MigrationRegistry`, `MigrationRunner`)
- Parallel node execution algorithm (`ParallelExecutionAlgorithm`) with eager dispatch
- Value typecasting graph visualization in README

### Changed
- Registration logs changed from INFO to DEBUG level
- Migrated from Poetry to uv for dependency management

## [0.3.3] - 2024-12-20

### Added
- Release script for streamlined publishing

## [0.3.2] - 2024-12-15

### Added
- JSON type support
- CLAUDE.md for Claude Code guidance

### Fixed
- Double-typecast issue with Files to JSONFile

## [0.3.1] - 2024-12-10

### Added
- MIME type support for file values
- Iteration and property accessors on collection types to reduce `.root` usage
- `__getitem__` access on `SequenceValue` and `StringMapValue`
- Direct equality comparison with Values

### Fixed
- MIME type inconsistencies
- Type inconsistencies in file handling
- Missing exports

## [0.3.0] - 2024-12-01

### Added
- `ForEachNode` for workflow iteration over sequences
- `IfNode` and `IfElseNode` for conditional branching
- Composite node system (`ExpandSequenceNode`, `GatherSequenceNode`, `ExpandMappingNode`, `GatherMappingNode`, `ExpandDataNode`, `GatherDataNode`)
- Serializable metadata for node types
- Input/output schemas on nodes
- Support for type unions in JSON schemas

### Changed
- Require semantic versioning in node type definitions

## [0.2.0] - 2024-11-15

### Added
- Asynchronous workflow execution
- Value type casting system (async)
- Error handling with partial results (`WorkflowErrors`)

### Changed
- All node I/O now uses Value types
- Workflows are fully asynchronous

## [0.1.x] - 2024-10-01

### Added
- Initial workflow engine implementation
- Core types: `Node`, `Workflow`, `Edge`, `Context`, `Value`, `Data`
- Storage backends: `SupabaseContext`, `LocalContext`, `InMemoryContext`
- Basic arithmetic, constant, text, and error nodes
- Topological execution algorithm

[2.0.0rc2]: https://github.com/aceteam-ai/workflow-engine/compare/v2.0.0rc1...v2.0.0rc2
[2.0.0rc1]: https://github.com/aceteam-ai/workflow-engine/compare/v1.2.0rc1...v2.0.0rc1
[1.2.0rc1]: https://github.com/aceteam-ai/workflow-engine/compare/v1.1.0rc1...v1.2.0rc1
[1.1.0rc1]: https://github.com/aceteam-ai/workflow-engine/compare/v1.0.0rc2...v1.1.0rc1
[1.0.0rc2]: https://github.com/aceteam-ai/workflow-engine/compare/v1.0.0rc1...v1.0.0rc2
[1.0.0rc1]: https://github.com/aceteam-ai/workflow-engine/compare/v0.3.3...v1.0.0rc1
[0.3.3]: https://github.com/aceteam-ai/workflow-engine/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/aceteam-ai/workflow-engine/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/aceteam-ai/workflow-engine/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/aceteam-ai/workflow-engine/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/aceteam-ai/workflow-engine/compare/v0.1.0...v0.2.0
