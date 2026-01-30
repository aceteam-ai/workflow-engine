# Writing Node Migrations

This guide explains how to write migrations for nodes when you need to change their schema between versions.

## Table of Contents

- [When to Write a Migration](#when-to-write-a-migration)
- [Migration Basics](#migration-basics)
- [Writing Your First Migration](#writing-your-first-migration)
- [Testing Migrations](#testing-migrations)
- [Common Patterns](#common-patterns)
- [Best Practices](#best-practices)
- [Advanced Topics](#advanced-topics)

## When to Write a Migration

Write a migration when you make **breaking changes** to a node's schema:

### ✅ Requires Migration
- **Renaming fields**: `old_name` → `new_name`
- **Changing field types**: `str` → `int`, `required` → `optional`
- **Removing required fields**: Provide defaults for old data
- **Restructuring params**: Flattening nested objects, etc.
- **Changing semantics**: Same field name, different meaning

### ❌ No Migration Needed
- **Adding optional fields with defaults**: Backward compatible
- **Bug fixes**: No schema change
- **Internal implementation changes**: Schema stays the same
- **Documentation updates**: No data changes

## Migration Basics

### Anatomy of a Migration

```python
from collections.abc import Mapping
from typing import Any
from workflow_engine.core import Migration, migration_registry

@migration_registry.register
class MyNodeMigration_1_0_0_to_2_0_0(Migration):
    """Brief description of what this migration does."""

    node_type = "MyNode"           # Must match node's type field
    from_version = "1.0.0"         # Source version (semantic versioning)
    to_version = "2.0.0"           # Target version

    def migrate(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Transform node data from v1.0.0 to v2.0.0.

        Args:
            data: Raw node data (read-only Mapping) with keys: type, id, version, params

        Returns:
            Transformed node data as Mapping (typically a dict).
            Version field will be updated by runner.
        """
        # Create mutable copy for transformation
        result = dict(data)
        params = dict(result.get("params", {}))

        # Your transformation logic here
        params["new_field"] = params.pop("old_field", "default_value")

        result["params"] = params
        return result
```

### Key Points

1. **Naming convention**: `{NodeType}Migration_{from}_{to}` (underscores replace dots)
2. **Decorator**: Always use `@migration_registry.register`
3. **Immutability**: Never modify input dict - create copies
4. **Version field**: Don't modify it - the runner handles this
5. **Registration**: Import migration module to register it

## Writing Your First Migration

### Step 1: Update Node Version

First, increment your node's version:

```python
class MyNode(Node[MyInput, MyOutput, MyParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="MyNode",
        display_name="My Node",
        description="...",
        version="2.0.0",  # ← Increment from 1.0.0
        parameter_type=MyParams,
    )
```

### Step 2: Create Migration File

Create a new file in your project (e.g., `migrations/my_node.py`):

```python
# migrations/my_node.py
from typing import Any
from workflow_engine.core import Migration, migration_registry


@migration_registry.register
class MyNodeMigration_1_0_0_to_2_0_0(Migration):
    """Rename 'input_text' to 'text' for consistency."""

    node_type = "MyNode"
    from_version = "1.0.0"
    to_version = "2.0.0"

    def migrate(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
        result = dict(data)
        params = dict(result.get("params", {}))

        # Rename field
        if "input_text" in params:
            params["text"] = params.pop("input_text")

        result["params"] = params
        return result
```

### Step 3: Register Migration

Ensure your migration module is imported when your package loads:

```python
# your_package/__init__.py
from . import migrations  # This registers all migrations

# OR in your migrations/__init__.py
from .my_node import MyNodeMigration_1_0_0_to_2_0_0
```

### Step 4: Test It

```python
# Test loading old workflow data
from workflow_engine.core import load_workflow_with_migration

old_workflow_data = {
    "nodes": [{
        "type": "MyNode",
        "id": "node1",
        "version": "1.0.0",  # Old version
        "params": {"input_text": "Hello"}  # Old field name
    }],
    "edges": [],
    "input_edges": [],
    "output_edges": []
}

# Load with migration - field gets renamed automatically
workflow = load_workflow_with_migration(old_workflow_data)
print(workflow.nodes[0].params.text)  # "Hello"
```

## Testing Migrations

### Unit Test Example

```python
import pytest
from workflow_engine.core import Workflow, migration_registry
from your_package.migrations import MyNodeMigration_1_0_0_to_2_0_0

@pytest.mark.unit
def test_my_node_migration():
    """Test that input_text is renamed to text."""
    # Register migration
    migration_registry.register(MyNodeMigration_1_0_0_to_2_0_0)

    # Old workflow data
    old_data = {
        "nodes": [{
            "type": "MyNode",
            "id": "node1",
            "version": "1.0.0",
            "params": {"input_text": "Hello", "other_param": 42}
        }],
        "edges": [],
        "input_edges": [],
        "output_edges": []
    }

    # Load workflow - migration runs automatically
    workflow = Workflow.model_validate(old_data)

    # Verify migration worked
    node = workflow.nodes[0]
    assert node.version == "2.0.0"
    assert node.params.text == "Hello"
    assert node.params.other_param == 42

    # Verify old field is gone
    assert not hasattr(node.params, "input_text")
```

### Test Migration Chain

```python
def test_migration_chain():
    """Test migrating through multiple versions."""
    # Register migrations for 1.0.0 → 1.5.0 → 2.0.0
    migration_registry.register(MyNodeMigration_1_0_0_to_1_5_0)
    migration_registry.register(MyNodeMigration_1_5_0_to_2_0_0)

    old_data = {
        "nodes": [{
            "type": "MyNode",
            "id": "node1",
            "version": "1.0.0",
            "params": {...}
        }],
        ...
    }

    workflow = Workflow.model_validate(old_data)
    assert workflow.nodes[0].version == "2.0.0"
```

## Common Patterns

### Pattern 1: Rename Field

```python
def migrate(self, data: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(data)
    params = dict(result.get("params", {}))

    # Rename with fallback
    params["new_name"] = params.pop("old_name", "default")

    result["params"] = params
    return result
```

### Pattern 2: Change Type

```python
def migrate(self, data: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(data)
    params = dict(result.get("params", {}))

    # Convert string to int
    if "port" in params and isinstance(params["port"], str):
        params["port"] = int(params["port"])

    result["params"] = params
    return result
```

### Pattern 3: Split Field

```python
def migrate(self, data: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(data)
    params = dict(result.get("params", {}))

    # Split "full_name" into "first_name" and "last_name"
    if "full_name" in params:
        parts = params.pop("full_name").split(" ", 1)
        params["first_name"] = parts[0]
        params["last_name"] = parts[1] if len(parts) > 1 else ""

    result["params"] = params
    return result
```

### Pattern 4: Restructure Nested Object

```python
def migrate(self, data: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(data)
    params = dict(result.get("params", {}))

    # Flatten nested config
    if "config" in params:
        config = params.pop("config")
        params["host"] = config.get("host", "localhost")
        params["port"] = config.get("port", 8080)

    result["params"] = params
    return result
```

### Pattern 5: Add Required Field with Logic

```python
def migrate(self, data: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(data)
    params = dict(result.get("params", {}))

    # Add new required field based on existing data
    if "timeout" not in params:
        # Derive from existing field
        retry_count = params.get("retry_count", 3)
        params["timeout"] = retry_count * 10  # 10s per retry

    result["params"] = params
    return result
```

### Pattern 6: Remove Field

```python
def migrate(self, data: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(data)
    params = dict(result.get("params", {}))

    # Remove deprecated field
    params.pop("deprecated_field", None)

    result["params"] = params
    return result
```

## Best Practices

### ✅ DO

1. **Write clear migration descriptions**
   ```python
   """
   Rename 'retries' to 'max_retries' to match common terminology.
   Old workflows will have their retry counts preserved.
   """
   ```

2. **Provide sensible defaults**
   ```python
   params["timeout"] = params.pop("timeout", 30)  # 30s default
   ```

3. **Preserve backward compatibility when possible**
   ```python
   # Support both old and new field names temporarily
   if "old_field" in params and "new_field" not in params:
       params["new_field"] = params["old_field"]
   ```

4. **Use validation for complex migrations**
   ```python
   def validate(self, data: Mapping[str, Any]) -> list[str]:
       errors = []
       params = data.get("params", {})

       if "required_field" not in params:
           errors.append("Missing required_field in params")

       return errors
   ```

5. **Test with real workflow data**
   - Export workflows before migration
   - Test migration on copies
   - Verify all nodes migrate correctly

### ❌ DON'T

1. **Don't mutate input or output** (enforced by type system)
   ```python
   # ❌ Bad - won't work, data is Mapping (read-only)
   def migrate(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
       data["params"]["new_field"] = "value"  # Type error!
       return data

   # ✅ Good - creates new dicts (functional style)
   def migrate(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
       result = dict(data)  # Create mutable copy
       params = dict(result.get("params", {}))
       params["new_field"] = "value"
       result["params"] = params
       return result  # Returns dict (which is a Mapping)
   ```

   **Why Mapping for return type?**
   - Discourages mutation of migration results
   - Enforces functional programming style
   - `dict` is a valid `Mapping`, so you can still return dicts
   - Type checker prevents accidental mutations downstream

2. **Don't modify the version field**
   ```python
   # ❌ Bad - runner handles this
   result["version"] = self.to_version

   # ✅ Good - let runner update version
   return result
   ```

3. **Don't skip validation for destructive changes**
   ```python
   # ✅ Good - validate before dropping data
   def validate(self, data: Mapping[str, Any]) -> list[str]:
       if "critical_field" not in data.get("params", {}):
           return ["Cannot migrate: missing critical_field"]
       return []
   ```

4. **Don't use magic values without comments**
   ```python
   # ❌ Bad
   params["priority"] = params.get("level", 5) * 2

   # ✅ Good
   # Convert level (1-5) to priority (1-10 scale)
   level = params.get("level", 3)  # Default medium priority
   params["priority"] = min(level * 2, 10)
   ```

## Advanced Topics

### Multi-Step Migration Chains

The system automatically chains migrations:

```python
# 1.0.0 → 2.0.0 is broken into:
# 1.0.0 → 1.5.0 → 2.0.0

@migration_registry.register
class MyNodeMigration_1_0_0_to_1_5_0(Migration):
    node_type = "MyNode"
    from_version = "1.0.0"
    to_version = "1.5.0"
    # ...

@migration_registry.register
class MyNodeMigration_1_5_0_to_2_0_0(Migration):
    node_type = "MyNode"
    from_version = "1.5.0"
    to_version = "2.0.0"
    # ...
```

The runner uses BFS to find the shortest path.

### Conditional Migrations

Handle different data shapes in the same version:

```python
def migrate(self, data: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(data)
    params = dict(result.get("params", {}))

    # Handle two different v1.0.0 formats
    if "old_style_config" in params:
        config = params.pop("old_style_config")
        params["host"] = config["server"]
        params["port"] = config["port"]
    elif "server_url" in params:
        # Parse URL format
        url = params.pop("server_url")
        host, port = url.split(":")
        params["host"] = host
        params["port"] = int(port)
    else:
        # New format, already has host/port
        pass

    result["params"] = params
    return result
```

### Validation with Helpful Error Messages

```python
def validate(self, data: Mapping[str, Any]) -> list[str]:
    """Validate that data can be safely migrated."""
    errors = []
    params = data.get("params", {})

    # Check required fields exist
    if "input_file" not in params:
        errors.append(
            "Missing 'input_file'. This node was created incorrectly "
            "and cannot be migrated automatically."
        )

    # Check data types
    if "retry_count" in params and not isinstance(params["retry_count"], int):
        errors.append(
            f"Field 'retry_count' must be an integer, got {type(params['retry_count']).__name__}. "
            f"Please fix the data before migrating."
        )

    # Check value ranges
    if "priority" in params:
        priority = params["priority"]
        if not (1 <= priority <= 10):
            errors.append(
                f"Field 'priority' must be between 1-10, got {priority}. "
                f"Cannot determine appropriate new priority level."
            )

    return errors
```

### Migrating Complex Nested Structures

```python
def migrate(self, data: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(data)
    params = dict(result.get("params", {}))

    # Migrate list of objects
    if "connections" in params:
        old_connections = params.pop("connections")
        new_connections = []

        for conn in old_connections:
            new_conn = {
                "name": conn.get("id", "unnamed"),  # Rename field
                "url": conn["host"],  # Rename field
                "timeout": conn.get("timeout", 30),  # Add default
                # Drop deprecated 'protocol' field
            }
            new_connections.append(new_conn)

        params["connections"] = new_connections

    result["params"] = params
    return result
```

## Edge Cleanup After Migration

When migrations rename input/output fields, edges may become invalid. Use the migration utilities:

```python
from workflow_engine.core import load_workflow_with_migration

# This automatically:
# 1. Migrates nodes
# 2. Detects invalid edges (broken by field renames)
# 3. Removes them with warnings
workflow = load_workflow_with_migration(workflow_data)
```

See [Workflow Loading Guide](./WORKFLOW_LOADING.md) for details on edge cleanup.

## Questions?

- Check existing migrations in `workflow_engine/nodes/` for examples
- Read migration tests in `tests/test_migration_*.py`
- See API docs: `workflow_engine.core.migration`

## Quick Reference

```python
from collections.abc import Mapping
from typing import Any
from workflow_engine.core import Migration, migration_registry

@migration_registry.register
class MyMigration(Migration):
    node_type = "NodeType"      # Must match node's type field
    from_version = "X.Y.Z"      # Source version
    to_version = "X.Y.Z"        # Target version

    def migrate(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
        """Transform data. Immutable in/out (functional style)."""
        result = dict(data)  # Create mutable copy for transformation
        params = dict(result.get("params", {}))
        # Transform params here
        result["params"] = params
        return result  # dict is a Mapping

    def validate(self, data: Mapping[str, Any]) -> list[str]:
        """Optional: validate before migrating."""
        return []  # Empty list = valid
```
