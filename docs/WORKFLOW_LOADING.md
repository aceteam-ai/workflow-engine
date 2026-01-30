# Loading Workflows with Migration Support

This guide explains how to load workflows from JSON/database when migrations may be needed.

## Quick Start

```python
from workflow_engine.core import load_workflow_with_migration
import json

# Load workflow from JSON file
with open("workflow.json") as f:
    workflow_data = json.load(f)

# Load with automatic migration support
workflow = load_workflow_with_migration(workflow_data)
```

That's it! The function handles:
- ✅ Node migrations (version upgrades)
- ✅ Edge cleanup (removes edges broken by field renames)
- ✅ Validation (ensures workflow is valid)

## When to Use Which Method

### Use `load_workflow_with_migration()` when:

- Loading workflows from files or database
- Workflows may have old node versions
- You want automatic edge cleanup
- You want graceful handling of migration issues

```python
from workflow_engine.core import load_workflow_with_migration

workflow = load_workflow_with_migration(workflow_data)
```

### Use `Workflow.model_validate()` when:

- Creating workflows programmatically
- Loading new workflows (current versions)
- You want strict validation (fail on any invalid edge)

```python
from workflow_engine.core import Workflow

workflow = Workflow.model_validate(workflow_data)
```

## How It Works

### Step-by-Step Process

When you call `load_workflow_with_migration(workflow_data)`:

1. **Parse nodes** - Each node is loaded via `Node.model_validate()`
2. **Check for migrations** - Compare node versions in data vs current
3. **Migrate nodes** - If version mismatch, run migrations
4. **Detect edge issues** - Check if any edges reference invalid fields
5. **Filter edges** - Remove edges broken by migrations (with warnings)
6. **Validate workflow** - Ensure result is valid
7. **Return workflow** - Ready to execute

### What Gets Migrated

**Nodes:**
- Version field updated to current version
- Params transformed according to migration rules
- All migrations in the chain are applied

**Edges:**
- Invalid edges are removed (if migrations occurred)
- Valid edges are preserved
- Warnings logged for each removed edge

### What Happens to Invalid Edges

If node migrations rename/remove fields, edges may become invalid:

```python
# Before migration: Node v1.0.0 has output field "result"
workflow_data = {
    "nodes": [{
        "type": "MyNode",
        "id": "node1",
        "version": "1.0.0",
        "params": {...}
    }],
    "edges": [],
    "output_edges": [{
        "source_id": "node1",
        "source_key": "result",  # ← This field gets renamed to "output" in v2.0.0
        "output_key": "final_result"
    }]
}

# Load with migration
workflow = load_workflow_with_migration(workflow_data)

# Result:
# - Node migrated to v2.0.0
# - Output edge removed (field "result" no longer exists)
# - Warning logged: "Removing output edge 'final_result' from node1.result: field does not exist"
```

## Edge Filtering Details

### When Edges Are Removed

Edges are removed when:

1. **Missing field reference**
   ```
   Edge references "old_field" but node now has "new_field"
   ```

2. **Type incompatibility**
   ```
   Edge connects StringValue output to IntegerValue input
   ```

3. **Missing node**
   ```
   Edge references node ID that doesn't exist
   ```

### Only After Migrations

Edge filtering **only** happens when migrations occur. If all nodes are already at current versions, strict validation applies:

```python
# All nodes at v2.0.0 - no migrations needed
workflow_data = {
    "nodes": [{
        "type": "MyNode",
        "version": "2.0.0",  # Current version
        "params": {...}
    }],
    "output_edges": [{
        "source_key": "invalid_field"  # ← This will FAIL validation
    }]
}

# This raises ValueError - strict mode
workflow = load_workflow_with_migration(workflow_data)
```

### Logging

All edge removals are logged:

```python
import logging

# Enable logging to see removed edges
logging.basicConfig(level=logging.WARNING)

workflow = load_workflow_with_migration(workflow_data)

# Output:
# WARNING: Node node1 migrated from 1.0.0 to 2.0.0
# INFO: Node migrations detected, filtering invalid edges
# WARNING: Removing output edge 'result' from node1.old_field: field does not exist
```

## Advanced Usage

### Custom Edge Filtering

If you want more control over edge filtering:

```python
from workflow_engine.core import clean_edges_after_migration, Workflow

# Clean edges manually
cleaned_data = clean_edges_after_migration(workflow_data)

# Inspect what was cleaned
print(f"Original edges: {len(workflow_data['edges'])}")
print(f"Cleaned edges: {len(cleaned_data['edges'])}")

# Create workflow
workflow = Workflow.model_validate(cleaned_data)
```

### Skip Edge Filtering

If you don't want edge filtering even with migrations:

```python
from workflow_engine.core import Workflow

# This will fail if edges are invalid, even with migrations
workflow = Workflow.model_validate(workflow_data)
```

### Programmatic Workflow Creation

When creating workflows in code, you don't need migration support:

```python
from workflow_engine.core import Workflow
from workflow_engine.nodes import ConstantStringNode

# Create nodes programmatically
node = ConstantStringNode(
    id="greeting",
    params={"value": StringValue("Hello")}
    # version automatically set to current
)

# Create workflow directly
workflow = Workflow(
    nodes=[node],
    edges=[],
    input_edges=[],
    output_edges=[]
)
```

## Error Handling

### Migration Errors

If migration fails, you'll get a descriptive error:

```python
from workflow_engine.core import MigrationNotFoundError, MigrationValidationError

try:
    workflow = load_workflow_with_migration(workflow_data)
except MigrationNotFoundError as e:
    print(f"No migration path available: {e}")
    # Workflow has old version but no migration defined
    # Action: Create migration or manually update data

except MigrationValidationError as e:
    print(f"Migration validation failed: {e}")
    # Data doesn't meet migration requirements
    # Action: Fix data or update migration

except ValidationError as e:
    print(f"Workflow validation failed: {e}")
    # Workflow structure is invalid
    # Action: Fix workflow structure
```

### Graceful Degradation

If no migration path exists, the system tries to load anyway:

```python
# Node v1.0.0 exists, but no migration to v2.0.0
workflow_data = {
    "nodes": [{"type": "MyNode", "version": "1.0.0", ...}]
}

# Loads with warning, no edge filtering
workflow = load_workflow_with_migration(workflow_data)
# WARNING: Node version 1.0.0 is older than latest version (2.0.0), and may need to be migrated

# Node still at v1.0.0, may fail during execution
```

## Examples

### Example 1: Load from Database

```python
import json
from workflow_engine.core import load_workflow_with_migration
from your_db import get_workflow

# Get workflow from database
db_workflow = get_workflow(workflow_id)
workflow_data = json.loads(db_workflow.graph)

# Load with migration support
workflow = load_workflow_with_migration(workflow_data)

# Execute
from workflow_engine.execution import TopologicalExecutionAlgorithm
algorithm = TopologicalExecutionAlgorithm()
errors, output = await algorithm.execute(context, workflow, input_data)
```

### Example 2: Import from File

```python
import json
from workflow_engine.core import load_workflow_with_migration

# Import workflow from JSON file
with open("exported_workflow.json") as f:
    workflow_data = json.load(f)

# Load with automatic migration
workflow = load_workflow_with_migration(workflow_data)

# Save to database with current versions
updated_data = workflow.model_dump()
save_to_database(updated_data)
```

### Example 3: Batch Migration

```python
from workflow_engine.core import load_workflow_with_migration
import json

# Migrate multiple workflows
workflows_to_migrate = get_all_old_workflows()

for wf_data in workflows_to_migrate:
    try:
        # Load and migrate
        workflow = load_workflow_with_migration(wf_data)

        # Save back with current versions
        updated_data = workflow.model_dump()
        save_workflow(updated_data)

        print(f"✓ Migrated workflow {wf_data['id']}")

    except Exception as e:
        print(f"✗ Failed to migrate workflow {wf_data['id']}: {e}")
```

## Best Practices

### ✅ DO

1. **Use migration support for user data**
   ```python
   # When loading from storage
   workflow = load_workflow_with_migration(workflow_data)
   ```

2. **Log migration activities**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   # Now you'll see migration logs
   ```

3. **Test migrations before deploying**
   ```python
   # In tests
   old_workflow = load_test_data("v1_workflow.json")
   new_workflow = load_workflow_with_migration(old_workflow)
   assert_workflow_valid(new_workflow)
   ```

4. **Save workflows after migration**
   ```python
   # Update stored version
   migrated = load_workflow_with_migration(old_data)
   save_to_database(migrated.model_dump())
   ```

### ❌ DON'T

1. **Don't use migration for new workflows**
   ```python
   # ❌ Unnecessary overhead
   new_workflow = load_workflow_with_migration(just_created_data)

   # ✅ Use direct validation
   new_workflow = Workflow.model_validate(just_created_data)
   ```

2. **Don't ignore migration warnings**
   ```python
   # ❌ Silently discarding edges may break workflows
   import warnings
   warnings.filterwarnings("ignore")

   # ✅ Review warnings and update workflows
   logging.basicConfig(level=logging.WARNING)
   ```

3. **Don't assume migrations preserve all edges**
   ```python
   # ❌ May fail - edges might be removed
   workflow = load_workflow_with_migration(old_data)
   assert len(workflow.edges) == len(old_data['edges'])

   # ✅ Check and handle missing edges
   if len(workflow.edges) < len(old_data['edges']):
       log_error("Some edges were removed during migration")
   ```

## See Also

- [Writing Migrations](./MIGRATIONS.md) - How to create migrations
- [Node Versioning](./VERSIONING.md) - Semantic versioning for nodes
- API Reference: `workflow_engine.core.migration`
