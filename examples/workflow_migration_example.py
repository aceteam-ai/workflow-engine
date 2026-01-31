#!/usr/bin/env python3
"""
Example: Loading workflows with migration support

This example demonstrates how to load workflows that may have been created
with older node versions, handling migrations and edge cleanup automatically.
"""

import json
from workflow_engine.core import Workflow, load_workflow_with_migration

# Example 1: Using load_workflow_with_migration (recommended)
# This handles migrations and cleans up invalid edges automatically
workflow_data = {
    "nodes": [
        {
            "type": "ConstantString",
            "id": "greeting",
            "version": "1.0.0",  # May be migrated to newer version
            "params": {"text": "Hello, World!"},
        }
    ],
    "edges": [],
    "input_edges": [],
    "output_edges": [
        {
            "source_id": "greeting",
            "source_key": "result",
            "output_key": "message",
        }
    ],
}

# Load with migration support - handles node version upgrades
# and removes edges that became invalid due to field renames
workflow = load_workflow_with_migration(workflow_data)
print(f"✓ Workflow loaded with {len(workflow.nodes)} node(s)")
print("  Nodes migrated if needed, invalid edges removed")

# Example 2: Loading from JSON file
json_data = """
{
  "nodes": [
    {
      "type": "ConstantString",
      "id": "node1",
      "version": "1.0.0",
      "params": {"text": "test"}
    }
  ],
  "edges": [],
  "input_edges": [],
  "output_edges": []
}
"""

workflow_data = json.loads(json_data)
workflow = load_workflow_with_migration(workflow_data)
print("\n✓ Workflow loaded from JSON")

# Example 3: Strict validation (no migration support)
# If you want strict validation without migration tolerance:
current_version_data = {
    "nodes": [
        {
            "type": "ConstantString",
            "id": "node1",
            "version": "1.0.0",  # Must exactly match expectations
            "params": {"text": "test"},
        }
    ],
    "edges": [],
    "input_edges": [],
    "output_edges": [],
}

try:
    # This will fail if edges are invalid (no migration tolerance)
    workflow_strict = Workflow.model_validate(current_version_data)
    print("\n✓ Strict validation passed")
except ValueError as e:
    print(f"\n✗ Strict validation failed: {e}")

print(
    """
Summary:
- Use load_workflow_with_migration() when loading workflows from disk/database
- Nodes automatically migrate to current versions
- Invalid edges (broken by migrations) are removed with warnings
- For new workflows created programmatically, use Workflow() directly
"""
)
