"""Tests for automatic filtering of invalid edges after node migrations."""

import logging
from typing import ClassVar, Literal, Type

import pytest

from workflow_engine.core import (
    Context,
    Empty,
    Migration,
    Node,
    NodeTypeInfo,
    Params,
    StringValue,
    Workflow,
    load_workflow_with_migration,
    migration_registry,
)


class _NodeParams(Params):
    value: int


class _NodeInput(Empty):
    pass


class _NodeOutput_V1(Params):
    result: StringValue


class _NodeOutput_V2(Params):
    output: StringValue  # Field renamed from 'result' to 'output'


class _TestNodeForEdgeFiltering(Node[_NodeInput, _NodeOutput_V2, _NodeParams]):
    """Test node version 2.0.0 with 'output' output field (was 'result' in v1.0.0)."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="TestNodeForEdgeFiltering",
        display_name="Test Node For Edge Filtering",
        description="A test node for migration testing",
        version="2.0.0",
        parameter_type=_NodeParams,
    )
    type: Literal["TestNodeForEdgeFiltering"] = "TestNodeForEdgeFiltering"

    @property
    def input_type(self) -> Type[_NodeInput]:
        return _NodeInput

    @property
    def output_type(self) -> Type[_NodeOutput_V2]:
        return _NodeOutput_V2

    async def run(self, context: Context, input: _NodeInput) -> _NodeOutput_V2:
        return _NodeOutput_V2(output=StringValue(str(self.params.value)))


class TestNodeMigration_1_to_2(Migration):
    """Migration that doesn't change params, just the output field name."""

    node_type = "TestNodeForEdgeFiltering"
    from_version = "1.0.0"
    to_version = "2.0.0"

    def migrate(self, data: dict) -> dict:
        # No changes needed to params, just version bump
        return dict(data)


@pytest.mark.unit
class TestWorkflowEdgeFiltering:
    """Test that invalid edges are automatically removed during workflow loading."""

    def test_removes_edge_with_invalid_source_field(self, caplog):
        """Test that an edge referencing a non-existent source field is removed."""
        # Register migration
        migration_registry.register(TestNodeMigration_1_to_2)

        # Create workflow data with old schema (v1.0.0) that has 'result' field
        # but after migration to v2.0.0, the field will be 'output'
        workflow_data = {
            "nodes": [
                {
                    "type": "TestNodeForEdgeFiltering",
                    "id": "node1",
                    "version": "1.0.0",
                    "params": {"value": 42},
                },
                {
                    "type": "TestNodeForEdgeFiltering",
                    "id": "node2",
                    "version": "1.0.0",
                    "params": {"value": 100},
                },
            ],
            "edges": [
                {
                    "source_id": "node1",
                    "source_key": "result",  # This field no longer exists after migration
                    "target_id": "node2",
                    "target_key": "input_that_doesnt_exist",
                }
            ],
            "input_edges": [],
            "output_edges": [],
        }

        # Load workflow with migration support
        with caplog.at_level(logging.WARNING):
            workflow = load_workflow_with_migration(workflow_data)

        # Verify the edge was removed
        assert len(workflow.edges) == 0

        # Verify warning was logged
        assert any("Removing invalid edge" in record.message for record in caplog.records)

        # Verify nodes were migrated successfully
        assert len(workflow.nodes) == 2
        assert all(node.version == "2.0.0" for node in workflow.nodes)

    def test_removes_output_edge_with_invalid_source_field(self, caplog):
        """Test that an output edge referencing a non-existent source field is removed."""
        # Register migration
        migration_registry.register(TestNodeMigration_1_to_2)

        workflow_data = {
            "nodes": [
                {
                    "type": "TestNodeForEdgeFiltering",
                    "id": "node1",
                    "version": "1.0.0",
                    "params": {"value": 42},
                }
            ],
            "edges": [],
            "input_edges": [],
            "output_edges": [
                {
                    "source_id": "node1",
                    "source_key": "result",  # This field no longer exists after migration
                    "output_key": "final_output",
                }
            ],
        }

        # Load workflow with migration support
        with caplog.at_level(logging.WARNING):
            workflow = load_workflow_with_migration(workflow_data)

        # Verify the output edge was removed
        assert len(workflow.output_edges) == 0

        # Verify warning was logged
        assert any(
            "Removing output edge" in record.message and "field does not exist" in record.message
            for record in caplog.records
        )

    def test_keeps_valid_edges_after_migration(self):
        """Test that valid edges are kept after node migration."""
        # Register migration
        migration_registry.register(TestNodeMigration_1_to_2)

        workflow_data = {
            "nodes": [
                {
                    "type": "TestNodeForEdgeFiltering",
                    "id": "node1",
                    "version": "1.0.0",
                    "params": {"value": 42},
                }
            ],
            "edges": [],
            "input_edges": [],
            "output_edges": [
                {
                    "source_id": "node1",
                    "source_key": "output",  # This field exists in v2.0.0
                    "output_key": "final_output",
                }
            ],
        }

        # Load workflow with migration support
        workflow = load_workflow_with_migration(workflow_data)

        # Verify the output edge was kept
        assert len(workflow.output_edges) == 1
        assert workflow.output_edges[0].source_key == "output"

    def test_removes_edge_to_nonexistent_node(self, caplog):
        """Test that edges to non-existent nodes are removed during migration."""
        # Register migration
        migration_registry.register(TestNodeMigration_1_to_2)

        workflow_data = {
            "nodes": [
                {
                    "type": "TestNodeForEdgeFiltering",
                    "id": "node1",
                    "version": "1.0.0",  # Old version to trigger migration
                    "params": {"value": 42},
                }
            ],
            "edges": [
                {
                    "source_id": "node1",
                    "source_key": "output",
                    "target_id": "node_that_doesnt_exist",
                    "target_key": "some_input",
                }
            ],
            "input_edges": [],
            "output_edges": [],
        }

        # Load workflow with migration support
        with caplog.at_level(logging.WARNING):
            workflow = load_workflow_with_migration(workflow_data)

        # Verify the edge was removed
        assert len(workflow.edges) == 0

        # Verify warning was logged
        assert any("node not found" in record.message for record in caplog.records)

    def test_workflow_deserializes_with_mixed_valid_and_invalid_edges(self):
        """Test that workflows with a mix of valid and invalid edges can be deserialized."""
        # Register migration
        migration_registry.register(TestNodeMigration_1_to_2)

        # Create workflow with both valid and invalid edges
        workflow_data = {
            "nodes": [
                {
                    "type": "TestNodeForEdgeFiltering",
                    "id": "node1",
                    "version": "1.0.0",
                    "params": {"value": 42},
                }
            ],
            "edges": [
                {
                    "source_id": "node1",
                    "source_key": "result",  # Invalid field (removed after migration)
                    "target_id": "node1",
                    "target_key": "some_input",
                }
            ],
            "input_edges": [],
            "output_edges": [
                {
                    "source_id": "node1",
                    "source_key": "output",  # Valid field
                    "output_key": "result",
                }
            ],
        }

        # Load workflow with migration support
        workflow = load_workflow_with_migration(workflow_data)

        # Verify deserialization results
        assert len(workflow.nodes) == 1
        assert len(workflow.edges) == 0  # Invalid edge removed
        assert len(workflow.output_edges) == 1  # Valid output edge kept

        # Node was migrated successfully
        node = workflow.nodes[0]
        assert isinstance(node, _TestNodeForEdgeFiltering)
        assert node.version == "2.0.0"

        # Note: Workflow may not execute successfully (e.g., if required inputs are missing),
        # but at least it can be loaded and inspected

    def test_strict_validation_without_migration(self):
        """Test that strict validation applies when no migrations occur."""
        # Create workflow with current version (no migration needed) and invalid edge
        workflow_data = {
            "nodes": [
                {
                    "type": "TestNodeForEdgeFiltering",
                    "id": "node1",
                    "version": "2.0.0",  # Current version, no migration
                    "params": {"value": 42},
                },
                {
                    "type": "TestNodeForEdgeFiltering",
                    "id": "node2",
                    "version": "2.0.0",
                    "params": {"value": 100},
                },
            ],
            "edges": [
                {
                    "source_id": "node1",
                    "source_key": "nonexistent_field",  # Invalid field
                    "target_id": "node2",
                    "target_key": "some_input",
                }
            ],
            "input_edges": [],
            "output_edges": [],
        }

        # Should raise ValueError because no migration occurred, so strict validation applies
        with pytest.raises(ValueError, match="does not have.*field"):
            Workflow.model_validate(workflow_data)
