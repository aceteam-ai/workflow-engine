# tests/test_node_migration.py
"""Integration tests for node migration during deserialization."""

import warnings
from typing import Any, ClassVar, Literal, Type

import pytest

from workflow_engine import StringValue
from workflow_engine.core import Empty, Node, NodeTypeInfo, Params
from workflow_engine.core.migration import Migration, migration_registry
from workflow_engine.core.values import Data


# Test fixtures: a node type with migrations


class MigratableParams(Params):
    """Parameters for migratable test node."""

    value: StringValue


class MigratableOutput(Data):
    """Output for migratable test node."""

    result: StringValue


class MigratableNode(Node[Empty, MigratableOutput, MigratableParams]):
    """A node type used for testing migrations."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="MigratableNode",
        display_name="Migratable Node",
        description="A test node for migration testing",
        version="2.0.0",  # Current version
        parameter_type=MigratableParams,
    )

    type: Literal["MigratableNode"] = "MigratableNode"

    @property
    def input_type(self) -> Type[Empty]:
        return Empty

    @property
    def output_type(self) -> Type[MigratableOutput]:
        return MigratableOutput

    async def run(self, context: Any, input: Empty) -> MigratableOutput:
        return MigratableOutput(result=self.params.value)


class MigratableNodeMigration_1_0_0_to_2_0_0(Migration):
    """Migration from 1.0.0 to 2.0.0 for MigratableNode."""

    node_type = "MigratableNode"
    from_version = "1.0.0"
    to_version = "2.0.0"

    def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        params = dict(result.get("params", {}))
        # In v1, the field was called 'text', in v2 it's 'value'
        if "text" in params:
            params["value"] = params.pop("text")
        result["params"] = params
        return result


@pytest.fixture(autouse=True)
def clean_migration_registry():
    """Clean the migration registry before and after each test."""
    # Store original migrations
    original_migrations = dict(migration_registry._migrations)

    yield

    # Restore original state
    migration_registry._migrations = original_migrations


class TestNodeMigration:
    """Integration tests for automatic node migration."""

    @pytest.mark.unit
    def test_load_current_version_no_migration(self):
        """Test that loading current version doesn't trigger migration."""
        data = {
            "type": "MigratableNode",
            "id": "test",
            "version": "2.0.0",
            "params": {"value": "hello"},
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            node = Node.model_validate(data)

            # Should not warn
            assert len(w) == 0

        assert isinstance(node, MigratableNode)
        assert node.version == "2.0.0"
        assert node.params.value.root == "hello"

    @pytest.mark.unit
    def test_load_old_version_with_migration(self):
        """Test that old version is automatically migrated."""
        # Register the migration
        migration_registry.register(MigratableNodeMigration_1_0_0_to_2_0_0)

        data = {
            "type": "MigratableNode",
            "id": "test",
            "version": "1.0.0",
            "params": {"text": "hello"},  # Old field name
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            node = Node.model_validate(data)

            # Should not warn since migration was successful
            assert len(w) == 0

        assert isinstance(node, MigratableNode)
        assert node.version == "2.0.0"  # Migrated to current version
        assert node.params.value.root == "hello"  # Field renamed

    @pytest.mark.unit
    def test_load_old_version_without_migration_warns(self):
        """Test that old version without migration issues warning."""
        # Don't register any migration

        data = {
            "type": "MigratableNode",
            "id": "test",
            "version": "1.5.0",  # Some old version with no migration
            "params": {"value": "hello"},
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            node = Node.model_validate(data)

            # Should have at least one warning about old version
            version_warnings = [
                warning
                for warning in w
                if "1.5.0 is older than" in str(warning.message)
            ]
            assert len(version_warnings) >= 1
            assert "may need to be migrated" in str(version_warnings[0].message)

        assert isinstance(node, MigratableNode)
        assert node.version == "1.5.0"  # Version unchanged (no migration)

    @pytest.mark.unit
    def test_migration_via_workflow_deserialization(self):
        """Test that migration works when loading workflow from JSON."""
        from workflow_engine import Workflow

        # Register the migration
        migration_registry.register(MigratableNodeMigration_1_0_0_to_2_0_0)

        workflow_data = {
            "nodes": [
                {
                    "type": "MigratableNode",
                    "id": "old_node",
                    "version": "1.0.0",
                    "params": {"text": "old_value"},
                }
            ],
            "edges": [],
            "input_edges": [],
            "output_edges": [
                {
                    "source_id": "old_node",
                    "source_key": "result",
                    "output_key": "output",
                }
            ],
        }

        workflow = Workflow.model_validate(workflow_data)

        # Node should be migrated
        node = workflow.nodes[0]
        assert isinstance(node, MigratableNode)
        assert node.version == "2.0.0"
        assert node.params.value.root == "old_value"

    @pytest.mark.unit
    def test_chained_migration(self):
        """Test that multi-step migrations work."""

        # Define intermediate migration
        class Migration_1_5_0_to_2_0_0(Migration):
            node_type = "MigratableNode"
            from_version = "1.5.0"
            to_version = "2.0.0"

            def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
                result = dict(data)
                params = dict(result.get("params", {}))
                # In 1.5.0, field was 'content', in 2.0.0 it's 'value'
                if "content" in params:
                    params["value"] = params.pop("content")
                result["params"] = params
                return result

        class Migration_1_0_0_to_1_5_0(Migration):
            node_type = "MigratableNode"
            from_version = "1.0.0"
            to_version = "1.5.0"

            def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
                result = dict(data)
                params = dict(result.get("params", {}))
                # In 1.0.0, field was 'text', in 1.5.0 it's 'content'
                if "text" in params:
                    params["content"] = params.pop("text")
                result["params"] = params
                return result

        migration_registry.register(Migration_1_0_0_to_1_5_0)
        migration_registry.register(Migration_1_5_0_to_2_0_0)

        data = {
            "type": "MigratableNode",
            "id": "test",
            "version": "1.0.0",
            "params": {"text": "hello"},
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            node = Node.model_validate(data)
            assert len(w) == 0

        assert node.version == "2.0.0"
        assert node.params.value.root == "hello"

    @pytest.mark.unit
    def test_migration_preserves_extra_fields(self):
        """Test that extra fields like position are preserved during migration."""
        migration_registry.register(MigratableNodeMigration_1_0_0_to_2_0_0)

        data = {
            "type": "MigratableNode",
            "id": "test",
            "version": "1.0.0",
            "params": {"text": "hello"},
            "position": {"x": 100, "y": 200},
        }

        node = Node.model_validate(data)

        # Extra field should be preserved
        dumped = node.model_dump()
        assert dumped["position"] == {"x": 100, "y": 200}

    @pytest.mark.unit
    def test_direct_class_validation_requires_current_schema(self):
        """Test that calling concrete class directly requires current version schema.

        Migration only works when deserializing via Node.model_validate(), not
        when calling the concrete class directly. This is by design - direct
        class validation bypasses the dispatch mechanism where migration occurs.

        Users should use Node.model_validate() for loading workflows with
        potentially old node versions.
        """
        migration_registry.register(MigratableNodeMigration_1_0_0_to_2_0_0)

        # Old version data with old schema
        old_data = {
            "type": "MigratableNode",
            "id": "test",
            "version": "1.0.0",
            "params": {"text": "hello"},  # Old field name
        }

        # Direct validation fails because it doesn't go through migration
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MigratableNode.model_validate(old_data)

        # But Node.model_validate works with migration
        node = Node.model_validate(old_data)
        assert node.version == "2.0.0"
        assert node.params.value.root == "hello"

        # Direct validation works with current schema
        current_data = {
            "type": "MigratableNode",
            "id": "test",
            "version": "2.0.0",
            "params": {"value": "hello"},  # Current field name
        }
        node = MigratableNode.model_validate(current_data)
        assert node.version == "2.0.0"


class TestMigrationEdgeCases:
    """Test edge cases in migration."""

    @pytest.mark.unit
    def test_latest_version_not_migrated(self):
        """Test that 'latest' version is not migrated but resolved."""
        data = {
            "type": "MigratableNode",
            "id": "test",
            "version": "latest",
            "params": {"value": "hello"},
        }

        node = Node.model_validate(data)

        # Should resolve to current version
        assert node.version == "2.0.0"

    @pytest.mark.unit
    def test_newer_version_raises_error(self):
        """Test that newer version than current raises error."""
        from pydantic import ValidationError

        data = {
            "type": "MigratableNode",
            "id": "test",
            "version": "3.0.0",  # Newer than current 2.0.0
            "params": {"value": "hello"},
        }

        with pytest.raises(ValidationError) as exc_info:
            Node.model_validate(data)

        assert "newer than the latest version" in str(exc_info.value)
