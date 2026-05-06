# tests/test_node_migration.py
"""Integration tests for node migration during deserialization."""

import warnings
from collections.abc import Mapping
from typing import Any, ClassVar, Type

import pytest
from overrides import override

from workflow_engine import (
    ExecutionContext,
    NodeRegistry,
    StringValue,
    WorkflowEngine,
)
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
        display_name="Migratable Node",
        description="A test node for migration testing",
        version="2.0.0",  # Current version
        parameter_type=MigratableParams,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> Type[MigratableOutput]:
        return MigratableOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[MigratableOutput],
        input: Empty,
    ) -> MigratableOutput:
        return MigratableOutput(result=self.params.value)


# The registered type name (default_type_name strips the "Node" suffix).
NODE_TYPE = "Migratable"


class MigratableNodeMigration_1_0_0_to_2_0_0(Migration):
    """Migration from 1.0.0 to 2.0.0 for MigratableNode."""

    node_type = NODE_TYPE
    from_version = "1.0.0"
    to_version = "2.0.0"

    def migrate(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
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
    original_migrations = dict(migration_registry._migrations)
    yield
    migration_registry._migrations = original_migrations


class TestNodeMigration:
    """Integration tests for automatic node migration."""

    @pytest.mark.unit
    def test_load_current_version_no_migration(self):
        """Test that loading current version doesn't trigger migration."""
        data = {
            "type": NODE_TYPE,
            "id": "test",
            "version": "2.0.0",
            "params": {"value": "hello"},
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            untyped_node = Node.model_validate(data)
            node = NodeRegistry.DEFAULT.load(untyped_node)

            assert len(w) == 0

        assert isinstance(node, MigratableNode)
        assert node.version == "2.0.0"
        assert node.params.value.root == "hello"

    @pytest.mark.unit
    def test_load_old_version_with_migration(self):
        """Test that old version is automatically migrated."""
        migration_registry.register(MigratableNodeMigration_1_0_0_to_2_0_0)

        data = {
            "type": NODE_TYPE,
            "id": "test",
            "version": "1.0.0",
            "params": {"text": "hello"},  # Old field name
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            untyped_node = Node.model_validate(data)
            node = NodeRegistry.DEFAULT.load(untyped_node)

            assert len(w) == 0

        assert isinstance(node, MigratableNode)
        assert node.version == "2.0.0"
        assert node.params.value.root == "hello"

    @pytest.mark.unit
    def test_load_old_version_without_migration_warns(self):
        """Test that old version without migration issues warning."""
        data = {
            "type": NODE_TYPE,
            "id": "test",
            "version": "1.5.0",
            "params": {"value": "hello"},
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            untyped_node = Node.model_validate(data)
            node = NodeRegistry.DEFAULT.load(untyped_node)

            version_warnings = [
                warning
                for warning in w
                if "1.5.0 is older than" in str(warning.message)
            ]
            assert len(version_warnings) >= 1
            assert "may need to be migrated" in str(version_warnings[0].message)

        assert isinstance(node, MigratableNode)
        assert node.version == "1.5.0"

    async def test_migration_via_workflow_deserialization(self):
        """Test that migration works when loading workflow from JSON."""
        from workflow_engine import Workflow

        migration_registry.register(MigratableNodeMigration_1_0_0_to_2_0_0)

        workflow_data = {
            "input_node": {
                "type": "Input",
                "version": "1.0.0",
                "id": "input",
                "params": {"fields": {}},
            },
            "inner_nodes": [
                {
                    "type": NODE_TYPE,
                    "id": "old_node",
                    "version": "1.0.0",
                    "params": {"text": "old_value"},
                }
            ],
            "output_node": {
                "type": "Output",
                "version": "1.0.0",
                "id": "output",
                "params": {"fields": {"output": {"type": "string"}}},
            },
            "edges": [
                {
                    "source_id": "old_node",
                    "source_key": "result",
                    "target_id": "output",
                    "target_key": "output",
                }
            ],
        }

        workflow = Workflow.model_validate(workflow_data)
        engine = WorkflowEngine()
        workflow = await engine.validate(workflow)

        node = workflow.inner_nodes[0]
        assert isinstance(node, MigratableNode)
        assert node.version == "2.0.0"
        assert node.params.value.root == "old_value"

    @pytest.mark.unit
    def test_chained_migration(self):
        """Test that multi-step migrations work."""

        class Migration_1_5_0_to_2_0_0(Migration):
            node_type = NODE_TYPE
            from_version = "1.5.0"
            to_version = "2.0.0"

            def migrate(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
                result = dict(data)
                params = dict(result.get("params", {}))
                if "content" in params:
                    params["value"] = params.pop("content")
                result["params"] = params
                return result

        class Migration_1_0_0_to_1_5_0(Migration):
            node_type = NODE_TYPE
            from_version = "1.0.0"
            to_version = "1.5.0"

            def migrate(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
                result = dict(data)
                params = dict(result.get("params", {}))
                if "text" in params:
                    params["content"] = params.pop("text")
                result["params"] = params
                return result

        migration_registry.register(Migration_1_0_0_to_1_5_0)
        migration_registry.register(Migration_1_5_0_to_2_0_0)

        data = {
            "type": NODE_TYPE,
            "id": "test",
            "version": "1.0.0",
            "params": {"text": "hello"},
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            untyped_node = Node.model_validate(data)
            node = NodeRegistry.DEFAULT.load(untyped_node)
            assert len(w) == 0

        assert node.version == "2.0.0"
        assert node.params.value.root == "hello"

    @pytest.mark.unit
    def test_migration_preserves_extra_fields(self):
        """Test that extra fields like position are preserved during migration."""
        migration_registry.register(MigratableNodeMigration_1_0_0_to_2_0_0)

        data = {
            "type": NODE_TYPE,
            "id": "test",
            "version": "1.0.0",
            "params": {"text": "hello"},
            "position": {"x": 100, "y": 200},
        }

        untyped_node = Node.model_validate(data)
        node = NodeRegistry.DEFAULT.load(untyped_node)

        dumped = node.model_dump()
        assert dumped["position"] == {"x": 100, "y": 200}

    @pytest.mark.unit
    def test_direct_class_validation_requires_current_schema(self):
        """Test that calling concrete class directly requires current version schema."""
        migration_registry.register(MigratableNodeMigration_1_0_0_to_2_0_0)

        old_data = {
            "type": NODE_TYPE,
            "id": "test",
            "version": "1.0.0",
            "params": {"text": "hello"},
        }

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MigratableNode.model_validate(old_data)

        untyped_node = Node.model_validate(old_data)
        node = NodeRegistry.DEFAULT.load(untyped_node)
        assert node.version == "2.0.0"
        assert node.params.value.root == "hello"

        current_data = {
            "type": NODE_TYPE,
            "id": "test",
            "version": "2.0.0",
            "params": {"value": "hello"},
        }
        node = MigratableNode.model_validate(current_data)
        assert node.version == "2.0.0"


class TestMigrationEdgeCases:
    """Test edge cases in migration."""

    @pytest.mark.unit
    def test_latest_version_not_migrated(self):
        """Test that 'latest' version is not migrated but resolved."""
        data = {
            "type": NODE_TYPE,
            "id": "test",
            "version": "latest",
            "params": {"value": "hello"},
        }

        untyped_node = Node.model_validate(data)
        node = NodeRegistry.DEFAULT.load(untyped_node)

        assert node.version == "2.0.0"

    @pytest.mark.unit
    def test_newer_version_raises_error(self):
        """Test that newer version than current raises error."""
        from pydantic import ValidationError

        data = {
            "type": NODE_TYPE,
            "id": "test",
            "version": "3.0.0",
            "params": {"value": "hello"},
        }

        untyped_node = Node.model_validate(data)

        with pytest.raises(ValidationError) as exc_info:
            NodeRegistry.DEFAULT.load(untyped_node)

        assert "newer than the latest version" in str(exc_info.value)
