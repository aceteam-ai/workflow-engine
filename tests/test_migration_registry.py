# tests/test_migration_registry.py
"""Tests for MigrationRegistry."""

from typing import Any

import pytest

from workflow_engine.core.migration import Migration, MigrationRegistry


class MockMigration_1_0_0_to_2_0_0(Migration):
    """Test migration from 1.0.0 to 2.0.0."""

    node_type = "TestNode"
    from_version = "1.0.0"
    to_version = "2.0.0"

    def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        return dict(data)


class MockMigration_2_0_0_to_3_0_0(Migration):
    """Test migration from 2.0.0 to 3.0.0."""

    node_type = "TestNode"
    from_version = "2.0.0"
    to_version = "3.0.0"

    def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        return dict(data)


class MockMigration_1_0_0_to_3_0_0(Migration):
    """Direct migration from 1.0.0 to 3.0.0 (shortcut)."""

    node_type = "TestNode"
    from_version = "1.0.0"
    to_version = "3.0.0"

    def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        return dict(data)


class OtherNodeMigration(Migration):
    """Migration for a different node type."""

    node_type = "OtherNode"
    from_version = "1.0.0"
    to_version = "2.0.0"

    def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        return dict(data)


class TestMigrationRegistry:
    """Tests for MigrationRegistry."""

    @pytest.fixture
    def registry(self) -> MigrationRegistry:
        """Create a fresh registry for each test."""
        return MigrationRegistry()

    @pytest.mark.unit
    def test_register_migration(self, registry: MigrationRegistry):
        """Test basic migration registration."""
        result = registry.register(MockMigration_1_0_0_to_2_0_0)

        # Should return the class for decorator usage
        assert result is MockMigration_1_0_0_to_2_0_0

        # Should be retrievable
        migration = registry.get("TestNode", "1.0.0")
        assert migration is MockMigration_1_0_0_to_2_0_0

    @pytest.mark.unit
    def test_register_duplicate_raises_error(self, registry: MigrationRegistry):
        """Test that registering duplicate migration raises error."""
        registry.register(MockMigration_1_0_0_to_2_0_0)

        # Create a different class with same node_type and from_version
        class DuplicateMigration(Migration):
            node_type = "TestNode"
            from_version = "1.0.0"
            to_version = "2.5.0"

            def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
                return data

        with pytest.raises(ValueError, match="already registered"):
            registry.register(DuplicateMigration)

    @pytest.mark.unit
    def test_register_same_class_twice_is_ok(self, registry: MigrationRegistry):
        """Test that registering the same class twice is allowed."""
        registry.register(MockMigration_1_0_0_to_2_0_0)
        registry.register(MockMigration_1_0_0_to_2_0_0)  # Should not raise

    @pytest.mark.unit
    def test_get_unregistered_returns_none(self, registry: MigrationRegistry):
        """Test that getting unregistered migration returns None."""
        result = registry.get("NonExistent", "1.0.0")
        assert result is None

    @pytest.mark.unit
    def test_get_migration_path_same_version(self, registry: MigrationRegistry):
        """Test that same version returns empty path."""
        path = registry.get_migration_path("TestNode", "1.0.0", "1.0.0")
        assert path == []

    @pytest.mark.unit
    def test_get_migration_path_single_step(self, registry: MigrationRegistry):
        """Test finding a single-step migration path."""
        registry.register(MockMigration_1_0_0_to_2_0_0)

        path = registry.get_migration_path("TestNode", "1.0.0", "2.0.0")
        assert len(path) == 1
        assert path[0] is MockMigration_1_0_0_to_2_0_0

    @pytest.mark.unit
    def test_get_migration_path_multi_step(self, registry: MigrationRegistry):
        """Test finding a multi-step migration path."""
        registry.register(MockMigration_1_0_0_to_2_0_0)
        registry.register(MockMigration_2_0_0_to_3_0_0)

        path = registry.get_migration_path("TestNode", "1.0.0", "3.0.0")
        assert len(path) == 2
        assert path[0] is MockMigration_1_0_0_to_2_0_0
        assert path[1] is MockMigration_2_0_0_to_3_0_0

    @pytest.mark.unit
    def test_get_migration_path_finds_available_route(self, registry: MigrationRegistry):
        """Test that BFS finds available migration route."""
        # Register multi-step path
        registry.register(MockMigration_1_0_0_to_2_0_0)
        registry.register(MockMigration_2_0_0_to_3_0_0)

        path = registry.get_migration_path("TestNode", "1.0.0", "3.0.0")
        # Should find the 2-step path
        assert len(path) == 2
        assert path[0] is MockMigration_1_0_0_to_2_0_0
        assert path[1] is MockMigration_2_0_0_to_3_0_0

    @pytest.mark.unit
    def test_get_migration_path_no_path_exists(self, registry: MigrationRegistry):
        """Test that missing path returns empty list."""
        registry.register(MockMigration_1_0_0_to_2_0_0)

        # No path from 2.0.0 to 4.0.0
        path = registry.get_migration_path("TestNode", "2.0.0", "4.0.0")
        assert path == []

    @pytest.mark.unit
    def test_get_migration_path_different_node_types(self, registry: MigrationRegistry):
        """Test that paths don't cross node types."""
        registry.register(MockMigration_1_0_0_to_2_0_0)
        registry.register(OtherNodeMigration)

        # TestNode migrations shouldn't work for OtherNode
        path = registry.get_migration_path("OtherNode", "1.0.0", "2.0.0")
        assert len(path) == 1
        assert path[0] is OtherNodeMigration

        # And vice versa
        path = registry.get_migration_path("TestNode", "1.0.0", "2.0.0")
        assert len(path) == 1
        assert path[0] is MockMigration_1_0_0_to_2_0_0

    @pytest.mark.unit
    def test_has_migrations_for(self, registry: MigrationRegistry):
        """Test checking if migrations exist for a node type."""
        assert not registry.has_migrations_for("TestNode")

        registry.register(MockMigration_1_0_0_to_2_0_0)

        assert registry.has_migrations_for("TestNode")
        assert not registry.has_migrations_for("OtherNode")

    @pytest.mark.unit
    def test_clear(self, registry: MigrationRegistry):
        """Test clearing all migrations."""
        registry.register(MockMigration_1_0_0_to_2_0_0)
        assert registry.has_migrations_for("TestNode")

        registry.clear()

        assert not registry.has_migrations_for("TestNode")
        assert registry.get("TestNode", "1.0.0") is None

    @pytest.mark.unit
    def test_decorator_usage(self, registry: MigrationRegistry):
        """Test using register as a decorator."""

        @registry.register
        class DecoratedMigration(Migration):
            node_type = "DecoratedNode"
            from_version = "1.0.0"
            to_version = "2.0.0"

            def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
                return data

        assert registry.get("DecoratedNode", "1.0.0") is DecoratedMigration
