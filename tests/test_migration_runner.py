# tests/test_migration_runner.py
"""Tests for MigrationRunner."""

from typing import Any

import pytest

from workflow_engine.core.migration import (
    Migration,
    MigrationNotFoundError,
    MigrationRegistry,
    MigrationRunner,
    MigrationValidationError,
)


class ParamRenameMigration(Migration):
    """Migration that renames a parameter."""

    node_type = "TestNode"
    from_version = "1.0.0"
    to_version = "2.0.0"

    def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        params = dict(result.get("params", {}))
        # Rename 'old_field' to 'new_field'
        if "old_field" in params:
            params["new_field"] = params.pop("old_field")
        result["params"] = params
        return result


class AddDefaultFieldMigration(Migration):
    """Migration that adds a default field."""

    node_type = "TestNode"
    from_version = "2.0.0"
    to_version = "3.0.0"

    def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        params = dict(result.get("params", {}))
        # Add new required field with default
        if "extra_field" not in params:
            params["extra_field"] = "default_value"
        result["params"] = params
        return result


class ValidationMigration(Migration):
    """Migration with validation."""

    node_type = "ValidatedNode"
    from_version = "1.0.0"
    to_version = "2.0.0"

    def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        return dict(data)

    def validate(self, data: dict[str, Any]) -> list[str]:
        errors = []
        if "required_field" not in data.get("params", {}):
            errors.append("Missing required_field in params")
        return errors


class TestMigrationRunner:
    """Tests for MigrationRunner."""

    @pytest.fixture
    def registry(self) -> MigrationRegistry:
        """Create a fresh registry for each test."""
        return MigrationRegistry()

    @pytest.fixture
    def runner(self, registry: MigrationRegistry) -> MigrationRunner:
        """Create a runner with the test registry."""
        return MigrationRunner(registry)

    @pytest.mark.unit
    def test_migrate_same_version_returns_data(
        self, runner: MigrationRunner, registry: MigrationRegistry
    ):
        """Test that migrating to same version returns original data."""
        data = {
            "type": "TestNode",
            "id": "test",
            "version": "1.0.0",
            "params": {"field": "value"},
        }

        result = runner.migrate(data, "1.0.0")
        assert result == data

    @pytest.mark.unit
    def test_migrate_single_step(
        self, runner: MigrationRunner, registry: MigrationRegistry
    ):
        """Test single-step migration."""
        registry.register(ParamRenameMigration)

        data = {
            "type": "TestNode",
            "id": "test",
            "version": "1.0.0",
            "params": {"old_field": "value"},
        }

        result = runner.migrate(data, "2.0.0")

        assert result["version"] == "2.0.0"
        assert result["params"]["new_field"] == "value"
        assert "old_field" not in result["params"]

    @pytest.mark.unit
    def test_migrate_multi_step(
        self, runner: MigrationRunner, registry: MigrationRegistry
    ):
        """Test multi-step migration chain."""
        registry.register(ParamRenameMigration)
        registry.register(AddDefaultFieldMigration)

        data = {
            "type": "TestNode",
            "id": "test",
            "version": "1.0.0",
            "params": {"old_field": "value"},
        }

        result = runner.migrate(data, "3.0.0")

        assert result["version"] == "3.0.0"
        assert result["params"]["new_field"] == "value"
        assert result["params"]["extra_field"] == "default_value"
        assert "old_field" not in result["params"]

    @pytest.mark.unit
    def test_migrate_no_path_raises_error(
        self, runner: MigrationRunner, registry: MigrationRegistry
    ):
        """Test that missing migration path raises error."""
        data = {
            "type": "TestNode",
            "id": "test",
            "version": "1.0.0",
            "params": {},
        }

        with pytest.raises(MigrationNotFoundError) as exc_info:
            runner.migrate(data, "2.0.0")

        assert exc_info.value.node_type == "TestNode"
        assert exc_info.value.from_version == "1.0.0"
        assert exc_info.value.to_version == "2.0.0"

    @pytest.mark.unit
    def test_migrate_missing_type_raises_error(
        self, runner: MigrationRunner, registry: MigrationRegistry
    ):
        """Test that missing type field raises error."""
        data = {
            "id": "test",
            "version": "1.0.0",
            "params": {},
        }

        with pytest.raises(MigrationValidationError, match="missing 'type' field"):
            runner.migrate(data, "2.0.0")

    @pytest.mark.unit
    def test_migrate_missing_version_raises_error(
        self, runner: MigrationRunner, registry: MigrationRegistry
    ):
        """Test that missing version field raises error."""
        data = {
            "type": "TestNode",
            "id": "test",
            "params": {},
        }

        with pytest.raises(MigrationValidationError, match="missing 'version' field"):
            runner.migrate(data, "2.0.0")

    @pytest.mark.unit
    def test_migrate_validation_failure(
        self, runner: MigrationRunner, registry: MigrationRegistry
    ):
        """Test that validation errors are raised."""
        registry.register(ValidationMigration)

        data = {
            "type": "ValidatedNode",
            "id": "test",
            "version": "1.0.0",
            "params": {},  # Missing required_field
        }

        with pytest.raises(MigrationValidationError, match="validation failed"):
            runner.migrate(data, "2.0.0")

    @pytest.mark.unit
    def test_migrate_validation_passes(
        self, runner: MigrationRunner, registry: MigrationRegistry
    ):
        """Test that valid data passes validation."""
        registry.register(ValidationMigration)

        data = {
            "type": "ValidatedNode",
            "id": "test",
            "version": "1.0.0",
            "params": {"required_field": "present"},
        }

        result = runner.migrate(data, "2.0.0")
        assert result["version"] == "2.0.0"

    @pytest.mark.unit
    def test_can_migrate_same_version(
        self, runner: MigrationRunner, registry: MigrationRegistry
    ):
        """Test can_migrate returns True for same version."""
        assert runner.can_migrate("TestNode", "1.0.0", "1.0.0")

    @pytest.mark.unit
    def test_can_migrate_with_path(
        self, runner: MigrationRunner, registry: MigrationRegistry
    ):
        """Test can_migrate returns True when path exists."""
        registry.register(ParamRenameMigration)

        assert runner.can_migrate("TestNode", "1.0.0", "2.0.0")

    @pytest.mark.unit
    def test_can_migrate_without_path(
        self, runner: MigrationRunner, registry: MigrationRegistry
    ):
        """Test can_migrate returns False when no path exists."""
        assert not runner.can_migrate("TestNode", "1.0.0", "2.0.0")

    @pytest.mark.unit
    def test_migrate_preserves_extra_fields(
        self, runner: MigrationRunner, registry: MigrationRegistry
    ):
        """Test that extra fields are preserved during migration."""
        registry.register(ParamRenameMigration)

        data = {
            "type": "TestNode",
            "id": "test",
            "version": "1.0.0",
            "params": {"old_field": "value"},
            "position": {"x": 100, "y": 200},  # Extra field
            "metadata": {"author": "test"},  # Another extra field
        }

        result = runner.migrate(data, "2.0.0")

        assert result["position"] == {"x": 100, "y": 200}
        assert result["metadata"] == {"author": "test"}
