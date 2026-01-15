# workflow_engine/core/migration/runner.py
"""Migration runner for applying migrations to node data."""

import logging
from typing import Any

from .exceptions import MigrationNotFoundError, MigrationValidationError
from .registry import MigrationRegistry, migration_registry

logger = logging.getLogger(__name__)


class MigrationRunner:
    """
    Applies migrations to transform node data between versions.

    The runner finds the migration path from the source version to the
    target version and applies each migration in sequence.

    Example:
        ```python
        from workflow_engine.core.migration import migration_runner

        old_data = {
            "type": "MyNode",
            "id": "node1",
            "version": "1.0.0",
            "params": {"old_field": "value"}
        }

        new_data = migration_runner.migrate(old_data, target_version="2.0.0")
        # new_data now has version="2.0.0" and migrated params
        ```
    """

    def __init__(self, registry: MigrationRegistry | None = None):
        """
        Initialize the migration runner.

        Args:
            registry: The migration registry to use. Defaults to the
                      global migration_registry singleton.
        """
        self._registry = registry or migration_registry

    def migrate(
        self,
        data: dict[str, Any],
        target_version: str,
    ) -> dict[str, Any]:
        """
        Migrate node data to the target version.

        Args:
            data: Raw serialized node data containing 'type', 'version', etc.
            target_version: The version to migrate to

        Returns:
            Migrated node data with version == target_version

        Raises:
            MigrationNotFoundError: If no migration path exists
            MigrationValidationError: If migration validation fails
        """
        node_type = data.get("type")
        current_version = data.get("version")

        if current_version == target_version:
            return data

        if node_type is None:
            raise MigrationValidationError("Node data missing 'type' field")

        if current_version is None:
            raise MigrationValidationError("Node data missing 'version' field")

        # Find migration path
        migrations = self._registry.get_migration_path(
            node_type, current_version, target_version
        )

        if not migrations:
            raise MigrationNotFoundError(node_type, current_version, target_version)

        logger.debug(
            "Migrating %s from %s to %s via %d migration(s)",
            node_type,
            current_version,
            target_version,
            len(migrations),
        )

        # Apply migrations in sequence
        result = dict(data)
        for migration_cls in migrations:
            migration = migration_cls()

            # Validate before migrating
            errors = migration.validate(result)
            if errors:
                raise MigrationValidationError(
                    f"Migration {migration_cls.__name__} validation failed: {errors}"
                )

            # Apply migration
            logger.debug(
                "Applying migration %s: %s -> %s",
                migration_cls.__name__,
                migration_cls.from_version,
                migration_cls.to_version,
            )
            result = migration.migrate(result)
            result["version"] = migration_cls.to_version

        return result

    def can_migrate(
        self,
        node_type: str,
        from_version: str,
        to_version: str,
    ) -> bool:
        """
        Check if a migration path exists.

        Args:
            node_type: The node type to migrate
            from_version: Starting version
            to_version: Target version

        Returns:
            True if a migration path exists
        """
        if from_version == to_version:
            return True
        return bool(
            self._registry.get_migration_path(node_type, from_version, to_version)
        )


# Global singleton runner
migration_runner = MigrationRunner()


__all__ = ["MigrationRunner", "migration_runner"]
