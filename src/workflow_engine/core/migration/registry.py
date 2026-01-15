# workflow_engine/core/migration/registry.py
"""Registry for node migrations."""

import logging
from collections import deque
from typing import Type

from .migration import Migration

logger = logging.getLogger(__name__)


class MigrationRegistry:
    """
    Registry for node migrations.

    Migrations are indexed by (node_type, from_version) pairs.
    The registry supports finding migration paths between versions
    using breadth-first search.

    Example:
        ```python
        from workflow_engine.core.migration import Migration, migration_registry

        @migration_registry.register
        class MyMigration(Migration):
            node_type = "MyNode"
            from_version = "1.0.0"
            to_version = "2.0.0"

            def migrate(self, data):
                return data
        ```
    """

    def __init__(self):
        # Key: (node_type, from_version), Value: Migration class
        self._migrations: dict[tuple[str, str], Type[Migration]] = {}

    def register(self, migration_cls: Type[Migration]) -> Type[Migration]:
        """
        Register a migration class.

        Can be used as a decorator:
            @migration_registry.register
            class MyMigration(Migration):
                ...

        Args:
            migration_cls: The Migration subclass to register

        Returns:
            The migration class (for decorator usage)

        Raises:
            ValueError: If a migration for this (node_type, from_version) is already registered
        """
        key = (migration_cls.node_type, migration_cls.from_version)
        if key in self._migrations:
            existing = self._migrations[key]
            if migration_cls is not existing:
                raise ValueError(
                    f"Migration for {migration_cls.node_type} from {migration_cls.from_version} "
                    f"already registered: {existing.__name__}"
                )
        self._migrations[key] = migration_cls
        logger.debug(
            "Registered migration %s for %s: %s -> %s",
            migration_cls.__name__,
            migration_cls.node_type,
            migration_cls.from_version,
            migration_cls.to_version,
        )
        return migration_cls

    def get(self, node_type: str, from_version: str) -> Type[Migration] | None:
        """
        Get a migration for the given node type and source version.

        Args:
            node_type: The node type (e.g., "ConstantString")
            from_version: The source version to migrate from

        Returns:
            The Migration class, or None if not found
        """
        return self._migrations.get((node_type, from_version))

    def get_migration_path(
        self,
        node_type: str,
        from_version: str,
        to_version: str,
    ) -> list[Type[Migration]]:
        """
        Find the chain of migrations needed to go from from_version to to_version.

        Uses breadth-first search to find the shortest path through the
        version graph.

        Args:
            node_type: The node type to migrate
            from_version: Starting version
            to_version: Target version

        Returns:
            Ordered list of Migration classes to apply, or empty list if
            no path exists or versions are the same.
        """
        if from_version == to_version:
            return []

        # Build adjacency list for this node type
        adj: dict[str, list[tuple[str, Type[Migration]]]] = {}
        for (ntype, fv), migration_cls in self._migrations.items():
            if ntype == node_type:
                adj.setdefault(fv, []).append(
                    (migration_cls.to_version, migration_cls)
                )

        # BFS to find shortest path
        queue: deque[tuple[str, list[Type[Migration]]]] = deque(
            [(from_version, [])]
        )
        visited = {from_version}

        while queue:
            current, path = queue.popleft()

            for next_version, migration_cls in adj.get(current, []):
                new_path = path + [migration_cls]

                if next_version == to_version:
                    return new_path

                if next_version not in visited:
                    visited.add(next_version)
                    queue.append((next_version, new_path))

        return []  # No path found

    def has_migrations_for(self, node_type: str) -> bool:
        """
        Check if any migrations are registered for a node type.

        Args:
            node_type: The node type to check

        Returns:
            True if at least one migration exists for this node type
        """
        return any(ntype == node_type for ntype, _ in self._migrations.keys())

    def clear(self):
        """Clear all registered migrations. Useful for testing."""
        self._migrations.clear()


# Global singleton registry
migration_registry = MigrationRegistry()


__all__ = ["MigrationRegistry", "migration_registry"]
