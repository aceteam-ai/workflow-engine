# workflow_engine/core/migration/migration.py
"""Base class for node migrations."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar


class Migration(ABC):
    """
    Base class for node migrations.

    A migration transforms serialized node data from one version to another.
    Migrations operate on raw dict data BEFORE Pydantic validation, allowing
    schema changes between versions.

    Example:
        ```python
        from collections.abc import Mapping
        from typing import Any
        from workflow_engine.core.migration import Migration, migration_registry

        @migration_registry.register
        class MyNodeMigration_1_0_0_to_2_0_0(Migration):
            node_type = "MyNode"
            from_version = "1.0.0"
            to_version = "2.0.0"

            def migrate(self, data: Mapping[str, Any]) -> dict[str, Any]:
                # Create mutable copy - data is read-only
                result = dict(data)
                params = dict(result.get("params", {}))
                # Rename 'old_field' to 'new_field'
                params["new_field"] = params.pop("old_field", "default")
                result["params"] = params
                return result
        ```
    """

    # The node type this migration applies to (e.g., "ConstantString")
    node_type: ClassVar[str]

    # Source version (the version being migrated FROM)
    from_version: ClassVar[str]

    # Target version (the version being migrated TO)
    to_version: ClassVar[str]

    @abstractmethod
    def migrate(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Transform node data from from_version to to_version.

        Args:
            data: Raw serialized node data with version == from_version.
                  Contains keys like 'type', 'id', 'version', 'params'.
                  This is a read-only Mapping to enforce immutability.

        Returns:
            Transformed node data (typically a dict, which satisfies Mapping).
            The 'version' field will be updated by the MigrationRunner.

        Note:
            - The input 'version' field will be from_version
            - Do NOT modify the 'version' field (handled by runner)
            - The 'type' and 'id' fields should be preserved
            - Return a new dict created from dict(data) - don't mutate input
            - Return type is Mapping to discourage mutation of the result
        """
        raise NotImplementedError

    def validate(self, data: Mapping[str, Any]) -> list[str]:
        """
        Optionally validate that the data can be migrated.

        Override this method to add pre-migration validation checks.

        Args:
            data: Raw serialized node data to validate (read-only)

        Returns:
            List of validation error messages (empty if valid)
        """
        return []


__all__ = ["Migration"]
