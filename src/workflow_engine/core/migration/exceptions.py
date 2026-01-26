# workflow_engine/core/migration/exceptions.py
"""Migration-related exceptions."""


class MigrationError(Exception):
    """Base exception for migration errors."""

    pass


class MigrationNotFoundError(MigrationError):
    """Raised when no migration path exists between versions."""

    def __init__(self, node_type: str, from_version: str, to_version: str):
        self.node_type = node_type
        self.from_version = from_version
        self.to_version = to_version
        super().__init__(
            f"No migration path found for {node_type} "
            f"from version {from_version} to {to_version}"
        )


class MigrationValidationError(MigrationError):
    """Raised when migration validation fails."""

    pass


__all__ = [
    "MigrationError",
    "MigrationNotFoundError",
    "MigrationValidationError",
]
