# workflow_engine/contexts/in_memory.py
from typing import TypeVar

from overrides import override

from ..core import ExecutionContext, FileValue, ValidationContext
from ..core.limits import LimitCoordinator, LimitPolicy

F = TypeVar("F", bound=FileValue)


class InMemoryExecutionContext(ExecutionContext):
    """
    Pretends to be a file system, but actually stores files in memory.
    """

    def __init__(
        self,
        *,
        validation_context: ValidationContext | None = None,
        limit_coordinator: LimitCoordinator | None = None,
        limit_policy: LimitPolicy | None = None,
        run_id: str | None = None,
    ):
        super().__init__(
            validation_context=validation_context,
            limit_coordinator=limit_coordinator,
            limit_policy=limit_policy,
            run_id=run_id,
        )
        self.data: dict[str, bytes] = {}

    @override
    async def read(
        self,
        file: FileValue,
    ) -> bytes:
        return self.data[file.path]

    @override
    async def write(
        self,
        file: F,
        content: bytes,
    ) -> F:
        self.data[file.path] = content
        return file


__all__ = [
    "InMemoryExecutionContext",
]
