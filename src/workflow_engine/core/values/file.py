# workflow_engine/core/values/file.py
from abc import ABC
from collections.abc import Mapping
from logging import getLogger
from typing import TYPE_CHECKING, Any, ClassVar, Self

from pydantic import ConfigDict, Field

from ...utils.immutable import ImmutableBaseModel
from .value import Value

if TYPE_CHECKING:
    from ..context import ExecutionContext

logger = getLogger(__name__)


class File(ImmutableBaseModel, ABC):
    """
    A serializable reference to a file.

    A Context provides the actual implementation to read the file's contents.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    metadata: Mapping[str, Any] = Field(default_factory=dict)
    path: str


class FileValue(Value[File]):
    """
    A Value that represents a file.
    """

    mime_type: ClassVar[str]

    async def read(self, context: "ExecutionContext") -> bytes:
        return await context.read(file=self)

    async def write(self, context: "ExecutionContext", content: bytes) -> Self:
        return await context.write(file=self, content=content)

    async def copy_from_local_file(
        self, context: "ExecutionContext", path: str
    ) -> Self:
        with open(path, "rb") as f:
            data = f.read()
            return await self.write(context, data)

    def write_metadata(self, key: str, value: Any, overwrite: bool = False) -> Self:
        """
        Adds the given key-value pair to the file's metadata.

        If the key already exists, it must have the same value.
        Otherwise, an error is raised unless `overwrite` is True.
        """
        if key in self.metadata:
            old_value = self.metadata[key]
            if old_value == value:
                return self
            elif not overwrite:
                raise ValueError(
                    f"Metadata key {key} already exists with value {old_value}, which is different from the new value {value}. Pass overwrite=True to overwrite."
                )
            else:
                logger.warning(
                    f"Metadata key {key} already exists with value {old_value}. Overwriting with new value {value}."
                )

        metadata = dict(self.metadata)
        metadata[key] = value
        return type(self)(self.root.model_copy(update={"metadata": metadata}))

    @classmethod
    def from_path(cls, path: str, **metadata: Any) -> Self:
        return cls(root=File(path=path, metadata=metadata))

    @property
    def path(self) -> str:
        return self.root.path

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.root.metadata


__all__ = [
    "File",
    "FileValue",
]
