# workflow_engine/files/docx.py
from typing import ClassVar

from ..core import FileValue


class DocXFileValue(FileValue):
    mime_type: ClassVar[str] = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


__all__ = [
    "DocXFileValue",
]
