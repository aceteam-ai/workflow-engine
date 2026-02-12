# workflow_engine/files/text.py
from typing import ClassVar, Self

from ..core import Context, File, FileValue, StringValue


class TextFileValue(FileValue):
    mime_type: ClassVar[str] = "text/plain"

    async def read_text(self, context: Context) -> str:
        return (await self.read(context)).decode("utf-8")

    async def write_text(self, context: Context, text: str) -> Self:
        return await self.write(context, text.encode("utf-8"))


@TextFileValue.register_cast_to(StringValue)
async def cast_text_to_string(value: TextFileValue, context: Context) -> StringValue:
    return StringValue(await value.read_text(context))


@StringValue.register_cast_to(TextFileValue)
async def cast_string_to_text(value: StringValue, context: Context) -> TextFileValue:
    file = TextFileValue(File(path=f"{value.md5}.txt"))
    return await file.write_text(context, value.root)


class MarkdownFileValue(TextFileValue):
    mime_type: ClassVar[str] = "text/markdown"


@StringValue.register_cast_to(MarkdownFileValue)
async def cast_string_to_markdown(value: StringValue, context: Context) -> MarkdownFileValue:
    file = MarkdownFileValue(File(path=f"{value.md5}.md"))
    return await file.write_text(context, value.root)


__all__ = [
    "TextFileValue",
    "MarkdownFileValue",
]
