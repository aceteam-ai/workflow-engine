# workflow_engine/nodes/text.py
import os
from typing import ClassVar, Literal, Self, Type
from overrides import override
from pydantic import Field

from ..core import (
    ExecutionContext,
    Data,
    File,
    Node,
    NodeTypeInfo,
    Params,
    StringValue,
    ValidationContext,
)
from ..files import TextFileValue


class AppendToFileInput(Data):
    file: TextFileValue = Field(
        title="File",
        description="The file to append to.",
    )
    text: StringValue = Field(
        title="Text",
        description="The text to append.",
    )


class AppendToFileOutput(Data):
    file: TextFileValue = Field(
        title="File",
        description="The resulting file with the text appended.",
    )


class AppendToFileParams(Params):
    suffix: StringValue = Field(
        title="Suffix",
        description="The suffix to add to the output filename.",
    )


class AppendToFileNode(Node[AppendToFileInput, AppendToFileOutput, AppendToFileParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="AppendToFile",
        display_name="AppendToFile",
        description="Appends a string to the end of a file.",
        version="0.4.0",
        parameter_type=AppendToFileParams,
    )

    type: Literal["AppendToFile"] = "AppendToFile"  # pyright: ignore[reportIncompatibleVariableOverride]

    @override
    async def input_type(self, context: ValidationContext) -> Type[AppendToFileInput]:
        return AppendToFileInput

    @override
    async def output_type(self, context: ValidationContext) -> Type[AppendToFileOutput]:
        return AppendToFileOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[AppendToFileInput],
        output_type: Type[AppendToFileOutput],
        input: AppendToFileInput,
    ) -> AppendToFileOutput:
        old_text = await input.file.read_text(context)
        new_text = old_text + input.text.root
        filename, ext = os.path.splitext(input.file.path)
        new_file = TextFileValue(File(path=filename + self.params.suffix.root + ext))
        new_file = await new_file.write_text(context, text=new_text)
        return AppendToFileOutput(file=new_file)

    @classmethod
    def from_suffix(cls, id: str, suffix: str) -> Self:
        return cls(id=id, params=AppendToFileParams(suffix=StringValue(suffix)))


__all__ = [
    "AppendToFileNode",
]
