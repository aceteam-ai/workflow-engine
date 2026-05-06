# workflow_engine/nodes/text.py
import os
from typing import ClassVar, Type

from overrides import override
from pydantic import Field

from ..core import (
    Data,
    ExecutionContext,
    File,
    Node,
    NodeTypeInfo,
    Params,
    StringValue,
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
        display_name="Append to File",
        description="Appends a string to the end of a file.",
        version="0.4.0",
        parameter_type=AppendToFileParams,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[AppendToFileInput]:
        return AppendToFileInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[AppendToFileOutput]:
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


__all__ = [
    "AppendToFileNode",
]
