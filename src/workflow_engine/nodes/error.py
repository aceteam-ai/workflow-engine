# workflow_engine/nodes/error.py

from typing import ClassVar, Literal, Type

from overrides import override
from pydantic import Field

from ..core import (
    ExecutionContext,
    Data,
    Empty,
    Node,
    NodeTypeInfo,
    Params,
    StringValue,
    UserException,
    ValidationContext,
)


class ErrorInput(Data):
    info: StringValue = Field(
        title="Info", description="Additional information about the error."
    )


class ErrorParams(Params):
    error_name: StringValue = Field(
        title="Error Name", description="The name of the error to raise."
    )


class ErrorNode(Node[ErrorInput, Empty, ErrorParams]):
    """
    A node that always raises an error.
    """

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="Error",
        display_name="Error",
        description="A node that always raises an error.",
        version="0.4.0",
        parameter_type=ErrorParams,
    )

    type: Literal["Error"] = "Error"  # pyright: ignore[reportIncompatibleVariableOverride]

    @override
    async def input_type(self, context: ValidationContext) -> Type[ErrorInput]:
        return ErrorInput

    @override
    async def output_type(self, context: ValidationContext) -> Type[Empty]:
        return Empty

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[ErrorInput],
        output_type: Type[Empty],
        input: ErrorInput,
    ) -> Empty:
        raise UserException(f"{self.params.error_name}: {input.info}")

    @classmethod
    def from_name(cls, id: str, name: str) -> "ErrorNode":
        return cls(id=id, params=ErrorParams(error_name=StringValue(name)))


__all__ = [
    "ErrorNode",
]
