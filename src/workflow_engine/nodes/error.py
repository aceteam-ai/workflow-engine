# workflow_engine/nodes/error.py

from typing import ClassVar, Type

from overrides import override
from pydantic import Field

from ..core import (
    Data,
    Empty,
    ExecutionContext,
    Node,
    NodeTypeInfo,
    Params,
    StringValue,
    WorkflowException,
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
        display_name="Error",
        description="A node that always raises an error.",
        version="0.4.0",
        parameter_type=ErrorParams,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[ErrorInput]:
        return ErrorInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[Empty]:
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
        raise WorkflowException.for_user(
            f"{self.params.error_name}: {input.info}",
        )


__all__ = [
    "ErrorNode",
]
