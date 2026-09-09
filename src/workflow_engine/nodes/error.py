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
        # error_name is an arbitrary, author-supplied string (this node exists
        # to let a workflow author trigger a failure on demand for testing).
        # The engine has no way to infer a cause from it, so error_class is
        # left unset here rather than guessed at; it materializes as
        # systemic, which is the honest answer for a genuinely unknown cause.
        raise WorkflowException.for_user(
            f"{self.params.error_name}: {input.info}",
        )


__all__ = [
    "ErrorNode",
]
