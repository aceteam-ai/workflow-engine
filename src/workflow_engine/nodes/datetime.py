# workflow_engine/nodes/datetime.py
"""Date and time nodes."""

from datetime import datetime, timezone
from typing import ClassVar, Type

from overrides import override
from pydantic import Field

from ..core import (
    Data,
    DateValue,
    Empty,
    ExecutionContext,
    Node,
    NodeTypeInfo,
)


class NowOutput(Data):
    now: DateValue = Field(
        title="Now",
        description="The current date and time in UTC.",
    )


class NowNode(Node[Empty, NowOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Now",
        description="Outputs the current UTC date and time.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> Type[NowOutput]:
        return NowOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[NowOutput],
        input: Empty,
    ) -> NowOutput:
        return NowOutput(now=DateValue(datetime.now(timezone.utc)))


__all__ = ("NowNode",)
