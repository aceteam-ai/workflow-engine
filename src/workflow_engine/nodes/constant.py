# workflow_engine/nodes/constant.py
from typing import ClassVar, Self, Type

from overrides import override
from pydantic import Field

from ..core import (
    BooleanValue,
    Data,
    Empty,
    ExecutionContext,
    IntegerValue,
    Node,
    NodeTypeInfo,
    Params,
    StringValue,
)


class ConstantBoolean(Params):
    value: BooleanValue = Field(
        title="Value",
        description="The constant boolean value.",
    )

    @classmethod
    def from_value(cls, value: bool) -> Self:
        return cls(value=BooleanValue(value))


class ConstantBooleanNode(Node[Empty, ConstantBoolean, ConstantBoolean]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Constant Boolean",
        description="A node that outputs a constant boolean value.",
        version="0.4.0",
        parameter_type=ConstantBoolean,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> Type[ConstantBoolean]:
        return ConstantBoolean

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[ConstantBoolean],
        input: Empty,
    ) -> ConstantBoolean:
        return self.params


class ConstantInteger(Params):
    value: IntegerValue = Field(
        title="Value", description="The constant integer value."
    )

    @classmethod
    def from_value(cls, value: int) -> Self:
        return cls(value=IntegerValue(value))


class ConstantIntegerNode(Node[Empty, ConstantInteger, ConstantInteger]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Constant Integer",
        description="A node that outputs a constant integer value.",
        version="0.4.0",
        parameter_type=ConstantInteger,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> Type[ConstantInteger]:
        return ConstantInteger

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[ConstantInteger],
        input: Empty,
    ) -> ConstantInteger:
        return self.params


class ConstantString(Params):
    value: StringValue = Field(title="Value", description="The constant string value.")


class ConstantStringNode(Node[Empty, ConstantString, ConstantString]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Constant String",
        description="A node that outputs a constant string value.",
        version="0.4.0",
        parameter_type=ConstantString,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> Type[ConstantString]:
        return ConstantString

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Data],
        output_type: Type[ConstantString],
        input: Data,
    ) -> ConstantString:
        return self.params


__all__ = [
    "ConstantBooleanNode",
    "ConstantIntegerNode",
    "ConstantStringNode",
]
