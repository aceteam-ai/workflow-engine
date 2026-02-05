# workflow_engine/nodes/constant.py
from functools import cached_property
from typing import ClassVar, Literal

from ..core import (
    BooleanValue,
    Context,
    Data,
    Empty,
    IntegerValue,
    Node,
    NodeTypeInfo,
    Params,
    StringValue,
)


class ConstantBoolean(Params):
    value: BooleanValue


class ConstantBooleanNode(Node[Empty, ConstantBoolean, ConstantBoolean]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="ConstantBoolean",
        display_name="ConstantBoolean",
        description="A node that outputs a constant boolean value.",
        version="0.4.0",
        parameter_type=ConstantBoolean,
    )

    type: Literal["ConstantBoolean"] = "ConstantBoolean"  # pyright: ignore[reportIncompatibleVariableOverride]

    @cached_property
    def output_type(self):
        return ConstantBoolean

    async def run(self, context: Context, input: Empty) -> ConstantBoolean:
        return self.params

    @classmethod
    def from_value(cls, *, id: str, value: bool) -> "ConstantBooleanNode":
        return cls(id=id, params=ConstantBoolean(value=BooleanValue(value)))


class ConstantInteger(Params):
    value: IntegerValue


class ConstantIntegerNode(Node[Empty, ConstantInteger, ConstantInteger]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="ConstantInteger",
        display_name="ConstantInteger",
        description="A node that outputs a constant integer value.",
        version="0.4.0",
        parameter_type=ConstantInteger,
    )

    type: Literal["ConstantInteger"] = "ConstantInteger"  # pyright: ignore[reportIncompatibleVariableOverride]

    @cached_property
    def output_type(self):
        return ConstantInteger

    async def run(self, context: Context, input: Empty) -> ConstantInteger:
        return self.params

    @classmethod
    def from_value(cls, *, id: str, value: int) -> "ConstantIntegerNode":
        return cls(id=id, params=ConstantInteger(value=IntegerValue(value)))


class ConstantString(Params):
    value: StringValue


class ConstantStringNode(Node[Empty, ConstantString, ConstantString]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="ConstantString",
        display_name="ConstantString",
        description="A node that outputs a constant string value.",
        version="0.4.0",
        parameter_type=ConstantString,
    )

    type: Literal["ConstantString"] = "ConstantString"  # pyright: ignore[reportIncompatibleVariableOverride]

    @cached_property
    def output_type(self):
        return ConstantString

    async def run(self, context: Context, input: Data) -> ConstantString:
        return self.params

    @classmethod
    def from_value(cls, *, id: str, value: str) -> "ConstantStringNode":
        return cls(id=id, params=ConstantString(value=StringValue(value)))


__all__ = [
    "ConstantBooleanNode",
    "ConstantIntegerNode",
    "ConstantStringNode",
]
