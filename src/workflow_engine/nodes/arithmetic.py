# workflow_engine/nodes/arithmetic.py
"""
Simple nodes for testing the workflow engine, with limited usefulness otherwise.
"""

from functools import cached_property
from typing import ClassVar, Literal

from pydantic import Field

from ..core import (
    Context,
    Data,
    Empty,
    FloatValue,
    IntegerValue,
    Node,
    NodeTypeInfo,
    SequenceValue,
)


class AddNodeInput(Data):
    a: FloatValue = Field(title="A", description="The first number.")
    b: FloatValue = Field(title="B", description="The second number.")


class SumOutput(Data):
    sum: FloatValue = Field(title="Sum", description="The sum of the two numbers.")


class AddNode(Node[AddNodeInput, SumOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo(
        name="Add",
        display_name="Add",
        description="Adds two numbers.",
        version="0.4.0",
    )

    type: Literal["Add"] = "Add"  # pyright: ignore[reportIncompatibleVariableOverride]

    @cached_property
    def input_type(self):
        return AddNodeInput

    @cached_property
    def output_type(self):
        return SumOutput

    async def run(self, context: Context, input: AddNodeInput) -> SumOutput:
        return SumOutput(sum=FloatValue(input.a.root + input.b.root))


class SumNodeInput(Data):
    values: SequenceValue[FloatValue] = Field(
        title="Values", description="The numbers to sum."
    )


class SumNodeOutput(Data):
    sum: FloatValue = Field(title="Sum", description="The sum of all the numbers.")


class SumNode(Node[SumNodeInput, SumNodeOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="Sum",
        display_name="Sum",
        description="Sums a sequence of numbers.",
        version="0.4.0",
        parameter_type=Empty,
    )

    type: Literal["Sum"] = "Sum"  # pyright: ignore[reportIncompatibleVariableOverride]

    @cached_property
    def input_type(self):
        return SumNodeInput

    @cached_property
    def output_type(self):
        return SumNodeOutput

    async def run(self, context: Context, input: SumNodeInput) -> SumNodeOutput:
        return SumNodeOutput(sum=FloatValue(sum(v.root for v in input.values)))


class IntegerData(Data):
    value: IntegerValue = Field(title="Value", description="The integer value.")


class FactorizationData(Data):
    factors: SequenceValue[IntegerValue] = Field(
        title="Factors", description="The factors of the integer."
    )


class FactorizationNode(Node[IntegerData, FactorizationData, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="Factorization",
        display_name="Factorization",
        description="Factorizes an integer into a sequence of its factors.",
        version="0.4.0",
        parameter_type=Empty,
    )

    type: Literal["Factorization"] = "Factorization"  # pyright: ignore[reportIncompatibleVariableOverride]

    @cached_property
    def input_type(self):
        return IntegerData

    @cached_property
    def output_type(self):
        return FactorizationData

    async def run(self, context: Context, input: IntegerData) -> FactorizationData:
        value = input.value.root
        if value > 0:
            return FactorizationData(
                factors=SequenceValue(
                    [IntegerValue(i) for i in range(1, value + 1) if value % i == 0]
                )
            )
        raise ValueError("Can only factorize positive integers")


__all__ = [
    "AddNode",
    "FactorizationNode",
    "SumNode",
]
