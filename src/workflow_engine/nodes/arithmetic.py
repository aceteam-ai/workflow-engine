# workflow_engine/nodes/arithmetic.py
"""
Simple nodes for testing the workflow engine, with limited usefulness otherwise.
"""

from functools import cached_property
from typing import ClassVar, Literal, Self

from pydantic import Field, create_model

from ..core import (
    Context,
    Data,
    Empty,
    FloatValue,
    IntegerValue,
    Node,
    NodeTypeInfo,
    Params,
    SequenceValue,
)


def _argument_field_name(index: int) -> str:
    """
    Generate a field name for the given zero-based index.
    Produces "a", "b", ..., "z", "aa", "ab", ..., "az", "ba", ...
    """
    letters = []
    n = index
    while True:
        letters.append(chr(ord("a") + n % 26))
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(reversed(letters))


class AddNodeParams(Params):
    num_arguments: IntegerValue = Field(
        title="Number of Arguments",
        description="The number of numbers to add.",
        default=IntegerValue(2),
        json_schema_extra={"minimum": 2},
    )


class SumOutput(Data):
    sum: FloatValue = Field(title="Sum", description="The sum of the numbers.")


class AddNode(Node[Data, SumOutput, AddNodeParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="Add",
        display_name="Add",
        description="Adds two or more numbers.",
        version="1.0.0",
        parameter_type=AddNodeParams,
    )

    type: Literal["Add"] = "Add"  # pyright: ignore[reportIncompatibleVariableOverride]
    # we can do this because an empty AddNodeParams is valid
    params: AddNodeParams = Field(default=AddNodeParams())  # pyright: ignore[reportIncompatibleVariableOverride]

    @cached_property
    def input_type(self):
        n = self.params.num_arguments.root
        field_names = [_argument_field_name(i) for i in range(n)]
        fields = {
            name: (
                FloatValue,
                Field(title=name.upper(), description=f"The number {name}."),
            )
            for name in field_names
        }
        return create_model("AddNodeInput", __base__=Data, **fields)  # type: ignore

    @cached_property
    def output_type(self):
        return SumOutput

    async def run(self, context: Context, input: Data) -> SumOutput:
        total = sum(
            getattr(input, _argument_field_name(i)).root
            for i in range(self.params.num_arguments.root)
        )
        return SumOutput(sum=FloatValue(total))

    @classmethod
    def with_arity(cls, id: str, arity: int) -> Self:
        return cls(id=id, params=AddNodeParams(num_arguments=IntegerValue(arity)))


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
