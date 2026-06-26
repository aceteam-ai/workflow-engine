# workflow_engine/nodes/arithmetic.py
"""
Built-in arithmetic nodes.
"""

from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from math import prod
from typing import ClassVar, Type

from overrides import override
from pydantic import Field

from workflow_engine.core.values import build_data_type

from ..core import (
    Data,
    Empty,
    ExecutionContext,
    FloatValue,
    IntegerValue,
    Node,
    NodeException,
    NodeTypeInfo,
    Params,
    SequenceValue,
    StringValue,
    UnionValue,
    ValidationContext,
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


_FLOAT_OR_FLOAT_SEQUENCE = UnionValue[FloatValue, SequenceValue[FloatValue]]

_ROUNDING_MODES = {
    "half_even": ROUND_HALF_EVEN,
    "half_up": ROUND_HALF_UP,
}


def _decimal_values_from_union(
    value: FloatValue | SequenceValue[FloatValue],
) -> list[Decimal]:
    if isinstance(value, FloatValue):
        return [value.root]
    return [item.root for item in value.root]


def _require_nonzero(
    divisor: Decimal,
    *,
    node: Node,
    label: str = "divisor",
) -> None:
    if divisor == 0:
        raise NodeException.for_user(f"Cannot divide by zero: {label} is 0.", node=node)


def _round_decimal(value: Decimal, *, ndigits: int, mode: str) -> Decimal:
    rounding = _ROUNDING_MODES[mode]
    return value.quantize(Decimal(10) ** -ndigits, rounding=rounding)


class BinaryFloatInput(Data):
    a: FloatValue = Field(title="A", description="The first number.")
    b: FloatValue = Field(title="B", description="The second number.")


class PowerInput(Data):
    base: FloatValue = Field(title="Base", description="The base number.")
    exponent: FloatValue = Field(title="Exponent", description="The exponent.")


class UnaryFloatInput(Data):
    a: FloatValue = Field(title="A", description="The number.")


class UnionFloatInput(Data):
    values: _FLOAT_OR_FLOAT_SEQUENCE = Field(
        title="Values",
        description="The numbers to combine, as a single value or a sequence.",
    )


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
        display_name="Add",
        description="Adds two or more numbers.",
        version="1.0.0",
        parameter_type=AddNodeParams,
    )

    # we can do this because an empty AddNodeParams is valid
    params: AddNodeParams = Field(default=AddNodeParams())  # pyright: ignore[reportIncompatibleVariableOverride]

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> Type[Data]:
        n = self.params.num_arguments.root
        field_names = [_argument_field_name(i) for i in range(n)]
        fields = {
            name: (
                FloatValue,
                Field(title=name.upper()),
            )
            for name in field_names
        }
        return build_data_type(
            name="AddNodeInput",
            fields=fields,
            base_cls=Data,
        )

    @classmethod
    @override
    def static_output_type(cls) -> Type[SumOutput]:
        return SumOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Data],
        output_type: Type[SumOutput],
        input: Data,
    ) -> SumOutput:
        total = sum(
            getattr(input, _argument_field_name(i)).root
            for i in range(self.params.num_arguments.root)
        )
        return SumOutput(sum=FloatValue(total))


class SumNodeInput(Data):
    values: SequenceValue[FloatValue] = Field(
        title="Values",
        description="The numbers to sum.",
    )


class SumNodeOutput(Data):
    sum: FloatValue = Field(
        title="Sum",
        description="The sum of all the numbers.",
    )


class SumNode(Node[SumNodeInput, SumNodeOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Sum",
        description="Sums a sequence of numbers.",
        version="0.4.0",
        parameter_type=Empty,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[SumNodeInput]:
        return SumNodeInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[SumNodeOutput]:
        return SumNodeOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[SumNodeInput],
        output_type: Type[SumNodeOutput],
        input: SumNodeInput,
    ) -> SumNodeOutput:
        return output_type(sum=FloatValue(sum(v.root for v in input.values)))


class IntegerData(Data):
    value: IntegerValue = Field(
        title="Value",
        description="The integer value.",
    )


class FactorizationData(Data):
    factors: SequenceValue[IntegerValue] = Field(
        title="Factors",
        description="The factors of the integer.",
    )


class FactorizationNode(Node[IntegerData, FactorizationData, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Factorization",
        description="Factorizes an integer into a sequence of its factors.",
        version="0.4.0",
        parameter_type=Empty,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[IntegerData]:
        return IntegerData

    @classmethod
    @override
    def static_output_type(cls) -> Type[FactorizationData]:
        return FactorizationData

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[IntegerData],
        output_type: Type[FactorizationData],
        input: IntegerData,
    ) -> FactorizationData:
        value = input.value.root
        if value > 0:
            return output_type(
                factors=SequenceValue(
                    [IntegerValue(i) for i in range(1, value + 1) if value % i == 0]
                )
            )
        raise ValueError("Can only factorize positive integers")


class SubtractOutput(Data):
    difference: FloatValue = Field(
        title="Difference",
        description="The result of subtracting the second number from the first.",
    )


class SubtractNode(Node[BinaryFloatInput, SubtractOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Subtract",
        description="Subtracts one number from another.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[BinaryFloatInput]:
        return BinaryFloatInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[SubtractOutput]:
        return SubtractOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[BinaryFloatInput],
        output_type: Type[SubtractOutput],
        input: BinaryFloatInput,
    ) -> SubtractOutput:
        return output_type(difference=FloatValue(input.a.root - input.b.root))


class DivideOutput(Data):
    quotient: FloatValue = Field(
        title="Quotient",
        description="The result of dividing the first number by the second.",
    )


class DivideNode(Node[BinaryFloatInput, DivideOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Divide",
        description="Divides one number by another.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[BinaryFloatInput]:
        return BinaryFloatInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[DivideOutput]:
        return DivideOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[BinaryFloatInput],
        output_type: Type[DivideOutput],
        input: BinaryFloatInput,
    ) -> DivideOutput:
        _require_nonzero(input.b.root, node=self, label="divisor")
        return output_type(quotient=FloatValue(input.a.root / input.b.root))


class PowerOutput(Data):
    power: FloatValue = Field(
        title="Power",
        description="The base raised to the exponent.",
    )


class PowerNode(Node[PowerInput, PowerOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Power",
        description="Raises a base number to an exponent.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[PowerInput]:
        return PowerInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[PowerOutput]:
        return PowerOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[PowerInput],
        output_type: Type[PowerOutput],
        input: PowerInput,
    ) -> PowerOutput:
        return output_type(power=FloatValue(input.base.root**input.exponent.root))


class MultiplyOutput(Data):
    product: FloatValue = Field(
        title="Product",
        description="The product of the numbers.",
    )


class MultiplyNode(Node[UnionFloatInput, MultiplyOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Multiply",
        description="Multiplies one number or a sequence of numbers.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[UnionFloatInput]:
        return UnionFloatInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[MultiplyOutput]:
        return MultiplyOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[UnionFloatInput],
        output_type: Type[MultiplyOutput],
        input: UnionFloatInput,
    ) -> MultiplyOutput:
        values = _decimal_values_from_union(input.values)
        return output_type(product=FloatValue(prod(values, start=Decimal(1))))


class MinOutput(Data):
    minimum: FloatValue = Field(
        title="Minimum",
        description="The smallest number.",
    )


class MinNode(Node[UnionFloatInput, MinOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Min",
        description="Finds the smallest of one number or a sequence of numbers.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[UnionFloatInput]:
        return UnionFloatInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[MinOutput]:
        return MinOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[UnionFloatInput],
        output_type: Type[MinOutput],
        input: UnionFloatInput,
    ) -> MinOutput:
        values = _decimal_values_from_union(input.values)
        if not values:
            raise NodeException.for_user(
                "Cannot compute the minimum of an empty sequence.",
                node=self,
            )
        return output_type(minimum=FloatValue(min(values)))


class MaxOutput(Data):
    maximum: FloatValue = Field(
        title="Maximum",
        description="The largest number.",
    )


class MaxNode(Node[UnionFloatInput, MaxOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Max",
        description="Finds the largest of one number or a sequence of numbers.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[UnionFloatInput]:
        return UnionFloatInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[MaxOutput]:
        return MaxOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[UnionFloatInput],
        output_type: Type[MaxOutput],
        input: UnionFloatInput,
    ) -> MaxOutput:
        values = _decimal_values_from_union(input.values)
        if not values:
            raise NodeException.for_user(
                "Cannot compute the maximum of an empty sequence.",
                node=self,
            )
        return output_type(maximum=FloatValue(max(values)))


class NegateOutput(Data):
    negated: FloatValue = Field(
        title="Negated",
        description="The number with its sign flipped.",
    )


class NegateNode(Node[UnaryFloatInput, NegateOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Negate",
        description="Flips the sign of a number.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[UnaryFloatInput]:
        return UnaryFloatInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[NegateOutput]:
        return NegateOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[UnaryFloatInput],
        output_type: Type[NegateOutput],
        input: UnaryFloatInput,
    ) -> NegateOutput:
        return output_type(negated=FloatValue(-input.a.root))


class AbsOutput(Data):
    absolute: FloatValue = Field(
        title="Absolute Value",
        description="The number without its sign.",
    )


class AbsNode(Node[UnaryFloatInput, AbsOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Abs",
        description="Computes the absolute value of a number.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[UnaryFloatInput]:
        return UnaryFloatInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[AbsOutput]:
        return AbsOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[UnaryFloatInput],
        output_type: Type[AbsOutput],
        input: UnaryFloatInput,
    ) -> AbsOutput:
        return output_type(absolute=FloatValue(abs(input.a.root)))


class RoundParams(Params):
    ndigits: IntegerValue = Field(
        title="Decimal Places",
        description="The number of decimal places to round to.",
        default=IntegerValue(0),
    )
    rounding_mode: StringValue = Field(
        title="Rounding Mode",
        description=(
            "The rounding rule to apply: "
            '"half_even" for banker\'s rounding, "half_up" for half-up rounding.'
        ),
        default=StringValue("half_even"),
    )


class RoundOutput(Data):
    rounded: FloatValue = Field(
        title="Rounded",
        description="The rounded number.",
    )


class RoundNode(Node[UnaryFloatInput, RoundOutput, RoundParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Round",
        description="Rounds a number to a chosen number of decimal places.",
        version="1.0.0",
        parameter_type=RoundParams,
    )

    params: RoundParams = Field(default=RoundParams())  # pyright: ignore[reportIncompatibleVariableOverride]

    @classmethod
    @override
    def static_input_type(cls) -> Type[UnaryFloatInput]:
        return UnaryFloatInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[RoundOutput]:
        return RoundOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[UnaryFloatInput],
        output_type: Type[RoundOutput],
        input: UnaryFloatInput,
    ) -> RoundOutput:
        mode = self.params.rounding_mode.root
        if mode not in _ROUNDING_MODES:
            supported = ", ".join(sorted(_ROUNDING_MODES))
            raise NodeException.for_user(
                f"Unknown rounding mode {mode!r}; expected one of: {supported}.",
                node=self,
            )
        rounded = _round_decimal(
            input.a.root,
            ndigits=self.params.ndigits.root,
            mode=mode,
        )
        return output_type(rounded=FloatValue(rounded))


__all__ = [
    "AbsNode",
    "AddNode",
    "DivideNode",
    "FactorizationNode",
    "MaxNode",
    "MinNode",
    "MultiplyNode",
    "NegateNode",
    "PowerNode",
    "RoundNode",
    "SubtractNode",
    "SumNode",
]
