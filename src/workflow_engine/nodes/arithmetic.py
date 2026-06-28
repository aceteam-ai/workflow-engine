# workflow_engine/nodes/arithmetic.py
"""
Built-in arithmetic nodes.
"""

from decimal import (
    ROUND_CEILING,
    ROUND_DOWN,
    ROUND_FLOOR,
    ROUND_HALF_DOWN,
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    ROUND_UP,
    Decimal,
)
from math import prod
from typing import ClassVar, Type

from overrides import override
from pydantic import Field

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
from ..core.values import build_data_type


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

_DECIMAL_ROUNDING_MODES = {
    "floor": ROUND_FLOOR,
    "ceiling": ROUND_CEILING,
    "toward_zero": ROUND_DOWN,
    "away_from_zero": ROUND_UP,
    "half_toward_zero": ROUND_HALF_DOWN,
    "half_away_from_zero": ROUND_HALF_UP,
    "half_even": ROUND_HALF_EVEN,
}

_CUSTOM_ROUNDING_MODES = frozenset(
    {
        "half_toward_negative_infinity",
        "half_toward_positive_infinity",
        "half_odd",
    }
)

_SUPPORTED_ROUNDING_MODES = frozenset(
    _DECIMAL_ROUNDING_MODES.keys() | _CUSTOM_ROUNDING_MODES
)

_ROUNDING_MODE_DESCRIPTION = (
    "The rounding rule to apply. Directed modes: "
    '"floor", "ceiling", "toward_zero", "away_from_zero". '
    "Nearest modes: "
    '"half_even", "half_odd", "half_toward_zero", "half_away_from_zero", '
    '"half_toward_positive_infinity", "half_toward_negative_infinity".'
)


def _rounding_quantizer(digits: int) -> Decimal:
    return Decimal(10) ** -digits


def _round_half_toward_positive_infinity(value: Decimal, quantizer: Decimal) -> Decimal:
    if value >= 0:
        return value.quantize(quantizer, rounding=ROUND_HALF_UP)
    return value.quantize(quantizer, rounding=ROUND_HALF_DOWN)


def _round_half_toward_negative_infinity(value: Decimal, quantizer: Decimal) -> Decimal:
    if value >= 0:
        return value.quantize(quantizer, rounding=ROUND_HALF_DOWN)
    return value.quantize(quantizer, rounding=ROUND_HALF_UP)


def _round_half_odd(value: Decimal, quantizer: Decimal) -> Decimal:
    rounded_up = value.quantize(quantizer, rounding=ROUND_HALF_UP)
    rounded_down = value.quantize(quantizer, rounding=ROUND_HALF_DOWN)
    if rounded_up == rounded_down:
        return rounded_up
    for candidate in (rounded_down, rounded_up):
        scaled = candidate / quantizer
        if int(scaled) % 2 != 0:
            return candidate
    return rounded_up


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


def _round_decimal(value: Decimal, *, digits: int, mode: str) -> Decimal:
    quantizer = _rounding_quantizer(digits)

    if mode in _DECIMAL_ROUNDING_MODES:
        return value.quantize(
            quantizer,
            rounding=_DECIMAL_ROUNDING_MODES[mode],
        )

    if mode == "half_toward_positive_infinity":
        return _round_half_toward_positive_infinity(value, quantizer)
    if mode == "half_toward_negative_infinity":
        return _round_half_toward_negative_infinity(value, quantizer)
    if mode == "half_odd":
        return _round_half_odd(value, quantizer)

    supported = ", ".join(sorted(_SUPPORTED_ROUNDING_MODES))
    raise ValueError(f"Unknown rounding mode {mode!r}; expected one of: {supported}.")


def _validate_rounding_mode(mode: str, *, node: Node) -> None:
    if mode not in _SUPPORTED_ROUNDING_MODES:
        supported = ", ".join(sorted(_SUPPORTED_ROUNDING_MODES))
        raise NodeException.for_user(
            f"Unknown rounding mode {mode!r}; expected one of: {supported}.",
            node=node,
        )


def _divide_with_remainder(
    dividend: Decimal,
    divisor: Decimal,
    *,
    mode: str,
) -> tuple[Decimal, Decimal, Decimal]:
    quotient = dividend / divisor
    integer_quotient = _round_decimal(quotient, digits=0, mode=mode)
    remainder = dividend - integer_quotient * divisor
    return quotient, integer_quotient, remainder


class RoundingParams(Params):
    digits: IntegerValue = Field(
        title="Decimal Places",
        description="The number of decimal places to round to.",
        default=IntegerValue(0),
    )
    rounding_mode: StringValue = Field(
        title="Rounding Mode",
        description=_ROUNDING_MODE_DESCRIPTION,
        default=StringValue("half_even"),
    )


class DivideParams(Params):
    rounding_mode: StringValue = Field(
        title="Rounding Mode",
        description=(
            "The rounding rule used to compute the integer quotient from the exact "
            f"quotient. {_ROUNDING_MODE_DESCRIPTION}"
        ),
        default=StringValue("floor"),
    )


class SubtractInput(Data):
    minuend: FloatValue = Field(
        title="Minuend",
        description="The number to subtract from.",
    )
    subtrahend: FloatValue = Field(
        title="Subtrahend",
        description="The number to subtract.",
    )


class DivideInput(Data):
    dividend: FloatValue = Field(
        title="Dividend",
        description="The number to be divided.",
    )
    divisor: FloatValue = Field(
        title="Divisor",
        description="The number to divide by.",
    )


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
    params: AddNodeParams = Field(default=AddNodeParams())

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
        description="The result of subtracting the subtrahend from the minuend.",
    )


class SubtractNode(Node[SubtractInput, SubtractOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Subtract",
        description="Subtracts one number from another.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[SubtractInput]:
        return SubtractInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[SubtractOutput]:
        return SubtractOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[SubtractInput],
        output_type: Type[SubtractOutput],
        input: SubtractInput,
    ) -> SubtractOutput:
        return output_type(
            difference=FloatValue(input.minuend.root - input.subtrahend.root)
        )


class DivideOutput(Data):
    quotient: FloatValue = Field(
        title="Quotient",
        description="The exact result of dividing the dividend by the divisor.",
    )
    integer_quotient: FloatValue = Field(
        title="Integer Quotient",
        description=(
            "The quotient rounded to a whole number using the chosen rounding mode."
        ),
    )
    remainder: FloatValue = Field(
        title="Remainder",
        description=(
            "The amount left over after subtracting the integer quotient times "
            "the divisor from the dividend."
        ),
    )


class DivideNode(Node[DivideInput, DivideOutput, DivideParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Divide",
        description=(
            "Divides one number by another, returning the exact quotient, an "
            "integer quotient, and the remainder relative to that integer quotient."
        ),
        version="1.0.0",
        parameter_type=DivideParams,
    )

    params: DivideParams = Field(default=DivideParams())

    @classmethod
    @override
    def static_input_type(cls) -> Type[DivideInput]:
        return DivideInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[DivideOutput]:
        return DivideOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[DivideInput],
        output_type: Type[DivideOutput],
        input: DivideInput,
    ) -> DivideOutput:
        _require_nonzero(input.divisor.root, node=self, label="divisor")
        mode = self.params.rounding_mode.root
        _validate_rounding_mode(mode, node=self)
        quotient, integer_quotient, remainder = _divide_with_remainder(
            input.dividend.root,
            input.divisor.root,
            mode=mode,
        )
        return output_type(
            quotient=FloatValue(quotient),
            integer_quotient=FloatValue(integer_quotient),
            remainder=FloatValue(remainder),
        )


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


class MinimumOutput(Data):
    minimum: FloatValue = Field(
        title="Minimum",
        description="The smallest number.",
    )


class MinimumNode(Node[UnionFloatInput, MinimumOutput, Empty]):
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
    def static_output_type(cls) -> Type[MinimumOutput]:
        return MinimumOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[UnionFloatInput],
        output_type: Type[MinimumOutput],
        input: UnionFloatInput,
    ) -> MinimumOutput:
        values = _decimal_values_from_union(input.values)
        if not values:
            raise NodeException.for_user(
                "Cannot compute the minimum of an empty sequence.",
                node=self,
            )
        return output_type(minimum=FloatValue(min(values)))


class MaximumOutput(Data):
    maximum: FloatValue = Field(
        title="Maximum",
        description="The largest number.",
    )


class MaximumNode(Node[UnionFloatInput, MaximumOutput, Empty]):
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
    def static_output_type(cls) -> Type[MaximumOutput]:
        return MaximumOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[UnionFloatInput],
        output_type: Type[MaximumOutput],
        input: UnionFloatInput,
    ) -> MaximumOutput:
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


class AbsoluteValueOutput(Data):
    absolute: FloatValue = Field(
        title="Absolute Value",
        description="The number without its sign.",
    )


class AbsoluteValueNode(Node[UnaryFloatInput, AbsoluteValueOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Absolute Value",
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
    def static_output_type(cls) -> Type[AbsoluteValueOutput]:
        return AbsoluteValueOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[UnaryFloatInput],
        output_type: Type[AbsoluteValueOutput],
        input: UnaryFloatInput,
    ) -> AbsoluteValueOutput:
        return output_type(absolute=FloatValue(abs(input.a.root)))


class RoundParams(RoundingParams):
    digits: IntegerValue = Field(
        title="Decimal Places",
        description="The number of decimal places to round to.",
        default=IntegerValue(0),
    )
    rounding_mode: StringValue = Field(
        title="Rounding Mode",
        description=_ROUNDING_MODE_DESCRIPTION,
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

    params: RoundParams = Field(default=RoundParams())

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
        _validate_rounding_mode(mode, node=self)
        rounded = _round_decimal(
            input.a.root,
            digits=self.params.digits.root,
            mode=mode,
        )
        return output_type(rounded=FloatValue(rounded))


__all__ = [
    "AbsoluteValueNode",
    "AddNode",
    "DivideNode",
    "FactorizationNode",
    "MaximumNode",
    "MinimumNode",
    "MultiplyNode",
    "NegateNode",
    "PowerNode",
    "RoundNode",
    "SubtractNode",
    "SumNode",
]
