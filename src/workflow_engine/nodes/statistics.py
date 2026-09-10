"""Decimal-preserving summary statistics with explicit sampling and interpolation."""

from abc import abstractmethod
from decimal import ROUND_HALF_EVEN, Decimal
from enum import StrEnum
from statistics import StatisticsError, median, mode, pstdev, pvariance, stdev, variance
from typing import ClassVar, Generic, Self, TypeVar

from overrides import override
from pydantic import Field, model_validator

from ..core import (
    BooleanValue,
    Data,
    Empty,
    ErrorClass,
    ExecutionContext,
    FloatValue,
    Node,
    NodeException,
    NodeTypeInfo,
    Params,
    SequenceValue,
    Value,
)


class StatisticsInput(Data):
    values: SequenceValue[FloatValue] = Field(
        title="Values", description="The numeric values to summarize."
    )


class StatisticsOutput(Data):
    value: FloatValue = Field(title="Value", description="The computed statistic.")


P = TypeVar("P", bound=Params, covariant=True)


class _StatisticsNode(Node[StatisticsInput, StatisticsOutput, P], Generic[P]):
    @classmethod
    @override
    def static_input_type(cls) -> type[StatisticsInput]:
        return StatisticsInput

    @classmethod
    @override
    def static_output_type(cls) -> type[StatisticsOutput]:
        return StatisticsOutput

    @abstractmethod
    def compute(self, values: list[Decimal]) -> Decimal:
        raise NotImplementedError

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[StatisticsInput],
        output_type: type[StatisticsOutput],
        input: StatisticsInput,
    ) -> StatisticsOutput:
        values = [item.root for item in input.values]
        if not values:
            raise NodeException.for_user(
                f"Cannot compute {self.TYPE_INFO.display_name.lower()} of an empty sequence.",
                node=self,
                error_class=ErrorClass.VALIDATION,
            )
        try:
            result = self.compute(values)
        except StatisticsError as error:
            raise NodeException.for_user(
                f"Cannot compute {self.TYPE_INFO.display_name.lower()}: {error}.",
                node=self,
                error_class=ErrorClass.VALIDATION,
            ) from error
        return output_type(value=FloatValue(result))


class MedianNode(_StatisticsNode[Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Median",
        description="Finds the middle value, averaging the middle pair when needed.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @override
    def compute(self, values: list[Decimal]) -> Decimal:
        return median(values)


class ModeNode(_StatisticsNode[Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Mode",
        description="Finds the most frequent value, taking the first encountered value when tied.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @override
    def compute(self, values: list[Decimal]) -> Decimal:
        return mode(values)


class PopulationParams(Params):
    population: BooleanValue = Field(
        default=BooleanValue(False),
        title="Population",
        description="The choice to treat the values as an entire population. False uses the sample estimate and requires at least two values.",
    )


class VarianceNode(_StatisticsNode[PopulationParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Variance",
        description="Computes sample variance by default, or population variance when selected.",
        version="1.0.0",
        parameter_type=PopulationParams,
    )

    @override
    def compute(self, values: list[Decimal]) -> Decimal:
        return pvariance(values) if self.params.population.root else variance(values)


class StandardDeviationNode(_StatisticsNode[PopulationParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Standard Deviation",
        description="Computes sample standard deviation by default, or population standard deviation when selected.",
        version="1.0.0",
        parameter_type=PopulationParams,
    )

    @override
    def compute(self, values: list[Decimal]) -> Decimal:
        return pstdev(values) if self.params.population.root else stdev(values)


class RangeNode(_StatisticsNode[Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Range",
        description="Computes the difference between the largest and smallest values.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @override
    def compute(self, values: list[Decimal]) -> Decimal:
        return max(values) - min(values)


class QuantileInterpolation(StrEnum):
    LINEAR = "linear"
    LOWER = "lower"
    HIGHER = "higher"
    NEAREST = "nearest"
    MIDPOINT = "midpoint"


class QuantileInterpolationValue(Value[QuantileInterpolation]):
    pass


class _QuantileParams(Params):
    interpolation: QuantileInterpolationValue = Field(
        default=QuantileInterpolationValue(QuantileInterpolation.LINEAR),
        title="Interpolation",
        description="The method used between adjacent sorted values. Linear blends them by the fractional position; nearest uses the even index on a tie.",
    )


class PercentileParams(_QuantileParams):
    q: FloatValue = Field(
        title="Percentile",
        description="The percentile from 0 to 100, including both endpoints.",
        json_schema_extra={"minimum": 0, "maximum": 100},
    )

    @model_validator(mode="after")
    def validate_percentile(self) -> Self:
        if not 0 <= self.q.root <= 100:
            raise ValueError("Percentile q must be between 0 and 100.")
        return self


class QuantileParams(_QuantileParams):
    q: FloatValue = Field(
        title="Quantile",
        description="The quantile from 0 to 1, including both endpoints.",
        json_schema_extra={"minimum": 0, "maximum": 1},
    )

    @model_validator(mode="after")
    def validate_quantile(self) -> Self:
        if not 0 <= self.q.root <= 1:
            raise ValueError("Quantile q must be between 0 and 1.")
        return self


def _quantile(
    values: list[Decimal], q: Decimal, interpolation: QuantileInterpolation
) -> Decimal:
    """Inclusive position (n - 1) * q, computed without a float conversion."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    fraction = position - lower
    if fraction == 0:
        return ordered[lower]
    upper = lower + 1
    match interpolation:
        case QuantileInterpolation.LOWER:
            return ordered[lower]
        case QuantileInterpolation.HIGHER:
            return ordered[upper]
        case QuantileInterpolation.NEAREST:
            return ordered[int(position.to_integral_value(rounding=ROUND_HALF_EVEN))]
        case QuantileInterpolation.MIDPOINT:
            return (ordered[lower] + ordered[upper]) / 2
        case QuantileInterpolation.LINEAR:
            return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class PercentileNode(_StatisticsNode[PercentileParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Percentile",
        description="Computes a percentile from 0 to 100 using explicit interpolation.",
        version="1.0.0",
        parameter_type=PercentileParams,
    )

    @override
    def compute(self, values: list[Decimal]) -> Decimal:
        return _quantile(
            values, self.params.q.root / 100, self.params.interpolation.root
        )


class QuantileNode(_StatisticsNode[QuantileParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Quantile",
        description="Computes a quantile from 0 to 1 using explicit interpolation.",
        version="1.0.0",
        parameter_type=QuantileParams,
    )

    @override
    def compute(self, values: list[Decimal]) -> Decimal:
        return _quantile(values, self.params.q.root, self.params.interpolation.root)


__all__ = [
    "MedianNode",
    "ModeNode",
    "PercentileNode",
    "QuantileInterpolation",
    "QuantileInterpolationValue",
    "QuantileNode",
    "RangeNode",
    "StandardDeviationNode",
    "VarianceNode",
]
