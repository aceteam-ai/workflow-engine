# workflow_engine/core/values/rounding.py
"""Rounding mode value type and decimal rounding helpers."""

from __future__ import annotations

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
from enum import StrEnum
from typing import TYPE_CHECKING

from .primitives import StringValue
from .value import Value

if TYPE_CHECKING:
    from ..context import ExecutionContext


class RoundingMode(StrEnum):
    """The rounding rule to apply.

    Directed modes: "down", "up", "toward_zero", "away_from_zero".
    Nearest modes: "half_even", "half_odd", "half_toward_zero",
    "half_away_from_zero", "half_up", "half_down".
    """

    DOWN = "down"
    UP = "up"
    TOWARD_ZERO = "toward_zero"
    AWAY_FROM_ZERO = "away_from_zero"
    HALF_TOWARD_ZERO = "half_toward_zero"
    HALF_AWAY_FROM_ZERO = "half_away_from_zero"
    HALF_EVEN = "half_even"
    HALF_ODD = "half_odd"
    HALF_UP = "half_up"
    HALF_DOWN = "half_down"

    def round(self, value: Decimal, *, digits: int) -> Decimal:
        quantizer = Decimal(10) ** -digits
        match self:
            case RoundingMode.DOWN:
                return value.quantize(quantizer, rounding=ROUND_FLOOR)
            case RoundingMode.UP:
                return value.quantize(quantizer, rounding=ROUND_CEILING)
            case RoundingMode.TOWARD_ZERO:
                return value.quantize(quantizer, rounding=ROUND_DOWN)
            case RoundingMode.AWAY_FROM_ZERO:
                return value.quantize(quantizer, rounding=ROUND_UP)
            case RoundingMode.HALF_TOWARD_ZERO:
                return value.quantize(quantizer, rounding=ROUND_HALF_DOWN)
            case RoundingMode.HALF_AWAY_FROM_ZERO:
                return value.quantize(quantizer, rounding=ROUND_HALF_UP)
            case RoundingMode.HALF_EVEN:
                return value.quantize(quantizer, rounding=ROUND_HALF_EVEN)
            case RoundingMode.HALF_UP:
                return self._round_half_up(value, quantizer)
            case RoundingMode.HALF_DOWN:
                return self._round_half_down(value, quantizer)
            case RoundingMode.HALF_ODD:
                return self._round_half_odd(value, quantizer)

    @staticmethod
    def _round_half_up(value: Decimal, quantizer: Decimal) -> Decimal:
        if value >= 0:
            return value.quantize(quantizer, rounding=ROUND_HALF_UP)
        return value.quantize(quantizer, rounding=ROUND_HALF_DOWN)

    @staticmethod
    def _round_half_down(value: Decimal, quantizer: Decimal) -> Decimal:
        if value >= 0:
            return value.quantize(quantizer, rounding=ROUND_HALF_DOWN)
        return value.quantize(quantizer, rounding=ROUND_HALF_UP)

    @staticmethod
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


class RoundingModeValue(Value[RoundingMode]):
    pass


@StringValue.register_cast_to(RoundingModeValue)
def cast_string_to_rounding_mode(
    value: StringValue,
    context: ExecutionContext,
) -> RoundingModeValue:
    return RoundingModeValue(RoundingMode(value.root))


__all__ = (
    "RoundingMode",
    "RoundingModeValue",
)
