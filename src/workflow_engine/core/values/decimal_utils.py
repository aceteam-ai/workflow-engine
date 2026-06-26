# workflow_engine/core/values/decimal_utils.py
"""Helpers for exact decimal numeric values."""

from decimal import Decimal


def to_decimal(value: int | float | Decimal | str) -> Decimal:
    """
    Convert a numeric value to Decimal without unnecessary float round-trips.

    Integers and Decimals pass through exactly. Floats and strings are converted
    via ``Decimal(str(value))``, matching Pydantic's Decimal coercion and
    preserving common decimal literals like ``0.1`` when sourced from JSON text.

    Do not replace float handling with bare ``Decimal(value)``: that encodes the
    float's exact IEEE-754 bits (e.g. ``Decimal(3.14)`` →
    ``3.1400000000000001243...``), while ``str(3.14)`` is the shortest decimal
    that round-trips back to the same float, so ``Decimal(str(3.14))`` →
    ``3.14`` — which is what ``FloatValue(3.14)`` stores and what ``__eq__``
    against ``3.14`` must match.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, str):
        return Decimal(value)
    raise ValueError(f"Expected int, float, Decimal, or str, got {type(value)}")


__all__ = ("to_decimal",)
