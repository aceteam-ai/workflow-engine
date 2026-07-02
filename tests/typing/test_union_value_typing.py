# tests/typing/test_union_value_typing.py
"""
Pyright-checked examples for UnionValue / OptionalValue construction.

These functions are not executed in pytest; they exist so ``pyright`` verifies
that common node-author patterns type-check without ``cast()``.
"""

from __future__ import annotations

from workflow_engine.core import (
    Data,
    IntegerValue,
    NullValue,
    StringValue,
)
from workflow_engine.core.values import OptionalValue

OptionalInteger = OptionalValue[IntegerValue]
OptionalString = OptionalValue[StringValue]


class MessageItem(Data):
    sender_id: OptionalInteger
    text: OptionalString


def build_message_from_members() -> MessageItem:
    return MessageItem(
        sender_id=IntegerValue(42),
        text=StringValue("hello"),
    )


def build_message_with_null_values() -> MessageItem:
    return MessageItem(
        sender_id=NullValue(None),
        text=NullValue(None),
    )
