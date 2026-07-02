# workflow_engine/core/values/union.py
"""
Union Value types.

``UnionValue[A, B, ...]`` accepts any of several member types. Validated and
cast values are always an instance of one member (``FloatValue`` or
``SequenceValue[FloatValue]``), never a wrapper object. Use
``isinstance(x, FloatValue)`` / ``isinstance(x, SequenceValue)`` in node code.

The public ``UnionValue`` helper wraps an internal ``_UnionType`` in
``Annotated`` so pyright accepts concrete members (and raw Python coercions) at
``Data`` construction time. For optional fields use ``OptionalValue[T]``
(shorthand for ``UnionValue[T, NullValue]``).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    TypeVar,
    get_args,
    get_origin,
)

from pydantic import GetCoreSchemaHandler
from pydantic.fields import FieldInfo
from pydantic_core import core_schema

from ...utils.asynchronous import is_coroutine
from .value import Caster, Value, ValueType, get_value_type_key

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from .schema import ValueSchema

_T = TypeVar("_T", bound=Value)

# Reuse identical unions built from schema round-trips or repeated subscripts.
_UNION_TYPE_CACHE: dict[tuple[tuple[str, tuple], ...], type[Value]] = {}


def get_union_members(value_type: ValueType) -> tuple[ValueType, ...] | None:
    """Return member types if *value_type* is a union, else None."""
    members: tuple[ValueType, ...] | None = getattr(value_type, "_union_members_", None)
    if members:
        return members
    return None


def union_value_type(*members: ValueType) -> ValueType:
    """Build or reuse an internal union type accepting any of *members*."""
    if not members:
        raise TypeError("UnionValue requires at least one member type")
    for member in members:
        if not issubclass(member, Value):
            raise TypeError(f"Union member {member!r} must be a Value type")

    key = tuple(get_value_type_key(member) for member in members)
    cached = _UNION_TYPE_CACHE.get(key)
    if cached is not None:
        return cached

    member_names = ", ".join(member.__name__ for member in members)
    union_cls = type(
        f"UnionValue[{member_names}]",
        (_UnionType,),
        {
            "_union_members_": members,
            "__module__": _UnionType.__module__,
        },
        register=False,
    )
    _UNION_TYPE_CACHE[key] = union_cls
    return union_cls


class _UnionType(Value[Any], register=False):
    """
    Internal union type accepting any one of several Value types.

    Use the public ``UnionValue`` helper on ``Data`` fields instead of
    referencing this class directly.
    """

    _union_members_: ClassVar[tuple[ValueType, ...]] = ()

    @classmethod
    def __class_getitem__(cls, members: Any) -> ValueType:
        if not isinstance(members, tuple):
            members = (members,)
        return union_value_type(*members)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        members = get_union_members(source_type)
        if not members:
            return core_schema.any_schema()
        return core_schema.union_schema(
            [handler.generate_schema(member) for member in members]
        )

    @classmethod
    def to_value_schema(cls) -> ValueSchema:
        from .schema import UnionValueSchema

        members = get_union_members(cls)
        if not members:
            return super().to_value_schema()
        return UnionValueSchema(
            anyOf=[member.to_value_schema() for member in members],
        )


class _UnionMarker:
    """Pydantic metadata binding a construction-time union to its runtime type."""

    __slots__ = ("union_type",)

    def __init__(self, union_type: ValueType) -> None:
        self.union_type = union_type

    def __get_pydantic_core_schema__(
        self,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return self.union_type.__get_pydantic_core_schema__(self.union_type, handler)

    def __repr__(self) -> str:
        members = get_union_members(self.union_type)
        if members:
            names = ", ".join(member.__name__ for member in members)
            return f"_UnionMarker(UnionValue[{names}])"
        return f"_UnionMarker({self.union_type!r})"


def _member_construction_types(member: ValueType) -> tuple[type[Any], ...]:
    """Return Value and raw Python types accepted at Data field construction."""
    from .datetime_value import DateValue
    from .primitives import (
        BooleanValue,
        FloatValue,
        IntegerValue,
        NullValue,
        StringValue,
    )

    if member is NullValue:
        return (NullValue, type(None))
    if member is BooleanValue:
        return (BooleanValue, bool)
    if member is IntegerValue:
        return (IntegerValue, int)
    if member is FloatValue:
        return (FloatValue, int, float)
    if member is StringValue:
        return (StringValue, str)
    if member is DateValue:
        return (DateValue, datetime, Decimal, int, float, str)
    return (member,)


def _construction_union(*members: ValueType) -> Any:
    construction_type: Any | None = None
    for member in members:
        for candidate in _member_construction_types(member):
            construction_type = (
                candidate
                if construction_type is None
                else construction_type | candidate
            )
    if construction_type is None:
        raise TypeError("UnionValue requires at least one member type")
    return construction_type


def _union_value(*members: ValueType) -> Any:
    union_type = union_value_type(*members)
    return Annotated[_construction_union(*members), _UnionMarker(union_type)]


class _UnionValueFactory:
    """Build union annotations via call or subscript syntax."""

    def __call__(self, *members: ValueType) -> Any:
        return _union_value(*members)

    def __getitem__(self, members: ValueType | tuple[ValueType, ...]) -> Any:
        if not isinstance(members, tuple):
            members = (members,)
        return _union_value(*members)


UnionValue = _UnionValueFactory()


class _OptionalValueFactory:
    """Build optional union annotations: ``member | NullValue``."""

    def __call__(self, member: type[_T]) -> Any:
        return _optional_value(member)

    def __getitem__(self, member: type[_T]) -> Any:
        return _optional_value(member)


def _optional_value(member: type[_T]) -> Any:
    from .primitives import NullValue

    return _union_value(member, NullValue)


OptionalValue = _OptionalValueFactory()


def resolve_union_type(
    annotation: Any,
    *,
    metadata: Iterable[Any] = (),
) -> ValueType:
    """
    Resolve a ``Data`` field annotation to its runtime union ``Value`` type.

    Accepts internal union subclasses and ``Annotated[..., _UnionMarker(...)]``
    from ``UnionValue`` / ``OptionalValue``.
    """
    for item in metadata:
        if isinstance(item, _UnionMarker):
            return item.union_type

    origin = get_origin(annotation)
    if origin is Annotated:
        for item in get_args(annotation)[1:]:
            if isinstance(item, _UnionMarker):
                return item.union_type
        raise TypeError(
            "Annotated Data field must include union metadata "
            f"(use UnionValue or OptionalValue), got {annotation!r}"
        )

    if isinstance(annotation, type) and issubclass(annotation, Value):
        return annotation

    raise TypeError(f"Field annotation is not a Value type: {annotation!r}")


def resolve_union_type_from_field(field_info: FieldInfo) -> ValueType:
    """Resolve a ``Data`` model field to its runtime union ``Value`` type."""
    assert field_info.annotation is not None
    return resolve_union_type(field_info.annotation, metadata=field_info.metadata)


SourceType = Value
TargetType = Value


@Value.register_generic_cast_to(_UnionType)  # pyright: ignore[reportArgumentType]
def cast_to_union(
    source_type: type[SourceType],
    target_type: type[_UnionType],
) -> Caster[SourceType, TargetType] | None:
    members = get_union_members(target_type)
    if not members:
        return None
    if not any(source_type.get_caster(member) is not None for member in members):
        return None

    async def _cast(
        value: SourceType,
        context: ExecutionContext,
    ) -> TargetType:
        for member in members:
            if isinstance(value, member):
                return value  # type: ignore[return-value]
        for member in members:
            caster = type(value).get_caster(member)
            if caster is not None:
                result = caster(value, context)
                casted = (await result) if is_coroutine(result) else result  # pyright: ignore[reportGeneralTypeIssues]
                return casted  # type: ignore[return-value]
        raise ValueError(f"Cannot convert {value} to {target_type}")

    return _cast


__all__ = [
    "OptionalValue",
    "UnionValue",
]
