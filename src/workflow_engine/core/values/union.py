# workflow_engine/core/values/union.py
"""
Type-level union ports for the Value system.

UnionValue is annotation-only: it describes a port that accepts any of several
member types, but validated and cast values are always an instance of one member
(``FloatValue`` or ``SequenceValue[FloatValue]``), never a ``UnionValue`` wrapper.
Use ``isinstance(x, FloatValue)`` / ``isinstance(x, SequenceValue)`` in node code,
not ``isinstance(x, UnionValue)``.

UnionValue is intentionally unlike ``SequenceValue[T]`` and other generic Value
types. Those wrap runtime data in a ``root`` field and parameterize Pydantic's
single RootModel generic slot (``Value[Sequence[T]]``). A union port has no
``root`` — it only constrains which concrete types a field may hold.

That combination — variadic arity (``anyOf`` can have N members) and no ``root``
— cannot be expressed as ``Generic[Unpack[Ts]]`` on ``Value[Any]``. Pydantic's
``RootModel.__class_getitem__`` accepts only one type argument (the root), so
``UnionValue[FloatValue, SequenceValue[FloatValue]]`` raises "Too many arguments"
if we try TypeVarTuple like a normal generic.

Instead, ``UnionValue[A, B, ...]`` builds (or reuses) a cached dynamic subclass
that stores members on ``_union_members_``. ``get_origin_and_args`` in value.py
recognizes that attribute and normalizes to ``(UnionValue, (A, B, ...))`` so the
existing casting and type-key machinery keeps working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from ...utils.asynchronous import is_coroutine
from .value import Caster, Value, ValueType, get_value_type_key

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from .schema import ValueSchema

# Reuse identical unions built from schema round-trips or repeated subscripts.
_UNION_TYPE_CACHE: dict[tuple[tuple[str, tuple], ...], type[Value]] = {}


def get_union_members(value_type: ValueType) -> tuple[ValueType, ...] | None:
    """Return member types if *value_type* is a UnionValue, else None."""
    members: tuple[ValueType, ...] | None = getattr(value_type, "_union_members_", None)
    if members:
        return members
    return None


def union_value_type(*members: ValueType) -> ValueType:
    """Build or reuse a UnionValue type accepting any of *members*."""
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
    # Dynamic subclass rather than a Pydantic generic specialization — see module
    # docstring. register=False keeps these out of the ValueRegistry.
    union_cls = type(
        f"UnionValue[{member_names}]",
        (UnionValue,),
        {
            "_union_members_": members,
            "__module__": UnionValue.__module__,
        },
        register=False,
    )
    _UNION_TYPE_CACHE[key] = union_cls
    return union_cls


class UnionValue(Value[Any], register=False):
    """
    A port type accepting any one of several Value types.

    Annotation-only: when used as a ``Data`` field type, validation and casting
    produce an instance of one member (``A`` or ``B``), not a ``UnionValue`` instance.

    Construct parameterized unions with ``UnionValue[A, B, ...]``.
    """

    # Set on dynamic subclasses built by union_value_type(); empty on the base.
    _union_members_: ClassVar[tuple[ValueType, ...]] = ()

    @classmethod
    def __class_getitem__(cls, members: Any) -> ValueType:
        # Bypass Pydantic RootModel.__class_getitem__ — it only supports one arg.
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
            # Bare UnionValue (unparameterized) — valid as a type form, not as data.
            return core_schema.any_schema()
        # Validate field values against any member; stored value is the concrete member.
        return core_schema.union_schema(
            [handler.generate_schema(member) for member in members]
        )

    @classmethod
    def to_value_schema(cls) -> ValueSchema:
        from .schema import UnionValueSchema

        members = get_union_members(cls)
        if not members:
            return super().to_value_schema()
        # No root field to derive a schema from — assemble anyOf from members.
        return UnionValueSchema(
            anyOf=[member.to_value_schema() for member in members],
        )


SourceType = Value
TargetType = Value


# Registered on Value so every source type inherits it; get_caster looks up by
# target origin name "UnionValue" (see get_origin_and_args in value.py).
@Value.register_generic_cast_to(UnionValue)  # pyright: ignore[reportArgumentType]
def cast_to_union(
    source_type: type[SourceType],
    target_type: type[UnionValue],
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
        # Casting to a union yields a concrete member, not a UnionValue wrapper.
        # Precedence: exact member match first, then first castable member in order.
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
    "UnionValue",
    "get_union_members",
    "union_value_type",
]
