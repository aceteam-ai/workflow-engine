# workflow_engine/core/values/value.py
import inspect
import re
from functools import cached_property
from hashlib import md5
from logging import getLogger
from typing import (
    TYPE_CHECKING,
    Awaitable,
    ClassVar,
    Generic,
    Protocol,
    Self,
    Type,
    TypeVar,
    get_origin,
)

from pydantic import PrivateAttr

from ...utils.immutable import ImmutableRootModel

if TYPE_CHECKING:
    from ..context import Context
    from .schema import ValueSchema


logger = getLogger(__name__)


T = TypeVar("T")
V = TypeVar("V", bound="Value")
type ValueType = Type[Value]


def get_origin_and_args(t: ValueType) -> tuple[ValueType, tuple[ValueType, ...]]:
    """
    For a non-generic value type NonGenericValue, returns (NonGenericValue, ()).

    For a generic value type GenericValue[Argument1Value, Argument2Value, ...],
    returns (GenericValue, (Argument1Value, Argument2Value, ...)).
    All arguments must themselves be Value subclasses.
    """
    # Pydantic RootModels don't play nice with get_origin and get_args, so we
    # get the root type directly from the model fields.
    assert issubclass(t, Value)
    info = t.__pydantic_generic_metadata__
    origin = info["origin"]
    args = info["args"]
    if origin is None:
        assert len(args) == 0
        return t, ()
    else:
        assert issubclass(origin, Value)
        assert len(args) > 0
        return origin, tuple(args)


type ValueTypeKey = tuple[str, tuple[ValueTypeKey, ...]]


def get_value_type_key(t: ValueType) -> ValueTypeKey:
    """
    Get a unique hashable key for a Value type.
    If t is a generic type, recursively call `get_value_type_key` to expand any
    Value types in the args.
    """
    origin, args = get_origin_and_args(t)
    return origin.__name__, tuple(
        get_value_type_key(arg) if issubclass(arg, Value) else arg for arg in args
    )


SourceType = TypeVar("SourceType", bound="Value")
TargetType = TypeVar("TargetType", bound="Value")


class Caster(Protocol, Generic[SourceType, TargetType]):  # type: ignore
    """
    A caster is a contextual function that transforms the type of a Value.

    May be either a sync or async function.
    """

    def __call__(
        self,
        value: SourceType,
        context: "Context",
    ) -> TargetType | Awaitable[TargetType]: ...


class GenericCaster(Protocol, Generic[SourceType, TargetType]):  # type: ignore
    """
    A generic caster is a contextual function that takes a source type and a
    target type, and outputs a caster between the two types, or None if the cast
    is not possible.
    This is an advanced feature intended for use on generic types.

    The purpose of this two-step approach is to explicitly allow or deny type
    casts before the exact type of the value is known. This is necessary because
    the type of a Value is not known until the Value is created.
    """

    def __call__(
        self,
        source_type: Type[SourceType],
        target_type: Type[TargetType],
    ) -> Caster[SourceType, TargetType] | None: ...


generic_pattern = re.compile(r"^[a-zA-Z]\w+\[.*\]$")


class Value(ImmutableRootModel[T], Generic[T]):
    """
    Wraps an arbitrary read-only value which can be passed as input to a node.

    Each Value subclass defines a specific type (possibly generic) of value.
    After defining the subclass, you can register Caster functions to convert
    other Value classes to that type, using the register_cast_to decorator.
    Casts are registered in any order.

    Each Value subclass inherits its parent classes' Casters.
    To avoid expanding the type tree every time, we cache the Casters at each
    class the first time a cast is used.
    Once that cache is created, the casts are locked and can no longer be
    changed.
    """

    # these properties force us to implement __eq__ and __hash__ to ignore them
    _casters: ClassVar[dict[str, GenericCaster]] = {}
    _resolved_casters: ClassVar[dict[str, GenericCaster] | None] = None
    _cast_cache: dict[ValueTypeKey, "Value"] = PrivateAttr(
        default_factory=dict,
    )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # reinitialize for each subclass so it doesn't just reference the parent
        cls._casters = {}
        cls._resolved_casters = None

        # NOTE: something about this hack does not work when using
        # `from __future__ import annotations`.
        while generic_pattern.match(cls.__name__) is not None:
            assert cls.__base__ is not None
            cls = cls.__base__
        if get_origin(cls) is None:
            value_type_registry.register(cls.__name__, cls)

    @classmethod
    def _get_casters(cls) -> dict[str, GenericCaster]:
        """
        Get all type casting functions for this class, including those inherited
        from parent classes.
        This inherits from all parents classes, though they will be overridden
        if the child class has its own casting function for the same type.
        """
        if cls._resolved_casters is not None:
            return cls._resolved_casters

        resolved_casters: dict[str, GenericCaster] = cls._casters.copy()

        # Add converters from all classes in MRO order
        # (starting from this class, then parents, then parents of parents, ...)
        for parent in cls.__bases__:
            if issubclass(parent, Value):
                parent_casters = parent._get_casters()
            else:
                continue

            for origin, caster in parent_casters.items():
                # converters in the child class override those in the parent
                # class
                if origin not in resolved_casters:
                    resolved_casters[origin] = caster

        cls._resolved_casters = resolved_casters
        return resolved_casters

    @classmethod
    def register_cast_to(cls, t: Type[V]):
        """
        A decorator to register a possible type cast from this class to the
        class T, neither of which are generic.
        """

        def wrap(caster: Caster[Self, V]):
            cls.register_generic_cast_to(t)(lambda source_type, target_type: caster)
            return caster

        return wrap

    @classmethod
    def register_generic_cast_to(cls, t: Type[V]):
        """
        A decorator to register a possible type cast from this class to the
        class T, either of which may be generic.
        """

        def wrap(caster: GenericCaster[Self, V]):
            if cls._resolved_casters is not None:
                raise RuntimeError(
                    f"Cannot add casters for {cls.__name__} after it has been used to cast values"
                )

            target_origin, _ = get_origin_and_args(t)
            name = target_origin.__name__
            if name in cls._casters:
                raise AssertionError(
                    f"Type caster from {cls.__name__} to {name} already registered"
                )
            cls._casters[name] = caster

        return wrap

    @classmethod
    def get_caster(cls, t: Type[V]) -> Caster[Self, V] | None:
        converters = cls._get_casters()
        target_origin, _ = get_origin_and_args(t)
        if target_origin.__name__ in converters:
            generic_caster = converters[target_origin.__name__]
            caster = generic_caster(cls, t)
            if caster is not None:
                return caster

        if issubclass(cls, t):
            return lambda value, context: value  # type: ignore

        return None

    @classmethod
    def can_cast_to(cls, t: Type[V]) -> bool:
        """
        Returns True if there is any hope of casting this value to the type t.
        """
        return cls.get_caster(t) is not None

    def __eq__(self, other) -> bool:
        if isinstance(other, Value):
            return self.root == other.root
        return self.root == other

    def __hash__(self):
        return hash(self.root)

    def __str__(self) -> str:
        return str(self.root)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.root})"

    def __bool__(self) -> bool:
        return bool(self.root)

    @cached_property
    def md5(self) -> str:
        return md5(str(self).encode()).hexdigest()

    async def cast_to(self, t: Type[V], *, context: "Context") -> V:
        key = get_value_type_key(t)
        if key in self._cast_cache:
            casted: V = self._cast_cache[key]  # type: ignore
            return casted

        cast_fn = self.__class__.get_caster(t)
        if cast_fn is not None:
            result = cast_fn(self, context)
            casted: V = (await result) if inspect.iscoroutine(result) else result  # type: ignore
            self._cast_cache[key] = casted
            return casted

        raise ValueError(f"Cannot convert {self} to {t}")

    @classmethod
    async def cast_from(cls, v: "Value", *, context: "Context") -> Self:
        return await v.cast_to(cls, context=context)

    @classmethod
    def to_value_schema(cls) -> "ValueSchema":
        from .schema import validate_value_schema  # avoid circular import

        return validate_value_schema(cls.model_json_schema())


class ValueRegistry:
    def __init__(self):
        self.types: dict[str, ValueType] = {}

    def register(self, name: str, cls: ValueType):
        if name in self.types:
            conflict = self.types[name]
            if cls is not conflict:
                raise ValueError(
                    f'Value type "{name}" (class {cls.__name__}) is already registered to a different class ({conflict.__name__})'
                )
        self.types[name] = cls
        logger.debug("Registering class %s as value type %s", cls.__name__, name)

    def __contains__(self, name: str) -> bool:
        return name in self.types

    def __getitem__(self, name: str) -> ValueType:
        if name not in self.types:
            raise ValueError(f'Value type "{name}" is not registered')
        return self.types[name]


value_type_registry = ValueRegistry()


__all__ = [
    "Caster",
    "GenericCaster",
    "get_origin_and_args",
    "Value",
    "ValueType",
    "value_type_registry",
]
