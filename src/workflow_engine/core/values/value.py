# workflow_engine/core/values/value.py
from __future__ import annotations

import inspect
import re

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from functools import cached_property
from hashlib import md5
from logging import getLogger
from typing import (
    TYPE_CHECKING,
    Awaitable,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    Self,
    TypeVar,
    get_origin,
    overload,
)

from overrides import override
from pydantic import PrivateAttr

from ...utils.immutable import ImmutableRootModel

if TYPE_CHECKING:
    from ..context import Context
    from .schema import ValueSchema


logger = getLogger(__name__)


T = TypeVar("T")
V = TypeVar("V", bound="Value")
type ValueType = type[Value]


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
        source_type: type[SourceType],
        target_type: type[TargetType],
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

    def __init_subclass__(cls, register: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)

        # reinitialize for each subclass so it doesn't just reference the parent
        cls._casters = {}
        cls._resolved_casters = None

        if not register:
            return

        while generic_pattern.match(cls.__name__) is not None:
            assert cls.__base__ is not None
            cls = cls.__base__
        # Skip generic base classes (e.g. SequenceValue, StringMapValue) whose
        # __pydantic_generic_metadata__["parameters"] still contains unbound
        # TypeVars — only fully-concrete classes belong in the registry.
        if (
            get_origin(cls) is None
            and len(cls.__pydantic_generic_metadata__["parameters"]) == 0
        ):
            ValueRegistry.DEFAULT.register_value_class(cls)

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
    def register_cast_to(cls, t: type[V]):
        """
        A decorator to register a possible type cast from this class to the
        class T, neither of which are generic.
        """

        def wrap(caster: Caster[Self, V]):
            cls.register_generic_cast_to(t)(lambda source_type, target_type: caster)
            return caster

        return wrap

    @classmethod
    def register_generic_cast_to(cls, t: type[V]):
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
    def get_caster(cls, t: type[V]) -> Caster[Self, V] | None:
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
    def can_cast_to(cls, t: type[V]) -> bool:
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

    async def cast_to(self, t: type[V], *, context: "Context") -> V:
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

        schema = cls.model_json_schema()
        schema["x-value-type"] = cls.__name__
        return validate_value_schema(schema)


class ValueRegistry(ABC):
    """
    An immutable registry of value types by name.
    """

    DEFAULT: ClassVar[LazyValueRegistry]

    @abstractmethod
    def get_value_class(self, name: str) -> ValueType:
        """Get a value type by name."""
        pass

    @abstractmethod
    def has_name(self, name: str) -> bool:
        """Check if a value type name is registered."""
        pass

    @abstractmethod
    def all_value_classes(self) -> Iterable[tuple[str, ValueType]]:
        """Return all registered value classes as (name, class) pairs."""
        pass

    def load_value(self, schema: ValueSchema) -> ValueType | None:
        """
        Load a value type from a schema by looking up the registry.

        Checks x-value-type first, then falls back to title for backwards
        compatibility. If no match is found, returns None to indicate that the
        caller should fall back to building the value class from the schema.

        Args:
            schema: A ValueSchema with an optional value_type or title field

        Returns:
            The registered value class if a match is found, None otherwise
        """
        if schema.value_type is not None and self.has_name(schema.value_type):
            return self.get_value_class(schema.value_type)
        if schema.title is not None and self.has_name(schema.title):
            return self.get_value_class(schema.title)
        return None

    @overload
    @staticmethod
    def builder(*, lazy: Literal[True]) -> LazyValueRegistry: ...

    @overload
    @staticmethod
    def builder(*, lazy: Literal[False] = False) -> EagerValueRegistryBuilder: ...

    @staticmethod
    def builder(*, lazy: bool = False):
        return LazyValueRegistry() if lazy else EagerValueRegistryBuilder()

    @overload
    def extend(self, *, lazy: Literal[True]) -> LazyValueRegistry: ...

    @overload
    def extend(self, *, lazy: Literal[False] = False) -> EagerValueRegistryBuilder: ...

    def extend(self, *, lazy: bool = False):
        """Create a new builder pre-populated with all value classes from this registry."""
        builder = ValueRegistry.builder(lazy=lazy)
        for _name, value_cls in self.all_value_classes():
            builder.register_value_class(value_cls)
        return builder


class ValueRegistryBuilder(ABC):
    """
    A builder for creating value registries.
    """

    @abstractmethod
    def register_value_class(self, value_cls: ValueType) -> Self:
        """Register a value type using its class name."""
        pass

    @abstractmethod
    def remove_value_class(self, value_cls: ValueType, *, missing_ok: bool = False) -> Self:
        """Remove a value type by class."""
        pass

    @abstractmethod
    def build(self) -> ValueRegistry:
        """Build and return the registry."""
        pass


class ImmutableValueRegistry(ValueRegistry):
    """
    An ImmutableValueRegistry is a ValueRegistry that is immutable after
    construction; enforced by shallow copying the input data structures.
    """

    def __init__(self, *, value_classes: Mapping[str, ValueType]):
        self._value_classes = dict(value_classes)

    @override
    def has_name(self, name: str) -> bool:
        return name in self._value_classes

    @override
    def get_value_class(self, name: str) -> ValueType:
        if name not in self._value_classes:
            raise ValueError(f'Value type "{name}" is not registered')
        return self._value_classes[name]

    @override
    def all_value_classes(self) -> Iterable[tuple[str, ValueType]]:
        return self._value_classes.items()


class EagerValueRegistryBuilder(ValueRegistryBuilder):
    """
    A builder that validates registrations immediately.
    """

    def __init__(self):
        self._value_classes: dict[str, ValueType] = {}

    @override
    def register_value_class(self, value_cls: ValueType) -> Self:
        name = value_cls.__name__
        if name in self._value_classes:
            conflict = self._value_classes[name]
            if value_cls is not conflict:
                raise ValueError(
                    f'Value type "{name}" (class {value_cls.__name__}) is already '
                    f"registered to a different class ({conflict.__name__})"
                )
        self._value_classes[name] = value_cls
        logger.debug("Registering value type %s", name)
        return self

    @override
    def remove_value_class(self, value_cls: ValueType, *, missing_ok: bool = False) -> Self:
        name = value_cls.__name__
        if name not in self._value_classes:
            if not missing_ok:
                raise ValueError(f'Value type "{name}" is not registered')
        else:
            del self._value_classes[name]
        return self

    @override
    def build(self) -> ValueRegistry:
        return ImmutableValueRegistry(value_classes=self._value_classes)


class LazyValueRegistry(ValueRegistry, ValueRegistryBuilder):
    """
    A LazyValueRegistry is both a ValueRegistry and a ValueRegistryBuilder with a
    2-part lifecycle:
    1. Registration phase: acts as a ValueRegistryBuilder until .build() is called.
    2. Registry phase: acts as a ValueRegistry.

    Validations are deferred until the registry is frozen; this allows for a
    LazyValueRegistry to exist as long as it is not used, making it suitable for
    automatic value registration.
    """

    def __init__(self):
        self._registrations: list[ValueType] = []
        self._removals: dict[ValueType, bool] = {}  # cls -> missing_ok
        self._frozen: bool = False

        # initialized after freeze
        self._value_classes: Mapping[str, ValueType]

    # REGISTRATION PHASE

    @override
    def register_value_class(self, value_cls: ValueType) -> Self:
        if self._frozen:
            # Allow re-registration of the same class (for testing/reloading)
            name = value_cls.__name__
            if name in self._value_classes and self._value_classes[name] is value_cls:
                logger.debug(
                    "Value type %s already registered, skipping",
                    name,
                )
                return self
            raise ValueError(
                f"Value registry is frozen, cannot register new value type '{name}' (class {value_cls.__name__})"
            )
        self._registrations.append(value_cls)
        return self

    @override
    def remove_value_class(self, value_cls: ValueType, *, missing_ok: bool = False) -> Self:
        if self._frozen:
            raise ValueError("Value registry is frozen, cannot remove value types.")
        self._removals[value_cls] = missing_ok
        return self

    @override
    def build(self) -> ValueRegistry:
        if self._frozen:
            return self

        self._frozen = True
        _value_classes = {}
        for value_cls in self._registrations:
            name = value_cls.__name__
            if name in _value_classes:
                conflict = _value_classes[name]
                if value_cls is conflict:
                    logger.warning(
                        "Value type %s is already registered to class %s, skipping registration",
                        name,
                        value_cls.__name__,
                    )
                else:
                    raise ValueError(
                        f'Value type "{name}" (class {value_cls.__name__}) is already '
                        f"registered to a different class ({conflict.__name__})"
                    )
            else:
                _value_classes[name] = value_cls
                logger.debug("Registering value type %s", name)

        for value_cls, missing_ok in self._removals.items():
            name = value_cls.__name__
            if name not in _value_classes:
                if not missing_ok:
                    raise ValueError(f'Value type "{name}" is not registered')
            else:
                del _value_classes[name]

        del self._registrations  # Memory optimization
        del self._removals
        self._value_classes = _value_classes

        return self

    # USE PHASE

    @override
    def has_name(self, name: str) -> bool:
        self.build()
        return name in self._value_classes

    @override
    def get_value_class(self, name: str) -> ValueType:
        self.build()
        if name not in self._value_classes:
            raise ValueError(f'Value type "{name}" is not registered')
        return self._value_classes[name]

    @override
    def all_value_classes(self) -> Iterable[tuple[str, ValueType]]:
        self.build()
        return self._value_classes.items()


ValueRegistry.DEFAULT = ValueRegistry.builder(lazy=True)


__all__ = [
    "Caster",
    "GenericCaster",
    "get_origin_and_args",
    "Value",
    "ValueRegistry",
    "ValueType",
]
