# workflow_engine/core/values/mapping.py

from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from typing import TYPE_CHECKING, Generic, Literal, Type, TypeVar, cast

from overrides import override

from ...utils.asynchronous import gather
from .primitives import StringValue
from .value import Caster, Value, get_origin_and_args

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from .schema import ValueSchema


V = TypeVar("V", bound=Value)


class StringMapValue(Value[Mapping[str, V]], Generic[V]):
    def __getitem__(self, key: str | StringValue) -> V:
        if isinstance(key, StringValue):
            key = key.root
        return self.root[key]

    def get(self, key: str | StringValue, default: V | None = None) -> V | None:
        if isinstance(key, StringValue):
            key = key.root
        return self.root.get(key, default)

    def __len__(self) -> int:
        return len(self.root)

    def __iter__(self) -> Iterator[str]:  # pyright: ignore[reportIncompatibleMethodOverride]
        # NOTE: This convenience method breaks Pydantic's dict(value) behaviour,
        # for better or worse. We will revert if this actually causes problems.
        yield from self.root

    def items(self) -> ItemsView[str, V]:
        return self.root.items()

    def keys(self) -> KeysView[str]:
        return self.root.keys()

    def values(self) -> ValuesView[V]:
        return self.root.values()

    def __contains__(self, key: str | StringValue) -> bool:
        if isinstance(key, StringValue):
            key = key.root
        return key in self.root

    @classmethod
    @override
    def to_value_schema(cls) -> "ValueSchema":
        """
        Delegates to the value type's own ``to_value_schema()``. See
        ``SequenceValue.to_value_schema()`` for why the generic default
        (raw ``model_json_schema()``) is wrong for value types like
        ``Result[T]`` that publish their own wire shape.

        ``StringMapValue[Value]`` (the fully-open map, used when a schema's
        ``additionalProperties`` is bare ``True``) has no concrete item type
        to delegate to, so it round-trips as ``additionalProperties: True``
        directly, matching ``StringMapValueSchema.build_value_cls()``.
        """
        from .schema import StringMapValueSchema

        _origin, args = get_origin_and_args(cls)
        if not args:
            # Bare, unparameterized StringMapValue: no item type to delegate to.
            return super().to_value_schema()
        (item_type,) = args

        raw = dict(cls.model_json_schema())
        raw.pop("$defs", None)
        raw.pop("additionalProperties", None)

        additional_properties: "ValueSchema | Literal[True]" = (
            True if item_type is Value else item_type.to_value_schema()
        )

        return StringMapValueSchema(
            **raw,
            additionalProperties=additional_properties,
            value_type=cls.__name__,
        )


SourceType = TypeVar("SourceType", bound=Value)
TargetType = TypeVar("TargetType", bound=Value)


@StringMapValue.register_generic_cast_to(StringMapValue)
def cast_string_map_to_string_map(
    source_type: Type[StringMapValue[SourceType]],
    target_type: Type[StringMapValue[TargetType]],
) -> Caster[StringMapValue[SourceType], StringMapValue[TargetType]] | None:
    source_origin, (source_value_type,) = get_origin_and_args(source_type)
    target_origin, (target_value_type,) = get_origin_and_args(target_type)

    assert issubclass(source_origin, StringMapValue)
    assert issubclass(target_origin, StringMapValue)
    if not source_value_type.can_cast_to(target_value_type):
        return None

    async def _cast(
        value: source_type,  # pyright: ignore[reportInvalidTypeForm]
        context: "ExecutionContext",
    ) -> target_type:  # pyright: ignore[reportInvalidTypeForm]
        assert isinstance(value, StringMapValue)
        # Cast all values in parallel
        keys, values = zip(*value.items())
        casted_values = await gather(
            cast(source_value_type, v).cast_to(target_value_type, context=context)
            for v in values
        )
        return target_type(dict(zip(keys, casted_values)))  # type: ignore

    return _cast


__all__ = [
    "StringMapValue",
]
