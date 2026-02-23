# workflow_engine/core/values/mapping.py

import asyncio
from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from typing import TYPE_CHECKING, Generic, Type, TypeVar

from .primitives import StringValue
from .value import Caster, Value, get_origin_and_args

if TYPE_CHECKING:
    from ..context import Context


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
        context: "Context",
    ) -> target_type:  # pyright: ignore[reportInvalidTypeForm]
        assert isinstance(value, StringMapValue)
        # Cast all values in parallel
        keys, values = zip(*value.items())
        cast_tasks = [
            v.cast_to(target_value_type, context=context)  # type: ignore
            for v in values
        ]
        casted_values = await asyncio.gather(*cast_tasks)
        return target_type(dict(zip(keys, casted_values)))  # type: ignore

    return _cast


__all__ = [
    "StringMapValue",
]
