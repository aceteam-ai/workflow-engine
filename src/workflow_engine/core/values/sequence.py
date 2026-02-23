# workflow_engine/core/values/sequence.py

import asyncio
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .primitives import IntegerValue
from .value import Caster, Value, get_origin_and_args

if TYPE_CHECKING:
    from ..context import Context

T = TypeVar("T", bound=Value)


class SequenceValue(Value[Sequence[T]], Generic[T]):
    def __getitem__(self, index: int | IntegerValue) -> T:
        if isinstance(index, IntegerValue):
            index = index.root
        return self.root[index]

    def __len__(self) -> int:
        return len(self.root)

    def __iter__(self) -> Iterator[T]:  # pyright: ignore[reportIncompatibleMethodOverride]
        # NOTE: This convenience method breaks Pydantic's dict(value) behaviour,
        # for better or worse. We will revert if this actually causes problems.
        yield from self.root

    def __contains__(self, item: Any) -> bool:
        return any(x == item for x in self.root)


SourceType = TypeVar("SourceType", bound=Value)
TargetType = TypeVar("TargetType", bound=Value)


@SequenceValue.register_generic_cast_to(SequenceValue)
def cast_sequence_to_sequence(
    source_type: type[SequenceValue[SourceType]],
    target_type: type[SequenceValue[TargetType]],
) -> Caster[SequenceValue[SourceType], SequenceValue[TargetType]] | None:
    source_origin, (source_item_type,) = get_origin_and_args(source_type)
    target_origin, (target_item_type,) = get_origin_and_args(target_type)

    assert issubclass(source_origin, SequenceValue)
    assert issubclass(target_origin, SequenceValue)
    if not source_item_type.can_cast_to(target_item_type):
        return None

    async def _cast(
        value: source_type,  # pyright: ignore[reportInvalidTypeForm]
        context: "Context",
    ) -> target_type:  # pyright: ignore[reportInvalidTypeForm]
        # Cast all items in parallel
        cast_tasks = [
            x.cast_to(target_item_type, context=context)  # type: ignore
            for x in value.root
        ]
        casted_items = await asyncio.gather(*cast_tasks)
        return target_type(casted_items)  # type: ignore

    return _cast


__all__ = [
    "SequenceValue",
]
