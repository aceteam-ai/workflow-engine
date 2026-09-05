# workflow_engine/core/values/sequence.py

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from overrides import override

from ...utils.asynchronous import gather
from .primitives import IntegerValue
from .value import Caster, Value, get_origin_and_args

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from .schema import ValueSchema

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

    @classmethod
    @override
    def to_value_schema(cls) -> "ValueSchema":
        """
        Delegates to the item type's own ``to_value_schema()`` instead of
        trusting Pydantic's ``model_json_schema()`` to describe the item.

        The generic default (``Value.to_value_schema()``) embeds whatever raw
        JSON Schema Pydantic generates for the item type. That is correct for
        item types with no custom ``to_value_schema()`` (Pydantic's schema and
        ours coincide), but wrong for item types like ``Result[T]`` that
        publish a different wire shape than their raw Pydantic schema: the
        embedded ``$ref`` then points at something ``validate_value_schema()``
        cannot rebuild. Calling ``item_type.to_value_schema()`` directly keeps
        the two in sync at every nesting depth, the same way ``Result[T]``
        already delegates to its own item type.

        Constraints (``minItems``/``maxItems``) and any other schema-level
        extras still come from ``model_json_schema()``, since those describe
        this sequence itself, not its item type.
        """
        from .schema import SequenceValueSchema

        _origin, args = get_origin_and_args(cls)
        if not args:
            # Bare, unparameterized SequenceValue: no item type to delegate to.
            return super().to_value_schema()
        (item_type,) = args

        raw = dict(cls.model_json_schema())
        raw.pop("$defs", None)
        raw.pop("items", None)

        return SequenceValueSchema(
            **raw,
            items=item_type.to_value_schema(),
            value_type=cls.__name__,
        )


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
        context: "ExecutionContext",
    ) -> target_type:  # pyright: ignore[reportInvalidTypeForm]
        # Cast all items in parallel
        casted_items = await gather(
            cast(source_item_type, x).cast_to(target_item_type, context=context)
            for x in value.root
        )
        return target_type(casted_items)  # type: ignore

    return _cast


__all__ = [
    "SequenceValue",
]
