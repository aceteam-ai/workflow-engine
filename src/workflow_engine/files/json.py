# workflow_engine/files/json.py
import datetime
import json
from collections.abc import Sequence
from logging import getLogger
from typing import Any, ClassVar, Self, Type

from ..core import (
    BooleanValue,
    Caster,
    Context,
    File,
    FileValue,
    FloatValue,
    IntegerValue,
    JSONValue,
    NullValue,
    SequenceValue,
    StringMapValue,
    StringValue,
    Value,
)
from ..core.values import get_origin_and_args
from .text import TextFileValue

logger = getLogger(__name__)


# HACK: serialize datetime objects
def _custom_json_serializer(obj: object) -> Any:
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    return None


class JSONFileValue(TextFileValue):
    """
    A Value that represents a JSON file.
    """

    mime_type: ClassVar[str] = "application/json"

    async def read_data(self, context: "Context") -> Any:
        return json.loads(await self.read_text(context))

    async def write_data(self, context: "Context", data: Any) -> Self:
        text = json.dumps(data, default=_custom_json_serializer)
        return await self.write_text(context, text)


@JSONFileValue.register_cast_to(NullValue)
async def cast_json_file_to_null(value: JSONFileValue, context: "Context") -> NullValue:
    data = await value.read_data(context)
    if data is None:
        return NullValue(None)
    raise ValueError(f"Expected null, got {type(data)}")


@JSONFileValue.register_cast_to(BooleanValue)
async def cast_json_file_to_boolean(
    value: JSONFileValue, context: "Context"
) -> BooleanValue:
    data = await value.read_data(context)
    if isinstance(data, bool):
        return BooleanValue(data)
    raise ValueError(f"Expected bool, got {type(data)}")


@JSONFileValue.register_cast_to(IntegerValue)
async def cast_json_file_to_integer(
    value: JSONFileValue, context: "Context"
) -> IntegerValue:
    data = await value.read_data(context)
    if isinstance(data, int):
        return IntegerValue(data)
    raise ValueError(f"Expected int, got {type(data)}")


@JSONFileValue.register_cast_to(FloatValue)
async def cast_json_file_to_float(
    value: JSONFileValue, context: "Context"
) -> FloatValue:
    data = await value.read_data(context)
    if isinstance(data, float):
        return FloatValue(data)
    raise ValueError(f"Expected float, got {type(data)}")


@JSONFileValue.register_cast_to(StringValue)
async def cast_json_file_to_string(
    value: JSONFileValue, context: "Context"
) -> StringValue:
    data = await value.read_data(context)
    if isinstance(data, str):
        return StringValue(data)
    raise ValueError(f"Expected str, got {type(data)}")


@JSONFileValue.register_generic_cast_to(SequenceValue)
def cast_json_file_to_sequence(
    source_type: Type[JSONFileValue],
    target_type: Type[SequenceValue],
) -> Caster[JSONFileValue, SequenceValue]:
    assert issubclass(target_type, SequenceValue)

    async def _cast(value: JSONFileValue, context: "Context") -> SequenceValue[Any]:
        return target_type(await value.read_data(context))

    return _cast


@JSONFileValue.register_generic_cast_to(StringMapValue)
def cast_json_file_to_string_map(
    source_type: Type[JSONFileValue],
    target_type: Type[StringMapValue],
) -> Caster[JSONFileValue, StringMapValue]:
    assert issubclass(target_type, StringMapValue)

    async def _cast(value: JSONFileValue, context: "Context") -> StringMapValue[Any]:
        return target_type(await value.read_data(context))

    return _cast


@JSONFileValue.register_cast_to(JSONValue)
async def cast_json_file_to_json(value: JSONFileValue, context: "Context") -> JSONValue:
    return JSONValue(await value.read_data(context))


@Value.register_cast_to(JSONFileValue)
async def cast_any_to_json_file(value: Value, context: "Context") -> JSONFileValue:
    if isinstance(value, JSONFileValue):
        return value
    if isinstance(value, FileValue):
        logger.warning(
            f"Forcibly casting file {value.path} of MIME type {value.mime_type} and source type {type(value).__name__} "
            f"(which might not necessarily be JSON compatible) to JSONFileValue. Ensure the source data is valid JSON."
        )
        return JSONFileValue(value.root)
    file = JSONFileValue(File(path=f"{value.md5}.json"))
    return await file.write_data(context=context, data=value.model_dump())


class JSONLinesFileValue(TextFileValue):
    """
    A file that contains a list of Python objects serialized as JSON.
    """

    mime_type: ClassVar[str] = "application/jsonl"

    async def read_data(self, context: "Context") -> Sequence[Any]:
        return [
            json.loads(line) for line in (await self.read_text(context)).splitlines()
        ]

    async def write_data(self, context: "Context", data: Sequence[Any]) -> Self:
        text = "\n".join(
            json.dumps(item, default=_custom_json_serializer) for item in data
        )
        return await self.write_text(context, text)


@JSONLinesFileValue.register_generic_cast_to(SequenceValue)
def cast_json_lines_to_sequence(
    source_type: Type[JSONLinesFileValue],
    target_type: Type[SequenceValue],
) -> Caster[JSONLinesFileValue, SequenceValue] | None:
    target_origin, (target_item_type,) = get_origin_and_args(target_type)
    assert issubclass(target_origin, SequenceValue)

    if any(
        issubclass(target_item_type, t)
        for t in (
            BooleanValue,
            FloatValue,
            IntegerValue,
            JSONValue,
            NullValue,
            SequenceValue,
            StringMapValue,
            StringValue,
        )
    ):

        async def _read_lines(
            value: JSONLinesFileValue,
            context: Context,
        ) -> SequenceValue[Any]:
            return target_type(
                [
                    target_item_type.model_validate(item)
                    for item in await value.read_data(context)
                ]
            )

        return _read_lines

    if issubclass(target_item_type, JSONFileValue):

        async def _read_then_recast(
            value: JSONLinesFileValue,
            context: Context,
        ):
            items = SequenceValue[Value](
                [Value(item) for item in await value.read_data(context)]
            )
            return await items.cast_to(target_type, context=context)

        return _read_then_recast

    return None


@SequenceValue.register_cast_to(JSONLinesFileValue)
async def cast_sequence_to_json_lines(
    value: SequenceValue[Value],
    context: Context,
) -> JSONLinesFileValue:
    file = JSONLinesFileValue(File(path=f"{value.md5}.jsonl"))
    return await file.write_data(context, [v.model_dump() for v in value])


__all__ = [
    "JSONFileValue",
    "JSONLinesFileValue",
]
