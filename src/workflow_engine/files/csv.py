# workflow_engine/files/csv.py
from collections.abc import Mapping, Sequence
from csv import DictReader, DictWriter
from hashlib import md5
from io import StringIO
from typing import Any, ClassVar, Self, TypeVar

from workflow_engine.core.values import get_origin_and_args

from ..core import (
    Caster,
    Context,
    File,
    JSONValue,
    SequenceValue,
    StringMapValue,
    UserException,
    Value,
)
from .text import TextFileValue


V = TypeVar("V", bound=Value)


class CSVFileValue(TextFileValue):
    mime_type: ClassVar[str] = "text/csv"

    async def read_data(self, context: Context) -> Sequence[Mapping[str, Any]]:
        text = await self.read_text(context)
        text_io = StringIO(text)
        reader = DictReader(text_io)
        return tuple(reader)

    async def write_data(
        self,
        context: Context,
        data: Sequence[Mapping[str, Any]],
    ) -> Self:
        text_io = StringIO()
        writer = DictWriter(text_io, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        text = text_io.getvalue()
        return await self.write_text(context, text)


@JSONValue.register_cast_to(CSVFileValue)
async def json_to_csv(value: JSONValue, context: Context) -> CSVFileValue:
    data = value.root
    if not isinstance(data, (Mapping, Sequence)):
        raise UserException(
            "JSON value must be a mapping or sequence to be cast to CSV"
        )

    if isinstance(data, Mapping):
        # treat a mapping as a single row
        data = [value.root]

    rows: list[Mapping[str, Any]] = []
    for row in data:
        if isinstance(row, Sequence):
            rows.append({str(i): v for i, v in enumerate(row)})
        elif isinstance(row, Mapping):
            rows.append(row)
        else:
            raise UserException(
                f"Object {row} is not a mapping or sequence, need mappings or sequences to write to CSV"
            )

    data_hash = md5(str(rows).encode()).hexdigest()
    output = CSVFileValue(File(path=f"{data_hash}.csv"))
    output = await output.write_data(context, rows)
    return output


@StringMapValue.register_generic_cast_to(CSVFileValue)
def mapping_to_csv(
    source_type: type[StringMapValue[V]],
    target_type: type[CSVFileValue],
) -> Caster[StringMapValue[V], CSVFileValue] | None:
    """
    Transitive cast from StringMapValue to CSVFileValue via JSON.
    """
    if not source_type.can_cast_to(JSONValue):
        return None

    async def cast(
        value: source_type,  # pyright: ignore[reportInvalidTypeForm]
        context: Context,
    ) -> target_type:  # pyright: ignore[reportInvalidTypeForm]
        assert isinstance(value, StringMapValue)
        json_obj = await value.cast_to(JSONValue, context=context)
        return await json_obj.cast_to(CSVFileValue, context=context)

    return cast


@SequenceValue.register_generic_cast_to(CSVFileValue)
def sequence_to_csv(
    source_type: type[SequenceValue[V]],
    target_type: type[CSVFileValue],
) -> Caster[SequenceValue[V], CSVFileValue] | None:
    source_origin, (source_value_type,) = get_origin_and_args(source_type)
    target_origin, () = get_origin_and_args(target_type)
    assert issubclass(source_origin, SequenceValue)
    assert issubclass(target_origin, CSVFileValue)
    if not source_value_type.can_cast_to(JSONValue):
        return None

    async def cast(
        value: source_type,  # pyright: ignore[reportInvalidTypeForm]
        context: Context,
    ) -> target_type:  # pyright: ignore[reportInvalidTypeForm]
        assert isinstance(value, SequenceValue)
        value_as_json_sequence = await value.cast_to(
            SequenceValue[JSONValue], context=context
        )
        rows: list[Mapping[str, Any]] = []
        for obj in value_as_json_sequence.root:
            row = obj.root
            if isinstance(row, Sequence):
                row = {str(i): v for i, v in enumerate(row)}
            if not isinstance(row, Mapping):
                raise UserException(
                    f"Object {obj} is not a mapping, need mappings to write to CSV"
                )
            rows.append(row)

        data_hash = md5(str(rows).encode()).hexdigest()
        output = CSVFileValue(File(path=f"{data_hash}.csv"))
        output = await output.write_data(context, rows)
        return output

    return cast


@CSVFileValue.register_cast_to(JSONValue)
async def csv_file_to_json(value: CSVFileValue, context: Context) -> JSONValue:
    rows = await value.read_data(context)
    return JSONValue(rows)


@CSVFileValue.register_generic_cast_to(SequenceValue)
def csv_file_to_sequence(
    source_type: type[CSVFileValue],
    target_type: type[SequenceValue[V]],
) -> Caster[CSVFileValue, SequenceValue[V]] | None:
    source_origin, () = get_origin_and_args(source_type)
    target_origin, (_target_value_type,) = get_origin_and_args(target_type)
    assert issubclass(source_origin, CSVFileValue)
    assert issubclass(target_origin, SequenceValue)

    async def cast(
        value: source_type,  # pyright: ignore[reportInvalidTypeForm]
        context: Context,
    ) -> target_type:  # pyright: ignore[reportInvalidTypeForm]
        data = await value.read_data(context)
        return target_type.model_validate(data)

    return cast


__all__ = [
    "CSVFileValue",
]
