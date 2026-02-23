# workflow_engine/core/values/data.py
import asyncio
import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from pydantic import ConfigDict, create_model

from ...utils.immutable import ImmutableBaseModel
from ...utils.iter import only
from .mapping import StringMapValue
from .value import Caster, Value, ValueType, get_origin_and_args

if TYPE_CHECKING:
    from ..context import Context
    from .schema import ValueSchema

logger = logging.getLogger(__name__)


class Data(ImmutableBaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    def __init_subclass__(cls, **kwargs):
        """Ensure all fields in subclasses are Value types."""
        super().__init_subclass__(**kwargs)

        for field_name, field_info in cls.model_fields.items():
            if not issubclass(field_info.annotation, Value):  # type: ignore
                raise TypeError(
                    f"Field '{field_name}' in {cls.__name__} must be a Value type, got {field_info.annotation}"
                )

    def to_dict(self) -> Mapping[str, Value]:
        data: dict[str, Value] = {}
        for key in self.__class__.model_fields.keys():
            value = getattr(self, key)
            assert isinstance(value, Value)
            data[key] = value
        return data

    @classmethod
    def to_value_schema(cls) -> "ValueSchema":
        from .schema import validate_value_schema  # avoid circular import

        return validate_value_schema(cls.model_json_schema())

    @classmethod
    def field_annotations(cls) -> Mapping[str, type[Value]]:
        """Iterate through the name and type annotation of each field."""
        fields: Mapping[str, type[Value]] = {}
        for field_name, field_info in cls.model_fields.items():
            assert field_info.annotation is not None
            assert issubclass(field_info.annotation, Value)
            fields[field_name] = field_info.annotation
        return fields

    @classmethod
    def only_field(cls) -> tuple[str, type[Value]]:
        """
        Get the name and type annotation of the only field. Will raise an error
        if there is not exactly one field.
        """
        fields = cls.field_annotations()
        if len(fields) != 1:
            raise ValueError(f"Expected 1 field, got {len(fields)}")
        return only(fields.items())


type DataMapping = Mapping[str, Value]


def dump_data_mapping(data: DataMapping) -> Mapping[str, Any]:
    return {k: v.model_dump() for k, v in data.items()}


def serialize_data_mapping(data: DataMapping) -> str:
    return json.dumps(dump_data_mapping(data))


Input_contra = TypeVar("Input_contra", bound=Data, contravariant=True)
Output_co = TypeVar("Output_co", bound=Data, covariant=True)


def get_data_fields(cls: type[Data]) -> Mapping[str, tuple[ValueType, bool]]:
    """
    Extract the fields of a Data subclass.

    Args:
        cls: The Data subclass to extract fields from

    Returns:
        A mapping of field names to (ValueType, is_required) tuples
    """
    fields: Mapping[str, tuple[ValueType, bool]] = {}
    for k, v in cls.model_fields.items():
        assert v.annotation is not None
        assert issubclass(v.annotation, Value)
        fields[k] = (v.annotation, v.is_required())
    return fields


D = TypeVar("D", bound=Data)


def build_data_type(
    name: str,
    fields: Mapping[str, tuple[ValueType, bool]],
    base_cls: type[D] = Data,
) -> type[D]:
    """
    Create a Data subclass whose fields are given by a mapping of field names to
    (ValueType, is_required) tuples.

    This is the inverse of get_fields() - it constructs a class that would return
    the same mapping when passed to get_fields().

    Args:
        name: The name of the class to create
        fields: Mapping of field names to (ValueType, required) tuples
        base_class: The base class to inherit from (defaults to Data)

    Returns:
        A new Pydantic BaseModel class with the specified fields
    """
    # Create field annotations dictionary
    annotations: dict[str, ValueType | tuple[ValueType, Any]] = {
        field_name: value_type if required else (value_type, None)
        for field_name, (value_type, required) in fields.items()
    }

    # Create the class dynamically
    cls = create_model(name, __base__=base_cls, **annotations)  # type: ignore

    return cls


class DataValue(Value[D], Generic[D]):
    """
    A Value subclass that wraps an arbitrary Data object.
    """

    pass


V = TypeVar("V", bound=Value)


@DataValue.register_generic_cast_to(DataValue)
def cast_data_to_data(
    source_type: type[DataValue],
    target_type: type[DataValue],
) -> Caster[DataValue, DataValue] | None:
    source_origin, (source_value_type,) = get_origin_and_args(source_type)
    assert issubclass(source_origin, DataValue)
    assert issubclass(source_value_type, Data)

    target_origin, (target_value_type,) = get_origin_and_args(target_type)
    assert issubclass(target_origin, DataValue)
    assert issubclass(target_value_type, Data)

    source_fields = get_data_fields(source_value_type)
    target_fields = get_data_fields(target_value_type)

    for name, (target_field_type, is_required) in target_fields.items():
        if name not in source_fields:
            if is_required:
                return None
            continue

        source_field_type, _ = source_fields[name]
        if not source_field_type.can_cast_to(target_field_type):
            return None

    async def _cast(
        value: source_type,  # pyright: ignore[reportInvalidTypeForm]
        context: "Context",
    ) -> target_type:  # pyright: ignore[reportInvalidTypeForm]
        assert isinstance(value.root, source_value_type)

        items = list(value.root.to_dict().items())
        keys = [k for k, v in items]
        cast_tasks = [v.cast_to(target_fields[k][0], context=context) for k, v in items]
        casted_values = await asyncio.gather(*cast_tasks)
        data_dict = dict(zip(keys, casted_values))
        return target_type(data_dict)

    return _cast


@DataValue.register_generic_cast_to(StringMapValue)
def cast_data_to_string_map(
    source_type: type[DataValue],
    target_type: type[StringMapValue[V]],
) -> Caster[DataValue, StringMapValue[V]] | None:
    """
    Casts a DataValue[D] object to a StringMapValue[V] object, if all of the
    fields of D can be cast to V.
    """

    source_origin, (source_value_type,) = get_origin_and_args(source_type)
    assert issubclass(source_origin, DataValue)
    assert issubclass(source_value_type, Data)

    target_origin, (target_value_type,) = get_origin_and_args(target_type)
    assert issubclass(target_origin, StringMapValue)
    assert issubclass(target_value_type, Value)

    source_fields = get_data_fields(source_value_type)
    for source_field_type, _ in source_fields.values():
        if not source_field_type.can_cast_to(target_value_type):
            return None

    async def _cast(
        value: source_type,  # pyright: ignore[reportInvalidTypeForm]
        context: "Context",
    ) -> target_type:  # pyright: ignore[reportInvalidTypeForm]
        assert isinstance(value.root, Data)

        # Cast all fields in parallel
        items = list(value.root.to_dict().items())
        keys = [k for k, v in items]
        cast_tasks = [v.cast_to(target_value_type, context=context) for k, v in items]
        casted_values = await asyncio.gather(*cast_tasks)
        return target_type(dict(zip(keys, casted_values)))  # type: ignore

    return _cast


@StringMapValue.register_generic_cast_to(DataValue)
def cast_string_map_to_data(
    source_type: type[StringMapValue],
    target_type: type[DataValue],
) -> Caster[StringMapValue, DataValue] | None:
    """
    Casts a StringMapValue[V] object to a DataValue[D] object by trying to cast
    each value in the map at runtime.

    We don't require statically that V can be cast to the fields of D, because
    in practice V will just be a higher up supertype.
    """

    source_origin, (source_value_type,) = get_origin_and_args(source_type)
    assert issubclass(source_origin, StringMapValue)
    assert issubclass(source_value_type, Value)

    target_origin, (target_value_type,) = get_origin_and_args(target_type)
    assert issubclass(target_origin, DataValue)
    assert issubclass(target_value_type, Data)

    target_fields = get_data_fields(target_value_type)

    for target_field_name, (target_field_type, _) in target_fields.items():
        if not source_value_type.can_cast_to(target_field_type):
            logger.warning(
                "%s to %s: cannot statically cast value type %s to %s (of field %s); will need to rely on a runtime cast which may fail.",
                source_type,
                target_type,
                source_value_type,
                target_field_type,
                target_field_name,
            )

    async def _cast(
        value: source_type,  # pyright: ignore[reportInvalidTypeForm]
        context: "Context",
    ) -> target_type:  # pyright: ignore[reportInvalidTypeForm]
        assert isinstance(value, StringMapValue)

        async def cast_field(
            field_name: str,
            field_value: Value,
        ) -> Value:
            if field_name in target_fields:
                target_field_type, _ = target_fields[field_name]
                casted_value = await field_value.cast_to(
                    target_field_type,
                    context=context,
                )
                return casted_value
            return field_value

        items = list(value.root.items())
        keys = [k for k, v in items]
        cast_tasks = [cast_field(k, v) for k, v in items]
        casted_values = await asyncio.gather(*cast_tasks)
        return target_type(dict(zip(keys, casted_values)))  # type: ignore

    return _cast


__all__ = [
    "build_data_type",
    "Data",
    "DataMapping",
    "DataValue",
    "dump_data_mapping",
    "get_data_fields",
    "Input_contra",
    "Output_co",
    "serialize_data_mapping",
]
