# workflow_engine/core/values/data.py
import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
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


type DataMapping = Mapping[str, Value]


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


# These are module-level functions rather than methods on Data to avoid
# namespace collisions with user-defined field names.


def get_data_dict(data: Data) -> Mapping[str, Value]:
    result: dict[str, Value] = {}
    for key in data.__class__.model_fields.keys():
        value = getattr(data, key)
        assert isinstance(value, Value)
        result[key] = value
    return result


def get_data_schema(cls: type[Data]) -> "ValueSchema":
    from .schema import validate_value_schema  # avoid circular import

    return validate_value_schema(cls.model_json_schema())


def get_field_annotations(cls: type[Data]) -> Mapping[str, type[Value]]:
    """Return the name and type annotation of each field."""
    fields: Mapping[str, type[Value]] = {}
    for field_name, field_info in cls.model_fields.items():
        assert field_info.annotation is not None
        assert issubclass(field_info.annotation, Value)
        fields[field_name] = field_info.annotation
    return fields


def get_only_field(cls: type[Data]) -> tuple[str, type[Value]]:
    """
    Get the name and type annotation of the only field. Will raise an error
    if there is not exactly one field.
    """
    fields = get_field_annotations(cls)
    if len(fields) != 1:
        raise ValueError(f"Expected 1 field, got {len(fields)}")
    return only(fields.items())


def dump_data_mapping(data: DataMapping) -> Mapping[str, Any]:
    return {k: v.model_dump() for k, v in data.items()}


def serialize_data_mapping(data: DataMapping) -> str:
    return json.dumps(dump_data_mapping(data))


def get_value_at_path(
    *,
    data: DataMapping | Data | Value,
    path: Sequence[str],
) -> Value:
    """
    Get the Value at a path in Data or nested Values.
    Traverses into Data, DataValue, and StringMapValue for multi-segment paths.

    Args:
        data: The Data instance or Value to look up in (e.g. node output, nested DataValue)
        path: Sequence of keys (e.g. ("foo",) or ("foo", "bar"))

    Returns:
        The Value at that path

    Raises:
        KeyError: If the path does not exist
    """
    # base case: we've arrived at the value
    if len(path) == 0:
        if isinstance(data, Mapping):
            raise KeyError("Empty path on DataMapping; use a field name.")
        if isinstance(data, Data):
            return DataValue[type(data)](data)
        return data

    head, *tail = path
    if isinstance(data, Data):
        container = get_data_dict(data)
    elif isinstance(data, DataValue):
        container = get_data_dict(data.root)
    elif isinstance(data, StringMapValue):
        container = data.root
    elif isinstance(data, Mapping):
        container = data
    else:
        raise KeyError(f"Cannot traverse into {type(data).__name__} at path {path}")
    if head not in container:
        raise KeyError(head)
    value = container[head]
    return get_value_at_path(data=value, path=tail)


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


def get_data_field(cls: type[Data], name: str) -> ValueType | None:
    if name not in cls.model_fields:
        return None
    field = cls.model_fields[name].annotation
    assert field is not None
    assert issubclass(field, Value)
    return field


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


def resolve_path(
    *,
    data_type: ValueType | type[Data],
    path: Sequence[str],
) -> ValueType:
    """
    Resolve a path through a Data type to the Value type at that path.

    A path like `("foo", "bar")` means: field "foo" of data_type, then field "bar"
    of the inner structure. Nested traversal works for DataValue (named fields) and
    StringMapValue (dynamic keys; value type is known, keys are not statically required).

    Args:
        data_type: The Data or ValueType subclass to resolve the path in
        path: Sequence of field names (e.g. ("foo",) or ("foo", "bar"))

    Returns:
        The ValueType at the given path.

    Raises:
        ValueError: If any segment of the path cannot be resolved on the given type.
    """
    # base case: empty path, return the current type (as a ValueType)
    if len(path) == 0:
        if issubclass(data_type, Data):
            return DataValue[data_type]
        return data_type

    head, *tail = path

    # Data type: look up head in fields
    if issubclass(data_type, (Data, DataValue)):
        # unwrap DataValue to its inner Data type if present
        if issubclass(data_type, DataValue):
            origin, (inner_data_type,) = get_origin_and_args(data_type)
            assert issubclass(origin, DataValue)
            assert issubclass(inner_data_type, Data)
            data_type = inner_data_type
        field_type = get_data_field(data_type, name=head)

    # StringMapValue: all fields have the same type
    elif issubclass(data_type, StringMapValue):
        origin, (value_type,) = get_origin_and_args(data_type)
        assert issubclass(origin, StringMapValue)
        assert issubclass(value_type, Value)
        field_type = value_type

    # other ValueType: no fields
    else:
        field_type = None

    if field_type is None:
        # raising an exception is more informative than returning None, because
        # None can happen at any depth of recursion, whereas the exception
        # explicitly indicates what level we encountered the problem.
        raise ValueError(f"{data_type.__name__} does not have field {head}")

    # recursive step
    return resolve_path(data_type=field_type, path=tail)


def has_path(
    *,
    data_type: ValueType | type[Data],
    path: Sequence[str],
) -> bool:
    """
    Check whether a path exists in a Data type.

    Args:
        data_type: The Data or ValueType subclass to check
        path: Sequence of field names

    Returns:
        True if the path exists and resolves to a Value type
    """
    try:
        _ = resolve_path(data_type=data_type, path=path)
        return True
    except ValueError:
        return False


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

        items = list(get_data_dict(value.root).items())
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
        items = list(get_data_dict(value.root).items())
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
    "get_data_dict",
    "get_data_fields",
    "get_data_schema",
    "get_field_annotations",
    "get_only_field",
    "has_path",
    "Input_contra",
    "Output_co",
    "resolve_path",
    "serialize_data_mapping",
]
