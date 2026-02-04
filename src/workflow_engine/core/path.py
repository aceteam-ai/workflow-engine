# workflow_engine/core/path.py
"""
Support for deep field access in edges using paths like ("foo", 0, "bar").

This module provides utilities for traversing and type-checking deep paths
through Value objects (Data fields, SequenceValue indices, StringMapValue keys).
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Type, get_args, get_origin

from .values import Data, SequenceValue, StringMapValue, Value, ValueType, get_data_fields

if TYPE_CHECKING:
    pass


# Type definitions
PathSegment = str | int
FieldPath = str | Sequence[PathSegment]


def normalize_path(path: FieldPath) -> tuple[PathSegment, ...]:
    """
    Normalize a path to a tuple of segments.

    Args:
        path: Either a string (single segment) or sequence of segments

    Returns:
        Tuple of path segments

    Examples:
        >>> normalize_path("foo")
        ("foo",)
        >>> normalize_path(("foo", 0, "bar"))
        ("foo", 0, "bar")
        >>> normalize_path(["foo", "bar"])
        ("foo", "bar")
    """
    if isinstance(path, str):
        return (path,)
    return tuple(path)


def traverse_value(value: Value, path: FieldPath) -> Value:
    """
    Traverse a Value using a field path.

    Supports traversal through:
    - Data fields (string keys)
    - SequenceValue indices (integer keys)
    - StringMapValue keys (string keys)

    Args:
        value: The starting Value
        path: Path to traverse (string or sequence of str/int segments)

    Returns:
        The Value at the end of the path

    Raises:
        KeyError: If a string segment doesn't exist
        IndexError: If an integer segment is out of bounds
        TypeError: If trying to index/access incompatible Value type
        AttributeError: If a field doesn't exist on a Data object

    Examples:
        >>> data = MyData(foo=StringValue("hello"))
        >>> traverse_value(data, "foo")
        StringValue("hello")

        >>> seq = SequenceValue([StringValue("a"), StringValue("b")])
        >>> traverse_value(seq, (0,))
        StringValue("a")

        >>> data = MyData(items=SequenceValue([MyItem(name=StringValue("x"))]))
        >>> traverse_value(data, ("items", 0, "name"))
        StringValue("x")
    """
    segments = normalize_path(path)
    current = value

    for segment in segments:
        if isinstance(segment, str):
            # String key - works for Data fields or StringMapValue
            if isinstance(current, Data):
                # Access field attribute
                if not hasattr(current, segment):
                    raise AttributeError(
                        f"Data object {type(current).__name__} has no field '{segment}'"
                    )
                current = getattr(current, segment)
            elif isinstance(current, StringMapValue):
                # Access map key
                current = current[segment]
            else:
                raise TypeError(
                    f"Cannot access string key '{segment}' on {type(current).__name__}"
                )
        elif isinstance(segment, int):
            # Integer index - works for SequenceValue
            if not isinstance(current, SequenceValue):
                raise TypeError(
                    f"Cannot access integer index {segment} on {type(current).__name__}"
                )
            current = current[segment]
        else:
            raise TypeError(f"Invalid path segment type: {type(segment)}")

    return current


def resolve_path_type(value_type: ValueType, path: FieldPath) -> ValueType:
    """
    Resolve the Value type at the end of a path through a Value type.

    This is used for static type checking of edges - given a source node's
    output type and a path, determine what type would be at that path.

    Args:
        value_type: Starting Value type
        path: Path to traverse

    Returns:
        The Value type at the end of the path

    Raises:
        ValueError: If the path is invalid for the given type
        TypeError: If trying to traverse through a non-container type

    Examples:
        >>> class MyData(Data):
        ...     foo: StringValue
        ...     items: SequenceValue[IntegerValue]
        >>> resolve_path_type(MyData, "foo")
        StringValue
        >>> resolve_path_type(MyData, ("items", 0))
        IntegerValue
    """
    segments = normalize_path(path)
    current_type = value_type

    for i, segment in enumerate(segments):
        if isinstance(segment, str):
            # String key - check if it's a Data type or StringMapValue
            # First try to see if it's a Data subclass
            is_data_subclass = False
            try:
                if isinstance(current_type, type) and issubclass(current_type, Data):
                    is_data_subclass = True
            except TypeError:
                # Not a class or not a subclass check
                pass

            if is_data_subclass:
                # Get fields from Data type
                fields = get_data_fields(current_type)
                if segment not in fields:
                    raise ValueError(
                        f"Data type {current_type.__name__} has no field '{segment}' "
                        f"(at path segment {i})"
                    )
                current_type, _ = fields[segment]
            else:
                # Check if it's a generic StringMapValue using get_origin_and_args
                try:
                    from .values import get_origin_and_args as value_get_origin_and_args

                    origin, args = value_get_origin_and_args(current_type)
                    if origin is StringMapValue:
                        # Get value type from StringMapValue[V]
                        if not args:
                            raise TypeError(
                                f"Cannot resolve path through unparameterized StringMapValue "
                                f"(at path segment {i})"
                            )
                        current_type = args[0]
                    else:
                        type_name = getattr(current_type, "__name__", str(current_type))
                        raise TypeError(
                            f"Cannot access string key '{segment}' on {type_name} "
                            f"(at path segment {i})"
                        )
                except Exception:
                    type_name = getattr(current_type, "__name__", str(current_type))
                    raise TypeError(
                        f"Cannot access string key '{segment}' on {type_name} "
                        f"(at path segment {i})"
                    )
        elif isinstance(segment, int):
            # Integer index - check if it's a SequenceValue using get_origin_and_args
            try:
                from .values import get_origin_and_args as value_get_origin_and_args

                origin, args = value_get_origin_and_args(current_type)
                if origin is SequenceValue:
                    # Get item type from SequenceValue[T]
                    if not args:
                        raise TypeError(
                            f"Cannot resolve path through unparameterized SequenceValue "
                            f"(at path segment {i})"
                        )
                    current_type = args[0]
                else:
                    type_name = getattr(current_type, "__name__", str(current_type))
                    raise TypeError(
                        f"Cannot access integer index {segment} on {type_name} "
                        f"(at path segment {i})"
                    )
            except Exception as e:
                if isinstance(e, TypeError) and "Cannot access integer index" in str(e):
                    raise
                type_name = getattr(current_type, "__name__", str(current_type))
                raise TypeError(
                    f"Cannot access integer index {segment} on {type_name} "
                    f"(at path segment {i})"
                )
        else:
            raise TypeError(
                f"Invalid path segment type: {type(segment)} (at path segment {i})"
            )

    return current_type


def validate_path(value_type: ValueType, path: FieldPath) -> bool:
    """
    Check if a path is valid for a given Value type.

    Args:
        value_type: Starting Value type
        path: Path to validate

    Returns:
        True if the path is valid, False otherwise

    Examples:
        >>> class MyData(Data):
        ...     foo: StringValue
        >>> validate_path(MyData, "foo")
        True
        >>> validate_path(MyData, "bar")
        False
    """
    try:
        resolve_path_type(value_type, path)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


__all__ = [
    "FieldPath",
    "normalize_path",
    "PathSegment",
    "resolve_path_type",
    "traverse_value",
    "validate_path",
]
