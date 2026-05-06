import importlib
from collections.abc import Callable
from types import ModuleType
from typing import Any, TypeVar, overload

T = TypeVar("T")


@overload
def dynamic_import(
    module: str,
) -> ModuleType: ...


@overload
def dynamic_import(
    module: str,
    name: str,
    *,
    validate_predicate: Callable[[Any], bool] | None = None,
) -> Any: ...


@overload
def dynamic_import(
    module: str,
    name: str | None = None,
    *,
    validate_instance: type[T],
    validate_predicate: Callable[[T], bool] | None = None,
) -> T: ...


@overload
def dynamic_import(
    module: str,
    name: str | None = None,
    *,
    validate_subclass: type[T],
    validate_predicate: Callable[[type[T]], bool] | None = None,
) -> type[T]: ...


def dynamic_import(
    module: str,
    name: str | None = None,
    *,
    validate_instance: type[T] | None = None,
    validate_subclass: type[T] | None = None,
    validate_predicate: Callable[[Any], bool] | None = None,
) -> ModuleType | Any | T | type[T]:
    """Dynamically import:

    If `name` is not provided, the module is imported and returned. Roughly:
    ```python
    import {module}
    return {module}
    ```

    If `name` is provided, the name is imported from the module. Roughly:
    ```python
    from {module} import {name}
    return {name}
    ```

    Thus, roughly, the imported object can be thought of as `{module}.{name}`.
    """

    imported = importlib.import_module(module)
    if name is not None:
        imported = getattr(imported, name)

    imported_name = f"{module}.{name}" if name is not None else module
    if validate_instance is not None and not isinstance(
        imported,
        validate_instance,
    ):
        raise ValueError(
            f"{imported_name} is not an instance of {validate_instance.__name__}"
        )
    if validate_subclass is not None:
        if not isinstance(imported, type):
            raise ValueError(
                f"{imported_name} is not a type but a {type(imported).__name__}"
            )
        if not issubclass(imported, validate_subclass):
            raise ValueError(
                f"{imported_name} is not a subclass of {validate_subclass.__name__}"
            )
    if validate_predicate is not None:
        if not validate_predicate(imported):
            raise ValueError(f"{imported_name} does not satisfy the predicate")
    return imported


__all__ = [
    "dynamic_import",
]
