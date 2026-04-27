from collections.abc import Mapping
from tomllib import load as load_toml
from tomllib import loads as loads_toml
from typing import Any, Self, TextIO

from pydantic import BaseModel
from pydantic.config import ExtraValues
from tomli_w import dump as dump_toml
from tomli_w import dumps as dumps_toml
from yaml import dump as _yaml_dump
from yaml import load as _yaml_load

try:
    from yaml import CSafeDumper as YamlDumper
    from yaml import CSafeLoader as YamlLoader
except ImportError:
    from yaml import SafeDumper as YamlDumper
    from yaml import SafeLoader as YamlLoader


class PydanticTomlMixin:
    @classmethod
    def model_validate_toml(
        cls,
        toml_data: TextIO | str,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        if not issubclass(cls, BaseModel):
            raise TypeError(
                f"{cls.__name__}.model_validate_toml() requires cls to inherit from BaseModel"
            )
        if not isinstance(toml_data, str):
            toml_data = toml_data.read()
        obj = loads_toml(toml_data)
        return cls.model_validate(
            obj,
            strict=strict,
            extra=extra,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    def model_dump_toml(self) -> str:
        if not isinstance(self, BaseModel):
            raise TypeError(
                f"{self.__class__.__name__}.model_dump_toml() requires self to be a BaseModel instance"
            )
        obj = self.model_dump(mode="json")
        if not isinstance(obj, Mapping):
            raise TypeError(
                f"Expected the serialized object to be a Mapping, got {type(obj)}"
            )
        return dumps_toml(obj)


def load_yaml(stream: TextIO) -> Any:
    return _yaml_load(stream, Loader=YamlLoader)


def loads_yaml(s: str) -> Any:
    return _yaml_load(s, Loader=YamlLoader)


def dump_yaml(data: Any, stream: TextIO) -> None:
    _yaml_dump(data, stream, Dumper=YamlDumper)


def dumps_yaml(data: Any) -> str:
    return _yaml_dump(data, Dumper=YamlDumper)


class PydanticYamlMixin:
    @classmethod
    def model_validate_yaml(
        cls,
        yaml_data: TextIO | str,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        if not issubclass(cls, BaseModel):
            raise TypeError(
                f"{cls.__name__}.model_validate_yaml() requires cls to inherit from BaseModel"
            )
        if isinstance(yaml_data, str):
            obj = loads_yaml(yaml_data)
        else:
            obj = load_yaml(yaml_data)
        return cls.model_validate(
            obj,
            strict=strict,
            extra=extra,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    def model_dump_yaml(self) -> str:
        if not isinstance(self, BaseModel):
            raise TypeError(
                f"{self.__class__.__name__}.model_dump_yaml() requires self to be a BaseModel instance"
            )
        return dumps_yaml(self.model_dump(mode="json"))


__all__ = [
    "PydanticTomlMixin",
    "PydanticYamlMixin",
    "dump_toml",
    "dump_yaml",
    "dumps_toml",
    "dumps_yaml",
    "load_toml",
    "load_yaml",
    "loads_toml",
    "loads_yaml",
]
