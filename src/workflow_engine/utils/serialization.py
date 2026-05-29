from collections.abc import Mapping
from io import StringIO
from tomllib import load as load_toml
from tomllib import loads as loads_toml
from typing import Any, Self, TextIO

from pydantic import BaseModel
from pydantic.config import ExtraValues
from ruamel.yaml import YAML
from tomli_w import dump as dump_toml
from tomli_w import dumps as dumps_toml

# Plain-data YAML (not round-trip): loads to / dumps from native Python types,
# the right mode for model serialization where there are no comments to keep.
# Block style, and key sorting disabled so a model's field order (and a dict's
# insertion order) survive a dump — matching pyyaml's `sort_keys=False`.
_yaml = YAML(typ="safe")
_yaml.default_flow_style = False
_yaml.representer.sort_base_mapping_type_on_output = False


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
    return _yaml.load(stream)


def loads_yaml(s: str) -> Any:
    return _yaml.load(s)


def dump_yaml(data: Any, stream: TextIO) -> None:
    _yaml.dump(data, stream)


def dumps_yaml(data: Any) -> str:
    buffer = StringIO()
    _yaml.dump(data, buffer)
    return buffer.getvalue()


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
