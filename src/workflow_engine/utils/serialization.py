from typing import Any, Self, TextIO

from pydantic import BaseModel
from yaml import dump as yaml_dump
from yaml import load as yaml_load

try:
    from yaml import CSafeDumper as YamlDumper
    from yaml import CSafeLoader as YamlLoader
except ImportError:
    from yaml import SafeDumper as YamlDumper
    from yaml import SafeLoader as YamlLoader


def load_yaml(stream: TextIO) -> Any:
    return yaml_load(stream, Loader=YamlLoader)


def loads_yaml(s: str) -> Any:
    return yaml_load(s, Loader=YamlLoader)


def dump_yaml(data: Any, stream: TextIO) -> None:
    yaml_dump(data, stream, Dumper=YamlDumper)


def dumps_yaml(data: Any) -> str:
    return yaml_dump(data, Dumper=YamlDumper)


class PydanticYamlMixin:
    @classmethod
    def model_validate_yaml(cls, stream: TextIO | str) -> Self:
        assert issubclass(cls, BaseModel)
        if isinstance(stream, str):
            return cls.model_validate(loads_yaml(stream))
        return cls.model_validate(load_yaml(stream))

    def model_dump_yaml(self) -> str:
        assert isinstance(self, BaseModel)
        return dumps_yaml(self.model_dump(mode="json"))


__all__ = [
    "PydanticYamlMixin",
    "dump_yaml",
    "dumps_yaml",
    "load_yaml",
    "loads_yaml",
]
