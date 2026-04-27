import io

import pytest
from pydantic import BaseModel

from workflow_engine.utils.model import BaseModel as WEBaseModel
from workflow_engine.utils.model import RootModel as WERootModel
from workflow_engine.utils.serialization import (
    PydanticTomlMixin,
    PydanticYamlMixin,
    dumps_yaml,
    load_yaml,
    loads_yaml,
)


class Sample(WEBaseModel):
    name: str
    age: int


class SampleRoot(WERootModel[list[int]]):
    pass


class SampleDictRoot(WERootModel[dict[str, int]]):
    pass


def test_loads_yaml_roundtrip():
    data = {"a": 1, "b": [1, 2, 3]}
    assert loads_yaml(dumps_yaml(data)) == data


def test_load_yaml_from_stream():
    yaml_str = "a: 1\nb: 2\n"
    assert load_yaml(io.StringIO(yaml_str)) == {"a": 1, "b": 2}


def test_model_validate_yaml_from_string():
    model = Sample.model_validate_yaml("name: alice\nage: 30\n")
    assert model == Sample(name="alice", age=30)


def test_model_validate_yaml_from_stream():
    model = Sample.model_validate_yaml(io.StringIO("name: bob\nage: 25\n"))
    assert model == Sample(name="bob", age=25)


def test_model_dump_yaml_roundtrip():
    original = Sample(name="carol", age=42)
    dumped = original.model_dump_yaml()
    assert Sample.model_validate_yaml(dumped) == original


def test_model_validate_yaml_root_model():
    model = SampleRoot.model_validate_yaml("[1, 2, 3]")
    assert model.root == [1, 2, 3]


def test_model_dump_yaml_root_model_roundtrip():
    original = SampleRoot(root=[4, 5, 6])
    assert SampleRoot.model_validate_yaml(original.model_dump_yaml()) == original


def test_model_validate_yaml_requires_base_model():
    class NotAModel(PydanticYamlMixin):
        pass

    with pytest.raises(TypeError, match="requires cls to inherit from BaseModel"):
        NotAModel.model_validate_yaml("foo: bar")


def test_model_dump_yaml_requires_base_model_instance():
    class NotAModel(PydanticYamlMixin):
        pass

    with pytest.raises(TypeError, match="requires self to be a BaseModel instance"):
        NotAModel().model_dump_yaml()


def test_model_validate_yaml_validation_error():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Sample.model_validate_yaml("name: dave\nage: not_a_number\n")


def test_pydantic_yaml_mixin_is_inherited():
    assert issubclass(Sample, PydanticYamlMixin)
    assert issubclass(Sample, BaseModel)


def test_model_validate_toml_from_string():
    model = Sample.model_validate_toml('name = "alice"\nage = 30\n')
    assert model == Sample(name="alice", age=30)


def test_model_validate_toml_from_stream():
    model = Sample.model_validate_toml(io.StringIO('name = "bob"\nage = 25\n'))
    assert model == Sample(name="bob", age=25)


def test_model_dump_toml_roundtrip():
    original = Sample(name="carol", age=42)
    dumped = original.model_dump_toml()
    assert Sample.model_validate_toml(dumped) == original


def test_model_validate_toml_dict_root_model():
    model = SampleDictRoot.model_validate_toml("a = 1\nb = 2\n")
    assert model.root == {"a": 1, "b": 2}


def test_model_dump_toml_dict_root_model_roundtrip():
    original = SampleDictRoot(root={"x": 1, "y": 2})
    assert SampleDictRoot.model_validate_toml(original.model_dump_toml()) == original


def test_model_validate_toml_requires_base_model():
    class NotAModel(PydanticTomlMixin):
        pass

    with pytest.raises(TypeError, match="requires cls to inherit from BaseModel"):
        NotAModel.model_validate_toml('foo = "bar"')


def test_model_dump_toml_requires_base_model_instance():
    class NotAModel(PydanticTomlMixin):
        pass

    with pytest.raises(TypeError, match="requires self to be a BaseModel instance"):
        NotAModel().model_dump_toml()


def test_model_validate_toml_validation_error():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Sample.model_validate_toml('name = "dave"\nage = "not_a_number"\n')


def test_model_dump_toml_non_mapping_raises():
    original = SampleRoot(root=[1, 2, 3])
    with pytest.raises(
        TypeError, match="Expected the serialized object to be a Mapping"
    ):
        original.model_dump_toml()


def test_pydantic_toml_mixin_is_inherited():
    assert issubclass(Sample, PydanticTomlMixin)
    assert issubclass(Sample, BaseModel)
