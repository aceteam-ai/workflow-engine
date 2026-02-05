from typing import ClassVar

import pytest
from pydantic import ConfigDict, ValidationError

from workflow_engine.utils.immutable import (
    ImmutableBaseModel,
    ImmutableRootModel,
)


@pytest.fixture
def TestImmutableModel():
    """Test class that inherits from ImmutableBaseModel."""

    class TestImmutableModel(ImmutableBaseModel):
        name: str
        age: int
        active: bool = True

    return TestImmutableModel


@pytest.fixture
def TestImmutableRootModel():
    """Test class that inherits from ImmutableRootModel."""

    class TestImmutableRootModel(ImmutableRootModel[str]):
        pass

    return TestImmutableRootModel


@pytest.mark.unit
def test_model_config_inheritance():
    """Test that each subclass gets its own model config copy."""

    class ModelA(ImmutableBaseModel):
        model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    class ModelB(ImmutableBaseModel):
        model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    assert ModelA.model_config.get("extra") == "allow"
    assert ModelB.model_config.get("extra") == "forbid"

    assert ModelA.model_config.get("frozen") is True
    assert ModelB.model_config.get("frozen") is True

    assert ModelA.model_config.get("validate_assignment") is True
    assert ModelB.model_config.get("validate_assignment") is True


@pytest.mark.unit
def test_immutable_base_model_immutability(TestImmutableModel):
    """Test that ImmutableBaseModel instances are immutable."""
    model = TestImmutableModel(name="John", age=30)

    # Attempting to modify attributes should raise an error
    with pytest.raises(ValidationError):
        model.name = "Jane"
    with pytest.raises(ValidationError):
        model.age = 25

    # Original values should remain unchanged
    assert model.name == "John"
    assert model.age == 30


@pytest.mark.unit
def test_immutable_root_model_immutability(TestImmutableRootModel):
    """Test that ImmutableRootModel instances are immutable."""
    model = TestImmutableRootModel(root="test string")

    # Attempting to modify the root should raise an error
    with pytest.raises(ValidationError):
        model.root = "modified string"

    # Original value should remain unchanged
    assert model.root == "test string"


@pytest.mark.unit
def test_model_update_method(TestImmutableModel):
    """Test that model_update method creates a new instance with updated values."""
    original = TestImmutableModel(name="John", age=30, active=False)
    updated = original.model_update(name="Jane", age=25)

    # referentially different
    assert updated is not original

    # Original should remain unchanged
    assert original.name == "John"
    assert original.age == 30
    assert original.active is False

    # Updated should have new values
    assert updated.name == "Jane"
    assert updated.age == 25
    assert updated.active is False


@pytest.mark.unit
def test_model_mutate_with_immutable_base_model(TestImmutableModel):
    """Test that _model_mutate works with ImmutableBaseModel."""
    model = TestImmutableModel(name="John", age=30, active=False)

    # Use _model_mutate to update fields
    model._model_mutate(name="Jane", age=25)
    assert model.name == "Jane"
    assert model.age == 25
    assert model.active is False

    # Model remains immutable
    with pytest.raises(ValidationError):
        model.name = "Modified"
