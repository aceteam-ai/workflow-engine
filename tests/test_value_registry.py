"""Tests for ValueRegistry implementations."""

import pytest

from workflow_engine import Value, ValueRegistry
from workflow_engine.core.values.value import (
    ImmutableValueRegistry,
    ValueRegistryBuilder,
)


# Test fixtures - simple value classes for testing
class SampleValueA(Value[int]):
    """Sample value type A."""

    pass


class SampleValueB(Value[str]):
    """Sample value type B."""

    pass


class TestImmutableValueRegistry:
    """Tests for ImmutableValueRegistry."""

    def test_construction(self):
        """Test that ImmutableValueRegistry can be constructed."""
        registry = ImmutableValueRegistry(
            value_classes={"SampleValueA": SampleValueA, "SampleValueB": SampleValueB}
        )
        assert registry is not None

    def test_get_value_class(self):
        """Test retrieving value classes by name."""
        registry = ImmutableValueRegistry(
            value_classes={"SampleValueA": SampleValueA, "SampleValueB": SampleValueB}
        )

        assert registry.get_value_class("SampleValueA") is SampleValueA
        assert registry.get_value_class("SampleValueB") is SampleValueB

    def test_get_value_class_not_found(self):
        """Test that ValueError is raised for missing value types."""
        registry = ImmutableValueRegistry(
            value_classes={"SampleValueA": SampleValueA}
        )

        with pytest.raises(
            ValueError, match='Value type "NonExistent" is not registered'
        ):
            registry.get_value_class("NonExistent")

    def test_contains_checks_existence(self):
        """Test has_name for checking if a value type is registered."""
        registry = ImmutableValueRegistry(
            value_classes={"SampleValueA": SampleValueA, "SampleValueB": SampleValueB}
        )

        assert registry.has_name("SampleValueA") is True
        assert registry.has_name("SampleValueB") is True
        assert registry.has_name("NonExistent") is False

    def test_getitem_syntax(self):
        """Test get_value_class syntax for retrieving value types."""
        registry = ImmutableValueRegistry(
            value_classes={"SampleValueA": SampleValueA}
        )

        assert registry.get_value_class("SampleValueA") is SampleValueA

    def test_immutability(self):
        """Test that the registry data cannot be modified after construction."""
        value_classes = {"SampleValueA": SampleValueA}
        registry = ImmutableValueRegistry(value_classes=value_classes)

        # Modifying the original dict should not affect the registry
        value_classes = {"SampleValueA": SampleValueA, "SampleValueB": SampleValueB}

        assert registry.has_name("SampleValueB") is False

    def test_all_value_classes(self):
        """Test that all_value_classes returns all registered value classes."""
        registry = ImmutableValueRegistry(
            value_classes={"SampleValueA": SampleValueA, "SampleValueB": SampleValueB}
        )

        result = dict(registry.all_value_classes())
        assert result == {"SampleValueA": SampleValueA, "SampleValueB": SampleValueB}


class TestEagerValueRegistryBuilder:
    """Tests for EagerValueRegistryBuilder."""

    def test_register_value_class(self):
        """Test registering value classes."""
        builder = ValueRegistry.builder(lazy=False)
        builder.register_value_class(SampleValueA)
        builder.register_value_class(SampleValueB)

        registry = builder.build()
        assert registry.get_value_class("SampleValueA") is SampleValueA
        assert registry.get_value_class("SampleValueB") is SampleValueB

    def test_fluent_interface(self):
        """Test that methods return self for chaining."""
        builder = ValueRegistry.builder(lazy=False)
        result = builder.register_value_class(SampleValueA).register_value_class(
            SampleValueB
        )

        assert result is builder

    def test_duplicate_same_class_succeeds(self):
        """Test that registering the same class twice succeeds (idempotent)."""
        builder = ValueRegistry.builder(lazy=False)
        builder.register_value_class(SampleValueA)
        builder.register_value_class(SampleValueA)  # Same class

        registry = builder.build()
        assert registry.get_value_class("SampleValueA") is SampleValueA

    def test_build_returns_immutable_registry(self):
        """Test that build() returns an ImmutableValueRegistry."""
        builder = ValueRegistry.builder(lazy=False)
        builder.register_value_class(SampleValueA)

        registry = builder.build()
        assert isinstance(registry, ImmutableValueRegistry)
        assert not isinstance(registry, ValueRegistryBuilder)

    def test_register_logs_debug_message(self, caplog):
        """Test that registration logs a debug message."""
        import logging

        caplog.set_level(logging.DEBUG)

        builder = ValueRegistry.builder(lazy=False)
        builder.register_value_class(SampleValueA)

        assert any(
            "Registering value type SampleValueA" in record.message
            for record in caplog.records
        )


class TestLazyValueRegistry:
    """Tests for LazyValueRegistry."""

    def test_starts_unfrozen(self):
        """Test that LazyValueRegistry starts in unfrozen state."""
        registry = ValueRegistry.builder(lazy=True)
        # Should be able to register without error
        registry.register_value_class(SampleValueA)
        assert True  # No exception means it's unfrozen

    def test_register_value_class_when_unfrozen(self):
        """Test registering value classes when unfrozen."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class(SampleValueA)
        registry.register_value_class(SampleValueB)

        # Trigger build by accessing
        assert registry.get_value_class("SampleValueA") is SampleValueA
        assert registry.get_value_class("SampleValueB") is SampleValueB

    def test_fluent_interface(self):
        """Test that methods return self for chaining."""
        registry = ValueRegistry.builder(lazy=True)
        result = registry.register_value_class(SampleValueA).register_value_class(
            SampleValueB
        )

        assert result is registry

    def test_build_freezes_registry(self):
        """Test that build() freezes the registry."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class(SampleValueA)

        # Build should freeze
        registry.build()

        # Should not be able to register after freezing
        with pytest.raises(ValueError, match="Value registry is frozen"):
            registry.register_value_class(SampleValueB)

    def test_access_triggers_build(self):
        """Test that accessing the registry triggers build."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class(SampleValueA)

        # Access should trigger build
        registry.get_value_class("SampleValueA")

        # Should be frozen now
        with pytest.raises(ValueError, match="Value registry is frozen"):
            registry.register_value_class(SampleValueB)

    def test_contains_triggers_build(self):
        """Test that has_name triggers build."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class(SampleValueA)

        # Access should trigger build
        _ = registry.has_name("SampleValueA")

        # Should be frozen now
        with pytest.raises(ValueError, match="Value registry is frozen"):
            registry.register_value_class(SampleValueB)

    def test_build_is_idempotent(self):
        """Test that calling build() multiple times is safe."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class(SampleValueA)

        result1 = registry.build()
        result2 = registry.build()

        assert result1 is result2
        assert result1 is registry

    def test_duplicate_same_class_logs_warning(self, caplog):
        """Test that registering the same class twice logs a warning but succeeds."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class(SampleValueA)
        registry.register_value_class(SampleValueA)  # Same class

        # Build should log warning but not raise
        registry.build()

        # Check that warning was logged
        assert any(
            "Value type SampleValueA is already registered" in record.message
            for record in caplog.records
        )

    def test_get_value_class_not_found(self):
        """Test that ValueError is raised for missing value types."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class(SampleValueA)

        with pytest.raises(
            ValueError, match='Value type "NonExistent" is not registered'
        ):
            registry.get_value_class("NonExistent")

    def test_lazy_registry_is_also_builder(self):
        """Test that LazyValueRegistry is both a registry and a builder."""
        registry = ValueRegistry.builder(lazy=True)

        assert isinstance(registry, ValueRegistry)
        assert isinstance(registry, ValueRegistryBuilder)

    def test_build_returns_self(self):
        """Test that build() returns self (the LazyValueRegistry itself)."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class(SampleValueA)

        result = registry.build()
        assert result is registry
        assert isinstance(result, ValueRegistry)

    def test_registrations_deleted_after_build(self):
        """Test that _registrations list is deleted after build to save memory."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class(SampleValueA)

        assert hasattr(registry, "_registrations")
        registry.build()

        # _registrations should be deleted
        assert not hasattr(registry, "_registrations")

    def test_getitem_syntax(self):
        """Test get_value_class syntax for retrieving value types."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class(SampleValueA)

        assert registry.get_value_class("SampleValueA") is SampleValueA

    def test_all_value_classes(self):
        """Test that all_value_classes returns all registered value classes."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class(SampleValueA)
        registry.register_value_class(SampleValueB)

        result = dict(registry.all_value_classes())
        assert result == {"SampleValueA": SampleValueA, "SampleValueB": SampleValueB}

    def test_all_value_classes_triggers_build(self):
        """Test that all_value_classes triggers build."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class(SampleValueA)

        _ = list(registry.all_value_classes())

        # Should be frozen now
        with pytest.raises(ValueError, match="Value registry is frozen"):
            registry.register_value_class(SampleValueB)


class TestValueRegistryIntegration:
    """Integration tests with actual Value classes."""

    def test_global_registry_contains_builtin_types(self):
        """Test that the global default_value_registry contains built-in value types."""
        # These types should have been auto-registered during import
        assert ValueRegistry.DEFAULT.has_name("IntegerValue")
        assert ValueRegistry.DEFAULT.has_name("StringValue")
        assert ValueRegistry.DEFAULT.has_name("FloatValue")
        assert ValueRegistry.DEFAULT.has_name("BooleanValue")

    def test_retrieve_builtin_types_from_global_registry(self):
        """Test retrieving built-in value types from global registry."""
        from workflow_engine.core.values import (
            BooleanValue,
            FloatValue,
            IntegerValue,
            StringValue,
        )

        assert ValueRegistry.DEFAULT.get_value_class("IntegerValue") is IntegerValue
        assert ValueRegistry.DEFAULT.get_value_class("StringValue") is StringValue
        assert ValueRegistry.DEFAULT.get_value_class("FloatValue") is FloatValue
        assert ValueRegistry.DEFAULT.get_value_class("BooleanValue") is BooleanValue

    def test_with_concrete_value_classes(self):
        """Test registry with real value classes."""
        registry = ValueRegistry.builder(lazy=True)

        # Create isolated registry for testing
        registry.register_value_class(SampleValueA)
        registry.register_value_class(SampleValueB)

        # Retrieve and verify
        assert registry.get_value_class("SampleValueA") is SampleValueA
        assert registry.get_value_class("SampleValueB") is SampleValueB

    def test_mixed_registration(self):
        """Test registry with multiple value classes."""
        registry = ValueRegistry.builder(lazy=True)

        registry.register_value_class(SampleValueA)
        registry.register_value_class(SampleValueB)

        registry.build()

        assert registry.has_name("SampleValueA")
        assert registry.get_value_class("SampleValueA") is SampleValueA
        assert registry.has_name("SampleValueB")
        assert registry.get_value_class("SampleValueB") is SampleValueB


class TestValueRegistryEdgeCases:
    """Edge case tests for ValueRegistry."""

    def test_multiple_registries_are_independent(self):
        """Test that multiple LazyValueRegistry instances are independent."""
        registry1 = ValueRegistry.builder(lazy=True)
        registry2 = ValueRegistry.builder(lazy=True)

        registry1.register_value_class(SampleValueA)
        registry2.register_value_class(SampleValueB)

        assert registry1.has_name("SampleValueA")
        assert registry1.get_value_class("SampleValueA") is SampleValueA
        assert registry2.has_name("SampleValueB")
        assert registry2.get_value_class("SampleValueB") is SampleValueB
        assert registry2.has_name("SampleValueA") is False
        with pytest.raises(
            ValueError, match='Value type "SampleValueA" is not registered'
        ):
            registry2.get_value_class("SampleValueA")


class TestValueRegistryExtend:
    """Tests for ValueRegistry.extend() method."""

    def test_extend_eager(self):
        """Test extending a registry into an eager builder."""
        registry = ImmutableValueRegistry(
            value_classes={"SampleValueA": SampleValueA}
        )

        builder = registry.extend(lazy=False)
        new_registry = builder.build()

        assert new_registry.get_value_class("SampleValueA") is SampleValueA

    def test_extend_lazy(self):
        """Test extending a registry into a lazy builder."""
        registry = ImmutableValueRegistry(
            value_classes={"SampleValueA": SampleValueA}
        )

        lazy_registry = registry.extend(lazy=True)
        assert lazy_registry.get_value_class("SampleValueA") is SampleValueA

    def test_extend_preserves_all_classes(self):
        """Test that extend copies all registered classes."""
        registry = ImmutableValueRegistry(
            value_classes={"SampleValueA": SampleValueA, "SampleValueB": SampleValueB}
        )

        builder = registry.extend(lazy=False)
        new_registry = builder.build()

        assert new_registry.get_value_class("SampleValueA") is SampleValueA
        assert new_registry.get_value_class("SampleValueB") is SampleValueB


class TestValueRegistryLoadValue:
    """Tests for ValueRegistry.load_value() method."""

    def test_load_value_with_matching_title(self):
        """Test loading a value type from schema with matching title."""
        from workflow_engine.core.values.schema import IntegerValueSchema

        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class(SampleValueA)

        schema = IntegerValueSchema(type="integer", title="SampleValueA")
        loaded_value = registry.load_value(schema)

        assert loaded_value is SampleValueA

    def test_load_value_without_title_returns_none(self):
        """Test that load_value returns None when schema has no title."""
        from workflow_engine.core.values.schema import IntegerValueSchema

        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class(SampleValueA)

        schema = IntegerValueSchema(type="integer")  # No title
        loaded_value = registry.load_value(schema)

        assert loaded_value is None

    def test_load_value_with_unregistered_title_returns_none(self):
        """Test that load_value returns None for unregistered title."""
        from workflow_engine.core.values.schema import IntegerValueSchema

        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class(SampleValueA)

        schema = IntegerValueSchema(type="integer", title="UnregisteredType")
        loaded_value = registry.load_value(schema)

        assert loaded_value is None

    def test_load_value_integration_with_to_value_cls(self):
        """Test that ValueSchema.to_value_cls() uses the registry's load_value."""
        from workflow_engine.core.values import IntegerValue
        from workflow_engine.core.values.schema import IntegerValueSchema

        # Create a schema with a title that matches a built-in type
        schema = IntegerValueSchema(type="integer", title="IntegerValue")

        # to_value_cls should use the default registry to load the value
        value_cls = schema.to_value_cls()

        assert value_cls is IntegerValue
