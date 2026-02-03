"""Tests for ValueRegistry implementations."""

import pytest

from workflow_engine.core.values import Value
from workflow_engine.core.values.value import (
    EagerValueRegistryBuilder,
    ImmutableValueRegistry,
    LazyValueRegistry,
    ValueRegistry,
    ValueRegistryBuilder,
    default_value_registry,
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
            value_classes={"SampleA": SampleValueA, "SampleB": SampleValueB}
        )
        assert registry is not None

    def test_get_value_class(self):
        """Test retrieving value classes by name."""
        registry = ImmutableValueRegistry(
            value_classes={"SampleA": SampleValueA, "SampleB": SampleValueB}
        )

        assert registry.get_value_class("SampleA") is SampleValueA
        assert registry.get_value_class("SampleB") is SampleValueB

    def test_get_value_class_not_found(self):
        """Test that ValueError is raised for missing value types."""
        registry = ImmutableValueRegistry(value_classes={"SampleA": SampleValueA})

        with pytest.raises(
            ValueError, match='Value type "NonExistent" is not registered'
        ):
            registry.get_value_class("NonExistent")

    def test_contains_checks_existence(self):
        """Test __contains__ for checking if a value type is registered."""
        registry = ImmutableValueRegistry(
            value_classes={"SampleA": SampleValueA, "SampleB": SampleValueB}
        )

        assert registry.has_name("SampleA") is True
        assert registry.has_name("SampleB") is True
        assert registry.has_name("NonExistent") is False

    def test_getitem_syntax(self):
        """Test indexer syntax for retrieving value types."""
        registry = ImmutableValueRegistry(value_classes={"SampleA": SampleValueA})

        assert registry.get_value_class("SampleA") is SampleValueA

    def test_immutability(self):
        """Test that the registry data cannot be modified after construction."""
        value_classes = {"SampleA": SampleValueA}
        registry = ImmutableValueRegistry(value_classes=value_classes)

        # Modifying the original dict should not affect the registry
        value_classes = {"SampleA": SampleValueA, "SampleB": SampleValueB}

        assert registry.has_name("SampleB") is False


class TestEagerValueRegistryBuilder:
    """Tests for EagerValueRegistryBuilder."""

    def test_register_value_class(self):
        """Test registering value classes."""
        builder = EagerValueRegistryBuilder()
        builder.register_value_class("SampleA", SampleValueA)
        builder.register_value_class("SampleB", SampleValueB)

        registry = builder.build()
        assert registry.get_value_class("SampleA") is SampleValueA
        assert registry.get_value_class("SampleB") is SampleValueB

    def test_fluent_interface(self):
        """Test that methods return self for chaining."""
        builder = EagerValueRegistryBuilder()
        result = builder.register_value_class(
            "SampleA", SampleValueA
        ).register_value_class("SampleB", SampleValueB)

        assert result is builder

    def test_duplicate_same_class_succeeds(self):
        """Test that registering the same class twice succeeds (idempotent)."""
        builder = EagerValueRegistryBuilder()
        builder.register_value_class("SampleA", SampleValueA)
        builder.register_value_class("SampleA", SampleValueA)  # Same class

        registry = builder.build()
        assert registry.get_value_class("SampleA") is SampleValueA

    def test_duplicate_different_class_raises_error(self):
        """Test that registering a different class to the same name raises ValueError."""
        builder = EagerValueRegistryBuilder()
        builder.register_value_class("Sample", SampleValueA)

        with pytest.raises(
            ValueError,
            match='Value type "Sample" .* is already registered to a different class',
        ):
            builder.register_value_class("Sample", SampleValueB)

    def test_build_returns_immutable_registry(self):
        """Test that build() returns an ImmutableValueRegistry."""
        builder = EagerValueRegistryBuilder()
        builder.register_value_class("SampleA", SampleValueA)

        registry = builder.build()
        assert isinstance(registry, ImmutableValueRegistry)
        assert not isinstance(registry, ValueRegistryBuilder)

    def test_register_logs_debug_message(self, caplog):
        """Test that registration logs a debug message."""
        import logging

        caplog.set_level(logging.DEBUG)

        builder = EagerValueRegistryBuilder()
        builder.register_value_class("SampleA", SampleValueA)

        assert any(
            "Registering class SampleValueA as value type SampleA" in record.message
            for record in caplog.records
        )


class TestLazyValueRegistry:
    """Tests for LazyValueRegistry."""

    def test_starts_unfrozen(self):
        """Test that LazyValueRegistry starts in unfrozen state."""
        registry = LazyValueRegistry()
        # Should be able to register without error
        registry.register_value_class("SampleA", SampleValueA)
        assert True  # No exception means it's unfrozen

    def test_register_value_class_when_unfrozen(self):
        """Test registering value classes when unfrozen."""
        registry = LazyValueRegistry()
        registry.register_value_class("SampleA", SampleValueA)
        registry.register_value_class("SampleB", SampleValueB)

        # Trigger build by accessing
        assert registry.get_value_class("SampleA") is SampleValueA
        assert registry.get_value_class("SampleB") is SampleValueB

    def test_fluent_interface(self):
        """Test that methods return self for chaining."""
        registry = LazyValueRegistry()
        result = registry.register_value_class(
            "SampleA", SampleValueA
        ).register_value_class("SampleB", SampleValueB)

        assert result is registry

    def test_build_freezes_registry(self):
        """Test that build() freezes the registry."""
        registry = LazyValueRegistry()
        registry.register_value_class("SampleA", SampleValueA)

        # Build should freeze
        registry.build()

        # Should not be able to register after freezing
        with pytest.raises(ValueError, match="Value registry is frozen"):
            registry.register_value_class("SampleB", SampleValueB)

    def test_access_triggers_build(self):
        """Test that accessing the registry triggers build."""
        registry = LazyValueRegistry()
        registry.register_value_class("SampleA", SampleValueA)

        # Access should trigger build
        registry.get_value_class("SampleA")

        # Should be frozen now
        with pytest.raises(ValueError, match="Value registry is frozen"):
            registry.register_value_class("SampleB", SampleValueB)

    def test_contains_triggers_build(self):
        """Test that __contains__ triggers build."""
        registry = LazyValueRegistry()
        registry.register_value_class("SampleA", SampleValueA)

        # Access should trigger build
        _ = registry.has_name("SampleA")

        # Should be frozen now
        with pytest.raises(ValueError, match="Value registry is frozen"):
            registry.register_value_class("SampleB", SampleValueB)

    def test_build_is_idempotent(self):
        """Test that calling build() multiple times is safe."""
        registry = LazyValueRegistry()
        registry.register_value_class("SampleA", SampleValueA)

        result1 = registry.build()
        result2 = registry.build()

        assert result1 is result2
        assert result1 is registry

    def test_duplicate_same_class_logs_warning(self, caplog):
        """Test that registering the same class twice logs a warning but succeeds."""
        registry = LazyValueRegistry()
        registry.register_value_class("SampleA", SampleValueA)
        registry.register_value_class("SampleA", SampleValueA)  # Same class

        # Build should log warning but not raise
        registry.build()

        # Check that warning was logged
        assert any(
            "Value type SampleA is already registered" in record.message
            for record in caplog.records
        )

    def test_duplicate_different_class_raises_error(self):
        """Test that registering a different class to the same name raises ValueError."""
        registry = LazyValueRegistry()
        registry.register_value_class("Sample", SampleValueA)
        registry.register_value_class("Sample", SampleValueB)  # Different class

        # Build should raise error
        with pytest.raises(
            ValueError,
            match='Value type "Sample" .* is already registered to a different class',
        ):
            registry.build()

    def test_get_value_class_not_found(self):
        """Test that ValueError is raised for missing value types."""
        registry = LazyValueRegistry()
        registry.register_value_class("SampleA", SampleValueA)

        with pytest.raises(
            ValueError, match='Value type "NonExistent" is not registered'
        ):
            registry.get_value_class("NonExistent")

    def test_lazy_registry_is_also_builder(self):
        """Test that LazyValueRegistry is both a registry and a builder."""
        registry = LazyValueRegistry()

        assert isinstance(registry, ValueRegistry)
        assert isinstance(registry, ValueRegistryBuilder)

    def test_build_returns_self(self):
        """Test that build() returns self (the LazyValueRegistry itself)."""
        registry = LazyValueRegistry()
        registry.register_value_class("SampleA", SampleValueA)

        result = registry.build()
        assert result is registry
        assert isinstance(result, ValueRegistry)

    def test_registrations_deleted_after_build(self):
        """Test that _registrations list is deleted after build to save memory."""
        registry = LazyValueRegistry()
        registry.register_value_class("SampleA", SampleValueA)

        assert hasattr(registry, "_registrations")
        registry.build()

        # _registrations should be deleted
        assert not hasattr(registry, "_registrations")

    def test_getitem_syntax(self):
        """Test indexer syntax for retrieving value types."""
        registry = LazyValueRegistry()
        registry.register_value_class("SampleA", SampleValueA)

        assert registry.get_value_class("SampleA") is SampleValueA


class TestValueRegistryIntegration:
    """Integration tests with actual Value classes."""

    def test_global_registry_contains_builtin_types(self):
        """Test that the global default_value_registry contains built-in value types."""
        # These types should have been auto-registered during import
        assert default_value_registry.has_name("IntegerValue")
        assert default_value_registry.has_name("StringValue")
        assert default_value_registry.has_name("FloatValue")
        assert default_value_registry.has_name("BooleanValue")

    def test_retrieve_builtin_types_from_global_registry(self):
        """Test retrieving built-in value types from global registry."""
        from workflow_engine.core.values import (
            BooleanValue,
            FloatValue,
            IntegerValue,
            StringValue,
        )

        assert default_value_registry.get_value_class("IntegerValue") is IntegerValue
        assert default_value_registry.get_value_class("StringValue") is StringValue
        assert default_value_registry.get_value_class("FloatValue") is FloatValue
        assert default_value_registry.get_value_class("BooleanValue") is BooleanValue

    def test_with_concrete_value_classes(self):
        """Test registry with real value classes."""
        registry = LazyValueRegistry()

        # Create isolated registry for testing
        registry.register_value_class("SampleA", SampleValueA)
        registry.register_value_class("SampleB", SampleValueB)

        # Retrieve and verify
        assert registry.get_value_class("SampleA") is SampleValueA
        assert registry.get_value_class("SampleB") is SampleValueB

    def test_mixed_registration(self):
        """Test registry with multiple value classes."""
        registry = LazyValueRegistry()

        registry.register_value_class("SampleA", SampleValueA)
        registry.register_value_class("SampleB", SampleValueB)

        registry.build()

        assert registry.has_name("SampleA")
        assert registry.get_value_class("SampleA") is SampleValueA
        assert registry.has_name("SampleB")
        assert registry.get_value_class("SampleB") is SampleValueB


class TestValueRegistryEdgeCases:
    """Edge case tests for ValueRegistry."""

    def test_register_with_different_name_than_class_name(self):
        """Test registering a class with a name different from its __name__."""
        registry = LazyValueRegistry()
        registry.register_value_class("CustomName", SampleValueA)

        assert registry.has_name("CustomName")
        assert registry.get_value_class("CustomName") is SampleValueA

    def test_multiple_registries_are_independent(self):
        """Test that multiple LazyValueRegistry instances are independent."""
        registry1 = LazyValueRegistry()
        registry2 = LazyValueRegistry()

        registry1.register_value_class("SampleA", SampleValueA)
        registry2.register_value_class("SampleB", SampleValueB)

        assert registry1.has_name("SampleA")
        assert registry1.get_value_class("SampleA") is SampleValueA
        assert registry2.has_name("SampleB")
        assert registry2.get_value_class("SampleB") is SampleValueB
        assert registry2.has_name("SampleA") is False
        with pytest.raises(ValueError, match='Value type "SampleA" is not registered'):
            registry2.get_value_class("SampleA")
