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
        builder = ValueRegistry.builder(lazy=False)
        builder.register_value_class("SampleA", SampleValueA)
        builder.register_value_class("SampleB", SampleValueB)

        registry = builder.build()
        assert registry.get_value_class("SampleA") is SampleValueA
        assert registry.get_value_class("SampleB") is SampleValueB

    def test_fluent_interface(self):
        """Test that methods return self for chaining."""
        builder = ValueRegistry.builder(lazy=False)
        result = builder.register_value_class(
            "SampleA", SampleValueA
        ).register_value_class("SampleB", SampleValueB)

        assert result is builder

    def test_duplicate_same_class_succeeds(self):
        """Test that registering the same class twice succeeds (idempotent)."""
        builder = ValueRegistry.builder(lazy=False)
        builder.register_value_class("SampleA", SampleValueA)
        builder.register_value_class("SampleA", SampleValueA)  # Same class

        registry = builder.build()
        assert registry.get_value_class("SampleA") is SampleValueA

    def test_duplicate_different_class_raises_error(self):
        """Test that registering a different class to the same name raises ValueError."""
        builder = ValueRegistry.builder(lazy=False)
        builder.register_value_class("Sample", SampleValueA)

        with pytest.raises(
            ValueError,
            match='Value type "Sample" .* is already registered to a different class',
        ):
            builder.register_value_class("Sample", SampleValueB)

    def test_build_returns_immutable_registry(self):
        """Test that build() returns an ImmutableValueRegistry."""
        builder = ValueRegistry.builder(lazy=False)
        builder.register_value_class("SampleA", SampleValueA)

        registry = builder.build()
        assert isinstance(registry, ImmutableValueRegistry)
        assert not isinstance(registry, ValueRegistryBuilder)

    def test_register_logs_debug_message(self, caplog):
        """Test that registration logs a debug message."""
        import logging

        caplog.set_level(logging.DEBUG)

        builder = ValueRegistry.builder(lazy=False)
        builder.register_value_class("SampleA", SampleValueA)

        assert any(
            "Registering class SampleValueA as value type SampleA" in record.message
            for record in caplog.records
        )


class TestLazyValueRegistry:
    """Tests for LazyValueRegistry."""

    def test_starts_unfrozen(self):
        """Test that LazyValueRegistry starts in unfrozen state."""
        registry = ValueRegistry.builder(lazy=True)
        # Should be able to register without error
        registry.register_value_class("SampleA", SampleValueA)
        assert True  # No exception means it's unfrozen

    def test_register_value_class_when_unfrozen(self):
        """Test registering value classes when unfrozen."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class("SampleA", SampleValueA)
        registry.register_value_class("SampleB", SampleValueB)

        # Trigger build by accessing
        assert registry.get_value_class("SampleA") is SampleValueA
        assert registry.get_value_class("SampleB") is SampleValueB

    def test_fluent_interface(self):
        """Test that methods return self for chaining."""
        registry = ValueRegistry.builder(lazy=True)
        result = registry.register_value_class(
            "SampleA", SampleValueA
        ).register_value_class("SampleB", SampleValueB)

        assert result is registry

    def test_build_freezes_registry(self):
        """Test that build() freezes the registry."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class("SampleA", SampleValueA)

        # Build should freeze
        registry.build()

        # Should not be able to register after freezing
        with pytest.raises(ValueError, match="Value registry is frozen"):
            registry.register_value_class("SampleB", SampleValueB)

    def test_access_triggers_build(self):
        """Test that accessing the registry triggers build."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class("SampleA", SampleValueA)

        # Access should trigger build
        registry.get_value_class("SampleA")

        # Should be frozen now
        with pytest.raises(ValueError, match="Value registry is frozen"):
            registry.register_value_class("SampleB", SampleValueB)

    def test_contains_triggers_build(self):
        """Test that __contains__ triggers build."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class("SampleA", SampleValueA)

        # Access should trigger build
        _ = registry.has_name("SampleA")

        # Should be frozen now
        with pytest.raises(ValueError, match="Value registry is frozen"):
            registry.register_value_class("SampleB", SampleValueB)

    def test_build_is_idempotent(self):
        """Test that calling build() multiple times is safe."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class("SampleA", SampleValueA)

        result1 = registry.build()
        result2 = registry.build()

        assert result1 is result2
        assert result1 is registry

    def test_duplicate_same_class_logs_warning(self, caplog):
        """Test that registering the same class twice logs a warning but succeeds."""
        registry = ValueRegistry.builder(lazy=True)
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
        registry = ValueRegistry.builder(lazy=True)
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
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class("SampleA", SampleValueA)

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
        registry.register_value_class("SampleA", SampleValueA)

        result = registry.build()
        assert result is registry
        assert isinstance(result, ValueRegistry)

    def test_registrations_deleted_after_build(self):
        """Test that _registrations list is deleted after build to save memory."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class("SampleA", SampleValueA)

        assert hasattr(registry, "_registrations")
        registry.build()

        # _registrations should be deleted
        assert not hasattr(registry, "_registrations")

    def test_getitem_syntax(self):
        """Test indexer syntax for retrieving value types."""
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class("SampleA", SampleValueA)

        assert registry.get_value_class("SampleA") is SampleValueA


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
        registry.register_value_class("SampleA", SampleValueA)
        registry.register_value_class("SampleB", SampleValueB)

        # Retrieve and verify
        assert registry.get_value_class("SampleA") is SampleValueA
        assert registry.get_value_class("SampleB") is SampleValueB

    def test_mixed_registration(self):
        """Test registry with multiple value classes."""
        registry = ValueRegistry.builder(lazy=True)

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
        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class("CustomName", SampleValueA)

        assert registry.has_name("CustomName")
        assert registry.get_value_class("CustomName") is SampleValueA

    def test_multiple_registries_are_independent(self):
        """Test that multiple LazyValueRegistry instances are independent."""
        registry1 = ValueRegistry.builder(lazy=True)
        registry2 = ValueRegistry.builder(lazy=True)

        registry1.register_value_class("SampleA", SampleValueA)
        registry2.register_value_class("SampleB", SampleValueB)

        assert registry1.has_name("SampleA")
        assert registry1.get_value_class("SampleA") is SampleValueA
        assert registry2.has_name("SampleB")
        assert registry2.get_value_class("SampleB") is SampleValueB
        assert registry2.has_name("SampleA") is False
        with pytest.raises(ValueError, match='Value type "SampleA" is not registered'):
            registry2.get_value_class("SampleA")


class TestValueRegistryLoadValue:
    """Tests for ValueRegistry.load_value() method."""

    def test_load_value_with_matching_title(self):
        """Test loading a value type from schema with matching title."""
        from workflow_engine.core.values.schema import IntegerValueSchema

        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class("SampleA", SampleValueA)

        schema = IntegerValueSchema(type="integer", title="SampleA")
        loaded_value = registry.load_value(schema)

        assert loaded_value is SampleValueA

    def test_load_value_without_title_returns_none(self):
        """Test that load_value returns None when schema has no title."""
        from workflow_engine.core.values.schema import IntegerValueSchema

        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class("SampleA", SampleValueA)

        schema = IntegerValueSchema(type="integer")  # No title
        loaded_value = registry.load_value(schema)

        assert loaded_value is None

    def test_load_value_with_unregistered_title_returns_none(self):
        """Test that load_value returns None for unregistered title."""
        from workflow_engine.core.values.schema import IntegerValueSchema

        registry = ValueRegistry.builder(lazy=True)
        registry.register_value_class("SampleA", SampleValueA)

        schema = IntegerValueSchema(type="integer", title="UnregisteredType")
        loaded_value = registry.load_value(schema)

        assert loaded_value is None

    def test_load_value_with_different_registries(self):
        """Test that different registries resolve to different value types."""
        from workflow_engine.core.values.schema import StringValueSchema

        registry_a = ValueRegistry.builder(lazy=True)
        registry_a.register_value_class("CustomValue", SampleValueA)

        registry_b = ValueRegistry.builder(lazy=True)
        registry_b.register_value_class("CustomValue", SampleValueB)

        schema = StringValueSchema(type="string", title="CustomValue")

        loaded_a = registry_a.load_value(schema)
        loaded_b = registry_b.load_value(schema)

        assert loaded_a is SampleValueA
        assert loaded_b is SampleValueB

    def test_load_value_integration_with_to_value_cls(self):
        """Test that ValueSchema.to_value_cls() uses the registry's load_value."""
        from workflow_engine.core.values import IntegerValue
        from workflow_engine.core.values.schema import IntegerValueSchema

        # Create a schema with a title that matches a built-in type
        schema = IntegerValueSchema(type="integer", title="IntegerValue")

        # to_value_cls should use the default registry to load the value
        value_cls = schema.to_value_cls()

        assert value_cls is IntegerValue

    def test_load_value_with_custom_registry(self):
        """Test using load_value with custom registry for multi-tenancy."""
        from workflow_engine.core.values.schema import StringValueSchema

        # Create two tenant-specific registries
        tenant_a_registry = ValueRegistry.builder(lazy=True)
        tenant_a_registry.register_value_class("TenantValue", SampleValueA)

        tenant_b_registry = ValueRegistry.builder(lazy=True)
        tenant_b_registry.register_value_class("TenantValue", SampleValueB)

        schema = StringValueSchema(type="string", title="TenantValue")

        # Each tenant should get their own value type
        value_a = tenant_a_registry.load_value(schema)
        value_b = tenant_b_registry.load_value(schema)

        assert value_a is SampleValueA
        assert value_b is SampleValueB
        assert value_a is not value_b
