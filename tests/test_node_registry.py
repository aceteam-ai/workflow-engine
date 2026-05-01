"""Tests for NodeRegistry implementations."""

from typing import Literal, Type

import pytest
from overrides import override

from workflow_engine import (
    Empty,
    ExecutionContext,
    Node,
    NodeRegistry,
    NodeTypeInfo,
)
from workflow_engine.core.node import ImmutableNodeRegistry, NodeRegistryBuilder


# Test fixtures - simple node classes for testing
class SampleNodeA(Node[Empty, Empty, Empty]):
    TYPE_INFO = NodeTypeInfo.from_parameter_type(
        name="TestA",
        display_name="Test A",
        version="1.0.0",
        parameter_type=Empty,
    )
    type: Literal["TestA"] = "TestA"  # pyright: ignore[reportIncompatibleVariableOverride]

    @classmethod
    @override
    def static_input_type(cls) -> Type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> Type[Empty]:
        return Empty

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[Empty],
        input: Empty,
    ):
        return Empty()


class SampleNodeB(Node[Empty, Empty, Empty]):
    TYPE_INFO = NodeTypeInfo.from_parameter_type(
        name="TestB",
        display_name="Test B",
        version="1.0.0",
        parameter_type=Empty,
    )
    type: Literal["TestB"] = "TestB"  # pyright: ignore[reportIncompatibleVariableOverride]

    @classmethod
    @override
    def static_input_type(cls) -> Type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> Type[Empty]:
        return Empty

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[Empty],
        input: Empty,
    ):
        return Empty()


class SampleBaseNode(Node[Empty, Empty, Empty]):
    """A base node class without a type discriminator."""

    pass


class TestImmutableNodeRegistry:
    """Tests for ImmutableNodeRegistry."""

    def test_construction(self):
        """Test that ImmutableNodeRegistry can be constructed."""
        registry = ImmutableNodeRegistry(
            node_classes={"TestA": SampleNodeA, "TestB": SampleNodeB},
        )
        assert registry is not None

    def test_get_node_class(self):
        """Test retrieving node classes by name."""
        registry = ImmutableNodeRegistry(
            node_classes={"TestA": SampleNodeA, "TestB": SampleNodeB},
        )

        assert registry.get("TestA") is SampleNodeA
        assert registry.get("TestB") is SampleNodeB

    def test_get_node_class_not_found(self):
        """Test that missing names yield None."""
        registry = ImmutableNodeRegistry(
            node_classes={"TestA": SampleNodeA},
        )

        assert registry.get("NonExistent") is None

    def test_immutability(self):
        """Test that the registry data cannot be modified after construction."""
        node_classes: dict[str, type[Node]] = {"TestA": SampleNodeA}

        registry = ImmutableNodeRegistry(
            node_classes=node_classes,
        )

        # Modifying the original dicts should not affect the registry
        node_classes["TestB"] = SampleNodeB

        assert registry.get("TestB") is None


class TestEagerNodeRegistryBuilder:
    """Tests for EagerNodeRegistryBuilder."""

    def test_register(self):
        """Test registering node classes."""
        registry = (
            NodeRegistry.builder(lazy=False)
            .register(SampleNodeA)
            .register(SampleNodeB)
            .build()
        )
        assert registry.get("TestA") is SampleNodeA
        assert registry.get("TestB") is SampleNodeB

    def test_fluent_interface(self):
        """Test that methods return self for chaining."""
        builder1 = NodeRegistry.builder(lazy=False)
        builder2 = builder1.register(SampleNodeA).register(SampleNodeB)
        assert builder1 is builder2

    def test_duplicate_node_class_raises_error(self):
        """Test that registering the same node type twice raises ValueError."""
        builder = NodeRegistry.builder(lazy=False)
        builder.register(SampleNodeA)

        with pytest.raises(ValueError, match='Node type "TestA" is already registered'):
            builder.register(SampleNodeA)

    def test_build_returns_immutable_registry(self):
        """Test that build() returns an ImmutableNodeRegistry."""
        builder = NodeRegistry.builder(lazy=False)
        builder.register(SampleNodeA)

        registry = builder.build()
        assert isinstance(registry, ImmutableNodeRegistry)
        assert not isinstance(registry, NodeRegistryBuilder)


class TestLazyNodeRegistry:
    """Tests for LazyNodeRegistry."""

    def test_starts_unfrozen(self):
        """Test that LazyNodeRegistry starts in unfrozen state."""
        registry = NodeRegistry.builder(lazy=True)
        # Should be able to register without error
        registry.register(SampleNodeA)
        assert True  # No exception means it's unfrozen

    def test_register_node_class_when_unfrozen(self):
        """Test registering node classes when unfrozen."""
        (
            NodeRegistry.builder(lazy=True)
            .register(Node)
            .register(SampleNodeA)
            .register(SampleNodeB)
            .build()
        )

    def test_fluent_interface(self):
        """Test that methods return self for chaining."""
        registry = NodeRegistry.builder(lazy=True)
        result = registry.register(SampleNodeB).register(Node).register(SampleNodeA)

        assert result is registry

    def test_build_freezes_registry(self):
        """Test that build() freezes the registry."""
        registry = NodeRegistry.builder(lazy=True)
        registry.register(SampleNodeA)

        # Build should freeze
        registry.build()

        # Should not be able to register after freezing
        with pytest.raises(ValueError, match="Node registry is frozen"):
            registry.register(SampleNodeB)

    def test_access_triggers_build(self):
        """Test that accessing the registry triggers build."""
        registry = NodeRegistry.builder(lazy=True)
        registry.register(SampleNodeA)

        # Access should trigger build
        registry.get("TestA")

        # Should be frozen now
        with pytest.raises(ValueError, match="Node registry is frozen"):
            registry.register(SampleNodeB)

    def test_build_is_idempotent(self):
        """Test that calling build() multiple times is safe."""
        registry = NodeRegistry.builder(lazy=True)
        registry.register(SampleNodeA)

        result1 = registry.build()
        result2 = registry.build()

        assert result1 is result2
        assert result1 is registry

    def test_duplicate_same_class_logs_warning(self, caplog):
        """Test that registering the same class twice logs a warning but succeeds."""
        registry = NodeRegistry.builder(lazy=True)
        registry.register(SampleNodeA)
        registry.register(SampleNodeA)  # Same class

        # Build should log warning but not raise
        registry.build()

        # Check that warning was logged
        assert any(
            "Node type TestA is already registered" in record.message
            for record in caplog.records
        )

    def test_duplicate_different_class_raises_error(self):
        """Test that registering a different class with the same type raises ValueError."""
        # Create a duplicate class inside the test, patching DEFAULT to avoid
        # polluting it (since __init_subclass__ registers into DEFAULT).
        original_default = NodeRegistry.DEFAULT
        NodeRegistry.DEFAULT = NodeRegistry.builder(lazy=True)
        try:

            class DuplicateTestA(Node[Empty, Empty, Empty]):
                TYPE_INFO = NodeTypeInfo.from_parameter_type(
                    name="TestA",
                    display_name="Test A Dup",
                    version="1.0.0",
                    parameter_type=Empty,
                )
                type: Literal["TestA"] = "TestA"  # pyright: ignore[reportIncompatibleVariableOverride]

                @classmethod
                @override
                def static_input_type(cls) -> Type[Empty]:
                    return Empty

                @classmethod
                @override
                def static_output_type(cls) -> Type[Empty]:
                    return Empty

                @override
                async def run(
                    self,
                    *,
                    context: ExecutionContext,
                    input_type: Type[Empty],
                    output_type: Type[Empty],
                    input: Empty,
                ):
                    return Empty()

        finally:
            NodeRegistry.DEFAULT = original_default

        registry = NodeRegistry.builder(lazy=True)
        registry.register(SampleNodeA)
        registry.register(DuplicateTestA)

        # Build should raise error
        with pytest.raises(
            ValueError,
            match=r'Node type "TestA" .* is already registered to a different class',
        ):
            registry.build()

    def test_duplicate_base_class_logs_warning(self, caplog):
        """Test that registering the same base class twice logs a warning."""
        registry = NodeRegistry.builder(lazy=True)
        registry.register(Node)
        registry.register(Node)

        registry.build()

        assert any(
            "Node type Node is already registered to class Node" in record.message
            for record in caplog.records
        )

    def test_get_not_found_returns_none(self):
        """Missing registry keys yield None."""
        registry = NodeRegistry.builder(lazy=True)
        registry.register(SampleNodeA)

        registry.build()
        assert registry.get("NonExistent") is None

    def test_lazy_registry_is_also_registry(self):
        """Test that LazyNodeRegistry is both a builder and a registry."""
        registry = NodeRegistry.builder(lazy=True)

        assert isinstance(registry, NodeRegistry)
        assert isinstance(registry, NodeRegistryBuilder)

    def test_build_returns_self(self):
        """Test that build() returns self (the LazyNodeRegistry itself)."""
        registry = NodeRegistry.builder(lazy=True)
        registry.register(SampleNodeA)

        result = registry.build()
        assert result is registry
        assert isinstance(result, NodeRegistry)

    def test_registrations_deleted_after_build(self):
        """Test that _registrations list is deleted after build to save memory."""
        registry = NodeRegistry.builder(lazy=True)
        registry.register(SampleNodeA)

        assert hasattr(registry, "_registrations")
        registry.build()

        # _registrations should be deleted
        assert not hasattr(registry, "_registrations")


class TestNodeRegistryIntegration:
    """Integration tests with actual Node classes."""

    def test_with_concrete_node_classes(self):
        """Test registry with real node classes that have type discriminators."""
        registry = NodeRegistry.builder(lazy=True)

        # These classes should have been registered via __init_subclass__
        # but we're creating a fresh registry for isolation
        registry.register(SampleNodeA)
        registry.register(SampleNodeB)

        # Retrieve and verify
        assert registry.get("TestA") is SampleNodeA
        assert registry.get("TestB") is SampleNodeB


class TestNodeRegistryLoad:
    """Tests for NodeRegistry.load()."""

    def test_load_node_with_base_class(self):
        """Test loading a base node instance into concrete type."""
        registry = (
            NodeRegistry.builder(lazy=True).register(Node).register(SampleNodeA).build()
        )

        # Create a base Node instance using model_construct to bypass validators
        base_node = Node.model_construct(type="TestA", id="test-1", version="1.0.0")

        # Load it into the concrete type
        loaded_node = registry.load(base_node)

        assert isinstance(loaded_node, SampleNodeA)
        assert loaded_node.type == "TestA"
        assert loaded_node.id == "test-1"
        assert loaded_node.version == "1.0.0"

    def test_load_node_with_concrete_class_returns_unchanged(self):
        """Test that loading a concrete node returns it unchanged."""
        registry = NodeRegistry.builder(lazy=True).register(SampleNodeA).build()

        # Create a concrete node instance
        concrete_node = SampleNodeA(type="TestA", id="test-1", version="1.0.0")

        # Load should return the same instance
        loaded_node = registry.load(concrete_node)

        assert loaded_node is concrete_node
        assert isinstance(loaded_node, SampleNodeA)

    def test_load_node_unregistered_type_raises_error(self):
        """Test that loading a node with unregistered type raises ValueError."""
        registry = NodeRegistry.builder(lazy=True).register(Node).build()

        # Create a base Node with unregistered type using model_construct to bypass validation
        base_node = Node.model_construct(
            type="UnregisteredType", id="test-1", version="1.0.0"
        )

        with pytest.raises(
            ValueError, match='Node type "UnregisteredType" is not registered'
        ):
            registry.load(base_node)

    def test_load_node_with_different_registries(self):
        """Test that different registries load to different concrete types."""
        registry_a = (
            NodeRegistry.builder(lazy=True).register(Node).register(SampleNodeA).build()
        )

        registry_b = (
            NodeRegistry.builder(lazy=True).register(Node).register(SampleNodeB).build()
        )

        base_node_a = Node.model_construct(type="TestA", id="test-1", version="1.0.0")
        base_node_b = Node.model_construct(type="TestB", id="test-2", version="1.0.0")

        loaded_a = registry_a.load(base_node_a)
        loaded_b = registry_b.load(base_node_b)

        assert isinstance(loaded_a, SampleNodeA)
        assert isinstance(loaded_b, SampleNodeB)

        # Cross-registry loading should fail
        with pytest.raises(ValueError, match='Node type "TestB" is not registered'):
            registry_a.load(base_node_b)

    def test_load_node_preserves_params(self):
        """Test that loading preserves node parameters."""
        registry = (
            NodeRegistry.builder(lazy=True).register(SampleNodeA).register(Node).build()
        )

        base_node = Node.model_construct(
            type="TestA",
            id="test-1",
            version="1.0.0",
            params=Empty(),
        )

        loaded_node = registry.load(base_node)

        assert isinstance(loaded_node, SampleNodeA)
        assert loaded_node.params is not None
        assert isinstance(loaded_node.params, Empty)
