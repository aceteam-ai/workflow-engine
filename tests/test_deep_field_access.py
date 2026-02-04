"""Tests for deep field access in edges using paths like ("foo", 0, "bar")."""

from typing import ClassVar, Literal, Type

import pytest

from workflow_engine.core import (
    Context,
    Data,
    Edge,
    Empty,
    FieldPath,
    InputEdge,
    IntegerValue,
    Node,
    NodeTypeInfo,
    OutputEdge,
    Params,
    SequenceValue,
    StringMapValue,
    StringValue,
    Workflow,
    resolve_path_type,
    traverse_value,
)
from workflow_engine.execution import TopologicalExecutionAlgorithm


# Test Data types
class SimpleData(Data):
    """Data with simple fields."""

    text: StringValue
    number: IntegerValue


class ContainerData(Data):
    """Data with nested structures."""

    items: SequenceValue[StringValue]
    mapping: StringMapValue[IntegerValue]
    text_field: StringValue


# Test Nodes
class ProducerNodeParams(Params):
    pass


class ProducerNodeOutput(Params):
    items: SequenceValue[StringValue]
    mapping: StringMapValue[IntegerValue]
    text_field: StringValue


class ProducerNode(Node[Empty, ProducerNodeOutput, ProducerNodeParams]):
    """Node that produces nested data."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="ProducerNode",
        display_name="Producer Node",
        description="Produces nested data",
        version="1.0.0",
        parameter_type=ProducerNodeParams,
    )
    type: Literal["ProducerNode"] = "ProducerNode"  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def input_type(self) -> Type[Empty]:
        return Empty

    @property
    def output_type(self) -> Type[ProducerNodeOutput]:
        return ProducerNodeOutput

    async def run(self, context: Context, input: Empty) -> ProducerNodeOutput:
        return ProducerNodeOutput(
            items=SequenceValue[StringValue]([StringValue("first"), StringValue("second")]),
            mapping=StringMapValue[IntegerValue]({"key1": IntegerValue(100), "key2": IntegerValue(200)}),
            text_field=StringValue("nested_text"),
        )


class ConsumerNodeInput(Params):
    text: StringValue


class ConsumerNodeOutput(Params):
    result: StringValue


class ConsumerNode(Node[ConsumerNodeInput, ConsumerNodeOutput, ProducerNodeParams]):
    """Node that consumes a string."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="ConsumerNode",
        display_name="Consumer Node",
        description="Consumes a string",
        version="1.0.0",
        parameter_type=ProducerNodeParams,
    )
    type: Literal["ConsumerNode"] = "ConsumerNode"  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def input_type(self) -> Type[ConsumerNodeInput]:
        return ConsumerNodeInput

    @property
    def output_type(self) -> Type[ConsumerNodeOutput]:
        return ConsumerNodeOutput

    async def run(self, context: Context, input: ConsumerNodeInput) -> ConsumerNodeOutput:
        return ConsumerNodeOutput(result=StringValue(f"processed: {input.text.root}"))


@pytest.mark.unit
class TestPathTraversal:
    """Test the path traversal utilities."""

    def test_traverse_data_field(self):
        """Test traversing through Data fields."""
        data = SimpleData(text=StringValue("test"), number=IntegerValue(42))
        result = traverse_value(data, "text")
        assert result == StringValue("test")

    def test_traverse_sequence_index(self):
        """Test traversing through SequenceValue."""
        seq = SequenceValue([StringValue("a"), StringValue("b")])
        result = traverse_value(seq, (0,))
        assert result == StringValue("a")

    def test_traverse_mapping_key(self):
        """Test traversing through StringMapValue."""
        mapping = StringMapValue({"key": StringValue("value")})
        result = traverse_value(mapping, ("key",))
        assert result == StringValue("value")

    def test_traverse_deep_path(self):
        """Test traversing through nested structures."""
        container = ContainerData(
            items=SequenceValue[StringValue]([StringValue("first"), StringValue("second")]),
            mapping=StringMapValue[IntegerValue]({"key": IntegerValue(42)}),
            text_field=StringValue("hello"),
        )

        # Access container.items[0]
        result = traverse_value(container, ("items", 0))
        assert result == StringValue("first")

        # Access container.mapping["key"]
        result = traverse_value(container, ("mapping", "key"))
        assert result == IntegerValue(42)

        # Access container.text_field
        result = traverse_value(container, "text_field")
        assert result == StringValue("hello")

    def test_traverse_invalid_field(self):
        """Test error handling for invalid fields."""
        data = SimpleData(text=StringValue("test"), number=IntegerValue(42))
        with pytest.raises(AttributeError, match="has no field"):
            traverse_value(data, "nonexistent")

    def test_traverse_invalid_index(self):
        """Test error handling for out of bounds index."""
        seq = SequenceValue[StringValue]([StringValue("a")])
        with pytest.raises(IndexError):
            traverse_value(seq, (5,))

    def test_traverse_type_mismatch(self):
        """Test error handling for type mismatches."""
        data = SimpleData(text=StringValue("test"), number=IntegerValue(42))

        # Try to index a StringValue
        with pytest.raises(TypeError, match="Cannot access integer index"):
            traverse_value(data, ("text", 0))


@pytest.mark.unit
class TestPathTypeResolution:
    """Test the path type resolution utilities."""

    def test_resolve_simple_field(self):
        """Test resolving type for simple field access."""
        result_type = resolve_path_type(SimpleData, "text")
        assert result_type == StringValue

    def test_resolve_sequence_index(self):
        """Test resolving type for sequence index access."""
        result_type = resolve_path_type(SequenceValue[StringValue], (0,))
        assert result_type == StringValue

    def test_resolve_mapping_key(self):
        """Test resolving type for mapping key access."""
        result_type = resolve_path_type(StringMapValue[IntegerValue], ("key",))
        assert result_type == IntegerValue

    def test_resolve_deep_path(self):
        """Test resolving type for deep path."""
        # ContainerData.items[0] -> StringValue
        result_type = resolve_path_type(ContainerData, ("items", 0))
        assert result_type == StringValue

        # ContainerData.mapping["key"] -> IntegerValue
        result_type = resolve_path_type(ContainerData, ("mapping", "key"))
        assert result_type == IntegerValue

        # ContainerData.text_field -> StringValue
        result_type = resolve_path_type(ContainerData, "text_field")
        assert result_type == StringValue

    def test_resolve_invalid_field(self):
        """Test error handling for invalid field."""
        with pytest.raises(ValueError, match="has no field"):
            resolve_path_type(SimpleData, "nonexistent")

    def test_resolve_type_mismatch(self):
        """Test error handling for type mismatches."""
        # Try to index a StringValue
        with pytest.raises(TypeError, match="Cannot access integer index"):
            resolve_path_type(SimpleData, ("text", 0))


@pytest.mark.unit
class TestEdgeValidation:
    """Test that edges validate deep paths correctly."""

    def test_valid_deep_path_edge(self):
        """Test creating an edge with valid deep path."""
        producer = ProducerNode(id="producer", params=ProducerNodeParams())
        consumer = ConsumerNode(id="consumer", params=ProducerNodeParams())

        # Create edge: producer.items[0] -> consumer.text
        edge = Edge.from_nodes(
            source=producer,
            source_key=("items", 0),
            target=consumer,
            target_key="text",
        )

        assert edge.source_key == ("items", 0)
        assert edge.target_key == "text"

    def test_invalid_source_path(self):
        """Test that invalid source paths are rejected."""
        producer = ProducerNode(id="producer", params=ProducerNodeParams())
        consumer = ConsumerNode(id="consumer", params=ProducerNodeParams())

        # Try to create edge with non-existent field
        with pytest.raises(ValueError, match="Invalid source path"):
            Edge.from_nodes(
                source=producer,
                source_key="nonexistent",
                target=consumer,
                target_key="text",
            )

    def test_invalid_path_segment_type(self):
        """Test that invalid path segment types are rejected."""
        producer = ProducerNode(id="producer", params=ProducerNodeParams())
        consumer = ConsumerNode(id="consumer", params=ProducerNodeParams())

        # Try to index a mapping with integer (should use string key)
        with pytest.raises(ValueError, match="Invalid source path"):
            Edge.from_nodes(
                source=producer,
                source_key=("mapping", 0),  # Mapping needs string key, not int
                target=consumer,
                target_key="text",
            )

    def test_type_mismatch_at_path_end(self):
        """Test that type compatibility is checked at path end."""
        producer = ProducerNode(id="producer", params=ProducerNodeParams())
        consumer = ConsumerNode(id="consumer", params=ProducerNodeParams())

        # IntegerValue can cast to StringValue, so this should succeed
        edge = Edge.from_nodes(
            source=producer,
            source_key=("mapping", "key1"),  # IntegerValue
            target=consumer,
            target_key="text",  # StringValue
        )
        assert edge.source_key == ("mapping", "key1")
        assert edge.target_key == "text"

    def test_backwards_compatibility_simple_string(self):
        """Test that simple string keys still work (backwards compatibility)."""
        producer = ProducerNode(id="producer", params=ProducerNodeParams())
        consumer = ConsumerNode(id="consumer", params=ProducerNodeParams())

        # This should work - text_field is StringValue
        edge = Edge.from_nodes(
            source=producer,
            source_key="text_field",  # StringValue
            target=consumer,
            target_key="text",  # StringValue
        )
        assert edge.source_key == "text_field"
        assert edge.target_key == "text"


@pytest.mark.integration
class TestDeepFieldAccessWorkflow:
    """Test deep field access in complete workflow execution."""

    @pytest.mark.asyncio
    async def test_workflow_with_deep_path(self):
        """Test executing a workflow with deep field access."""
        from workflow_engine.contexts import InMemoryContext

        producer = ProducerNode(id="producer", params=ProducerNodeParams())
        consumer = ConsumerNode(id="consumer", params=ProducerNodeParams())

        # Edge: producer.items[0] -> consumer.text
        edge = Edge(
            source_id="producer",
            source_key=("items", 0),
            target_id="consumer",
            target_key="text",
        )

        # Output edge: consumer.result -> workflow output
        output_edge = OutputEdge(
            source_id="consumer",
            source_key="result",
            output_key="final_result",
        )

        workflow = Workflow(
            nodes=[producer, consumer],
            edges=[edge],
            input_edges=[],
            output_edges=[output_edge],
        )

        # Execute
        context = InMemoryContext()
        algorithm = TopologicalExecutionAlgorithm()
        errors, result = await algorithm.execute(context=context, workflow=workflow, input={})

        assert len(errors.workflow_errors) == 0
        assert len(errors.node_errors) == 0
        assert result["final_result"] == StringValue("processed: first")

    @pytest.mark.asyncio
    async def test_workflow_with_mapping_access(self):
        """Test executing a workflow with mapping key access."""
        from workflow_engine.contexts import InMemoryContext

        producer = ProducerNode(id="producer", params=ProducerNodeParams())
        consumer = ConsumerNode(id="consumer", params=ProducerNodeParams())

        # Edge: producer.text_field -> consumer.text
        edge = Edge(
            source_id="producer",
            source_key="text_field",
            target_id="consumer",
            target_key="text",
        )

        # Output edge: consumer.result -> workflow output
        output_edge = OutputEdge(
            source_id="consumer",
            source_key="result",
            output_key="final_result",
        )

        workflow = Workflow(
            nodes=[producer, consumer],
            edges=[edge],
            input_edges=[],
            output_edges=[output_edge],
        )

        # Execute
        context = InMemoryContext()
        algorithm = TopologicalExecutionAlgorithm()
        errors, result = await algorithm.execute(context=context, workflow=workflow, input={})

        assert len(errors.workflow_errors) == 0
        assert len(errors.node_errors) == 0
        assert result["final_result"] == StringValue("processed: nested_text")

    @pytest.mark.asyncio
    async def test_output_edge_with_deep_path(self):
        """Test OutputEdge with deep field access."""
        from workflow_engine.contexts import InMemoryContext

        producer = ProducerNode(id="producer", params=ProducerNodeParams())

        # Output edge with deep path: producer.items[1] -> output
        output_edge = OutputEdge(
            source_id="producer",
            source_key=("items", 1),
            output_key="extracted_value",
        )

        workflow = Workflow(
            nodes=[producer],
            edges=[],
            input_edges=[],
            output_edges=[output_edge],
        )

        # Execute
        context = InMemoryContext()
        algorithm = TopologicalExecutionAlgorithm()
        errors, result = await algorithm.execute(context=context, workflow=workflow, input={})

        assert len(errors.workflow_errors) == 0
        assert len(errors.node_errors) == 0
        assert result["extracted_value"] == StringValue("second")
