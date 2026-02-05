"""Tests for WorkflowEngine."""

from functools import cached_property
from typing import Literal

import pytest

from workflow_engine import (
    Edge,
    Empty,
    IntegerValue,
    Node,
    NodeRegistry,
    NodeTypeInfo,
    ValueRegistry,
    Workflow,
    WorkflowEngine,
)
from workflow_engine.contexts import InMemoryContext
from workflow_engine.core.io import InputNode, OutputNode
from workflow_engine.execution import ParallelExecutionAlgorithm
from workflow_engine.nodes import AddNode


# Test fixtures - simple node classes for testing
class SampleAddNode(Node[Empty, Empty, Empty]):
    TYPE_INFO = NodeTypeInfo.from_parameter_type(
        name="SampleAdd",
        display_name="Sample Add",
        version="1.0.0",
        parameter_type=Empty,
    )
    type: Literal["SampleAdd"] = "SampleAdd"  # pyright: ignore[reportIncompatibleVariableOverride]

    @cached_property
    def input_type(self):
        return Empty

    @cached_property
    def output_type(self):
        return Empty

    async def run(self, context, input):
        return Empty()


class SampleMultiplyNode(Node[Empty, Empty, Empty]):
    TYPE_INFO = NodeTypeInfo.from_parameter_type(
        name="SampleMultiply",
        display_name="Sample Multiply",
        version="1.0.0",
        parameter_type=Empty,
    )
    type: Literal["SampleMultiply"] = "SampleMultiply"  # pyright: ignore[reportIncompatibleVariableOverride]

    @cached_property
    def input_type(self):
        return Empty

    @cached_property
    def output_type(self):
        return Empty

    async def run(self, context, input):
        return Empty()


class TestWorkflowEngine:
    """Tests for WorkflowEngine class."""

    def test_construction(self):
        """Test that WorkflowEngine can be constructed."""
        node_registry = NodeRegistry.builder(lazy=True)
        value_registry = ValueRegistry.builder(lazy=True)

        engine = WorkflowEngine(
            node_registry=node_registry,
            value_registry=value_registry,
        )

        assert engine.node_registry is node_registry
        assert engine.value_registry is value_registry

    def test_construction_with_defaults(self):
        """Test that WorkflowEngine uses default registries when none provided."""
        engine = WorkflowEngine()

        assert engine.node_registry is NodeRegistry.DEFAULT
        assert engine.value_registry is ValueRegistry.DEFAULT

    def test_default_engine_loads_workflow(self):
        """Test that default engine can load workflows with built-in node types."""
        # Create engine with no arguments - uses global registries
        engine = WorkflowEngine()

        # Use SampleAddNode which is registered in our test's global registry
        input_node = InputNode.empty()
        output_node = OutputNode.empty()

        workflow = Workflow.model_construct(
            input_node=input_node,
            inner_nodes=[
                Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")
            ],
            output_node=output_node,
            edges=[],
        )

        # Should be able to load it
        typed_workflow = engine.load(workflow)
        assert isinstance(typed_workflow.inner_nodes[0], SampleAddNode)

    def test_load_workflow_with_untyped_nodes(self):
        """Test loading a workflow with untyped nodes into typed nodes."""
        # Set up registry
        node_registry = NodeRegistry.builder(lazy=True)
        node_registry.register_node_class("SampleAdd", SampleAddNode)
        node_registry.register_base_node_class(Node)

        value_registry = ValueRegistry.builder(lazy=True)

        engine = WorkflowEngine(
            node_registry=node_registry,
            value_registry=value_registry,
        )

        # Create workflow with untyped node
        input_node = InputNode.empty()
        output_node = OutputNode.empty()
        untyped_node = Node.model_construct(
            type="SampleAdd", id="node1", version="1.0.0"
        )
        workflow = Workflow(
            input_node=input_node,
            inner_nodes=[untyped_node],
            output_node=output_node,
            edges=[],
        )

        # Load workflow
        typed_workflow = engine.load(workflow)

        # Verify the node is now typed
        assert len(typed_workflow.inner_nodes) == 1
        assert isinstance(typed_workflow.inner_nodes[0], SampleAddNode)
        assert typed_workflow.inner_nodes[0].type == "SampleAdd"
        assert typed_workflow.inner_nodes[0].id == "node1"

    def test_load_workflow_with_already_typed_nodes(self):
        """Test that loading a workflow with typed nodes returns equivalent workflow."""
        node_registry = NodeRegistry.builder(lazy=True)
        node_registry.register_node_class("SampleAdd", SampleAddNode)

        value_registry = ValueRegistry.builder(lazy=True)

        engine = WorkflowEngine(
            node_registry=node_registry,
            value_registry=value_registry,
        )

        # Create workflow with already typed node
        input_node = InputNode.empty()
        output_node = OutputNode.empty()
        typed_node = SampleAddNode(
            id="node1",
            version="1.0.0",
        )
        workflow = Workflow(
            input_node=input_node,
            inner_nodes=[typed_node],
            output_node=output_node,
            edges=[],
        )

        # Load workflow
        loaded_workflow = engine.load(workflow)

        # Verify the node is still the same type
        assert len(loaded_workflow.inner_nodes) == 1
        assert isinstance(loaded_workflow.inner_nodes[0], SampleAddNode)
        assert loaded_workflow.inner_nodes[0].id == "node1"

    def test_load_workflow_with_multiple_nodes(self):
        """Test loading a workflow with multiple untyped nodes."""
        node_registry = NodeRegistry.builder(lazy=True)
        node_registry.register_node_class("SampleAdd", SampleAddNode)
        node_registry.register_node_class("SampleMultiply", SampleMultiplyNode)
        node_registry.register_base_node_class(Node)

        value_registry = ValueRegistry.builder(lazy=True)

        engine = WorkflowEngine(
            node_registry=node_registry,
            value_registry=value_registry,
        )

        # Create workflow with multiple untyped nodes
        input_node = InputNode.empty()
        output_node = OutputNode.empty()
        node1 = Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")
        node2 = Node.model_construct(
            type="SampleMultiply", id="node2", version="1.0.0"
        )

        workflow = Workflow(
            input_node=input_node,
            inner_nodes=[node1, node2],
            output_node=output_node,
            edges=[],
        )

        # Load workflow
        typed_workflow = engine.load(workflow)

        # Verify both nodes are typed
        assert len(typed_workflow.inner_nodes) == 2
        assert isinstance(typed_workflow.inner_nodes[0], SampleAddNode)
        assert isinstance(typed_workflow.inner_nodes[1], SampleMultiplyNode)

    def test_load_workflow_with_unregistered_node_type_raises_error(self):
        """Test that loading a workflow with unregistered node type raises ValueError."""
        node_registry = NodeRegistry.builder(lazy=True)
        node_registry.register_base_node_class(Node)

        value_registry = ValueRegistry.builder(lazy=True)

        engine = WorkflowEngine(
            node_registry=node_registry,
            value_registry=value_registry,
        )

        # Create workflow with unregistered node type using model_construct to bypass validation
        input_node = InputNode.empty()
        output_node = OutputNode.empty()
        untyped_node = Node.model_construct(
            type="UnregisteredType",
            id="node1",
            version="1.0.0",
        )
        workflow = Workflow.model_construct(
            input_node=input_node,
            inner_nodes=[untyped_node],
            output_node=output_node,
            edges=[],
        )

        # Loading should raise error
        with pytest.raises(
            ValueError, match='Node type "UnregisteredType" is not registered'
        ):
            engine.load(workflow)

    def test_load_workflow_with_different_registries(self):
        """Test that different engines with different registries have different capabilities."""
        # Engine A with both SampleAddNode and SampleMultiplyNode
        node_registry_a = NodeRegistry.builder(lazy=True)
        node_registry_a.register_node_class("SampleAdd", SampleAddNode)
        node_registry_a.register_node_class("SampleMultiply", SampleMultiplyNode)
        node_registry_a.register_base_node_class(Node)

        engine_a = WorkflowEngine(
            node_registry=node_registry_a,
            value_registry=ValueRegistry.builder(lazy=True),
        )

        # Engine B with only SampleAddNode
        node_registry_b = NodeRegistry.builder(lazy=True)
        node_registry_b.register_node_class("SampleAdd", SampleAddNode)
        node_registry_b.register_base_node_class(Node)

        engine_b = WorkflowEngine(
            node_registry=node_registry_b,
            value_registry=ValueRegistry.builder(lazy=True),
        )

        # Workflow with SampleAdd node - both can load
        input_node = InputNode.empty()
        output_node = OutputNode.empty()

        workflow_add = Workflow.model_construct(
            input_node=input_node,
            inner_nodes=[
                Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")
            ],
            output_node=output_node,
            edges=[],
        )

        workflow_a = engine_a.load(workflow_add)
        workflow_b = engine_b.load(workflow_add)

        assert isinstance(workflow_a.inner_nodes[0], SampleAddNode)
        assert isinstance(workflow_b.inner_nodes[0], SampleAddNode)

        # Workflow with SampleMultiply node - only engine A can load
        workflow_multiply = Workflow.model_construct(
            input_node=input_node,
            inner_nodes=[
                Node.model_construct(
                    type="SampleMultiply", id="node1", version="1.0.0"
                )
            ],
            output_node=output_node,
            edges=[],
        )

        workflow_a_mult = engine_a.load(workflow_multiply)
        assert isinstance(workflow_a_mult.inner_nodes[0], SampleMultiplyNode)

        # Engine B cannot load SampleMultiply
        with pytest.raises(
            ValueError, match='Node type "SampleMultiply" is not registered'
        ):
            engine_b.load(workflow_multiply)

    def test_load_preserves_workflow_structure(self):
        """Test that loading preserves edges and other workflow structure."""
        node_registry = NodeRegistry.builder(lazy=True)
        node_registry.register_node_class("SampleAdd", SampleAddNode)
        node_registry.register_base_node_class(Node)

        value_registry = ValueRegistry.builder(lazy=True)

        engine = WorkflowEngine(
            node_registry=node_registry,
            value_registry=value_registry,
        )

        # Create workflow with multiple nodes (no edges for simplicity)
        input_node = InputNode.empty()
        output_node = OutputNode.empty()
        node1 = Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")
        node2 = Node.model_construct(type="SampleAdd", id="node2", version="1.0.0")

        workflow = Workflow.model_construct(
            input_node=input_node,
            inner_nodes=[node1, node2],
            output_node=output_node,
            edges=[],
        )

        # Load workflow
        typed_workflow = engine.load(workflow)

        # Verify structure is preserved
        assert len(typed_workflow.inner_nodes) == 2
        assert typed_workflow.input_node.id == "input"
        assert typed_workflow.output_node.id == "output"
        assert typed_workflow.inner_nodes[0].id == "node1"
        assert typed_workflow.inner_nodes[1].id == "node2"


class TestWorkflowEngineMultiTenancy:
    """Tests for multi-tenancy scenarios with WorkflowEngine."""

    def test_tenant_isolation(self):
        """Test that different tenants can have different node types available."""
        # Tenant A has only SampleAddNode
        tenant_a_registry = NodeRegistry.builder(lazy=True)
        tenant_a_registry.register_node_class("SampleAdd", SampleAddNode)
        tenant_a_registry.register_base_node_class(Node)

        engine_a = WorkflowEngine(
            node_registry=tenant_a_registry,
            value_registry=ValueRegistry.builder(lazy=True),
        )

        # Tenant B has only SampleMultiplyNode
        tenant_b_registry = NodeRegistry.builder(lazy=True)
        tenant_b_registry.register_node_class("SampleMultiply", SampleMultiplyNode)
        tenant_b_registry.register_base_node_class(Node)

        engine_b = WorkflowEngine(
            node_registry=tenant_b_registry,
            value_registry=ValueRegistry.builder(lazy=True),
        )

        # Workflow with SampleAdd node - use model_construct to bypass validation
        input_node = InputNode.empty()
        output_node = OutputNode.empty()

        workflow_add = Workflow.model_construct(
            input_node=input_node,
            inner_nodes=[
                Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")
            ],
            output_node=output_node,
            edges=[],
        )

        # Tenant A can load it
        loaded_a = engine_a.load(workflow_add)
        assert isinstance(loaded_a.inner_nodes[0], SampleAddNode)

        # Tenant B cannot load it
        with pytest.raises(ValueError, match='Node type "SampleAdd" is not registered'):
            engine_b.load(workflow_add)

    def test_shared_workflow_different_engines(self):
        """Test that the same workflow JSON can be loaded by different engines."""
        # Create two engines with different registries
        registry_1 = NodeRegistry.builder(lazy=True)
        registry_1.register_node_class("SampleAdd", SampleAddNode)
        registry_1.register_base_node_class(Node)

        registry_2 = NodeRegistry.builder(lazy=True)
        registry_2.register_node_class("SampleAdd", SampleAddNode)
        registry_2.register_base_node_class(Node)

        engine_1 = WorkflowEngine(
            node_registry=registry_1,
            value_registry=ValueRegistry.builder(lazy=True),
        )

        engine_2 = WorkflowEngine(
            node_registry=registry_2,
            value_registry=ValueRegistry.builder(lazy=True),
        )

        # Create a workflow
        input_node = InputNode.empty()
        output_node = OutputNode.empty()

        workflow = Workflow(
            input_node=input_node,
            inner_nodes=[
                Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")
            ],
            output_node=output_node,
            edges=[],
        )

        # Both engines can load the same workflow independently
        loaded_1 = engine_1.load(workflow)
        loaded_2 = engine_2.load(workflow)

        # Both should produce typed workflows
        assert isinstance(loaded_1.inner_nodes[0], SampleAddNode)
        assert isinstance(loaded_2.inner_nodes[0], SampleAddNode)


class TestWorkflowEngineExecution:
    """Tests for WorkflowEngine.execute() method."""

    @pytest.mark.asyncio
    async def test_execute_with_default_algorithm(self):
        """Test that execute() works with default TopologicalExecutionAlgorithm."""

        # Create engine with defaults
        engine = WorkflowEngine()

        # Create a simple addition workflow
        input_node = InputNode.from_fields(
            a=IntegerValue,
            b=IntegerValue,
        )
        output_node = OutputNode.from_fields(
            result=IntegerValue,
        )
        add_node = AddNode(id="add1")

        workflow = Workflow(
            input_node=input_node,
            inner_nodes=[add_node],
            output_node=output_node,
            edges=[
                Edge.from_nodes(
                    source=input_node, source_key="a", target=add_node, target_key="a"
                ),
                Edge.from_nodes(
                    source=input_node, source_key="b", target=add_node, target_key="b"
                ),
                Edge.from_nodes(
                    source=add_node,
                    source_key="sum",
                    target=output_node,
                    target_key="result",
                ),
            ],
        )

        # Execute workflow
        context = InMemoryContext()
        input_data = {"a": IntegerValue(5), "b": IntegerValue(3)}

        errors, output = await engine.execute(workflow, input_data, context)

        # Verify results
        assert len(errors.workflow_errors) == 0
        assert len(errors.node_errors) == 0
        assert output["result"].root == 8

    @pytest.mark.asyncio
    async def test_execute_with_custom_algorithm(self):
        """Test that execute() uses the provided execution algorithm."""

        # Create engine with custom algorithm
        custom_algorithm = ParallelExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=custom_algorithm)

        # Verify the algorithm is stored
        assert engine.execution_algorithm is custom_algorithm

        # Create a simple workflow
        input_node = InputNode.from_fields(
            a=IntegerValue,
            b=IntegerValue,
        )
        output_node = OutputNode.from_fields(
            result=IntegerValue,
        )
        add_node = AddNode(id="add1")

        workflow = Workflow(
            input_node=input_node,
            inner_nodes=[add_node],
            output_node=output_node,
            edges=[
                Edge.from_nodes(
                    source=input_node, source_key="a", target=add_node, target_key="a"
                ),
                Edge.from_nodes(
                    source=input_node, source_key="b", target=add_node, target_key="b"
                ),
                Edge.from_nodes(
                    source=add_node,
                    source_key="sum",
                    target=output_node,
                    target_key="result",
                ),
            ],
        )

        # Execute workflow
        context = InMemoryContext()
        input_data = {"a": IntegerValue(10), "b": IntegerValue(20)}

        errors, output = await engine.execute(workflow, input_data, context)

        # Verify results
        assert len(errors.workflow_errors) == 0
        assert len(errors.node_errors) == 0
        assert output["result"].root == 30

    @pytest.mark.asyncio
    async def test_execute_calls_load_internally(self):
        """Test that execute() calls load() internally before execution."""

        # Create engine
        engine = WorkflowEngine()

        # Create a simple workflow
        input_node = InputNode.from_fields(
            a=IntegerValue,
            b=IntegerValue,
        )
        output_node = OutputNode.from_fields(
            result=IntegerValue,
        )
        add_node = AddNode(id="add1")

        workflow = Workflow(
            input_node=input_node,
            inner_nodes=[add_node],
            output_node=output_node,
            edges=[
                Edge.from_nodes(
                    source=input_node, source_key="a", target=add_node, target_key="a"
                ),
                Edge.from_nodes(
                    source=input_node, source_key="b", target=add_node, target_key="b"
                ),
                Edge.from_nodes(
                    source=add_node,
                    source_key="sum",
                    target=output_node,
                    target_key="result",
                ),
            ],
        )

        # Execute workflow - load is called internally
        context = InMemoryContext()
        input_data = {"a": IntegerValue(7), "b": IntegerValue(4)}

        errors, output = await engine.execute(workflow, input_data, context)

        # Verify it worked
        assert len(errors.workflow_errors) == 0
        assert len(errors.node_errors) == 0
        assert output["result"].root == 11

    @pytest.mark.asyncio
    async def test_execute_with_typed_workflow(self):
        """Test that execute() works with already typed workflows."""

        # Create engine
        engine = WorkflowEngine()

        # Create workflow with typed node
        input_node = InputNode.from_fields(
            a=IntegerValue,
            b=IntegerValue,
        )
        output_node = OutputNode.from_fields(
            result=IntegerValue,
        )
        typed_node = AddNode(id="add1")

        workflow = Workflow(
            input_node=input_node,
            inner_nodes=[typed_node],
            output_node=output_node,
            edges=[
                Edge.from_nodes(
                    source=input_node, source_key="a", target=typed_node, target_key="a"
                ),
                Edge.from_nodes(
                    source=input_node, source_key="b", target=typed_node, target_key="b"
                ),
                Edge.from_nodes(
                    source=typed_node,
                    source_key="sum",
                    target=output_node,
                    target_key="result",
                ),
            ],
        )

        # Execute workflow
        context = InMemoryContext()
        input_data = {"a": IntegerValue(15), "b": IntegerValue(25)}

        errors, output = await engine.execute(workflow, input_data, context)

        # Verify results
        assert len(errors.workflow_errors) == 0
        assert len(errors.node_errors) == 0
        assert output["result"].root == 40

    @pytest.mark.asyncio
    async def test_execute_with_multiple_nodes(self):
        """Test execute() with a workflow containing multiple nodes."""

        # Create engine
        engine = WorkflowEngine()

        # Create workflow: (a + b) + c
        input_node = InputNode.from_fields(
            a=IntegerValue,
            b=IntegerValue,
            c=IntegerValue,
        )
        output_node = OutputNode.from_fields(
            result=IntegerValue,
        )
        add_node1 = AddNode(id="add1")
        add_node2 = AddNode(id="add2")

        workflow = Workflow(
            input_node=input_node,
            inner_nodes=[add_node1, add_node2],
            output_node=output_node,
            edges=[
                Edge.from_nodes(
                    source=input_node, source_key="a", target=add_node1, target_key="a"
                ),
                Edge.from_nodes(
                    source=input_node, source_key="b", target=add_node1, target_key="b"
                ),
                Edge.from_nodes(
                    source=add_node1, source_key="sum", target=add_node2, target_key="a"
                ),
                Edge.from_nodes(
                    source=input_node, source_key="c", target=add_node2, target_key="b"
                ),
                Edge.from_nodes(
                    source=add_node2,
                    source_key="sum",
                    target=output_node,
                    target_key="result",
                ),
            ],
        )

        # Execute: (2 + 3) + 4 = 9
        context = InMemoryContext()
        input_data = {
            "a": IntegerValue(2),
            "b": IntegerValue(3),
            "c": IntegerValue(4),
        }

        errors, output = await engine.execute(workflow, input_data, context)

        # Verify results
        assert len(errors.workflow_errors) == 0
        assert len(errors.node_errors) == 0
        assert output["result"].root == 9

    @pytest.mark.asyncio
    async def test_execute_with_tenant_specific_engine(self):
        """Test that tenant-specific engines execute with isolated registries."""

        # Create tenant-specific registry with only SampleAddNode
        tenant_registry = NodeRegistry.builder(lazy=True)
        tenant_registry.register_node_class("SampleAdd", SampleAddNode)
        tenant_registry.register_base_node_class(Node)

        engine = WorkflowEngine(
            node_registry=tenant_registry,
            value_registry=ValueRegistry.builder(lazy=True),
        )

        # Create workflow with SampleAdd node
        input_node = InputNode.empty()
        output_node = OutputNode.empty()

        workflow = Workflow.model_construct(
            input_node=input_node,
            inner_nodes=[
                Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")
            ],
            output_node=output_node,
            edges=[],
        )

        # Execute workflow
        context = InMemoryContext()
        errors, output = await engine.execute(workflow, {}, context)

        # Should execute successfully (no errors)
        assert len(errors.workflow_errors) == 0
        assert len(errors.node_errors) == 0
