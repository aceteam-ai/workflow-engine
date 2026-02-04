"""Tests for WorkflowEngine."""

import pytest
from typing import Literal

from workflow_engine import (
    ValueRegistry,
    WorkflowEngine,
    Workflow,
    Node,
    Empty,
    NodeTypeInfo,
    NodeRegistry,
)


# Test fixtures - simple node classes for testing
class SampleAddNode(Node[Empty, Empty, Empty]):
    TYPE_INFO = NodeTypeInfo.from_parameter_type(
        name="SampleAdd",
        display_name="Sample Add",
        version="1.0.0",
        parameter_type=Empty,
    )
    type: Literal["SampleAdd"] = "SampleAdd"

    @property
    def input_type(self):
        return Empty

    @property
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
    type: Literal["SampleMultiply"] = "SampleMultiply"

    @property
    def input_type(self):
        return Empty

    @property
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
        # (it auto-registered when the class was defined at module level)
        workflow = Workflow.model_construct(
            nodes=[Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")],
            edges=[],
            input_edges=[],
            output_edges=[],
        )

        # Should be able to load it
        typed_workflow = engine.load(workflow)
        assert isinstance(typed_workflow.nodes[0], SampleAddNode)

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
        untyped_node = Node.model_construct(
            type="SampleAdd", id="node1", version="1.0.0"
        )
        workflow = Workflow(
            nodes=[untyped_node],
            edges=[],
            input_edges=[],
            output_edges=[],
        )

        # Load workflow
        typed_workflow = engine.load(workflow)

        # Verify the node is now typed
        assert len(typed_workflow.nodes) == 1
        assert isinstance(typed_workflow.nodes[0], SampleAddNode)
        assert typed_workflow.nodes[0].type == "SampleAdd"
        assert typed_workflow.nodes[0].id == "node1"

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
        typed_node = SampleAddNode(type="SampleAdd", id="node1", version="1.0.0")
        workflow = Workflow(
            nodes=[typed_node],
            edges=[],
            input_edges=[],
            output_edges=[],
        )

        # Load workflow
        loaded_workflow = engine.load(workflow)

        # Verify the node is still the same type
        assert len(loaded_workflow.nodes) == 1
        assert isinstance(loaded_workflow.nodes[0], SampleAddNode)
        assert loaded_workflow.nodes[0].id == "node1"

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
        node1 = Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")
        node2 = Node.model_construct(type="SampleMultiply", id="node2", version="1.0.0")

        workflow = Workflow(
            nodes=[node1, node2],
            edges=[],
            input_edges=[],
            output_edges=[],
        )

        # Load workflow
        typed_workflow = engine.load(workflow)

        # Verify both nodes are typed
        assert len(typed_workflow.nodes) == 2
        assert isinstance(typed_workflow.nodes[0], SampleAddNode)
        assert isinstance(typed_workflow.nodes[1], SampleMultiplyNode)

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
        untyped_node = Node.model_construct(
            type="UnregisteredType", id="node1", version="1.0.0"
        )
        workflow = Workflow.model_construct(
            nodes=[untyped_node],
            edges=[],
            input_edges=[],
            output_edges=[],
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
        workflow_add = Workflow.model_construct(
            nodes=[Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")],
            edges=[],
            input_edges=[],
            output_edges=[],
        )

        workflow_a = engine_a.load(workflow_add)
        workflow_b = engine_b.load(workflow_add)

        assert isinstance(workflow_a.nodes[0], SampleAddNode)
        assert isinstance(workflow_b.nodes[0], SampleAddNode)

        # Workflow with SampleMultiply node - only engine A can load
        workflow_multiply = Workflow.model_construct(
            nodes=[
                Node.model_construct(type="SampleMultiply", id="node1", version="1.0.0")
            ],
            edges=[],
            input_edges=[],
            output_edges=[],
        )

        workflow_a_mult = engine_a.load(workflow_multiply)
        assert isinstance(workflow_a_mult.nodes[0], SampleMultiplyNode)

        # Engine B cannot load SampleMultiply
        with pytest.raises(
            ValueError, match='Node type "SampleMultiply" is not registered'
        ):
            engine_b.load(workflow_multiply)

    def test_load_preserves_workflow_structure(self):
        """Test that loading preserves edges and other workflow structure."""
        from workflow_engine.core.edge import InputEdge, OutputEdge

        node_registry = NodeRegistry.builder(lazy=True)
        node_registry.register_node_class("SampleAdd", SampleAddNode)
        node_registry.register_base_node_class(Node)

        value_registry = ValueRegistry.builder(lazy=True)

        engine = WorkflowEngine(
            node_registry=node_registry,
            value_registry=value_registry,
        )

        # Create workflow with input/output edges (simpler than internal edges)
        node1 = Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")

        input_edge = InputEdge(
            input_key="x",
            target_id="node1",
            target_key="input",
        )

        output_edge = OutputEdge(
            source_id="node1",
            source_key="output",
            output_key="y",
        )

        workflow = Workflow.model_construct(
            nodes=[node1],
            edges=[],
            input_edges=[input_edge],
            output_edges=[output_edge],
        )

        # Load workflow
        typed_workflow = engine.load(workflow)

        # Verify edges are preserved
        assert len(typed_workflow.input_edges) == 1
        assert len(typed_workflow.output_edges) == 1
        assert typed_workflow.input_edges[0].input_key == "x"
        assert typed_workflow.output_edges[0].output_key == "y"


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
        workflow_add = Workflow.model_construct(
            nodes=[Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")],
            edges=[],
            input_edges=[],
            output_edges=[],
        )

        # Tenant A can load it
        loaded_a = engine_a.load(workflow_add)
        assert isinstance(loaded_a.nodes[0], SampleAddNode)

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
        workflow = Workflow(
            nodes=[Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")],
            edges=[],
            input_edges=[],
            output_edges=[],
        )

        # Both engines can load the same workflow independently
        loaded_1 = engine_1.load(workflow)
        loaded_2 = engine_2.load(workflow)

        # Both should produce typed workflows
        assert isinstance(loaded_1.nodes[0], SampleAddNode)
        assert isinstance(loaded_2.nodes[0], SampleAddNode)


class TestWorkflowEngineExecution:
    """Tests for WorkflowEngine.execute() method."""

    @pytest.mark.asyncio
    async def test_execute_with_default_algorithm(self):
        """Test that execute() works with default TopologicalExecutionAlgorithm."""
        from workflow_engine import IntegerValue
        from workflow_engine.contexts import InMemoryContext
        from workflow_engine.nodes import AddNode

        # Create engine with defaults
        engine = WorkflowEngine()

        # Create a simple addition workflow
        node = AddNode(id="add1")

        workflow = Workflow(
            nodes=[node],
            edges=[],
            input_edges=[
                {"input_key": "a", "target_id": "add1", "target_key": "a"},
                {"input_key": "b", "target_id": "add1", "target_key": "b"},
            ],
            output_edges=[
                {"source_id": "add1", "source_key": "sum", "output_key": "result"}
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
        from workflow_engine import IntegerValue
        from workflow_engine.contexts import InMemoryContext
        from workflow_engine.execution import ParallelExecutionAlgorithm
        from workflow_engine.nodes import AddNode

        # Create engine with custom algorithm
        custom_algorithm = ParallelExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=custom_algorithm)

        # Verify the algorithm is stored
        assert engine.execution_algorithm is custom_algorithm

        # Create a simple workflow
        node = AddNode(id="add1")

        workflow = Workflow(
            nodes=[node],
            edges=[],
            input_edges=[
                {"input_key": "a", "target_id": "add1", "target_key": "a"},
                {"input_key": "b", "target_id": "add1", "target_key": "b"},
            ],
            output_edges=[
                {"source_id": "add1", "source_key": "sum", "output_key": "result"}
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
        from workflow_engine import IntegerValue
        from workflow_engine.contexts import InMemoryContext
        from workflow_engine.nodes import AddNode

        # Create engine
        engine = WorkflowEngine()

        # Create a simple workflow
        node = AddNode(id="add1")

        workflow = Workflow(
            nodes=[node],
            edges=[],
            input_edges=[
                {"input_key": "a", "target_id": "add1", "target_key": "a"},
                {"input_key": "b", "target_id": "add1", "target_key": "b"},
            ],
            output_edges=[
                {"source_id": "add1", "source_key": "sum", "output_key": "result"}
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
        from workflow_engine import IntegerValue
        from workflow_engine.contexts import InMemoryContext
        from workflow_engine.nodes import AddNode

        # Create engine
        engine = WorkflowEngine()

        # Create workflow with typed node
        typed_node = AddNode(id="add1")

        workflow = Workflow(
            nodes=[typed_node],
            edges=[],
            input_edges=[
                {"input_key": "a", "target_id": "add1", "target_key": "a"},
                {"input_key": "b", "target_id": "add1", "target_key": "b"},
            ],
            output_edges=[
                {"source_id": "add1", "source_key": "sum", "output_key": "result"}
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
        from workflow_engine import IntegerValue
        from workflow_engine.contexts import InMemoryContext
        from workflow_engine.nodes import AddNode

        # Create engine
        engine = WorkflowEngine()

        # Create workflow: (a + b) + c
        add_node1 = AddNode(id="add1")
        add_node2 = AddNode(id="add2")

        workflow = Workflow(
            nodes=[add_node1, add_node2],
            edges=[
                {
                    "source_id": "add1",
                    "source_key": "sum",
                    "target_id": "add2",
                    "target_key": "a",
                }
            ],
            input_edges=[
                {"input_key": "a", "target_id": "add1", "target_key": "a"},
                {"input_key": "b", "target_id": "add1", "target_key": "b"},
                {"input_key": "c", "target_id": "add2", "target_key": "b"},
            ],
            output_edges=[
                {
                    "source_id": "add2",
                    "source_key": "sum",
                    "output_key": "result",
                }
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
        from workflow_engine.contexts import InMemoryContext

        # Create tenant-specific registry with only SampleAddNode
        tenant_registry = NodeRegistry.builder(lazy=True)
        tenant_registry.register_node_class("SampleAdd", SampleAddNode)
        tenant_registry.register_base_node_class(Node)

        engine = WorkflowEngine(
            node_registry=tenant_registry,
            value_registry=ValueRegistry.builder(lazy=True),
        )

        # Create workflow with SampleAdd node
        workflow = Workflow.model_construct(
            nodes=[Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")],
            edges=[],
            input_edges=[],
            output_edges=[],
        )

        # Execute workflow
        context = InMemoryContext()
        errors, output = await engine.execute(workflow, {}, context)

        # Should execute successfully (no errors)
        assert len(errors.workflow_errors) == 0
        assert len(errors.node_errors) == 0
