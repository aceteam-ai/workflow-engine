"""Tests for WorkflowEngine."""

from typing import Type

import pytest
from overrides import override

from workflow_engine import (
    Edge,
    Empty,
    ExecutionContext,
    InputNode,
    IntegerValue,
    Node,
    NodeRegistry,
    NodeTypeInfo,
    OutputNode,
    ValueRegistry,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.execution import ParallelExecutionAlgorithm
from workflow_engine.nodes import AddNode


# Test fixtures - simple node classes for testing
class SampleAddNode(Node[Empty, Empty, Empty]):
    TYPE_INFO = NodeTypeInfo.from_parameter_type(
        display_name="Sample Add",
        version="1.0.0",
        parameter_type=Empty,
    )

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
    ) -> Empty:
        return Empty()


class SampleMultiplyNode(Node[Empty, Empty, Empty]):
    TYPE_INFO = NodeTypeInfo.from_parameter_type(
        display_name="Sample Multiply",
        version="1.0.0",
        parameter_type=Empty,
    )

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
    ) -> Empty:
        return Empty()


def _build_registry(*node_classes: type[Node]) -> NodeRegistry:
    """Build a registry with the given node classes plus Node, Input, Output."""
    builder = (
        NodeRegistry.builder(lazy=True)
        .register(Node, name="Node")
        .register(InputNode)
        .register(OutputNode)
    )
    for cls in node_classes:
        builder.register(cls)
    return builder.build()


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

    async def test_default_engine_loads_workflow(self):
        """Test that default engine can load workflows with built-in node types."""
        engine = WorkflowEngine()

        workflow = Workflow.model_construct(
            input_node=engine.create_input_node(),
            inner_nodes=[
                Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")
            ],
            output_node=engine.create_output_node(),
            edges=[],
        )

        typed_workflow = await engine.validate(workflow)
        assert isinstance(typed_workflow.inner_nodes[0], SampleAddNode)

    async def test_load_workflow_with_untyped_nodes(self):
        """Test loading a workflow with untyped nodes into typed nodes."""
        engine = WorkflowEngine(
            node_registry=_build_registry(SampleAddNode),
            value_registry=ValueRegistry.builder(lazy=True),
        )

        untyped_node = Node.model_construct(
            type="SampleAdd", id="node1", version="1.0.0"
        )
        workflow = Workflow(
            input_node=engine.create_input_node(),
            inner_nodes=[untyped_node],
            output_node=engine.create_output_node(),
            edges=[],
        )

        typed_workflow = await engine.validate(workflow)

        assert len(typed_workflow.inner_nodes) == 1
        assert isinstance(typed_workflow.inner_nodes[0], SampleAddNode)
        assert typed_workflow.inner_nodes[0].type == "SampleAdd"
        assert typed_workflow.inner_nodes[0].id == "node1"

    async def test_load_workflow_with_already_typed_nodes(self):
        """Test that loading a workflow with typed nodes returns equivalent workflow."""
        engine = WorkflowEngine(
            node_registry=_build_registry(SampleAddNode),
            value_registry=ValueRegistry.builder(lazy=True),
        )

        typed_node = engine.create_node(SampleAddNode, id="node1")
        workflow = Workflow(
            input_node=engine.create_input_node(),
            inner_nodes=[typed_node],
            output_node=engine.create_output_node(),
            edges=[],
        )

        loaded_workflow = await engine.validate(workflow)

        assert len(loaded_workflow.inner_nodes) == 1
        assert isinstance(loaded_workflow.inner_nodes[0], SampleAddNode)
        assert loaded_workflow.inner_nodes[0].id == "node1"

    async def test_load_workflow_with_multiple_nodes(self):
        """Test loading a workflow with multiple untyped nodes."""
        engine = WorkflowEngine(
            node_registry=_build_registry(SampleAddNode, SampleMultiplyNode),
            value_registry=ValueRegistry.builder(lazy=True),
        )

        node1 = Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")
        node2 = Node.model_construct(type="SampleMultiply", id="node2", version="1.0.0")

        workflow = Workflow(
            input_node=engine.create_input_node(),
            inner_nodes=[node1, node2],
            output_node=engine.create_output_node(),
            edges=[],
        )

        typed_workflow = await engine.validate(workflow)

        assert len(typed_workflow.inner_nodes) == 2
        assert isinstance(typed_workflow.inner_nodes[0], SampleAddNode)
        assert isinstance(typed_workflow.inner_nodes[1], SampleMultiplyNode)

    async def test_load_workflow_with_unregistered_node_type_raises_error(self):
        """Test that loading a workflow with unregistered node type raises ValueError."""
        engine = WorkflowEngine(
            node_registry=_build_registry(),
            value_registry=ValueRegistry.builder(lazy=True),
        )

        untyped_node = Node.model_construct(
            type="UnregisteredType",
            id="node1",
            version="1.0.0",
        )
        workflow = Workflow.model_construct(
            input_node=engine.create_input_node(),
            inner_nodes=[untyped_node],
            output_node=engine.create_output_node(),
            edges=[],
        )

        with pytest.raises(
            ValueError, match='Node type "UnregisteredType" is not registered'
        ):
            await engine.validate(workflow)

    async def test_load_workflow_with_different_registries(self):
        """Test that different engines with different registries have different capabilities."""
        engine_a = WorkflowEngine(
            node_registry=_build_registry(SampleAddNode, SampleMultiplyNode),
            value_registry=ValueRegistry.builder(lazy=True),
        )
        engine_b = WorkflowEngine(
            node_registry=_build_registry(SampleAddNode),
            value_registry=ValueRegistry.builder(lazy=True),
        )

        workflow_add = Workflow.model_construct(
            input_node=engine_a.create_input_node(),
            inner_nodes=[
                Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")
            ],
            output_node=engine_a.create_output_node(),
            edges=[],
        )

        workflow_a = await engine_a.validate(workflow_add)
        workflow_b = await engine_b.validate(workflow_add)

        assert isinstance(workflow_a.inner_nodes[0], SampleAddNode)
        assert isinstance(workflow_b.inner_nodes[0], SampleAddNode)

        workflow_multiply = Workflow.model_construct(
            input_node=engine_a.create_input_node(),
            inner_nodes=[
                Node.model_construct(type="SampleMultiply", id="node1", version="1.0.0")
            ],
            output_node=engine_a.create_output_node(),
            edges=[],
        )

        workflow_a_mult = await engine_a.validate(workflow_multiply)
        assert isinstance(workflow_a_mult.inner_nodes[0], SampleMultiplyNode)

        with pytest.raises(
            ValueError, match='Node type "SampleMultiply" is not registered'
        ):
            await engine_b.validate(workflow_multiply)

    async def test_load_preserves_workflow_structure(self):
        """Test that loading preserves edges and other workflow structure."""
        engine = WorkflowEngine(
            node_registry=_build_registry(SampleAddNode),
            value_registry=ValueRegistry.builder(lazy=True),
        )

        node1 = Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")
        node2 = Node.model_construct(type="SampleAdd", id="node2", version="1.0.0")

        workflow = Workflow.model_construct(
            input_node=engine.create_input_node(),
            inner_nodes=[node1, node2],
            output_node=engine.create_output_node(),
            edges=[],
        )

        typed_workflow = await engine.validate(workflow)

        assert len(typed_workflow.inner_nodes) == 2
        assert typed_workflow.input_node.id == "input"
        assert typed_workflow.output_node.id == "output"
        assert typed_workflow.inner_nodes[0].id == "node1"
        assert typed_workflow.inner_nodes[1].id == "node2"


class TestWorkflowEngineMultiTenancy:
    """Tests for multi-tenancy scenarios with WorkflowEngine."""

    async def test_tenant_isolation(self):
        """Test that different tenants can have different node types available."""
        engine_a = WorkflowEngine(node_registry=_build_registry(SampleAddNode))
        engine_b = WorkflowEngine(node_registry=_build_registry(SampleMultiplyNode))

        workflow_add = Workflow.model_construct(
            input_node=engine_a.create_input_node(),
            inner_nodes=[
                Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")
            ],
            output_node=engine_a.create_output_node(),
            edges=[],
        )

        loaded_a = await engine_a.validate(workflow_add)
        assert isinstance(loaded_a.inner_nodes[0], SampleAddNode)

        with pytest.raises(ValueError, match='Node type "SampleAdd" is not registered'):
            await engine_b.validate(workflow_add)

    async def test_shared_workflow_different_engines(self):
        """Test that the same workflow JSON can be loaded by different engines."""
        engine_1 = WorkflowEngine(
            node_registry=_build_registry(SampleAddNode),
            value_registry=ValueRegistry.builder(lazy=True),
        )
        engine_2 = WorkflowEngine(
            node_registry=_build_registry(SampleAddNode),
            value_registry=ValueRegistry.builder(lazy=True),
        )

        workflow = Workflow(
            input_node=engine_1.create_input_node(),
            inner_nodes=[
                Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")
            ],
            output_node=engine_1.create_output_node(),
            edges=[],
        )

        validated_workflow_1 = await engine_1.validate(workflow)
        validated_workflow_2 = await engine_2.validate(workflow)

        assert isinstance(validated_workflow_1.inner_nodes[0], SampleAddNode)
        assert isinstance(validated_workflow_2.inner_nodes[0], SampleAddNode)


class TestWorkflowEngineExecution:
    """Tests for WorkflowEngine.execute() method."""

    async def test_execute_with_default_algorithm(self):
        """Test that execute() works with default TopologicalExecutionAlgorithm."""
        engine = WorkflowEngine()

        workflow = Workflow(
            input_node=(
                input_node := engine.create_input_node(
                    a=IntegerValue,
                    b=IntegerValue,
                )
            ),
            output_node=(output_node := engine.create_output_node(result=IntegerValue)),
            inner_nodes=[
                add_node := engine.create_node(AddNode, id="add1"),
            ],
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

        context = InMemoryExecutionContext()
        input_data = {"a": IntegerValue(5), "b": IntegerValue(3)}
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input=input_data,
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.output["result"].root == 8

    async def test_execute_with_custom_algorithm(self):
        """Test that execute() uses the provided execution algorithm."""
        custom_algorithm = ParallelExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=custom_algorithm)

        assert engine.execution_algorithm is custom_algorithm

        workflow = Workflow(
            input_node=(
                input_node := engine.create_input_node(
                    a=IntegerValue,
                    b=IntegerValue,
                )
            ),
            output_node=(output_node := engine.create_output_node(result=IntegerValue)),
            inner_nodes=[
                add_node := engine.create_node(AddNode, id="add1"),
            ],
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

        context = InMemoryExecutionContext()
        input_data = {"a": IntegerValue(10), "b": IntegerValue(20)}
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input=input_data,
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.output["result"].root == 30

    async def test_execute_calls_load_internally(self):
        """Test that execute() calls load() internally before execution."""
        engine = WorkflowEngine()

        workflow = Workflow(
            input_node=(
                input_node := engine.create_input_node(
                    a=IntegerValue,
                    b=IntegerValue,
                )
            ),
            output_node=(output_node := engine.create_output_node(result=IntegerValue)),
            inner_nodes=[
                add_node := engine.create_node(AddNode, id="add1"),
            ],
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

        context = InMemoryExecutionContext()
        input_data = {"a": IntegerValue(7), "b": IntegerValue(4)}
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input=input_data,
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.output["result"].root == 11

    async def test_execute_with_typed_workflow(self):
        """Test that execute() works with already typed workflows."""
        engine = WorkflowEngine()

        workflow = Workflow(
            input_node=(
                input_node := engine.create_input_node(
                    a=IntegerValue,
                    b=IntegerValue,
                )
            ),
            output_node=(output_node := engine.create_output_node(result=IntegerValue)),
            inner_nodes=[
                typed_node := engine.create_node(AddNode, id="add1"),
            ],
            edges=[
                Edge.from_nodes(
                    source=input_node,
                    source_key="a",
                    target=typed_node,
                    target_key="a",
                ),
                Edge.from_nodes(
                    source=input_node,
                    source_key="b",
                    target=typed_node,
                    target_key="b",
                ),
                Edge.from_nodes(
                    source=typed_node,
                    source_key="sum",
                    target=output_node,
                    target_key="result",
                ),
            ],
        )

        context = InMemoryExecutionContext()
        input_data = {"a": IntegerValue(15), "b": IntegerValue(25)}
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input=input_data,
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.output["result"].root == 40

    async def test_execute_with_multiple_nodes(self):
        """Test execute() with a workflow containing multiple nodes."""
        engine = WorkflowEngine()

        workflow = Workflow(
            input_node=(
                input_node := engine.create_input_node(
                    a=IntegerValue,
                    b=IntegerValue,
                    c=IntegerValue,
                )
            ),
            output_node=(output_node := engine.create_output_node(result=IntegerValue)),
            inner_nodes=[
                add_node1 := engine.create_node(AddNode, id="add1"),
                add_node2 := engine.create_node(AddNode, id="add2"),
            ],
            edges=[
                Edge.from_nodes(
                    source=input_node, source_key="a", target=add_node1, target_key="a"
                ),
                Edge.from_nodes(
                    source=input_node, source_key="b", target=add_node1, target_key="b"
                ),
                Edge.from_nodes(
                    source=add_node1,
                    source_key="sum",
                    target=add_node2,
                    target_key="a",
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

        context = InMemoryExecutionContext()
        input_data = {
            "a": IntegerValue(2),
            "b": IntegerValue(3),
            "c": IntegerValue(4),
        }
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input=input_data,
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.output["result"].root == 9

    async def test_execute_with_tenant_specific_engine(self):
        """Test that tenant-specific engines execute with isolated registries."""
        engine = WorkflowEngine(
            node_registry=_build_registry(SampleAddNode),
            value_registry=ValueRegistry.builder(lazy=True),
        )

        workflow = Workflow.model_construct(
            input_node=engine.create_input_node(),
            inner_nodes=[
                Node.model_construct(type="SampleAdd", id="node1", version="1.0.0")
            ],
            output_node=engine.create_output_node(),
            edges=[],
        )

        context = InMemoryExecutionContext()
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
