# workflow_engine/core/engine.py
from collections.abc import Mapping
from typing import Any, TypeVar

from typing_extensions import Self

from .config import WorkflowEngineConfig
from .context import ExecutionContext, ValidationContext
from .edge import Edge
from .execution import ExecutionAlgorithm, WorkflowExecutionResult
from .io import InputNode, OutputNode
from .node import Node, NodeRegistry, Params
from .values import Data, ValueRegistry, ValueType, get_data_dict, get_data_fields
from .workflow import ValidatedWorkflow, Workflow

N = TypeVar("N", bound=Node)


def _value_fields_from_data_type(data_type: type[Data]) -> dict[str, ValueType]:
    return {
        name: value_type for name, (value_type, _) in get_data_fields(data_type).items()
    }


class WorkflowEngine:
    """
    WorkflowEngine manages type resolution and execution for workflows using
    isolated registries.

    Unlike a Context, a WorkflowEngine instance can be shared by multiple
    workflows, possibly at the same time.

    Each engine instance has its own registries, enabling multi-tenancy where
    different organizations can have different sets of available nodes and values.
    """

    def __init__(
        self,
        *,
        node_registry: NodeRegistry = NodeRegistry.DEFAULT,
        value_registry: ValueRegistry = ValueRegistry.DEFAULT,
        execution_algorithm: ExecutionAlgorithm | None = None,
    ):
        """
        Create a WorkflowEngine with isolated registries.

        Args:
            node_registry: Registry of available node types.
                Defaults to the global _default_registry if not provided.
            value_registry: Registry of available value types.
                Defaults to the global default_value_registry if not provided.
            execution_algorithm: Strategy for executing workflows.
                Defaults to TopologicalExecutionAlgorithm if not provided.
        """
        self.node_registry = (
            node_registry if node_registry is not None else NodeRegistry.DEFAULT
        )
        self.value_registry = (
            value_registry if value_registry is not None else ValueRegistry.DEFAULT
        )
        if execution_algorithm is None:
            # Import here to avoid circular dependency
            from ..execution import TopologicalExecutionAlgorithm

            execution_algorithm = TopologicalExecutionAlgorithm()
        self.execution_algorithm = execution_algorithm

    @classmethod
    async def from_config(cls, config: WorkflowEngineConfig) -> Self:
        """
        Turns a WorkflowEngineConfig into a WorkflowEngine instance.
        Potentially asynchronous, to enable lazy initialization of the engine.
        This method lives in the WorkflowEngine class so that subclasses can
        override how the config is handled.
        """
        execution_algorithm = await config.build_execution_algorithm()
        return cls(
            node_registry=config.node_registry,
            value_registry=ValueRegistry.DEFAULT,
            execution_algorithm=execution_algorithm,
        )

    async def _get_validation_context(self) -> ValidationContext:
        """
        Builds a validation context. Override this for custom validation logic.
        """
        return ValidationContext(
            node_registry=self.node_registry,
            value_registry=self.value_registry,
        )

    def create_node(
        self,
        name: str | type[N],
        /,
        *,
        id: str,
        params: Mapping[str, Any] | Params | None = None,
        **kwargs: Any,
    ) -> N:
        """
        Create a new node instance by name.
        If a Node type is provided, we will use its default_type_name()
        """
        return self.node_registry.create_node(
            name,
            id=id,
            params=params,
            **kwargs,
        )

    def create_input_node(
        self,
        **fields: ValueType,
    ) -> InputNode:
        """
        Create a new input node instance, using whatever has been registered as
        the "Input" node type.
        """
        return self.node_registry.create_input_node(**fields)

    def create_output_node(
        self,
        **fields: ValueType,
    ) -> OutputNode:
        """
        Create a new output node instance, using whatever has been registered as
        the "Output" node type.
        """
        return self.node_registry.create_output_node(**fields)

    async def validate(
        self,
        workflow: Workflow,
    ) -> ValidatedWorkflow:
        validation_context = await self._get_validation_context()
        return await workflow.validate(context=validation_context)

    async def execute(
        self,
        *,
        context: ExecutionContext,
        workflow: Workflow,
        input: Mapping[str, Any],
    ) -> WorkflowExecutionResult:
        """
        Load and execute a workflow with the given context.

        Args:
            workflow: Workflow to execute (typed or untyped)
            input: Input data for the workflow
            context: Execution context (must be fresh for each execution)

        Returns:
            WorkflowExecutionResult
        """
        # Load workflow to ensure it's typed, even if it was already validated
        validated_workflow = await self.validate(workflow)
        validated_input = validated_workflow.input_type.model_validate(input)

        # Execute using the configured algorithm
        return await self.execution_algorithm.execute(
            context=context,
            workflow=validated_workflow,
            input=get_data_dict(validated_input),
        )

    async def build_single_node_workflow(
        self,
        node: str | type[N],
        /,
        *,
        node_id: str = "node",
        params: Mapping[str, Any] | Params | None = None,
        input_fields: Mapping[str, ValueType] | None = None,
        output_fields: Mapping[str, ValueType] | None = None,
    ) -> Workflow:
        """
        Build a minimal workflow that wires one inner node between input and output.

        When ``input_fields`` or ``output_fields`` are omitted, they are inferred
        from the node's resolved input and output types (including dynamic types
        that depend on ``params``).
        """
        inner_node = self.create_node(node, id=node_id, params=params)
        validation_context = await self._get_validation_context()
        if input_fields is None:
            input_type = await inner_node.input_type(validation_context)
            input_fields = _value_fields_from_data_type(input_type)
        if output_fields is None:
            output_type = await inner_node.output_type(validation_context)
            output_fields = _value_fields_from_data_type(output_type)

        input_node = self.create_input_node(**input_fields)
        output_node = self.create_output_node(**output_fields)
        edges = [
            Edge.from_nodes(
                source=input_node,
                source_key=key,
                target=inner_node,
                target_key=key,
            )
            for key in input_fields
        ] + [
            Edge.from_nodes(
                source=inner_node,
                source_key=key,
                target=output_node,
                target_key=key,
            )
            for key in output_fields
        ]
        return Workflow(
            input_node=input_node,
            output_node=output_node,
            inner_nodes=[inner_node],
            edges=edges,
        )

    async def execute_node(
        self,
        *,
        context: ExecutionContext,
        node: str | type[N],
        input: Mapping[str, Any],
        node_id: str = "node",
        params: Mapping[str, Any] | Params | None = None,
        input_fields: Mapping[str, ValueType] | None = None,
        output_fields: Mapping[str, ValueType] | None = None,
    ) -> WorkflowExecutionResult:
        """
        Run a single node through the full execution pipeline.

        This is the programmatic counterpart of treating each node as an
        independent tool call: a one-node workflow is assembled, validated,
        and executed with the configured algorithm and context hooks.
        """
        workflow = await self.build_single_node_workflow(
            node,
            node_id=node_id,
            params=params,
            input_fields=input_fields,
            output_fields=output_fields,
        )
        return await self.execute(
            context=context,
            workflow=workflow,
            input=input,
        )


__all__ = [
    "WorkflowEngine",
]
