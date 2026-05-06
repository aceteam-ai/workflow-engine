# workflow_engine/core/engine.py
from collections.abc import Mapping
from typing import Any, TypeVar

from typing_extensions import Self

from .config import WorkflowEngineConfig
from .context import ExecutionContext, ValidationContext
from .execution import ExecutionAlgorithm, WorkflowExecutionResult
from .io import InputNode, OutputNode
from .node import Node, NodeRegistry, Params
from .values import ValueRegistry, ValueType, get_data_dict
from .workflow import ValidatedWorkflow, Workflow

N = TypeVar("N", bound=Node)


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


__all__ = [
    "WorkflowEngine",
]
