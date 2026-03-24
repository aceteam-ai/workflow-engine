# workflow_engine/core/engine.py
from collections.abc import Mapping
from typing import Any
from .context import ExecutionContext, ValidationContext
from .execution import ExecutionAlgorithm, WorkflowExecutionResult
from .node import NodeRegistry
from .values import get_data_dict
from .values.value import ValueRegistry
from .workflow import ValidatedWorkflow, Workflow


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

    async def _get_validation_context(self) -> ValidationContext:
        """
        Builds a validation context. Override this for custom validation logic.
        """
        return ValidationContext(
            node_registry=self.node_registry,
            value_registry=self.value_registry,
        )

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
