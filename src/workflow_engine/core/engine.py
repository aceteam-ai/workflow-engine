# workflow_engine/core/workflow_engine.py
from .context import Context
from .error import WorkflowErrors
from .execution import ExecutionAlgorithm
from .node import NodeRegistry
from .values import DataMapping
from .values.value import ValueRegistry
from .workflow import Workflow


class WorkflowEngine:
    """
    WorkflowEngine manages type resolution and execution for workflows using isolated registries.

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

    def load(self, workflow: Workflow) -> Workflow:
        """
        Convert an untyped workflow to a typed workflow.

        Walks the workflow graph and:
        1. Looks up concrete node types in node_registry
        2. Applies migrations via the registry's load_node method
        3. Returns a new Workflow with typed nodes

        Args:
            workflow: Untyped workflow (nodes may be base Node instances)

        Returns:
            Typed workflow (nodes are concrete subclass instances)
        """
        typed_inner_nodes = []

        for node in workflow.inner_nodes:
            # Load the node (will return unchanged if already concrete)
            typed_node = self.node_registry.load_node(node)
            typed_inner_nodes.append(typed_node)

        # Return new workflow with typed nodes
        return workflow.model_update(inner_nodes=typed_inner_nodes)

    async def execute(
        self,
        workflow: Workflow,
        input: DataMapping,
        context: Context,
    ) -> tuple[WorkflowErrors, DataMapping]:
        """
        Load and execute a workflow with the given context.

        Args:
            workflow: Workflow to execute (typed or untyped)
            input: Input data for the workflow
            context: Execution context (must be fresh for each execution)

        Returns:
            Tuple of (errors, output_data)
        """
        # Load workflow to ensure it's typed
        typed_workflow = self.load(workflow)

        # Execute using the configured algorithm
        return await self.execution_algorithm.execute(
            context=context,
            workflow=typed_workflow,
            input=input,
        )


__all__ = [
    "WorkflowEngine",
]
