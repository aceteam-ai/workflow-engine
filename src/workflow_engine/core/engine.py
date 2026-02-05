# workflow_engine/core/workflow_engine.py
from .node import NodeRegistry
from .values.value import ValueRegistry
from .workflow import Workflow


class WorkflowEngine:
    """
    WorkflowEngine manages type resolution for workflows using isolated registries.

    Each engine instance has its own registries, enabling multi-tenancy where
    different organizations can have different sets of available nodes and values.
    """

    def __init__(
        self,
        *,
        node_registry: NodeRegistry = NodeRegistry.DEFAULT,
        value_registry: ValueRegistry = ValueRegistry.DEFAULT,
    ):
        """
        Create a WorkflowEngine with isolated registries.

        Args:
            node_registry: Registry of available node types.
                Defaults to the global _default_registry if not provided.
            value_registry: Registry of available value types.
                Defaults to the global default_value_registry if not provided.
        """
        self.node_registry = (
            node_registry if node_registry is not None else NodeRegistry.DEFAULT
        )
        self.value_registry = (
            value_registry if value_registry is not None else ValueRegistry.DEFAULT
        )

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
        typed_nodes = []

        for node in workflow.nodes:
            # Load the node (will return unchanged if already concrete)
            typed_node = self.node_registry.load_node(node)
            typed_nodes.append(typed_node)

        # Return new workflow with typed nodes
        return workflow.model_update(nodes=typed_nodes)


__all__ = [
    "WorkflowEngine",
]
