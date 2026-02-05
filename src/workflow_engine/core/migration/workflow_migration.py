# workflow_engine/core/migration/workflow_migration.py
"""Utilities for handling workflow data after node migrations."""

import logging
from typing import TYPE_CHECKING, Any

from ..edge import Edge
from ..node import Node, NodeRegistry

if TYPE_CHECKING:
    from ..workflow import Workflow

logger = logging.getLogger(__name__)


def clean_edges_after_migration(workflow_data: dict[str, Any]) -> dict[str, Any]:
    """
    Remove invalid edges from workflow data after node migrations.

    This function should be called after loading workflow data that may contain
    migrated nodes. It filters out edges that reference non-existent fields or
    have incompatible types, which can occur when node schemas change between
    versions.

    Only performs filtering if at least one node was migrated. Otherwise returns
    the data unchanged (strict validation will apply).

    Args:
        workflow_data: Raw workflow data dictionary with keys:
                      - input_node: input node dict
                      - inner_nodes: list of node dicts
                      - output_node: output node dict
                      - edges: list of edge dicts

    Returns:
        Modified workflow data with invalid edges removed. If no migrations
        occurred, returns the original data unchanged.

    Example:
        ```python
        # Load workflow JSON
        workflow_data = json.load(f)

        # Clean edges if migrations occurred
        cleaned_data = clean_edges_after_migration(workflow_data)

        # Create workflow
        workflow = Workflow.model_validate(cleaned_data)
        ```
    """
    if not isinstance(workflow_data, dict):
        return workflow_data

    # Parse all nodes (input_node, inner_nodes, output_node)
    inner_nodes_data = workflow_data.get("inner_nodes", [])
    input_node_data = workflow_data.get("input_node")
    output_node_data = workflow_data.get("output_node")

    if not inner_nodes_data and not input_node_data and not output_node_data:
        return workflow_data

    try:
        # Parse all nodes - first deserialize as base Node, then load to concrete type
        registry = NodeRegistry.DEFAULT
        inner_nodes = [
            registry.load_node(Node.model_validate(node_data))
            for node_data in inner_nodes_data
        ]

        # Parse input/output nodes - also need registry dispatch for proper typing
        input_node = (
            registry.load_node(Node.model_validate(input_node_data))
            if input_node_data
            else None
        )
        output_node = (
            registry.load_node(Node.model_validate(output_node_data))
            if output_node_data
            else None
        )
    except Exception:
        # If nodes can't be parsed, return unchanged
        return workflow_data

    # Check if any inner nodes were migrated
    any_migration_occurred = False
    for node, node_data in zip(inner_nodes, inner_nodes_data, strict=False):
        if not isinstance(node_data, dict):
            continue
        original_version = node_data.get("version")
        current_version = node.version
        if original_version != current_version:
            any_migration_occurred = True
            logger.info(
                f"Node {node.id} migrated from {original_version} to {current_version}"
            )
            break

    # Only filter edges if migrations occurred
    if not any_migration_occurred:
        return workflow_data

    logger.info("Node migrations detected, filtering invalid edges")

    # Build nodes_by_id including input/output nodes
    nodes_by_id = {node.id: node for node in inner_nodes}
    if input_node:
        nodes_by_id[input_node.id] = input_node
    if output_node:
        nodes_by_id[output_node.id] = output_node

    # Filter edges
    edges_data = workflow_data.get("edges", [])
    valid_edges = _filter_edges(edges_data, nodes_by_id)

    # Return modified data with filtered edges
    result = dict(workflow_data)
    result["edges"] = valid_edges
    return result


def _filter_edges(
    edges_data: list[dict[str, Any]], nodes_by_id: dict[str, Node]
) -> list[dict[str, Any]]:
    """Filter edges, removing those with invalid references."""
    valid_edges = []
    for edge_data in edges_data:
        try:
            edge = Edge.model_validate(edge_data)
            if edge.source_id not in nodes_by_id or edge.target_id not in nodes_by_id:
                logger.warning(
                    f"Removing edge from {edge.source_id}.{edge.source_key} to "
                    f"{edge.target_id}.{edge.target_key}: node not found"
                )
                continue
            edge.validate_types(
                source=nodes_by_id[edge.source_id],
                target=nodes_by_id[edge.target_id],
            )
            valid_edges.append(edge_data)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Removing invalid edge from {edge_data.get('source_id')}.{edge_data.get('source_key')} "
                f"to {edge_data.get('target_id')}.{edge_data.get('target_key')}: {e}"
            )
    return valid_edges


def load_workflow_with_migration(
    workflow_data: dict[str, Any],
    node_registry: NodeRegistry | None = None,
) -> "Workflow":
    """
    Load a workflow from data, applying migrations and cleaning invalid edges.

    This is the recommended way to load workflows from JSON or other serialized
    formats. It handles the complete migration process:
    1. Nodes are automatically migrated to current versions
    2. Invalid edges (broken by migrations) are removed
    3. Workflow is validated and returned with typed nodes

    Args:
        workflow_data: Raw workflow data dictionary (e.g., from json.load())
        node_registry: Registry to use for node type resolution.
            Defaults to NodeRegistry.DEFAULT.

    Returns:
        A validated Workflow instance with typed nodes

    Example:
        ```python
        import json
        from workflow_engine.core.migration import load_workflow_with_migration

        # Load workflow JSON
        with open("workflow.json") as f:
            workflow_data = json.load(f)

        # Load with migration support
        workflow = load_workflow_with_migration(workflow_data)
        ```

    Note:
        If you want strict validation without migration support, use
        `Workflow.model_validate()` directly instead.
    """
    from ..engine import WorkflowEngine
    from ..workflow import Workflow

    # Clean edges that may have been broken by node migrations
    cleaned_data = clean_edges_after_migration(workflow_data)

    # Validate and create workflow
    workflow = Workflow.model_validate(cleaned_data)

    # Load workflow to convert untyped nodes to typed nodes
    registry = node_registry if node_registry is not None else NodeRegistry.DEFAULT
    engine = WorkflowEngine(node_registry=registry)
    return engine.load(workflow)


__all__ = ["clean_edges_after_migration", "load_workflow_with_migration"]
