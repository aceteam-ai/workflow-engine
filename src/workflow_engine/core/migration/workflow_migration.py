# workflow_engine/core/migration/workflow_migration.py
"""Utilities for handling workflow data after node migrations."""

import logging
from typing import TYPE_CHECKING, Any

from ..edge import Edge, InputEdge, OutputEdge
from ..node import Node

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
                      - nodes: list of node dicts
                      - edges: list of edge dicts
                      - input_edges: list of input edge dicts
                      - output_edges: list of output edge dicts

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

    # Parse nodes first
    nodes_data = workflow_data.get("nodes", [])
    if not nodes_data:
        return workflow_data

    try:
        nodes = [Node.model_validate(node_data) for node_data in nodes_data]
    except Exception:
        # If nodes can't be parsed, return unchanged
        return workflow_data

    # Check if any nodes were migrated
    any_migration_occurred = False
    for node, node_data in zip(nodes, nodes_data, strict=False):
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

    nodes_by_id = {node.id: node for node in nodes}

    # Filter regular edges
    edges_data = workflow_data.get("edges", [])
    valid_edges = _filter_edges(edges_data, nodes_by_id)

    # Filter input edges
    input_edges_data = workflow_data.get("input_edges", [])
    valid_input_edges = _filter_input_edges(input_edges_data, nodes_by_id)

    # Filter output edges
    output_edges_data = workflow_data.get("output_edges", [])
    valid_output_edges = _filter_output_edges(output_edges_data, nodes_by_id)

    # Return modified data with filtered edges
    result = dict(workflow_data)
    result["edges"] = valid_edges
    result["input_edges"] = valid_input_edges
    result["output_edges"] = valid_output_edges
    return result


def _filter_edges(
    edges_data: list[dict[str, Any]], nodes_by_id: dict[str, Node]
) -> list[dict[str, Any]]:
    """Filter regular edges, removing those with invalid references."""
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


def _filter_input_edges(
    input_edges_data: list[dict[str, Any]], nodes_by_id: dict[str, Node]
) -> list[dict[str, Any]]:
    """Filter input edges, removing those with invalid references."""
    valid_input_edges = []
    for edge_data in input_edges_data:
        try:
            edge = InputEdge.model_validate(edge_data)
            if edge.target_id not in nodes_by_id:
                logger.warning(
                    f"Removing input edge to {edge.target_id}.{edge.target_key}: node not found"
                )
                continue

            target = nodes_by_id[edge.target_id]

            # Check if target field exists
            if edge.target_key not in target.input_fields:
                logger.warning(
                    f"Removing input edge '{edge.input_key}' to {edge.target_id}.{edge.target_key}: "
                    f"field does not exist on target node"
                )
                continue

            # If edge has a schema, validate type compatibility
            if edge.input_schema is not None:
                edge_input_type = edge.input_schema.to_value_cls()
                target_input_type, _ = target.input_fields[edge.target_key]
                if not edge_input_type.can_cast_to(target_input_type):
                    logger.warning(
                        f"Removing input edge '{edge.input_key}' to {edge.target_id}.{edge.target_key}: "
                        f"type {edge_input_type} is not assignable to {target_input_type}"
                    )
                    continue

            valid_input_edges.append(edge_data)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Removing invalid input edge '{edge_data.get('input_key')}' "
                f"to {edge_data.get('target_id')}.{edge_data.get('target_key')}: {e}"
            )
    return valid_input_edges


def _filter_output_edges(
    output_edges_data: list[dict[str, Any]], nodes_by_id: dict[str, Node]
) -> list[dict[str, Any]]:
    """Filter output edges, removing those with invalid references."""
    valid_output_edges = []
    for edge_data in output_edges_data:
        try:
            edge = OutputEdge.model_validate(edge_data)
            if edge.source_id not in nodes_by_id:
                logger.warning(
                    f"Removing output edge '{edge.output_key}' from {edge.source_id}.{edge.source_key}: "
                    f"node not found"
                )
                continue

            source = nodes_by_id[edge.source_id]

            # Check if source field exists
            if edge.source_key not in source.output_fields:
                logger.warning(
                    f"Removing output edge '{edge.output_key}' from {edge.source_id}.{edge.source_key}: "
                    f"field does not exist on source node"
                )
                continue

            # If edge has a schema, validate type compatibility
            if edge.output_schema is not None:
                source_output_type, _ = source.output_fields[edge.source_key]
                edge_output_type = edge.output_schema.to_value_cls()
                if not source_output_type.can_cast_to(edge_output_type):
                    logger.warning(
                        f"Removing output edge '{edge.output_key}' from {edge.source_id}.{edge.source_key}: "
                        f"type {source_output_type} is not assignable to {edge_output_type}"
                    )
                    continue

            valid_output_edges.append(edge_data)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Removing invalid output edge '{edge_data.get('output_key')}' "
                f"from {edge_data.get('source_id')}.{edge_data.get('source_key')}: {e}"
            )
    return valid_output_edges


def load_workflow_with_migration(workflow_data: dict[str, Any]) -> "Workflow":
    """
    Load a workflow from data, applying migrations and cleaning invalid edges.

    This is the recommended way to load workflows from JSON or other serialized
    formats. It handles the complete migration process:
    1. Nodes are automatically migrated to current versions
    2. Invalid edges (broken by migrations) are removed
    3. Workflow is validated and returned

    Args:
        workflow_data: Raw workflow data dictionary (e.g., from json.load())

    Returns:
        A validated Workflow instance

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
    from ..workflow import Workflow

    # Clean edges that may have been broken by node migrations
    cleaned_data = clean_edges_after_migration(workflow_data)

    # Validate and create workflow
    return Workflow.model_validate(cleaned_data)


__all__ = ["clean_edges_after_migration", "load_workflow_with_migration"]
