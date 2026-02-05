# workflow_engine/core/workflow.py
import asyncio
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Type

import networkx as nx
from pydantic import ConfigDict, ValidationError, model_validator

from ..utils.immutable import ImmutableBaseModel
from .edge import Edge
from .error import NodeExpansionException, UserException
from .node import Node
from .values import Data, DataMapping, Value, ValueType
from .io import InputNode, OutputNode
from .values.schema import DataValueSchema

if TYPE_CHECKING:
    from .context import Context


class Workflow(ImmutableBaseModel):
    model_config = ConfigDict(frozen=True)

    input_node: InputNode
    inner_nodes: Sequence[Node]
    output_node: OutputNode
    edges: Sequence[Edge]

    @cached_property
    def nodes(self) -> Sequence[Node]:
        return (self.input_node, *self.inner_nodes, self.output_node)

    @cached_property
    def nodes_by_id(self) -> Mapping[str, Node]:
        nodes_by_id: dict[str, Node] = {}
        for node in self.nodes:
            if node.id in nodes_by_id:
                raise ValueError(f"Node {node.id} is already in the graph")
            nodes_by_id[node.id] = node
        return nodes_by_id

    @cached_property
    def edges_by_target(self) -> Mapping[str, Mapping[str, Edge]]:
        """
        A mapping from each node and input key to the (unique) edge that targets
        the node at that key.
        """
        edges_by_target: dict[str, dict[str, Edge]] = {
            node.id: {} for node in self.nodes
        }
        for edge in self.edges:
            if edge.target_key in edges_by_target[edge.target_id]:
                raise ValueError(
                    f"In-edge to {edge.target_id}.{edge.target_key} is already in the graph"
                )
            edges_by_target[edge.target_id][edge.target_key] = edge
        return edges_by_target

    @cached_property
    def input_fields(self) -> Mapping[str, tuple[ValueType, bool]]:
        """
        Returns the input fields for this workflow.

        If an InputNode has an input_schema specified, that schema's type is used.
        Otherwise, the type is inferred from the target node's input field.
        """
        return self.input_node.input_fields

    @cached_property
    def output_fields(self) -> Mapping[str, tuple[ValueType, bool]]:
        """
        Returns the output fields for this workflow.

        If an OutputNode has an output_schema specified, that schema's type is used.
        Otherwise, the type is inferred from the source node's output field.
        """
        return self.output_node.output_fields

    @cached_property
    def input_type(self) -> Type[Data]:
        return self.input_node.input_type

    @cached_property
    def output_type(self) -> Type[Data]:
        return self.output_node.output_type

    @cached_property
    def input_schema(self) -> DataValueSchema:
        return self.input_node.input_schema

    @cached_property
    def output_schema(self) -> DataValueSchema:
        return self.output_node.output_schema

    @cached_property
    def input_edges(self) -> Sequence[Edge]:
        return [edge for edge in self.edges if edge.source_id == self.input_node.id]

    @cached_property
    def output_edges(self) -> Sequence[Edge]:
        return [edge for edge in self.edges if edge.target_id == self.output_node.id]

    @cached_property
    def nx_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()

        for node in self.nodes:
            graph.add_node(node.id, data=node)

        for edge in self.edges:
            graph.add_edge(edge.source_id, edge.target_id, data=edge)

        return graph

    @model_validator(mode="after")
    def _validate_dag(self):
        if not nx.is_directed_acyclic_graph(self.nx_graph):
            cycles = list(nx.simple_cycles(self.nx_graph))
            raise ValueError(f"Workflow graph is not a DAG. Cycles found: {cycles}")
        return self

    @model_validator(mode="after")
    def _validate_no_id_prefix_collisions(self):
        """
        Ensure no node ID is a prefix of another when followed by '/'.

        This prevents ID collisions when composite nodes are expanded.
        For example, this prevents having both 'foo' and 'foo/bar' nodes.
        """
        node_ids = [node.id for node in self.nodes]
        sorted_ids = sorted(node_ids)

        for i in range(len(sorted_ids) - 1):
            current = sorted_ids[i]
            next_id = sorted_ids[i + 1]
            if next_id.startswith(current + "/"):
                raise ValueError(
                    f"Node ID collision detected: '{current}' is a prefix of '{next_id}'. "
                    f"This would cause conflicts when composite nodes are expanded. "
                    f"Please ensure no node ID is a prefix of another when followed by '/'."
                )
        return self

    def get_ready_nodes(
        self,
        node_outputs: Mapping[str, DataMapping] | None = None,
        partial_results: Mapping[str, DataMapping] | None = None,
    ) -> Mapping[str, DataMapping]:
        """
        Given the input and the set of nodes which have already finished, return
        the nodes that are now ready to be executed and their arguments.

        Should only return an empty map if the entire workflow is finished.

        For efficiency, this method can use partial results to avoid
        recalculating already finished nodes.
        """
        if node_outputs is None:
            node_outputs = {}

        ready_nodes: dict[str, DataMapping] = (
            {} if partial_results is None else dict(partial_results)
        )
        for node in self.nodes:
            # remove the node if it is now finished
            if node.id in node_outputs:
                if node.id in ready_nodes:
                    ready_nodes.pop(node.id)
                continue
            # skip the node if it is already in the ready set
            if node.id in ready_nodes:
                continue

            # node might be ready, we have to check all its input edges
            ready: bool = True
            node_input_dict: DataMapping = {}
            for target_key, edge in self.edges_by_target[node.id].items():
                # if the input is missing, we will let the node figure it out
                if edge.source_id in node_outputs:
                    node_input_dict[target_key] = node_outputs[edge.source_id][
                        edge.source_key
                    ]
                else:
                    ready = False
                    break
            if not ready:
                continue

            try:
                ready_nodes[node.id] = node_input_dict
            except ValidationError as e:
                raise UserException(
                    f"Input {node_input_dict} for node {node.id} is invalid: {e}",
                )
        return ready_nodes

    async def get_output(
        self,
        *,
        context: "Context",
        node_outputs: Mapping[str, DataMapping],
        partial: bool = False,
    ) -> DataMapping:
        """
        Get the output of the workflow, casting values to expected output types.

        This method validates that all outputs can be cast to their expected types
        and performs the casting in parallel.

        Args:
            node_outputs: Mapping from node IDs to their output data
            context: Execution context used for casting operations
            partial: If True, skip missing outputs instead of raising exceptions

        Returns:
            DataMapping with all outputs cast to their expected types

        Raises:
            UserException: If a required output is missing or cannot be cast
        """
        # First pass: Validate all outputs exist and can be cast
        outputs_to_cast: list[tuple[str, Value, ValueType]] = []

        for edge in self.output_edges:
            if edge.source_id not in node_outputs:
                if partial:
                    continue
                raise UserException(
                    f"Cannot get output from node {edge.source_id}.",
                )
            node_output = node_outputs[edge.source_id]
            if edge.source_key not in node_output:
                if partial:
                    continue
                raise UserException(
                    f"Cannot get output from node {edge.source_id} at key '{edge.source_key}'.",
                )

            output_field = node_output[edge.source_key]
            expected_type, _ = self.output_fields[edge.target_key]

            # Validate that the output can be cast to the expected type
            if not output_field.can_cast_to(expected_type):
                raise UserException(
                    f"Output '{edge.target_key}' from node {edge.source_id}.{edge.source_key} "
                    f"cannot be cast: {output_field} is not assignable to {expected_type}"
                )

            outputs_to_cast.append((edge.target_key, output_field, expected_type))

        # Second pass: Cast all outputs in parallel
        cast_tasks = [
            output_field.cast_to(expected_type, context=context)
            for _, output_field, expected_type in outputs_to_cast
        ]

        if len(cast_tasks) == 0:
            return {}

        casted_values = await asyncio.gather(*cast_tasks)

        # Build the result dictionary
        output: DataMapping = {}
        for (output_key, _, _), casted_value in zip(
            outputs_to_cast,
            casted_values,
            strict=True,
        ):
            output[output_key] = casted_value

        return output

    def expand_node(
        self,
        node_id: str,
        subgraph: "Workflow",
    ) -> "Workflow":
        """
        Replace a node in this workflow with a subgraph.

        This method performs graph surgery to replace a node with a subgraph.
        The subgraph's nodes are namespaced with the original node's ID to prevent
        ID collisions. Input and output edges are reconnected appropriately.

        Args:
            node_id: ID of the node to replace
            subgraph: The workflow to insert in place of the node
            node_input: The input that was passed to the original node

        Returns:
            A new Workflow with the node replaced by the subgraph

        Raises:
            ValueError: If the node_id doesn't exist or if the replacement would
                       create an invalid graph
        """
        try:
            if node_id not in self.nodes_by_id:
                raise ValueError(f"Node {node_id} not found in workflow")

            subgraph = subgraph.with_namespace(node_id)

            # Collect all edges that need to be modified
            new_inner_nodes: list[Node] = [
                node for node in self.inner_nodes if node.id != node_id
            ] + list(subgraph.nodes)
            new_edges: list[Edge] = list(subgraph.edges)

            for edge in self.edges:
                if edge.target_id == node_id:
                    # Only create the edge if the subgraph's input_node has this field
                    if edge.target_key in subgraph.input_node.input_fields:
                        new_edges.append(
                            Edge(
                                source_id=edge.source_id,
                                source_key=edge.source_key,
                                target_id=subgraph.input_node.id,
                                target_key=edge.target_key,
                            )
                        )
                    # Edges to fields not in the subgraph input are dropped
                    # (e.g., control fields like 'condition' for IfElseNode)
                elif edge.source_id == node_id:
                    new_edges.append(
                        Edge(
                            source_id=subgraph.output_node.id,
                            source_key=edge.source_key,
                            target_id=edge.target_id,
                            target_key=edge.target_key,
                        )
                    )
                else:
                    new_edges.append(edge)

            return Workflow(
                inner_nodes=new_inner_nodes,
                input_node=self.input_node,
                output_node=self.output_node,
                edges=new_edges,
            )
        except Exception as e:
            raise NodeExpansionException(node_id, workflow=subgraph) from e

    def with_namespace(self, namespace: str) -> "Workflow":
        """
        Create a copy of this workflow with all node IDs namespaced.

        Args:
            namespace: The namespace to prefix all node IDs with

        Returns:
            A new Workflow with all node IDs prefixed with '{namespace}/'
        """
        # Create namespaced nodes
        return Workflow(
            input_node=self.input_node.with_namespace(namespace),
            inner_nodes=[node.with_namespace(namespace) for node in self.inner_nodes],
            output_node=self.output_node.with_namespace(namespace),
            edges=[edge.with_namespace(namespace) for edge in self.edges],
        )


class WorkflowValue(Value[Workflow]):
    pass


__all__ = [
    "Workflow",
    "WorkflowValue",
]
