# workflow_engine/core/workflow.py
import asyncio
from collections.abc import Mapping, Sequence
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, Type

import networkx as nx
from pydantic import ConfigDict, ValidationError, model_validator

from ..utils.immutable import ImmutableBaseModel
from .edge import Edge, InputEdge, OutputEdge
from .error import NodeExpansionException, UserException
from .node import Node
from .values import Data, DataMapping, Value, ValueType, build_data_type

if TYPE_CHECKING:
    from .context import Context


class Workflow(ImmutableBaseModel):
    model_config = ConfigDict(frozen=True)

    nodes: Sequence[Node]
    edges: Sequence[Edge]
    input_edges: Sequence[InputEdge]
    output_edges: Sequence[OutputEdge]

    @cached_property
    def nodes_by_id(self) -> Mapping[str, Node]:
        nodes_by_id: dict[str, Node] = {}
        for node in self.nodes:
            if node.id in nodes_by_id:
                raise ValueError(f"Node {node.id} is already in the graph")
            nodes_by_id[node.id] = node
        return nodes_by_id

    @cached_property
    def edges_by_target(self) -> Mapping[str, Mapping[str, Edge | InputEdge]]:
        """
        A mapping from each node and input key to the (unique) edge that targets
        the node at that key.
        """
        edges_by_target: dict[str, dict[str, Edge | InputEdge]] = {
            node.id: {} for node in self.nodes
        }
        for edge in chain(self.edges, self.input_edges):
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

        If an InputEdge has an input_schema specified, that schema's type is used.
        Otherwise, the type is inferred from the target node's input field.
        """
        return {
            edge.input_key: (
                (edge.input_schema.to_value_cls(), True)
                if edge.input_schema is not None
                else self.nodes_by_id[edge.target_id].input_fields[edge.target_key]
            )
            for edge in self.input_edges
        }

    @cached_property
    def output_fields(self) -> Mapping[str, tuple[ValueType, bool]]:
        """
        Returns the output fields for this workflow.

        If an OutputEdge has an output_schema specified, that schema's type is used.
        Otherwise, the type is inferred from the source node's output field.
        """
        return {
            edge.output_key: (
                (edge.output_schema.to_value_cls(), True)
                if edge.output_schema is not None
                else self.nodes_by_id[edge.source_id].output_fields[edge.source_key]
            )
            for edge in self.output_edges
        }

    @cached_property
    def input_type(self) -> Type[Data]:
        return build_data_type(
            "WorkflowInput",
            {
                edge.input_key: self.input_fields[edge.input_key]
                for edge in self.input_edges
            },
        )

    @cached_property
    def output_type(self) -> Type[Data]:
        return build_data_type(
            "WorkflowOutput",
            {
                edge.output_key: self.output_fields[edge.output_key]
                for edge in self.output_edges
            },
        )

    @cached_property
    def nx_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()

        for node in self.nodes:
            graph.add_node(node.id, data=node)

        for edge in self.edges:
            graph.add_edge(edge.source_id, edge.target_id, data=edge)

        return graph

    @cached_property
    def input_edges_by_key(self) -> Mapping[str, InputEdge]:
        """Index of input edges by their input_key."""
        return {edge.input_key: edge for edge in self.input_edges}

    @cached_property
    def output_edges_by_key(self) -> Mapping[str, OutputEdge]:
        """Index of output edges by their output_key."""
        return {edge.output_key: edge for edge in self.output_edges}

    @model_validator(mode="after")
    def _validate_nodes(self):
        # make sure that for each node, all input edges are present
        for node in self.nodes:
            for key, (_, required) in node.input_fields.items():
                if required and key not in self.edges_by_target[node.id]:
                    raise ValueError(f"Node {node.id} has no required input edge {key}")
        return self

    @model_validator(mode="after")
    def _validate_edges(self):
        for edge in self.edges:
            edge.validate_types(
                source=self.nodes_by_id[edge.source_id],
                target=self.nodes_by_id[edge.target_id],
            )
        return self

    @model_validator(mode="after")
    def _validate_input_edges(self):
        """
        Validate that input edges with schemas have compatible types.

        For each InputEdge with input_schema, validates that the schema can cast
        to the target node's input field type.
        """
        for edge in self.input_edges:
            if edge.input_schema is not None:
                target = self.nodes_by_id[edge.target_id]

                if edge.target_key not in target.input_fields:
                    raise ValueError(
                        f"Target node {target.id} does not have input field '{edge.target_key}'"
                    )

                edge_input_type = edge.input_schema.to_value_cls()
                target_input_type, _ = target.input_fields[edge.target_key]

                # Validate: edge schema -> target input
                if not edge_input_type.can_cast_to(target_input_type):
                    raise TypeError(
                        f"Input edge '{edge.input_key}' has invalid schema: "
                        f"{edge_input_type} is not assignable to target "
                        f"{target.id}.{edge.target_key} ({target_input_type})"
                    )

        return self

    @model_validator(mode="after")
    def _validate_output_edges(self):
        """
        Validate that output edges with schemas have compatible types.

        For each OutputEdge with output_schema, validates that the source node's
        output can cast to the schema type.
        """
        for edge in self.output_edges:
            if edge.output_schema is not None:
                source = self.nodes_by_id[edge.source_id]

                if edge.source_key not in source.output_fields:
                    raise ValueError(
                        f"Source node {source.id} does not have output field '{edge.source_key}'"
                    )

                source_output_type, _ = source.output_fields[edge.source_key]
                edge_output_type = edge.output_schema.to_value_cls()

                # Validate: source output -> edge schema
                if not source_output_type.can_cast_to(edge_output_type):
                    raise TypeError(
                        f"Output edge '{edge.output_key}' has invalid schema: "
                        f"source {source.id}.{edge.source_key} ({source_output_type}) "
                        f"is not assignable to {edge_output_type}"
                    )

        return self

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
        input: DataMapping,
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
                if isinstance(edge, InputEdge):
                    node_input_dict[target_key] = input[edge.input_key]
                elif edge.source_id in node_outputs:
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
            expected_type, _ = self.output_fields[edge.output_key]

            # Validate that the output can be cast to the expected type
            if not output_field.can_cast_to(expected_type):
                raise UserException(
                    f"Output '{edge.output_key}' from node {edge.source_id}.{edge.source_key} "
                    f"cannot be cast: {output_field} is not assignable to {expected_type}"
                )

            outputs_to_cast.append((edge.output_key, output_field, expected_type))

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
            new_nodes: list[Node] = [
                node for node in self.nodes if node.id != node_id
            ] + list(subgraph.nodes)
            new_edges: list[Edge] = list(subgraph.edges)
            new_input_edges: list[InputEdge] = []
            new_output_edges: list[OutputEdge] = []

            # Use cached properties for subgraph edge indexing
            subgraph_input_by_key = subgraph.input_edges_by_key
            subgraph_output_by_key = subgraph.output_edges_by_key

            # Handle input edges - reconnect them to subgraph input nodes
            for input_edge in self.input_edges:
                if input_edge.target_id == node_id:
                    if input_edge.target_key in subgraph_input_by_key:
                        subgraph_input_edge = subgraph_input_by_key[
                            input_edge.target_key
                        ]
                        new_input_edges.append(
                            InputEdge(
                                input_key=input_edge.input_key,
                                target_id=subgraph_input_edge.target_id,
                                target_key=subgraph_input_edge.target_key,
                            )
                        )
                    # If no matching input edge found, the edge is dropped
                else:
                    new_input_edges.append(input_edge)

            # Handle regular edges
            for edge in self.edges:
                if edge.target_id == node_id:
                    if edge.target_key in subgraph_input_by_key:
                        subgraph_input_edge = subgraph_input_by_key[edge.target_key]
                        new_edges.append(
                            Edge(
                                source_id=edge.source_id,
                                source_key=edge.source_key,
                                target_id=subgraph_input_edge.target_id,
                                target_key=subgraph_input_edge.target_key,
                            )
                        )
                    # If no matching input edge found, the edge is dropped
                elif edge.source_id == node_id:
                    if edge.source_key in subgraph_output_by_key:
                        subgraph_output_edge = subgraph_output_by_key[edge.source_key]
                        new_edges.append(
                            Edge(
                                source_id=subgraph_output_edge.source_id,
                                source_key=subgraph_output_edge.source_key,
                                target_id=edge.target_id,
                                target_key=edge.target_key,
                            )
                        )
                    else:
                        raise ValueError(
                            f"Node {node_id} has output '{edge.source_key}' that is required by "
                            f"workflow, but the subgraph does not provide a matching output edge"
                        )
                else:
                    new_edges.append(edge)

            # Handle output edges
            for output_edge in self.output_edges:
                if output_edge.source_id == node_id:
                    if output_edge.source_key in subgraph_output_by_key:
                        subgraph_output_edge = subgraph_output_by_key[
                            output_edge.source_key
                        ]
                        new_output_edges.append(
                            OutputEdge(
                                source_id=subgraph_output_edge.source_id,
                                source_key=subgraph_output_edge.source_key,
                                output_key=output_edge.output_key,
                            )
                        )
                    else:
                        raise ValueError(
                            f"Node {node_id} has output '{output_edge.source_key}' that is required by "
                            f"workflow output '{output_edge.output_key}', but the subgraph does not "
                            f"provide a matching output edge"
                        )
                else:
                    new_output_edges.append(output_edge)

            return Workflow(
                nodes=new_nodes,
                edges=new_edges,
                input_edges=new_input_edges,
                output_edges=new_output_edges,
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
        namespaced_nodes = [node.with_namespace(namespace) for node in self.nodes]

        # Create namespaced edges (update source_id and target_id)
        namespaced_edges = [
            edge.model_copy(
                update={
                    "source_id": f"{namespace}/{edge.source_id}",
                    "target_id": f"{namespace}/{edge.target_id}",
                }
            )
            for edge in self.edges
        ]

        # Create namespaced input edges (update target_id only)
        namespaced_input_edges = [
            input_edge.model_copy(
                update={"target_id": f"{namespace}/{input_edge.target_id}"}
            )
            for input_edge in self.input_edges
        ]

        # Create namespaced output edges (update source_id only)
        namespaced_output_edges = [
            output_edge.model_copy(
                update={"source_id": f"{namespace}/{output_edge.source_id}"}
            )
            for output_edge in self.output_edges
        ]

        return Workflow(
            nodes=namespaced_nodes,
            edges=namespaced_edges,
            input_edges=namespaced_input_edges,
            output_edges=namespaced_output_edges,
        )


class WorkflowValue(Value[Workflow]):
    pass


__all__ = [
    "Workflow",
    "WorkflowValue",
]
