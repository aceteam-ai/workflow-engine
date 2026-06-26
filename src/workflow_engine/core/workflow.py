# workflow_engine/core/workflow.py
from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence, Set
from functools import cached_property
from typing import TYPE_CHECKING, Awaitable, Self, Type

import networkx as nx
from overrides import override
from pydantic import ConfigDict, Field, ValidationError, model_validator

from ..utils.asynchronous import gather
from ..utils.model import ImmutableBaseModel
from .edge import Edge
from .error import NodeException, NodeExpansionException, WorkflowException
from .io import InputNode, OutputNode
from .node import Node, get_id_with_namespace
from .values import (
    Data,
    DataMapping,
    Value,
    get_value_at_path,
    resolve_path,
)

if TYPE_CHECKING:
    from .context import ExecutionContext, ValidationContext


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
    def successors_by_source(self) -> Mapping[str, Set[str]]:
        """A mapping from each node to the set of nodes that depend on it."""
        successors: dict[str, set[str]] = {node.id: set() for node in self.nodes}
        for edge in self.edges:
            successors[edge.source_id].add(edge.target_id)
        return successors

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

    def with_namespace(self, namespace: str) -> Self:
        """
        Create a copy of this workflow with all node IDs namespaced.

        Args:
            namespace: The namespace to prefix all node IDs with

        Returns:
            A new Workflow with all node IDs prefixed with '{namespace}/'
        """
        # Create namespaced nodes
        return self.model_update(
            input_node=self.input_node.with_namespace(namespace),
            inner_nodes=[node.with_namespace(namespace) for node in self.inner_nodes],
            output_node=self.output_node.with_namespace(namespace),
            edges=[edge.with_namespace(namespace) for edge in self.edges],
        )

    # NOTE: this clobbers a long-deprecated method of the same name by Pydantic but we don't care
    async def validate(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        context: "ValidationContext",
    ) -> ValidatedWorkflow:
        """
        Convert an untyped workflow to a typed workflow.

        Walks the workflow graph and:
        1. Looks up concrete node types in node_registry
        2. Applies migrations via the registry's ``load`` method
        3. Validates node inputs and edge types
        4. Returns a new Workflow with typed nodes

        Args:
            workflow: Untyped workflow (nodes may be base Node instances)

        Returns:
            Typed workflow (nodes are concrete subclass instances)

        Raises:
            ValueError: If edges reference non-existent fields or nodes are missing required inputs
            TypeError: If edge types are incompatible
        """
        typed_input_node = context.node_registry.load(self.input_node)
        if not isinstance(typed_input_node, InputNode):
            raise ValueError(f"Node {typed_input_node.id} is not an InputNode")
        typed_inner_nodes = await gather(
            asyncio.to_thread(context.node_registry.load, node)
            for node in self.inner_nodes
        )
        typed_output_node = context.node_registry.load(self.output_node)
        if not isinstance(typed_output_node, OutputNode):
            raise ValueError(f"Node {typed_output_node.id} is not an OutputNode")
        typed_nodes = (typed_input_node, *typed_inner_nodes, typed_output_node)

        async def get_input_output_types(
            node: Node,
        ) -> tuple[str, tuple[type[Data], type[Data]]]:
            # NOTE: it would be faster to do this in parallel but later on we
            # are going to have to do this in series anyway due to type variable
            # resolution, so we might as well do it in series here too.
            input_type = await node.input_type(context)
            output_type = await node.output_type(context)
            return node.id, (input_type, output_type)

        node_input_output_types = dict(
            await gather(get_input_output_types(node) for node in typed_nodes)
        )
        node_input_types = {
            node_id: input_type
            for node_id, (input_type, _) in node_input_output_types.items()
        }
        node_output_types = {
            node_id: output_type
            for node_id, (_, output_type) in node_input_output_types.items()
        }

        for edge in self.edges:
            if edge.source_id not in node_output_types:
                raise ValueError(
                    f"Edge {edge.source_id} -> {edge.target_id} has a source that is not a node"
                )
            if edge.target_id not in node_input_types:
                raise ValueError(
                    f"Edge {edge.source_id} -> {edge.target_id} has a target that is not a node"
                )
            edge.validate_types(
                source_type=node_output_types[edge.source_id],
                target_type=node_input_types[edge.target_id],
            )

        return ValidatedWorkflow(
            input_node=typed_input_node,
            inner_nodes=typed_inner_nodes,
            output_node=typed_output_node,
            edges=self.edges,
            node_input_types=node_input_types,
            node_output_types=node_output_types,
        )


class ValidatedWorkflow(Workflow):
    node_input_types: Mapping[str, type[Data]] = Field(exclude=True)
    node_output_types: Mapping[str, type[Data]] = Field(exclude=True)

    @override
    def with_namespace(self, namespace: str) -> Self:
        return (
            super()
            .with_namespace(namespace)
            .model_update(
                node_input_types={
                    get_id_with_namespace(node_id, namespace): node_input_type
                    for node_id, node_input_type in self.node_input_types.items()
                },
                node_output_types={
                    get_id_with_namespace(node_id, namespace): node_output_type
                    for node_id, node_output_type in self.node_output_types.items()
                },
            )
        )

    @cached_property
    def input_type(self) -> Type[Data]:
        return self.node_input_types[self.input_node.id]

    @cached_property
    def output_type(self) -> Type[Data]:
        return self.node_output_types[self.output_node.id]

    def get_node_input_if_ready(
        self,
        node_id: str,
        node_outputs: Mapping[str, DataMapping],
    ) -> DataMapping | None:
        """
        Build a node's input from completed upstream outputs, or None if not ready.
        """
        node_edges = self.edges_by_target.get(node_id)
        if node_edges is None or len(node_edges) == 0:
            return {}
        node_input: DataMapping = {}
        for target_key, edge in node_edges.items():
            source_output = node_outputs.get(edge.source_id)
            if source_output is None:
                return None
            node_input[target_key] = get_value_at_path(
                data=source_output,
                path=edge.source_key_path,
            )
        return node_input

    def get_initial_ready_nodes(self, input: DataMapping) -> Mapping[str, DataMapping]:
        """
        Return all nodes ready to execute at workflow start, including the input node.
        """
        return {
            **self.get_ready_nodes(node_outputs={}),
            self.input_node.id: input,
        }

    def get_ready_successors(
        self,
        completed_node_ids: Iterable[str],
        node_outputs: Mapping[str, DataMapping],
        *,
        skip: Set[str] = frozenset(),
    ) -> Mapping[str, DataMapping]:
        """
        Return newly ready successor nodes after the given nodes complete.

        Only checks direct successors of completed_node_ids. Nodes in skip are
        ignored, as are nodes already present in the returned map.
        """
        ready_nodes: dict[str, DataMapping] = {}
        for completed_id in completed_node_ids:
            for succ_id in self.successors_by_source.get(completed_id, ()):
                if succ_id in node_outputs or succ_id in skip or succ_id in ready_nodes:
                    continue
                succ_input = self.get_node_input_if_ready(succ_id, node_outputs)
                if succ_input is not None:
                    ready_nodes[succ_id] = succ_input
        return ready_nodes

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
                ready_nodes.pop(node.id, None)
                continue
            # skip the node if it is already in the ready set
            if node.id in ready_nodes:
                continue

            node_input = self.get_node_input_if_ready(node.id, node_outputs)
            if node_input is None:
                continue

            try:
                ready_nodes[node.id] = node_input
            except ValidationError as e:
                raise NodeException.for_user(
                    f"Input {node_input} for node {node.id} is invalid: {e}",
                    node=node,
                ) from e
        return ready_nodes

    async def get_output(
        self,
        *,
        context: "ExecutionContext",
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
            WorkflowException: If a required output is missing or cannot be cast
        """
        # Validate and cast
        output: DataMapping = {}
        cast_keys: list[str] = []
        cast_tasks: list[Awaitable[Value]] = []

        for edge in self.edges:
            if edge.target_id != self.output_node.id:
                continue
            if edge.source_id not in node_outputs:
                if partial:
                    continue
                raise WorkflowException.for_user(
                    f"Cannot get output from node {edge.source_id}.",
                )
            node_output = node_outputs[edge.source_id]
            try:
                output_field = get_value_at_path(
                    data=node_output,
                    path=edge.source_key_path,
                )
            except KeyError as e:
                if partial:
                    continue
                raise WorkflowException.for_user(
                    f"Cannot get output from node {edge.source_id} at path {edge.source_key_path_string}.",
                ) from e

            expected_type = resolve_path(
                data_type=self.output_type,
                path=[edge.target_key],
            )

            # Fast path: value already matches expected type
            if isinstance(output_field, expected_type):
                output[edge.target_key] = output_field
                continue

            # Validate that the output can be cast to the expected type
            if not output_field.can_cast_to(expected_type):
                raise WorkflowException.for_builder(
                    f"Output '{edge.target_key}' from {edge.source_id}.{edge.source_key_path_string} "
                    f"cannot be cast: {output_field} is not assignable to {expected_type}",
                )

            cast_keys.append(edge.target_key)
            cast_tasks.append(output_field.cast_to(expected_type, context=context))

        casted_values = await gather(cast_tasks)

        # Build the result dictionary
        for key, value in zip(cast_keys, casted_values):
            assert key not in output
            output[key] = value

        return output

    def expand_node(
        self,
        node_id: str,
        subgraph: "ValidatedWorkflow",
    ) -> Self:
        """
        Replace a node in this workflow with a subgraph.

        This method performs graph surgery to replace a node with a subgraph.
        The subgraph's nodes are namespaced with the original node's ID to prevent
        ID collisions. Input and output edges are reconnected appropriately.

        Perhaps foolishly, we do not re-validate that the new workflow's input
        and output types are compatible with those of the node being replaced.

        Args:
            node_id: ID of the node to replace
            subgraph: The workflow to insert in place of the node
            node_input: The input that was passed to the original node

        Returns:
            A new Workflow with the node replaced by the subgraph

        Raises:
            WorkflowException: If the node_id doesn't exist
            NodeExpansionException: If the expansion fails for any other reason
        """
        if node_id not in self.nodes_by_id:
            raise WorkflowException.for_engineer(
                f"Node '{node_id}' is a target for expansion, but is not found in the workflow.",
            )
        node = self.nodes_by_id[node_id]

        try:
            subgraph = subgraph.with_namespace(namespace=node_id)

            # Collect all edges that need to be modified
            new_inner_nodes: list[Node] = [
                node for node in self.inner_nodes if node.id != node_id
            ] + list(subgraph.nodes)
            new_edges: list[Edge] = list(subgraph.edges)

            for edge in self.edges:
                if edge.target_id == node_id:
                    # Only create the edge if the subgraph's input_node has this field
                    if edge.target_key in subgraph.input_type.model_fields:
                        new_edges.append(
                            Edge(
                                source_id=edge.source_id,
                                source_key=edge.source_key,
                                target_id=subgraph.input_node.id,
                                target_key=edge.target_key,
                            )
                        )
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

            # the new node types contain all nodes except the one being expanded,
            # plus all nodes from the namespaced subgraph
            new_node_input_types = dict(self.node_input_types)
            del new_node_input_types[node_id]
            new_node_input_types.update(subgraph.node_input_types)

            new_node_output_types = dict(self.node_output_types)
            del new_node_output_types[node_id]
            new_node_output_types.update(subgraph.node_output_types)

            return self.model_update(
                inner_nodes=new_inner_nodes,
                edges=new_edges,
                node_input_types=new_node_input_types,
                node_output_types=new_node_output_types,
            )
        except Exception as e:
            raise NodeExpansionException(node=node, workflow=subgraph) from e


class WorkflowValue(Value[Workflow]):
    pass


__all__ = [
    "Workflow",
    "WorkflowValue",
]
