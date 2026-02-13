# workflow_engine/nodes/iteration.py
"""
Nodes that iterate over a sequence of items.
"""

from functools import cached_property
from typing import ClassVar, Literal, Self

from overrides import override

from workflow_engine.core.io import SchemaParams

from ..core import (
    Context,
    DataValue,
    Edge,
    InputNode,
    Node,
    NodeTypeInfo,
    OutputNode,
    Params,
    SequenceValue,
    Workflow,
    WorkflowValue,
)
from .data import (
    ExpandDataNode,
    ExpandSequenceNode,
    GatherDataNode,
    GatherSequenceNode,
    SequenceData,
)


class ForEachParams(Params):
    workflow: WorkflowValue


class ForEachNode(Node[SequenceData, SequenceData, ForEachParams]):
    """
    A node that executes the internal workflow W for each item in the input
    sequence.

    For each item i in the input sequence, create a copy of W, call it W[i].
    We expand the sequence into its individual data objects and expand
    sequence[i] into the input fields of W[i].
    Then, we gather the output of each W[i] into a single object, before
    gathering them further into a single sequence.

    The output of this node is a sequence of the same length as the input
    sequence, with each item being the output of the internal workflow.
    """

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="ForEach",
        display_name="ForEach",
        description="Executes the internal workflow for each item in the input sequence.",
        version="0.4.0",
        parameter_type=ForEachParams,
    )

    type: Literal["ForEach"] = "ForEach"  # pyright: ignore[reportIncompatibleVariableOverride]

    @cached_property
    def workflow(self) -> Workflow:
        return self.params.workflow.root

    @cached_property
    def input_type(self):
        return SequenceData[DataValue[self.workflow.input_type]]

    @cached_property
    def output_type(self):
        return SequenceData[DataValue[self.workflow.output_type]]

    @override
    async def run(self, context: Context, input: SequenceData) -> Workflow:
        N = len(input.sequence)

        input_params = SchemaParams.from_fields(
            sequence=SequenceValue[DataValue[self.workflow.input_type]],
        )
        output_params = SchemaParams.from_fields(
            sequence=SequenceValue[DataValue[self.workflow.output_type]],
        )

        input_node = InputNode(id="input", params=input_params)
        output_node = OutputNode(id="output", params=output_params)

        inner_nodes: list[Node] = []
        edges: list[Edge] = []

        expand = ExpandSequenceNode.from_length(
            id="expand",
            length=N,
            element_type=DataValue[self.workflow.input_type],
        )
        gather = GatherSequenceNode.from_length(
            id="gather",
            length=N,
            element_type=DataValue[self.workflow.output_type],
        )
        inner_nodes.append(expand)
        inner_nodes.append(gather)

        # Connect input_node -> expand and gather -> output_node
        edges.append(
            Edge.from_nodes(
                source=input_node,
                source_key="sequence",
                target=expand,
                target_key="sequence",
            )
        )
        edges.append(
            Edge.from_nodes(
                source=gather,
                source_key="sequence",
                target=output_node,
                target_key="sequence",
            )
        )

        for i in range(N):
            namespace = f"element_{i}"
            input_adapter = ExpandDataNode.from_data_type(
                id="input_adapter",
                data_type=self.workflow.input_type,
            ).with_namespace(namespace)
            item_workflow = self.workflow.with_namespace(namespace)
            output_adapter = GatherDataNode.from_data_type(
                id="output_adapter",
                data_type=self.workflow.output_type,
            ).with_namespace(namespace)

            inner_nodes.append(input_adapter)
            inner_nodes.extend(item_workflow.inner_nodes)
            inner_nodes.append(output_adapter)

            edges.append(
                Edge.from_nodes(
                    source=expand,
                    source_key=expand.key(i),
                    target=input_adapter,
                    target_key="data",
                )
            )
            for edge in item_workflow.edges:
                if edge.source_id == item_workflow.input_node.id:
                    source_id = input_adapter.id
                else:
                    source_id = edge.source_id
                if edge.target_id == item_workflow.output_node.id:
                    target_id = output_adapter.id
                else:
                    target_id = edge.target_id
                edges.append(
                    Edge(
                        source_id=source_id,
                        source_key=edge.source_key,
                        target_id=target_id,
                        target_key=edge.target_key,
                    )
                )
            edges.append(
                Edge.from_nodes(
                    source=output_adapter,
                    source_key="data",
                    target=gather,
                    target_key=gather.key(i),
                )
            )

        return Workflow(
            input_node=input_node,
            inner_nodes=inner_nodes,
            output_node=output_node,
            edges=edges,
        )

    @classmethod
    def from_workflow(
        cls,
        id: str,
        workflow: Workflow,
    ) -> Self:
        return cls(id=id, params=ForEachParams(workflow=WorkflowValue(workflow)))


__all__ = [
    "ForEachNode",
]
