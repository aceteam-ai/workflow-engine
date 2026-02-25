# workflow_engine/nodes/iteration.py
"""
Nodes that iterate over a sequence of items.
"""

from functools import cached_property
from typing import ClassVar, Literal, Self

from overrides import override
from pydantic import Field

from workflow_engine.core.io import SchemaParams

from ..core import (
    Context,
    DataValue,
    Edge,
    Empty,
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
    workflow: WorkflowValue = Field(
        title="Workflow", description="The workflow to run for each item."
    )


class ForEachNode(Node[SequenceData, SequenceData | Empty, ForEachParams]):
    """
    A node that executes the internal workflow W for each item in the input
    sequence.

    For each item i in the input sequence, create a copy of W, call it W[i].
    The implementation automatically chooses the most efficient path:

    - **Single input field**: Sequence of scalar values, direct wiring from
      expand to each workflow (no ExpandDataNode).
    - **Multiple input fields**: Sequence of Data objects, ExpandDataNode
      adapter per item.
    - **Single output field**: Direct wiring from each workflow to gather
      (no GatherDataNode).
    - **Multiple output fields**: GatherDataNode adapter per item.
    - **Zero output fields**: Outputs nothing (Empty); inner workflows run for
      side effects only.
    """

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="ForEach",
        display_name="ForEach",
        description="Executes the internal workflow for each item in the input sequence.",
        version="1.0.0",
        parameter_type=ForEachParams,
    )

    type: Literal["ForEach"] = "ForEach"  # pyright: ignore[reportIncompatibleVariableOverride]

    @cached_property
    def workflow(self) -> Workflow:
        return self.params.workflow.root

    @cached_property
    def _input_field_count(self) -> int:
        return len(self.workflow.input_type.field_annotations())

    @cached_property
    def _output_field_count(self) -> int:
        return len(self.workflow.output_type.field_annotations())

    def _has_single_input(self) -> bool:
        return self._input_field_count == 1

    def _has_single_output(self) -> bool:
        return self._output_field_count == 1

    def _has_no_output(self) -> bool:
        return self._output_field_count == 0

    @cached_property
    def _input_item_type(self):
        """Single input field type; only valid when _has_single_input()."""
        return self.workflow.input_type.only_field()[1]

    @cached_property
    def _output_item_type(self):
        """Single output field type; only valid when _has_single_output()."""
        return self.workflow.output_type.only_field()[1]

    def _input_element_type(self):
        if self._has_single_input():
            return self._input_item_type
        return DataValue[self.workflow.input_type]

    def _output_element_type(self):
        """Only valid when _has_no_output() is False."""
        assert not self._has_no_output()
        if self._has_single_output():
            return self._output_item_type
        return DataValue[self.workflow.output_type]

    @cached_property
    def input_type(self):
        return SequenceData[self._input_element_type()]

    @cached_property
    def output_type(self):
        if self._has_no_output():
            return Empty
        return SequenceData[self._output_element_type()]

    def _build_input_output_nodes(self, n: int) -> tuple[InputNode, OutputNode]:
        input_seq_type = SequenceValue[self._input_element_type()]
        input_params = SchemaParams.from_fields(sequence=input_seq_type)
        input_node = InputNode(id="input", params=input_params)
        if self._has_no_output():
            output_node = OutputNode.empty()
        else:
            output_seq_type = SequenceValue[self._output_element_type()]
            output_params = SchemaParams.from_fields(sequence=output_seq_type)
            output_node = OutputNode(id="output", params=output_params)
        return input_node, output_node

    @override
    async def run(self, context: Context, input: SequenceData) -> Workflow:
        n = len(input.sequence)
        input_node, output_node = self._build_input_output_nodes(n)
        has_no_output = self._has_no_output()

        expand_element_type = self._input_element_type()
        expand = ExpandSequenceNode.from_length(
            id="expand",
            length=n,
            element_type=expand_element_type,
        )

        inner_nodes: list[Node] = [expand]
        edges: list[Edge] = [
            Edge.from_nodes(
                source=input_node,
                source_key="sequence",
                target=expand,
                target_key="sequence",
            ),
        ]

        gather: GatherSequenceNode | None = None
        if not has_no_output:
            gather = GatherSequenceNode.from_length(
                id="gather",
                length=n,
                element_type=self._output_element_type(),
            )
            inner_nodes.append(gather)
            edges.append(
                Edge.from_nodes(
                    source=gather,
                    source_key="sequence",
                    target=output_node,
                    target_key="sequence",
                )
            )

        for i in range(n):
            namespace = f"element_{i}"
            item_workflow = self.workflow.with_namespace(namespace)

            input_adapter = (
                ExpandDataNode.from_data_type(
                    id="input_adapter",
                    data_type=self.workflow.input_type,
                ).with_namespace(namespace)
                if not self._has_single_input()
                else None
            )
            output_adapter = (
                GatherDataNode.from_data_type(
                    id="output_adapter",
                    data_type=self.workflow.output_type,
                ).with_namespace(namespace)
                if self._has_single_output() is False and has_no_output is False
                else None
            )

            if input_adapter is not None:
                inner_nodes.append(input_adapter)
            inner_nodes.extend(item_workflow.inner_nodes)
            if output_adapter is not None:
                inner_nodes.append(output_adapter)

            if input_adapter is not None:
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
                    source_id = (
                        input_adapter.id if input_adapter is not None else expand.id
                    )
                    source_key = (
                        edge.source_key if input_adapter is not None else expand.key(i)
                    )
                else:
                    source_id = edge.source_id
                    source_key = edge.source_key
                if edge.target_id == item_workflow.output_node.id:
                    if has_no_output:
                        continue
                    assert gather is not None
                    target_id = (
                        output_adapter.id if output_adapter is not None else gather.id
                    )
                    target_key = (
                        edge.target_key if output_adapter is not None else gather.key(i)
                    )
                else:
                    target_id = edge.target_id
                    target_key = edge.target_key
                edges.append(
                    Edge(
                        source_id=source_id,
                        source_key=source_key,
                        target_id=target_id,
                        target_key=target_key,
                    )
                )

            if output_adapter is not None:
                assert gather is not None
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
    def from_workflow(cls, id: str, workflow: Workflow) -> Self:
        return cls(id=id, params=ForEachParams(workflow=WorkflowValue(workflow)))


__all__ = [
    "ForEachNode",
]
