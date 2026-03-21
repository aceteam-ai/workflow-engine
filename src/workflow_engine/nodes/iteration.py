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
from ..core.values.data import get_field_annotations, get_only_field
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
        return len(get_field_annotations(self.workflow.input_type))

    @cached_property
    def _output_field_count(self) -> int:
        return len(get_field_annotations(self.workflow.output_type))

    def _has_single_input(self) -> bool:
        return self._input_field_count == 1

    def _has_single_output(self) -> bool:
        return self._output_field_count == 1

    def _has_no_output(self) -> bool:
        return self._output_field_count == 0

    @cached_property
    def _input_item_type(self):
        """Single input field type; only valid when _has_single_input()."""
        return get_only_field(self.workflow.input_type)[1]

    @cached_property
    def _output_item_type(self):
        """Single output field type; only valid when _has_single_output()."""
        return get_only_field(self.workflow.output_type)[1]

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
        has_single_input = self._has_single_input()
        has_single_output = self._has_single_output()

        expand_element_type = self._input_element_type()
        expand = ExpandSequenceNode.from_length(
            id="expand",
            length=n,
            element_type=expand_element_type,
        )

        inner_nodes: list[Node] = [expand]
        edges: list[Edge] = [
            Edge(
                source_id=input_node.id,
                source_key="sequence",
                target_id=expand.id,
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
                Edge(
                    source_id=gather.id,
                    source_key="sequence",
                    target_id=output_node.id,
                    target_key="sequence",
                )
            )

        # Pre-compute adapter templates (created once, namespaced per iteration)
        needs_input_adapter = not has_single_input
        needs_output_adapter = not has_single_output and not has_no_output
        input_adapter_template = (
            ExpandDataNode.from_data_type(
                id="input_adapter",
                data_type=self.workflow.input_type,
            )
            if needs_input_adapter
            else None
        )
        output_adapter_template = (
            GatherDataNode.from_data_type(
                id="output_adapter",
                data_type=self.workflow.output_type,
            )
            if needs_output_adapter
            else None
        )

        # Cache expand/gather IDs
        expand_id = expand.id
        gather_id = gather.id if gather is not None else None

        # Cache the base workflow's input/output node IDs (before namespacing)
        base_input_id = self.workflow.input_node.id
        base_output_id = self.workflow.output_node.id
        base_edges = self.workflow.edges
        base_inner_nodes = self.workflow.inner_nodes

        for i in range(n):
            namespace = f"element_{i}"
            prefix = f"{namespace}/"

            # Namespace inner nodes directly
            namespaced_inner = [node.with_namespace(namespace) for node in base_inner_nodes]
            inner_nodes.extend(namespaced_inner)

            # Namespace the input/output node IDs
            ns_input_id = f"{prefix}{base_input_id}"
            ns_output_id = f"{prefix}{base_output_id}"

            input_adapter = (
                input_adapter_template.with_namespace(namespace)
                if input_adapter_template is not None
                else None
            )
            output_adapter = (
                output_adapter_template.with_namespace(namespace)
                if output_adapter_template is not None
                else None
            )

            if input_adapter is not None:
                inner_nodes.append(input_adapter)
                edges.append(
                    Edge(
                        source_id=expand_id,
                        source_key=expand.key(i),
                        target_id=input_adapter.id,
                        target_key="data",
                    )
                )
            if output_adapter is not None:
                inner_nodes.append(output_adapter)

            # Rewire edges from the base workflow
            input_adapter_id = input_adapter.id if input_adapter is not None else None
            output_adapter_id = output_adapter.id if output_adapter is not None else None

            for edge in base_edges:
                # Determine source
                if edge.source_id == base_input_id:
                    source_id = input_adapter_id if input_adapter_id is not None else expand_id
                    source_key = edge.source_key if input_adapter_id is not None else expand.key(i)
                else:
                    source_id = f"{prefix}{edge.source_id}"
                    source_key = edge.source_key

                # Determine target
                if edge.target_id == base_output_id:
                    if has_no_output:
                        continue
                    target_id = output_adapter_id if output_adapter_id is not None else gather_id
                    target_key = edge.target_key if output_adapter_id is not None else gather.key(i)
                else:
                    target_id = f"{prefix}{edge.target_id}"
                    target_key = edge.target_key

                edges.append(
                    Edge.model_construct(
                        source_id=source_id,
                        source_key=source_key,
                        target_id=target_id,
                        target_key=target_key,
                    )
                )

            if output_adapter is not None:
                assert gather is not None
                edges.append(
                    Edge(
                        source_id=output_adapter.id,
                        source_key="data",
                        target_id=gather_id,
                        target_key=gather.key(i),
                    )
                )

        return Workflow._construct_trusted(
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
