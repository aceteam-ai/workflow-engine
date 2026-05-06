# workflow_engine/nodes/iteration.py
"""
Nodes that iterate over a sequence of items.
"""

from typing import ClassVar, Type

from overrides import override
from pydantic import Field, PrivateAttr

from ..core import (
    DataValue,
    Edge,
    Empty,
    ExecutionContext,
    InputNode,
    IntegerValue,
    Node,
    NodeTypeInfo,
    OutputNode,
    Params,
    SequenceValue,
    ValidatedWorkflow,
    ValidationContext,
    Value,
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
    SequenceParams,
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
        display_name="For Each",
        description="Executes the internal workflow for each item in the input sequence.",
        version="1.0.0",
        parameter_type=ForEachParams,
    )

    _workflow: ValidatedWorkflow | None = PrivateAttr(default=None)

    async def workflow(self, context: ValidationContext) -> ValidatedWorkflow:
        if self._workflow is None:
            self._workflow = await self.params.workflow.root.validate(context=context)
        return self._workflow

    def _input_field_count(self, workflow: ValidatedWorkflow) -> int:
        return len(get_field_annotations(workflow.input_type))

    def _output_field_count(self, workflow: ValidatedWorkflow) -> int:
        return len(get_field_annotations(workflow.output_type))

    def _has_single_input(self, workflow: ValidatedWorkflow) -> bool:
        return self._input_field_count(workflow) == 1

    def _has_single_output(self, workflow: ValidatedWorkflow) -> bool:
        return self._output_field_count(workflow) == 1

    def _has_no_output(self, workflow: ValidatedWorkflow) -> bool:
        return self._output_field_count(workflow) == 0

    def _input_item_type(self, workflow: ValidatedWorkflow) -> Type[Value]:
        """Single input field type; only valid when _has_single_input()."""
        return get_only_field(workflow.input_type)[1]

    def _output_item_type(self, workflow: ValidatedWorkflow) -> Type[Value]:
        """Single output field type; only valid when _has_single_output()."""
        return get_only_field(workflow.output_type)[1]

    def _input_element_type(self, workflow: ValidatedWorkflow) -> Type[Value]:
        if self._has_single_input(workflow):
            return self._input_item_type(workflow)
        return DataValue[workflow.input_type]

    def _output_element_type(self, workflow: ValidatedWorkflow) -> Type[Value]:
        """Only valid when _has_no_output() is False."""
        assert not self._has_no_output(workflow)
        if self._has_single_output(workflow):
            return self._output_item_type(workflow)
        return DataValue[workflow.output_type]

    @override
    async def dynamic_input_type(
        self,
        context: ValidationContext,
    ) -> Type[SequenceData]:
        workflow = await self.workflow(context)
        return SequenceData[self._input_element_type(workflow)]

    @override
    async def dynamic_output_type(
        self,
        context: ValidationContext,
    ) -> Type[SequenceData | Empty]:
        workflow = await self.workflow(context)
        if self._has_no_output(workflow):
            return Empty
        return SequenceData[self._output_element_type(workflow)]

    def _build_input_output_nodes(
        self,
        context: ValidationContext,
        workflow: ValidatedWorkflow,
    ) -> tuple[InputNode, OutputNode]:
        """
        Uses the Node registry to build the input and output nodes for the ForEachNode.

        If "Input" and "Output" have been overridden, then we will use the
        overridden node classes.
        """
        node_registry = context.node_registry
        input_seq_type = SequenceValue[self._input_element_type(workflow)]
        input_node = node_registry.create_input_node(sequence=input_seq_type)
        if self._has_no_output(workflow):
            output_node = node_registry.create_output_node()
        else:
            output_seq_type = SequenceValue[self._output_element_type(workflow)]
            output_node = node_registry.create_output_node(sequence=output_seq_type)
        return input_node, output_node

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[SequenceData],
        output_type: Type[SequenceData | Empty],
        input: SequenceData,
    ) -> Workflow | SequenceData | Empty:
        n = len(input.sequence)
        if n == 0:
            if issubclass(output_type, Empty):
                return output_type()
            else:
                return output_type.empty()

        workflow = await self.workflow(context.validation_context)
        input_node, output_node = self._build_input_output_nodes(
            context.validation_context,
            workflow,
        )
        has_no_output = self._has_no_output(workflow)
        expand_element_type = self._input_element_type(workflow)
        expand = context.validation_context.node_registry.create_node(
            ExpandSequenceNode,
            id="expand",
            params=SequenceParams(length=IntegerValue(root=n)),
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
            gather = context.validation_context.node_registry.create_node(
                GatherSequenceNode,
                id="gather",
                params=SequenceParams(length=IntegerValue(root=n)),
                element_type=self._output_element_type(workflow),
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
            item_workflow = workflow.with_namespace(namespace)

            input_adapter = (
                context.validation_context.node_registry.create_node(
                    ExpandDataNode,
                    id="input_adapter",
                    data_type=workflow.input_type,
                ).with_namespace(namespace)
                if not self._has_single_input(workflow)
                else None
            )
            output_adapter = (
                context.validation_context.node_registry.create_node(
                    GatherDataNode,
                    id="output_adapter",
                    data_type=workflow.output_type,
                ).with_namespace(namespace)
                if self._has_single_output(workflow) is False and has_no_output is False
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


__all__ = [
    "ForEachNode",
]
