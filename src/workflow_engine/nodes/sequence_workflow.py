"""Typed sequence algebra with flat, replayable workflow expansion.

ForEach is traverse. Fold chains steps; Filter and GroupBy traverse a
predicate/key workflow and combine its outputs using pure nodes. No operator
interprets Result tags or introduces an error-handling policy.
"""

from typing import ClassVar

from overrides import override
from pydantic import Field

from ..core import (
    BooleanValue,
    Data,
    Edge,
    ExecutionContext,
    IntegerValue,
    Node,
    NodeTypeInfo,
    Params,
    SequenceValue,
    StringValue,
    ValidatedWorkflow,
    ValidationContext,
    Value,
    Workflow,
    WorkflowValue,
)
from ..core.values import build_data_type, get_data_dict
from ..core.values.data import get_data_fields, get_field_annotations, get_only_field
from .data import (
    ExpandDataNode,
    ExpandSequenceNode,
    SequenceParams,
    single_field_or_wrapped,
)
from .iteration import ForEachNode
from .sequence import GroupSequenceNode, SelectSequenceNode, _data


class WorkflowSequenceParams(Params):
    workflow: WorkflowValue = Field(
        title="Workflow", description="The workflow to run for each item."
    )


class _WorkflowSequenceNode(Node[Data, Data, WorkflowSequenceParams]):
    async def workflow(self, context: ValidationContext) -> ValidatedWorkflow:
        return await self.params.workflow.root.validate(context=context)


class _ClassifyNode(_WorkflowSequenceNode):
    decision_type: ClassVar[type[Value]]
    combine_type: ClassVar[type[SelectSequenceNode] | type[GroupSequenceNode]]
    decision_key: ClassVar[str]

    async def mapper(self, context: ValidationContext) -> ForEachNode:
        workflow = await self.workflow(context)
        fields = get_field_annotations(workflow.output_type)
        if len(fields) != 1 or not issubclass(
            next(iter(fields.values())), self.decision_type
        ):
            raise ValueError(
                f"{self.type} workflow must output exactly one {self.decision_type.__name__} field; resolve Result explicitly before this output."
            )
        return context.node_registry.create_node(
            ForEachNode, id="traverse", params={"workflow": self.params.workflow}
        )

    async def combiner(
        self, context: ValidationContext
    ) -> SelectSequenceNode | GroupSequenceNode:
        workflow = await self.workflow(context)
        item = single_field_or_wrapped(workflow.input_type)
        return context.node_registry.create_node(
            self.combine_type,
            id="combine",
            params={"element_schema": item.to_value_schema()},
        )

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> type[Data]:
        return await (await self.mapper(context)).input_type(context)

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> type[Data]:
        await self.mapper(context)
        return await (await self.combiner(context)).output_type(context)

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[Data],
        output_type: type[Data],
        input: Data,
    ) -> Workflow:
        validation = context.validation_context
        registry = validation.node_registry
        mapper, combine = await self.mapper(validation), await self.combiner(validation)
        source = registry.create_input_node(**get_field_annotations(input_type))
        target = registry.create_output_node(**get_field_annotations(output_type))
        output_key, _ = get_only_field(output_type)
        return Workflow(
            input_node=source,
            output_node=target,
            inner_nodes=[mapper, combine],
            edges=[
                Edge.from_nodes(
                    source=source,
                    source_key="sequence",
                    target=mapper,
                    target_key="sequence",
                ),
                Edge.from_nodes(
                    source=source,
                    source_key="sequence",
                    target=combine,
                    target_key="sequence",
                ),
                Edge.from_nodes(
                    source=mapper,
                    source_key="sequence",
                    target=combine,
                    target_key=self.decision_key,
                ),
                Edge.from_nodes(
                    source=combine,
                    source_key=output_key,
                    target=target,
                    target_key=output_key,
                ),
            ],
        )


class FilterNode(_ClassifyNode):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Filter",
        description="Keeps items whose predicate workflow returns true.",
        version="1.0.0",
        parameter_type=WorkflowSequenceParams,
    )
    decision_type = BooleanValue
    combine_type = SelectSequenceNode
    decision_key = "decisions"


class GroupByNode(_ClassifyNode):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Group By",
        description="Groups items by a key workflow while preserving their order.",
        version="1.0.0",
        parameter_type=WorkflowSequenceParams,
    )
    decision_type = StringValue
    combine_type = GroupSequenceNode
    decision_key = "keys"


class FoldNode(_WorkflowSequenceNode):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Fold",
        description="Accumulates items in order through checkpointed workflow steps.",
        version="1.0.0",
        parameter_type=WorkflowSequenceParams,
    )

    async def signature(
        self, context: ValidationContext
    ) -> tuple[ValidatedWorkflow, type[Value], type[Data]]:
        workflow = await self.workflow(context)
        inputs, outputs = (
            get_field_annotations(workflow.input_type),
            get_field_annotations(workflow.output_type),
        )
        if "acc" not in inputs or len(inputs) < 2 or set(outputs) != {"acc"}:
            raise ValueError(
                "Fold step must take 'acc' and at least one item field, and output only 'acc'."
            )
        acc = inputs["acc"]
        if acc.to_value_schema() != outputs["acc"].to_value_schema():
            raise ValueError("Fold input and output accumulator schemas must match.")
        item = build_data_type(
            name="FoldItem",
            fields={
                key: field
                for key, field in get_data_fields(workflow.input_type).items()
                if key != "acc"
            },
        )
        return workflow, acc, item

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> type[Data]:
        _, acc, item = await self.signature(context)
        return _data(
            "FoldInput", seed=acc, sequence=SequenceValue[single_field_or_wrapped(item)]
        )

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> type[Data]:
        _, acc, _ = await self.signature(context)
        return _data("FoldOutput", acc=acc)

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[Data],
        output_type: type[Data],
        input: Data,
    ) -> Workflow | Data:
        values = get_data_dict(input)
        items = values["sequence"].root
        if not items:
            return output_type.model_validate({"acc": values["seed"]})
        validation = context.validation_context
        registry = validation.node_registry
        _, acc_type, item_type = await self.signature(validation)
        source = registry.create_input_node(**get_field_annotations(input_type))
        target = registry.create_output_node(acc=acc_type)
        expand = registry.create_node(
            ExpandSequenceNode,
            id="expand",
            params=SequenceParams(length=IntegerValue(len(items))),
            element_type=single_field_or_wrapped(item_type),
        )
        nodes: list[Node] = [expand]
        edges = [
            Edge.from_nodes(
                source=source,
                source_key="sequence",
                target=expand,
                target_key="sequence",
            )
        ]
        previous_id, previous_key = source.id, "seed"
        single_item = len(get_field_annotations(item_type)) == 1
        for i in range(len(items)):
            # Expansion itself depends on acc, so independent nodes inside a
            # later step cannot dispatch before its predecessor produces acc.
            step = registry.create_node(
                FoldStepNode, id=f"step_{i}", params=self.params
            )
            nodes.append(step)
            edges.append(
                Edge(
                    source_id=previous_id,
                    source_key=previous_key,
                    target_id=step.id,
                    target_key="acc",
                )
            )
            if single_item:
                item_key, _ = get_only_field(item_type)
                edges.append(
                    Edge.from_nodes(
                        source=expand,
                        source_key=expand.key(i),
                        target=step,
                        target_key=item_key,
                    )
                )
            else:
                adapter = registry.create_node(
                    ExpandDataNode, id=f"item_{i}", data_type=item_type
                )
                nodes.append(adapter)
                edges.append(
                    Edge.from_nodes(
                        source=expand,
                        source_key=expand.key(i),
                        target=adapter,
                        target_key="data",
                    )
                )
                edges.extend(
                    Edge.from_nodes(
                        source=adapter, source_key=key, target=step, target_key=key
                    )
                    for key in get_field_annotations(item_type)
                )
            previous_id, previous_key = step.id, "acc"
        edges.append(
            Edge(
                source_id=previous_id,
                source_key=previous_key,
                target_id=target.id,
                target_key="acc",
            )
        )
        return Workflow(
            input_node=source, output_node=target, inner_nodes=nodes, edges=edges
        )


class FoldStepNode(_WorkflowSequenceNode):
    """Flat call adapter used to gate discovery of each fold step on acc."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Fold Step",
        description="Runs one fold step after its accumulator is available.",
        version="1.0.0",
        parameter_type=WorkflowSequenceParams,
    )

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> type[Data]:
        return (await self.workflow(context)).input_type

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> type[Data]:
        return (await self.workflow(context)).output_type

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[Data],
        output_type: type[Data],
        input: Data,
    ) -> Workflow:
        return self.params.workflow.root


__all__ = ["FilterNode", "FoldNode", "FoldStepNode", "GroupByNode"]
