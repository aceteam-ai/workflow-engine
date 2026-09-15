"""Bounded sequence generation through flat, checkpointed step expansion."""

from functools import cached_property
from typing import ClassVar

from overrides import override
from pydantic import Field, PrivateAttr

from ..core import (
    BooleanValue,
    Data,
    Edge,
    ErrorClass,
    ExecutionContext,
    IntegerValue,
    Node,
    NodeException,
    NodeTypeInfo,
    Params,
    SequenceValue,
    ValidatedWorkflow,
    ValidationContext,
    Value,
    Workflow,
    WorkflowValue,
)
from ..core.values import get_data_dict
from ..core.values.data import get_field_annotations
from ..core.values.value import get_origin_and_args
from .data import SequenceData
from .sequence import ElementParams, _data


class UnfoldParams(Params):
    workflow: WorkflowValue = Field(
        title="Workflow",
        description="The step workflow that produces items, a next seed, and a completion flag.",
    )
    max_iterations: IntegerValue = Field(
        title="Maximum Iterations",
        description="The maximum number of step executions allowed.",
        json_schema_extra={"minimum": 1},
    )
    truncate: BooleanValue = Field(
        default=BooleanValue(False),
        title="Truncate",
        description="The choice to return generated items when the iteration limit is reached before completion.",
    )


class _UnfoldBase(Node[Data, Data, UnfoldParams]):
    _workflow: ValidatedWorkflow | None = PrivateAttr(default=None)

    async def signature(
        self, context: ValidationContext
    ) -> tuple[ValidatedWorkflow, type[Value], type[Value]]:
        if self.params.max_iterations.root < 1:
            raise ValueError("Unfold max_iterations must be positive.")
        if self._workflow is None:
            self._workflow = await self.params.workflow.root.validate(context=context)
        workflow = self._workflow
        inputs = get_field_annotations(workflow.input_type)
        outputs = get_field_annotations(workflow.output_type)
        if set(inputs) != {"seed"} or set(outputs) != {"items", "next", "done"}:
            raise ValueError(
                "Unfold step must take only 'seed' and output exactly 'items', 'next', and 'done'."
            )
        seed_type = inputs["seed"]
        if seed_type.to_value_schema() != outputs["next"].to_value_schema():
            raise ValueError("Unfold seed and next schemas must match.")
        if not issubclass(outputs["done"], BooleanValue):
            raise ValueError(
                "Unfold done must be a BooleanValue; resolve Result explicitly."
            )
        if not issubclass(outputs["items"], SequenceValue):
            raise ValueError("Unfold items must be a SequenceValue.")
        _, (element_type,) = get_origin_and_args(outputs["items"])
        return workflow, seed_type, element_type

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> type[Data]:
        _, _, element_type = await self.signature(context)
        return SequenceData[element_type]

    async def expand_step(
        self,
        *,
        context: ValidationContext,
        input_type: type[Data],
        output_type: type[Data],
        previous_items: bool,
    ) -> Workflow:
        workflow, _, element_type = await self.signature(context)
        registry = context.node_registry
        source = registry.create_input_node(**get_field_annotations(input_type))
        output = registry.create_output_node(**get_field_annotations(output_type))
        step = workflow.with_namespace("step")
        next_params = (
            self.params.model_update(
                max_iterations=IntegerValue(self.params.max_iterations.root - 1)
            )
            if previous_items
            else self.params
        )
        successor = registry.create_node(UnfoldNextNode, id="next", params=next_params)
        nodes: list[Node] = [*step.nodes, successor]
        edges = [
            *step.edges,
            Edge.from_nodes(
                source=source,
                source_key="next" if previous_items else "seed",
                target=step.input_node,
                target_key="seed",
            ),
            *[
                Edge.from_nodes(
                    source=step.output_node,
                    source_key=key,
                    target=successor,
                    target_key=key,
                )
                for key in ("items", "next", "done")
            ],
        ]
        if previous_items:
            join = registry.create_node(
                UnfoldJoinNode,
                id="join",
                params={"element_schema": element_type.to_value_schema()},
            )
            nodes.append(join)
            edges.extend(
                [
                    Edge.from_nodes(
                        source=source,
                        source_key="items",
                        target=join,
                        target_key="first",
                    ),
                    Edge.from_nodes(
                        source=successor,
                        source_key="sequence",
                        target=join,
                        target_key="second",
                    ),
                    Edge.from_nodes(
                        source=join,
                        source_key="sequence",
                        target=output,
                        target_key="sequence",
                    ),
                ]
            )
        else:
            edges.append(
                Edge.from_nodes(
                    source=successor,
                    source_key="sequence",
                    target=output,
                    target_key="sequence",
                )
            )
        return Workflow(
            input_node=source, output_node=output, inner_nodes=nodes, edges=edges
        )


class UnfoldNode(_UnfoldBase):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Unfold",
        description="Generates a sequence from a seed through a bounded step workflow.",
        version="1.0.0",
        parameter_type=UnfoldParams,
    )

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> type[Data]:
        _, seed_type, _ = await self.signature(context)
        return _data("UnfoldInput", seed=seed_type)

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[Data],
        output_type: type[Data],
        input: Data,
    ) -> Workflow:
        return await self.expand_step(
            context=context.validation_context,
            input_type=input_type,
            output_type=output_type,
            previous_items=False,
        )


class UnfoldNextNode(_UnfoldBase):
    """Checkpointed control node between pages of an Unfold expansion."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Continue Unfold",
        description="Continues sequence generation after a step, subject to its remaining budget.",
        version="1.0.0",
        parameter_type=UnfoldParams,
    )

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> type[Data]:
        _, seed_type, element_type = await self.signature(context)
        return _data(
            "UnfoldNextInput",
            items=SequenceValue[element_type],
            next=seed_type,
            done=BooleanValue,
        )

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
        if values["done"].root:
            return output_type.model_validate({"sequence": values["items"]})
        if self.params.max_iterations.root == 1:
            if self.params.truncate.root:
                return output_type.model_validate({"sequence": values["items"]})
            raise NodeException.for_user(
                "Unfold reached its maximum iteration count before the step reported completion.",
                node=self,
                error_class=ErrorClass.VALIDATION,
            )
        return await self.expand_step(
            context=context.validation_context,
            input_type=input_type,
            output_type=output_type,
            previous_items=True,
        )


class UnfoldJoinNode(Node[Data, Data, ElementParams]):
    """Pure concatenation stage: page items stay on ordinary checkpointed edges."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Collect Unfold Items",
        description="Combines a page's items with the items generated by later pages.",
        version="1.0.0",
        parameter_type=ElementParams,
    )

    @cached_property
    def element_type(self) -> type[Value]:
        return self.params.element_schema.root.to_value_cls()

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> type[Data]:
        return _data(
            "UnfoldJoinInput",
            first=SequenceValue[self.element_type],
            second=SequenceValue[self.element_type],
        )

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> type[Data]:
        return SequenceData[self.element_type]

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[Data],
        output_type: type[Data],
        input: Data,
    ) -> Data:
        values = get_data_dict(input)
        return output_type.model_validate(
            {"sequence": [*values["first"].root, *values["second"].root]}
        )


__all__ = ["UnfoldJoinNode", "UnfoldNextNode", "UnfoldNode", "UnfoldParams"]
