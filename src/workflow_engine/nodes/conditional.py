# workflow_engine/nodes/conditional.py
"""
Conditional nodes that run different workflows depending on a condition input.
"""

from typing import ClassVar, Type

from overrides import override
from pydantic import ConfigDict, Field

from ..core import (
    BooleanValue,
    Data,
    Empty,
    ExecutionContext,
    Node,
    NodeTypeInfo,
    Params,
    ValidationContext,
    Workflow,
    WorkflowValue,
)
from ..core.values import build_data_type, compare_fields, get_data_fields
from ..utils.mappings import mapping_intersection


class IfParams(Params):
    if_true: WorkflowValue = Field(
        title="If True", description="The workflow to run when the condition is true."
    )


class IfElseParams(Params):
    if_true: WorkflowValue = Field(
        title="If True", description="The workflow to run when the condition is true."
    )
    if_false: WorkflowValue = Field(
        title="If False", description="The workflow to run when the condition is false."
    )


class ConditionalInput(Data):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    condition: BooleanValue = Field(
        title="Condition", description="The condition to evaluate."
    )


class IfNode(Node[ConditionalInput, Empty, IfParams]):
    """
    A node that optionally executes the internal workflow if the boolean
    condition is true.

    The output of this node is always empty, since there would be no valid
    output if the condition is false.
    """

    # TODO: allow conditional nodes with optional output

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="If",
        description="Executes the internal workflow if the boolean condition is true.",
        version="0.4.0",
        parameter_type=IfParams,
    )

    @override
    async def dynamic_input_type(
        self, context: ValidationContext
    ) -> Type[ConditionalInput]:
        workflow_if_true = await self.params.if_true.root.validate(context)
        fields = dict(get_data_fields(ConditionalInput))
        for key, field in get_data_fields(workflow_if_true.input_type).items():
            assert key not in fields
            fields[key] = field
        return build_data_type(
            name="IfInput",
            fields=fields,
            base_cls=ConditionalInput,
        )

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> Type[Empty]:
        return Empty

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[ConditionalInput],
        output_type: Type[Empty],
        input: ConditionalInput,
    ) -> Empty | Workflow:
        return self.params.if_true.root if input.condition else Empty()


class IfElseNode(Node[ConditionalInput, Data, IfElseParams]):
    """
    A node that executes one of the two internal workflows based on the boolean
    condition.

    The output of this node is the intersection of the if_true and if_false
    workflows.
    """

    # TODO: allow union types

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="If Or Else",
        description="Executes one of the two internal workflows based on the boolean condition.",
        version="0.4.0",
        parameter_type=IfElseParams,
    )

    @override
    async def dynamic_input_type(
        self, context: ValidationContext
    ) -> Type[ConditionalInput]:
        fields = dict(get_data_fields(ConditionalInput))
        workflow_if_true = await self.params.if_true.root.validate(context)
        for key, field in get_data_fields(workflow_if_true.input_type).items():
            assert key not in fields
            fields[key] = field
        return build_data_type(
            name="IfElseInput",
            fields=fields,
            base_cls=ConditionalInput,
        )

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> Type[Data]:
        workflow_if_true = await self.params.if_true.root.validate(context)
        workflow_if_false = await self.params.if_false.root.validate(context)
        fields = mapping_intersection(
            get_data_fields(workflow_if_true.output_type),
            get_data_fields(workflow_if_false.output_type),
            compare_fn=compare_fields,
        )
        return build_data_type(
            name="IfElseOutput",
            fields=fields,
            base_cls=Data,
        )

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[ConditionalInput],
        output_type: Type[Data],
        input: ConditionalInput,
    ) -> Workflow:
        return (
            self.params.if_true.root if input.condition else self.params.if_false.root
        )


__all__ = [
    "ConditionalInput",
    "IfElseNode",
    "IfNode",
]
