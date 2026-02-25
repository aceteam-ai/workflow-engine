# workflow_engine/nodes/conditional.py
"""
Conditional nodes that run different workflows depending on a condition input.
"""

from functools import cached_property
from typing import ClassVar, Literal, Self

from overrides import override
from pydantic import ConfigDict, Field

from ..core.values import build_data_type, get_data_fields

from ..core import (
    BooleanValue,
    Context,
    Data,
    Empty,
    Node,
    NodeTypeInfo,
    Params,
    Workflow,
    WorkflowValue,
)
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
        name="If",
        display_name="If",
        description="Executes the internal workflow if the boolean condition is true.",
        version="0.4.0",
        parameter_type=IfParams,
    )

    type: Literal["If"] = "If"  # pyright: ignore[reportIncompatibleVariableOverride]

    @cached_property
    def input_type(self):
        fields = dict(get_data_fields(ConditionalInput))
        for key, field in self.params.if_true.root.input_fields.items():
            assert key not in fields
            fields[key] = field
        return build_data_type("IfInput", fields, base_cls=ConditionalInput)

    @cached_property
    def output_type(self):
        return Empty

    @override
    async def run(self, context: Context, input: ConditionalInput) -> Empty | Workflow:
        if input.condition:
            return self.params.if_true.root
        return Empty()

    @classmethod
    def from_workflow(
        cls,
        id: str,
        if_true: Workflow,
    ) -> Self:
        return cls(
            id=id,
            params=IfParams(if_true=WorkflowValue(if_true)),
        )


class IfElseNode(Node[ConditionalInput, Data, IfElseParams]):
    """
    A node that executes one of the two internal workflows based on the boolean
    condition.

    The output of this node is the intersection of the if_true and if_false
    workflows.
    """

    # TODO: allow union types

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="IfElse",
        display_name="IfElse",
        description="Executes one of the two internal workflows based on the boolean condition.",
        version="0.4.0",
        parameter_type=IfElseParams,
    )
    type: Literal["IfElse"] = "IfElse"  # pyright: ignore[reportIncompatibleVariableOverride]

    @cached_property
    def input_type(self):
        fields = dict(get_data_fields(ConditionalInput))
        for key, field in self.params.if_true.root.input_fields.items():
            assert key not in fields
            fields[key] = field
        return build_data_type("IfElseInput", fields, base_cls=ConditionalInput)

    @cached_property
    def output_type(self):
        fields = mapping_intersection(
            self.params.if_true.root.output_fields,
            self.params.if_false.root.output_fields,
        )
        return build_data_type("IfElseOutput", fields)

    @override
    async def run(self, context: Context, input: ConditionalInput) -> Workflow:
        return (
            self.params.if_true.root if input.condition else self.params.if_false.root
        )

    @classmethod
    def from_workflows(
        cls,
        id: str,
        if_true: Workflow,
        if_false: Workflow,
    ) -> Self:
        return cls(
            id=id,
            params=IfElseParams(
                if_true=WorkflowValue(if_true),
                if_false=WorkflowValue(if_false),
            ),
        )


__all__ = [
    "ConditionalInput",
    "IfNode",
    "IfElseNode",
]
