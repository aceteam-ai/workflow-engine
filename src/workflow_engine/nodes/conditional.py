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
    ErrorClass,
    ErrorClassValue,
    ExecutionContext,
    Node,
    NodeException,
    NodeTypeInfo,
    Params,
    StringMapValue,
    ValidationContext,
    Workflow,
    WorkflowValue,
)
from ..core.values import build_data_type, compare_fields, get_data_fields
from ..utils.asynchronous import gather
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


################################################################################
# match_error_class


def _validate_branches_exhaustive(
    *,
    node_id: str,
    branches: StringMapValue[WorkflowValue],
) -> None:
    """
    Raises a ``ValueError`` naming *node_id* and every ``ErrorClass`` value
    *branches* fails to cover, plus any key in *branches* that is not a
    current ``ErrorClass`` value.

    Reads ``ErrorClass`` from this module's own namespace rather than a
    fresh import, so a test can point it at a wider stand-in enum (a
    hypothetical "what if a seventh value existed") via monkeypatch and
    observe this check react to the substitute, not to a value baked in at
    import time. That is the property that keeps this check from silently
    weakening the day a real seventh value is added: the set of required
    branches is read from ``ErrorClass`` itself, never hardcoded here.
    """
    valid_values = [error_class.value for error_class in ErrorClass]
    valid_value_set = set(valid_values)
    branch_keys = set(branches.keys())
    missing = [value for value in valid_values if value not in branch_keys]
    unknown = sorted(key for key in branch_keys if key not in valid_value_set)
    if not missing and not unknown:
        return

    sentences: list[str] = []
    if missing:
        sentences.append(
            f"Node '{node_id}' (MatchErrorClass) is missing a branch for "
            f"error_class value(s): {', '.join(missing)}. Every ErrorClass "
            f"value needs its own branch: {', '.join(valid_values)}. Add a "
            "branch for each missing value to this node's 'branches'."
        )
    if unknown:
        sentences.append(
            f"Node '{node_id}' (MatchErrorClass) has a branch for "
            f"value(s) ErrorClass does not define: {', '.join(unknown)}. "
            f"Valid error_class values are: {', '.join(valid_values)}. "
            "Remove or rename the unrecognized branch(es)."
        )
    raise ValueError(" ".join(sentences))


class MatchErrorClassParams(Params):
    branches: StringMapValue[WorkflowValue] = Field(
        title="Branches",
        description=(
            "The sub-workflow to run for each error_class value, keyed by "
            "the ErrorClass string (e.g. 'timeout'). Must have exactly one "
            "entry per value ErrorClass currently defines: graph validation "
            "rejects a node whose branches do not cover every value."
        ),
    )


class MatchErrorClassInput(Data):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    error_class: ErrorClassValue = Field(
        title="Error Class", description="The error class to branch on."
    )


class MatchErrorClassNode(Node[MatchErrorClassInput, Data, MatchErrorClassParams]):
    """
    Branches on ``error_class``: runs the sub-workflow wired to whichever
    ``ErrorClass`` value the input carries.

    ``error_class`` is a closed, engine-owned vocabulary (``core/error.py``),
    which is what makes an exhaustiveness check legitimate for this node
    specifically, and not for a conditional keyed on a per-node error type:
    graph validation rejects a ``MatchErrorClass`` node whose ``branches``
    do not cover every value ``ErrorClass`` currently defines, naming the
    node and the missing value(s). See docs/nodes.md ("MatchErrorClass") for
    why that is deliberate rather than a defect, and what it costs the next
    person who adds a seventh ``ErrorClass`` value.
    """

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Match Error Class",
        description=(
            "Runs a different sub-workflow depending on which ErrorClass "
            "value the input carries. Requires a branch for every value "
            "ErrorClass currently defines."
        ),
        version="1.0.0",
        parameter_type=MatchErrorClassParams,
    )

    @override
    async def dynamic_input_type(
        self, context: ValidationContext
    ) -> Type[MatchErrorClassInput]:
        _validate_branches_exhaustive(node_id=self.id, branches=self.params.branches)
        branch_workflows = await gather(
            branch.root.validate(context) for branch in self.params.branches.values()
        )
        fields = dict(get_data_fields(MatchErrorClassInput))
        common_fields = mapping_intersection(
            *(get_data_fields(workflow.input_type) for workflow in branch_workflows),
            compare_fn=compare_fields,
        )
        for key, field in common_fields.items():
            assert key not in fields
            fields[key] = field
        return build_data_type(
            name="MatchErrorClassInput",
            fields=fields,
            base_cls=MatchErrorClassInput,
        )

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> Type[Data]:
        _validate_branches_exhaustive(node_id=self.id, branches=self.params.branches)
        branch_workflows = await gather(
            branch.root.validate(context) for branch in self.params.branches.values()
        )
        fields = mapping_intersection(
            *(get_data_fields(workflow.output_type) for workflow in branch_workflows),
            compare_fn=compare_fields,
        )
        return build_data_type(
            name="MatchErrorClassOutput",
            fields=fields,
            base_cls=Data,
        )

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[MatchErrorClassInput],
        output_type: Type[Data],
        input: MatchErrorClassInput,
    ) -> Workflow:
        key = input.error_class.root.value
        branch = self.params.branches.get(key)
        if branch is None:
            # Exhaustiveness is enforced at graph validation (see
            # dynamic_input_type/dynamic_output_type above), and
            # engine.execute() always re-validates before running, so this
            # is unreachable through that path. It stays defensive against a
            # host that calls run() directly against an already-validated
            # workflow, e.g. after ErrorClass has since gained a value this
            # node's branches were validated without.
            raise NodeException.for_engineer(
                f"No branch for error_class '{key}' on node '{self.id}'; "
                "graph validation should have rejected this workflow.",
                node=self,
                error_class=ErrorClass.SYSTEMIC,
            )
        return branch.root


__all__ = [
    "ConditionalInput",
    "IfElseNode",
    "IfNode",
    "MatchErrorClassInput",
    "MatchErrorClassNode",
    "MatchErrorClassParams",
]
