"""Runtime graph composition for Attempt retries; see docs/plans/attempt-retries.md."""

from collections.abc import Mapping, Sequence
from typing import ClassVar

from overrides import override
from pydantic import BaseModel, Field

from ..core import (
    Data,
    Edge,
    ErrorClass,
    ExecutionContext,
    IntegerValue,
    Node,
    NodeException,
    NodeTypeInfo,
    Result,
    StringValue,
    ValidatedWorkflow,
    ValidationContext,
    Workflow,
    WorkflowValue,
)
from ..core.values import (
    build_data_type,
    get_data_dict,
    get_data_fields,
    get_field_annotations,
)
from .attempt import AttemptNode, AttemptParams


async def validate_unmetered_workflow(
    owner: Node, workflow: ValidatedWorkflow, context: ValidationContext
) -> None:
    """Inspect functional params recursively, never hints or host annotations."""
    visited: set[int] = set()

    async def visit(value: object) -> None:
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, WorkflowValue):
            await visit(await value.root.validate(context))
        elif isinstance(value, Workflow):
            for node in value.nodes:
                if node.TYPE_INFO.metered:
                    raise NodeException.for_builder(
                        f"Attempt node '{owner.id}' retries metered node '{node.id}'; "
                        "set allow_metered to true to authorize additional charges.",
                        node=owner,
                        error_class=ErrorClass.VALIDATION,
                    )
                await visit(node.params)
        elif isinstance(value, BaseModel):
            for name in type(value).model_fields:
                await visit(getattr(value, name))
            if value.model_extra:
                await visit(value.model_extra)
        elif isinstance(value, Mapping):
            for item in value.values():
                await visit(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                await visit(item)

    await visit(workflow)


class AttemptRetryParams(AttemptParams):
    boundary_id: StringValue = Field(
        title="Boundary ID", description="The original attempt node's identifier."
    )
    completed_attempt: IntegerValue = Field(
        title="Completed Attempt",
        description="The zero-based number of the attempt whose result is being checked.",
    )


class AttemptRetryNode(Node[Data, Data, AttemptRetryParams]):
    """Continue a failed attempt using ordinary graph expansion and typed edges."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Attempt Retry",
        description="Continues a failed attempt within its declared retry budget.",
        version="1.0.0",
        parameter_type=AttemptRetryParams,
    )

    def policy(self) -> AttemptParams:
        return AttemptParams(
            workflow=self.params.workflow,
            retries=self.params.retries,
            retry_on=self.params.retry_on,
            allow_metered=self.params.allow_metered,
        )

    def boundary(self, context: ValidationContext) -> AttemptNode:
        return context.node_registry.create_node(
            AttemptNode,
            id=self.params.boundary_id.root,
            params=self.policy(),
        )

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> type[Data]:
        boundary = self.boundary(context)
        original_input = await boundary.input_type(context)
        output = await boundary.output_type(context)
        fields = {
            f"input_{key}": field
            for key, field in get_data_fields(original_input).items()
        }
        fields.update(get_data_fields(output))
        return build_data_type(name="AttemptRetryInput", fields=fields)

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> type[Data]:
        boundary = self.boundary(context)
        return await boundary.output_type(context)

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[Data],
        output_type: type[Data],
        input: Data,
    ) -> Data | Workflow:
        result = get_data_dict(input)["result"]
        assert isinstance(result, Result)
        next_attempt = self.params.completed_attempt.root + 1
        if (
            result.is_ok()
            or next_attempt > self.params.retries.root
            or result.unwrap_err().error_class.root
            not in {value.root for value in self.params.retry_on.root}
        ):
            return output_type(**{"result": result})

        await context.on_boundary_retry(
            node=self,
            boundary_id=self.params.boundary_id.root,
            error=result.unwrap_err(),
            attempt=next_attempt,
            max_retries=self.params.retries.root,
            allow_metered=self.params.allow_metered.root,
        )
        return await build_retry_workflow(
            context=context.validation_context,
            params=self.policy(),
            boundary_id=self.params.boundary_id.root,
            attempt=next_attempt,
            input_type=input_type,
            output_type=output_type,
            forwarded=True,
        )


async def build_retry_workflow(
    *,
    context: ValidationContext,
    params: AttemptParams,
    boundary_id: str,
    attempt: int,
    input_type: type[Data],
    output_type: type[Data],
    forwarded: bool,
) -> Workflow:
    """Run one single-shot child, then pass its result to a continuation."""
    registry = context.node_registry
    input_node = registry.create_input_node(**get_field_annotations(input_type))
    output_node = registry.create_output_node(**get_field_annotations(output_type))
    child = registry.create_node(
        AttemptNode,
        id=f"try_{attempt}",
        params=params.model_update(retries=IntegerValue(0)),
    )
    continuation = registry.create_node(
        AttemptRetryNode,
        id="next",
        params=AttemptRetryParams(
            workflow=params.workflow,
            retries=params.retries,
            retry_on=params.retry_on,
            allow_metered=params.allow_metered,
            boundary_id=StringValue(boundary_id),
            completed_attempt=IntegerValue(attempt),
        ),
    )
    child_input = await child.input_type(context)
    edges: list[Edge] = []
    for key in child_input.model_fields:
        source_key = f"input_{key}" if forwarded else key
        edges.extend(
            [
                Edge.from_nodes(
                    source=input_node,
                    source_key=source_key,
                    target=child,
                    target_key=key,
                ),
                Edge.from_nodes(
                    source=input_node,
                    source_key=source_key,
                    target=continuation,
                    target_key=f"input_{key}",
                ),
            ]
        )
    edges.extend(
        [
            Edge.from_nodes(
                source=child,
                source_key="result",
                target=continuation,
                target_key="result",
            ),
            Edge.from_nodes(
                source=continuation,
                source_key="result",
                target=output_node,
                target_key="result",
            ),
        ]
    )
    return Workflow(
        input_node=input_node,
        inner_nodes=[child, continuation],
        output_node=output_node,
        edges=edges,
    )
