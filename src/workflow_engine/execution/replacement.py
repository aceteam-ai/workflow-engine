"""Shared, iterative replacement completion for both flat schedulers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Self

from overrides import override
from pydantic import Field, model_validator

from ..core.boundary import ErrorBoundaryNode
from ..core.context import ExecutionContext
from ..core.error import NodeException, NodeReplacementException, WorkflowException
from ..core.node import Node
from ..core.replacement import (
    NodeReplacement,
    ReplacementFrame,
    ReplacementRetry,
    adapt_data,
    fingerprint,
)
from ..core.values import Data, DataMapping, get_data_fields
from ..core.values.data import get_data_schema
from ..core.workflow import ValidatedWorkflow
from .boundary import BoundaryTracker, flush_cancellations, handle_failure, materialize
from .retry import NodeRetryState, RetryTracker


class ReplacementGraph(ValidatedWorkflow):
    """Run-local graph: declared logical slots may coexist with their children."""

    replacement_slots: frozenset[str] = Field(default=frozenset(), exclude=True)
    seeded_inputs: Mapping[str, DataMapping] = Field(default_factory=dict, exclude=True)

    @model_validator(mode="after")
    def _validate_no_id_prefix_collisions(self):
        ids = sorted(node.id for node in self.nodes)
        for index, current in enumerate(ids[:-1]):
            if (
                ids[index + 1].startswith(current + "/")
                and current not in self.replacement_slots
            ):
                raise ValueError(
                    f"Node ID collision detected: '{current}' is a prefix of '{ids[index + 1]}'."
                )
        return self

    @override
    def get_node_input_if_ready(
        self, node_id: str, node_outputs: Mapping[str, DataMapping]
    ) -> DataMapping | None:
        if node_id in self.replacement_slots:
            return None
        if node_id in self.seeded_inputs:
            return self.seeded_inputs[node_id]
        return super().get_node_input_if_ready(node_id, node_outputs)

    @override
    def expand_node(self, node_id: str, subgraph: ValidatedWorkflow) -> Self:
        expanded = super().expand_node(node_id, subgraph)
        if node_id in self.seeded_inputs:
            seeds = dict(expanded.seeded_inputs)
            seeds[f"{node_id}/{subgraph.input_node.id}"] = seeds.pop(node_id)
            expanded = expanded.model_update(seeded_inputs=seeds)
        return expanded


@dataclass(frozen=True)
class PendingReplacement:
    frame: ReplacementFrame
    node: Node
    input: DataMapping
    input_type: type[Data]
    output_type: type[Data]
    child_output_type: type[Data]


def check_contract(
    source: type[Data], target: type[Data], *, node: Node, direction: str
) -> None:
    source_fields = get_data_fields(source)
    for name, (value_type, field) in get_data_fields(target).items():
        if name not in source_fields:
            if field.is_required():
                raise NodeReplacementException(
                    f"Replacement {direction} field '{name}' is required by {target.__name__} but absent from {source.__name__}.",
                    node=node,
                )
        elif not source_fields[name][0].can_cast_to(value_type):
            raise NodeReplacementException(
                f"Replacement {direction} field '{name}' cannot cast {source_fields[name][0].__name__} to {value_type.__name__}.",
                node=node,
            )


async def recover(
    node: Node,
    context: ExecutionContext,
    input_type: type[Data],
    output_type: type[Data],
    input: DataMapping,
    exception: Exception,
) -> DataMapping:
    """One error hook at the adapter/delegator; never attribute a child's error here."""
    if not isinstance(exception, WorkflowException):
        wrapped = NodeException.for_operator(
            f"Unhandled replacement lifecycle error: {exception}", node=node
        )
        wrapped.__cause__ = exception
        exception = wrapped
    if exception.node_id is None:
        exception.node_id = node.id
    elif exception.node_id != node.id:
        raise NodeException.for_operator(
            "Replacement lifecycle exception has a mismatched node ID.", node=node
        ) from exception
    result = await context.on_node_error(
        node=node,
        input_type=input_type,
        output_type=output_type,
        input=input,
        exception=exception,
    )
    if isinstance(result, Mapping):
        return await adapt_data(result, output_type, node=node, context=context)
    if isinstance(result, WorkflowException):
        raise result
    raise NodeException.for_operator(
        "Context returned an invalid replacement error override.", node=node
    ) from exception


class ReplacementTracker:
    def __init__(self, max_hops: int) -> None:
        if max_hops < 1:
            raise ValueError("max_replacement_hops must be positive")
        self.max_hops = max_hops
        self.frames: dict[str, PendingReplacement] = {}
        self.waiting: dict[str, str] = {}
        self.occupied: set[str] = set()

    async def install(
        self,
        workflow: ReplacementGraph,
        node: Node,
        request: NodeReplacement,
        context: ExecutionContext,
        retries: RetryTracker,
        boundaries: BoundaryTracker,
    ) -> tuple[ReplacementGraph, dict[str, DataMapping], DataMapping | None]:
        input_type = workflow.node_input_types[node.id]
        output_type = workflow.node_output_types[node.id]
        try:
            parent_id = self.waiting.get(node.id)
            parent = self.frames[parent_id].frame if parent_id is not None else None
            logical = parent.logical_node_id if parent else node.id
            hop = parent.hop + 1 if parent else 0
            if hop >= self.max_hops:
                raise NodeReplacementException(
                    f"Replacement exceeds max_replacement_hops={self.max_hops}.",
                    node=node,
                )
            child_id = f"{logical}/replacement_{hop}"
            self.occupied.update(workflow.nodes_by_id)
            if any(
                existing == child_id or existing.startswith(child_id + "/")
                for existing in self.occupied | self.frames.keys()
            ):
                raise NodeReplacementException(
                    f"Replacement ID '{child_id}' is already occupied.", node=node
                )
            child = context.validation_context.node_registry.load(request.replacement)
            child = child.model_copy(update={"id": child_id})
            for boundary in boundaries.enclosing(node.id):
                assert isinstance(boundary.node, ErrorBoundaryNode)
                await boundary.node.validate_replacement(
                    delegator=node,
                    replacement=child,
                    context=context.validation_context,
                )
            child_input_type = await child.input_type(context.validation_context)
            child_output_type = await child.output_type(context.validation_context)
            check_contract(input_type, child_input_type, node=node, direction="input")
            check_contract(
                child_output_type, output_type, node=node, direction="output"
            )
            child_input = await adapt_data(
                request.input, child_input_type, node=node, context=context
            )
            schemas = [
                get_data_schema(cls)
                for cls in (
                    input_type,
                    output_type,
                    child_input_type,
                    child_output_type,
                )
            ]
            frame = ReplacementFrame(
                logical_node_id=logical,
                delegator_id=node.id,
                replacement_id=child_id,
                original_label=request.replacement.id,
                hop=hop,
                max_replacement_hops=self.max_hops,
                delegator=node,
                replacement=child,
                input_schema=schemas[0],
                output_schema=schemas[1],
                replacement_input_schema=schemas[2],
                replacement_output_schema=schemas[3],
                input_fingerprint=fingerprint(
                    {k: v.model_dump(mode="json") for k, v in request.input.items()}
                ),
                contract_fingerprint=fingerprint(
                    [
                        node.without_hints().model_dump(mode="json"),
                        child.without_hints().model_dump(mode="json"),
                        *[s.model_dump(mode="json") for s in schemas],
                    ]
                ),
            )
            saved = await context.get_node_replacement_frame(node_id=node.id)
            if saved is not None:
                if saved.status == "failed":
                    raise NodeReplacementException(
                        "Cannot resume a terminal failed replacement frame.", node=node
                    )
                for key in (
                    "logical_node_id",
                    "delegator_id",
                    "replacement_id",
                    "hop",
                    "max_replacement_hops",
                    "input_fingerprint",
                    "contract_fingerprint",
                ):
                    if getattr(saved, key) != getattr(frame, key):
                        raise NodeReplacementException(
                            f"Replacement checkpoint mismatch: {key}.", node=node
                        )
                frame = frame.model_copy(
                    update={
                        "original_label": saved.original_label,
                        "completed_slots": saved.completed_slots,
                        "retries": saved.retries,
                    }
                )
                for retry_id, state in saved.retries.items():
                    current = retries.get_state(retry_id)
                    if state.attempt >= current.attempt:
                        retries.states[retry_id] = NodeRetryState(
                            retry_id,
                            attempt=state.attempt,
                            next_retry_at=state.next_retry_at,
                        )
            # Validate graph installation before making its event durable.
            updated = workflow.model_update(
                inner_nodes=[*workflow.inner_nodes, child],
                node_input_types={
                    **workflow.node_input_types,
                    child_id: child_input_type,
                },
                node_output_types={
                    **workflow.node_output_types,
                    child_id: child_output_type,
                },
                replacement_slots=workflow.replacement_slots | {node.id},
                seeded_inputs={**workflow.seeded_inputs, child_id: child_input},
            )
            await context.on_node_replace(
                node=node,
                replacement=child,
                input=request.input,
                replacement_info=frame,
            )
            self.frames[node.id] = PendingReplacement(
                frame, node, request.input, input_type, output_type, child_output_type
            )
            self.waiting[child_id] = node.id
            self.occupied.add(child_id)
            return updated, {child_id: child_input}, None
        except Exception as exc:
            if not isinstance(exc, WorkflowException):
                wrapped = NodeReplacementException(
                    f"Invalid replacement: {exc}", node=node
                )
                wrapped.__cause__ = exc
                exc = wrapped
            output = await recover(
                node, context, input_type, output_type, request.input, exc
            )
            return workflow, {}, output

    def expanded(self, node_id: str, subgraph: ValidatedWorkflow) -> None:
        if node_id in self.waiting:
            self.waiting[f"{node_id}/{subgraph.output_node.id}"] = self.waiting.pop(
                node_id
            )

    async def complete(
        self,
        completed: Iterable[str],
        outputs: MutableMapping[str, DataMapping],
        context: ExecutionContext,
        boundaries: BoundaryTracker,
    ) -> tuple[set[str], list[WorkflowException]]:
        """Unwind each completion iteratively; intermediate logical slots stay visible."""
        completed = set(completed)
        failures = []
        queue = list(completed)
        while queue:
            child_id = queue.pop()
            parent_id = self.waiting.get(child_id)
            if parent_id is None:
                continue
            pending = self.frames[parent_id]
            if pending.frame.status != "pending" or boundaries.is_blocked(parent_id):
                continue
            try:
                child_output = await adapt_data(
                    outputs[child_id],
                    pending.child_output_type,
                    node=pending.node,
                    context=context,
                )
                output = await adapt_data(
                    child_output,
                    pending.output_type,
                    node=pending.node,
                    context=context,
                )
                output = await context.on_node_finish(
                    node=pending.node,
                    input_type=pending.input_type,
                    output_type=pending.output_type,
                    input=pending.input,
                    output=output,
                )
                output = await adapt_data(
                    output, pending.output_type, node=pending.node, context=context
                )
            except Exception as exc:
                try:
                    output = await recover(
                        pending.node,
                        context,
                        pending.input_type,
                        pending.output_type,
                        pending.input,
                        exc,
                    )
                except WorkflowException as failure:
                    failures.append(failure)
                    continue
            frame = pending.frame.model_copy(
                update={
                    "status": "completed",
                    "output_json": pending.output_type.model_validate(
                        output
                    ).model_dump_json(),
                    "completed_slots": (*pending.frame.completed_slots, parent_id),
                }
            )
            await self._checkpoint(parent_id, frame, context)
            outputs[parent_id] = output
            del self.waiting[child_id]
            queue.append(parent_id)
            completed.add(parent_id)
        return completed, failures

    async def failed(
        self,
        exception: WorkflowException,
        boundaries: BoundaryTracker,
        context: ExecutionContext,
    ) -> None:
        affected = {exception.node_id}
        # Chain IDs are siblings, so walk explicit relations instead of guessing
        # their ancestry from prefixes. Descendants of an expanded child still
        # use ordinary flat namespace membership.
        for node_id, pending in sorted(
            self.frames.items(), key=lambda item: item[1].frame.hop, reverse=True
        ):
            frame = pending.frame
            if frame.status != "pending":
                continue
            child_failed = any(
                failed_id == frame.replacement_id
                or (failed_id or "").startswith(frame.replacement_id + "/")
                for failed_id in affected
            )
            boundary = boundaries.innermost(exception.node_id or "")
            if boundaries.is_blocked(node_id) or (
                (child_failed or node_id == exception.node_id) and boundary is None
            ):
                failed = frame.model_copy(
                    update={"status": "failed", "failure_node_id": exception.node_id}
                )
                await self._checkpoint(node_id, failed, context)
                affected.add(node_id)
                if node_id != exception.node_id:
                    await context.on_node_replacement_failed(
                        node=pending.node, replacement_info=failed, exception=exception
                    )
                owner = boundaries.innermost(node_id)
                if owner is not None:
                    # This caller was started; it is not an undispatched member.
                    owner.reported_cancelled.add(node_id)

    async def materialize_pending(
        self,
        workflow,
        context,
        boundaries,
        outputs,
        *,
        in_flight,
        node_yields,
        ready_nodes,
        pending_retry,
        retry_tracker,
    ):
        """Drain eligible boundaries and any new outer adapter failure iteratively."""
        completed = set()
        failures = []
        while True:
            progress = False
            for boundary in boundaries.pending():
                if not boundaries.can_materialize(
                    boundary, in_flight=in_flight, node_yields=set(node_yields)
                ):
                    continue
                await flush_cancellations(
                    boundaries,
                    workflow,
                    context,
                    boundary,
                    node_outputs=outputs,
                    in_flight=in_flight,
                    node_yields=set(node_yields),
                )
                output_id = await materialize(
                    boundaries, workflow, context, boundary, outputs
                )
                completed_slots, adaptation_errors = await self.complete(
                    [output_id], outputs, context, boundaries
                )
                completed.update(completed_slots)
                progress = True
                for error in adaptation_errors:
                    assert error.node_id is not None
                    contained = await handle_failure(
                        boundaries,
                        workflow,
                        context,
                        error.node_id,
                        error,
                        ready_nodes=ready_nodes,
                        pending_retry=pending_retry,
                        retry_tracker=retry_tracker,
                    )
                    await self.failed(error, boundaries, context)
                    if not contained:
                        failures.append(error)
            if not progress:
                return completed, failures

    async def record_retry(
        self, node_id: str, retries: RetryTracker, context: ExecutionContext
    ) -> None:
        state = retries.get_state(node_id)
        for parent_id, pending in list(self.frames.items()):
            frame = pending.frame
            if frame.status == "pending" and (
                node_id == frame.replacement_id
                or node_id.startswith(frame.replacement_id + "/")
                or node_id.startswith(frame.logical_node_id + "/replacement_")
            ):
                updated = frame.model_copy(
                    update={
                        "retries": {
                            **frame.retries,
                            node_id: ReplacementRetry(
                                attempt=state.attempt, next_retry_at=state.next_retry_at
                            ),
                        }
                    }
                )
                await self._checkpoint(parent_id, updated, context)

    async def _checkpoint(
        self, node_id: str, frame: ReplacementFrame, context: ExecutionContext
    ) -> None:
        old = self.frames[node_id]
        await context.on_node_replacement_checkpoint(frame=frame)
        self.frames[node_id] = PendingReplacement(
            frame,
            old.node,
            old.input,
            old.input_type,
            old.output_type,
            old.child_output_type,
        )
