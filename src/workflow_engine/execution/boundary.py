# workflow_engine/execution/boundary.py
"""
Execution-side tracking for error boundaries (``core.boundary.ErrorBoundaryNode``,
e.g. ``AttemptNode`` in ``nodes/attempt.py``).

Both execution algorithms register a boundary at the single site where an
``ErrorBoundaryNode`` is expanded into a subgraph, keyed by the node's flat
id. Boundary membership is a string-prefix test on flat ids
(``node_id == boundary_id or node_id.startswith(boundary_id + "/")``), the
same rule ``Workflow._validate_no_id_prefix_collisions`` enforces, so ids
stay flat and downstream ledger/resume/pin machinery keeps working
unmodified.

See #201 and discussion #198 for the motivation and the full design.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping, Set
from dataclasses import dataclass, field

from ..core.boundary import CancelReason, ErrorBoundaryNode
from ..core.context import ExecutionContext
from ..core.error import ErrorClass, WorkflowException
from ..core.node import Node
from ..core.stakeholder import StakeholderLevel
from ..core.values import Data, DataMapping, ErrorClassValue, ResultError, StringValue
from ..core.workflow import ValidatedWorkflow
from .retry import RetryTracker

__all__ = [
    "Boundary",
    "BoundaryTracker",
    "CancelReason",
    "flush_cancellations",
    "handle_failure",
    "materialize",
    "result_error_from_exception",
]


@dataclass
class Boundary:
    """
    The runtime state of a single registered error boundary.

    ``id`` is the boundary node's flat id at the time it was expanded (e.g.
    ``"for_each/element_16/attempt"``). ``node`` is the ``ErrorBoundaryNode``
    instance itself, which the executor drops from the graph once it expands
    it. ``input`` is the ``DataMapping`` the boundary node was expanded with,
    passed through unchanged to ``on_boundary_error``.
    """

    id: str
    node: Node
    input: DataMapping
    input_type: type[Data]
    output_type: type[Data]
    output_node_id: str
    failure: WorkflowException | None = None
    materialized: bool = False
    reported_cancelled: set[str] = field(default_factory=set)


def _is_member_id(boundary_id: str, node_id: str) -> bool:
    return node_id == boundary_id or node_id.startswith(boundary_id + "/")


class BoundaryTracker:
    """
    Tracks every error boundary registered during a single execution, and
    their failure/materialization state. One instance per ``execute()`` call,
    shared by both ``TopologicalExecutionAlgorithm`` and
    ``ParallelExecutionAlgorithm``.
    """

    def __init__(self) -> None:
        self._boundaries: dict[str, Boundary] = {}

    def register(
        self,
        *,
        node_id: str,
        node: Node,
        input: DataMapping,
        input_type: type[Data],
        output_type: type[Data],
        subgraph: ValidatedWorkflow,
    ) -> None:
        """Register node_id (which must be an ErrorBoundaryNode) as a boundary root."""
        assert isinstance(node, ErrorBoundaryNode)
        self._boundaries[node_id] = Boundary(
            id=node_id,
            node=node,
            input=input,
            input_type=input_type,
            output_type=output_type,
            output_node_id=f"{node_id}/{subgraph.output_node.id}",
        )

    def innermost(self, node_id: str) -> Boundary | None:
        """The most deeply nested registered boundary enclosing node_id, if any."""
        best: Boundary | None = None
        for boundary in self._boundaries.values():
            if _is_member_id(boundary.id, node_id) and (
                best is None or len(boundary.id) > len(best.id)
            ):
                best = boundary
        return best

    def is_blocked(self, node_id: str) -> bool:
        """Whether node_id is a member of any boundary that has already failed."""
        return any(
            boundary.failure is not None and _is_member_id(boundary.id, node_id)
            for boundary in self._boundaries.values()
        )

    def is_subsumed(self, b: Boundary) -> bool:
        """Whether some strictly-enclosing boundary has already failed."""
        return any(
            boundary is not b
            and boundary.failure is not None
            and b.id.startswith(boundary.id + "/")
            for boundary in self._boundaries.values()
        )

    def members(self, b: Boundary, workflow: ValidatedWorkflow) -> Iterator[str]:
        """The ids, currently present in workflow, that are members of b."""
        prefix = b.id + "/"
        for node_id in workflow.nodes_by_id:
            if node_id.startswith(prefix):
                yield node_id

    def fail(self, b: Boundary, exc: WorkflowException) -> bool:
        """Record b's failure. Returns True if this is the first failure for b."""
        if b.failure is not None:
            return False
        b.failure = exc
        return True

    def can_materialize(
        self,
        b: Boundary,
        *,
        in_flight: Set[str],
        node_yields: Set[str],
    ) -> bool:
        """
        Whether b can materialize its err arm right now: it has failed, has
        not already materialized, is not subsumed by an outer boundary's
        failure, and has no member currently in flight or yielded.
        """
        if b.failure is None or b.materialized or self.is_subsumed(b):
            return False
        if any(_is_member_id(b.id, nid) for nid in in_flight):
            return False
        if any(_is_member_id(b.id, nid) for nid in node_yields):
            return False
        return True

    def pending(self) -> list[Boundary]:
        """Every failed, not-yet-materialized, not-subsumed boundary."""
        return [
            b
            for b in self._boundaries.values()
            if b.failure is not None and not b.materialized and not self.is_subsumed(b)
        ]


async def handle_failure(
    tracker: BoundaryTracker,
    workflow: ValidatedWorkflow,
    context: ExecutionContext,
    node_id: str,
    exc: WorkflowException,
    *,
    ready_nodes: MutableMapping[str, DataMapping] | None,
    pending_retry: MutableMapping[str, DataMapping],
    retry_tracker: RetryTracker,
) -> bool:
    """
    Handle a node failure that may belong to a registered error boundary.

    Returns False if node_id is not inside any registered boundary; the
    caller should fall back to its existing (run-level) error handling.

    Returns True if the failure was absorbed by a boundary: either this is
    the first failure inside the boundary (in which case any of the
    boundary's members already sitting in ``pending_retry`` are purged and
    reported ``on_node_cancelled(RETRY_ABANDONED)``), or the boundary had
    already failed from a different member during drain, in which case
    nothing more happens here (the failing node's own ``on_node_error`` has
    already fired inside ``Node.__call__``).

    ``ready_nodes`` is purged in place when provided (the topological
    executor's single ready dict); the parallel executor passes None because
    it has no such dict between dispatch batches.
    """
    b = tracker.innermost(node_id)
    if b is None:
        return False

    if not tracker.fail(b, exc):
        return True

    if ready_nodes is not None:
        for member_id in [nid for nid in ready_nodes if _is_member_id(b.id, nid)]:
            del ready_nodes[member_id]

    for member_id in [nid for nid in pending_retry if _is_member_id(b.id, nid)]:
        member_input = pending_retry.pop(member_id)
        retry_tracker.discard(member_id)
        b.reported_cancelled.add(member_id)
        member_node = workflow.nodes_by_id[member_id]
        await context.on_node_cancelled(
            node=member_node,
            input_type=workflow.node_input_types[member_id],
            output_type=workflow.node_output_types[member_id],
            input=member_input,
            boundary_id=b.id,
            reason=CancelReason.RETRY_ABANDONED,
            cause=exc,
        )

    return True


async def abandon_retry_if_blocked(
    tracker: BoundaryTracker,
    workflow: ValidatedWorkflow,
    context: ExecutionContext,
    node_id: str,
    node_input: DataMapping,
) -> bool:
    """
    If node_id is a member of an already-failed boundary, report it
    RETRY_ABANDONED instead of letting the caller schedule it for retry.

    This covers the case where a boundary member's ShouldRetry is processed
    after its boundary has already failed within the same dispatch batch (the
    parallel executor processes a batch of completions as a set, so a
    boundary failure and a sibling's retry can land in the same batch in
    either order). Returns True if the node was abandoned (the caller must
    not add it to pending_retry); False if it is not blocked.
    """
    b = tracker.innermost(node_id)
    if b is None or b.failure is None:
        return False
    b.reported_cancelled.add(node_id)
    await context.on_node_cancelled(
        node=workflow.nodes_by_id[node_id],
        input_type=workflow.node_input_types[node_id],
        output_type=workflow.node_output_types[node_id],
        input=node_input,
        boundary_id=b.id,
        reason=CancelReason.RETRY_ABANDONED,
        cause=b.failure,
    )
    return True


async def flush_cancellations(
    tracker: BoundaryTracker,
    workflow: ValidatedWorkflow,
    context: ExecutionContext,
    b: Boundary,
    *,
    node_outputs: Mapping[str, DataMapping],
    in_flight: Set[str],
    node_yields: Set[str],
) -> None:
    """
    Fire ``on_node_cancelled(NOT_SCHEDULED)`` for every member of b that has
    settled into "will not run this pass": not already finished, not
    in-flight, not yielded, not the boundary's own output node, not the
    member whose failure caused b to fail (its own ``on_node_error`` already
    fired), and not already reported. Idempotent, so it is safe to call
    repeatedly across a pass: once at failure time (via the pending-boundary
    check), again immediately before materialization, and once more at pass
    end for boundaries still held open by a yielded member.
    """
    assert b.failure is not None
    failed_node_id = b.failure.node_id
    for member_id in tracker.members(b, workflow):
        if member_id == b.output_node_id:
            continue
        if member_id == failed_node_id:
            continue
        if member_id in node_outputs:
            continue
        if member_id in in_flight:
            continue
        if member_id in node_yields:
            continue
        if member_id in b.reported_cancelled:
            continue
        b.reported_cancelled.add(member_id)
        member_node = workflow.nodes_by_id[member_id]
        await context.on_node_cancelled(
            node=member_node,
            input_type=workflow.node_input_types[member_id],
            output_type=workflow.node_output_types[member_id],
            input=None,
            boundary_id=b.id,
            reason=CancelReason.NOT_SCHEDULED,
            cause=b.failure,
        )


async def materialize(
    tracker: BoundaryTracker,
    workflow: ValidatedWorkflow,
    context: ExecutionContext,
    b: Boundary,
    node_outputs: MutableMapping[str, DataMapping],
) -> str:
    """
    Materialize b's err arm as the output of its output node, calling
    ``on_boundary_error`` and recording the result in node_outputs. Returns
    b.output_node_id so the caller can compute newly-ready successors.
    """
    assert b.failure is not None
    node = b.node
    assert isinstance(node, ErrorBoundaryNode)
    error = result_error_from_exception(b.failure)
    output = node.materialize_error(output_type=b.output_type, error=error)
    output = await context.on_boundary_error(
        node=b.node,
        input_type=b.input_type,
        output_type=b.output_type,
        input=b.input,
        error=error,
        output=output,
        cause=b.failure,
    )
    node_outputs[b.output_node_id] = output
    b.materialized = True
    return b.output_node_id


def result_error_from_exception(exc: WorkflowException) -> ResultError:
    """
    Build the structured err-arm value for a boundary's failing exception.

    ``message`` is redacted unless the exception is already USER level:
    unlike ``WorkflowErrors``, which every viewer filters through
    ``WorkflowError.filter`` before it is rendered, a materialized ``err``
    value flows straight into user-visible workflow output.
    """
    assert exc.node_id is not None
    cause: BaseException = exc
    while cause.__cause__ is not None:
        cause = cause.__cause__
    name = type(cause).__name__
    message = (
        exc.message
        if exc.level >= StakeholderLevel.USER
        else "An internal error occurred"
    )
    error_class = (
        exc.error_class if exc.error_class is not None else ErrorClass.SYSTEMIC
    )
    return ResultError(
        error_class=ErrorClassValue(error_class),
        name=StringValue(name),
        message=StringValue(message),
        node_id=StringValue(exc.node_id),
    )
