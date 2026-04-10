# workflow_engine/core/error.py
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from traceback import format_exception
from typing import TYPE_CHECKING, Self

from pydantic import Field

from ..utils.immutable import ImmutableBaseModel
from .stakeholder import StakeholderLevel

if TYPE_CHECKING:
    from .node import Node
    from .workflow import Workflow


class WorkflowError(ImmutableBaseModel):
    """
    A serialized workflow exception.
    """

    message: str
    timestamp: float
    level: StakeholderLevel
    node_id: str | None = Field(default=None)
    cause: WorkflowError | str | None = Field(default=None)
    traceback: Sequence[str] | None = Field(default=None)

    def filter(self, level: StakeholderLevel) -> Self | None:
        # remove errors that require a lower level of visibility to be seen
        if self.level < level:
            return None
        # erase the traceback from all errors for stakeholders above operator level
        if level > StakeholderLevel.OPERATOR and self.traceback is not None:
            return self.model_update(traceback=None)
        return self


class WorkflowException(RuntimeError):
    """
    An exception that occurred within a workflow.
    All exceptions leaving a node or workflow stack frame are wrapped in this
    class or one of its subclasses.

    The level of the exception determines who can see the error message.
    Stack traces always remain visible to engineers and operators only.
    """

    def __init__(
        self,
        message: str,
        *,
        level: StakeholderLevel,
        node_id: str | None = None,
    ):
        super().__init__(message)
        self.timestamp = datetime.now(timezone.utc).timestamp()
        self.level = level
        self.message = message
        self.node_id = node_id

    def dump(self) -> WorkflowError:
        return WorkflowError(
            message=self.message,
            timestamp=self.timestamp,
            level=self.level,
            node_id=self.node_id,
            cause=(
                None
                if self.__cause__ is None
                else self.__cause__.dump()
                if isinstance(self.__cause__, WorkflowException)
                else str(self.__cause__)
            ),
            traceback=format_exception(self),
        )


class NodeException(WorkflowException):
    """
    An exception that occurred during the execution of a node.
    """

    def __init__(
        self,
        node: "Node",
        message: str,
        *,
        level: StakeholderLevel,
    ):
        super().__init__(message, level=level, node_id=node.id)
        self.node = node


class NodeExpansionException(NodeException):
    """
    An error that occurred while expanding a node into a workflow.
    """

    def __init__(
        self,
        node: "Node",
        workflow: "Workflow",
    ):
        super().__init__(
            node,
            f"Error expanding node {node.id} into the workflow {workflow}",
            level=StakeholderLevel.USER,
        )
        self.workflow = workflow


class ShouldRetry(NodeException):
    """
    An exception that indicates a temporary failure that should be retried.

    Nodes can raise this to signal that the current attempt failed due to a
    transient error (e.g., rate limit, network timeout) and should be retried
    after a backoff period.

    Retrying a node that raised ShouldRetry is always a courtesy of the
    execution algorithm.
    If refused, the error passes up uncaught as a regular NodeException.
    """

    def __init__(
        self,
        node: "Node",
        message: str,
        *,
        level: StakeholderLevel,
        backoff: timedelta = timedelta(seconds=1),
    ):
        super().__init__(node, message, level=level)
        self.backoff = backoff


class ShouldYield(Exception):
    """
    A control-flow signal that a node will not be returning a value anytime
    soon (e.g., it has dispatched work to an external system and is waiting
    for a callback).

    This exception is never surfaced to the user as an error. It is caught by
    the execution algorithm, which calls the ``on_node_yield`` context hook and
    then continues running other ready nodes. Once no more nodes can run, the
    algorithm returns with the partial output as well as the ShouldYield
    messages left by the yielded nodes.

    To resume a yielded workflow, re-run it with the same context. The node's
    ``run`` method is responsible for checking whether its condition is now met
    and either returning a result or raising ``ShouldYield`` again.

    Usage:
        async def run(self, context, input):
            job_id = await submit_long_running_job(input)
            raise ShouldYield(f"Waiting for job {job_id}")
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class WorkflowErrors(ImmutableBaseModel):
    """
    An error object that accumulates the errors that occurred during the
    execution of a workflow.

    None represents an error that is not visible due to filtering.

    workflow_errors contains errors which cannot be associated with a node.
    node_errors contains errors which can be associated with a node.
    """

    workflow_errors: Sequence[WorkflowError | None] = Field(default=())
    node_errors: Mapping[str, Sequence[WorkflowError | None]] = Field(
        default_factory=dict
    )

    @property
    def count(self) -> int:
        return len(self.workflow_errors) + sum(
            len(errors) for errors in self.node_errors.values()
        )

    def any(self) -> bool:
        return self.count > 0

    def filter(self, level: StakeholderLevel) -> Self:
        return self.model_copy(
            update={
                "workflow_errors": tuple(
                    error.filter(level) if isinstance(error, WorkflowError) else error
                    for error in self.workflow_errors
                ),
                "node_errors": {
                    node_id: tuple(
                        error.filter(level)
                        if isinstance(error, WorkflowError)
                        else error
                        for error in errors
                    )
                    for node_id, errors in self.node_errors.items()
                },
            },
        )


class LegacyWorkflowErrors(ImmutableBaseModel):
    """
    A legacy version of WorkflowErrors that is backwards-compatible with
    <=2.0.0rc7 error objects (str | None).
    """

    workflow_errors: Sequence[WorkflowError | str | None] = Field(default=())
    node_errors: Mapping[str, Sequence[WorkflowError | str | None]] = Field(
        default_factory=dict
    )

    @property
    def count(self) -> int:
        return len(self.workflow_errors) + sum(
            len(errors) for errors in self.node_errors.values()
        )

    def any(self) -> bool:
        return self.count > 0

    def filter(self, level: StakeholderLevel) -> Self:
        return self.model_copy(
            update={
                "workflow_errors": tuple(
                    error.filter(level) if isinstance(error, WorkflowError) else error
                    for error in self.workflow_errors
                ),
                "node_errors": {
                    node_id: tuple(
                        error.filter(level)
                        if isinstance(error, WorkflowError)
                        else error
                        for error in errors
                    )
                    for node_id, errors in self.node_errors.items()
                },
            },
        )


class WorkflowErrorsBuilder:
    """
    A mutable builder for WorkflowErrors.
    """

    def __init__(self):
        self._workflow_errors: list[WorkflowError] = []
        self._node_errors: dict[str, list[WorkflowError]] = defaultdict(list)
        self._count = 0

    def add(self, exception: WorkflowException):
        serialized_exception = exception.dump()
        node_id = serialized_exception.node_id
        if node_id is None:
            self._workflow_errors.append(serialized_exception)
        else:
            self._node_errors[node_id].append(serialized_exception)
        self._count += 1

    def build(self) -> WorkflowErrors:
        return WorkflowErrors(
            workflow_errors=tuple(self._workflow_errors),
            node_errors={
                node_id: tuple(errors) for node_id, errors in self._node_errors.items()
            },
        )

    @property
    def count(self) -> int:
        return self._count

    def any(self) -> bool:
        return self.count > 0


__all__ = [
    "LegacyWorkflowErrors",
    "NodeException",
    "NodeExpansionException",
    "ShouldRetry",
    "ShouldYield",
    "WorkflowError",
    "WorkflowErrors",
    "WorkflowErrorsBuilder",
]
