# workflow_engine/core/error.py
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from traceback import format_exception
from typing import TYPE_CHECKING, Sequence

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
    traceback: Sequence[str]


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
    An error object that accumulates the error messages that occurred during the
    execution of a workflow.

    None represents an error that is not visible to the engine operator.

    workflow_errors contains errors which cannot be associated with a node.
    node_errors contains errors which can be associated with a node.
    """

    workflow_errors: list[WorkflowError] = Field(default_factory=list)
    node_errors: dict[str, list[WorkflowError]] = Field(
        default_factory=lambda: defaultdict(list)
    )

    def add(self, exception: WorkflowException):
        serialized_exception = exception.dump()
        node_id = serialized_exception.node_id
        if node_id is None:
            self.workflow_errors.append(serialized_exception)
        else:
            self.node_errors[node_id].append(serialized_exception)

    @property
    def count(self) -> int:
        return len(self.workflow_errors) + sum(
            len(errors) for errors in self.node_errors.values()
        )

    def any(self) -> bool:
        return self.count > 0


__all__ = [
    "NodeException",
    "NodeExpansionException",
    "ShouldRetry",
    "ShouldYield",
    "WorkflowError",
    "WorkflowErrors",
]
