# workflow_engine/core/error.py

from collections import defaultdict
from datetime import timedelta
from typing import TYPE_CHECKING

from pydantic import Field

from ..utils.immutable import ImmutableBaseModel

if TYPE_CHECKING:
    from .workflow import Workflow


class UserException(RuntimeError):
    """
    Any exception that can be reported to the user.

    The node ID does not need to be included in contexts where it can be
    inferred (when the exception is raised from a node, or from a function
    called by a node).

    Usual usage:
    ```
    try:
        do_something_dangerous()
    except AnticipatedException as e:
        raise UserException("prepared message") from e
    ```
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NodeException(RuntimeError):
    """
    An exception that occured during the execution of a node.
    """

    def __init__(self, node_id: str):
        super().__init__()
        self.node_id = node_id

    @property
    def message(self) -> str | None:
        if isinstance(self.__cause__, UserException):
            return self.__cause__.message
        return None


class NodeExpansionException(UserException):
    """
    An error that occurred while expanding a node into a workflow.
    """

    def __init__(self, node_id: str, workflow: "Workflow"):
        super().__init__(f"Error expanding node {node_id} into the workflow {workflow}")
        self.node_id = node_id
        self.workflow = workflow

    @property
    def message(self) -> str | None:
        base_message = (
            f"Error expanding node {self.node_id} into the workflow {self.workflow}"
        )
        if isinstance(self.__cause__, UserException):
            return f"{base_message}, due to {self.__cause__}."
        else:
            return base_message


class ShouldRetry(UserException):
    """
    An exception that indicates a temporary failure that should be retried.

    Nodes can raise this to signal that the current attempt failed due to a
    transient error (e.g., rate limit, network timeout) and should be retried
    after a backoff period.

    Usage:
        try:
            response = await api_call()
        except RateLimitError as e:
            raise ShouldRetry(
                message="Rate limited by API",
                backoff=timedelta(seconds=e.retry_after),
            )
    """

    def __init__(
        self,
        message: str,
        backoff: timedelta = timedelta(seconds=1),
    ):
        super().__init__(message)
        self.backoff = backoff


class WorkflowErrors(ImmutableBaseModel):
    """
    An error object that accumulates the errors that occurred during the
    execution of a workflow.

    None represents an error that is not user-visible.

    workflow_errors contains errors which cannot be associated with a node.
    node_errors contains errors which can be associated with a node.
    """

    workflow_errors: list[str | None] = Field(default_factory=list)
    node_errors: dict[str, list[str | None]] = Field(
        default_factory=lambda: defaultdict(list)
    )

    def add(self, exception: Exception):
        if isinstance(exception, NodeException):
            node_id = exception.node_id
            message = exception.message
        else:
            node_id = None
            if isinstance(exception, UserException):
                message = exception.message
            else:
                message = None
        if node_id is None:
            self.workflow_errors.append(message)
        else:
            self.node_errors[node_id].append(message)

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
    "UserException",
    "WorkflowErrors",
]
