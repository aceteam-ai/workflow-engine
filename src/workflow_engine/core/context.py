# workflow_engine/core/context.py
from abc import ABC, abstractmethod
from typing import TypeVar

from overrides import EnforceOverrides

from .error import ShouldRetry, WorkflowErrors
from .node import Node
from .values import DataMapping, FileValue
from .workflow import Workflow

F = TypeVar("F", bound=FileValue)


class Context(ABC, EnforceOverrides):
    """
    Represents the environment in which a workflow is executed.
    A context's life is limited to the execution of a single workflow.
    """

    @abstractmethod
    async def read(
        self,
        file: FileValue,
    ) -> bytes:
        """
        Read the content of a file from the context.

        file: the file to read

        The context can modify the file by returning a different FileValue.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def write(
        self,
        file: F,
        content: bytes,
    ) -> F:
        """
        Write the content of a file to the context.

        file: the file to write
        content: the content to write

        The context can modify the file by returning a different FileValue.
        """
        raise NotImplementedError("Subclasses must implement this method")

    async def on_node_start(
        self,
        *,
        node: "Node",
        input: DataMapping,
    ) -> DataMapping | None:
        """
        A hook that is called when a node starts execution.

        If the context already knows what the node's output will be, return that
        output to skip node execution.
        """
        return None

    async def on_node_error(
        self,
        *,
        node: "Node",
        input: DataMapping,
        exception: Exception,
    ) -> Exception | DataMapping:
        """
        A hook that is called when a node raises an error.
        The context can modify the error by returning a different Exception, or
        it can silence the error by returning an output.
        """
        return exception

    async def on_node_retry(
        self,
        *,
        node: "Node",
        input: DataMapping,
        exception: ShouldRetry,
        attempt: int,
    ) -> None:
        """
        A hook that is called when a node is scheduled for retry after raising
        a ShouldRetry exception.

        node: the node that will be retried
        input: the input data to the node
        exception: the ShouldRetry exception that was raised
        attempt: the retry attempt number (1 for first retry, 2 for second, etc.)
        """
        pass

    async def on_node_finish(
        self,
        *,
        node: "Node",
        input: DataMapping,
        output: DataMapping,
    ) -> DataMapping:
        """
        A hook that is called when a node finishes execution by returning a
        DataMapping (not a Workflow).

        node: the node that finished execution
        input: the input data to the node
        output: the output data from the node

        The context can modify the output by returning a different DataMapping.
        """
        return output

    async def on_workflow_start(
        self,
        *,
        workflow: "Workflow",
        input: DataMapping,
    ) -> tuple[WorkflowErrors, DataMapping] | None:
        """
        A hook that is called when a workflow starts execution.

        workflow: the workflow that is starting execution
        input: the input data to the workflow

        If the context already knows what the workflow's output will be, return
        that output to skip workflow execution.
        """
        return None

    async def on_workflow_error(
        self,
        *,
        workflow: "Workflow",
        input: DataMapping,
        errors: WorkflowErrors,
        partial_output: DataMapping,
    ) -> tuple[WorkflowErrors, DataMapping]:
        """
        A hook that is called when a workflow raises an error.

        workflow: the workflow that raised the error
        input: the input data to the workflow
        errors: the errors that occurred
        partial_output: the partial output data from the workflow

        The context can modify the errors or partial output by returning a
        different tuple.
        """
        return errors, partial_output

    async def on_workflow_finish(
        self,
        *,
        workflow: "Workflow",
        input: DataMapping,
        output: DataMapping,
    ) -> DataMapping:
        """
        A hook that is called when a workflow finishes execution.

        workflow: the workflow that finished execution
        input: the input data to the workflow
        output: the output data from the workflow

        The context can modify the output by returning a different DataMapping.
        """
        return output


__all__ = [
    "Context",
]
