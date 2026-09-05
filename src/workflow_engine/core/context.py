# workflow_engine/core/context.py
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TypeVar

from overrides import EnforceOverrides

from ..utils.env import get_env as _resolve_env_var
from .boundary import CancelReason
from .error import ShouldRetry, ShouldYield, WorkflowErrors, WorkflowException
from .execution import WorkflowExecutionResult
from .node import Node, NodeRegistry
from .values import Data, DataMapping, FileValue, ResultError, ValueRegistry
from .workflow import ValidatedWorkflow, Workflow

F = TypeVar("F", bound=FileValue)


class ValidationContext:
    """
    Represents a context in which a node or workflow is validated.

    Validation includes inferring the types of nodes, their input and output
    types, and validating that edges are connected
    """

    def __init__(
        self,
        *,
        node_registry: NodeRegistry = NodeRegistry.DEFAULT,
        value_registry: ValueRegistry = ValueRegistry.DEFAULT,
    ):
        self.node_registry = node_registry
        self.value_registry = value_registry

    async def get_env(self, key: str, default: str | None = None) -> str:
        """
        Resolve an environment variable for use during validation or execution.

        The default implementation reads from the process environment (a loaded
        ``.env`` file included), returning ``default`` when the variable is
        unset and raising ``ValueError`` when it is unset and no default is
        given.

        This is the single source of environment resolution; ``ExecutionContext``
        delegates here. The method is awaitable so that interactive contexts can
        raise ``ShouldYield`` to pause execution and ask the user to supply a
        missing variable (returning the now-provided value on resume), or fetch
        secrets from an external store.

        key: the environment variable name
        default: value to return when the variable is unset
        """
        return _resolve_env_var(key, default=default)


class ExecutionContext(ABC, EnforceOverrides):
    """
    Represents the environment in which a workflow is executed.
    A context's life is limited to the execution of a single workflow.

    An execution context always contains a validation context, allowing it to
    validate sub-workflows that are emitted by nodes.
    """

    def __init__(self, *, validation_context: ValidationContext | None = None):
        if validation_context is None:
            validation_context = ValidationContext()
        self.validation_context = validation_context

    async def get_env(self, key: str, default: str | None = None) -> str:
        """
        Resolve an environment variable, delegating to the validation context.

        Nodes call this to obtain credentials and configuration at runtime. The
        underlying implementation may raise ``ShouldYield`` to pause execution
        and request a missing variable from the user; see
        ``ValidationContext.get_env``.

        key: the environment variable name
        default: value to return when the variable is unset
        """
        return await self.validation_context.get_env(key, default=default)

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
        node: Node,
        input_type: type[Data],
        output_type: type[Data],
        input: DataMapping,
    ) -> DataMapping | Workflow | None:
        """
        A hook that is called when a node starts execution.

        If the context already knows what the node's output will be, return that
        output to skip node execution.
        """
        return None

    async def on_node_error(
        self,
        *,
        node: Node,
        input_type: type[Data],
        output_type: type[Data],
        input: DataMapping,
        exception: WorkflowException,
    ) -> WorkflowException | DataMapping:
        """
        A hook that is called when a node raises an error.
        The context can modify the error by returning a different exception, or
        it can silence the error by returning an output.
        """
        return exception

    async def on_node_yield(
        self,
        *,
        node: Node,
        input_type: type[Data],
        output_type: type[Data],
        input: DataMapping,
        exception: ShouldYield,
    ) -> None:
        """
        A hook that is called when a node raises ShouldYield, signalling that
        it has dispatched work externally and cannot return a value yet.

        The context can use this hook to persist the node's yield state so that
        a future execution can detect that the external work is complete and
        allow the node to resume.

        node: the node that yielded
        input: the input data the node was given
        exception: the ShouldYield exception, whose message describes what the
                   node is waiting for
        """
        pass

    async def on_node_retry(
        self,
        *,
        node: Node,
        input_type: type[Data],
        output_type: type[Data],
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

    async def on_node_cancelled(
        self,
        *,
        node: Node,
        input_type: type[Data],
        output_type: type[Data],
        input: DataMapping | None,
        boundary_id: str,
        reason: CancelReason,
        cause: WorkflowException,
    ) -> None:
        """
        A hook that fires once per pass for each member of a failed error
        boundary (see ``nodes/attempt.py``) that will not run this pass.

        NOT_SCHEDULED: the boundary failed before this node was dispatched;
        ``input`` is None because an upstream member never completed.
        RETRY_ABANDONED: the node was in ``ShouldRetry`` backoff and is not
        re-dispatched; ``input`` is the node's input.

        ``cause`` is the exception that failed the boundary; ``cause.node_id``
        is the sibling that failed, which may differ from ``node.id``.

        Never fires for a node that was in flight when the boundary failed
        (it gets its normal ``on_node_finish`` / ``on_node_error`` /
        ``on_node_yield`` instead), nor for a yielded node, nor for the
        boundary's own output node (which gets ``on_boundary_error`` instead,
        once the boundary actually materializes).

        In a pass where the boundary is held open by a yielded member (see
        ``on_boundary_error``), this disposition is per pass: a node reported
        cancelled here may still run on a later, resumed pass.
        """
        pass

    async def on_boundary_error(
        self,
        *,
        node: Node,
        input_type: type[Data],
        output_type: type[Data],
        input: DataMapping,
        error: ResultError,
        output: DataMapping,
        cause: WorkflowException,
    ) -> DataMapping:
        """
        A hook that fires when an error boundary (see ``nodes/attempt.py``)
        materializes its err arm: every member has settled, and none yielded
        in this pass.

        node: the boundary node itself (e.g. the ``AttemptNode``).
        input: the input the boundary node was expanded with.
        error: the structured error being materialized.
        output: the mapping about to be written as the output of the
                boundary's own output node.
        cause: the exception that failed the boundary.

        Return the output (possibly replaced), mirroring ``on_node_finish``.
        A host that persists ``output`` against ``node.id`` may short-circuit
        the whole boundary on a later pass from ``on_node_start``: that is
        safe here specifically because, by construction, nothing inside the
        boundary is suspended when this hook fires.
        """
        return output

    async def on_node_finish(
        self,
        *,
        node: Node,
        input_type: type[Data],
        output_type: type[Data],
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

    async def on_node_expand(
        self,
        *,
        node: Node,
        input_type: type[Data],
        output_type: type[Data],
        input: DataMapping,
        workflow: ValidatedWorkflow,
    ) -> ValidatedWorkflow:
        """
        A hook that is called when a node finishes execution by returning a
        Workflow (i.e., it expands into a subgraph).

        node: the node that emitted the workflow
        input: the input data to the node
        workflow: the validated workflow emitted by the node

        The context can modify the workflow by returning a different Workflow.
        """
        return workflow

    async def on_workflow_start(
        self,
        *,
        workflow: ValidatedWorkflow,
        input: DataMapping,
    ) -> WorkflowExecutionResult | None:
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
        workflow: ValidatedWorkflow,
        input: DataMapping,
        errors: WorkflowErrors,
        partial_output: DataMapping,
        node_yields: Mapping[str, str],
    ) -> WorkflowExecutionResult:
        """
        A hook that is called when a workflow raises an error.

        workflow: the workflow that raised the error
        input: the input data to the workflow
        errors: the errors that occurred
        partial_output: the partial output data from the workflow
        node_yields: the per-node yield messages for any nodes that yielded
                     during execution

        The context can modify the execution result by returning a
        different WorkflowExecutionResult.
        """
        return WorkflowExecutionResult.error(
            errors=errors,
            partial_output=partial_output,
            node_yields=node_yields,
        )

    async def on_workflow_finish(
        self,
        *,
        workflow: ValidatedWorkflow,
        input: DataMapping,
        output: DataMapping,
    ) -> WorkflowExecutionResult:
        """
        A hook that is called when a workflow finishes execution with no errors.

        workflow: the workflow that finished execution
        input: the input data to the workflow
        output: the output data from the workflow

        The context can modify the execution result by returning a different
        WorkflowExecutionResult.
        """
        return WorkflowExecutionResult.success(output=output)

    async def on_workflow_yield(
        self,
        *,
        workflow: ValidatedWorkflow,
        input: DataMapping,
        partial_output: DataMapping,
        node_yields: Mapping[str, str],
    ) -> WorkflowExecutionResult:
        """
        A hook that is called when a workflow yields, signalling that one or
        more nodes have dispatched work externally and the workflow cannot
        complete yet.

        workflow: the workflow that yielded
        input: the input data to the workflow
        node_yields: the per-node yield messages for any nodes that yielded
                     during execution
        partial_output: the partial output data from nodes that completed
                        before the workflow yielded

        The context can modify the execution result by returning a
        different WorkflowExecutionResult.
        """
        return WorkflowExecutionResult.yielded(
            partial_output=partial_output,
            node_yields=node_yields,
        )


__all__ = [
    "ExecutionContext",
]
