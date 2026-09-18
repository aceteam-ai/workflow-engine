# workflow_engine/core/execution.py
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING, Mapping, Self

from overrides import EnforceOverrides
from pydantic import Field

from ..utils.model import ImmutableBaseModel
from .error import WorkflowErrors, WorkflowException
from .values import DataMapping
from .workflow import ValidatedWorkflow

if TYPE_CHECKING:
    from .context import ExecutionContext


class WorkflowExecutionResultStatus(StrEnum):
    """The status of a workflow execution result.

    Note that this is *not* the same as a workflow's execution status, which can
    also include running or pending states.
    """

    SUCCESS = "success"  # completed successfully
    ERROR = "error"  # errors occurred during execution
    YIELDED = "yielded"  # yielded with no or partial output


class WorkflowExecutionResult(ImmutableBaseModel):
    errors: WorkflowErrors = Field(
        description="The user-exposable errors that occurred during the execution of the workflow."
    )
    output: DataMapping = Field(
        description="The complete output of the workflow, or the partial output if the workflow yielded or had errors."
    )
    node_yields: Mapping[str, str] = Field(
        description="The messages provided by all nodes which yielded during execution."
    )

    @cached_property
    def status(self) -> WorkflowExecutionResultStatus:
        if self.errors.any():
            return WorkflowExecutionResultStatus.ERROR
        if len(self.node_yields) > 0:
            return WorkflowExecutionResultStatus.YIELDED
        return WorkflowExecutionResultStatus.SUCCESS

    @classmethod
    def success(cls, output: DataMapping) -> Self:
        return cls(
            errors=WorkflowErrors(),
            output=output,
            node_yields={},
        )

    @classmethod
    def error(
        cls,
        *,
        errors: WorkflowErrors,
        partial_output: DataMapping,
        node_yields: Mapping[str, str],
    ) -> Self:
        return cls(
            errors=errors,
            output=partial_output,
            node_yields=node_yields,
        )

    @classmethod
    def yielded(
        cls,
        *,
        partial_output: DataMapping,
        node_yields: Mapping[str, str],
    ) -> Self:
        return cls(
            errors=WorkflowErrors(),
            output=partial_output,
            node_yields=node_yields,
        )


class ExecutionAlgorithm(ABC, EnforceOverrides):
    """
    Handles the scheduling and execution of workflow nodes.
    Uses hooks to perform extra functionality at key points in the execution
    flow.
    """

    @abstractmethod
    async def execute(
        self,
        *,
        context: ExecutionContext,
        workflow: ValidatedWorkflow,
        input: DataMapping,
    ) -> WorkflowExecutionResult:
        pass

    @staticmethod
    def require_validated(workflow: ValidatedWorkflow) -> None:
        """
        Enforce, at runtime, that ``workflow`` is actually a ``ValidatedWorkflow``.

        ``execute()`` is typed to take a ``ValidatedWorkflow``, but a type
        annotation is advisory: nothing stops a caller from passing a plain
        ``Workflow`` that was never run through ``validate()``. Left
        unchecked, that mistake is not caught here. It surfaces several
        frames later, inside scheduler internals that assume
        validated-only attributes and methods (``node_input_types``,
        ``get_output``, ...), as an ``AttributeError`` that names the wrong
        thing and gives no hint that the workflow itself was never
        validated. Concrete schedulers must call this before doing anything
        else in their public ``execute()``, so the diagnostic points at the
        actual mistake instead of an incidental crash deep inside a retry
        or error-recovery path.
        """
        if not isinstance(workflow, ValidatedWorkflow):
            raise WorkflowException.for_engineer(
                f"ExecutionAlgorithm.execute() requires a ValidatedWorkflow, "
                f"got {type(workflow).__name__}. Call WorkflowEngine.validate() "
                f"(or Workflow.validate()) before executing."
            )


__all__ = [
    "ExecutionAlgorithm",
]
