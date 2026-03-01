# workflow_engine/core/execution.py
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from functools import cached_property
from typing import Mapping, Self, TYPE_CHECKING

from overrides import EnforceOverrides

from pydantic import Field
from ..utils.immutable import ImmutableBaseModel
from .error import WorkflowErrors
from .values import DataMapping
from .workflow import Workflow

if TYPE_CHECKING:
    from .context import Context


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
        context: Context,
        workflow: Workflow,
        input: DataMapping,
    ) -> WorkflowExecutionResult:
        pass


__all__ = [
    "ExecutionAlgorithm",
]
