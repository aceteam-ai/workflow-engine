# workflow_engine/core/boundary.py
"""
Core-level types shared by error-boundary nodes (e.g. ``AttemptNode``, in
``nodes/attempt.py``) and the execution algorithms that give them meaning
(``execution/boundary.py``).

This module exists so that ``core/context.py`` can declare the
``on_node_cancelled`` hook without importing ``execution/``, and so that
``execution/`` can recognize a boundary node without importing ``nodes/``.
See #201 and discussion #198 for the motivation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from .values import Data, DataMapping, ResultError


class ErrorBoundaryNode(ABC):
    """
    Mixin for a node whose expansion is an error boundary.

    A node that mixes this in (e.g. ``AttemptNode``) is registered by the
    execution algorithm, at the single site where it is expanded into a
    subgraph, as the root of a boundary. Every node whose flat id has the
    boundary's id as a ``/``-separated prefix is a member. If any member
    fails, the boundary's own output is replaced with the result of
    ``materialize_error`` instead of the failure propagating to the run.
    """

    @abstractmethod
    def materialize_error(
        self, *, output_type: type[Data], error: ResultError
    ) -> DataMapping:
        """
        Build the output of this boundary's output node for the err arm.

        output_type: the boundary node's own (static or dynamic) output type.
        error: the structured error to materialize.
        """
        raise NotImplementedError("Subclasses must implement this method")


class CancelReason(StrEnum):
    """
    Why a boundary member did not run in a pass where its boundary failed.
    """

    NOT_SCHEDULED = "not_scheduled"
    """The boundary failed before this member was dispatched."""

    RETRY_ABANDONED = "retry_abandoned"
    """The member was in ``ShouldRetry`` backoff and was not re-dispatched."""


__all__ = [
    "CancelReason",
    "ErrorBoundaryNode",
]
