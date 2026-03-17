# workflow_engine/core/edge.py

from collections.abc import Sequence
from functools import cached_property
from typing import Self

from pydantic import model_validator

from ..utils.immutable import ImmutableBaseModel
from .node import Node
from .values import resolve_path


class Edge(ImmutableBaseModel):
    """
    An edge connects an output of source node to the input of a target node.
    Outputs can be taken from any depth of the source node's output data.
    """

    source_id: str
    source_key: str | Sequence[str]
    target_id: str
    target_key: str

    @model_validator(mode="after")
    def _validate_keys(self):
        if isinstance(self.source_key, str):
            if len(self.source_key) == 0:
                raise ValueError("Source key cannot be empty")
        else:
            if len(self.source_key) == 0:
                raise ValueError("Source key path needs at least one segment")

        if len(self.target_key) == 0:
            raise ValueError("Target key cannot be empty")

        return self

    @cached_property
    def source_key_path(self) -> Sequence[str]:
        if isinstance(self.source_key, str):
            return (self.source_key,)
        return tuple(self.source_key)

    @cached_property
    def source_key_path_string(self) -> str:
        return ".".join(self.source_key)

    @classmethod
    def from_nodes(
        cls,
        *,
        source: Node,
        source_key: str | Sequence[str],
        target: Node,
        target_key: str,
    ) -> Self:
        """
        Self-validating factory method.
        """
        edge = cls(
            source_id=source.id,
            source_key=source_key,
            target_id=target.id,
            target_key=target_key,
        )
        edge.validate_types(source, target)
        return edge

    def validate_types(self, source: Node, target: Node):
        try:
            source_output_type = resolve_path(
                data_type=source.output_type,
                path=self.source_key_path,
            )
        except TypeError as e:
            e.add_note(
                f"Source node {source.id} does not have a {self.source_key_path_string} field"
            )
            raise

        try:
            target_input_type = resolve_path(
                data_type=target.input_type,
                path=(self.target_key,),
            )
        except TypeError as e:
            e.add_note(
                f"Target node {target.id} does not have a {self.target_key} field"
            )
            raise

        if not source_output_type.can_cast_to(target_input_type):
            raise TypeError(
                f"Edge from {source.id}.{self.source_key} to {target.id}.{self.target_key} has invalid types: {source_output_type} is not assignable to {target_input_type}"
            )

    def with_namespace(self, namespace: str) -> Self:
        return self.model_update(
            source_id=f"{namespace}/{self.source_id}",
            target_id=f"{namespace}/{self.target_id}",
        )


class SynchronizationEdge(ImmutableBaseModel):
    """
    An edge that carries no information, but indicates that the target node must
    run after the source node finishes.
    """

    source_id: str
    target_id: str


__all__ = [
    "Edge",
    "SynchronizationEdge",
]
