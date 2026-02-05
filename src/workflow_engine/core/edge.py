# workflow_engine/core/edge.py

from typing import Self

from ..utils.immutable import ImmutableBaseModel
from .node import Node
from .values import Value


class Edge(ImmutableBaseModel):
    """
    An edge connects the output of source node to the input of a target node.
    """

    source_id: str
    source_key: str
    target_id: str
    target_key: str

    @classmethod
    def from_nodes(
        cls,
        *,
        source: Node,
        source_key: str,
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
        if self.source_key not in source.output_fields:
            raise ValueError(
                f"Source node {source.id} does not have a {self.source_key} field"
            )

        if self.target_key not in target.input_fields:
            raise ValueError(
                f"Target node {target.id} does not have a {self.target_key} field"
            )

        source_output_type, _ = source.output_fields[self.source_key]
        assert issubclass(source_output_type, Value)
        target_input_type, _ = target.input_fields[self.target_key]
        assert issubclass(target_input_type, Value)

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
