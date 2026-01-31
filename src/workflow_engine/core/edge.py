# workflow_engine/core/edge.py

from ..utils.immutable import ImmutableBaseModel
from .node import Node
from .values import Value, ValueType, ValueSchema


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
    ) -> "Edge":
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


class SynchronizationEdge(ImmutableBaseModel):
    """
    An edge that carries no information, but indicates that the target node must
    run after the source node finishes.
    """

    source_id: str
    target_id: str


class InputEdge(ImmutableBaseModel):
    """
    An "edge" that maps a field from the workflow's input to the input of a
    target node.
    """

    input_key: str
    target_id: str
    target_key: str
    input_schema: ValueSchema | None = None

    @classmethod
    def from_node(
        cls,
        *,
        input_key: str,
        target: Node,
        target_key: str,
        input_schema: ValueSchema | None = None,
    ) -> "InputEdge":
        return cls(
            input_key=input_key,
            target_id=target.id,
            target_key=target_key,
            input_schema=input_schema,
        )

    def validate_types(self, input_type: ValueType, target: Node):
        if self.target_key not in target.input_fields:
            raise ValueError(
                f"Target node {target.id} does not have a {self.target_key} field"
            )

        target_input_type, _ = target.input_fields[self.target_key]
        assert issubclass(target_input_type, Value)

        if not input_type.can_cast_to(target_input_type):
            raise TypeError(
                f"Input edge to {target.id}.{self.target_key} has invalid types: {input_type} is not assignable to {target_input_type}"
            )


class OutputEdge(ImmutableBaseModel):
    """
    An "edge" that maps a source node's output to a special output of the
    workflow.
    """

    source_id: str
    source_key: str
    output_key: str
    output_schema: ValueSchema | None = None

    @classmethod
    def from_node(
        cls,
        *,
        source: Node,
        source_key: str,
        output_key: str,
        output_schema: ValueSchema | None = None,
    ) -> "OutputEdge":
        return cls(
            source_id=source.id,
            source_key=source_key,
            output_key=output_key,
            output_schema=output_schema,
        )

    def validate_types(self, source: Node, output_type: ValueType):
        if self.source_key not in source.output_fields:
            raise ValueError(
                f"Source node {source.id} does not have a {self.source_key} field"
            )

        source_output_type, _ = source.output_fields[self.source_key]
        assert issubclass(source_output_type, Value)

        if not source_output_type.can_cast_to(output_type):
            raise TypeError(
                f"Output edge from {source.id}.{self.source_key} has invalid types: {source_output_type} is not assignable to {output_type}"
            )


__all__ = [
    "Edge",
    "InputEdge",
    "OutputEdge",
    "SynchronizationEdge",
]
