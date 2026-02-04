# workflow_engine/core/edge.py

from collections.abc import Sequence

from ..utils.immutable import ImmutableBaseModel
from .node import Node
from .path import FieldPath, resolve_path_type
from .values import Value, ValueType, ValueSchema


class Edge(ImmutableBaseModel):
    """
    An edge connects the output of source node to the input of a target node.

    Supports deep field access using paths:
    - Simple string: "result" (backwards compatible)
    - Deep path: ("data", "items", 0, "name") to access nested fields
    """

    source_id: str
    source_key: FieldPath
    target_id: str
    target_key: FieldPath

    @classmethod
    def from_nodes(
        cls,
        *,
        source: Node,
        source_key: FieldPath,
        target: Node,
        target_key: FieldPath,
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
        """
        Validate that the edge's source and target paths are valid and type-compatible.

        For deep paths, this validates:
        1. The path is valid for the source/target node's fields
        2. The type at the end of the source path can cast to the target path's type
        """
        # Resolve source type through path
        try:
            source_output_type = self._resolve_source_type(source)
        except (ValueError, TypeError, AttributeError) as e:
            raise ValueError(
                f"Invalid source path {self.source_key} on node {source.id}: {e}"
            ) from e

        # Resolve target type through path
        try:
            target_input_type = self._resolve_target_type(target)
        except (ValueError, TypeError, AttributeError) as e:
            raise ValueError(
                f"Invalid target path {self.target_key} on node {target.id}: {e}"
            ) from e

        assert issubclass(source_output_type, Value)
        assert issubclass(target_input_type, Value)

        # Check type compatibility
        if not source_output_type.can_cast_to(target_input_type):
            raise TypeError(
                f"Edge from {source.id}.{self.source_key} to {target.id}.{self.target_key} "
                f"has invalid types: {source_output_type} is not assignable to {target_input_type}"
            )

    def _resolve_source_type(self, source: Node) -> ValueType:
        """Resolve the Value type at the end of the source path."""
        # Handle simple string path (backwards compatible)
        if isinstance(self.source_key, str):
            if self.source_key not in source.output_fields:
                raise ValueError(
                    f"Source node {source.id} does not have field '{self.source_key}'"
                )
            source_output_type, _ = source.output_fields[self.source_key]
            return source_output_type

        # Handle deep path
        path = self.source_key
        if not isinstance(path, Sequence) or len(path) == 0:
            raise ValueError("Path must be a non-empty string or sequence")

        # First segment must be a field name
        first_segment = path[0]
        if not isinstance(first_segment, str):
            raise ValueError(
                f"First path segment must be a field name (string), got {type(first_segment)}"
            )

        if first_segment not in source.output_fields:
            raise ValueError(
                f"Source node {source.id} does not have field '{first_segment}'"
            )

        # Get the field's type and traverse remaining path
        field_type, _ = source.output_fields[first_segment]

        # If only one segment, return the field type
        if len(path) == 1:
            return field_type

        # Traverse remaining path segments
        return resolve_path_type(field_type, path[1:])

    def _resolve_target_type(self, target: Node) -> ValueType:
        """Resolve the Value type at the end of the target path."""
        # Handle simple string path (backwards compatible)
        if isinstance(self.target_key, str):
            if self.target_key not in target.input_fields:
                raise ValueError(
                    f"Target node {target.id} does not have field '{self.target_key}'"
                )
            target_input_type, _ = target.input_fields[self.target_key]
            return target_input_type

        # Handle deep path
        path = self.target_key
        if not isinstance(path, Sequence) or len(path) == 0:
            raise ValueError("Path must be a non-empty string or sequence")

        # First segment must be a field name
        first_segment = path[0]
        if not isinstance(first_segment, str):
            raise ValueError(
                f"First path segment must be a field name (string), got {type(first_segment)}"
            )

        if first_segment not in target.input_fields:
            raise ValueError(
                f"Target node {target.id} does not have field '{first_segment}'"
            )

        # Get the field's type and traverse remaining path
        field_type, _ = target.input_fields[first_segment]

        # If only one segment, return the field type
        if len(path) == 1:
            return field_type

        # Traverse remaining path segments
        return resolve_path_type(field_type, path[1:])


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

    Supports deep field access for target_key (but not input_key, which is always
    a simple workflow input field name).
    """

    input_key: str  # Always a simple field name
    target_id: str
    target_key: FieldPath  # Can be a deep path
    input_schema: ValueSchema | None = None

    @classmethod
    def from_node(
        cls,
        *,
        input_key: str,
        target: Node,
        target_key: FieldPath,
        input_schema: ValueSchema | None = None,
    ) -> "InputEdge":
        return cls(
            input_key=input_key,
            target_id=target.id,
            target_key=target_key,
            input_schema=input_schema,
        )

    def validate_types(self, input_type: ValueType, target: Node):
        """
        Validate that the input type can be cast to the target's input type.

        For deep target paths, validates the path and resolves the target type.
        """
        # Resolve target type through path
        try:
            target_input_type = self._resolve_target_type(target)
        except (ValueError, TypeError, AttributeError) as e:
            raise ValueError(
                f"Invalid target path {self.target_key} on node {target.id}: {e}"
            ) from e

        assert issubclass(target_input_type, Value)

        if not input_type.can_cast_to(target_input_type):
            raise TypeError(
                f"Input edge to {target.id}.{self.target_key} has invalid types: "
                f"{input_type} is not assignable to {target_input_type}"
            )

    def _resolve_target_type(self, target: Node) -> ValueType:
        """Resolve the Value type at the end of the target path."""
        # Handle simple string path (backwards compatible)
        if isinstance(self.target_key, str):
            if self.target_key not in target.input_fields:
                raise ValueError(
                    f"Target node {target.id} does not have field '{self.target_key}'"
                )
            target_input_type, _ = target.input_fields[self.target_key]
            return target_input_type

        # Handle deep path
        path = self.target_key
        if not isinstance(path, Sequence) or len(path) == 0:
            raise ValueError("Path must be a non-empty string or sequence")

        # First segment must be a field name
        first_segment = path[0]
        if not isinstance(first_segment, str):
            raise ValueError(
                f"First path segment must be a field name (string), got {type(first_segment)}"
            )

        if first_segment not in target.input_fields:
            raise ValueError(
                f"Target node {target.id} does not have field '{first_segment}'"
            )

        # Get the field's type and traverse remaining path
        field_type, _ = target.input_fields[first_segment]

        # If only one segment, return the field type
        if len(path) == 1:
            return field_type

        # Traverse remaining path segments
        return resolve_path_type(field_type, path[1:])


class OutputEdge(ImmutableBaseModel):
    """
    An "edge" that maps a source node's output to a special output of the
    workflow.

    Supports deep field access for source_key (but not output_key, which is always
    a simple workflow output field name).
    """

    source_id: str
    source_key: FieldPath  # Can be a deep path
    output_key: str  # Always a simple field name
    output_schema: ValueSchema | None = None

    @classmethod
    def from_node(
        cls,
        *,
        source: Node,
        source_key: FieldPath,
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
        """
        Validate that the source output type can be cast to the workflow output type.

        For deep source paths, validates the path and resolves the source type.
        """
        # Resolve source type through path
        try:
            source_output_type = self._resolve_source_type(source)
        except (ValueError, TypeError, AttributeError) as e:
            raise ValueError(
                f"Invalid source path {self.source_key} on node {source.id}: {e}"
            ) from e

        assert issubclass(source_output_type, Value)

        if not source_output_type.can_cast_to(output_type):
            raise TypeError(
                f"Output edge from {source.id}.{self.source_key} has invalid types: "
                f"{source_output_type} is not assignable to {output_type}"
            )

    def _resolve_source_type(self, source: Node) -> ValueType:
        """Resolve the Value type at the end of the source path."""
        # Handle simple string path (backwards compatible)
        if isinstance(self.source_key, str):
            if self.source_key not in source.output_fields:
                raise ValueError(
                    f"Source node {source.id} does not have field '{self.source_key}'"
                )
            source_output_type, _ = source.output_fields[self.source_key]
            return source_output_type

        # Handle deep path
        path = self.source_key
        if not isinstance(path, Sequence) or len(path) == 0:
            raise ValueError("Path must be a non-empty string or sequence")

        # First segment must be a field name
        first_segment = path[0]
        if not isinstance(first_segment, str):
            raise ValueError(
                f"First path segment must be a field name (string), got {type(first_segment)}"
            )

        if first_segment not in source.output_fields:
            raise ValueError(
                f"Source node {source.id} does not have field '{first_segment}'"
            )

        # Get the field's type and traverse remaining path
        field_type, _ = source.output_fields[first_segment]

        # If only one segment, return the field type
        if len(path) == 1:
            return field_type

        # Traverse remaining path segments
        return resolve_path_type(field_type, path[1:])


__all__ = [
    "Edge",
    "InputEdge",
    "OutputEdge",
    "SynchronizationEdge",
]
