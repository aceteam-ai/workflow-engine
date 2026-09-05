# workflow_engine/nodes/attempt.py
"""
``attempt``: runs an inner workflow inside an error boundary, producing a
``Result[B]`` instead of letting a member's failure propagate to the run.

See discussion #198 for the motivation, #201 for the design, and
``schema/attempt.md`` for the published wire shape, the flat id table, the
reserved ``ok`` id, and the full boundary semantics (drain, not kill; yield
wins over err within a pass; the innermost boundary catches a nested
failure). ``core/boundary.py`` defines the ``ErrorBoundaryNode`` marker this
node implements, and ``execution/boundary.py`` gives it meaning during
execution.
"""

from typing import ClassVar, Type

from overrides import override
from pydantic import Field, PrivateAttr
from pydantic.fields import FieldInfo

from ..core import (
    Data,
    DataMapping,
    DataValue,
    Edge,
    Empty,
    ErrorBoundaryNode,
    ExecutionContext,
    Node,
    NodeException,
    NodeTypeInfo,
    Params,
    Result,
    ResultError,
    ValidatedWorkflow,
    ValidationContext,
    Workflow,
    WorkflowValue,
)
from ..core.values import build_data_type, get_data_dict
from ..core.values.data import get_field_annotations
from .data import single_field_or_wrapped

_RESERVED_OK_ID = "ok"


class AttemptParams(Params):
    workflow: WorkflowValue = Field(
        title="Workflow",
        description="The workflow to run inside the error boundary.",
    )


class AttemptNode(ErrorBoundaryNode, Node[Data, Data, AttemptParams]):
    """
    Runs the inner workflow inside an error boundary, producing ``Result[B]``.

    Wrapping this in the standard input/output nodes gives the workflow
    ``A -> Result[B]``: ``ok`` of the inner workflow's output if every member
    of the boundary succeeds (or yields and later succeeds on resume), or
    ``err`` of the first failure once every member has settled and none is
    suspended.

    This node's own ``run`` is not inside the boundary it creates: it only
    builds a graph from ``params``. A shape error in the inner workflow
    itself (e.g. a dangling edge) surfaces at outer validation time
    (``Workflow.validate``), which is an authoring error, not a runtime
    failure for the boundary to catch.
    """

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Attempt",
        description="Runs the inner workflow inside an error boundary, producing Result.",
        version="1.0.0",
        parameter_type=AttemptParams,
    )

    _workflow: ValidatedWorkflow | None = PrivateAttr(default=None)

    async def workflow(self, context: ValidationContext) -> ValidatedWorkflow:
        if self._workflow is None:
            self._workflow = await self.params.workflow.root.validate(context=context)
        return self._workflow

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> Type[Data]:
        w = await self.workflow(context)
        return w.input_type

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> Type[Data]:
        w = await self.workflow(context)
        b_type = single_field_or_wrapped(w.output_type)
        return build_data_type(
            name="AttemptOutput",
            fields={
                "result": (
                    Result[b_type],
                    FieldInfo(
                        title="Result",
                        description=(
                            "Ok of the inner workflow's output, or err of "
                            "the first failure once the boundary settles."
                        ),
                    ),
                ),
            },
        )

    @override
    def materialize_error(
        self, *, output_type: type[Data], error: ResultError
    ) -> DataMapping:
        result_type = get_field_annotations(output_type)["result"]
        assert issubclass(result_type, Result)
        return {"result": result_type.err(error)}

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Data],
        output_type: Type[Data],
        input: Data,
    ) -> Workflow:
        w = await self.workflow(context.validation_context)

        reserved_collision = _RESERVED_OK_ID in {node.id for node in w.inner_nodes} | {
            w.input_node.id,
            w.output_node.id,
        }
        if reserved_collision:
            raise NodeException.for_builder(
                f"The inner workflow of attempt node '{self.id}' has a node "
                f"with the reserved id '{_RESERVED_OK_ID}'; rename it.",
                node=self,
            )

        b_type = single_field_or_wrapped(w.output_type)
        node_registry = context.validation_context.node_registry
        ok_node = node_registry.create_node(
            OkNode,
            id=_RESERVED_OK_ID,
            data_type=w.output_type,
        )
        output_node = node_registry.create_output_node(result=Result[b_type])

        edges: list[Edge] = []
        for edge in w.edges:
            if edge.target_id == w.output_node.id:
                edges.append(
                    Edge(
                        source_id=edge.source_id,
                        source_key=edge.source_key,
                        target_id=ok_node.id,
                        target_key=edge.target_key,
                    )
                )
            else:
                edges.append(edge)
        edges.append(
            Edge.from_nodes(
                source=ok_node,
                source_key="result",
                target=output_node,
                target_key="result",
            )
        )

        return Workflow(
            input_node=w.input_node,
            inner_nodes=[*w.inner_nodes, ok_node],
            output_node=output_node,
            edges=edges,
        )


class OkNode(Node[Data, Data, Empty]):
    """
    Wraps an inner workflow's output as the ok arm of a ``Result``.

    Programmatically constructed by ``AttemptNode.run``, the same way
    ``GatherDataNode`` is constructed by its callers: ``data_type`` is only
    available when built in code, not from the wire.
    """

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Ok",
        description="Wraps a value as the ok arm of a Result.",
        version="1.0.0",
        parameter_type=Empty,
    )

    # The type of the wrapped data. For now, this field is only available
    # when the node is constructed programmatically (see nodes/data.py for
    # the same TODO on GatherDataNode).
    data_type: Type[Data] = Field(default=Data, exclude=True)

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> Type[Data]:
        return self.data_type

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> Type[Data]:
        b_type = single_field_or_wrapped(self.data_type)
        return build_data_type(
            name="OkOutput",
            fields={
                "result": (
                    Result[b_type],
                    FieldInfo(
                        title="Result",
                        description="Ok of the wrapped value.",
                    ),
                ),
            },
        )

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Data],
        output_type: Type[Data],
        input: Data,
    ) -> Data:
        b_type = single_field_or_wrapped(self.data_type)
        result_type = Result[b_type]
        fields = get_field_annotations(self.data_type)
        if len(fields) == 1:
            (key,) = fields.keys()
            value = get_data_dict(input)[key]
            return output_type(**{"result": result_type.ok(value)})
        return output_type(
            **{"result": result_type.ok(DataValue[self.data_type](root=input))}
        )


__all__ = [
    "AttemptNode",
    "AttemptParams",
    "OkNode",
]
