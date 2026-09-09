# tests/test_declared_errors.py
"""
Tests for declaring emitted error names on `NodeTypeInfo` (#237).

`declared_errors` is the surviving half of the per-node-error-types proposal
#234 otherwise rejects: `error_class` stays the closed vocabulary routing and
retry key on, and `name` becomes the open per-node vocabulary a node type can
document for a downstream author (e.g. for a branch node's dropdown).

Two properties are load-bearing and each has its own test below:

- Documentation-level, never wire-enforced. `ResultError`'s schema does not
  change, and a `name` absent from a node's declared list is still a valid
  `ResultError.name` on the wire (`test_undeclared_error_name_still_...`).
- Non-exhaustive by construction: the field is optional, and adding a name to
  it is not a schema change (`test_node_without_declared_errors_...`).
"""

from typing import ClassVar, Type

import pytest
from overrides import override

from workflow_engine import (
    DeclaredError,
    Empty,
    ErrorClass,
    ExecutionContext,
    Node,
    NodeException,
    NodeTypeInfo,
    Result,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core.values import ResultError
from workflow_engine.nodes import AttemptNode

# ---------------------------------------------------------------------------
# Test helper nodes.
# ---------------------------------------------------------------------------


class DeclaredErrorsProbeNode(Node[Empty, Empty, Empty]):
    """A node type that documents two error names it may raise."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="DeclaredErrorsProbe",
        description="Test helper that declares the error names it may raise.",
        version="1.0.0",
        parameter_type=Empty,
        declared_errors=[
            DeclaredError(
                name="not_found",
                error_class=ErrorClass.VALIDATION,
                description="The requested item was not found.",
            ),
            DeclaredError(
                name="upstream_timeout",
                error_class=ErrorClass.TIMEOUT,
                description="The upstream service did not respond in time.",
            ),
        ],
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> Type[Empty]:
        return Empty

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[Empty],
        input: Empty,
    ) -> Empty:
        return output_type()


class NoDeclaredErrorsProbeNode(Node[Empty, Empty, Empty]):
    """A node type that declares nothing; the field is optional."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="NoDeclaredErrorsProbe",
        description="Test helper that declares no error names.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> Type[Empty]:
        return Empty

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[Empty],
        input: Empty,
    ) -> Empty:
        return output_type()


class UndeclaredErrorProbeNode(Node[Empty, Empty, Empty]):
    """
    Declares only 'declared_only', but always raises a plain `NodeException`.

    `result_error_from_exception` (`execution/boundary.py`) materializes a
    plain exception's `ResultError.name` as `type(cause).__name__`, i.e.
    'NodeException' here, not the declared name. This mismatch is exactly
    the gap non-enforcement exists to allow: nothing here rebinds a raised
    error to something on the declared list, nor rejects it for not being on
    it.
    """

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="UndeclaredErrorProbe",
        description="Test helper that raises an error name absent from its declaration.",
        version="1.0.0",
        parameter_type=Empty,
        declared_errors=[
            DeclaredError(
                name="declared_only",
                error_class=ErrorClass.VALIDATION,
                description="A name declared here but never actually raised.",
            ),
        ],
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[Empty]:
        return Empty

    @classmethod
    @override
    def static_output_type(cls) -> Type[Empty]:
        return Empty

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[Empty],
        input: Empty,
    ) -> Empty:
        raise NodeException.for_user(
            "boom",
            node=self,
            error_class=ErrorClass.VALIDATION,
        )


# ---------------------------------------------------------------------------
# 1. Publication.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_declared_errors_publish_in_type_info():
    """A node type with declared errors publishes them in its type info."""
    declared = DeclaredErrorsProbeNode.TYPE_INFO.declared_errors
    assert list(declared) == [
        DeclaredError(
            name="not_found",
            error_class=ErrorClass.VALIDATION,
            description="The requested item was not found.",
        ),
        DeclaredError(
            name="upstream_timeout",
            error_class=ErrorClass.TIMEOUT,
            description="The upstream service did not respond in time.",
        ),
    ]
    dumped = DeclaredErrorsProbeNode.TYPE_INFO.model_dump(mode="json")[
        "declared_errors"
    ]
    assert dumped == [
        {
            "name": "not_found",
            "error_class": "validation",
            "description": "The requested item was not found.",
        },
        {
            "name": "upstream_timeout",
            "error_class": "timeout",
            "description": "The upstream service did not respond in time.",
        },
    ]


# ---------------------------------------------------------------------------
# 2. Optionality.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_declared_errors_field_itself_defaults_to_empty():
    """
    `NodeTypeInfo.declared_errors` is optional at the model level, not just
    via the `from_parameter_type` convenience constructor: constructing
    `NodeTypeInfo` directly with no `declared_errors` kwarg must not raise.
    """
    info = NodeTypeInfo(display_name="Direct", description=None, version="1.0.0")
    assert info.declared_errors == ()


@pytest.mark.asyncio
async def test_node_without_declared_errors_still_validates_and_runs():
    """A node with no declarations still validates and runs; the field is optional."""
    assert NoDeclaredErrorsProbeNode.TYPE_INFO.declared_errors == ()
    assert (
        NoDeclaredErrorsProbeNode.TYPE_INFO.model_dump(mode="json")["declared_errors"]
        == []
    )

    engine = WorkflowEngine()
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=NoDeclaredErrorsProbeNode,
        input={},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.errors.count == 0


# ---------------------------------------------------------------------------
# 3. Non-enforcement (the guard on property 1).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_undeclared_error_name_still_produces_valid_result_on_wire():
    """
    A node raising an undeclared name still produces a valid `Result` on the
    wire. This is the test that pins the non-enforcement decision: nothing
    about `declared_errors` changes `ResultError`'s schema or rejects a
    `name` absent from the declared list, either at validation time or at
    the wire.
    """
    # ResultError's own shape is unperturbed by this feature.
    assert set(ResultError.model_fields) == {
        "error_class",
        "name",
        "message",
        "node_id",
    }

    engine = WorkflowEngine()
    inner = Workflow(
        input_node=engine.create_input_node(),
        inner_nodes=[
            engine.create_node(UndeclaredErrorProbeNode, id="undeclared"),
        ],
        output_node=engine.create_output_node(),
        edges=[],
    )
    workflow = await engine.build_single_node_workflow(
        AttemptNode,
        node_id="attempt",
        params={"workflow": inner},
    )
    result = await engine.execute(
        context=InMemoryExecutionContext(),
        workflow=workflow,
        input={},
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.errors.count == 0
    result_value = result.output["result"]
    assert isinstance(result_value, Result)
    assert result_value.is_err()
    error = result_value.unwrap_err()

    declared_names = {
        d.name for d in UndeclaredErrorProbeNode.TYPE_INFO.declared_errors
    }
    assert error.name.root == "NodeException"
    assert error.name.root not in declared_names

    # The err arm still round-trips through the wire (JSON mode) unharmed:
    # non-enforcement means this name was never a candidate for rejection.
    result_type = type(result_value)
    dumped = result_value.model_dump(mode="json")
    revalidated = result_type.model_validate(dumped)
    assert revalidated.is_err()
    assert revalidated.unwrap_err().name.root == error.name.root
    assert revalidated.unwrap_err().node_id.root == error.node_id.root
