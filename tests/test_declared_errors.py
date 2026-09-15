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


class AlwaysRetriesNode(Node[Empty, Empty, Empty]):
    """Always raises an exhausted-immediately ShouldRetry with an explicit name."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="AlwaysRetries",
        description="Always raises ShouldRetry with an explicit name.",
        version="1.0.0",
        parameter_type=Empty,
        max_retries=0,
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
        from workflow_engine import ShouldRetry

        raise ShouldRetry.for_user(
            "still throttled",
            node=self,
            error_class=ErrorClass.RATE_LIMIT,
            name="Throttled",
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


class UpstreamRateLimited(NodeException):
    """
    A subclass with an author-selected class name. Subclassing alone is not
    the name= channel (#248): raising this with no explicit name= carries no
    special weight in result_error_from_exception, it is just another
    NodeException. See test_subclassing_alone_is_not_the_name_channel.
    """


class ChainedDeclaredErrorProbeNode(UndeclaredErrorProbeNode):
    """Raises `from` a TimeoutError, with an explicit name= on the outer exception."""

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[Empty],
        input: Empty,
    ) -> Empty:
        try:
            raise TimeoutError("provider transport details")
        except TimeoutError as cause:
            raise NodeException.for_user(
                "Provider throttled the request",
                node=self,
                error_class=ErrorClass.RATE_LIMIT,
                name="UpstreamRateLimited",
            ) from cause


@pytest.mark.asyncio
async def test_explicit_name_survives_raise_from_end_to_end(algorithm):
    """
    Design acceptance criterion #3 on #248: this is the case that silently
    regressed on the Option 2 implementation this branch replaces, an
    explicit name= surviving `raise ... from ...` through an attempt
    boundary. error_class, message, and node_id all still come from the
    raised exception itself.
    """
    engine = WorkflowEngine(execution_algorithm=algorithm)
    inner = Workflow(
        input_node=engine.create_input_node(),
        inner_nodes=[engine.create_node(ChainedDeclaredErrorProbeNode, id="provider")],
        output_node=engine.create_output_node(),
        edges=[],
    )
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=AttemptNode,
        input={},
        params={"workflow": inner},
    )
    value = result.output["result"]
    assert isinstance(value, Result)
    error = value.unwrap_err()
    assert error.name.root == "UpstreamRateLimited"
    assert error.error_class.root == ErrorClass.RATE_LIMIT
    assert error.message.root == "Provider throttled the request"
    assert error.node_id.root == "node/provider"


@pytest.mark.unit
def test_subclassing_alone_is_not_the_name_channel():
    """
    Design acceptance criterion #4 on #248: a concrete WorkflowException
    subclass, raised with no name=, is not itself a name. With a cause it
    still yields the root cause's type name (the non-regression case);
    without a cause it falls back to its own class name, same as any other
    unnamed exception.
    """
    from workflow_engine.execution.boundary import result_error_from_exception

    probe = WorkflowEngine().create_node(UndeclaredErrorProbeNode, id="probe")

    with_cause = UpstreamRateLimited.for_user("rate limit", node=probe)
    with_cause.__cause__ = TimeoutError("transport")
    assert result_error_from_exception(with_cause).name.root == "TimeoutError"

    without_cause = UpstreamRateLimited.for_user("rate limit", node=probe)
    assert result_error_from_exception(without_cause).name.root == "UpstreamRateLimited"


@pytest.mark.unit
def test_generic_wrappers_retain_root_diagnostic_name():
    """
    Diagnostic quality does not regress (design acceptance criterion #5): an
    unclassified failure with no name= anywhere in the chain still surfaces
    the deepest cause's type name, exactly as before this feature.
    """
    from workflow_engine import WorkflowException
    from workflow_engine.execution.boundary import result_error_from_exception

    try:
        try:
            raise ValueError("private provider detail")
        except ValueError as cause:
            raise WorkflowException.for_operator("wrapper") from cause
    except WorkflowException as cause:
        exc = NodeException.for_operator(
            "outer wrapper",
            node=WorkflowEngine().create_node(UndeclaredErrorProbeNode, id="probe"),
        )
        exc.__cause__ = cause
    error = result_error_from_exception(exc)
    assert error.name.root == "ValueError"
    assert error.error_class.root == ErrorClass.SYSTEMIC
    assert error.message.root == "An internal error occurred"


@pytest.mark.unit
def test_chain_precedence_outward_in():
    """
    Design acceptance criterion #6 on #248, a direct unit test on
    result_error_from_exception with hand-built chains. The resolver takes
    the first explicit name= walking the __cause__ chain outward-in (from
    the raised exception toward the root): an inner name wins over no outer
    name, and an outer name wins over an inner one. With nothing named
    anywhere, it falls back to the root cause's type name.

    Reverting the resolver to inward-out (innermost name wins) makes the
    second assertion below fail: it would return "InnerFailure" instead of
    "OuterFailure".
    """
    from workflow_engine.execution.boundary import result_error_from_exception

    probe = WorkflowEngine().create_node(UndeclaredErrorProbeNode, id="probe")

    def build_chain(*, outer_name: str | None, inner_name: str | None) -> NodeException:
        inner = NodeException.for_user(
            "inner failure",
            node=probe,
            error_class=ErrorClass.RATE_LIMIT,
            name=inner_name,
        )
        inner.__cause__ = TimeoutError("transport")
        outer = NodeException.for_user(
            "outer failure",
            node=probe,
            error_class=ErrorClass.SYSTEMIC,
            name=outer_name,
        )
        outer.__cause__ = inner
        return outer

    # Inner named, outer unnamed: the inner name wins, there is nothing else.
    chain = build_chain(outer_name=None, inner_name="InnerFailure")
    assert result_error_from_exception(chain).name.root == "InnerFailure"

    # Both named: the outer name wins (outward-in precedence).
    chain = build_chain(outer_name="OuterFailure", inner_name="InnerFailure")
    assert result_error_from_exception(chain).name.root == "OuterFailure"

    # Nothing named: falls back to the root cause's type name, unchanged
    # from before this feature.
    chain = build_chain(outer_name=None, inner_name=None)
    assert result_error_from_exception(chain).name.root == "TimeoutError"


@pytest.mark.unit
def test_cause_cycle_is_visited_once():
    """
    A self-referential __cause__ (which a raise site should never construct
    deliberately, but which the resolver must not loop on) terminates at the
    cyclic exception's own class name rather than hanging.
    """
    from workflow_engine import WorkflowException
    from workflow_engine.execution.boundary import result_error_from_exception

    cyclic = WorkflowException.for_user("cycle", node_id="probe")
    cyclic.__cause__ = cyclic
    assert result_error_from_exception(cyclic).name.root == "WorkflowException"


@pytest.mark.unit
def test_empty_name_raises_value_error():
    """
    name="" is rejected at construction (design acceptance criterion #1):
    an author who wants no explicit name omits the keyword (None) rather
    than passing an empty string, which could otherwise silently coerce
    into a name= that is never actually meaningful.
    """
    from workflow_engine import WorkflowException

    probe = WorkflowEngine().create_node(UndeclaredErrorProbeNode, id="probe")
    with pytest.raises(ValueError):
        WorkflowException.for_user("boom", node_id="probe", name="")
    with pytest.raises(ValueError):
        NodeException.for_user("boom", node=probe, name="")


@pytest.mark.unit
def test_should_retry_threads_explicit_name():
    """
    ShouldRetry accepts name= directly and through for_user, the same
    channel as WorkflowException and NodeException (design acceptance
    criterion #7).
    """
    from workflow_engine import ShouldRetry

    probe = WorkflowEngine().create_node(UndeclaredErrorProbeNode, id="probe")
    exc = ShouldRetry.for_user("throttled", node=probe, name="Throttled")
    assert exc.name == "Throttled"
    with pytest.raises(ValueError):
        ShouldRetry.for_user("throttled", node=probe, name="")


@pytest.mark.asyncio
async def test_should_retry_exhaustion_preserves_explicit_name(algorithm):
    """
    Design acceptance criterion #7 end to end: retry exhaustion surfaces the
    ShouldRetry itself (`failure = e` in both executors), so it needs the
    name= channel too, not just WorkflowException/NodeException. This is one
    of the two regressions the rejected Option 2 implementation reproduced:
    on that branch this materialized name == "ShouldRetry".
    """
    engine = WorkflowEngine(execution_algorithm=algorithm)
    inner = Workflow(
        input_node=engine.create_input_node(),
        inner_nodes=[engine.create_node(AlwaysRetriesNode, id="retrier")],
        output_node=engine.create_output_node(),
        edges=[],
    )
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=AttemptNode,
        input={},
        params={"workflow": inner},
    )
    value = result.output["result"]
    assert isinstance(value, Result)
    error = value.unwrap_err()
    assert error.name.root == "Throttled"
    assert error.error_class.root == ErrorClass.RATE_LIMIT


@pytest.mark.asyncio
async def test_error_node_uses_declared_name_not_message_smuggling(algorithm):
    """
    ErrorNode (nodes/error.py) is the one node with an author-chosen error
    name that is a runtime parameter, not a class (design acceptance
    criterion #10). Its error_name now reaches the wire through name=
    rather than being prefixed onto message.
    """
    from workflow_engine.nodes import ErrorNode

    engine = WorkflowEngine(execution_algorithm=algorithm)
    inner = await engine.build_single_node_workflow(
        ErrorNode, params={"error_name": "CustomFailure"}
    )
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=AttemptNode,
        input={"info": "something went wrong"},
        params={"workflow": inner},
    )
    value = result.output["result"]
    assert isinstance(value, Result)
    error = value.unwrap_err()
    assert error.name.root == "CustomFailure"
    assert error.message.root == "something went wrong"
    assert "CustomFailure" not in error.message.root


@pytest.mark.asyncio
async def test_error_node_empty_name_falls_back_without_operator_error(algorithm):
    """
    An empty error_name is a valid StringValue in a stored graph. It must
    still materialize a normal err arm via the resolver's own fallback, not
    turn into an "Unhandled exception" operator error from the name=""
    ValueError guard (design acceptance criterion #10).
    """
    from workflow_engine.nodes import ErrorNode

    engine = WorkflowEngine(execution_algorithm=algorithm)
    inner = await engine.build_single_node_workflow(
        ErrorNode, params={"error_name": ""}
    )
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=AttemptNode,
        input={"info": "something went wrong"},
        params={"workflow": inner},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.errors.count == 0
    value = result.output["result"]
    assert isinstance(value, Result)
    error = value.unwrap_err()
    assert error.name.root == "WorkflowException"
    assert error.message.root == "something went wrong"


@pytest.mark.asyncio
async def test_factorization_emitted_errors_match_declaration(algorithm):
    from workflow_engine import IntegerValue
    from workflow_engine.nodes import FactorizationNode

    engine = WorkflowEngine(execution_algorithm=algorithm)
    inner = await engine.build_single_node_workflow(FactorizationNode)
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=AttemptNode,
        input={"value": IntegerValue(0)},
        params={"workflow": inner},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    value = result.output["result"]
    assert isinstance(value, Result)
    error = value.unwrap_err()
    declarations = {d.name: d for d in FactorizationNode.TYPE_INFO.declared_errors}
    assert error.name.root == "InvalidFactorizationInput"
    assert declarations[error.name.root].error_class == error.error_class.root
