# tests/test_result_element_eliminators.py
"""
Tests for the element-level Result[T] eliminators (#235): unwrap,
unwrap_or (scalar), and the is_ok tag branch.

#232 made Result[T] assignable only to Result, which left a lone attempt
outside a sequence with no legal downstream node; these three nodes unstrand
it. See #234 for the epic and core/values/result.py's PropagatedResultError
for the provenance mechanism unwrap depends on to keep a re-raised err arm's
root cause attributed to the node that actually produced it, not to unwrap
itself.
"""

from typing import cast

import pytest

from workflow_engine.contexts.in_memory import InMemoryExecutionContext
from workflow_engine.core import (
    Edge,
    ErrorClass,
    ErrorClassValue,
    Result,
    ResultError,
    StringValue,
    ValidationContext,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.core.values import get_data_dict
from workflow_engine.nodes import AttemptNode, ConstantStringNode, IfElseNode
from workflow_engine.nodes.result import (
    IsOkNode,
    IsOkOutput,
    UnwrapNode,
    UnwrapOrValueNode,
)


def _value(data, key: str) -> StringValue:
    """
    Read a field off a build_data_type-constructed Data instance.

    UnwrapNode/UnwrapOrValueNode's dynamic input/output types are built with
    build_data_type and typed as the base Data class, so pyright doesn't see
    their fields statically; get_data_dict() is the same escape hatch their
    own run() implementations use.
    """
    value = get_data_dict(data)[key]
    assert isinstance(value, StringValue)
    return value


def _error(name: str, *, node_id: str = "node-1") -> ResultError:
    return ResultError(
        error_class=ErrorClassValue(ErrorClass.SYSTEMIC),
        name=StringValue(name),
        message=StringValue(f"{name} failed"),
        node_id=StringValue(node_id),
    )


def edge(source_id: str, source_key: str, target_id: str, target_key: str) -> Edge:
    return Edge(
        source_id=source_id,
        source_key=source_key,
        target_id=target_id,
        target_key=target_key,
    )


@pytest.fixture
def context() -> InMemoryExecutionContext:
    return InMemoryExecutionContext()


@pytest.fixture
def validation_context() -> ValidationContext:
    return ValidationContext()


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


################################################################################
# unwrap


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unwrap_ok_returns_the_payload(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    node = UnwrapNode(type="Unwrap", id="u", element_type=StringValue)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type.model_validate(
            {"result": Result[StringValue].ok(StringValue("hi"))}
        ),
    )

    assert _value(output, "value").root == "hi"


@pytest.mark.asyncio
async def test_unwrap_err_inside_attempt_preserves_root_provenance(
    engine: WorkflowEngine,
):
    """
    The test that matters (#235). A naive re-raise from unwrap's run() (e.g.
    ``raise WorkflowException.for_user(original.message.root, ...)``) makes
    ``Node.__call__`` stamp the exception's node_id to unwrap's own id, so
    the enclosing boundary would record unwrap, a bookkeeping node, as the
    failing node instead of the thing that actually failed upstream. This
    checks the materialized err carries the *original* error_class, name,
    message and node_id through unchanged.

    The original ResultError is hand-built and fed directly as the attempt's
    input (rather than produced by a genuinely failing upstream node), so its
    node_id ("somewhere/else/entirely") is deliberately not a real id in this
    graph: only a correct pass-through reconstructs it.
    """
    unwrap = engine.create_node(UnwrapNode, id="unwrap", element_type=StringValue)
    inner = Workflow(
        input_node=engine.create_input_node(result=Result[StringValue]),
        inner_nodes=[unwrap],
        output_node=engine.create_output_node(value=StringValue),
        edges=[
            edge("input", "result", "unwrap", "result"),
            edge("unwrap", "value", "output", "value"),
        ],
    )
    attempted = await engine.build_single_node_workflow(
        AttemptNode, node_id="attempt", params={"workflow": inner}
    )

    original = ResultError(
        error_class=ErrorClassValue(ErrorClass.RATE_LIMIT),
        name=StringValue("UpstreamRateLimited"),
        message=StringValue("upstream said slow down"),
        node_id=StringValue("somewhere/else/entirely"),
    )

    context = InMemoryExecutionContext()
    result = await engine.execute(
        context=context,
        workflow=attempted,
        input={"result": Result[StringValue].err(original)},
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    outer = cast(Result, result.output["result"])
    assert outer.is_err()
    got = outer.unwrap_err()

    assert got.error_class.root == original.error_class.root
    assert got.name.root == original.name.root
    assert got.message.root == original.message.root
    assert got.node_id.root == original.node_id.root
    # Legible negative: the bug this test guards against would attribute the
    # error to the bookkeeping node instead of the root cause.
    assert got.node_id.root != "attempt/unwrap"


@pytest.mark.asyncio
async def test_unwrap_err_outside_boundary_fails_the_run(engine: WorkflowEngine):
    unwrap = engine.create_node(UnwrapNode, id="unwrap", element_type=StringValue)
    workflow = Workflow(
        input_node=engine.create_input_node(result=Result[StringValue]),
        inner_nodes=[unwrap],
        output_node=engine.create_output_node(value=StringValue),
        edges=[
            edge("input", "result", "unwrap", "result"),
            edge("unwrap", "value", "output", "value"),
        ],
    )

    context = InMemoryExecutionContext()
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={"result": Result[StringValue].err(_error("Boom"))},
    )

    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert "unwrap" in result.errors.node_errors


################################################################################
# unwrap_or (scalar)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unwrap_or_value_returns_default_on_err(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    node = UnwrapOrValueNode(type="UnwrapOrValue", id="u", element_type=StringValue)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type.model_validate(
            {
                "result": Result[StringValue].err(_error("Boom")),
                "default": StringValue("fallback"),
            }
        ),
    )

    assert _value(output, "value").root == "fallback"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unwrap_or_value_returns_payload_on_ok(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    node = UnwrapOrValueNode(type="UnwrapOrValue", id="u", element_type=StringValue)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type.model_validate(
            {
                "result": Result[StringValue].ok(StringValue("real")),
                "default": StringValue("fallback"),
            }
        ),
    )

    assert _value(output, "value").root == "real"


################################################################################
# is_ok (tag branch)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_is_ok_true_for_ok(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    node = IsOkNode(type="IsOk", id="i", element_type=StringValue)
    input_type = await node.dynamic_input_type(validation_context)

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=IsOkOutput,
        input=input_type(result=Result[StringValue].ok(StringValue("x"))),
    )

    assert output.is_ok.root is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_is_ok_false_for_err(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    node = IsOkNode(type="IsOk", id="i", element_type=StringValue)
    input_type = await node.dynamic_input_type(validation_context)

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=IsOkOutput,
        input=input_type(result=Result[StringValue].err(_error("Boom"))),
    )

    assert output.is_ok.root is False


@pytest.mark.asyncio
async def test_is_ok_routes_both_arms_via_ifelse(engine: WorkflowEngine):
    """
    The tag branch routes both arms: is_ok feeding IfElse's condition picks
    the ok-arm inner workflow when the tag is ok, and the err-arm inner
    workflow when it is err.
    """

    def _branch_workflow(label: str) -> Workflow:
        const = engine.create_node(
            ConstantStringNode, id="c", params={"value": StringValue(label)}
        )
        return Workflow(
            input_node=engine.create_input_node(),
            inner_nodes=[const],
            output_node=engine.create_output_node(branch=StringValue),
            edges=[edge("c", "value", "output", "branch")],
        )

    is_ok = engine.create_node(IsOkNode, id="is_ok", element_type=StringValue)
    if_else = engine.create_node(
        IfElseNode,
        id="branch",
        params={
            "if_true": _branch_workflow("ok-branch"),
            "if_false": _branch_workflow("err-branch"),
        },
    )
    workflow = Workflow(
        input_node=engine.create_input_node(result=Result[StringValue]),
        inner_nodes=[is_ok, if_else],
        output_node=engine.create_output_node(branch=StringValue),
        edges=[
            edge("input", "result", "is_ok", "result"),
            edge("is_ok", "is_ok", "branch", "condition"),
            edge("branch", "branch", "output", "branch"),
        ],
    )

    ok_result = await engine.execute(
        context=InMemoryExecutionContext(),
        workflow=workflow,
        input={"result": Result[StringValue].ok(StringValue("x"))},
    )
    assert ok_result.status is WorkflowExecutionResultStatus.SUCCESS
    assert cast(StringValue, ok_result.output["branch"]).root == "ok-branch"

    err_result = await engine.execute(
        context=InMemoryExecutionContext(),
        workflow=workflow,
        input={"result": Result[StringValue].err(_error("Boom"))},
    )
    assert err_result.status is WorkflowExecutionResultStatus.SUCCESS
    assert cast(StringValue, err_result.output["branch"]).root == "err-branch"


################################################################################
# attempt then unwrap, end to end


@pytest.mark.asyncio
async def test_attempt_then_unwrap_validates_and_executes_end_to_end(
    engine: WorkflowEngine,
):
    const = engine.create_node(
        ConstantStringNode, id="c", params={"value": StringValue("done")}
    )
    inner = Workflow(
        input_node=engine.create_input_node(),
        inner_nodes=[const],
        output_node=engine.create_output_node(final=StringValue),
        edges=[edge("c", "value", "output", "final")],
    )
    attempt = engine.create_node(AttemptNode, id="attempt", params={"workflow": inner})
    unwrap = engine.create_node(UnwrapNode, id="unwrap", element_type=StringValue)
    workflow = Workflow(
        input_node=engine.create_input_node(),
        inner_nodes=[attempt, unwrap],
        output_node=engine.create_output_node(value=StringValue),
        edges=[
            edge("attempt", "result", "unwrap", "result"),
            edge("unwrap", "value", "output", "value"),
        ],
    )

    # Validates: the Result[T]-typed edge from attempt to unwrap type-checks
    # now that unwrap exists as a legal consumer.
    validated = await engine.validate(workflow)

    result = await engine.execute(
        context=InMemoryExecutionContext(), workflow=validated, input={}
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert cast(StringValue, result.output["value"]).root == "done"
