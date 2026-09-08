# tests/test_match_error_class.py
"""
Tests for MatchErrorClassNode (#236): a conditional keyed on error_class,
exhaustive over every ErrorClass value, checked at graph validation.

error_class is a closed, engine-owned vocabulary (core/error.py), which is
what makes an exhaustiveness check legitimate here and not for a per-node
error type; see #234 for the epic framing.

test_seventh_error_class_value_fails_validation is the test that pins the
intended, deliberately annoying behavior: extending ErrorClass must break
every stored graph that branches on it until a human updates each one, not
silently route the new class down whatever branch happens to be the
fallback. It monkeypatches the ErrorClass name inside
workflow_engine.nodes.conditional to a wider stand-in enum rather than
extending the real ErrorClass, which cannot gain a member after class
definition; this proves the exhaustiveness check reads its required set
from ErrorClass at call time instead of a hardcoded list of "the six
values," which is exactly the thing that would let it silently weaken the
day a real seventh value is added.
"""

import re
from enum import StrEnum

import pytest

import workflow_engine.nodes.conditional as conditional_module
from workflow_engine import (
    Edge,
    ErrorClass,
    ErrorClassValue,
    IntegerValue,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.nodes import ConstantIntegerNode
from workflow_engine.nodes.conditional import MatchErrorClassNode


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


def _constant_branch(engine: WorkflowEngine, value: int) -> Workflow:
    """A sub-workflow with no input that always outputs `value`."""
    return Workflow(
        input_node=engine.create_input_node(),
        output_node=(output_node := engine.create_output_node(result=IntegerValue)),
        inner_nodes=[
            const := engine.create_node(
                ConstantIntegerNode,
                id="const",
                params=dict(value=value),
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=const,
                source_key="value",
                target=output_node,
                target_key="result",
            ),
        ],
    )


def _all_branches(engine: WorkflowEngine) -> dict[str, Workflow]:
    """One distinct constant-output branch per real ErrorClass value."""
    return {
        error_class.value: _constant_branch(engine, index)
        for index, error_class in enumerate(ErrorClass, start=1)
    }


def _build_workflow(engine: WorkflowEngine, branches: dict[str, Workflow]) -> Workflow:
    match_node = engine.create_node(
        MatchErrorClassNode,
        id="route_error",
        params=dict(branches=branches),
    )
    return Workflow(
        input_node=(
            input_node := engine.create_input_node(error_class=ErrorClassValue)
        ),
        output_node=(output_node := engine.create_output_node(result=IntegerValue)),
        inner_nodes=[match_node],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="error_class",
                target=match_node,
                target_key="error_class",
            ),
            Edge.from_nodes(
                source=match_node,
                source_key="result",
                target=output_node,
                target_key="result",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_exhaustive_conditional_validates(engine: WorkflowEngine):
    workflow = _build_workflow(engine, _all_branches(engine))
    validated = await engine.validate(workflow)
    assert validated is not None


def _missing_values_named(message: str) -> str:
    """
    Extracts the exact comma-separated list of missing values the error
    message names, so a test can assert on precisely what is reported
    missing rather than merely that some substring appears somewhere in the
    message (which a message naming the wrong value, or every value, would
    also satisfy).
    """
    match = re.search(r"value\(s\): ([^.]*)\. Every ErrorClass", message)
    assert match is not None, (
        f"message did not contain a missing-values list: {message!r}"
    )
    return match.group(1)


@pytest.mark.asyncio
async def test_missing_branch_fails_validation_naming_node_and_value(
    engine: WorkflowEngine,
):
    branches = _all_branches(engine)
    del branches["permission"]
    workflow = _build_workflow(engine, branches)

    with pytest.raises(ValueError) as exc_info:
        await engine.validate(workflow)

    message = str(exc_info.value)
    assert "route_error" in message
    assert _missing_values_named(message) == "permission"


@pytest.mark.asyncio
async def test_two_missing_branches_are_both_named(engine: WorkflowEngine):
    branches = _all_branches(engine)
    del branches["permission"]
    del branches["timeout"]
    workflow = _build_workflow(engine, branches)

    with pytest.raises(ValueError) as exc_info:
        await engine.validate(workflow)

    message = str(exc_info.value)
    # ErrorClass order, not deletion order: timeout precedes permission.
    assert _missing_values_named(message) == "timeout, permission"


@pytest.mark.asyncio
async def test_unrecognized_branch_key_fails_validation(engine: WorkflowEngine):
    branches = _all_branches(engine)
    branches["not_a_real_error_class"] = branches.pop("systemic")
    branches["systemic"] = _constant_branch(engine, 99)
    workflow = _build_workflow(engine, branches)

    with pytest.raises(ValueError, match="not_a_real_error_class"):
        await engine.validate(workflow)


@pytest.mark.asyncio
async def test_seventh_error_class_value_fails_validation(
    engine: WorkflowEngine,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Pins the intended-consequence behavior from #236: a MatchErrorClass node
    that fully covers today's six ErrorClass values must fail validation the
    moment ErrorClass gains a seventh, until a human adds the missing
    branch. Without this test, the exhaustiveness check could silently
    degrade into checking a hardcoded list of "the six values" instead of
    ErrorClass itself, and nothing here would catch that regression.
    """

    class WidenedErrorClass(StrEnum):
        TIMEOUT = "timeout"
        UNREACHABLE = "unreachable"
        RATE_LIMIT = "rate_limit"
        VALIDATION = "validation"
        PERMISSION = "permission"
        SYSTEMIC = "systemic"
        QUOTA_EXCEEDED = "quota_exceeded"

    workflow = _build_workflow(engine, _all_branches(engine))

    # Fully exhaustive under the real, six-value ErrorClass.
    await engine.validate(workflow)

    monkeypatch.setattr(conditional_module, "ErrorClass", WidenedErrorClass)

    with pytest.raises(ValueError) as exc_info:
        await engine.validate(workflow)

    message = str(exc_info.value)
    assert "route_error" in message
    assert _missing_values_named(message) == "quota_exceeded"


@pytest.mark.asyncio
async def test_each_branch_routes_the_matching_class_at_execution_time(
    engine: WorkflowEngine,
):
    workflow = _build_workflow(engine, _all_branches(engine))
    context = InMemoryExecutionContext()

    for index, error_class in enumerate(ErrorClass, start=1):
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={"error_class": error_class.value},
        )
        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.output == {"result": index}
