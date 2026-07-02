import pytest

from workflow_engine import (
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.nodes import (
    AndNode,
    EqualNode,
    GreaterThanEqualNode,
    GreaterThanNode,
    LessThanEqualNode,
    LessThanNode,
    NotEqualNode,
    NotNode,
    OrNode,
)
from workflow_engine.nodes.comparison import _argument_field_name


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


@pytest.fixture
def context() -> InMemoryExecutionContext:
    return InMemoryExecutionContext()


async def _comparison_result(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
    node_cls: type,
    a: float,
    b: float,
    params: dict | None = None,
) -> bool:
    result = await engine.execute_node(
        context=context,
        node=node_cls,
        params=params,
        input={"a": a, "b": b},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    return result.output["result"].root


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node_cls, a, b, expected",
    [
        (EqualNode, 2.0, 2.0, True),
        (EqualNode, 2.0, 3.0, False),
        (NotEqualNode, 2.0, 3.0, True),
        (NotEqualNode, 2.0, 2.0, False),
        (GreaterThanNode, 3.0, 2.0, True),
        (GreaterThanNode, 2.0, 2.0, False),
        (GreaterThanEqualNode, 2.0, 2.0, True),
        (GreaterThanEqualNode, 1.0, 2.0, False),
        (LessThanNode, 1.0, 2.0, True),
        (LessThanNode, 2.0, 2.0, False),
        (LessThanEqualNode, 2.0, 2.0, True),
        (LessThanEqualNode, 3.0, 2.0, False),
    ],
)
async def test_comparison_nodes(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
    node_cls: type,
    a: float,
    b: float,
    expected: bool,
):
    assert await _comparison_result(engine, context, node_cls, a, b) is expected


@pytest.mark.asyncio
async def test_equal_default_is_exact(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    """By default (rel_tol=0, abs_tol=0) Equal is an exact comparison."""
    # 0.1 + 0.2 != 0.3 in binary floating point.
    assert await _comparison_result(engine, context, EqualNode, 0.1 + 0.2, 0.3) is False
    assert (
        await _comparison_result(engine, context, NotEqualNode, 0.1 + 0.2, 0.3) is True
    )
    # Large magnitudes that differ by 1 are NOT silently treated as equal.
    assert await _comparison_result(engine, context, EqualNode, 1e9, 1e9 + 1) is False


@pytest.mark.asyncio
async def test_equal_rel_tol_absorbs_rounding(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    """An explicit rel_tol lets Equal treat 0.1 + 0.2 and 0.3 as equal."""
    params = {"rel_tol": 1e-9}
    assert (
        await _comparison_result(engine, context, EqualNode, 0.1 + 0.2, 0.3, params)
        is True
    )
    assert (
        await _comparison_result(engine, context, NotEqualNode, 0.1 + 0.2, 0.3, params)
        is False
    )


@pytest.mark.asyncio
async def test_equal_abs_tol_near_zero(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    """abs_tol handles values near zero where rel_tol alone is too strict."""
    params = {"rel_tol": 0.0, "abs_tol": 1e-6}
    assert (
        await _comparison_result(engine, context, EqualNode, 0.0, 1e-9, params) is True
    )
    assert (
        await _comparison_result(engine, context, EqualNode, 0.0, 1e-3, params) is False
    )


@pytest.mark.asyncio
async def test_not_node(engine: WorkflowEngine, context: InMemoryExecutionContext):
    result = await engine.execute_node(
        context=context,
        node=NotNode,
        input={"a": True},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["result"].root is False


async def _variadic_logic_result(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
    node_cls: type,
    values: list[bool],
) -> bool:
    keys = [_argument_field_name(i) for i in range(len(values))]
    result = await engine.execute_node(
        context=context,
        node=node_cls,
        params={"num_arguments": len(values)},
        input=dict(zip(keys, values)),
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    return result.output["result"].root


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "values, expected",
    [
        ([True, True], True),
        ([True, False], False),
        ([True, True, True], True),
        ([True, True, False], False),
    ],
)
async def test_and_variadic(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
    values: list[bool],
    expected: bool,
):
    assert await _variadic_logic_result(engine, context, AndNode, values) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "values, expected",
    [
        ([False, False], False),
        ([False, True], True),
        ([False, False, False], False),
        ([False, False, True], True),
    ],
)
async def test_or_variadic(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
    values: list[bool],
    expected: bool,
):
    assert await _variadic_logic_result(engine, context, OrNode, values) is expected
