import pytest

from workflow_engine import (
    BooleanValue,
    Edge,
    FloatValue,
    Workflow,
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


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


async def _run_comparison(
    engine: WorkflowEngine,
    node_cls: type,
    a: float,
    b: float,
    params: dict | None = None,
) -> bool:
    workflow = Workflow(
        input_node=(input_node := engine.create_input_node(a=FloatValue, b=FloatValue)),
        output_node=(output_node := engine.create_output_node(result=BooleanValue)),
        inner_nodes=[
            cmp := engine.create_node(node_cls, id="cmp", params=params or {})
        ],
        edges=[
            Edge.from_nodes(
                source=input_node, source_key="a", target=cmp, target_key="a"
            ),
            Edge.from_nodes(
                source=input_node, source_key="b", target=cmp, target_key="b"
            ),
            Edge.from_nodes(
                source=cmp,
                source_key="result",
                target=output_node,
                target_key="result",
            ),
        ],
    )
    result = await engine.execute(
        context=InMemoryExecutionContext(),
        workflow=workflow,
        input={"a": FloatValue(a), "b": FloatValue(b)},
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
    engine: WorkflowEngine, node_cls: type, a: float, b: float, expected: bool
):
    assert await _run_comparison(engine, node_cls, a, b) is expected


@pytest.mark.asyncio
async def test_equal_default_is_exact(engine: WorkflowEngine):
    """By default (rel_tol=0, abs_tol=0) Equal is an exact comparison."""
    # 0.1 + 0.2 != 0.3 in binary floating point.
    assert await _run_comparison(engine, EqualNode, 0.1 + 0.2, 0.3) is False
    assert await _run_comparison(engine, NotEqualNode, 0.1 + 0.2, 0.3) is True
    # Large magnitudes that differ by 1 are NOT silently treated as equal.
    assert await _run_comparison(engine, EqualNode, 1e9, 1e9 + 1) is False


@pytest.mark.asyncio
async def test_equal_rel_tol_absorbs_rounding(engine: WorkflowEngine):
    """An explicit rel_tol lets Equal treat 0.1 + 0.2 and 0.3 as equal."""
    params = {"rel_tol": 1e-9}
    assert await _run_comparison(engine, EqualNode, 0.1 + 0.2, 0.3, params) is True
    assert await _run_comparison(engine, NotEqualNode, 0.1 + 0.2, 0.3, params) is False


@pytest.mark.asyncio
async def test_equal_abs_tol_near_zero(engine: WorkflowEngine):
    """abs_tol handles values near zero where rel_tol alone is too strict."""
    params = {"rel_tol": 0.0, "abs_tol": 1e-6}
    assert await _run_comparison(engine, EqualNode, 0.0, 1e-9, params) is True
    assert await _run_comparison(engine, EqualNode, 0.0, 1e-3, params) is False


@pytest.mark.asyncio
async def test_not_node(engine: WorkflowEngine):
    workflow = Workflow(
        input_node=(input_node := engine.create_input_node(a=BooleanValue)),
        output_node=(output_node := engine.create_output_node(result=BooleanValue)),
        inner_nodes=[node := engine.create_node(NotNode, id="not")],
        edges=[
            Edge.from_nodes(
                source=input_node, source_key="a", target=node, target_key="a"
            ),
            Edge.from_nodes(
                source=node,
                source_key="result",
                target=output_node,
                target_key="result",
            ),
        ],
    )
    result = await engine.execute(
        context=InMemoryExecutionContext(),
        workflow=workflow,
        input={"a": BooleanValue(True)},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["result"].root is False


async def _run_variadic_logic(
    engine: WorkflowEngine, node_cls: type, values: list[bool]
) -> bool:
    n = len(values)
    keys = [chr(ord("a") + i) for i in range(n)]
    workflow = Workflow(
        input_node=(
            input_node := engine.create_input_node(**{k: BooleanValue for k in keys})
        ),
        output_node=(output_node := engine.create_output_node(result=BooleanValue)),
        inner_nodes=[
            logic := engine.create_node(
                node_cls, id="logic", params=dict(num_arguments=n)
            )
        ],
        edges=[
            Edge.from_nodes(source=input_node, source_key=k, target=logic, target_key=k)
            for k in keys
        ]
        + [
            Edge.from_nodes(
                source=logic,
                source_key="result",
                target=output_node,
                target_key="result",
            ),
        ],
    )
    result = await engine.execute(
        context=InMemoryExecutionContext(),
        workflow=workflow,
        input={k: BooleanValue(v) for k, v in zip(keys, values)},
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
async def test_and_variadic(engine: WorkflowEngine, values: list[bool], expected: bool):
    assert await _run_variadic_logic(engine, AndNode, values) is expected


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
async def test_or_variadic(engine: WorkflowEngine, values: list[bool], expected: bool):
    assert await _run_variadic_logic(engine, OrNode, values) is expected
