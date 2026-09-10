# tests/test_error_class_raise_sites.py
"""
Tests for #269: `error_class` populated at raise sites, not left to default.

The trap this file exists to avoid (see #269 and the pre-existing
`TestShouldRetryErrorClass` in `test_retry.py`) is asserting the mechanism
works by handing a raise site `error_class=ErrorClass.X` and reading it back:
that only proves the field round-trips, not that any real code path ever
sets it. Every test below drives an actual failing code path (a real
`Node.__call__`, a real `Value.cast_to`, a real `LocalContext.read`/`write`,
or a real workflow through a real `AttemptNode` boundary) and observes
whatever `error_class` that path produces.
"""

import os

import pytest

from tests.test_attempt import _build_attempt_workflow, _run, as_result, edge
from workflow_engine import (
    ErrorClass,
    File,
    WorkflowEngine,
    WorkflowException,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.contexts.local import LocalContext
from workflow_engine.core.error import NodeException
from workflow_engine.core.values import FloatValue, JSONValue, SequenceValue
from workflow_engine.files.csv import CSVFileValue
from workflow_engine.files.text import TextFileValue
from workflow_engine.nodes import ConstantIntegerNode, DivideNode, ErrorNode


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


@pytest.fixture
def context() -> InMemoryExecutionContext:
    return InMemoryExecutionContext()


# ---------------------------------------------------------------------------
# node.py: Node._cast_input, driven through the real Node.__call__ dispatch.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unknown_input_field_is_validation(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    """
    node.py's ``_cast_input`` rejects a DataMapping key the node's input type
    does not declare. Called directly against ``Node.__call__`` (the same
    dispatch ``WorkflowEngine.execute_node`` uses internally) with a bogus
    extra key, bypassing the graph-level edge validation that would normally
    catch this earlier, to exercise the runtime check itself.
    """
    node = engine.create_node(DivideNode, id="node")
    validation_context = context.validation_context
    input_type = await node.input_type(validation_context)
    output_type = await node.output_type(validation_context)

    with pytest.raises(NodeException) as exc_info:
        await node(
            context=context,
            input_type=input_type,
            output_type=output_type,
            input={
                "dividend": FloatValue(1.0),
                "divisor": FloatValue(2.0),
                "bogus": FloatValue(1.0),
            },
        )

    assert exc_info.value.error_class == ErrorClass.VALIDATION


@pytest.mark.unit
@pytest.mark.asyncio
async def test_input_not_assignable_is_validation(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    """
    node.py's ``_cast_input`` rejects a value whose type structurally cannot
    cast to the declared input field type (``Value.can_cast_to`` is False).
    A sequence value has no caster to a scalar ``FloatValue``.
    """
    node = engine.create_node(DivideNode, id="node")
    validation_context = context.validation_context
    input_type = await node.input_type(validation_context)
    output_type = await node.output_type(validation_context)

    assert not SequenceValue[FloatValue].can_cast_to(FloatValue)

    with pytest.raises(NodeException) as exc_info:
        await node(
            context=context,
            input_type=input_type,
            output_type=output_type,
            input={
                "dividend": SequenceValue[FloatValue]([FloatValue(1.0)]),
                "divisor": FloatValue(2.0),
            },
        )

    assert exc_info.value.error_class == ErrorClass.VALIDATION


# ---------------------------------------------------------------------------
# files/csv.py: cast-time shape checks, driven through the real Value.cast_to.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_csv_cast_of_non_mapping_sequence_is_validation(
    context: InMemoryExecutionContext,
):
    """
    A JSON scalar cannot be cast to CSV: it is neither a mapping nor a
    sequence of rows. Exercises the real ``json_to_csv`` caster registered on
    ``JSONValue``, not a hand-built exception.
    """
    value = JSONValue(42)
    assert value.can_cast_to(CSVFileValue)

    with pytest.raises(WorkflowException) as exc_info:
        await value.cast_to(CSVFileValue, context=context)

    assert exc_info.value.error_class == ErrorClass.VALIDATION


# ---------------------------------------------------------------------------
# contexts/local.py: LocalContext.read/write, driven through a real OS-level
# permission failure (chmod), not a stubbed exception.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_context_read_permission_denied_is_permission(tmp_path):
    ctx = LocalContext(base_dir=str(tmp_path))
    file_value = TextFileValue(File(path="secret.txt"))
    path = ctx.get_file_path("secret.txt")
    with open(path, "w") as f:
        f.write("hello")
    os.chmod(path, 0o000)
    try:
        with pytest.raises(WorkflowException) as exc_info:
            await ctx.read(file_value)
        assert exc_info.value.error_class == ErrorClass.PERMISSION
    finally:
        # Restore permissions so pytest's tmp_path cleanup can remove the file.
        os.chmod(path, 0o644)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_context_write_permission_denied_is_permission(tmp_path):
    ctx = LocalContext(base_dir=str(tmp_path))
    os.chmod(ctx.files_dir, 0o555)
    try:
        with pytest.raises(WorkflowException) as exc_info:
            await ctx.write(TextFileValue(File(path="new.txt")), b"data")
        assert exc_info.value.error_class == ErrorClass.PERMISSION
    finally:
        # Restore permissions so pytest's tmp_path cleanup can remove the dir.
        os.chmod(ctx.files_dir, 0o755)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_context_read_missing_file_is_unclassified(tmp_path):
    """
    Contrast case: a missing file (as opposed to a permission failure) has no
    single knowable cause (bad reference vs. deleted-out-from-under-us vs.
    storage issue), so it is left unclassified and materializes as systemic.
    This pins that we deliberately did *not* guess a class here.
    """
    ctx = LocalContext(base_dir=str(tmp_path))
    file_value = TextFileValue(File(path="does-not-exist.txt"))
    with pytest.raises(WorkflowException) as exc_info:
        await ctx.read(file_value)
    assert exc_info.value.error_class is None


# ---------------------------------------------------------------------------
# nodes/error.py: ErrorNode, the deliberately-unclassified baseline.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_node_unset_class_materializes_as_systemic(
    engine: WorkflowEngine, context: InMemoryExecutionContext
):
    """
    ``ErrorNode`` raises with an author-supplied, arbitrary string the engine
    cannot classify, and deliberately leaves ``error_class`` unset. Pins that
    this stays honestly unclassified (systemic at the boundary) rather than
    guessing.
    """
    result = await engine.execute_node(
        context=context,
        node=ErrorNode,
        input={"info": "boom"},
        params={"error_name": "CustomFailure"},
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR
    error = result.errors.node_errors["node"][0]
    assert error is not None
    assert error.error_class is None


# ---------------------------------------------------------------------------
# Routes a real classified failure through a real AttemptNode boundary, to
# confirm the class survives materialization into the wire-level Result
# (`result_error_from_exception`), which is what a retry policy actually
# reads -- not just what the raised exception object carries.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validation_failure_survives_attempt_boundary_materialization(
    engine: WorkflowEngine,
):
    dividend = engine.create_node(
        ConstantIntegerNode, id="dividend", params=dict(value=1)
    )
    divisor = engine.create_node(
        ConstantIntegerNode, id="divisor", params=dict(value=0)
    )
    divide = engine.create_node(DivideNode, id="divide")
    workflow = await _build_attempt_workflow(
        engine,
        inner_nodes=[dividend, divisor, divide],
        edges=[
            edge("dividend", "value", "divide", "dividend"),
            edge("divisor", "value", "divide", "divisor"),
            edge("divide", "quotient", "output", "final"),
        ],
        output_fields={"final": FloatValue},
    )
    context = InMemoryExecutionContext()
    result = await _run(engine, workflow, context)

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    error = as_result(result.output["result"]).unwrap_err()
    assert error.error_class.root == ErrorClass.VALIDATION


# ---------------------------------------------------------------------------
# The property #269 actually cares about: the engine's own raise sites reach
# more than the historical {validation, systemic} pair.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_observed_error_classes_exceed_validation_and_systemic(
    engine: WorkflowEngine, context: InMemoryExecutionContext, tmp_path
):
    """
    Drives three independent real failing paths and collects whatever
    ``error_class`` each one actually produced, rather than asserting a
    single class in isolation. Before #269, every raise site in ``src/``
    that set ``error_class`` set it to ``validation`` or ``systemic``; this
    pins that the reachable set is now strictly larger.
    """
    observed: set[ErrorClass | None] = set()

    # A validation failure (arithmetic node, real user input).
    divide_result = await engine.execute_node(
        context=context,
        node=DivideNode,
        input={"dividend": 1.0, "divisor": 0.0},
    )
    assert divide_result.status is WorkflowExecutionResultStatus.ERROR
    divide_error = divide_result.errors.node_errors["node"][0]
    assert divide_error is not None
    observed.add(divide_error.error_class)

    # An unclassified failure (ErrorNode, deliberately left unset).
    error_result = await engine.execute_node(
        context=context,
        node=ErrorNode,
        input={"info": "boom"},
        params={"error_name": "CustomFailure"},
    )
    assert error_result.status is WorkflowExecutionResultStatus.ERROR
    error_node_error = error_result.errors.node_errors["node"][0]
    assert error_node_error is not None
    observed.add(error_node_error.error_class)

    # A permission failure (LocalContext, real chmod).
    ctx = LocalContext(base_dir=str(tmp_path))
    path = ctx.get_file_path("secret.txt")
    with open(path, "w") as f:
        f.write("hello")
    os.chmod(path, 0o000)
    try:
        with pytest.raises(WorkflowException) as exc_info:
            await ctx.read(TextFileValue(File(path="secret.txt")))
        observed.add(exc_info.value.error_class)
    finally:
        os.chmod(path, 0o644)

    assert observed == {ErrorClass.VALIDATION, None, ErrorClass.PERMISSION}
    assert observed - {ErrorClass.VALIDATION, ErrorClass.SYSTEMIC, None}, (
        "expected at least one class reachable from a real raise site beyond "
        "{validation, systemic}"
    )
