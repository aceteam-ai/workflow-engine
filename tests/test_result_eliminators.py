"""
Tests for the Seq[Result[T]] eliminators (#202): partition, unwrap_or,
all_ok, first_error.
"""

from typing import cast

import pytest
from pydantic import ValidationError

from workflow_engine.contexts.in_memory import InMemoryExecutionContext
from workflow_engine.core import (
    DataValue,
    Edge,
    ErrorClass,
    ErrorClassValue,
    FloatValue,
    IntegerValue,
    Result,
    ResultError,
    SequenceValue,
    StringValue,
    ValidationContext,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.nodes.result import (
    AllOkData,
    AllOkNode,
    FirstErrorData,
    FirstErrorNode,
    PartitionData,
    PartitionNode,
    UnwrapOrNode,
)


def _error(name: str, *, node_id: str = "node-1") -> ResultError:
    return ResultError(
        error_class=ErrorClassValue(ErrorClass.SYSTEMIC),
        name=StringValue(name),
        message=StringValue(f"{name} failed"),
        node_id=StringValue(node_id),
    )


def _seq(*items: Result) -> SequenceValue:
    return SequenceValue(root=list(items))


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
# partition


@pytest.mark.unit
@pytest.mark.asyncio
async def test_partition_splits_oks_and_errs_with_original_indices(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    element_type = FloatValue
    node = PartitionNode(type="Partition", id="p", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)

    err_a = _error("ErrA")
    err_b = _error("ErrB")
    sequence = _seq(
        Result[element_type].ok(FloatValue(1.0)),
        Result[element_type].err(err_a),
        Result[element_type].ok(FloatValue(3.0)),
        Result[element_type].err(err_b),
    )

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type(sequence=sequence),
    )

    assert [v.root for v in output.oks] == [1.0, 3.0]
    assert [v.root for v in output.ok_indices] == [0, 2]
    assert [e.root.name.root for e in output.errs] == ["ErrA", "ErrB"]
    assert [v.root for v in output.err_indices] == [1, 3]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_partition_indices_reconstruct_original_sequence(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    """
    Non-tautological proof of the index-preservation claim: rebuild the
    original sequence purely from oks/ok_indices/errs/err_indices and check
    it matches the input, rather than just asserting individual fields.
    """
    element_type = FloatValue
    node = PartitionNode(type="Partition", id="p", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)

    original = [
        Result[element_type].ok(FloatValue(10.0)),
        Result[element_type].err(_error("E0")),
        Result[element_type].err(_error("E1")),
        Result[element_type].ok(FloatValue(13.0)),
        Result[element_type].ok(FloatValue(14.0)),
    ]
    sequence = _seq(*original)

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type(sequence=sequence),
    )

    rebuilt: list[Result | None] = [None] * len(original)
    for index, value in zip(output.ok_indices, output.oks):
        rebuilt[index.root] = Result[element_type].ok(value)
    for index, error in zip(output.err_indices, output.errs):
        rebuilt[index.root] = Result[element_type].err(error.root)

    assert all(r is not None for r in rebuilt)
    for original_item, rebuilt_item in zip(original, rebuilt):
        assert rebuilt_item is not None
        assert original_item.is_ok() == rebuilt_item.is_ok()
        if original_item.is_ok():
            assert original_item.unwrap_ok().root == rebuilt_item.unwrap_ok().root
        else:
            assert original_item.unwrap_err().name.root == (
                rebuilt_item.unwrap_err().name.root
            )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_partition_all_ok(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    element_type = FloatValue
    node = PartitionNode(type="Partition", id="p", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)
    sequence = _seq(
        Result[element_type].ok(FloatValue(1.0)),
        Result[element_type].ok(FloatValue(2.0)),
    )

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type(sequence=sequence),
    )

    assert [v.root for v in output.oks] == [1.0, 2.0]
    assert [v.root for v in output.ok_indices] == [0, 1]
    assert len(output.errs) == 0
    assert len(output.err_indices) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_partition_all_err(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    element_type = FloatValue
    node = PartitionNode(type="Partition", id="p", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)
    sequence = _seq(
        Result[element_type].err(_error("E0")),
        Result[element_type].err(_error("E1")),
    )

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type(sequence=sequence),
    )

    assert len(output.oks) == 0
    assert len(output.ok_indices) == 0
    assert [e.root.name.root for e in output.errs] == ["E0", "E1"]
    assert [v.root for v in output.err_indices] == [0, 1]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_partition_empty(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    element_type = FloatValue
    node = PartitionNode(type="Partition", id="p", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type(sequence=_seq()),
    )

    assert len(output.oks) == 0
    assert len(output.ok_indices) == 0
    assert len(output.errs) == 0
    assert len(output.err_indices) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_partition_json_round_trip_preserves_errors(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    element_type = FloatValue
    node = PartitionNode(type="Partition", id="p", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)
    sequence = _seq(
        Result[element_type].ok(FloatValue(1.0)),
        Result[element_type].err(_error("E0")),
    )
    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type(sequence=sequence),
    )

    dumped = output.model_dump(mode="json")
    assert dumped["errs"][0]["name"] == "E0"
    restored = output_type.model_validate(dumped)
    assert restored.errs[0].root.name.root == "E0"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_partition_end_to_end_through_engine(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    """
    Wires PartitionNode behind a GatherSequenceNode, the way it is meant to be
    used in a real graph: fed by whatever upstream node (typically a
    for_each/attempt boundary) produced the Seq[Result[T]], via an ordinary
    inner-node-to-inner-node edge, rather than as a workflow's own external
    input.

    Historical note: this used to also be load-bearing, because
    SequenceValue[Result[T]] could not itself be an InputNode/OutputNode field
    type (#215 -- SequenceValue's to_value_schema() fell back to the generic
    default instead of delegating to Result's own overridden
    to_value_schema()). That gap is fixed now (see
    test_seq_result_output_field_round_trips below); this test still uses the
    inner-graph wiring because that is the realistic shape of
    for_each(attempt(w))'s output feeding an inner eliminator node, not
    because the direct wiring would fail.
    """
    from workflow_engine.nodes.data import (
        GatherSequenceNode,
        SequenceParams,
    )

    element_type = FloatValue
    result_type = Result[element_type]
    input_node = engine.create_input_node(
        element_0=result_type,
        element_1=result_type,
    )
    output_node = engine.create_output_node(
        oks=SequenceValue[element_type],
        ok_indices=SequenceValue[IntegerValue],
        errs=SequenceValue[DataValue[ResultError]],
        err_indices=SequenceValue[IntegerValue],
    )
    gather = engine.create_node(
        GatherSequenceNode,
        id="gather",
        params=SequenceParams(length=IntegerValue(2)),
        element_type=result_type,
    )
    partition = engine.create_node(
        PartitionNode, id="partition", element_type=element_type
    )
    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[gather, partition],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="element_0",
                target=gather,
                target_key="element_0",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="element_1",
                target=gather,
                target_key="element_1",
            ),
            Edge.from_nodes(
                source=gather,
                source_key="sequence",
                target=partition,
                target_key="sequence",
            ),
            *[
                Edge.from_nodes(
                    source=partition,
                    source_key=key,
                    target=output_node,
                    target_key=key,
                )
                for key in ("oks", "ok_indices", "errs", "err_indices")
            ],
        ],
    )

    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={
            "element_0": result_type.ok(FloatValue(1.0)),
            "element_1": result_type.err(_error("E0")),
        },
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    output = cast("dict[str, SequenceValue]", result.output)
    assert [v.root for v in output["oks"]] == [1.0]
    assert [v.root for v in output["ok_indices"]] == [0]
    assert [e.root.name.root for e in output["errs"]] == ["E0"]
    assert [v.root for v in output["err_indices"]] == [1]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_seq_result_output_field_round_trips(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    """
    Regression test for #215: SequenceValue[Result[T]] can now be declared
    directly as an OutputNode field, which is the shape for_each(attempt(w))
    is meant to produce as a workflow's own output.

    GatherSequenceNode still assembles the sequence (that is how
    for_each(attempt(w)) itself works), but here it is wired straight to the
    workflow's OutputNode field instead of through an inner eliminator node.
    Declaring that field forces SequenceValue[Result[FloatValue]] through
    to_value_schema() at node-construction time, and rebuilding the output
    node's dynamic data type forces the schema back through build_value_cls()
    -- both used to fail before generic containers delegated to their inner
    type's to_value_schema().
    """
    from workflow_engine.nodes.data import GatherSequenceNode, SequenceParams

    element_type = FloatValue
    result_type = Result[element_type]

    input_node = engine.create_input_node(
        element_0=result_type,
        element_1=result_type,
    )
    output_node = engine.create_output_node(results=SequenceValue[result_type])
    gather = engine.create_node(
        GatherSequenceNode,
        id="gather",
        params=SequenceParams(length=IntegerValue(2)),
        element_type=result_type,
    )
    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[gather],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="element_0",
                target=gather,
                target_key="element_0",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="element_1",
                target=gather,
                target_key="element_1",
            ),
            Edge.from_nodes(
                source=gather,
                source_key="sequence",
                target=output_node,
                target_key="results",
            ),
        ],
    )

    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={
            "element_0": result_type.ok(FloatValue(1.0)),
            "element_1": result_type.err(_error("E0")),
        },
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    output_seq = cast(SequenceValue, result.output["results"])
    assert output_seq[0].is_ok()
    assert output_seq[0].unwrap_ok() == FloatValue(1.0)
    assert output_seq[1].is_err()
    assert output_seq[1].unwrap_err().name.root == "E0"

    # The dynamic output type is rebuilt from the declared field's schema via
    # DataValueSchema.build_data_cls() -- the exact path that used to raise
    # NotImplementedError. Confirm the rebuilt type itself round-trips a value.
    # (output_type is a dynamically-built Data subclass, so field access goes
    # through getattr rather than a statically-known attribute.)
    validation_context = ValidationContext()
    output_type = await output_node.dynamic_output_type(validation_context)
    dumped = output_type.model_validate({"results": output_seq}).model_dump(mode="json")
    restored = output_type.model_validate(dumped)
    restored_results = cast(SequenceValue, getattr(restored, "results"))
    assert restored_results[0].is_ok()
    assert restored_results[1].is_err()


################################################################################
# unwrap_or


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unwrap_or_substitutes_default_for_errors(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    element_type = FloatValue
    node = UnwrapOrNode(type="UnwrapOr", id="u", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)
    sequence = _seq(
        Result[element_type].ok(FloatValue(1.0)),
        Result[element_type].err(_error("E0")),
        Result[element_type].ok(FloatValue(3.0)),
    )

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type.model_validate(
            {"sequence": sequence, "default": FloatValue(-1.0)}
        ),
    )

    assert [v.root for v in output.sequence] == [1.0, -1.0, 3.0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unwrap_or_all_ok(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    element_type = FloatValue
    node = UnwrapOrNode(type="UnwrapOr", id="u", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)
    sequence = _seq(
        Result[element_type].ok(FloatValue(1.0)),
        Result[element_type].ok(FloatValue(2.0)),
    )

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type.model_validate(
            {"sequence": sequence, "default": FloatValue(-1.0)}
        ),
    )

    assert [v.root for v in output.sequence] == [1.0, 2.0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unwrap_or_all_err(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    element_type = FloatValue
    node = UnwrapOrNode(type="UnwrapOr", id="u", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)
    sequence = _seq(
        Result[element_type].err(_error("E0")),
        Result[element_type].err(_error("E1")),
    )

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type.model_validate(
            {"sequence": sequence, "default": FloatValue(-1.0)}
        ),
    )

    assert [v.root for v in output.sequence] == [-1.0, -1.0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unwrap_or_empty(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    element_type = FloatValue
    node = UnwrapOrNode(type="UnwrapOr", id="u", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type.model_validate(
            {"sequence": _seq(), "default": FloatValue(-1.0)}
        ),
    )

    assert list(output.sequence) == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unwrap_or_migration_shim_for_marker_strings(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    """
    The single-scalar-output case the discussion says this shim is
    behavior-identical for: a marker string substituted for a failed page.
    """
    element_type = StringValue
    node = UnwrapOrNode(type="UnwrapOr", id="u", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)
    sequence = _seq(
        Result[element_type].ok(StringValue("page 1 text")),
        Result[element_type].err(_error("PageFailure")),
        Result[element_type].ok(StringValue("page 3 text")),
    )
    marker = StringValue("[item 2 of 3 failed]")

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type.model_validate({"sequence": sequence, "default": marker}),
    )

    assert [v.root for v in output.sequence] == [
        "page 1 text",
        "[item 2 of 3 failed]",
        "page 3 text",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unwrap_or_requires_default_to_be_wired_explicitly(
    validation_context: ValidationContext,
):
    """
    No built-in default is ever synthesized: omitting the wired `default`
    input fails validation instead of inventing a plausible-looking value.
    """
    element_type = FloatValue
    node = UnwrapOrNode(type="UnwrapOr", id="u", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)

    with pytest.raises(ValidationError):
        input_type.model_validate(
            {"sequence": _seq(Result[element_type].ok(FloatValue(1.0)))}
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unwrap_or_end_to_end_through_engine(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    """
    Same GatherSequenceNode-assembled wiring as
    test_partition_end_to_end_through_engine, for the same realistic-shape
    reason (see that test's docstring; the schema gap it references, #215, is
    now fixed). unwrap_or's own output (a plain Seq[T]) never had that
    problem.
    """
    from workflow_engine.nodes.data import (
        GatherSequenceNode,
        SequenceParams,
    )

    element_type = FloatValue
    result_type = Result[element_type]
    input_node = engine.create_input_node(
        element_0=result_type,
        element_1=result_type,
        default=element_type,
    )
    output_node = engine.create_output_node(sequence=SequenceValue[element_type])
    gather = engine.create_node(
        GatherSequenceNode,
        id="gather",
        params=SequenceParams(length=IntegerValue(2)),
        element_type=result_type,
    )
    unwrap_or = engine.create_node(
        UnwrapOrNode, id="unwrap_or", element_type=element_type
    )
    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[gather, unwrap_or],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="element_0",
                target=gather,
                target_key="element_0",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="element_1",
                target=gather,
                target_key="element_1",
            ),
            Edge.from_nodes(
                source=gather,
                source_key="sequence",
                target=unwrap_or,
                target_key="sequence",
            ),
            Edge.from_nodes(
                source=input_node,
                source_key="default",
                target=unwrap_or,
                target_key="default",
            ),
            Edge.from_nodes(
                source=unwrap_or,
                source_key="sequence",
                target=output_node,
                target_key="sequence",
            ),
        ],
    )

    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={
            "element_0": result_type.ok(FloatValue(1.0)),
            "element_1": result_type.err(_error("E0")),
            "default": FloatValue(-1.0),
        },
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    output_sequence = cast(SequenceValue, result.output["sequence"])
    assert [v.root for v in output_sequence] == [1.0, -1.0]


################################################################################
# all_ok


@pytest.mark.unit
@pytest.mark.asyncio
async def test_all_ok_returns_ok_of_full_sequence(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    element_type = FloatValue
    node = AllOkNode(type="AllOk", id="a", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)
    sequence = _seq(
        Result[element_type].ok(FloatValue(1.0)),
        Result[element_type].ok(FloatValue(2.0)),
    )

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type(sequence=sequence),
    )

    assert output.result.is_ok()
    assert [v.root for v in output.result.unwrap_ok()] == [1.0, 2.0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_all_ok_returns_first_error_when_error_in_middle(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    element_type = FloatValue
    node = AllOkNode(type="AllOk", id="a", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)
    sequence = _seq(
        Result[element_type].ok(FloatValue(1.0)),
        Result[element_type].err(_error("First")),
        Result[element_type].err(_error("Second")),
    )

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type(sequence=sequence),
    )

    assert output.result.is_err()
    assert output.result.unwrap_err().name.root == "First"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_all_ok_all_err(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    element_type = FloatValue
    node = AllOkNode(type="AllOk", id="a", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)
    sequence = _seq(
        Result[element_type].err(_error("First")),
        Result[element_type].err(_error("Second")),
    )

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type(sequence=sequence),
    )

    assert output.result.is_err()
    assert output.result.unwrap_err().name.root == "First"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_all_ok_empty_is_ok_of_empty_sequence(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    """Vacuous truth: no elements means none of them failed."""
    element_type = FloatValue
    node = AllOkNode(type="AllOk", id="a", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type(sequence=_seq()),
    )

    assert output.result.is_ok()
    assert list(output.result.unwrap_ok()) == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_all_ok_json_round_trip():
    element_type = FloatValue
    output_type = AllOkData[element_type]
    value = output_type(result=Result[SequenceValue[element_type]].err(_error("E0")))
    dumped = value.model_dump(mode="json")
    assert dumped["result"] == {
        "tag": "err",
        "err": {
            "error_class": "systemic",
            "name": "E0",
            "message": "E0 failed",
            "node_id": "node-1",
        },
    }
    restored = output_type.model_validate(dumped)
    assert restored.result.is_err()
    assert restored.result.unwrap_err().name.root == "E0"


################################################################################
# first_error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_first_error_finds_first_of_several(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    element_type = FloatValue
    node = FirstErrorNode(type="FirstError", id="f", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.output_type(validation_context)
    sequence = _seq(
        Result[element_type].ok(FloatValue(1.0)),
        Result[element_type].err(_error("First")),
        Result[element_type].err(_error("Second")),
    )

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type(sequence=sequence),
    )

    assert isinstance(output.error, DataValue)
    assert output.error.root.name.root == "First"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_first_error_all_err_returns_index_zero(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    element_type = FloatValue
    node = FirstErrorNode(type="FirstError", id="f", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.output_type(validation_context)
    sequence = _seq(
        Result[element_type].err(_error("First")),
        Result[element_type].err(_error("Second")),
    )

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type(sequence=sequence),
    )

    assert output.error.root.name.root == "First"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_first_error_all_ok_is_none(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    element_type = FloatValue
    node = FirstErrorNode(type="FirstError", id="f", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.output_type(validation_context)
    sequence = _seq(
        Result[element_type].ok(FloatValue(1.0)),
        Result[element_type].ok(FloatValue(2.0)),
    )

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type(sequence=sequence),
    )

    assert output.error.root is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_first_error_empty_is_none(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    element_type = FloatValue
    node = FirstErrorNode(type="FirstError", id="f", element_type=element_type)
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.output_type(validation_context)

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_type(sequence=_seq()),
    )

    assert output.error.root is None


@pytest.mark.unit
def test_first_error_output_type_is_static_and_element_independent():
    """
    Unlike the other three eliminators, first_error's output never depends on
    the element type, since it only ever reports on the err arm.
    """
    node_a = FirstErrorNode(type="FirstError", id="f", element_type=FloatValue)
    node_b = FirstErrorNode(type="FirstError", id="f", element_type=StringValue)
    assert FirstErrorNode.static_output_type() is FirstErrorData
    assert (
        node_a.__class__.static_output_type() is node_b.__class__.static_output_type()
    )


################################################################################
# default element_type=Value must not crash (registry / entry-point loading)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_nodes_resolve_types_with_default_element_type(
    validation_context: ValidationContext,
):
    """
    Nodes are registered generically (see pyproject.toml entry points) with
    no element_type set, i.e. the Field(default=Value) fallback. Resolving
    input/output types in that state must not blow up, the same guarantee
    GatherSequenceNode etc. already provide.
    """
    for node in (
        PartitionNode(type="Partition", id="p"),
        UnwrapOrNode(type="UnwrapOr", id="u"),
        AllOkNode(type="AllOk", id="a"),
        FirstErrorNode(type="FirstError", id="f"),
    ):
        await node.input_type(validation_context)
        await node.output_type(validation_context)


################################################################################
# schema generation for the published Data output shapes (not the full
# to_value_cls() reconstruction: see PR discussion for the pre-existing gap
# in nested Result[T]-inside-Data schema round-tripping, which predates and
# is orthogonal to these nodes).


@pytest.mark.unit
def test_partition_data_schema_generation_and_round_trip():
    from workflow_engine.core.values import get_data_schema, validate_value_schema

    cls = PartitionData[FloatValue]
    schema = get_data_schema(cls)
    dumped = schema.model_dump(by_alias=True)
    parsed = validate_value_schema(dumped)
    rebuilt = parsed.to_value_cls()
    assert issubclass(rebuilt, DataValue)


@pytest.mark.unit
def test_first_error_data_schema_generation_and_round_trip():
    from workflow_engine.core.values import get_data_schema, validate_value_schema

    schema = get_data_schema(FirstErrorData)
    dumped = schema.model_dump(by_alias=True)
    parsed = validate_value_schema(dumped)
    rebuilt = parsed.to_value_cls()
    assert issubclass(rebuilt, DataValue)


@pytest.mark.unit
def test_all_ok_data_schema_generation_does_not_crash():
    """
    Schema *generation* for AllOkData (which nests a Result[T] field) must
    still succeed, even though reconstructing it back from raw JSON Schema
    hits a pre-existing gap unrelated to this change: Data.model_json_schema()
    does not route nested Result[T] fields through Result's own
    to_value_schema(), so the discriminated-union shape it produces isn't one
    validate_value_schema() can rebuild. Result[T] itself still round-trips
    correctly as a standalone value type (see test_result_value.py).
    """
    from workflow_engine.core.values import get_data_schema

    cls = AllOkData[FloatValue]
    schema = get_data_schema(cls)
    assert schema.model_dump(by_alias=True)["title"] == "AllOkData[FloatValue]"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_all_ok_and_first_error_agree_on_which_error_is_first(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    """
    AllOk short-circuits on the first err and FirstError reports the first
    err. Nothing forces those two "firsts" to be the same one, because they
    are separate scans in separate nodes. A graph that branches a sequence
    into both and reports one error while failing on another would be
    confusing in exactly the situation the eliminators exist for.

    Pin the agreement with several errors present, so a node that scanned in
    reverse or returned the last match would fail here rather than passing
    by luck on a single-error input.
    """
    element_type = FloatValue
    sequence = _seq(
        Result[element_type].ok(FloatValue(1.0)),
        Result[element_type].err(_error("First")),
        Result[element_type].ok(FloatValue(2.0)),
        Result[element_type].err(_error("Second")),
        Result[element_type].err(_error("Third")),
    )

    all_ok = AllOkNode(type="AllOk", id="a", element_type=element_type)
    all_ok_input_type = await all_ok.dynamic_input_type(validation_context)
    all_ok_output = await all_ok.run(
        context=context,
        input_type=all_ok_input_type,
        output_type=await all_ok.dynamic_output_type(validation_context),
        input=all_ok_input_type(sequence=sequence),
    )

    first_error = FirstErrorNode(type="FirstError", id="f", element_type=element_type)
    first_error_input_type = await first_error.dynamic_input_type(validation_context)
    first_error_output = await first_error.run(
        context=context,
        input_type=first_error_input_type,
        output_type=await first_error.output_type(validation_context),
        input=first_error_input_type(sequence=sequence),
    )

    assert all_ok_output.result.is_err()
    reported_by_all_ok = all_ok_output.result.unwrap_err()
    reported_by_first_error = first_error_output.error.root

    assert reported_by_first_error is not None
    assert reported_by_all_ok.name.root == "First"
    assert reported_by_first_error.name.root == "First"
    assert reported_by_all_ok == reported_by_first_error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_partition_and_unwrap_or_agree_on_which_elements_failed(
    context: InMemoryExecutionContext,
    validation_context: ValidationContext,
):
    """
    Partition reports err positions via err_indices, and UnwrapOr substitutes
    the default at exactly the positions that failed. Those are two different
    code paths deciding the same thing, and a host that partitions to report
    failures while unwrapping to build its artifact depends on them matching.
    """
    element_type = StringValue
    default = StringValue("[missing]")
    sequence = _seq(
        Result[element_type].ok(StringValue("a")),
        Result[element_type].err(_error("Boom")),
        Result[element_type].ok(StringValue("c")),
        Result[element_type].err(_error("Bang")),
    )

    partition = PartitionNode(type="Partition", id="p", element_type=element_type)
    partition_input_type = await partition.dynamic_input_type(validation_context)
    partition_output = await partition.run(
        context=context,
        input_type=partition_input_type,
        output_type=await partition.dynamic_output_type(validation_context),
        input=partition_input_type(sequence=sequence),
    )

    unwrap = UnwrapOrNode(type="UnwrapOr", id="u", element_type=element_type)
    unwrap_input_type = await unwrap.dynamic_input_type(validation_context)
    unwrap_output = await unwrap.run(
        context=context,
        input_type=unwrap_input_type,
        output_type=await unwrap.dynamic_output_type(validation_context),
        input=unwrap_input_type.model_validate(
            {"sequence": sequence, "default": default}
        ),
    )

    substituted_positions = [
        index
        for index, item in enumerate(unwrap_output.sequence.root)
        if item == default
    ]
    err_positions = [index.root for index in partition_output.err_indices.root]

    assert err_positions == [1, 3]
    assert substituted_positions == err_positions
