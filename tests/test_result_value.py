"""Tests for Result[T]: a tagged ok/err value type (#200)."""

import pytest
from pydantic import ValidationError

from workflow_engine import (
    Data,
    Edge,
    FloatValue,
    IntegerValue,
    SequenceValue,
    StringValue,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts.in_memory import InMemoryExecutionContext
from workflow_engine.core import (
    ErrorClass,
    ErrorClassValue,
    NullValue,
    Result,
    ResultError,
    ValidationContext,
    Value,
)
from workflow_engine.core.values import validate_value_schema
from workflow_engine.core.values.result import _ErrRoot, _OkRoot
from workflow_engine.core.values.schema import ResultValueSchema
from workflow_engine.nodes.data import (
    ExpandSequenceNode,
    GatherSequenceNode,
    SequenceData,
    SequenceParams,
)


@pytest.fixture
def context() -> InMemoryExecutionContext:
    return InMemoryExecutionContext()


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


def _error(
    *,
    error_class: ErrorClass = ErrorClass.SYSTEMIC,
    name: str = "SomeError",
    message: str = "something went wrong",
    node_id: str = "node-1",
) -> ResultError:
    return ResultError(
        error_class=ErrorClassValue(error_class),
        name=StringValue(name),
        message=StringValue(message),
        node_id=StringValue(node_id),
    )


# --- Construction and identity ---


@pytest.mark.unit
def test_ok_is_tagged_ok():
    value = Result[FloatValue].ok(FloatValue(1.5))
    assert value.is_ok()
    assert not value.is_err()
    assert value.unwrap_ok().root == 1.5
    assert isinstance(value, Result)
    assert isinstance(value, Value)


@pytest.mark.unit
def test_err_is_tagged_err():
    error = _error()
    value = Result[FloatValue].err(error)
    assert value.is_err()
    assert not value.is_ok()
    assert value.unwrap_err() == error


@pytest.mark.unit
def test_unwrap_ok_on_err_raises():
    value = Result[FloatValue].err(_error())
    with pytest.raises(ValueError):
        value.unwrap_ok()


@pytest.mark.unit
def test_unwrap_err_on_ok_raises():
    value = Result[FloatValue].ok(FloatValue(1.0))
    with pytest.raises(ValueError):
        value.unwrap_err()


@pytest.mark.unit
def test_ok_root_rejects_err_tag():
    """
    The root is a discriminated union of two variant models (_OkRoot /
    _ErrRoot), not a single model with two optional sibling fields. That
    structurally rules out a mismatched tag/payload instead of relying on a
    hand-written invariant check.
    """
    with pytest.raises(ValidationError):
        _OkRoot[FloatValue](tag="err", ok=FloatValue(1.0))  # type: ignore[arg-type]


@pytest.mark.unit
def test_ok_root_requires_ok_payload():
    with pytest.raises(ValidationError):
        _OkRoot[FloatValue]()  # type: ignore[call-arg]


@pytest.mark.unit
def test_err_root_requires_err_payload():
    with pytest.raises(ValidationError):
        _ErrRoot()  # type: ignore[call-arg]


@pytest.mark.unit
def test_error_class_is_closed_vocabulary():
    with pytest.raises(ValidationError):
        ErrorClassValue.model_validate("not_a_real_class")


@pytest.mark.unit
def test_ok_of_null_value_round_trips():
    """
    Regression test: an earlier design used Python None as the "this arm is
    absent" sentinel on a single root model, which is indistinguishable from
    a populated `ok: NullValue` payload (NullValue's own wire form is also
    `null`), so Result[NullValue].ok(NullValue(None)) failed to validate back
    in from its own dump. The discriminated-union root fixes this: `tag`
    routes validation before the payload is ever checked against a sentinel.
    """
    value_type = Result[NullValue]
    value = value_type.ok(NullValue(None))
    dumped = value.model_dump(mode="json")
    restored = value_type.model_validate(dumped)
    assert restored.is_ok()
    assert restored.unwrap_ok().root is None


# --- Serialization: the published wire shape ---


@pytest.mark.unit
def test_ok_wire_shape():
    value = Result[FloatValue].ok(FloatValue(2.5))
    dumped = value.model_dump(mode="json")
    assert dumped == {"tag": "ok", "ok": 2.5}


@pytest.mark.unit
def test_err_wire_shape():
    error = _error(
        error_class=ErrorClass.RATE_LIMIT,
        name="RateLimited",
        message="too many requests",
        node_id="fetch-1",
    )
    value = Result[FloatValue].err(error)
    dumped = value.model_dump(mode="json")
    assert dumped == {
        "tag": "err",
        "err": {
            "error_class": "rate_limit",
            "name": "RateLimited",
            "message": "too many requests",
            "node_id": "fetch-1",
        },
    }


# --- Result[Result[T]]: both tags survive serialization ---


@pytest.mark.unit
def test_nested_result_ok_of_ok_round_trips_with_both_tags():
    value_type = Result[Result[FloatValue]]
    value = value_type.ok(Result[FloatValue].ok(FloatValue(9.0)))

    dumped = value.model_dump(mode="json")
    assert dumped == {"tag": "ok", "ok": {"tag": "ok", "ok": 9.0}}

    restored = value_type.model_validate(dumped)
    assert restored.is_ok()
    inner = restored.unwrap_ok()
    assert isinstance(inner, Result)
    assert inner.is_ok()
    assert inner.unwrap_ok().root == 9.0


@pytest.mark.unit
def test_nested_result_ok_of_err_round_trips_with_both_tags():
    value_type = Result[Result[FloatValue]]
    inner_error = _error(name="InnerFailure")
    value = value_type.ok(Result[FloatValue].err(inner_error))

    dumped = value.model_dump(mode="json")
    assert dumped["tag"] == "ok"
    assert dumped["ok"]["tag"] == "err"
    assert dumped["ok"]["err"]["name"] == "InnerFailure"

    restored = value_type.model_validate(dumped)
    assert restored.is_ok()
    inner = restored.unwrap_ok()
    assert inner.is_err()
    assert inner.unwrap_err().name.root == "InnerFailure"


@pytest.mark.unit
def test_nested_result_err_round_trips():
    value_type = Result[Result[FloatValue]]
    outer_error = _error(name="OuterFailure")
    value = value_type.err(outer_error)

    dumped = value.model_dump(mode="json")
    assert dumped == {
        "tag": "err",
        "err": {
            "error_class": "systemic",
            "name": "OuterFailure",
            "message": "something went wrong",
            "node_id": "node-1",
        },
    }
    restored = value_type.model_validate(dumped)
    assert restored.is_err()
    assert restored.unwrap_err().name.root == "OuterFailure"


# --- Schema round-trip: Value type -> schema -> Value type ---


@pytest.mark.unit
def test_result_schema_round_trips():
    value_cls = Result[FloatValue]
    schema = value_cls.to_value_schema()
    rebuilt = schema.to_value_cls()
    assert rebuilt is value_cls


@pytest.mark.unit
def test_nested_result_schema_round_trips():
    value_cls = Result[Result[IntegerValue]]
    schema = value_cls.to_value_schema()
    rebuilt = schema.to_value_cls()
    assert rebuilt is value_cls


def _round_trip_from_raw_json(value_cls: type[Result]) -> type[Value]:
    """
    Value type -> schema -> raw JSON (the wire form a host actually receives)
    -> parsed back through the whole ValueSchema union -> Value type.

    This is the real round-trip: it goes through validate_value_schema()
    against the full ValueSchema union, not a schema object constructed
    directly in Python (which trivially bypasses the union-matching this
    test exists to check).
    """
    raw = value_cls.to_value_schema().model_dump(by_alias=True)
    parsed = validate_value_schema(raw)
    return parsed.to_value_cls()


@pytest.mark.unit
def test_result_schema_round_trips_from_raw_json():
    value_cls = Result[FloatValue]
    parsed = value_cls.to_value_schema().model_dump(by_alias=True)
    # Must be recognized as a ResultValueSchema, not fall through to
    # DataValueSchema or StringMapValueSchema (both also `type: "object"`;
    # BaseValueSchema's extra="allow" would otherwise let `ok`/`err` validate
    # as harmless extras on either of those, silently losing Result's
    # ok/err identity on the way back in).
    assert isinstance(validate_value_schema(parsed), ResultValueSchema)
    assert _round_trip_from_raw_json(value_cls) is value_cls


@pytest.mark.unit
def test_nested_result_schema_round_trips_from_raw_json():
    value_cls = Result[Result[IntegerValue]]
    parsed = value_cls.to_value_schema().model_dump(by_alias=True)
    assert isinstance(validate_value_schema(parsed), ResultValueSchema)
    assert _round_trip_from_raw_json(value_cls) is value_cls


# --- Casting: Result[S] -> Result[T] when S can cast to T ---


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cast_result_ok_arm(context: InMemoryExecutionContext):
    source = Result[IntegerValue].ok(IntegerValue(7))
    assert source.can_cast_to(Result[FloatValue])
    casted = await source.cast_to(Result[FloatValue], context=context)
    assert isinstance(casted, Result)
    assert casted.is_ok()
    assert casted.unwrap_ok().root == 7


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cast_result_err_arm_preserves_error(context: InMemoryExecutionContext):
    error = _error(name="CastError")
    source = Result[IntegerValue].err(error)
    casted = await source.cast_to(Result[FloatValue], context=context)
    assert casted.is_err()
    assert casted.unwrap_err() == error


# --- Gather-side typing: index stability with Result[T] elements ---
#
# The design in discussion #198 claims GatherSequenceNode needs no
# modification for Seq[Result[T]]: dynamic_input_type already builds one
# *required* field per index, and run() does a direct positional lookup, so
# every index stays filled and stable regardless of whether the element at
# that index is ok or err.


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gather_sequence_of_results_requires_every_index(
    context: InMemoryExecutionContext,
):
    element_type = Result[FloatValue]
    node = GatherSequenceNode(
        type="GatherSequence",
        id="gather",
        params=SequenceParams(length=IntegerValue(3)),
        element_type=element_type,
    )

    input_type = await node.dynamic_input_type(ValidationContext())
    fields = input_type.model_fields
    assert set(fields.keys()) == {"element_0", "element_1", "element_2"}
    for field_info in fields.values():
        assert field_info.annotation is element_type
        assert field_info.is_required()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gather_sequence_of_results_preserves_index_and_tags(
    context: InMemoryExecutionContext,
):
    element_type = Result[FloatValue]
    node = GatherSequenceNode(
        type="GatherSequence",
        id="gather",
        params=SequenceParams(length=IntegerValue(3)),
        element_type=element_type,
    )
    validation_context = ValidationContext()
    input_type = await node.dynamic_input_type(validation_context)
    output_type = await node.dynamic_output_type(validation_context)

    error = _error(name="ElementFailure")
    input_data = input_type.model_validate(
        {
            "element_0": element_type.ok(FloatValue(1.0)),
            "element_1": element_type.err(error),
            "element_2": element_type.ok(FloatValue(3.0)),
        }
    )

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_data,
    )

    # No index skipped or shifted: the failed element stays at index 1,
    # tagged err, rather than being absent or forging a fake success value.
    assert len(output.sequence) == 3
    assert output.sequence[0].is_ok()
    assert output.sequence[0].unwrap_ok().root == 1.0
    assert output.sequence[1].is_err()
    assert output.sequence[1].unwrap_err().name.root == "ElementFailure"
    assert output.sequence[2].is_ok()
    assert output.sequence[2].unwrap_ok().root == 3.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_expand_sequence_of_results_preserves_index_and_tags(
    context: InMemoryExecutionContext,
):
    """
    ExpandSequenceNode is the inverse of GatherSequenceNode and makes the same
    claim: a required positional field per index, no absence handling. Checked
    here since docs/values.md mentions it alongside GatherSequenceNode.
    """
    element_type = Result[FloatValue]
    node = ExpandSequenceNode(
        type="ExpandSequence",
        id="expand",
        params=SequenceParams(length=IntegerValue(2)),
        element_type=element_type,
    )
    validation_context = ValidationContext()

    input_type = await node.dynamic_input_type(validation_context)
    assert input_type is SequenceData[element_type]
    output_type = await node.dynamic_output_type(validation_context)

    error = _error(name="ExpandFailure")
    input_data = input_type(
        sequence=SequenceValue[element_type](
            root=[element_type.ok(FloatValue(1.0)), element_type.err(error)]
        )
    )

    output = await node.run(
        context=context,
        input_type=input_type,
        output_type=output_type,
        input=input_data,
    )
    assert getattr(output, node.key(0)).is_ok()
    assert getattr(output, node.key(1)).is_err()
    assert getattr(output, node.key(1)).unwrap_err().name.root == "ExpandFailure"


# --- Acceptance: a graph carrying Result elements validates and executes,
# using only public workflow_engine APIs (no reference to any host). ---


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graph_with_result_elements_validates_and_executes(
    engine: WorkflowEngine,
    context: InMemoryExecutionContext,
):
    element_type = Result[FloatValue]
    input_node = engine.create_input_node(x=element_type)
    output_node = engine.create_output_node(y=element_type)
    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="x",
                target=output_node,
                target_key="y",
            ),
        ],
    )

    ok_result = await engine.execute(
        context=context,
        workflow=workflow,
        input={"x": element_type.ok(FloatValue(4.0))},
    )
    assert ok_result.status is WorkflowExecutionResultStatus.SUCCESS
    ok_output = ok_result.output["y"]
    assert isinstance(ok_output, Result)
    assert ok_output.is_ok()

    err_result = await engine.execute(
        context=context,
        workflow=workflow,
        input={"x": element_type.err(_error(name="GraphFailure"))},
    )
    assert err_result.status is WorkflowExecutionResultStatus.SUCCESS
    err_output = err_result.output["y"]
    assert isinstance(err_output, Result)
    assert err_output.is_err()
    assert err_output.unwrap_err().name.root == "GraphFailure"


@pytest.mark.unit
def test_result_field_on_data_class():
    """Result[T] is usable as an ordinary Data field, like any other Value type."""

    class Item(Data):
        outcome: Result[StringValue]

    item = Item(outcome=Result[StringValue].ok(StringValue("done")))
    assert item.outcome.is_ok()
