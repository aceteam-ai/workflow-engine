"""A union must retain its exact member validators and published constraints."""

import pytest
from pydantic import ValidationError

from workflow_engine import (
    Edge,
    IntegerValue,
    NullValue,
    StringValue,
    ValueSchemaValue,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core.values.union import union_value_type


@pytest.mark.parametrize(
    ("base", "schema", "good", "bad", "constraint", "expected"),
    [
        (IntegerValue, {"type": "integer", "minimum": 1}, 1, 0, "minimum", 1),
        (
            StringValue,
            {"type": "string", "pattern": "^yes$"},
            "yes",
            "no",
            "pattern",
            "^yes$",
        ),
        (
            StringValue,
            {"type": "string", "enum": ["yes"]},
            "yes",
            "no",
            "enum",
            ["yes"],
        ),
    ],
)
def test_plain_cached_union_cannot_replace_constrained_member(
    base, schema, good, bad, constraint, expected
):
    plain = union_value_type(base, NullValue)
    bounded = ValueSchemaValue.model_validate(schema).root.to_value_cls()
    assert bounded.__name__ == base.__name__
    constrained = union_value_type(bounded, NullValue)

    with pytest.raises(ValidationError):
        constrained.model_validate(bad)
    assert constrained.model_validate(good).root == good
    assert constrained.model_validate(None).root is None
    assert constrained is not plain
    published = constrained.model_json_schema()
    assert any(
        definition.get(constraint) == expected
        for definition in published.get("$defs", {}).values()
    )
    assert union_value_type(bounded, NullValue) is constrained


@pytest.mark.parametrize("bounds", [(2, 5), (5, 2)])
def test_distinct_same_name_constraints_do_not_contaminate_each_other(bounds):
    first_bound, second_bound = bounds
    first_member = ValueSchemaValue.model_validate(
        {"type": "integer", "minimum": first_bound}
    ).root.to_value_cls()
    second_member = ValueSchemaValue.model_validate(
        {"type": "integer", "minimum": second_bound}
    ).root.to_value_cls()
    first = union_value_type(first_member, NullValue)
    second = union_value_type(second_member, NullValue)

    for union, bound in ((first, first_bound), (second, second_bound)):
        with pytest.raises(ValidationError):
            union.model_validate(bound - 1)
        assert union.model_validate(bound).root == bound
        assert union.model_validate(None).root is None
        assert any(
            definition.get("minimum") == bound
            for definition in union.model_json_schema().get("$defs", {}).values()
        )
    assert first is not second


def test_equivalent_constraints_reuse_the_member_class():
    schema = {"type": "integer", "minimum": 3}
    first = ValueSchemaValue.model_validate(schema).root.to_value_cls()
    second = ValueSchemaValue.model_validate(schema).root.to_value_cls()
    assert first is second
    assert union_value_type(first, NullValue) is union_value_type(second, NullValue)


@pytest.mark.asyncio
async def test_cast_cache_does_not_reuse_a_weaker_union_target():
    context = InMemoryExecutionContext()
    loose = ValueSchemaValue.model_validate(
        {"type": "integer", "minimum": 1}
    ).root.to_value_cls()
    strict = ValueSchemaValue.model_validate(
        {"type": "integer", "minimum": 5}
    ).root.to_value_cls()
    value = loose.model_validate(2)
    loose_target = union_value_type(loose, NullValue)
    strict_target = union_value_type(strict, NullValue)
    assert (await value.cast_to(loose_target, context=context)).root == 2
    with pytest.raises((ValueError, ValidationError)):
        await value.cast_to(strict_target, context=context)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, 3])
async def test_equivalent_constrained_unions_cross_a_real_output_edge(algorithm, value):
    schema = {"anyOf": [{"type": "integer", "minimum": 3}, {"type": "null"}]}
    source_type = ValueSchemaValue.model_validate(schema).root.to_value_cls()
    target_type = ValueSchemaValue.model_validate(schema).root.to_value_cls()
    engine = WorkflowEngine(execution_algorithm=algorithm)
    input_node = engine.create_input_node(finding=source_type)
    output_node = engine.create_output_node(finding=target_type)
    workflow = Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="finding",
                target=output_node,
                target_key="finding",
            )
        ],
    )
    result = await engine.execute(
        context=InMemoryExecutionContext(), workflow=workflow, input={"finding": value}
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["finding"].root == value


def test_cached_constraint_metadata_is_an_immutable_snapshot():
    schema = ValueSchemaValue.model_validate(
        {"type": "integer", "allOf": [{"minimum": 3}]}
    )
    bounded = schema.root.to_value_cls()
    assert schema.root.model_extra is not None
    schema.root.model_extra["allOf"][0]["minimum"] = 0
    published = bounded.model_json_schema()
    assert published["allOf"] == [{"minimum": 3}]
    with pytest.raises(ValidationError):
        bounded.model_validate(0)
