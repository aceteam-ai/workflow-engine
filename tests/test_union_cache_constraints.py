"""A union must retain its exact member validators and published constraints."""

import pytest
from pydantic import ValidationError

from workflow_engine import IntegerValue, NullValue, StringValue, ValueSchemaValue
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
