"""ForEach closure capture with portable, validated constant input names."""

import pytest

from tests.test_sequence_ops import edge, run_roundtrip, workflow
from workflow_engine import ExecutionAlgorithm, FloatValue, Workflow, WorkflowEngine
from workflow_engine.nodes import AddNode, ForEachNode

pytestmark = pytest.mark.integration


@pytest.fixture
def engine(algorithm: ExecutionAlgorithm) -> WorkflowEngine:
    return WorkflowEngine(execution_algorithm=algorithm)


def sum_step(engine: WorkflowEngine, *, multi_item=False) -> Workflow:
    nodes = [engine.create_node(AddNode, id="add")]
    edges = [edge("input", "acc", "add", "a"), edge("input", "item", "add", "b")]
    inputs = {"acc": FloatValue, "item": FloatValue}
    if multi_item:
        inputs["extra"] = FloatValue
        nodes.append(engine.create_node(AddNode, id="add_extra"))
        edges.extend(
            [
                edge("add", "sum", "add_extra", "a"),
                edge("input", "extra", "add_extra", "b"),
                edge("add_extra", "sum", "output", "acc"),
            ]
        )
    else:
        edges.append(edge("add", "sum", "output", "acc"))
    return workflow(engine, inputs, {"acc": FloatValue}, nodes, edges)


@pytest.mark.parametrize("constant_inputs", [["acc"], ["acc", "extra"]])
async def test_traverse_broadcast_roundtrip(engine, constant_inputs):
    inner = sum_step(engine, multi_item=len(constant_inputs) == 2)
    inputs = {"sequence": [1, 2], "acc": 10}
    if len(constant_inputs) == 2:
        inputs["extra"] = 20
    expected = [31.0, 32.0] if len(constant_inputs) == 2 else [11.0, 12.0]
    assert await run_roundtrip(
        engine,
        ForEachNode,
        {"workflow": inner, "constant_inputs": constant_inputs},
        inputs,
    ) == {"sequence": expected}


async def test_traverse_broadcast_with_record_items(engine):
    assert await run_roundtrip(
        engine,
        ForEachNode,
        {"workflow": sum_step(engine, multi_item=True), "constant_inputs": ["acc"]},
        {"sequence": [{"item": 1, "extra": 2}, {"item": 3, "extra": 4}], "acc": 10},
    ) == {"sequence": [13.0, 17.0]}


@pytest.mark.parametrize(
    "names,message",
    [(["missing"], "Unknown"), (["acc", "item"], "vary"), (["acc", "acc"], "unique")],
)
async def test_traverse_rejects_invalid_constants(engine, names, message):
    with pytest.raises(ValueError, match=message):
        await engine.build_single_node_workflow(
            ForEachNode, params={"workflow": sum_step(engine), "constant_inputs": names}
        )


@pytest.mark.unit
def test_broadcast_default_survives_parameter_schema_reconstruction():
    from workflow_engine.core.values import get_data_dict
    from workflow_engine.core.values.schema import DataValueSchema

    schema = ForEachNode.TYPE_INFO.parameter_schema
    assert isinstance(schema, DataValueSchema)
    constant_schema = schema.properties["constant_inputs"]
    assert "constant_inputs" not in schema.required
    assert constant_schema.model_dump(mode="json")["default"] == []

    # Reconstruct the optional field through the public parameter schema.
    # Missing defaults cannot be reconstructed by portable graph consumers.
    defaults = schema.model_update(
        properties={"constant_inputs": constant_schema}, required=[]
    )
    rebuilt = defaults.build_data_cls()
    assert get_data_dict(rebuilt())["constant_inputs"].model_dump(mode="json") == []
