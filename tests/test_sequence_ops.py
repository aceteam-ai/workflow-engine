"""Portable sequence algebra, ordering, typed partiality, and replay contracts."""

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from workflow_engine import (
    Edge,
    ErrorClass,
    ExecutionAlgorithm,
    IntegerValue,
    Node,
    Result,
    SequenceValue,
    StringValue,
    Value,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.nodes import (
    AttemptNode,
    ChunkSequenceNode,
    EntriesNode,
    FlattenSequenceNode,
    GroupSequenceNode,
    SelectSequenceNode,
    ZipNode,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def engine(algorithm: ExecutionAlgorithm) -> WorkflowEngine:
    return WorkflowEngine(execution_algorithm=algorithm)


def edge(source: str, key: str, target: str, port: str) -> Edge:
    return Edge(source_id=source, source_key=key, target_id=target, target_key=port)


def workflow(
    engine: WorkflowEngine,
    inputs: Mapping[str, type[Value]],
    outputs: Mapping[str, type[Value]],
    nodes: Sequence[Node],
    edges: list[Edge],
) -> Workflow:
    return Workflow(
        input_node=engine.create_input_node(**inputs),
        output_node=engine.create_output_node(**outputs),
        inner_nodes=nodes,
        edges=edges,
    )


async def run_roundtrip(
    engine: WorkflowEngine,
    cls: type[Node],
    params: dict[str, Any],
    inputs: dict[str, Any],
):
    graph = await engine.build_single_node_workflow(cls, params=params)
    encoded = graph.model_dump_json()
    restored = Workflow.model_validate_json(encoded)
    assert json.loads(restored.model_dump_json()) == json.loads(encoded)
    result = await engine.execute(
        context=InMemoryExecutionContext(), workflow=restored, input=inputs
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    return {key: value.model_dump(mode="json") for key, value in result.output.items()}


def element_params(value: type[Value] = IntegerValue) -> dict[str, Any]:
    return {"element_schema": value.to_value_schema()}


@pytest.mark.parametrize(
    "items,size,chunks",
    [([], 2, []), ([1], 3, [[1]]), ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]])],
)
async def test_chunk_flatten_roundtrip(engine, items, size, chunks):
    assert await run_roundtrip(
        engine,
        ChunkSequenceNode,
        {**element_params(), "size": size},
        {"sequence": items},
    ) == {"sequence": chunks}
    assert await run_roundtrip(
        engine, FlattenSequenceNode, element_params(), {"sequence": chunks}
    ) == {"sequence": items}


@pytest.mark.parametrize("first,second", [([], []), ([1, 2], ["a", "b"])])
async def test_zip_roundtrip(engine, first, second):
    assert await run_roundtrip(
        engine,
        ZipNode,
        {
            "first_schema": IntegerValue.to_value_schema(),
            "second_schema": StringValue.to_value_schema(),
        },
        {"first": first, "second": second},
    ) == {
        "sequence": [
            {"first": a, "second": b} for a, b in zip(first, second, strict=True)
        ]
    }


async def test_zip_rejects_unequal_lengths(engine):
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=ZipNode,
        params={
            "first_schema": IntegerValue.to_value_schema(),
            "second_schema": StringValue.to_value_schema(),
        },
        input={"first": [1], "second": []},
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR


async def test_chunk_rejects_nonpositive_size(engine):
    with pytest.raises(ValueError, match="positive"):
        await engine.build_single_node_workflow(
            ChunkSequenceNode, params={**element_params(), "size": 0}
        )


async def test_flatten_rejects_scalar_sequence_at_graph_validation(engine):
    node = engine.create_node(
        FlattenSequenceNode, id="flatten", params=element_params()
    )
    graph = workflow(
        engine,
        {"sequence": SequenceValue[IntegerValue]},
        {"sequence": SequenceValue[IntegerValue]},
        [node],
        [
            edge("input", "sequence", "flatten", "sequence"),
            edge("flatten", "sequence", "output", "sequence"),
        ],
    )
    with pytest.raises(Exception, match=r"[Cc]ast|[Aa]ssign|compatible"):
        await engine.validate(graph)


async def test_entries_sorted_and_empty(engine):
    for mapping in [{}, {"z": 1, "a": 2, "m": 3}]:
        assert await run_roundtrip(
            engine, EntriesNode, element_params(), {"mapping": mapping}
        ) == {
            "sequence": [{"key": key, "value": mapping[key]} for key in sorted(mapping)]
        }


async def test_zip_preserves_result_positions(engine):
    results = [
        {"tag": "ok", "ok": 7},
        {
            "tag": "err",
            "err": {
                "error_class": "validation",
                "name": "BadItem",
                "message": "bad",
                "node_id": "root/item_1",
            },
        },
    ]
    assert await run_roundtrip(
        engine,
        ZipNode,
        {
            "first_schema": StringValue.to_value_schema(),
            "second_schema": Result[IntegerValue].to_value_schema(),
        },
        {"first": ["first", "second"], "second": results},
    ) == {
        "sequence": [
            {"first": name, "second": result}
            for name, result in zip(["first", "second"], results, strict=True)
        ]
    }


@pytest.mark.parametrize(
    "cls,params,inputs,message",
    [
        (
            ZipNode,
            {
                "first_schema": IntegerValue.to_value_schema(),
                "second_schema": IntegerValue.to_value_schema(),
            },
            {"first": [1], "second": []},
            "Zip requires equal lengths",
        ),
        (
            SelectSequenceNode,
            element_params(),
            {"sequence": [1], "decisions": []},
            "exactly one decision",
        ),
        (
            GroupSequenceNode,
            element_params(),
            {"sequence": [1], "keys": []},
            "exactly one group key",
        ),
    ],
)
async def test_shape_errors_remain_user_visible_validation_errors(
    engine, cls, params, inputs, message
):
    inner = await engine.build_single_node_workflow(cls, params=params)
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=AttemptNode,
        params={"workflow": inner},
        input=inputs,
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    captured = result.output["result"]
    assert isinstance(captured, Result)
    error = captured.unwrap_err()
    assert error.error_class.root is ErrorClass.VALIDATION
    assert message in error.message.root
    assert error.node_id.root.endswith("/node")
