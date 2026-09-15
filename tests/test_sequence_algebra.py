"""Portable sequence algebra, ordering, typed partiality, and replay contracts."""

from typing import Any, ClassVar

import pytest
from overrides import override
from pydantic import Field

from tests.test_sequence_ops import edge, element_params, run_roundtrip, workflow
from workflow_engine import (
    BooleanValue,
    Data,
    DataMapping,
    DataValue,
    Empty,
    ExecutionAlgorithm,
    ExecutionContext,
    FloatValue,
    IntegerValue,
    Node,
    NodeException,
    NodeTypeInfo,
    Result,
    SequenceValue,
    ShouldYield,
    StringValue,
    ValidatedWorkflow,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core.values import get_data_dict
from workflow_engine.nodes import (
    AddNode,
    AttemptNode,
    ForEachNode,
    GroupSequenceNode,
    SelectSequenceNode,
)
from workflow_engine.nodes.sequence_workflow import FilterNode, FoldNode, GroupByNode

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


@pytest.mark.parametrize("items,expected", [([], 5.0), ([1, 2, 3], 11.0)])
async def test_fold_sum_empty_and_roundtrip(engine, items, expected):
    assert await run_roundtrip(
        engine, FoldNode, {"workflow": sum_step(engine)}, {"seed": 5, "sequence": items}
    ) == {"acc": expected}


async def test_fold_multiple_item_fields_and_multinode_step(engine):
    assert await run_roundtrip(
        engine,
        FoldNode,
        {"workflow": sum_step(engine, multi_item=True)},
        {"seed": 1, "sequence": [{"item": 2, "extra": 3}, {"item": 4, "extra": 5}]},
    ) == {"acc": 15.0}


class FoldProbeInput(Data):
    acc: SequenceValue[IntegerValue] = Field(
        title="Accumulator", description="The accumulated items."
    )
    item: IntegerValue = Field(title="Item", description="The next item.")


class FoldProbeOutput(Data):
    acc: SequenceValue[IntegerValue] = Field(
        title="Accumulator", description="The accumulated items."
    )


class FoldProbeNode(Node[FoldProbeInput, FoldProbeOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Fold Probe", version="1.0.0", parameter_type=Empty
    )
    calls: ClassVar[list[tuple[str, int]]] = []
    yield_item: ClassVar[int | None] = None
    fail_item: ClassVar[int | None] = None

    @classmethod
    @override
    def static_input_type(cls) -> type[FoldProbeInput]:
        return FoldProbeInput

    @classmethod
    @override
    def static_output_type(cls) -> type[FoldProbeOutput]:
        return FoldProbeOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[FoldProbeInput],
        output_type: type[FoldProbeOutput],
        input: FoldProbeInput,
    ) -> FoldProbeOutput:
        self.calls.append((self.id, input.item.root))
        if input.item.root == self.yield_item:
            type(self).yield_item = None
            raise ShouldYield("Waiting for this item.")
        if input.item.root == self.fail_item:
            raise NodeException.for_user("Bad item.", node=self)
        return output_type(
            acc=SequenceValue[IntegerValue]([*input.acc.root, input.item])
        )


def list_step(engine):
    return workflow(
        engine,
        {"acc": SequenceValue[IntegerValue], "item": IntegerValue},
        {"acc": SequenceValue[IntegerValue]},
        [engine.create_node(FoldProbeNode, id="append")],
        [
            edge("input", "acc", "append", "acc"),
            edge("input", "item", "append", "item"),
            edge("append", "acc", "output", "acc"),
        ],
    )


class CacheContext(InMemoryExecutionContext):
    def __init__(self, cache: dict[str, str] | None = None):
        super().__init__()
        self.cache = {} if cache is None else cache
        self.replayed: list[str] = []
        self.expansions: dict[str, dict[str, Any]] = {}

    @override
    async def on_node_start(
        self,
        *,
        node: Node,
        input_type: type[Data],
        output_type: type[Data],
        input: DataMapping,
    ) -> DataMapping | None:
        if node.id in self.cache:
            self.replayed.append(node.id)
            return get_data_dict(output_type.model_validate_json(self.cache[node.id]))
        return None

    @override
    async def on_node_finish(
        self,
        *,
        node: Node,
        input_type: type[Data],
        output_type: type[Data],
        input: DataMapping,
        output: DataMapping,
    ) -> DataMapping:
        self.cache[node.id] = output_type.model_validate(output).model_dump_json()
        return output

    @override
    async def on_node_expand(
        self,
        *,
        node: Node,
        input_type: type[Data],
        output_type: type[Data],
        input: DataMapping,
        workflow: ValidatedWorkflow,
    ) -> ValidatedWorkflow:
        self.expansions[node.id] = workflow.model_dump(mode="json")
        return workflow


@pytest.fixture(autouse=True)
def reset_probes():
    FoldProbeNode.calls = []
    FoldProbeNode.yield_item = None
    FoldProbeNode.fail_item = None
    DecisionProbeNode.calls = []
    DecisionProbeNode.yield_item = None
    DecisionProbeNode.fail_item = None


async def test_fold_list_accumulator_resumes_from_checkpoint(engine):
    graph = await engine.build_single_node_workflow(
        FoldNode, params={"workflow": list_step(engine)}
    )
    FoldProbeNode.yield_item = 3
    context = CacheContext()
    first = await engine.execute(
        context=context, workflow=graph, input={"seed": [], "sequence": [1, 2, 3, 4]}
    )
    assert first.status is WorkflowExecutionResultStatus.YIELDED
    assert [item for _, item in FoldProbeNode.calls] == [1, 2, 3]
    resumed = CacheContext(dict(context.cache))
    restored = Workflow.model_validate_json(graph.model_dump_json())
    result = await engine.execute(
        context=resumed, workflow=restored, input={"seed": [], "sequence": [1, 2, 3, 4]}
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["acc"].model_dump() == [1, 2, 3, 4]
    assert [item for _, item in FoldProbeNode.calls] == [1, 2, 3, 3, 4]
    assert any("step_0" in key for key in resumed.replayed)
    assert any("step_1" in key for key in resumed.replayed)
    for key in context.expansions.keys() & resumed.expansions.keys():
        assert context.expansions[key] == resumed.expansions[key]


async def test_fold_failure_does_not_run_later_steps(engine):
    FoldProbeNode.fail_item = 2
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=FoldNode,
        params={"workflow": list_step(engine)},
        input={"seed": [], "sequence": [1, 2, 3]},
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert [item for _, item in FoldProbeNode.calls] == [1, 2]
    assert "acc" not in result.output


class DecisionInput(Data):
    item: IntegerValue = Field(title="Item", description="The item to classify.")


class DecisionOutput(Data):
    decision: BooleanValue = Field(
        title="Decision", description="The inclusion decision."
    )


class DecisionProbeNode(Node[DecisionInput, DecisionOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Decision Probe", version="1.0.0", parameter_type=Empty
    )
    calls: ClassVar[list[tuple[str, int]]] = []
    yield_item: ClassVar[int | None] = None
    fail_item: ClassVar[int | None] = None

    @classmethod
    @override
    def static_input_type(cls) -> type[DecisionInput]:
        return DecisionInput

    @classmethod
    @override
    def static_output_type(cls) -> type[DecisionOutput]:
        return DecisionOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[DecisionInput],
        output_type: type[DecisionOutput],
        input: DecisionInput,
    ) -> DecisionOutput:
        self.calls.append((self.id, input.item.root))
        if input.item.root == self.yield_item:
            type(self).yield_item = None
            raise ShouldYield("Waiting for decision.")
        if input.item.root == self.fail_item:
            raise NodeException.for_user("Predicate failed.", node=self)
        return output_type(decision=BooleanValue(input.item.root % 2 == 0))


def predicate(engine):
    return workflow(
        engine,
        {"item": IntegerValue},
        {"decision": BooleanValue},
        [engine.create_node(DecisionProbeNode, id="decide")],
        [
            edge("input", "item", "decide", "item"),
            edge("decide", "decision", "output", "decision"),
        ],
    )


@pytest.mark.parametrize(
    "items,expected", [([], []), ([2, 4], [2, 4]), ([1, 3], []), ([3, 2, 4, 1], [2, 4])]
)
async def test_filter_roundtrip_order(engine, items, expected):
    assert await run_roundtrip(
        engine, FilterNode, {"workflow": predicate(engine)}, {"sequence": items}
    ) == {"sequence": expected}


@pytest.mark.parametrize("node", [FilterNode, GroupByNode])
async def test_classification_rejects_unresolved_result_decisions(engine, node):
    attempted = await engine.build_single_node_workflow(
        AttemptNode, params={"workflow": predicate(engine)}
    )
    with pytest.raises(ValueError, match="resolve Result explicitly"):
        await engine.build_single_node_workflow(node, params={"workflow": attempted})


async def test_filter_failed_predicate_has_no_silent_drop(engine):
    DecisionProbeNode.fail_item = 2
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=FilterNode,
        params={"workflow": predicate(engine)},
        input={"sequence": [1, 2, 3]},
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert "sequence" not in result.output


async def test_filter_resume_preserves_completed_predicates(engine):
    graph = await engine.build_single_node_workflow(
        FilterNode, params={"workflow": predicate(engine)}
    )
    DecisionProbeNode.yield_item = 3
    context = CacheContext()
    result = await engine.execute(
        context=context, workflow=graph, input={"sequence": [1, 2, 3, 4]}
    )
    assert result.status is WorkflowExecutionResultStatus.YIELDED
    completed = {item for _, item in DecisionProbeNode.calls} - {3}
    resumed = CacheContext(dict(context.cache))
    result = await engine.execute(
        context=resumed,
        workflow=Workflow.model_validate_json(graph.model_dump_json()),
        input={"sequence": [1, 2, 3, 4]},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["sequence"].model_dump() == [2, 4]
    for item in completed:
        assert [value for _, value in DecisionProbeNode.calls].count(item) == 1
    for key in context.expansions.keys() & resumed.expansions.keys():
        assert context.expansions[key] == resumed.expansions[key]


def key_workflow(engine):
    return workflow(
        engine,
        {"key": StringValue, "value": IntegerValue},
        {"key": StringValue},
        [],
        [edge("input", "key", "output", "key")],
    )


@pytest.mark.parametrize(
    "items,expected",
    [
        ([], {}),
        ([{"key": "z", "value": 2}], {"z": [{"key": "z", "value": 2}]}),
        (
            [
                {"key": "z", "value": 1},
                {"key": "a", "value": 2},
                {"key": "z", "value": 3},
            ],
            {
                "z": [{"key": "z", "value": 1}, {"key": "z", "value": 3}],
                "a": [{"key": "a", "value": 2}],
            },
        ),
    ],
)
async def test_groupby_roundtrip_order(engine, items, expected):
    assert await run_roundtrip(
        engine, GroupByNode, {"workflow": key_workflow(engine)}, {"sequence": items}
    ) == {"mapping": expected}


@pytest.mark.parametrize(
    "cls,field,decisions",
    [(SelectSequenceNode, "decisions", [True]), (GroupSequenceNode, "keys", ["a"])],
)
async def test_combine_requires_positional_alignment(engine, cls, field, decisions):
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=cls,
        params=element_params(),
        input={"sequence": [1, 2], field: decisions},
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR


async def test_group_entries_traverse_composition(engine):
    from workflow_engine.core.values import get_field_annotations
    from workflow_engine.nodes import EntriesNode

    key = await engine.validate(key_workflow(engine))
    item_type = DataValue[key.input_type]
    grouped_type = SequenceValue[item_type]
    entry_mapper = workflow(
        engine,
        {"key": StringValue, "value": grouped_type},
        {"key": StringValue},
        [],
        [edge("input", "key", "output", "key")],
    )
    group = engine.create_node(
        GroupByNode, id="group", params={"workflow": key_workflow(engine)}
    )
    entries = engine.create_node(
        EntriesNode, id="entries", params=element_params(grouped_type)
    )
    each = engine.create_node(ForEachNode, id="each", params={"workflow": entry_mapper})
    graph = workflow(
        engine,
        {"sequence": SequenceValue[item_type]},
        {"sequence": SequenceValue[StringValue]},
        [group, entries, each],
        [
            edge("input", "sequence", "group", "sequence"),
            edge("group", "mapping", "entries", "mapping"),
            edge("entries", "sequence", "each", "sequence"),
            edge("each", "sequence", "output", "sequence"),
        ],
    )
    restored = Workflow.model_validate_json(graph.model_dump_json())
    result = await engine.execute(
        context=InMemoryExecutionContext(),
        workflow=restored,
        input={
            "sequence": [
                {"key": "z", "value": 1},
                {"key": "a", "value": 2},
                {"key": "z", "value": 3},
            ]
        },
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    assert result.output["sequence"].model_dump() == ["a", "z"]
    assert set(
        get_field_annotations((await engine.validate(restored)).output_type)
    ) == {"sequence"}


async def test_filter_multinode_predicate(engine):
    from workflow_engine.nodes import NotNode

    inner = predicate(engine)
    inner = inner.model_update(
        inner_nodes=[*inner.inner_nodes, engine.create_node(NotNode, id="invert")],
        edges=[
            edge("input", "item", "decide", "item"),
            edge("decide", "decision", "invert", "a"),
            edge("invert", "result", "output", "decision"),
        ],
    )
    assert await run_roundtrip(
        engine, FilterNode, {"workflow": inner}, {"sequence": [1, 2, 3, 4]}
    ) == {"sequence": [1, 3]}


async def test_fold_result_accumulator_preserves_err_as_value(engine):
    result_type = Result[IntegerValue]
    step = workflow(
        engine,
        {"acc": result_type, "item": result_type},
        {"acc": result_type},
        [],
        [edge("input", "item", "output", "acc")],
    )
    err = {
        "tag": "err",
        "err": {
            "name": "BadItem",
            "message": "bad",
            "node_id": "original/item",
            "error_class": "validation",
        },
    }
    assert await run_roundtrip(
        engine,
        FoldNode,
        {"workflow": step},
        {"seed": {"tag": "ok", "ok": 0}, "sequence": [{"tag": "ok", "ok": 1}, err]},
    ) == {"acc": err}


@pytest.mark.parametrize(
    "inputs,outputs",
    [
        ({"item": IntegerValue}, {"acc": IntegerValue}),
        ({"acc": IntegerValue}, {"acc": IntegerValue}),
        ({"acc": IntegerValue, "item": IntegerValue}, {"other": IntegerValue}),
        ({"acc": IntegerValue, "item": IntegerValue}, {"acc": StringValue}),
    ],
)
async def test_fold_rejects_invalid_step_signature(engine, inputs, outputs):
    output_key = next(iter(outputs))
    input_key = next(iter(inputs))
    step = workflow(
        engine, inputs, outputs, [], [edge("input", input_key, "output", output_key)]
    )
    with pytest.raises(ValueError, match="Fold"):
        await engine.build_single_node_workflow(FoldNode, params={"workflow": step})


class KeyOutput(Data):
    key: StringValue = Field(title="Key", description="The grouping key.")


class KeyProbeNode(Node[DecisionInput, KeyOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Key Probe", version="1.0.0", parameter_type=Empty
    )

    @classmethod
    @override
    def static_input_type(cls) -> type[DecisionInput]:
        return DecisionInput

    @classmethod
    @override
    def static_output_type(cls) -> type[KeyOutput]:
        return KeyOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[DecisionInput],
        output_type: type[KeyOutput],
        input: DecisionInput,
    ) -> KeyOutput:
        DecisionProbeNode.calls.append((self.id, input.item.root))
        if input.item.root == DecisionProbeNode.yield_item:
            DecisionProbeNode.yield_item = None
            raise ShouldYield("Waiting for key.")
        if input.item.root == DecisionProbeNode.fail_item:
            raise NodeException.for_user("Key failed.", node=self)
        return output_type(key=StringValue(str(input.item.root % 2)))


def probing_key_workflow(engine):
    return workflow(
        engine,
        {"item": IntegerValue},
        {"key": StringValue},
        [engine.create_node(KeyProbeNode, id="key")],
        [edge("input", "item", "key", "item"), edge("key", "key", "output", "key")],
    )


async def test_groupby_key_extraction_checkpoint_replay(engine):
    graph = await engine.build_single_node_workflow(
        GroupByNode, params={"workflow": probing_key_workflow(engine)}
    )
    DecisionProbeNode.yield_item = 3
    context = CacheContext()
    result = await engine.execute(
        context=context, workflow=graph, input={"sequence": [1, 2, 3, 4]}
    )
    assert result.status is WorkflowExecutionResultStatus.YIELDED
    completed = {item for _, item in DecisionProbeNode.calls} - {3}
    resumed = CacheContext(dict(context.cache))
    result = await engine.execute(
        context=resumed,
        workflow=Workflow.model_validate_json(graph.model_dump_json()),
        input={"sequence": [1, 2, 3, 4]},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["mapping"].model_dump() == {"1": [1, 3], "0": [2, 4]}
    for item in completed:
        assert [value for _, value in DecisionProbeNode.calls].count(item) == 1
    for key in context.expansions.keys() & resumed.expansions.keys():
        assert context.expansions[key] == resumed.expansions[key]


async def test_groupby_key_failure_is_not_an_implicit_gap_group(engine):
    DecisionProbeNode.fail_item = 2
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=GroupByNode,
        params={"workflow": probing_key_workflow(engine)},
        input={"sequence": [1, 2, 3]},
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert "mapping" not in result.output


async def test_attempt_collects_filter_failure_at_explicit_boundary(engine):
    DecisionProbeNode.fail_item = 2
    filtered = await engine.build_single_node_workflow(
        FilterNode, params={"workflow": predicate(engine)}
    )
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=AttemptNode,
        params={"workflow": filtered},
        input={"sequence": [1, 2, 3]},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    wrapped = result.output["result"]
    assert isinstance(wrapped, Result)
    assert wrapped.is_err()
    assert wrapped.unwrap_err().node_id.root.endswith("/element_1/decide")


async def test_fold_gates_independent_sources_in_later_steps(engine):
    from workflow_engine.nodes import ConstantIntegerNode

    inner = list_step(engine)
    source = engine.create_node(ConstantIntegerNode, id="source", params={"value": 1})
    inner = inner.model_update(
        inner_nodes=[*inner.inner_nodes, source],
        edges=[
            edge("input", "acc", "append", "acc"),
            edge("source", "value", "append", "item"),
            edge("append", "acc", "output", "acc"),
        ],
    )
    FoldProbeNode.fail_item = 1
    context = CacheContext()
    result = await engine.execute_node(
        context=context,
        node=FoldNode,
        params={"workflow": inner},
        input={"seed": [], "sequence": [10, 20]},
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR
    source_ids = [node_id for node_id in context.cache if node_id.endswith("/source")]
    assert source_ids == ["node/step_0/source"]
