"""Bounded flat generation, serialization, partiality, and checkpoint replay."""

from datetime import timedelta
from typing import ClassVar

import pytest
from overrides import override
from pydantic import Field, ValidationError

from tests.test_sequence_algebra import CacheContext
from tests.test_sequence_ops import edge, run_roundtrip, workflow
from workflow_engine import (
    BooleanValue,
    Data,
    DataValue,
    Empty,
    ErrorClass,
    ExecutionAlgorithm,
    ExecutionContext,
    IntegerValue,
    Node,
    NodeException,
    NodeTypeInfo,
    Params,
    Result,
    SequenceValue,
    ShouldRetry,
    ShouldYield,
    StringValue,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core.stakeholder import StakeholderLevel
from workflow_engine.nodes import AttemptNode, ForEachNode, UnfoldNode

pytestmark = pytest.mark.integration


@pytest.fixture
def engine(algorithm: ExecutionAlgorithm) -> WorkflowEngine:
    return WorkflowEngine(execution_algorithm=algorithm)


class UnfoldProbeInput(Data):
    seed: IntegerValue = Field(title="Seed", description="The page number.")


class UnfoldProbeOutput(Data):
    items: SequenceValue[IntegerValue] = Field(
        title="Items", description="The page items."
    )
    next: IntegerValue = Field(title="Next", description="The next page number.")
    done: BooleanValue = Field(title="Done", description="The completion flag.")


class UnfoldProbeParams(Params):
    final_seed: IntegerValue = Field(
        default=IntegerValue(2),
        title="Final Seed",
        description="The final page number.",
    )
    empty_page: IntegerValue = Field(
        default=IntegerValue(-1),
        title="Empty Page",
        description="The page whose item list is empty.",
    )


class UnfoldProbeNode(Node[UnfoldProbeInput, UnfoldProbeOutput, UnfoldProbeParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Unfold Probe", version="1.0.0", parameter_type=UnfoldProbeParams
    )
    calls: ClassVar[list[tuple[str, int]]] = []
    yield_seed: ClassVar[int | None] = None
    fail_seed: ClassVar[int | None] = None
    retry_seed: ClassVar[int | None] = None

    @classmethod
    @override
    def static_input_type(cls) -> type[UnfoldProbeInput]:
        return UnfoldProbeInput

    @classmethod
    @override
    def static_output_type(cls) -> type[UnfoldProbeOutput]:
        return UnfoldProbeOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[UnfoldProbeInput],
        output_type: type[UnfoldProbeOutput],
        input: UnfoldProbeInput,
    ) -> UnfoldProbeOutput:
        seed = input.seed.root
        self.calls.append((self.id, seed))
        if seed == self.yield_seed:
            type(self).yield_seed = None
            raise ShouldYield("Waiting for page.")
        if seed == self.retry_seed:
            type(self).retry_seed = None
            raise ShouldRetry(
                "Retry this page.",
                node=self,
                level=StakeholderLevel.USER,
                backoff=timedelta(0),
            )
        if seed == self.fail_seed:
            raise NodeException.for_user("Page failed.", node=self)
        return output_type(
            items=SequenceValue[IntegerValue](
                []
                if seed == self.params.empty_page.root
                else [IntegerValue(10 * seed), IntegerValue(10 * seed + 1)]
            ),
            next=IntegerValue(seed + 1),
            done=BooleanValue(seed >= self.params.final_seed.root),
        )


@pytest.fixture(autouse=True)
def reset_probe():
    UnfoldProbeNode.calls = []
    UnfoldProbeNode.yield_seed = None
    UnfoldProbeNode.fail_seed = None
    UnfoldProbeNode.retry_seed = None


def step(engine, *, final_seed=2, empty_page=-1):
    probe = engine.create_node(
        UnfoldProbeNode,
        id="probe",
        params={"final_seed": final_seed, "empty_page": empty_page},
    )
    return workflow(
        engine,
        {"seed": IntegerValue},
        {
            "items": SequenceValue[IntegerValue],
            "next": IntegerValue,
            "done": BooleanValue,
        },
        [probe],
        [
            edge("input", "seed", "probe", "seed"),
            *[
                edge("probe", name, "output", name)
                for name in ("items", "next", "done")
            ],
        ],
    )


@pytest.mark.parametrize(
    "seed,final,budget,expected",
    [
        (0, 2, 3, [0, 1, 10, 11, 20, 21]),
        (2, 2, 1, [20, 21]),
        (0, 0, 9, [0, 1]),
    ],
)
async def test_unfold_roundtrip_terminating_page_is_included(
    engine, seed, final, budget, expected
):
    assert await run_roundtrip(
        engine,
        UnfoldNode,
        {"workflow": step(engine, final_seed=final), "max_iterations": budget},
        {"seed": seed},
    ) == {"sequence": expected}
    assert len(UnfoldProbeNode.calls) == final - seed + 1


@pytest.mark.parametrize("final,empty,expected", [(2, 1, [0, 1, 20, 21]), (0, 0, [])])
async def test_unfold_empty_pages_and_empty_generation(engine, final, empty, expected):
    assert await run_roundtrip(
        engine,
        UnfoldNode,
        {
            "workflow": step(engine, final_seed=final, empty_page=empty),
            "max_iterations": 3,
        },
        {"seed": 0},
    ) == {"sequence": expected}


@pytest.mark.parametrize("budget", [1, 3])
async def test_unfold_exhausted_budget_is_an_error_without_extra_page(engine, budget):
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=UnfoldNode,
        params={"workflow": step(engine, final_seed=5), "max_iterations": budget},
        input={"seed": 0},
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert "sequence" not in result.output
    assert [seed for _, seed in UnfoldProbeNode.calls] == list(range(budget))


@pytest.mark.parametrize("budget,expected", [(1, [0, 1]), (3, [0, 1, 10, 11, 20, 21])])
async def test_unfold_explicit_truncation_returns_generated_pages(
    engine, budget, expected
):
    assert await run_roundtrip(
        engine,
        UnfoldNode,
        {
            "workflow": step(engine, final_seed=5),
            "max_iterations": budget,
            "truncate": True,
        },
        {"seed": 0},
    ) == {"sequence": expected}
    assert len(UnfoldProbeNode.calls) == budget


@pytest.mark.parametrize("budget", [0, -1])
async def test_unfold_requires_positive_budget(engine, budget):
    with pytest.raises(ValueError, match="positive"):
        await engine.build_single_node_workflow(
            UnfoldNode, params={"workflow": step(engine), "max_iterations": budget}
        )


async def test_unfold_budget_is_required(engine):
    with pytest.raises(ValidationError, match="max_iterations"):
        engine.create_node(UnfoldNode, id="unfold", params={"workflow": step(engine)})


@pytest.mark.parametrize(
    "replacement,expected",
    [
        (
            {"items": IntegerValue, "next": IntegerValue, "done": BooleanValue},
            "SequenceValue",
        ),
        (
            {
                "items": SequenceValue[IntegerValue],
                "next": StringValue,
                "done": BooleanValue,
            },
            "schemas must match",
        ),
        (
            {
                "items": SequenceValue[IntegerValue],
                "next": IntegerValue,
                "done": StringValue,
            },
            "BooleanValue",
        ),
    ],
)
async def test_unfold_rejects_wrong_step_output_types(engine, replacement, expected):
    original = step(engine)
    changed = original.model_update(
        output_node=engine.create_output_node(**replacement)
    )
    with pytest.raises((ValueError, TypeError), match=expected + "|not assignable"):
        await engine.build_single_node_workflow(
            UnfoldNode, params={"workflow": changed, "max_iterations": 3}
        )


async def test_unfold_rejects_extra_step_inputs(engine):
    original = step(engine)
    changed = original.model_update(
        input_node=engine.create_input_node(seed=IntegerValue, extra=IntegerValue)
    )
    with pytest.raises(ValueError, match="take only 'seed'"):
        await engine.build_single_node_workflow(
            UnfoldNode, params={"workflow": changed, "max_iterations": 3}
        )


async def test_unfold_resume_replays_completed_pages_with_identical_ids(engine):
    graph = await engine.build_single_node_workflow(
        UnfoldNode, params={"workflow": step(engine, final_seed=3), "max_iterations": 5}
    )
    context = CacheContext()
    UnfoldProbeNode.yield_seed = 2
    result = await engine.execute(context=context, workflow=graph, input={"seed": 0})
    assert result.status is WorkflowExecutionResultStatus.YIELDED
    assert [seed for _, seed in UnfoldProbeNode.calls] == [0, 1, 2]
    resumed = CacheContext(dict(context.cache))
    result = await engine.execute(
        context=resumed,
        workflow=Workflow.model_validate_json(graph.model_dump_json()),
        input={"seed": 0},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["sequence"].model_dump() == [0, 1, 10, 11, 20, 21, 30, 31]
    assert [seed for _, seed in UnfoldProbeNode.calls] == [0, 1, 2, 2, 3]
    assert [node_id for node_id in resumed.replayed if node_id.endswith("/probe")] == [
        "node/step/probe",
        "node/next/step/probe",
    ]
    for node_id in context.expansions.keys() & resumed.expansions.keys():
        assert context.expansions[node_id] == resumed.expansions[node_id]


async def test_unfold_step_failure_propagates_without_later_dispatch(engine):
    UnfoldProbeNode.fail_seed = 1
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=UnfoldNode,
        params={"workflow": step(engine), "max_iterations": 3},
        input={"seed": 0},
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert [seed for _, seed in UnfoldProbeNode.calls] == [0, 1]
    assert "sequence" not in result.output


async def test_attempt_catches_budget_exhaustion_as_validation_error(engine):
    inner = await engine.build_single_node_workflow(
        UnfoldNode, params={"workflow": step(engine), "max_iterations": 1}
    )
    result = await engine.execute_node(
        context=InMemoryExecutionContext(),
        node=AttemptNode,
        params={"workflow": inner},
        input={"seed": 0},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    captured = result.output["result"]
    assert isinstance(captured, Result)
    error = captured.unwrap_err()
    assert error.error_class.root is ErrorClass.VALIDATION
    assert "maximum iteration count" in error.message.root
    assert error.node_id.root.endswith("/next")


async def test_unfold_result_items_and_nested_step_expansion_roundtrip(engine):
    identity = workflow(
        engine,
        {"value": IntegerValue},
        {"value": IntegerValue},
        [],
        [edge("input", "value", "output", "value")],
    )
    attempted = await engine.build_single_node_workflow(
        AttemptNode, params={"workflow": identity}
    )
    each = engine.create_node(ForEachNode, id="tag", params={"workflow": attempted})
    original = step(engine, final_seed=1)
    tagged = original.model_update(
        inner_nodes=[*original.inner_nodes, each],
        output_node=engine.create_output_node(
            items=SequenceValue[Result[IntegerValue]],
            next=IntegerValue,
            done=BooleanValue,
        ),
        edges=[
            edge("input", "seed", "probe", "seed"),
            edge("probe", "items", "tag", "sequence"),
            edge("tag", "sequence", "output", "items"),
            edge("probe", "next", "output", "next"),
            edge("probe", "done", "output", "done"),
        ],
    )
    assert await run_roundtrip(
        engine, UnfoldNode, {"workflow": tagged, "max_iterations": 4}, {"seed": 0}
    ) == {"sequence": [{"tag": "ok", "ok": value} for value in [0, 1, 10, 11]]}


class UnfoldCursor(Data):
    page: IntegerValue = Field(title="Page", description="The current page index.")


class UnfoldRecordInput(Data):
    seed: DataValue[UnfoldCursor] = Field(
        title="Seed", description="The structured cursor."
    )


class UnfoldRecordOutput(Data):
    items: SequenceValue[StringValue] = Field(
        title="Items", description="The generated strings."
    )
    next: DataValue[UnfoldCursor] = Field(
        title="Next", description="The next structured cursor."
    )
    done: BooleanValue = Field(title="Done", description="The completion flag.")


class UnfoldRecordNode(Node[UnfoldRecordInput, UnfoldRecordOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Unfold Record", version="1.0.0", parameter_type=Empty
    )

    @classmethod
    @override
    def static_input_type(cls) -> type[UnfoldRecordInput]:
        return UnfoldRecordInput

    @classmethod
    @override
    def static_output_type(cls) -> type[UnfoldRecordOutput]:
        return UnfoldRecordOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[UnfoldRecordInput],
        output_type: type[UnfoldRecordOutput],
        input: UnfoldRecordInput,
    ) -> UnfoldRecordOutput:
        page = input.seed.root.page.root
        return output_type(
            items=SequenceValue[StringValue]([StringValue(str(page))]),
            next=DataValue[UnfoldCursor](UnfoldCursor(page=IntegerValue(page + 1))),
            done=BooleanValue(page == 1),
        )


async def test_unfold_structured_seed_json_roundtrip(engine):
    producer = engine.create_node(UnfoldRecordNode, id="record")
    body = workflow(
        engine,
        {"seed": DataValue[UnfoldCursor]},
        {
            "items": SequenceValue[StringValue],
            "next": DataValue[UnfoldCursor],
            "done": BooleanValue,
        },
        [producer],
        [
            edge("input", "seed", "record", "seed"),
            *[
                edge("record", field, "output", field)
                for field in ("items", "next", "done")
            ],
        ],
    )
    assert await run_roundtrip(
        engine,
        UnfoldNode,
        {"workflow": body, "max_iterations": 3},
        {"seed": {"page": 0}},
    ) == {"sequence": ["0", "1"]}


async def test_unfold_executor_retry_does_not_consume_iteration_budget(engine):
    UnfoldProbeNode.retry_seed = 0
    assert await run_roundtrip(
        engine,
        UnfoldNode,
        {"workflow": step(engine, final_seed=0), "max_iterations": 1},
        {"seed": 0},
    ) == {"sequence": [0, 1]}
    assert [seed for _, seed in UnfoldProbeNode.calls] == [0, 0]
