"""Missing required inputs must fail graph validation, before node dispatch."""

from typing import ClassVar

import pytest
from overrides import override
from pydantic import Field

from workflow_engine import (
    Data,
    DataValue,
    Edge,
    Empty,
    ExecutionContext,
    IntegerValue,
    Node,
    NodeTypeInfo,
    NullValue,
    ResolvedWorkflow,
    UnionValue,
    ValidatedWorkflow,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.nodes import AddNode, AttemptNode


class RequiredEdgeInput(Data):
    required: IntegerValue = Field(title="Required", description="The required input.")
    defaulted: IntegerValue = Field(
        default=IntegerValue(3), title="Defaulted", description="The defaulted input."
    )
    factory: IntegerValue = Field(
        default_factory=lambda: IntegerValue(5),
        title="Factory",
        description="The input with a default factory.",
    )


class RequiredEdgeOutput(Data):
    total: IntegerValue = Field(title="Total", description="The combined inputs.")


class RequiredEdgeProbeNode(Node[RequiredEdgeInput, RequiredEdgeOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Required Edge Probe", version="1.0.0", parameter_type=Empty
    )

    @classmethod
    @override
    def static_input_type(cls) -> type[RequiredEdgeInput]:
        return RequiredEdgeInput

    @classmethod
    @override
    def static_output_type(cls) -> type[RequiredEdgeOutput]:
        return RequiredEdgeOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[RequiredEdgeInput],
        output_type: type[RequiredEdgeOutput],
        input: RequiredEdgeInput,
    ) -> RequiredEdgeOutput:
        return output_type(
            total=IntegerValue(
                input.required.root + input.defaulted.root + input.factory.root
            )
        )


NullableEdgeValue = UnionValue[IntegerValue, NullValue]


class NullableRequiredInput(Data):
    value: NullableEdgeValue = Field(
        title="Value", description="The required input, which may contain null."
    )


class NullableRequiredNode(Node[NullableRequiredInput, Empty, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Nullable Required", version="1.0.0", parameter_type=Empty
    )

    @classmethod
    @override
    def static_input_type(cls) -> type[NullableRequiredInput]:
        return NullableRequiredInput

    @classmethod
    @override
    def static_output_type(cls) -> type[Empty]:
        return Empty

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[NullableRequiredInput],
        output_type: type[Empty],
        input: NullableRequiredInput,
    ) -> Empty:
        return Empty()


class RequiredEdgeRecord(Data):
    value: IntegerValue = Field(title="Value", description="The nested source value.")


@pytest.mark.unit
@pytest.mark.parametrize("wired", [False, True])
async def test_missing_required_inputs_name_node_and_fields(wired):
    engine = WorkflowEngine()
    source = engine.create_input_node(value=IntegerValue)
    node = engine.create_node(AddNode, id="adder")
    graph = Workflow(
        input_node=source,
        inner_nodes=[node],
        output_node=engine.create_output_node(),
        edges=[
            Edge.from_nodes(
                source=source, source_key="value", target=node, target_key="a"
            )
        ]
        if wired
        else [],
    )
    with pytest.raises(
        ValueError, match=r"Node 'adder'.*required input fields"
    ) as caught:
        await engine.validate(graph)
    assert "'b'" in str(caught.value)
    assert ("'a'" in str(caught.value)) is not wired


@pytest.mark.unit
async def test_output_fields_also_require_edges():
    engine = WorkflowEngine()
    graph = Workflow(
        input_node=engine.create_input_node(),
        inner_nodes=[],
        output_node=engine.create_output_node(result=IntegerValue),
        edges=[],
    )
    with pytest.raises(ValueError, match=r"Node 'output'.*'result'"):
        await engine.validate(graph)


@pytest.mark.unit
async def test_nullable_input_still_requires_an_edge():
    engine = WorkflowEngine()
    graph = Workflow(
        input_node=engine.create_input_node(),
        inner_nodes=[engine.create_node(NullableRequiredNode, id="nullable")],
        output_node=engine.create_output_node(),
        edges=[],
    )
    with pytest.raises(ValueError, match=r"Node 'nullable'.*'value'"):
        await engine.validate(graph)


@pytest.mark.unit
async def test_nested_attempt_rejects_missing_inner_input_during_validation():
    engine = WorkflowEngine()
    inner = Workflow(
        input_node=engine.create_input_node(),
        inner_nodes=[engine.create_node(AddNode, id="inner_add")],
        output_node=engine.create_output_node(),
        edges=[],
    )
    with pytest.raises(ValueError, match=r"Node 'inner_add'.*'a'.*'b'"):
        await engine.build_single_node_workflow(AttemptNode, params={"workflow": inner})


@pytest.mark.integration
@pytest.mark.parametrize("path", [False, True])
async def test_defaults_and_deep_source_paths_remain_valid(algorithm, path):
    engine = WorkflowEngine(execution_algorithm=algorithm)
    source = engine.create_input_node(
        **(
            {"record": DataValue[RequiredEdgeRecord]}
            if path
            else {"value": IntegerValue}
        )
    )
    node = engine.create_node(RequiredEdgeProbeNode, id="probe")
    target = engine.create_output_node(total=IntegerValue)
    graph = Workflow(
        input_node=source,
        inner_nodes=[node],
        output_node=target,
        edges=[
            Edge.from_nodes(
                source=source,
                source_key=["record", "value"] if path else "value",
                target=node,
                target_key="required",
            ),
            Edge.from_nodes(
                source=node, source_key="total", target=target, target_key="total"
            ),
        ],
    )
    restored = Workflow.model_validate_json(graph.model_dump_json())
    result = await engine.execute(
        context=InMemoryExecutionContext(),
        workflow=restored,
        input={"record": {"value": 2}} if path else {"value": 2},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["total"] == IntegerValue(10)


@pytest.mark.unit
async def test_invalid_source_path_is_reported_before_missing_target_inputs():
    engine = WorkflowEngine()
    source = engine.create_input_node(record=DataValue[RequiredEdgeRecord])
    node = engine.create_node(AddNode, id="adder")
    graph = Workflow(
        input_node=source,
        inner_nodes=[node],
        output_node=engine.create_output_node(),
        edges=[
            Edge.from_nodes(
                source=source,
                source_key=["record", "absent"],
                target=node,
                target_key="a",
            )
        ],
    )
    with pytest.raises(TypeError, match="absent"):
        await engine.validate(graph)


@pytest.mark.unit
async def test_resolved_draft_cannot_bypass_execution_validation():
    engine = WorkflowEngine()
    graph = Workflow(
        input_node=engine.create_input_node(),
        inner_nodes=[engine.create_node(AddNode, id="unwired")],
        output_node=engine.create_output_node(),
        edges=[],
    )
    draft = await engine.resolve(graph)
    assert isinstance(draft, ResolvedWorkflow)
    assert not isinstance(draft, ValidatedWorkflow)
    assert not hasattr(draft, "get_initial_ready_nodes")
    assert set(draft.node_input_types["unwired"].model_fields) == {"a", "b"}
    with pytest.raises(ValueError, match=r"Node 'unwired'.*'a'.*'b'"):
        await engine.execute(
            context=InMemoryExecutionContext(), workflow=draft, input={}
        )
