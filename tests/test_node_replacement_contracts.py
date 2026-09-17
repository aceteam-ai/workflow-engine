"""Replacement declaration checks and both concrete adaptation stages."""

from typing import ClassVar

import pytest
from overrides import override
from pydantic import Field

from tests.test_node_replacement import ReplacementContext
from tests.test_sequence_ops import edge, workflow
from workflow_engine import (
    Data,
    DataMapping,
    ErrorClass,
    FloatValue,
    IntegerValue,
    JSONValue,
    Node,
    NodeTypeInfo,
    NullValue,
    Params,
    Result,
    StringValue,
    UnionValue,
    Value,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.core.values import ErrorClassValue, ValueSchemaValue, get_data_dict
from workflow_engine.core.values.data import get_data_schema
from workflow_engine.core.values.schema import DataValueSchema

pytestmark = pytest.mark.integration


class ContractParams(Params):
    input_schema: ValueSchemaValue = Field(
        title="Input", description="The input contract."
    )
    output_schema: ValueSchemaValue = Field(
        title="Output", description="The output contract."
    )
    target_json: StringValue = Field(
        default=StringValue(""),
        title="Target",
        description="The delegated node configuration.",
    )
    output: JSONValue = Field(
        default=JSONValue({}),
        title="Output",
        description="The explicit output overrides.",
    )


class ContractNode(Node[Data, Data, ContractParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Contract probe", version="1.0.0", parameter_type=ContractParams
    )
    calls: ClassVar[list[tuple[str, DataMapping]]] = []

    @override
    async def dynamic_input_type(self, context) -> type[Data]:
        schema = self.params.input_schema.root
        assert isinstance(schema, DataValueSchema)
        return schema.build_data_cls()

    @override
    async def dynamic_output_type(self, context) -> type[Data]:
        schema = self.params.output_schema.root
        assert isinstance(schema, DataValueSchema)
        return schema.build_data_cls()

    @override
    async def run(self, *, context, input_type, output_type, input) -> Data | Node:
        self.calls.append((self.id, get_data_dict(input)))
        if self.params.target_json.root:
            return Node.model_validate_json(self.params.target_json.root)
        assert isinstance(self.params.output.root, dict)
        values = {**input.model_dump(), **self.params.output.root}
        return output_type.model_validate(
            {
                key: value
                for key, value in values.items()
                if key in output_type.model_fields
            }
        )


class IntegerRecord(Data):
    value: IntegerValue


class StringRecord(Data):
    value: StringValue


class NullableRecord(Data):
    value: IntegerValue
    nullable: UnionValue[StringValue, NullValue]  # pyright: ignore[reportInvalidTypeForm]


class ResultRecord(Data):
    value: Result[IntegerValue]


class EnumRecord(Data):
    value: ErrorClassValue


class CallerInput(IntegerRecord):
    ignored: StringValue


class ChildInput(Data):
    value: FloatValue
    option: StringValue = StringValue("default option")


class ChildOutput(Data):
    value: FloatValue
    extra: StringValue = StringValue("hidden")


class CallerOutput(StringRecord):
    fallback: StringValue = StringValue("default output")


@pytest.fixture(autouse=True)
def reset_contracts():
    ContractNode.calls = []


def contract_node(
    engine, input_type, output_type, *, target=None, output=None, id="contract"
):
    return engine.create_node(
        ContractNode,
        id=id,
        params={
            "input_schema": get_data_schema(input_type),
            "output_schema": get_data_schema(output_type),
            "target_json": "" if target is None else target.model_dump_json(),
            "output": {} if output is None else output,
        },
    )


async def execute_contract(
    algorithm,
    source,
    target_input,
    target_output,
    output,
    *,
    inputs,
    context=None,
    child_output=None,
):
    engine = WorkflowEngine(execution_algorithm=algorithm)
    child = contract_node(engine, target_input, target_output, output=child_output)
    parent = contract_node(engine, source, output, target=child)
    graph = await engine.build_single_node_workflow(ContractNode, params=parent.params)
    return await engine.execute(
        context=context or ReplacementContext(),
        workflow=graph.model_validate_json(graph.model_dump_json()),
        input=inputs,
    )


async def test_input_and_output_defaults_projection_and_casts(algorithm):
    result = await execute_contract(
        algorithm,
        CallerInput,
        ChildInput,
        ChildOutput,
        CallerOutput,
        inputs={"value": 7, "ignored": "not a child port"},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    assert result.output == {
        "value": StringValue("7"),
        "fallback": StringValue("default output"),
    }
    child_input = ContractNode.calls[-1][1]
    assert child_input == {
        "value": FloatValue(7),
        "option": StringValue("default option"),
    }
    assert isinstance(child_input["value"], FloatValue)


@pytest.mark.parametrize(
    "child_input,child_output,parent_output,field",
    [
        (NullableRecord, IntegerRecord, IntegerRecord, "nullable"),
        (IntegerRecord, IntegerRecord, NullableRecord, "nullable"),
        (ResultRecord, IntegerRecord, IntegerRecord, "value"),
        (IntegerRecord, ResultRecord, IntegerRecord, "value"),
    ],
)
async def test_incompatible_contract_is_rejected_before_child_dispatch(
    algorithm, child_input, child_output, parent_output, field
):
    result = await execute_contract(
        algorithm,
        IntegerRecord,
        child_input,
        child_output,
        parent_output,
        inputs={"value": 7},
    )
    assert result.status is WorkflowExecutionResultStatus.ERROR
    error = result.errors.node_errors["node"][0]
    assert error is not None
    assert error.error_class is ErrorClass.VALIDATION
    assert error.name == "NodeReplacementException"
    assert field in error.message
    assert [node_id for node_id, _ in ContractNode.calls] == ["node"]


@pytest.mark.parametrize("stage", ["input", "output"])
async def test_concrete_cast_failure_retains_delegator_and_cause(algorithm, stage):
    if stage == "input":
        result = await execute_contract(
            algorithm,
            StringRecord,
            IntegerRecord,
            IntegerRecord,
            IntegerRecord,
            inputs={"value": "bad integer"},
        )
    else:
        result = await execute_contract(
            algorithm,
            IntegerRecord,
            IntegerRecord,
            StringRecord,
            IntegerRecord,
            inputs={"value": 7},
            child_output={"value": "bad integer"},
        )
    assert result.status is WorkflowExecutionResultStatus.ERROR
    error = result.errors.node_errors["node"][0]
    assert error is not None
    assert error.error_class is ErrorClass.VALIDATION
    assert error.name == "NodeReplacementException"
    assert error.cause is not None


@pytest.mark.parametrize(
    "contract,value",
    [
        (
            ResultRecord,
            {
                "tag": "err",
                "err": {
                    "error_class": "validation",
                    "name": "Original",
                    "message": "bad",
                    "node_id": "origin",
                },
            },
        ),
        (EnumRecord, "timeout"),
    ],
)
async def test_registered_identity_and_result_tags_survive_dynamic_contracts(
    algorithm, contract, value
):
    result = await execute_contract(
        algorithm, contract, contract, contract, contract, inputs={"value": value}
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    assert result.output["value"].model_dump(mode="json") == value
    cls = Result if contract is ResultRecord else ErrorClassValue
    assert isinstance(ContractNode.calls[-1][1]["value"], cls)
    assert isinstance(result.output["value"], cls)


class SourceStage(Value[str]):
    pass


class CallerStage(Value[str]):
    pass


class ChildStage(Value[str]):
    pass


class CallerFinal(Value[str]):
    pass


class FinalStage(Value[str]):
    pass


@SourceStage.register_cast_to(CallerStage)
def source_to_caller(value, context):
    return CallerStage(value.root + "/caller")


@CallerStage.register_cast_to(ChildStage)
def caller_to_child(value, context):
    return ChildStage(value.root + "/child")


@SourceStage.register_cast_to(ChildStage)
def wrong_shortcut_input(value, context):
    return ChildStage(value.root + "/WRONG")


@ChildStage.register_cast_to(CallerFinal)
def child_to_caller(value, context):
    return CallerFinal(value.root + "/return")


@CallerFinal.register_cast_to(FinalStage)
def caller_to_final(value, context):
    return FinalStage(value.root + "/final")


@ChildStage.register_cast_to(FinalStage)
def wrong_shortcut_output(value, context):
    return FinalStage(value.root + "/WRONG")


class CallerStageData(Data):
    value: CallerStage


class ChildStageData(Data):
    value: ChildStage


class CallerFinalData(Data):
    value: CallerFinal


async def test_nontransitive_casts_do_not_bypass_caller_contract(algorithm):
    engine = WorkflowEngine(execution_algorithm=algorithm)
    child = contract_node(engine, ChildStageData, ChildStageData)
    parent = contract_node(engine, CallerStageData, CallerFinalData, target=child)
    graph = workflow(
        engine,
        {"value": SourceStage},
        {"value": FinalStage},
        [parent],
        [
            edge("input", "value", parent.id, "value"),
            edge(parent.id, "value", "output", "value"),
        ],
    )
    result = await engine.execute(
        context=ReplacementContext(),
        workflow=graph.model_validate_json(graph.model_dump_json()),
        input={"value": "start"},
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS, result.errors
    assert result.output["value"] == FinalStage("start/caller/child/return/final")
