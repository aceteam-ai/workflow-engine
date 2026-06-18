import pytest

from workflow_engine import (
    Edge,
    IntegerValue,
    StringValue,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core import StringMapValue
from workflow_engine.files import TextFileValue
from workflow_engine.nodes import BashNode


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


async def _read(result, key: str, context: InMemoryExecutionContext) -> str:
    value = result.output[key]
    assert isinstance(value, TextFileValue)
    return await value.read_text(context)


def _separate_workflow(engine: WorkflowEngine, command: str) -> Workflow:
    return Workflow(
        input_node=engine.create_input_node(),
        output_node=(
            output_node := engine.create_output_node(
                stdout=TextFileValue,
                stderr=TextFileValue,
                exit_code=IntegerValue,
            )
        ),
        inner_nodes=[
            bash := engine.create_node(
                BashNode,
                id="bash",
                params=dict(command=command),
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=bash, source_key=key, target=output_node, target_key=key
            )
            for key in ("stdout", "stderr", "exit_code")
        ],
    )


@pytest.mark.asyncio
async def test_basic_command(engine: WorkflowEngine):
    context = InMemoryExecutionContext()
    workflow = _separate_workflow(engine, "echo hello")

    result = await engine.execute(context=context, workflow=workflow, input={})

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert (await _read(result, "stdout", context)) == "hello\n"
    assert (await _read(result, "stderr", context)) == ""
    assert result.output["exit_code"].root == 0


@pytest.mark.asyncio
async def test_jinja_arguments(engine: WorkflowEngine):
    context = InMemoryExecutionContext()
    workflow = Workflow(
        input_node=(
            input_node := engine.create_input_node(
                arguments=StringMapValue[StringValue],
            )
        ),
        output_node=(output_node := engine.create_output_node(stdout=TextFileValue)),
        inner_nodes=[
            bash := engine.create_node(
                BashNode,
                id="bash",
                params=dict(command="echo Hello {{ name }}"),
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=input_node,
                source_key="arguments",
                target=bash,
                target_key="arguments",
            ),
            Edge.from_nodes(
                source=bash,
                source_key="stdout",
                target=output_node,
                target_key="stdout",
            ),
        ],
    )

    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={"arguments": StringMapValue({"name": StringValue("world")})},
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert (await _read(result, "stdout", context)) == "Hello world\n"


@pytest.mark.asyncio
async def test_nonzero_exit_and_stderr(engine: WorkflowEngine):
    context = InMemoryExecutionContext()
    workflow = _separate_workflow(engine, "echo oops >&2; exit 3")

    result = await engine.execute(context=context, workflow=workflow, input={})

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert (await _read(result, "stderr", context)) == "oops\n"
    assert result.output["exit_code"].root == 3


@pytest.mark.asyncio
async def test_combined_output(engine: WorkflowEngine):
    context = InMemoryExecutionContext()
    workflow = Workflow(
        input_node=engine.create_input_node(),
        output_node=(
            output_node := engine.create_output_node(
                output=TextFileValue,
                exit_code=IntegerValue,
            )
        ),
        inner_nodes=[
            bash := engine.create_node(
                BashNode,
                id="bash",
                params=dict(
                    command="echo out; echo err >&2",
                    combine_output=True,
                ),
            ),
        ],
        edges=[
            Edge.from_nodes(
                source=bash, source_key=key, target=output_node, target_key=key
            )
            for key in ("output", "exit_code")
        ],
    )

    result = await engine.execute(context=context, workflow=workflow, input={})

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert (await _read(result, "output", context)) == "out\nerr\n"
