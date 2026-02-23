# tests/test_hooks.py
"""Tests for context lifecycle hooks."""

from functools import cached_property
from typing import ClassVar, Literal
from unittest.mock import AsyncMock

import pytest

from workflow_engine import (
    Context,
    Data,
    Edge,
    Node,
    Params,
    ShouldYield,
    StringValue,
    Workflow,
    WorkflowErrors,
    WorkflowYield,
)
from workflow_engine.contexts import InMemoryContext
from workflow_engine.core import NodeTypeInfo
from workflow_engine.core.io import InputNode, OutputNode
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.execution.parallel import ParallelExecutionAlgorithm
from workflow_engine.nodes import ConstantStringNode, ErrorNode


@pytest.fixture(params=["topological", "parallel"])
def any_algorithm(request):
    if request.param == "topological":
        return TopologicalExecutionAlgorithm()
    return ParallelExecutionAlgorithm()


def _simple_workflow() -> Workflow:
    """A minimal workflow: (no input) -> constant -> output."""
    input_node = InputNode.empty()
    output_node = OutputNode.from_fields(value=StringValue)
    constant = ConstantStringNode.from_value(id="constant", value="hello")
    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[constant],
        edges=[
            Edge.from_nodes(
                source=constant,
                source_key="value",
                target=output_node,
                target_key="value",
            )
        ],
    )


def _error_workflow() -> Workflow:
    """A workflow that raises an error during execution."""
    input_node = InputNode.empty()
    output_node = OutputNode.from_fields(value=StringValue)
    constant = ConstantStringNode.from_value(id="constant", value="hello")
    error = ErrorNode.from_name(id="error", name="TestError")
    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[constant, error],
        edges=[
            Edge.from_nodes(
                source=constant,
                source_key="value",
                target=error,
                target_key="info",
            ),
            Edge.from_nodes(
                source=constant,
                source_key="value",
                target=output_node,
                target_key="value",
            ),
        ],
    )


class ExpandingOutput(Data):
    value: StringValue


# A node that expands into a subgraph (returns a Workflow from run())
# Its output_type matches the inner workflow's output schema.
class ExpandingNode(Node[Data, ExpandingOutput, Params]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="Expanding",
        display_name="Expanding",
        description="A node that expands into a subgraph.",
        version="1.0.0",
        parameter_type=Params,
    )
    type: Literal["Expanding"] = "Expanding"  # pyright: ignore[reportIncompatibleVariableOverride]

    output_value: str

    @cached_property
    def input_type(self):
        return Data

    @cached_property
    def output_type(self):
        return ExpandingOutput

    async def run(self, context: Context, input: Data) -> Workflow:
        inner_input = InputNode.empty()
        inner_output = OutputNode.from_fields(value=StringValue)
        constant = ConstantStringNode.from_value(
            id="inner_constant", value=self.output_value
        )
        return Workflow(
            input_node=inner_input,
            output_node=inner_output,
            inner_nodes=[constant],
            edges=[
                Edge.from_nodes(
                    source=constant,
                    source_key="value",
                    target=inner_output,
                    target_key="value",
                )
            ],
        )

    @classmethod
    def create(cls, id: str, output_value: str = "expanded") -> "ExpandingNode":
        return cls(id=id, params=Params(), output_value=output_value)


class YieldingNode(Node[Data, ExpandingOutput, Params]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="HookYielding",
        display_name="HookYielding",
        description="Always raises ShouldYield.",
        version="1.0.0",
        parameter_type=Params,
    )
    type: Literal["HookYielding"] = "HookYielding"  # pyright: ignore[reportIncompatibleVariableOverride]

    @cached_property
    def input_type(self):
        return Data

    @cached_property
    def output_type(self):
        return ExpandingOutput

    async def run(self, context: Context, input: Data) -> ExpandingOutput:
        raise ShouldYield("waiting for approval")


def _yielding_workflow() -> Workflow:
    yielding = YieldingNode(id="yielding", params=Params())
    input_node = InputNode.empty()
    output_node = OutputNode.from_fields(value=StringValue)
    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[yielding],
        edges=[
            Edge.from_nodes(
                source=yielding,
                source_key="value",
                target=output_node,
                target_key="value",
            )
        ],
    )


def _expanding_workflow(output_value: str = "expanded") -> Workflow:
    expanding = ExpandingNode.create(id="expanding", output_value=output_value)
    input_node = InputNode.empty()
    output_node = OutputNode.from_fields(value=StringValue)
    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[expanding],
        edges=[
            Edge.from_nodes(
                source=expanding,
                source_key="value",
                target=output_node,
                target_key="value",
            )
        ],
    )


class TestOnNodeStart:
    @pytest.mark.asyncio
    async def test_called_for_each_node(self):
        workflow = _simple_workflow()
        context = InMemoryContext()
        mock = AsyncMock(side_effect=context.on_node_start)
        context.on_node_start = mock

        algorithm = TopologicalExecutionAlgorithm()
        errors, output = await algorithm.execute(
            context=context, workflow=workflow, input={}
        )

        assert not errors.any()
        assert mock.call_count == 3  # input_node, constant, output_node
        called_ids = {call.kwargs["node"].id for call in mock.call_args_list}
        assert called_ids == {
            workflow.input_node.id,
            "constant",
            workflow.output_node.id,
        }

    @pytest.mark.asyncio
    async def test_can_skip_node_execution(self):
        """Returning a DataMapping from on_node_start bypasses run()."""
        workflow = _simple_workflow()
        context = InMemoryContext()
        original = context.on_node_start

        async def on_node_start(*, node, input):
            if node.id == "constant":
                return {"value": StringValue("cached")}
            return await original(node=node, input=input)

        context.on_node_start = on_node_start

        algorithm = TopologicalExecutionAlgorithm()
        errors, output = await algorithm.execute(
            context=context, workflow=workflow, input={}
        )

        assert not errors.any()
        assert output == {"value": StringValue("cached")}


class TestOnNodeFinish:
    @pytest.mark.asyncio
    async def test_called_for_each_node(self):
        workflow = _simple_workflow()
        context = InMemoryContext()
        mock = AsyncMock(side_effect=context.on_node_finish)
        context.on_node_finish = mock

        algorithm = TopologicalExecutionAlgorithm()
        errors, output = await algorithm.execute(
            context=context, workflow=workflow, input={}
        )

        assert not errors.any()
        assert mock.call_count == 3  # input_node, constant, output_node
        called_ids = {call.kwargs["node"].id for call in mock.call_args_list}
        assert called_ids == {
            workflow.input_node.id,
            "constant",
            workflow.output_node.id,
        }

    @pytest.mark.asyncio
    async def test_can_modify_output(self):
        """The context can transform a node's output via on_node_finish."""
        workflow = _simple_workflow()
        context = InMemoryContext()

        async def on_node_finish(*, node, input, output):
            if node.id == "constant":
                return {"value": StringValue("overridden")}
            return output

        context.on_node_finish = on_node_finish

        algorithm = TopologicalExecutionAlgorithm()
        errors, output = await algorithm.execute(
            context=context, workflow=workflow, input={}
        )

        assert not errors.any()
        assert output == {"value": StringValue("overridden")}

    @pytest.mark.asyncio
    async def test_not_called_for_expanding_nodes(self):
        """on_node_finish is not called when a node returns a Workflow."""
        workflow = _expanding_workflow()
        context = InMemoryContext()
        finish_ids: list[str] = []

        async def on_node_finish(*, node, input, output):
            finish_ids.append(node.id)
            return output

        context.on_node_finish = on_node_finish

        algorithm = TopologicalExecutionAlgorithm()
        errors, _ = await algorithm.execute(
            context=context, workflow=workflow, input={}
        )

        assert not errors.any()
        assert "expanding" not in finish_ids


class TestOnNodeExpand:
    @pytest.mark.asyncio
    async def test_called_when_node_emits_workflow(self):
        workflow = _expanding_workflow()
        context = InMemoryContext()
        mock = AsyncMock(side_effect=context.on_node_expand)
        context.on_node_expand = mock

        algorithm = TopologicalExecutionAlgorithm()
        errors, output = await algorithm.execute(
            context=context, workflow=workflow, input={}
        )

        assert not errors.any()
        assert output == {"value": StringValue("expanded")}
        mock.assert_called_once()
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["node"].id == "expanding"
        assert isinstance(call_kwargs["workflow"], Workflow)

    @pytest.mark.asyncio
    async def test_can_modify_emitted_workflow(self):
        """The context can replace the emitted workflow via on_node_expand."""
        workflow = _expanding_workflow()
        context = InMemoryContext()

        async def on_node_expand(*, node, input, workflow):
            inner_input = InputNode.empty()
            inner_output = OutputNode.from_fields(value=StringValue)
            constant = ConstantStringNode.from_value(
                id="replaced_constant", value="replaced"
            )
            return Workflow(
                input_node=inner_input,
                output_node=inner_output,
                inner_nodes=[constant],
                edges=[
                    Edge.from_nodes(
                        source=constant,
                        source_key="value",
                        target=inner_output,
                        target_key="value",
                    )
                ],
            )

        context.on_node_expand = on_node_expand

        algorithm = TopologicalExecutionAlgorithm()
        errors, output = await algorithm.execute(
            context=context, workflow=workflow, input={}
        )

        assert not errors.any()
        assert output == {"value": StringValue("replaced")}


class TestOnNodeYield:
    @pytest.mark.asyncio
    async def test_called_when_node_yields(self, any_algorithm):
        workflow = _yielding_workflow()
        context = InMemoryContext()
        mock = AsyncMock()
        context.on_node_yield = mock

        with pytest.raises(WorkflowYield):
            await any_algorithm.execute(context=context, workflow=workflow, input={})

        mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_called_with_correct_arguments(self, any_algorithm):
        workflow = _yielding_workflow()
        context = InMemoryContext()
        mock = AsyncMock()
        context.on_node_yield = mock

        with pytest.raises(WorkflowYield):
            await any_algorithm.execute(context=context, workflow=workflow, input={})

        kwargs = mock.call_args.kwargs
        assert kwargs["node"].id == "yielding"
        assert isinstance(kwargs["exception"], ShouldYield)
        assert kwargs["exception"].message == "waiting for approval"

    @pytest.mark.asyncio
    async def test_not_called_for_regular_error(self, any_algorithm):
        """on_node_yield must not fire when a node raises a plain error."""
        workflow = _error_workflow()
        context = InMemoryContext()
        mock = AsyncMock()
        context.on_node_yield = mock

        await any_algorithm.execute(context=context, workflow=workflow, input={})

        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_not_called_on_success(self, any_algorithm):
        workflow = _simple_workflow()
        context = InMemoryContext()
        mock = AsyncMock()
        context.on_node_yield = mock

        errors, _ = await any_algorithm.execute(
            context=context, workflow=workflow, input={}
        )

        assert not errors.any()
        mock.assert_not_called()


class TestOnWorkflowStart:
    @pytest.mark.asyncio
    async def test_called_at_start(self):
        workflow = _simple_workflow()
        context = InMemoryContext()
        mock = AsyncMock(side_effect=context.on_workflow_start)
        context.on_workflow_start = mock

        algorithm = TopologicalExecutionAlgorithm()
        errors, output = await algorithm.execute(
            context=context, workflow=workflow, input={}
        )

        assert not errors.any()
        mock.assert_called_once()
        assert mock.call_args.kwargs["workflow"] is workflow

    @pytest.mark.asyncio
    async def test_can_skip_execution(self):
        """Returning a result from on_workflow_start bypasses all node execution."""
        workflow = _simple_workflow()
        context = InMemoryContext()
        cached_output = {"value": StringValue("cached_workflow")}

        async def on_workflow_start(*, workflow, input):
            return WorkflowErrors(), cached_output

        context.on_workflow_start = on_workflow_start

        node_start_ids: list[str] = []
        original_on_node_start = context.on_node_start

        async def tracking_on_node_start(*, node, input):
            node_start_ids.append(node.id)
            return await original_on_node_start(node=node, input=input)

        context.on_node_start = tracking_on_node_start

        algorithm = TopologicalExecutionAlgorithm()
        errors, output = await algorithm.execute(
            context=context, workflow=workflow, input={}
        )

        assert not errors.any()
        assert output == cached_output
        assert node_start_ids == []  # no nodes ran


class TestOnWorkflowFinish:
    @pytest.mark.asyncio
    async def test_called_on_success(self):
        workflow = _simple_workflow()
        context = InMemoryContext()
        mock = AsyncMock(side_effect=context.on_workflow_finish)
        context.on_workflow_finish = mock

        algorithm = TopologicalExecutionAlgorithm()
        errors, output = await algorithm.execute(
            context=context, workflow=workflow, input={}
        )

        assert not errors.any()
        mock.assert_called_once()
        assert mock.call_args.kwargs["workflow"] is workflow
        assert mock.call_args.kwargs["output"] == {"value": StringValue("hello")}

    @pytest.mark.asyncio
    async def test_can_modify_output(self):
        workflow = _simple_workflow()
        context = InMemoryContext()

        async def on_workflow_finish(*, workflow, input, output):
            return {"value": StringValue("modified")}

        context.on_workflow_finish = on_workflow_finish

        algorithm = TopologicalExecutionAlgorithm()
        errors, output = await algorithm.execute(
            context=context, workflow=workflow, input={}
        )

        assert not errors.any()
        assert output == {"value": StringValue("modified")}

    @pytest.mark.asyncio
    async def test_not_called_on_error(self):
        workflow = _error_workflow()
        context = InMemoryContext()
        mock = AsyncMock()
        context.on_workflow_finish = mock

        algorithm = TopologicalExecutionAlgorithm()
        errors, _ = await algorithm.execute(
            context=context, workflow=workflow, input={}
        )

        assert errors.any()
        mock.assert_not_called()


class TestOnWorkflowError:
    @pytest.mark.asyncio
    async def test_called_on_error(self):
        workflow = _error_workflow()
        context = InMemoryContext()
        mock = AsyncMock(side_effect=context.on_workflow_error)
        context.on_workflow_error = mock

        algorithm = TopologicalExecutionAlgorithm()
        errors, _ = await algorithm.execute(
            context=context, workflow=workflow, input={}
        )

        assert errors.any()
        mock.assert_called_once()
        assert mock.call_args.kwargs["workflow"] is workflow

    @pytest.mark.asyncio
    async def test_can_modify_errors(self):
        """The context can clear errors via on_workflow_error."""
        workflow = _error_workflow()
        context = InMemoryContext()

        async def on_workflow_error(*, workflow, input, errors, partial_output):
            return WorkflowErrors(), partial_output

        context.on_workflow_error = on_workflow_error

        algorithm = TopologicalExecutionAlgorithm()
        errors, _ = await algorithm.execute(
            context=context, workflow=workflow, input={}
        )

        assert not errors.any()

    @pytest.mark.asyncio
    async def test_not_called_on_success(self):
        workflow = _simple_workflow()
        context = InMemoryContext()
        mock = AsyncMock()
        context.on_workflow_error = mock

        algorithm = TopologicalExecutionAlgorithm()
        errors, _ = await algorithm.execute(
            context=context, workflow=workflow, input={}
        )

        assert not errors.any()
        mock.assert_not_called()
