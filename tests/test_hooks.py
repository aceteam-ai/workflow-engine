# tests/test_hooks.py
"""Tests for context lifecycle hooks."""

from collections.abc import Mapping
from typing import ClassVar, Literal, Type
from unittest.mock import AsyncMock

from overrides import override
import pytest

from workflow_engine import (
    ExecutionContext,
    Data,
    DataMapping,
    Edge,
    ExecutionAlgorithm,
    Node,
    NodeTypeInfo,
    Params,
    ShouldYield,
    StringValue,
    ValidationContext,
    Workflow,
    WorkflowEngine,
    WorkflowErrors,
    WorkflowExecutionResult,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core import ValidatedWorkflow
from workflow_engine.core.io import InputNode, OutputNode
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.execution.parallel import (
    ErrorHandlingMode,
    ParallelExecutionAlgorithm,
)
from workflow_engine.nodes import ConstantStringNode, ErrorNode


@pytest.fixture(params=["topological", "parallel"])
def algorithm(request) -> ExecutionAlgorithm:
    if request.param == "topological":
        return TopologicalExecutionAlgorithm()
    elif request.param == "parallel":
        return ParallelExecutionAlgorithm()
    else:
        raise ValueError(f"Invalid algorithm: {request.param}")


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

    @override
    async def input_type(self, context: ValidationContext) -> Type[Data]:
        return Data

    @override
    async def output_type(self, context: ValidationContext) -> Type[ExpandingOutput]:
        return ExpandingOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Data],
        output_type: Type[ExpandingOutput],
        input: Data,
    ) -> Workflow:
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

    @override
    async def input_type(self, context: ValidationContext) -> Type[Data]:
        return Data

    @override
    async def output_type(self, context: ValidationContext) -> Type[ExpandingOutput]:
        return ExpandingOutput

    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Data],
        output_type: Type[ExpandingOutput],
        input: Data,
    ) -> ExpandingOutput:
        raise ShouldYield("waiting for approval")


def _error_and_yield_workflow() -> Workflow:
    """A workflow where one independent branch errors and another yields."""
    input_node = InputNode.empty()
    output_node = OutputNode.from_fields(value=StringValue)
    constant = ConstantStringNode.from_value(id="constant", value="hello")
    error = ErrorNode.from_name(id="error", name="TestError")
    yielding = YieldingNode(id="yielding", params=Params())
    return Workflow(
        input_node=input_node,
        output_node=output_node,
        inner_nodes=[constant, error, yielding],
        edges=[
            Edge.from_nodes(
                source=constant,
                source_key="value",
                target=error,
                target_key="info",
            ),
            Edge.from_nodes(
                source=yielding,
                source_key="value",
                target=output_node,
                target_key="value",
            ),
        ],
    )


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
        context = InMemoryExecutionContext()
        called_ids: list[str] = []
        original = context.on_node_start

        async def on_node_start(
            *,
            node: Node,
            input_type: Type[Data],
            output_type: Type[Data],
            input: DataMapping,
        ) -> DataMapping | None:
            assert isinstance(node, Node)
            assert input_type is not None
            assert output_type is not None
            assert isinstance(input, Mapping)
            called_ids.append(node.id)
            return await original(
                node=node,
                input_type=input_type,
                output_type=output_type,
                input=input,
            )

        context.on_node_start = on_node_start

        algorithm = TopologicalExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert len(called_ids) == 3  # input_node, constant, output_node
        assert set(called_ids) == {
            workflow.input_node.id,
            "constant",
            workflow.output_node.id,
        }

    @pytest.mark.asyncio
    async def test_can_skip_node_execution(self):
        """Returning a DataMapping from on_node_start bypasses run()."""
        workflow = _simple_workflow()
        context = InMemoryExecutionContext()
        original = context.on_node_start

        async def on_node_start(
            *,
            node: Node,
            input_type: Type[Data],
            output_type: Type[Data],
            input: DataMapping,
        ) -> DataMapping | None:
            assert isinstance(node, Node)
            assert input_type is not None
            assert output_type is not None
            assert isinstance(input, Mapping)
            if node.id == "constant":
                return {"value": StringValue("cached")}
            return await original(
                node=node,
                input_type=input_type,
                output_type=output_type,
                input=input,
            )

        context.on_node_start = on_node_start

        algorithm = TopologicalExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.output == {"value": StringValue("cached")}


class TestOnNodeFinish:
    @pytest.mark.asyncio
    async def test_called_for_each_node(self):
        workflow = _simple_workflow()
        context = InMemoryExecutionContext()
        called_ids: list[str] = []
        original = context.on_node_finish

        async def on_node_finish(
            *,
            node: Node,
            input_type: Type[Data],
            output_type: Type[Data],
            input: DataMapping,
            output: DataMapping,
        ) -> DataMapping:
            assert isinstance(node, Node)
            assert input_type is not None
            assert output_type is not None
            assert isinstance(input, Mapping)
            assert isinstance(output, Mapping)
            called_ids.append(node.id)
            return await original(
                node=node,
                input_type=input_type,
                output_type=output_type,
                input=input,
                output=output,
            )

        context.on_node_finish = on_node_finish

        algorithm = TopologicalExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert len(called_ids) == 3  # input_node, constant, output_node
        assert set(called_ids) == {
            workflow.input_node.id,
            "constant",
            workflow.output_node.id,
        }

    @pytest.mark.asyncio
    async def test_can_modify_output(self):
        """The context can transform a node's output via on_node_finish."""
        workflow = _simple_workflow()
        context = InMemoryExecutionContext()

        async def on_node_finish(
            *,
            node: Node,
            input_type: Type[Data],
            output_type: Type[Data],
            input: DataMapping,
            output: DataMapping,
        ) -> DataMapping:
            assert isinstance(node, Node)
            assert input_type is not None
            assert output_type is not None
            assert isinstance(input, Mapping)
            assert isinstance(output, Mapping)
            if node.id == "constant":
                return {"value": StringValue("overridden")}
            return output

        context.on_node_finish = on_node_finish

        algorithm = TopologicalExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.output == {"value": StringValue("overridden")}

    @pytest.mark.asyncio
    async def test_not_called_for_expanding_nodes(self):
        """on_node_finish is not called when a node returns a Workflow."""
        workflow = _expanding_workflow()
        context = InMemoryExecutionContext()
        finish_ids: list[str] = []

        async def on_node_finish(
            *,
            node: Node,
            input_type: Type[Data],
            output_type: Type[Data],
            input: DataMapping,
            output: DataMapping,
        ) -> DataMapping:
            assert isinstance(node, Node)
            assert input_type is not None
            assert output_type is not None
            assert isinstance(input, Mapping)
            assert isinstance(output, Mapping)
            finish_ids.append(node.id)
            return output

        context.on_node_finish = on_node_finish

        algorithm = TopologicalExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert "expanding" not in finish_ids


class TestOnNodeExpand:
    @pytest.mark.asyncio
    async def test_called_when_node_emits_workflow(self):
        workflow = _expanding_workflow()
        context = InMemoryExecutionContext()
        original = context.on_node_expand

        async def on_node_expand(
            *,
            node: Node,
            input_type: Type[Data],
            output_type: Type[Data],
            input: DataMapping,
            workflow: ValidatedWorkflow,
        ) -> ValidatedWorkflow:
            assert node.id == "expanding"
            assert input_type is not None
            assert output_type is not None
            assert input == {}
            assert isinstance(workflow, ValidatedWorkflow)
            return await original(
                node=node,
                input_type=input_type,
                output_type=output_type,
                input=input,
                workflow=workflow,
            )

        context.on_node_expand = on_node_expand

        algorithm = TopologicalExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.output == {"value": StringValue("expanded")}

    @pytest.mark.asyncio
    async def test_can_modify_emitted_workflow(self):
        """The context can replace the emitted workflow via on_node_expand."""
        workflow = _expanding_workflow()
        context = InMemoryExecutionContext()

        async def on_node_expand(
            *,
            node: Node,
            input_type: Type[Data],
            output_type: Type[Data],
            input: DataMapping,
            workflow: ValidatedWorkflow,
        ) -> ValidatedWorkflow:
            assert node.id == "expanding"
            assert input_type is not None
            assert output_type is not None
            assert input == {}
            assert isinstance(workflow, ValidatedWorkflow)
            inner_input = InputNode.empty()
            inner_output = OutputNode.from_fields(value=StringValue)
            constant = ConstantStringNode.from_value(
                id="replaced_constant", value="replaced"
            )
            overridden_workflow = Workflow(
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
            return await overridden_workflow.validate(context.validation_context)

        context.on_node_expand = on_node_expand

        algorithm = TopologicalExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.output == {"value": StringValue("replaced")}


class TestOnNodeYield:
    @pytest.mark.asyncio
    async def test_called_when_node_yields(
        self,
        algorithm: ExecutionAlgorithm,
    ):
        workflow = _yielding_workflow()
        context = InMemoryExecutionContext()
        yielded = []

        async def on_node_yield(
            *,
            node: Node,
            input_type: Type[Data],
            output_type: Type[Data],
            input: DataMapping,
            exception: ShouldYield,
        ) -> None:
            assert isinstance(node, Node)
            assert input_type is not None
            assert output_type is not None
            assert isinstance(input, Mapping)
            assert isinstance(exception, ShouldYield)
            yielded.append(exception.message)

        context.on_node_yield = on_node_yield

        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )
        assert result.status is WorkflowExecutionResultStatus.YIELDED

        assert len(yielded) == 1
        assert yielded[0] == "waiting for approval"

    @pytest.mark.asyncio
    async def test_called_with_correct_arguments(
        self,
        algorithm: ExecutionAlgorithm,
    ):
        workflow = _yielding_workflow()
        context = InMemoryExecutionContext()

        async def on_node_yield(
            *,
            node: Node,
            input_type: Type[Data],
            output_type: Type[Data],
            input: DataMapping,
            exception: ShouldYield,
        ) -> None:
            assert node.id == "yielding"
            assert input_type is not None
            assert output_type is not None
            assert input == {}
            assert isinstance(exception, ShouldYield)
            assert exception.message == "waiting for approval"

        context.on_node_yield = on_node_yield

        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )
        assert result.status is WorkflowExecutionResultStatus.YIELDED

    @pytest.mark.asyncio
    async def test_not_called_for_regular_error(
        self,
        algorithm: ExecutionAlgorithm,
    ):
        """on_node_yield must not fire when a node raises a plain error."""
        workflow = _error_workflow()
        context = InMemoryExecutionContext()
        mock = AsyncMock()
        context.on_node_yield = mock

        engine = WorkflowEngine(execution_algorithm=algorithm)
        await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_not_called_on_success(
        self,
        algorithm: ExecutionAlgorithm,
    ):
        workflow = _simple_workflow()
        context = InMemoryExecutionContext()
        mock = AsyncMock()
        context.on_node_yield = mock

        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )
        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        mock.assert_not_called()


class TestOnWorkflowStart:
    @pytest.mark.asyncio
    async def test_called_at_start(self):
        workflow = _simple_workflow()
        context = InMemoryExecutionContext()
        original = context.on_workflow_start

        async def on_workflow_start(
            *,
            workflow: ValidatedWorkflow,
            input: DataMapping,
        ) -> WorkflowExecutionResult | None:
            assert isinstance(workflow, ValidatedWorkflow)
            assert input == {}
            return await original(workflow=workflow, input=input)

        context.on_workflow_start = on_workflow_start

        algorithm = TopologicalExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_can_skip_execution(self):
        """Returning a result from on_workflow_start bypasses all node execution."""
        workflow = _simple_workflow()
        context = InMemoryExecutionContext()
        cached_output = {"value": StringValue("cached_workflow")}

        async def on_workflow_start(
            *,
            workflow: ValidatedWorkflow,
            input: DataMapping,
        ) -> WorkflowExecutionResult | None:
            assert isinstance(workflow, ValidatedWorkflow)
            assert input == {}
            return WorkflowExecutionResult.success(cached_output)

        context.on_workflow_start = on_workflow_start

        node_start_ids: list[str] = []
        original_on_node_start = context.on_node_start

        async def tracking_on_node_start(
            *,
            node: Node,
            input_type: Type[Data],
            output_type: Type[Data],
            input: DataMapping,
        ) -> DataMapping | None:
            node_start_ids.append(node.id)
            return await original_on_node_start(
                node=node,
                input_type=input_type,
                output_type=output_type,
                input=input,
            )

        context.on_node_start = tracking_on_node_start

        algorithm = TopologicalExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.output == cached_output
        assert node_start_ids == []  # no nodes ran


class TestOnWorkflowFinish:
    @pytest.mark.asyncio
    async def test_called_on_success(self):
        workflow = _simple_workflow()
        context = InMemoryExecutionContext()
        original = context.on_workflow_finish

        async def on_workflow_finish(
            *,
            workflow: ValidatedWorkflow,
            input: DataMapping,
            output: DataMapping,
        ) -> WorkflowExecutionResult:
            assert isinstance(workflow, ValidatedWorkflow)
            assert input == {}
            assert output == {"value": StringValue("hello")}
            return await original(
                workflow=workflow,
                input=input,
                output=output,
            )

        context.on_workflow_finish = on_workflow_finish

        algorithm = TopologicalExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_can_modify_output(self):
        workflow = _simple_workflow()
        context = InMemoryExecutionContext()

        async def on_workflow_finish(
            *,
            workflow: ValidatedWorkflow,
            input: DataMapping,
            output: DataMapping,
        ) -> WorkflowExecutionResult:
            assert isinstance(workflow, ValidatedWorkflow)
            assert input == {}
            assert isinstance(output, Mapping)
            return WorkflowExecutionResult.success({"value": StringValue("modified")})

        context.on_workflow_finish = on_workflow_finish

        algorithm = TopologicalExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert result.output == {"value": StringValue("modified")}

    @pytest.mark.asyncio
    async def test_not_called_on_error(self):
        workflow = _error_workflow()
        context = InMemoryExecutionContext()
        mock = AsyncMock()
        context.on_workflow_finish = mock

        algorithm = TopologicalExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.ERROR
        mock.assert_not_called()


class TestOnWorkflowYield:
    @pytest.mark.asyncio
    async def test_called_when_workflow_yields(
        self,
        algorithm: ExecutionAlgorithm,
    ):
        workflow = _yielding_workflow()
        context = InMemoryExecutionContext()
        original = context.on_workflow_yield

        async def on_workflow_yield(
            *,
            workflow: ValidatedWorkflow,
            input: DataMapping,
            partial_output: DataMapping,
            node_yields: Mapping[str, str],
        ) -> WorkflowExecutionResult:
            assert isinstance(workflow, ValidatedWorkflow)
            assert input == {}
            assert isinstance(partial_output, Mapping)
            assert isinstance(node_yields, Mapping)
            return await original(
                workflow=workflow,
                input=input,
                partial_output=partial_output,
                node_yields=node_yields,
            )

        context.on_workflow_yield = on_workflow_yield

        engine = WorkflowEngine(execution_algorithm=algorithm)

        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )
        assert result.status is WorkflowExecutionResultStatus.YIELDED

    @pytest.mark.asyncio
    async def test_called_with_correct_arguments(
        self,
        algorithm: ExecutionAlgorithm,
    ):
        workflow = _yielding_workflow()
        context = InMemoryExecutionContext()

        async def on_workflow_yield(
            *,
            workflow: ValidatedWorkflow,
            input: DataMapping,
            partial_output: DataMapping,
            node_yields: Mapping[str, str],
        ) -> WorkflowExecutionResult:
            assert isinstance(workflow, ValidatedWorkflow)
            assert input == {}
            assert partial_output == {}
            assert node_yields == {"yielding": "waiting for approval"}
            return WorkflowExecutionResult.yielded(
                partial_output=partial_output,
                node_yields=node_yields,
            )

        context.on_workflow_yield = on_workflow_yield

        engine = WorkflowEngine(execution_algorithm=algorithm)

        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )
        assert result.status is WorkflowExecutionResultStatus.YIELDED

    @pytest.mark.asyncio
    async def test_not_called_on_success(
        self,
        algorithm: ExecutionAlgorithm,
    ):
        workflow = _simple_workflow()
        context = InMemoryExecutionContext()
        mock = AsyncMock()
        context.on_workflow_yield = mock

        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_not_called_on_error(
        self,
        algorithm: ExecutionAlgorithm,
    ):
        workflow = _error_workflow()
        context = InMemoryExecutionContext()
        mock = AsyncMock()
        context.on_workflow_yield = mock

        engine = WorkflowEngine(execution_algorithm=algorithm)
        _ = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        mock.assert_not_called()


class TestOnWorkflowError:
    @pytest.mark.asyncio
    async def test_called_on_error(self):
        workflow = _error_workflow()
        context = InMemoryExecutionContext()
        original = context.on_workflow_error

        async def on_workflow_error(
            *,
            workflow: ValidatedWorkflow,
            input: DataMapping,
            errors: WorkflowErrors,
            partial_output: DataMapping,
            node_yields: Mapping[str, str],
        ) -> WorkflowExecutionResult:
            assert isinstance(workflow, ValidatedWorkflow)
            assert input == {}
            assert isinstance(errors, WorkflowErrors)
            assert isinstance(partial_output, Mapping)
            assert node_yields == {}
            return await original(
                workflow=workflow,
                input=input,
                errors=errors,
                partial_output=partial_output,
                node_yields=node_yields,
            )

        context.on_workflow_error = on_workflow_error

        algorithm = TopologicalExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_can_modify_errors(self):
        """The context can clear errors via on_workflow_error."""
        workflow = _error_workflow()
        context = InMemoryExecutionContext()

        async def on_workflow_error(
            *,
            workflow: Workflow,
            input: DataMapping,
            errors: WorkflowErrors,
            partial_output: DataMapping,
            node_yields: Mapping[str, str],
        ):
            assert isinstance(workflow, ValidatedWorkflow)
            assert input == {}
            assert isinstance(errors, WorkflowErrors)
            assert isinstance(partial_output, Mapping)
            assert isinstance(node_yields, Mapping)
            return WorkflowExecutionResult.success(partial_output)

        context.on_workflow_error = on_workflow_error

        algorithm = TopologicalExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_not_called_on_success(self):
        workflow = _simple_workflow()
        context = InMemoryExecutionContext()
        mock = AsyncMock()
        context.on_workflow_error = mock

        algorithm = TopologicalExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        mock.assert_not_called()


class TestWorkflowYieldPartialOutput:
    @pytest.mark.asyncio
    async def test_partial_output_passed_to_on_workflow_yield(
        self, algorithm: ExecutionAlgorithm
    ):
        """on_workflow_yield receives the partial output from completed nodes."""
        workflow = _yielding_workflow()
        context = InMemoryExecutionContext()
        received: dict = {}

        async def on_workflow_yield(
            *,
            workflow: Workflow,
            input: DataMapping,
            partial_output: DataMapping,
            node_yields: Mapping[str, str],
        ):
            assert isinstance(workflow, ValidatedWorkflow)
            assert input == {}
            received["partial_output"] = partial_output
            assert isinstance(node_yields, Mapping)
            return WorkflowExecutionResult.success(partial_output)

        context.on_workflow_yield = on_workflow_yield
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.SUCCESS
        assert received["partial_output"] == {}
        assert result.output == {}


class TestWorkflowErrorNodeYields:
    @pytest.mark.asyncio
    async def test_node_yields_passed_to_on_workflow_error(self):
        """on_workflow_error receives node_yields (empty when no nodes yielded)."""
        workflow = _error_workflow()
        context = InMemoryExecutionContext()
        received: dict = {}

        async def on_workflow_error(
            *,
            workflow: Workflow,
            input: DataMapping,
            errors: WorkflowErrors,
            partial_output: DataMapping,
            node_yields: Mapping[str, str],
        ):
            assert isinstance(workflow, ValidatedWorkflow)
            assert input == {}
            assert isinstance(errors, WorkflowErrors)
            assert isinstance(partial_output, Mapping)
            received["node_yields"] = node_yields
            return WorkflowExecutionResult.error(
                errors=errors,
                partial_output=partial_output,
                node_yields=node_yields,
            )

        context.on_workflow_error = on_workflow_error

        algorithm = TopologicalExecutionAlgorithm()
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.ERROR
        assert received["node_yields"] == {}


class TestErrorPrecedenceOverYield:
    @pytest.mark.asyncio
    async def test_error_takes_precedence_over_yield_in_continue_mode(self):
        """When both errors and yields occur, on_workflow_error is called, not on_workflow_yield."""
        workflow = _error_and_yield_workflow()
        context = InMemoryExecutionContext()
        yield_mock = AsyncMock(side_effect=context.on_workflow_yield)
        original_on_error = context.on_workflow_error

        async def on_workflow_error(
            *,
            workflow: ValidatedWorkflow,
            input: DataMapping,
            errors: WorkflowErrors,
            partial_output: DataMapping,
            node_yields: Mapping[str, str],
        ) -> WorkflowExecutionResult:
            assert isinstance(workflow, ValidatedWorkflow)
            assert input == {}
            assert isinstance(errors, WorkflowErrors)
            assert isinstance(partial_output, Mapping)
            assert isinstance(node_yields, Mapping)
            return await original_on_error(
                workflow=workflow,
                input=input,
                errors=errors,
                partial_output=partial_output,
                node_yields=node_yields,
            )

        context.on_workflow_yield = yield_mock
        context.on_workflow_error = on_workflow_error

        algorithm = ParallelExecutionAlgorithm(
            error_handling=ErrorHandlingMode.CONTINUE,
        )
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.ERROR
        yield_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_node_yields_included_in_error_hook_when_both_occur(self):
        """When both errors and yields occur, on_workflow_error receives the node_yields."""
        workflow = _error_and_yield_workflow()
        context = InMemoryExecutionContext()
        received: dict = {}

        async def on_workflow_error(
            *,
            workflow: Workflow,
            input: DataMapping,
            errors: WorkflowErrors,
            partial_output: DataMapping,
            node_yields: Mapping[str, str],
        ):
            assert isinstance(workflow, ValidatedWorkflow)
            assert input == {}
            assert isinstance(errors, WorkflowErrors)
            assert isinstance(partial_output, Mapping)
            received["node_yields"] = node_yields
            return WorkflowExecutionResult.error(
                errors=errors,
                partial_output=partial_output,
                node_yields=node_yields,
            )

        context.on_workflow_error = on_workflow_error

        algorithm = ParallelExecutionAlgorithm(
            error_handling=ErrorHandlingMode.CONTINUE,
        )
        engine = WorkflowEngine(execution_algorithm=algorithm)
        result = await engine.execute(
            context=context,
            workflow=workflow,
            input={},
        )

        assert result.status is WorkflowExecutionResultStatus.ERROR
        assert "yielding" in received["node_yields"]
