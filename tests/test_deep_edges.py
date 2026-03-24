"""
Tests for deep edges: extracting values from nested DataValue structures.

Tests run against both TopologicalExecutionAlgorithm and
ParallelExecutionAlgorithm via the `algorithm` fixture.
"""

from typing import ClassVar, Literal, Type

from overrides import override
import pytest
from pydantic import Field

from workflow_engine import (
    Data,
    DataValue,
    Edge,
    Empty,
    ExecutionAlgorithm,
    ExecutionContext,
    IntegerValue,
    StringMapValue,
    StringValue,
    ValidationContext,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core.io import InputNode, OutputNode
from workflow_engine.core.node import Node, NodeTypeInfo
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.execution.parallel import ParallelExecutionAlgorithm


@pytest.fixture(params=["topological", "parallel"])
def algorithm(request) -> ExecutionAlgorithm:
    if request.param == "topological":
        return TopologicalExecutionAlgorithm()
    return ParallelExecutionAlgorithm()


class Level2Data(Data):
    """Deepest level."""

    alpha: IntegerValue = Field(title="Alpha", description="Deep integer.")
    beta: StringValue = Field(title="Beta", description="Deep string.")


class Level1Data(Data):
    """Middle level."""

    x: StringValue = Field(title="X", description="Mid-level string.")
    y: DataValue[Level2Data] = Field(title="Y", description="Nested level-2.")


class Level0Output(Data):
    """Top level with multiple nested DataValues."""

    a: IntegerValue = Field(title="A", description="Top-level integer.")
    b: DataValue[Level1Data] = Field(title="B", description="Nested level-1.")


class MultiLevelOutputNode(Node[Empty, Level0Output, Empty]):
    """Node that outputs a structure with multiple nesting levels."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="MultiLevelOutput",
        display_name="Multi-Level Output",
        description="Outputs nested data for testing deep edges.",
        version="1.0.0",
        parameter_type=Empty,
    )

    type: Literal["MultiLevelOutput"] = "MultiLevelOutput"  # pyright: ignore[reportIncompatibleVariableOverride]
    params: Empty = Field(default_factory=Empty)

    @override
    async def input_type(self, context: ValidationContext) -> Type[Empty]:
        return Empty

    @override
    async def output_type(self, context: ValidationContext) -> Type[Level0Output]:
        return Level0Output

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[Level0Output],
        input: Empty,
    ) -> Level0Output:
        return Level0Output(
            a=IntegerValue(100),
            b=DataValue[Level1Data](
                root=Level1Data(
                    x=StringValue("mid"),
                    y=DataValue[Level2Data](
                        root=Level2Data(
                            alpha=IntegerValue(7),
                            beta=StringValue("deep"),
                        ),
                    ),
                ),
            ),
        )


@pytest.mark.asyncio
async def test_deep_edges(algorithm: ExecutionAlgorithm):
    """
    Extract values at multiple nesting depths: leaf values, and whole nested
    DataValue objects.
    - 1 level:  ["a"] (leaf)
    - 2 levels: ["b", "x"] (leaf), ["b", "y"] (whole DataValue)
    - 3 levels: ["b", "y", "alpha"], ["b", "y", "beta"] (leaves)
    """
    node = MultiLevelOutputNode(id="multilevel")
    output_node = OutputNode.from_fields(
        top_num=IntegerValue,
        mid_str=StringValue,
        nested=DataValue[Level2Data],
        deep_num=IntegerValue,
        deep_str=StringValue,
    )

    workflow = Workflow(
        input_node=InputNode.from_fields(),
        output_node=output_node,
        inner_nodes=[node],
        edges=[
            Edge.from_nodes(
                source=node,
                source_key="a",
                target=output_node,
                target_key="top_num",
            ),
            Edge.from_nodes(
                source=node,
                source_key=["b", "x"],
                target=output_node,
                target_key="mid_str",
            ),
            Edge.from_nodes(
                source=node,
                source_key=["b", "y"],
                target=output_node,
                target_key="nested",
            ),
            Edge.from_nodes(
                source=node,
                source_key=["b", "y", "alpha"],
                target=output_node,
                target_key="deep_num",
            ),
            Edge.from_nodes(
                source=node,
                source_key=["b", "y", "beta"],
                target=output_node,
                target_key="deep_str",
            ),
        ],
    )

    context = InMemoryExecutionContext()
    engine = WorkflowEngine(execution_algorithm=algorithm)
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={},
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["top_num"].root == 100
    assert result.output["mid_str"].root == "mid"
    assert isinstance(result.output["nested"], DataValue)
    assert result.output["nested"].root.alpha.root == 7
    assert result.output["nested"].root.beta.root == "deep"
    assert result.output["deep_num"].root == 7
    assert result.output["deep_str"].root == "deep"


# StringMapValue tests
class MapOutput(Data):
    """Output with a StringMapValue (dynamic keys)."""

    map: StringMapValue[IntegerValue] = Field(
        title="Map",
        description="A map of string keys to integer values.",
    )


class MapOutputNode(Node[Empty, MapOutput, Empty]):
    """Node that outputs a StringMapValue with known keys."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="MapOutput",
        display_name="Map Output",
        description="Outputs a StringMapValue for testing dynamic key extraction.",
        version="1.0.0",
        parameter_type=Empty,
    )

    type: Literal["MapOutput"] = "MapOutput"  # pyright: ignore[reportIncompatibleVariableOverride]
    params: Empty = Field(default_factory=Empty)

    @override
    async def input_type(self, context: ValidationContext) -> Type[Empty]:
        return Empty

    @override
    async def output_type(self, context: ValidationContext) -> Type[MapOutput]:
        return MapOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Empty],
        output_type: Type[MapOutput],
        input: Empty,
    ) -> MapOutput:
        return MapOutput(
            map=StringMapValue[IntegerValue](
                root={"found": IntegerValue(42), "also_found": IntegerValue(99)}
            )
        )


@pytest.mark.asyncio
async def test_string_map_value_happy_path(algorithm: ExecutionAlgorithm):
    """Extract a value from StringMapValue when the key exists."""
    node = MapOutputNode(id="map_node")
    output_node = OutputNode.from_fields(num=IntegerValue)

    workflow = Workflow(
        input_node=InputNode.from_fields(),
        output_node=output_node,
        inner_nodes=[node],
        edges=[
            Edge.from_nodes(
                source=node,
                source_key=["map", "found"],
                target=output_node,
                target_key="num",
            ),
        ],
    )

    context = InMemoryExecutionContext()
    engine = WorkflowEngine(execution_algorithm=algorithm)
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={},
    )

    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output["num"].root == 42


@pytest.mark.asyncio
async def test_string_map_value_missing_key(algorithm: ExecutionAlgorithm):
    """Workflow fails when edge references a key that does not exist in StringMapValue."""
    node = MapOutputNode(id="map_node")
    output_node = OutputNode.from_fields(num=IntegerValue)

    workflow = Workflow(
        input_node=InputNode.from_fields(),
        output_node=output_node,
        inner_nodes=[node],
        edges=[
            Edge.from_nodes(
                source=node,
                source_key=["map", "missing_key"],
                target=output_node,
                target_key="num",
            ),
        ],
    )

    context = InMemoryExecutionContext()
    engine = WorkflowEngine(execution_algorithm=algorithm)
    result = await engine.execute(
        context=context,
        workflow=workflow,
        input={},
    )

    assert result.status is WorkflowExecutionResultStatus.ERROR
    assert result.errors.any()
