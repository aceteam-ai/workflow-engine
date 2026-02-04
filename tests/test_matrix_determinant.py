"""
Example: 2x2 Matrix Determinant Calculator using Deep Field Access

This example demonstrates the power of deep field access in edges by calculating
the determinant of a 2x2 matrix:

    [[a, b],
     [c, d]]

The determinant is: a*d - b*c

The workflow:
1. Takes a SequenceValue[SequenceValue[FloatValue]] as input (2x2 matrix)
2. A MatrixInputNode outputs the matrix
3. Uses deep field access to extract individual elements from the matrix:
   - matrix_input.matrix[0][0] -> a (top-left)
   - matrix_input.matrix[0][1] -> b (top-right)
   - matrix_input.matrix[1][0] -> c (bottom-left)
   - matrix_input.matrix[1][1] -> d (bottom-right)
4. Calculates a*d and b*c using MultiplyNode
5. Subtracts b*c from a*d using SubtractNode

This showcases:
- Deep indexing through nested SequenceValue types using paths like ("matrix", 0, 0)
- Type-safe path validation at workflow construction time
- Clean workflow structure leveraging deep field access
"""

from typing import ClassVar, Literal, Type

import pytest

from workflow_engine import (
    Edge,
    FloatValue,
    InputEdge,
    OutputEdge,
    SequenceValue,
    Workflow,
)
from workflow_engine.core import (
    Context,
    Data,
    Empty,
    Node,
    NodeTypeInfo,
    Params,
)
from workflow_engine.contexts import InMemoryContext
from workflow_engine.execution import TopologicalExecutionAlgorithm
from workflow_engine.nodes import MultiplyNode, SubtractNode


class MatrixInput(Data):
    """Input type for a node that receives a 2x2 matrix."""

    matrix: SequenceValue[SequenceValue[FloatValue]]


class MatrixOutput(Data):
    """Output type for a node that outputs a 2x2 matrix."""

    matrix: SequenceValue[SequenceValue[FloatValue]]


class MatrixInputNode(Node[MatrixInput, MatrixOutput, Empty]):
    """
    A pass-through node that takes a matrix as input and outputs it unchanged.

    This node enables deep field access on its outputs via edges.
    """

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo(
        name="MatrixInput",
        display_name="Matrix Input",
        description="Pass-through node for matrix input",
        version="1.0.0",
    )

    type: Literal["MatrixInput"] = "MatrixInput"  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def input_type(self) -> Type[MatrixInput]:
        return MatrixInput

    @property
    def output_type(self) -> Type[MatrixOutput]:
        return MatrixOutput

    async def run(self, context: Context, input: MatrixInput) -> MatrixOutput:
        return MatrixOutput(matrix=input.matrix)


@pytest.fixture
def workflow():
    """
    Create a workflow that calculates the determinant of a 2x2 matrix.

    Input: matrix (SequenceValue[SequenceValue[FloatValue]]) - a 2x2 matrix
    Output: determinant (FloatValue) - the determinant value

    The workflow uses deep field access to extract matrix elements:
    - matrix_input.matrix[0][0] and matrix_input.matrix[1][1] -> multiply to get a*d
    - matrix_input.matrix[0][1] and matrix_input.matrix[1][0] -> multiply to get b*c
    - a*d - b*c -> final determinant
    """
    return Workflow(
        nodes=[
            # Input node that receives and outputs the matrix
            matrix_node := MatrixInputNode(id="matrix_input"),
            # Calculate a*d (top-left * bottom-right)
            ad := MultiplyNode(id="a_times_d"),
            # Calculate b*c (top-right * bottom-left)
            bc := MultiplyNode(id="b_times_c"),
            # Calculate determinant: a*d - b*c
            det := SubtractNode(id="determinant"),
        ],
        edges=[
            # Deep field access: Extract matrix[0][0] -> a (top-left)
            Edge.from_nodes(
                source=matrix_node,
                source_key=("matrix", 0, 0),  # Deep path!
                target=ad,
                target_key="a",
            ),
            # Deep field access: Extract matrix[1][1] -> d (bottom-right)
            Edge.from_nodes(
                source=matrix_node,
                source_key=("matrix", 1, 1),  # Deep path!
                target=ad,
                target_key="b",
            ),
            # Deep field access: Extract matrix[0][1] -> b (top-right)
            Edge.from_nodes(
                source=matrix_node,
                source_key=("matrix", 0, 1),  # Deep path!
                target=bc,
                target_key="a",
            ),
            # Deep field access: Extract matrix[1][0] -> c (bottom-left)
            Edge.from_nodes(
                source=matrix_node,
                source_key=("matrix", 1, 0),  # Deep path!
                target=bc,
                target_key="b",
            ),
            # Connect a*d result to subtraction node
            Edge.from_nodes(
                source=ad,
                source_key="product",
                target=det,
                target_key="a",
            ),
            # Connect b*c result to subtraction node
            Edge.from_nodes(
                source=bc,
                source_key="product",
                target=det,
                target_key="b",
            ),
        ],
        input_edges=[
            # Workflow input goes to matrix node
            InputEdge.from_node(
                input_key="matrix",
                target=matrix_node,
                target_key="matrix",
            ),
        ],
        output_edges=[
            # Deep field access in output: Extract determinant.difference
            OutputEdge.from_node(
                source=det,
                source_key="difference",
                output_key="determinant",
            ),
        ],
    )


@pytest.mark.unit
def test_workflow_serialization(workflow: Workflow):
    """Test that the workflow can be serialized and deserialized correctly."""
    workflow_json = workflow.model_dump_json(indent=2)
    with open("examples/matrix_determinant.json", "w") as f:
        f.write(workflow_json)

    # Verify round-trip - deserialize and re-serialize should match
    deserialized_workflow = Workflow.model_validate_json(workflow_json)
    assert deserialized_workflow.model_dump_json(indent=2) == workflow_json


@pytest.mark.asyncio
async def test_workflow_execution(workflow: Workflow):
    """Test that the workflow correctly calculates the determinant of a 2x2 matrix."""
    context = InMemoryContext()
    algorithm = TopologicalExecutionAlgorithm()

    # Test case 1: Identity matrix [[1, 0], [0, 1]]
    # Determinant = 1*1 - 0*0 = 1
    matrix1 = SequenceValue[SequenceValue[FloatValue]]([
        SequenceValue[FloatValue]([FloatValue(1.0), FloatValue(0.0)]),
        SequenceValue[FloatValue]([FloatValue(0.0), FloatValue(1.0)]),
    ])

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={"matrix": matrix1},
    )
    assert not errors.any()
    assert output["determinant"].root == 1.0

    # Test case 2: [[3, 8], [4, 6]]
    # Determinant = 3*6 - 8*4 = 18 - 32 = -14
    matrix2 = SequenceValue[SequenceValue[FloatValue]]([
        SequenceValue[FloatValue]([FloatValue(3.0), FloatValue(8.0)]),
        SequenceValue[FloatValue]([FloatValue(4.0), FloatValue(6.0)]),
    ])

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={"matrix": matrix2},
    )
    assert not errors.any()
    assert output["determinant"].root == -14.0

    # Test case 3: [[2, 5], [1, 3]]
    # Determinant = 2*3 - 5*1 = 6 - 5 = 1
    matrix3 = SequenceValue[SequenceValue[FloatValue]]([
        SequenceValue[FloatValue]([FloatValue(2.0), FloatValue(5.0)]),
        SequenceValue[FloatValue]([FloatValue(1.0), FloatValue(3.0)]),
    ])

    errors, output = await algorithm.execute(
        context=context,
        workflow=workflow,
        input={"matrix": matrix3},
    )
    assert not errors.any()
    assert output["determinant"].root == 1.0
