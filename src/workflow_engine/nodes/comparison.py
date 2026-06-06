# workflow_engine/nodes/comparison.py
"""Comparison and logical operator nodes."""

import math
from typing import ClassVar, Type

from overrides import override
from pydantic import Field

from ..core import (
    BooleanValue,
    Data,
    Empty,
    ExecutionContext,
    FloatValue,
    IntegerValue,
    Node,
    NodeTypeInfo,
    Params,
    ValidationContext,
)
from ..core.values import build_data_type


def _argument_field_name(index: int) -> str:
    """
    Generate a field name for the given zero-based index.
    Produces "a", "b", ..., "z", "aa", "ab", ..., "az", "ba", ...
    """
    letters = []
    n = index
    while True:
        letters.append(chr(ord("a") + n % 26))
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(reversed(letters))


class ComparisonParams(Params):
    pass


class EqualityParams(Params):
    rel_tol: FloatValue = Field(
        title="Relative Tolerance",
        description=(
            "The maximum difference between the two values relative to the larger "
            "of their magnitudes, for them to count as equal. Use this to absorb "
            "floating-point rounding error. Defaults to 0, meaning an exact "
            "comparison."
        ),
        default=FloatValue(0.0),
        json_schema_extra={"minimum": 0},
    )
    abs_tol: FloatValue = Field(
        title="Absolute Tolerance",
        description=(
            "The maximum absolute difference between the two values for them to "
            "count as equal. Useful when comparing values near zero, where the "
            "relative tolerance is too strict."
        ),
        default=FloatValue(0.0),
        json_schema_extra={"minimum": 0},
    )


class LogicalParams(Params):
    num_arguments: IntegerValue = Field(
        title="Number of Arguments",
        description="The number of boolean inputs to combine.",
        default=IntegerValue(2),
        json_schema_extra={"minimum": 2},
    )


class ComparisonInput(Data):
    a: FloatValue = Field(
        title="A",
        description="The left value in the comparison.",
    )
    b: FloatValue = Field(
        title="B",
        description="The right value in the comparison.",
    )


class ComparisonOutput(Data):
    result: BooleanValue = Field(
        title="Result",
        description="The result of the comparison.",
    )


class NotInput(Data):
    a: BooleanValue = Field(
        title="A",
        description="The boolean input to negate.",
    )


class LogicalOutput(Data):
    result: BooleanValue = Field(
        title="Result",
        description="The result of the logical operation.",
    )


# --- Comparison Nodes ---


class EqualNode(Node[ComparisonInput, ComparisonOutput, EqualityParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Equal",
        description=(
            "Outputs true if the two input values are equal, within the "
            "configured relative and absolute tolerances."
        ),
        version="1.0.0",
        parameter_type=EqualityParams,
    )

    # we can do this because an empty EqualityParams is valid
    params: EqualityParams = Field(default=EqualityParams())  # pyright: ignore[reportIncompatibleVariableOverride]

    @classmethod
    @override
    def static_input_type(cls) -> Type[ComparisonInput]:
        return ComparisonInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[ComparisonOutput]:
        return ComparisonOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[ComparisonInput],
        output_type: Type[ComparisonOutput],
        input: ComparisonInput,
    ) -> ComparisonOutput:
        result = math.isclose(
            input.a.root,
            input.b.root,
            rel_tol=self.params.rel_tol.root,
            abs_tol=self.params.abs_tol.root,
        )
        return output_type(result=BooleanValue(result))


class NotEqualNode(Node[ComparisonInput, ComparisonOutput, EqualityParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Not Equal",
        description=(
            "Outputs true if the two input values are not equal, within the "
            "configured relative and absolute tolerances."
        ),
        version="1.0.0",
        parameter_type=EqualityParams,
    )

    # we can do this because an empty EqualityParams is valid
    params: EqualityParams = Field(default=EqualityParams())  # pyright: ignore[reportIncompatibleVariableOverride]

    @classmethod
    @override
    def static_input_type(cls) -> Type[ComparisonInput]:
        return ComparisonInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[ComparisonOutput]:
        return ComparisonOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[ComparisonInput],
        output_type: Type[ComparisonOutput],
        input: ComparisonInput,
    ) -> ComparisonOutput:
        result = math.isclose(
            input.a.root,
            input.b.root,
            rel_tol=self.params.rel_tol.root,
            abs_tol=self.params.abs_tol.root,
        )
        return output_type(result=BooleanValue(not result))


class GreaterThanNode(Node[ComparisonInput, ComparisonOutput, ComparisonParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Greater Than",
        description="Outputs true if the first value is greater than the second.",
        version="1.0.0",
        parameter_type=ComparisonParams,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[ComparisonInput]:
        return ComparisonInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[ComparisonOutput]:
        return ComparisonOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[ComparisonInput],
        output_type: Type[ComparisonOutput],
        input: ComparisonInput,
    ) -> ComparisonOutput:
        return output_type(result=BooleanValue(input.a.root > input.b.root))


class GreaterThanEqualNode(Node[ComparisonInput, ComparisonOutput, ComparisonParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Greater Than or Equal",
        description="Outputs true if the first value is greater than or equal to the second.",
        version="1.0.0",
        parameter_type=ComparisonParams,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[ComparisonInput]:
        return ComparisonInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[ComparisonOutput]:
        return ComparisonOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[ComparisonInput],
        output_type: Type[ComparisonOutput],
        input: ComparisonInput,
    ) -> ComparisonOutput:
        return output_type(result=BooleanValue(input.a.root >= input.b.root))


class LessThanNode(Node[ComparisonInput, ComparisonOutput, ComparisonParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Less Than",
        description="Outputs true if the first value is less than the second.",
        version="1.0.0",
        parameter_type=ComparisonParams,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[ComparisonInput]:
        return ComparisonInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[ComparisonOutput]:
        return ComparisonOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[ComparisonInput],
        output_type: Type[ComparisonOutput],
        input: ComparisonInput,
    ) -> ComparisonOutput:
        return output_type(result=BooleanValue(input.a.root < input.b.root))


class LessThanEqualNode(Node[ComparisonInput, ComparisonOutput, ComparisonParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Less Than or Equal",
        description="Outputs true if the first value is less than or equal to the second.",
        version="1.0.0",
        parameter_type=ComparisonParams,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[ComparisonInput]:
        return ComparisonInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[ComparisonOutput]:
        return ComparisonOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[ComparisonInput],
        output_type: Type[ComparisonOutput],
        input: ComparisonInput,
    ) -> ComparisonOutput:
        return output_type(result=BooleanValue(input.a.root <= input.b.root))


# --- Logical Nodes ---


def _logical_input_type(n: int, name: str) -> Type[Data]:
    fields = {
        _argument_field_name(i): (
            BooleanValue,
            Field(title=_argument_field_name(i).upper()),
        )
        for i in range(n)
    }
    return build_data_type(name=name, fields=fields, base_cls=Data)


class AndNode(Node[Data, LogicalOutput, LogicalParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Logical AND",
        description="Outputs true only when all inputs are true.",
        version="1.0.0",
        parameter_type=LogicalParams,
    )

    # we can do this because an empty LogicalParams is valid
    params: LogicalParams = Field(default=LogicalParams())  # pyright: ignore[reportIncompatibleVariableOverride]

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> Type[Data]:
        return _logical_input_type(self.params.num_arguments.root, "AndNodeInput")

    @classmethod
    @override
    def static_output_type(cls) -> Type[LogicalOutput]:
        return LogicalOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Data],
        output_type: Type[LogicalOutput],
        input: Data,
    ) -> LogicalOutput:
        result = all(
            getattr(input, _argument_field_name(i)).root
            for i in range(self.params.num_arguments.root)
        )
        return output_type(result=BooleanValue(result))


class OrNode(Node[Data, LogicalOutput, LogicalParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Logical OR",
        description="Outputs true when at least one input is true.",
        version="1.0.0",
        parameter_type=LogicalParams,
    )

    # we can do this because an empty LogicalParams is valid
    params: LogicalParams = Field(default=LogicalParams())  # pyright: ignore[reportIncompatibleVariableOverride]

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> Type[Data]:
        return _logical_input_type(self.params.num_arguments.root, "OrNodeInput")

    @classmethod
    @override
    def static_output_type(cls) -> Type[LogicalOutput]:
        return LogicalOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Data],
        output_type: Type[LogicalOutput],
        input: Data,
    ) -> LogicalOutput:
        result = any(
            getattr(input, _argument_field_name(i)).root
            for i in range(self.params.num_arguments.root)
        )
        return output_type(result=BooleanValue(result))


class NotNode(Node[NotInput, LogicalOutput, Empty]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Logical NOT",
        description="Returns the opposite of the input value.",
        version="1.0.0",
        parameter_type=Empty,
    )

    @classmethod
    @override
    def static_input_type(cls) -> Type[NotInput]:
        return NotInput

    @classmethod
    @override
    def static_output_type(cls) -> Type[LogicalOutput]:
        return LogicalOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[NotInput],
        output_type: Type[LogicalOutput],
        input: NotInput,
    ) -> LogicalOutput:
        return output_type(result=BooleanValue(not input.a.root))
