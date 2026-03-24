# workflow_engine/core/io.py
from typing import TYPE_CHECKING, ClassVar, Literal, Self, Type

from pydantic import Field, PrivateAttr
from overrides import override

from .node import Node, NodeTypeInfo, Params
from .values import (
    Data,
    ValueSchemaValue,
    ValueType,
    FieldSchemaMappingValue,
)

if TYPE_CHECKING:
    from .context import ExecutionContext, ValidationContext


class SchemaParams(Params):
    fields: FieldSchemaMappingValue = Field(
        title="Fields",
        description="A mapping of field names to value schemas.",
        default_factory=lambda: FieldSchemaMappingValue({}),
    )

    @classmethod
    def from_fields(cls, **fields: ValueType) -> Self:
        return cls(fields=FieldSchemaMappingValue.from_fields(**fields))


class InputNode(Node[Data, Data, SchemaParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="Input",
        display_name="Input Node",
        description="The unique node that provides the input for the workflow",
        version="1.0.0",
        parameter_type=SchemaParams,
    )

    type: Literal["Input"] = "Input"  # pyright: ignore[reportIncompatibleVariableOverride]

    _cached_data_cls: Type[Data] | None = PrivateAttr(default=None)

    @override
    async def input_type(self, context: "ValidationContext") -> Type[Data]:
        if self._cached_data_cls is None:
            self._cached_data_cls = self.params.fields.to_data_schema(
                "InputData"
            ).build_data_cls()
        return self._cached_data_cls

    @override
    async def output_type(self, context: "ValidationContext") -> Type[Data]:
        return await self.input_type(context)

    @override
    async def run(
        self,
        *,
        context: "ExecutionContext",
        input_type: Type[Data],
        output_type: Type[Data],
        input: Data,
    ) -> Data:
        return input

    @classmethod
    def from_fields(
        cls,
        **fields: ValueType,
    ) -> Self:
        """Create an InputNode with the specified output fields.

        Args:
            fields: Mapping from field name to Value type (e.g., {"a": IntegerValue})
        """
        schema_fields = FieldSchemaMappingValue(
            {
                name: ValueSchemaValue(vtype.to_value_schema())
                for name, vtype in fields.items()
            }
        )
        return cls(
            id="input",
            params=SchemaParams(fields=schema_fields),
        )

    @classmethod
    def empty(cls) -> Self:
        """Create an InputNode with no fields."""
        return cls.from_fields()


class OutputNode(Node[Data, Data, SchemaParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="Output",
        display_name="Output Node",
        description="The unique node that provides the output for the workflow",
        version="1.0.0",
        parameter_type=SchemaParams,
    )

    type: Literal["Output"] = "Output"  # pyright: ignore[reportIncompatibleVariableOverride]

    _cached_data_cls: Type[Data] | None = PrivateAttr(default=None)

    @override
    async def input_type(self, context: "ValidationContext") -> Type[Data]:
        if self._cached_data_cls is None:
            self._cached_data_cls = self.params.fields.to_data_schema(
                "OutputData"
            ).build_data_cls()
        return self._cached_data_cls

    @override
    async def output_type(self, context: "ValidationContext") -> Type[Data]:
        return await self.input_type(context)

    @override
    async def run(
        self,
        *,
        context: "ExecutionContext",
        input_type: Type[Data],
        output_type: Type[Data],
        input: Data,
    ) -> Data:
        return input

    @classmethod
    def from_fields(
        cls,
        **fields: ValueType,
    ) -> Self:
        """Create an OutputNode with the specified input/output fields.

        Args:
            fields: Mapping from field name to Value type (e.g., {"result": IntegerValue})
        """
        schema_fields = FieldSchemaMappingValue(
            {
                name: ValueSchemaValue(vtype.to_value_schema())
                for name, vtype in fields.items()
            }
        )
        return cls(
            id="output",
            params=SchemaParams(fields=schema_fields),
        )

    @classmethod
    def empty(cls) -> Self:
        """Create an OutputNode with no fields."""
        return cls.from_fields()


__all__ = [
    "InputNode",
    "OutputNode",
    "SchemaParams",
]
