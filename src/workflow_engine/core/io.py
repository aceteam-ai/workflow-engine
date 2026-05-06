# workflow_engine/core/io.py
from typing import TYPE_CHECKING, ClassVar, Self, Type

from overrides import override
from pydantic import Field, PrivateAttr

from .node import Node, NodeTypeInfo, Params
from .values import (
    Data,
    FieldSchemaMappingValue,
    ValueType,
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
        display_name="Input Node",
        description="The unique node that provides the input for the workflow",
        version="1.0.0",
        parameter_type=SchemaParams,
    )

    _cached_data_cls: Type[Data] | None = PrivateAttr(default=None)

    @override
    async def dynamic_input_type(self, context: "ValidationContext") -> Type[Data]:
        if self._cached_data_cls is None:
            self._cached_data_cls = self.params.fields.to_data_schema(
                "InputData"
            ).build_data_cls()
        return self._cached_data_cls

    @override
    async def dynamic_output_type(self, context: "ValidationContext") -> Type[Data]:
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


class OutputNode(Node[Data, Data, SchemaParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Output Node",
        description="The unique node that provides the output for the workflow",
        version="1.0.0",
        parameter_type=SchemaParams,
    )

    _cached_data_cls: Type[Data] | None = PrivateAttr(default=None)

    @override
    async def dynamic_input_type(self, context: "ValidationContext") -> Type[Data]:
        if self._cached_data_cls is None:
            self._cached_data_cls = self.params.fields.to_data_schema(
                "OutputData"
            ).build_data_cls()
        return self._cached_data_cls

    @override
    async def dynamic_output_type(self, context: "ValidationContext") -> Type[Data]:
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


__all__ = [
    "InputNode",
    "OutputNode",
    "SchemaParams",
]
