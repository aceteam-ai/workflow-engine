# workflow_engine/core/io.py
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, Literal, Self

from overrides import override

from .node import Empty, Node, NodeTypeInfo, Params
from .values import Data, StringMapValue, ValueSchemaValue, ValueType
from .values.schema import DataValueSchema

if TYPE_CHECKING:
    from .context import Context


class SchemaParams(Params):
    fields: StringMapValue[ValueSchemaValue] = StringMapValue({})


class InputNode(Node[Data, Data, SchemaParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="Input",
        display_name="Input Node",
        description="The unique node that provides the input for the workflow",
        version="1.0.0",
        parameter_type=SchemaParams,
    )

    type: Literal["Input"] = "Input"  # pyright: ignore[reportIncompatibleVariableOverride]

    @cached_property
    def input_type(self):
        return self.input_schema.build_data_cls()

    @cached_property
    def output_type(self):
        return self.input_type

    @cached_property
    def input_schema(self) -> DataValueSchema:
        return DataValueSchema(
            type="object",
            title="InputData",
            properties={key: field.root for key, field in self.params.fields.items()},
            additionalProperties=False,
        )

    @cached_property
    def output_schema(self) -> DataValueSchema:
        return self.input_schema

    @override
    async def run(self, context: "Context", input: Data) -> Data:
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
        schema_fields = StringMapValue[ValueSchemaValue](
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

    @cached_property
    def input_type(self):
        return self.output_type

    @cached_property
    def output_type(self):
        return self.output_schema.build_data_cls()

    @cached_property
    def input_schema(self) -> DataValueSchema:
        return self.output_schema

    @cached_property
    def output_schema(self) -> DataValueSchema:
        return DataValueSchema(
            type="object",
            title="OutputData",
            properties={key: field.root for key, field in self.params.fields.items()},
            additionalProperties=False,
        )

    @override
    async def run(self, context: "Context", input: Data) -> Data:
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
        schema_fields = StringMapValue[ValueSchemaValue](
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
