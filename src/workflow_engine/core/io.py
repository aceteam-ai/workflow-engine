from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, Literal, Mapping, Self, Type

from pydantic import Field

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

    type: Literal["Input"] = "Input"
    params: SchemaParams = Field(default_factory=lambda: SchemaParams(fields=StringMapValue({})))

    @cached_property
    def input_type(self) -> Type[Data]:
        return self.input_schema.build_data_cls()

    @property
    def output_type(self) -> Type[Data]:
        return self.input_type

    @cached_property
    def input_schema(self) -> DataValueSchema:
        return DataValueSchema(
            type="object",
            title="InputData",
            properties={key: field.root for key, field in self.params.fields.items()},
            additionalProperties=False,
        )

    @property
    def output_schema(self) -> DataValueSchema:
        return self.input_schema

    async def run(self, context: "Context", input: Data) -> Data:
        return input

    @classmethod
    def from_fields(
        cls,
        id: str,
        fields: Mapping[str, ValueType] | None = None,
    ) -> Self:
        """Create an InputNode with the specified output fields.

        Args:
            id: Node ID
            fields: Mapping from field name to Value type (e.g., {"a": IntegerValue})
        """
        if fields is None:
            fields = {}

        schema_fields = StringMapValue[ValueSchemaValue](
            {name: ValueSchemaValue(vtype.to_value_schema()) for name, vtype in fields.items()}
        )
        return cls(id=id, params=SchemaParams(fields=schema_fields))


class OutputNode(Node[Data, Empty, SchemaParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        name="Output",
        display_name="Output Node",
        description="The unique node that provides the output for the workflow",
        version="1.0.0",
        parameter_type=SchemaParams,
    )

    type: Literal["Output"] = "Output"
    params: SchemaParams = Field(default_factory=lambda: SchemaParams(fields=StringMapValue({})))

    @property
    def input_type(self) -> Type[Data]:
        return self.output_type

    @cached_property
    def output_type(self) -> Type[Data]:
        return self.output_schema.build_data_cls()

    @property
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

    async def run(self, context: "Context", input: Data) -> Data:
        return input

    @classmethod
    def from_fields(
        cls,
        id: str,
        fields: Mapping[str, ValueType] | None = None,
    ) -> Self:
        """Create an OutputNode with the specified input/output fields.

        Args:
            id: Node ID
            fields: Mapping from field name to Value type (e.g., {"result": IntegerValue})
        """
        if fields is None:
            fields = {}

        schema_fields = StringMapValue[ValueSchemaValue](
            {name: ValueSchemaValue(vtype.to_value_schema()) for name, vtype in fields.items()}
        )
        return cls(id=id, params=SchemaParams(fields=schema_fields))


__all__ = [
    "InputNode",
    "OutputNode",
    "SchemaParams",
]
