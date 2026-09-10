"""Typed sequence algebra with flat, replayable workflow expansion.

ForEach is traverse. Fold chains steps; Filter and GroupBy traverse a
predicate/key workflow and combine its outputs using pure nodes. No operator
interprets Result tags or introduces an error-handling policy.
"""

from functools import cached_property
from typing import ClassVar

from overrides import override
from pydantic import Field

from ..core import (
    BooleanValue,
    Data,
    DataValue,
    ExecutionContext,
    IntegerValue,
    Node,
    NodeTypeInfo,
    Params,
    SequenceValue,
    StringMapValue,
    StringValue,
    ValidationContext,
    Value,
)
from ..core.values import ValueSchemaValue, build_data_type, get_data_dict
from ..core.values.data import get_field_annotations
from .data import (
    SequenceData,
)


def _data(name: str, **fields: type[Value]) -> type[Data]:
    return build_data_type(
        name=name,
        fields={
            key: (
                value,
                Field(
                    title=key.replace("_", " ").title(),
                    description=f"The {key.replace('_', ' ')} value.",
                ),
            )
            for key, value in fields.items()
        },
    )


class ElementParams(Params):
    element_schema: ValueSchemaValue = Field(
        title="Element Schema", description="The value schema of each sequence item."
    )


class _ElementNode(Node[Data, Data, ElementParams]):
    @cached_property
    def element_type(self) -> type[Value]:
        return self.params.element_schema.root.build_value_cls()


class FlattenSequenceNode(_ElementNode):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Flatten Sequence",
        description="Joins one level of nested sequences.",
        version="1.0.0",
        parameter_type=ElementParams,
    )

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> type[Data]:
        return SequenceData[SequenceValue[self.element_type]]

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> type[Data]:
        return SequenceData[self.element_type]

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[Data],
        output_type: type[Data],
        input: Data,
    ) -> Data:
        sequence = get_data_dict(input)["sequence"]
        return output_type.model_validate(
            {"sequence": [item for group in sequence.root for item in group.root]}
        )


class ChunkSequenceParams(ElementParams):
    size: IntegerValue = Field(
        title="Size", description="The maximum number of items in each chunk."
    )


class ChunkSequenceNode(Node[Data, Data, ChunkSequenceParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Chunk Sequence",
        description="Splits a sequence into bounded chunks.",
        version="1.0.0",
        parameter_type=ChunkSequenceParams,
    )

    @cached_property
    def element_type(self) -> type[Value]:
        if self.params.size.root < 1:
            raise ValueError("Chunk size must be positive.")
        return self.params.element_schema.root.build_value_cls()

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> type[Data]:
        return SequenceData[self.element_type]

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> type[Data]:
        return SequenceData[SequenceValue[self.element_type]]

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[Data],
        output_type: type[Data],
        input: Data,
    ) -> Data:
        items = get_data_dict(input)["sequence"].root
        size = self.params.size.root
        return output_type.model_validate(
            {"sequence": [items[i : i + size] for i in range(0, len(items), size)]}
        )


class ZipParams(Params):
    first_schema: ValueSchemaValue = Field(
        title="First Schema",
        description="The value schema of each item in the first sequence.",
    )
    second_schema: ValueSchemaValue = Field(
        title="Second Schema",
        description="The value schema of each item in the second sequence.",
    )


class ZipNode(Node[Data, Data, ZipParams]):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Zip",
        description="Pairs equally sized sequences by position.",
        version="1.0.0",
        parameter_type=ZipParams,
    )

    @cached_property
    def pair_type(self) -> type[Data]:
        return _data(
            "ZipPair",
            first=self.params.first_schema.root.build_value_cls(),
            second=self.params.second_schema.root.build_value_cls(),
        )

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> type[Data]:
        fields = get_field_annotations(self.pair_type)
        return _data(
            "ZipInput", **{key: SequenceValue[value] for key, value in fields.items()}
        )

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> type[Data]:
        return SequenceData[DataValue[self.pair_type]]

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[Data],
        output_type: type[Data],
        input: Data,
    ) -> Data:
        fields = get_data_dict(input)
        first, second = fields["first"].root, fields["second"].root
        if len(first) != len(second):
            raise ValueError(
                f"Zip requires equal lengths, got {len(first)} and {len(second)}."
            )
        return output_type.model_validate(
            {
                "sequence": [
                    {"first": a, "second": b}
                    for a, b in zip(first, second, strict=True)
                ]
            }
        )


class EntriesNode(_ElementNode):
    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Entries",
        description="Lists a mapping's entries in sorted key order.",
        version="1.0.0",
        parameter_type=ElementParams,
    )

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> type[Data]:
        return _data("EntriesInput", mapping=StringMapValue[self.element_type])

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> type[Data]:
        return SequenceData[
            DataValue[_data("MappingEntry", key=StringValue, value=self.element_type)]
        ]

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[Data],
        output_type: type[Data],
        input: Data,
    ) -> Data:
        mapping = get_data_dict(input)["mapping"].root
        return output_type.model_validate(
            {
                "sequence": [
                    {"key": key, "value": mapping[key]} for key in sorted(mapping)
                ]
            }
        )


class SelectSequenceNode(_ElementNode):
    """Pure final stage of Filter; also usable with precomputed decisions."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Select Sequence",
        description="Keeps items selected by positional boolean decisions.",
        version="1.0.0",
        parameter_type=ElementParams,
    )

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> type[Data]:
        return _data(
            "SelectInput",
            sequence=SequenceValue[self.element_type],
            decisions=SequenceValue[BooleanValue],
        )

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> type[Data]:
        return SequenceData[self.element_type]

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[Data],
        output_type: type[Data],
        input: Data,
    ) -> Data:
        fields = get_data_dict(input)
        items, decisions = fields["sequence"].root, fields["decisions"].root
        if len(items) != len(decisions):
            raise ValueError("Each sequence item must have exactly one decision.")
        return output_type.model_validate(
            {
                "sequence": [
                    item
                    for item, decision in zip(items, decisions, strict=True)
                    if decision.root
                ]
            }
        )


class GroupSequenceNode(_ElementNode):
    """Pure final stage of GroupBy; keys and items remain positionally aligned."""

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Group Sequence",
        description="Groups items using positional string keys.",
        version="1.0.0",
        parameter_type=ElementParams,
    )

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> type[Data]:
        return _data(
            "GroupInput",
            sequence=SequenceValue[self.element_type],
            keys=SequenceValue[StringValue],
        )

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> type[Data]:
        return _data(
            "GroupOutput",
            mapping=StringMapValue[SequenceValue[self.element_type]],
        )

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: type[Data],
        output_type: type[Data],
        input: Data,
    ) -> Data:
        fields = get_data_dict(input)
        items, keys = fields["sequence"].root, fields["keys"].root
        if len(items) != len(keys):
            raise ValueError("Each sequence item must have exactly one group key.")
        groups: dict[str, list[Value]] = {}
        for item, key in zip(items, keys, strict=True):
            groups.setdefault(key.root, []).append(item)
        return output_type.model_validate({"mapping": groups})


__all__ = [
    "ChunkSequenceNode",
    "EntriesNode",
    "FlattenSequenceNode",
    "GroupSequenceNode",
    "SelectSequenceNode",
    "ZipNode",
]
