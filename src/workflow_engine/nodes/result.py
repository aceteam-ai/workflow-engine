# workflow_engine/nodes/result.py
"""
Eliminators for ``Seq[Result[T]]``.

A small, closed vocabulary for consuming a sequence of ``Result[T]`` values,
decided once so that no combinator built on top of it has to grow its own
error-handling policy. See discussion #198 for the motivation and #200 /
``core/values/result.py`` for ``Result[T]`` itself.

- ``PartitionNode``: splits into oks and errs, as two separate outputs.
- ``UnwrapOrNode``: collapses ``Seq[Result[T]]`` to ``Seq[T]`` using a
  caller-supplied default/marker in place of each err.
- ``AllOkNode``: the all-or-nothing collapse, ``Seq[Result[T]] -> Result[Seq[T]]``.
- ``FirstErrorNode``: the first err in the sequence, if any.

None of these run or retry a workflow; that is ``attempt`` (#201), a separate
piece. These only ever consume a sequence that already contains ``Result[T]``
elements, typically produced by ``for_each(attempt(w))``.
"""

from typing import ClassVar, Generic, Type, TypeVar, cast

from overrides import override
from pydantic import Field
from pydantic.fields import FieldInfo

from ..core import (
    Data,
    DataValue,
    Empty,
    ExecutionContext,
    IntegerValue,
    Node,
    NodeTypeInfo,
    NullValue,
    OptionalValue,
    Result,
    ResultError,
    SequenceValue,
    ValidationContext,
    Value,
    ValueType,
)
from ..core.values import build_data_type, get_data_dict
from .data import SequenceData

V = TypeVar("V", bound=Value)

# ResultError is a Data (a record), not a Value, so it must be wrapped in
# DataValue to appear inside a SequenceValue/OptionalValue, the same way
# NestedData wraps an arbitrary Data object in nodes/data.py. Result[T]
# itself gets to use ResultError directly only because its root model
# (_ErrRoot) is a plain pydantic model outside the Data/Value field system,
# not a Data subclass.
ResultErrorValue = DataValue[ResultError]

# Module-level alias: OptionalValue produces an Annotated construction-time
# union, which pyright requires be bound to a name rather than used inline.
OptionalResultError = OptionalValue[ResultErrorValue]


################################################################################
# partition


class PartitionData(Data, Generic[V]):
    """
    The two-way split of a ``Seq[Result[T]]`` into oks and errs.

    Both halves keep the 0-based index each element held in the original
    input sequence, via a same-length, parallel indices sequence, rather than
    only compacting values into place and dropping where they came from. A
    single string convention that means "this element failed" (the problem
    ``Result[T]`` itself exists to fix) is exactly as unrecoverable for
    *position* as it is for the failure itself: once "oks" and "errs" are
    compacted separately, an ok at position 2 of ``oks`` no longer tells you
    whether it was element 2 or element 17 of the original sequence. A
    consumer that needs to know which original page went missing (the
    motivating case in #198) needs that index; a consumer that does not can
    ignore the ``*_indices`` fields entirely. ``oks`` and ``errs`` themselves
    stay plain sequences of ``T`` / ``ResultError``, so they can still be fed
    directly into whatever consumes them next.
    """

    oks: SequenceValue[V] = Field(
        title="Oks",
        description="The ok values, in their original relative order.",
    )
    ok_indices: SequenceValue[IntegerValue] = Field(
        title="Ok Indices",
        description=(
            "The 0-based index each element of 'oks' held in the original "
            "input sequence. Same length as 'oks'."
        ),
    )
    errs: SequenceValue[ResultErrorValue] = Field(
        title="Errs",
        description="The errors, in their original relative order.",
    )
    err_indices: SequenceValue[IntegerValue] = Field(
        title="Err Indices",
        description=(
            "The 0-based index each element of 'errs' held in the original "
            "input sequence. Same length as 'errs'."
        ),
    )


class PartitionNode(Node[SequenceData, PartitionData, Empty]):
    """
    Splits a ``Seq[Result[T]]`` into its oks and errs, each paired with the
    0-based index it held in the original sequence.
    """

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Partition",
        description="Splits a sequence of Results into oks and errs.",
        version="1.0.0",
        parameter_type=Empty,
    )

    # The type of the ok element. For now, only available when the node is
    # constructed programmatically (see nodes/data.py for the same TODO).
    element_type: ValueType = Field(default=Value, exclude=True)

    @override
    async def dynamic_input_type(
        self, context: ValidationContext
    ) -> Type[SequenceData]:
        return SequenceData[Result[self.element_type]]

    @override
    async def dynamic_output_type(
        self, context: ValidationContext
    ) -> Type[PartitionData]:
        return PartitionData[self.element_type]

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[SequenceData],
        output_type: Type[PartitionData],
        input: SequenceData,
    ) -> PartitionData:
        oks: list[Value] = []
        ok_indices: list[IntegerValue] = []
        errs: list[ResultErrorValue] = []
        err_indices: list[IntegerValue] = []
        for index, item in enumerate(input.sequence):
            if item.is_ok():
                oks.append(item.unwrap_ok())
                ok_indices.append(IntegerValue(index))
            else:
                errs.append(ResultErrorValue(root=item.unwrap_err()))
                err_indices.append(IntegerValue(index))
        return output_type(
            oks=SequenceValue[self.element_type](root=oks),
            ok_indices=SequenceValue[IntegerValue](root=ok_indices),
            errs=SequenceValue[ResultErrorValue](root=errs),
            err_indices=SequenceValue[IntegerValue](root=err_indices),
        )


################################################################################
# unwrap_or


_DEFAULT_FIELD_DESCRIPTION = (
    "The value used in place of each error element. It must be provided "
    "explicitly; there is no built-in default."
)


class UnwrapOrNode(Node[Data, SequenceData, Empty]):
    """
    Collapses a ``Seq[Result[T]]`` to a ``Seq[T]``, substituting a
    caller-supplied default/marker value for each err element.

    This is also the shim for hosts migrating off the old absorb-and-substitute
    behavior, where a failed element silently became a forged success value
    typed as the failed node's output. ``unwrap_or`` makes that substitution
    explicit and total instead of implicit and type-lying: the author wires in
    the exact replacement value, of the exact element type, rather than the
    engine inventing one. That is behavior-identical to the old default only
    for single-scalar outputs (e.g. a marker string); a multi-field output
    needs a caller-authored placeholder record with per-field defaults of its
    own, mirroring whatever the previous absorb behavior actually filled in
    per field, and for types with no sensible default at all (e.g. a file)
    there simply isn't one to wire in, which is the same "no reasonable
    default" refusal as the current implementation, made structural instead
    of case-by-case.
    """

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Unwrap Or",
        description=(
            "Collapses a sequence of Results to a plain sequence, using a "
            "default value in place of each error."
        ),
        version="1.0.0",
        parameter_type=Empty,
    )

    # The type of the element. For now, only available when the node is
    # constructed programmatically (see nodes/data.py for the same TODO).
    element_type: ValueType = Field(default=Value, exclude=True)

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> Type[Data]:
        # `default` must be the element type itself, not wrapped in a
        # container Value (SequenceValue, DataValue, ...). A generic Data
        # subclass can't declare a bare-typevar field (Data validates every
        # field is a concrete Value type at class-definition time, before any
        # parametrization), so this node builds its input type dynamically
        # from `self.element_type`, the same idiom GatherSequenceNode /
        # GatherMappingNode use for a field whose type *is* the variable
        # element type rather than some fixed container of it.
        return build_data_type(
            name="UnwrapOrInput",
            fields={
                "sequence": (
                    SequenceValue[Result[self.element_type]],
                    FieldInfo(
                        title="Sequence",
                        description="The sequence of Results to collapse.",
                    ),
                ),
                "default": (
                    self.element_type,
                    FieldInfo(
                        title="Default",
                        description=_DEFAULT_FIELD_DESCRIPTION,
                    ),
                ),
            },
        )

    @override
    async def dynamic_output_type(
        self, context: ValidationContext
    ) -> Type[SequenceData]:
        return SequenceData[self.element_type]

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Data],
        output_type: Type[SequenceData],
        input: Data,
    ) -> SequenceData:
        input_dict = get_data_dict(input)
        # get_data_dict()'s static return type is Mapping[str, Value]; the
        # actual runtime type of "sequence" is SequenceValue[Result[V]] (it
        # was just built that way above), so cast rather than lie to pyright
        # with an ignore comment.
        sequence = cast(SequenceValue[Result[Value]], input_dict["sequence"])
        default = input_dict["default"]
        items = [item.unwrap_ok() if item.is_ok() else default for item in sequence]
        return output_type(sequence=SequenceValue[self.element_type](root=items))


################################################################################
# all_ok


class AllOkData(Data, Generic[V]):
    """The all-or-nothing collapse of a ``Seq[Result[T]]``."""

    result: Result[SequenceValue[V]] = Field(
        title="Result",
        description=(
            "Ok of the full sequence of values if every element was ok, "
            "otherwise err of the first error encountered."
        ),
    )


class AllOkNode(Node[SequenceData, AllOkData, Empty]):
    """
    ``Seq[Result[T]] -> Result[Seq[T]]``: ok of every value if all elements
    were ok, otherwise err of the first error, scanning in order.

    An empty input sequence is ok of an empty sequence: there is no element to
    fail, so "all elements are ok" holds vacuously. This mirrors Haskell's
    ``sequence [] = pure []`` and keeps ``all_ok`` composable with whatever
    upstream produced zero elements, rather than needing special-case
    handling for that case wherever ``all_ok`` is used.
    """

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="All Ok",
        description=(
            "Collapses a sequence of Results to a single Result: ok of all "
            "values, or err of the first failure."
        ),
        version="1.0.0",
        parameter_type=Empty,
    )

    # The type of the ok element. For now, only available when the node is
    # constructed programmatically (see nodes/data.py for the same TODO).
    element_type: ValueType = Field(default=Value, exclude=True)

    @override
    async def dynamic_input_type(
        self, context: ValidationContext
    ) -> Type[SequenceData]:
        return SequenceData[Result[self.element_type]]

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> Type[AllOkData]:
        return AllOkData[self.element_type]

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[SequenceData],
        output_type: Type[AllOkData],
        input: SequenceData,
    ) -> AllOkData:
        result_type = Result[SequenceValue[self.element_type]]
        oks: list[Value] = []
        for item in input.sequence:
            if item.is_err():
                return output_type(result=result_type.err(item.unwrap_err()))
            oks.append(item.unwrap_ok())
        return output_type(
            result=result_type.ok(SequenceValue[self.element_type](root=oks))
        )


################################################################################
# first_error


class FirstErrorData(Data):
    """The first error in a ``Seq[Result[T]]``, if any."""

    error: OptionalResultError = Field(
        title="First Error",
        description="The first error in the sequence, or null if there were none.",
    )


class FirstErrorNode(Node[SequenceData, FirstErrorData, Empty]):
    """
    ``Seq[Result[T]] -> Result Error | null``: the first err in the sequence,
    scanning in order, or null if every element was ok.

    Unlike ``all_ok``, this never needs the ok values themselves, so its
    output type does not depend on the element type ``T`` at all; only the
    input does.
    """

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="First Error",
        description="Finds the first error in a sequence of Results, if any.",
        version="1.0.0",
        parameter_type=Empty,
    )

    # The type of the ok element. For now, only available when the node is
    # constructed programmatically (see nodes/data.py for the same TODO).
    element_type: ValueType = Field(default=Value, exclude=True)

    @override
    async def dynamic_input_type(
        self, context: ValidationContext
    ) -> Type[SequenceData]:
        return SequenceData[Result[self.element_type]]

    @classmethod
    @override
    def static_output_type(cls) -> Type[FirstErrorData]:
        return FirstErrorData

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[SequenceData],
        output_type: Type[FirstErrorData],
        input: SequenceData,
    ) -> FirstErrorData:
        for item in input.sequence:
            if item.is_err():
                return output_type(error=ResultErrorValue(root=item.unwrap_err()))
        return output_type(error=NullValue(None))


__all__ = [
    "AllOkData",
    "AllOkNode",
    "FirstErrorData",
    "FirstErrorNode",
    "PartitionData",
    "PartitionNode",
    "UnwrapOrNode",
]
