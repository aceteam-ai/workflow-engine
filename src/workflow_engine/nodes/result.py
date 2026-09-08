# workflow_engine/nodes/result.py
"""
Eliminators for ``Result[T]``, at both sequence and element granularity.

A small, closed vocabulary for consuming ``Result[T]`` values, decided once
so that no combinator built on top of it has to grow its own error-handling
policy. See discussion #198 for the motivation and #200 / #234 (#235) /
``core/values/result.py`` for ``Result[T]`` itself.

``Seq[Result[T]]`` eliminators:

- ``PartitionNode``: splits into oks and errs, as two separate outputs.
- ``UnwrapOrNode``: collapses ``Seq[Result[T]]`` to ``Seq[T]`` using a
  caller-supplied default/marker in place of each err.
- ``AllOkNode``: the all-or-nothing collapse, ``Seq[Result[T]] -> Result[Seq[T]]``.
- ``FirstErrorNode``: the first err in the sequence, if any.

Element-level ``Result[T]`` eliminators (#235), added once #232 made
``Result[T]`` assignable only to ``Result``, which left a lone ``attempt``
outside a sequence with no legal downstream node:

- ``UnwrapNode``: ``Result[T] -> T``, failing the node on err. Inside a
  boundary this is exactly ``?``; outside any boundary it is exactly
  ``.unwrap()`` panicking, both falling out of the same node. See
  ``PropagatedResultError`` (``core/values/result.py``) for how it keeps the
  err arm's original provenance through the re-raise.
- ``UnwrapOrValueNode``: the element-level analogue of ``UnwrapOrNode``,
  ``Result[T]`` plus a default of type ``T`` -> ``T``.
- ``IsOkNode``: ``Result[T] -> bool``, for branching a graph on the tag by
  feeding ``If``/``IfElse``'s ``condition`` (see #235's PR description for
  why this shape was chosen over a two-armed match node).

None of these run or retry a workflow; that is ``attempt`` (#201), a separate
piece. The ``Seq``-level eliminators consume a sequence that already contains
``Result[T]`` elements, typically produced by ``for_each(attempt(w))``; the
element-level ones consume a lone ``Result[T]``, typically produced directly
by ``attempt(w)``.
"""

from typing import ClassVar, Generic, Type, TypeVar, cast

from overrides import override
from pydantic import Field
from pydantic.fields import FieldInfo

from ..core import (
    BooleanValue,
    Data,
    DataValue,
    Empty,
    ExecutionContext,
    IntegerValue,
    Node,
    NodeTypeInfo,
    NullValue,
    OptionalValue,
    PropagatedResultError,
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


################################################################################
# unwrap


class UnwrapNode(Node[Data, Data, Empty]):
    """
    ``Result[T] -> T``: the ok value, or fail the node on err.

    This is the value-granularity ``?``: ``attempt`` is already ``?`` at
    function granularity (fail-fast inside the boundary is early return, and
    the boundary is the function body), but nothing could re-raise a
    ``Result[T]`` that arrived on an edge from an earlier boundary back
    inside the current one, at the value it actually failed on, until this
    node existed. ``unwrap`` inside a boundary is exactly that re-raise;
    ``unwrap`` outside any boundary is exactly ``.unwrap()`` panicking, which
    fails the run. Both behaviors fall out of one node: this one never checks
    whether it is inside a boundary, it just raises, and the boundary
    machinery (or its absence) decides what that raise means.

    The dynamic input/output types build the field directly from
    ``self.element_type`` rather than declaring a ``Generic[V]`` ``Data``
    subclass, the same idiom ``UnwrapOrNode``'s ``default`` field uses: a
    generic ``Data`` subclass can't declare a bare-typevar field, since
    ``Data`` validates every field is a concrete ``Value`` type at
    class-definition time, before any parametrization.
    """

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Unwrap",
        description="Returns the ok value of a Result, failing the node on err.",
        version="1.0.0",
        parameter_type=Empty,
    )

    # The type of the ok element. For now, only available when the node is
    # constructed programmatically (see nodes/data.py for the same TODO).
    element_type: ValueType = Field(default=Value, exclude=True)

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> Type[Data]:
        return build_data_type(
            name="UnwrapInput",
            fields={
                "result": (
                    Result[self.element_type],
                    FieldInfo(
                        title="Result",
                        description="The Result to unwrap.",
                    ),
                ),
            },
        )

    @override
    async def dynamic_output_type(self, context: ValidationContext) -> Type[Data]:
        return build_data_type(
            name="UnwrapOutput",
            fields={
                "value": (
                    self.element_type,
                    FieldInfo(
                        title="Value",
                        description="The unwrapped ok value.",
                    ),
                ),
            },
        )

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Data],
        output_type: Type[Data],
        input: Data,
    ) -> Data:
        input_dict = get_data_dict(input)
        # get_data_dict()'s static return type is Mapping[str, Value]; the
        # actual runtime type of "result" is Result[Value] (it was just built
        # that way above), so cast rather than lie to pyright with an ignore
        # comment.
        result = cast(Result[Value], input_dict["result"])
        if result.is_ok():
            return output_type(**{"value": result.unwrap_ok()})
        # Re-raise carrying the original error_class/name/message/node_id
        # unchanged; see PropagatedResultError for why a fresh
        # WorkflowException can't do that through its own node_id field.
        raise PropagatedResultError(original=result.unwrap_err())


################################################################################
# unwrap_or (scalar)


class UnwrapOrValueNode(Node[Data, Data, Empty]):
    """
    ``Result[T]`` plus a default of type ``T`` -> ``T``: the element-level
    analogue of ``UnwrapOrNode``. Returns the ok value, or the caller-supplied
    default if err.

    Unlike ``unwrap``, this never fails: it is the total, no-boundary-needed
    way to consume a lone ``Result[T]`` when any placeholder value is an
    acceptable substitute for "this failed here." See ``UnwrapOrNode`` for
    the rationale behind requiring an explicit default rather than the engine
    inventing one.
    """

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Unwrap Or",
        description=(
            "Returns the ok value of a Result, or a default value in place of err."
        ),
        version="1.0.0",
        parameter_type=Empty,
    )

    # The type of the element. For now, only available when the node is
    # constructed programmatically (see nodes/data.py for the same TODO).
    element_type: ValueType = Field(default=Value, exclude=True)

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> Type[Data]:
        return build_data_type(
            name="UnwrapOrValueInput",
            fields={
                "result": (
                    Result[self.element_type],
                    FieldInfo(
                        title="Result",
                        description="The Result to unwrap.",
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
    async def dynamic_output_type(self, context: ValidationContext) -> Type[Data]:
        return build_data_type(
            name="UnwrapOrValueOutput",
            fields={
                "value": (
                    self.element_type,
                    FieldInfo(
                        title="Value",
                        description="The ok value, or the default if err.",
                    ),
                ),
            },
        )

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[Data],
        output_type: Type[Data],
        input: Data,
    ) -> Data:
        input_dict = get_data_dict(input)
        result = cast(Result[Value], input_dict["result"])
        default = input_dict["default"]
        value = result.unwrap_ok() if result.is_ok() else default
        return output_type(**{"value": value})


################################################################################
# is_ok (tag branch)


class IsOkData(Data, Generic[V]):
    """The single ``Result[T]`` input to ``is_ok``."""

    result: Result[V] = Field(
        title="Result",
        description="The Result to check.",
    )


class IsOkOutput(Data):
    """Whether a ``Result[T]`` was ok."""

    is_ok: BooleanValue = Field(
        title="Is Ok",
        description="True if the Result was ok, false if it was err.",
    )


class IsOkNode(Node[IsOkData, IsOkOutput, Empty]):
    """
    ``Result[T] -> bool``: true if ok, false if err.

    The tag branch for a lone ``Result[T]``: wire this node's ``is_ok``
    output into ``If``/``IfElse``'s ``condition`` to run a different inner
    workflow depending on the tag, the same conditional every other boolean
    branch in a graph already uses. Composes with ``And``/``Or``/``Not`` for
    a compound condition, for free, since the output is a plain
    ``BooleanValue`` rather than a bespoke branch shape.

    See the PR description for #235 for why this shape (a boolean feeding
    the existing conditional) was chosen over a two-armed ``match_result``
    node with typed ok/err ports.
    """

    TYPE_INFO: ClassVar[NodeTypeInfo] = NodeTypeInfo.from_parameter_type(
        display_name="Is Ok",
        description="Reports whether a Result is ok, to branch a graph on the tag.",
        version="1.0.0",
        parameter_type=Empty,
    )

    # The type of the element. For now, only available when the node is
    # constructed programmatically (see nodes/data.py for the same TODO).
    element_type: ValueType = Field(default=Value, exclude=True)

    @override
    async def dynamic_input_type(self, context: ValidationContext) -> Type[IsOkData]:
        return IsOkData[self.element_type]

    @classmethod
    @override
    def static_output_type(cls) -> Type[IsOkOutput]:
        return IsOkOutput

    @override
    async def run(
        self,
        *,
        context: ExecutionContext,
        input_type: Type[IsOkData],
        output_type: Type[IsOkOutput],
        input: IsOkData,
    ) -> IsOkOutput:
        return output_type(is_ok=BooleanValue(input.result.is_ok()))


__all__ = [
    "AllOkData",
    "AllOkNode",
    "FirstErrorData",
    "FirstErrorNode",
    "IsOkData",
    "IsOkNode",
    "IsOkOutput",
    "PartitionData",
    "PartitionNode",
    "UnwrapNode",
    "UnwrapOrNode",
    "UnwrapOrValueNode",
]
