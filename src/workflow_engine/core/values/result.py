# workflow_engine/core/values/result.py
"""
Result[T]: a tagged value type for fallible elements.

``Result[T]`` is ok or err. The ok arm carries a value of type ``T``; the err
arm carries a structured, closed-vocabulary error. Unlike ``UnionValue[T,
ErrValue]`` (see ``union.py``), a validated ``Result[T]`` is always an
instance of the single ``Result`` class, with the ok/err tag carried
explicitly in the serialized form. That is what makes ``Result[Result[T]]``
representable: each level keeps its own tag, so nothing is lost or collapsed
on the way to the wire and back.

The root is a Pydantic discriminated union of two plain models (``_OkRoot`` /
``_ErrRoot``) rather than a single model with two optional, sibling fields.
The earlier sibling-fields shape used Python ``None`` as the "this arm is
absent" sentinel, which is indistinguishable from a populated
``ok: NullValue`` payload (``NullValue``'s own wire form is also ``null``),
so ``Result[NullValue].ok(NullValue(None))`` failed to round-trip. The
discriminated union routes on ``tag`` first, so the payload field is only
ever validated against its own type, never against the sentinel.

See the published wire shape in schema/result.md and docs/values.md, and
discussion #198 for the motivation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Generic, Literal, Self, TypeVar, Union

from overrides import override
from pydantic import Field

from ...utils.model import ImmutableBaseModel
from .data import Data, get_data_schema
from .primitives import StringValue
from .value import Caster, Value, ValueType, get_origin_and_args

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from .schema import ValueSchema

T = TypeVar("T", bound=Value)


class ErrorClass(StrEnum):
    """
    The closed-vocabulary, machine-readable classification of a ``Result``
    err arm.

    A single field lets callers key retry policy, circuit breakers, and a run
    ledger off of one vocabulary instead of three.
    """

    TIMEOUT = "timeout"
    UNREACHABLE = "unreachable"
    RATE_LIMIT = "rate_limit"
    VALIDATION = "validation"
    PERMISSION = "permission"
    SYSTEMIC = "systemic"


class ErrorClassValue(Value[ErrorClass]):
    pass


@StringValue.register_cast_to(ErrorClassValue)
def cast_string_to_error_class(
    value: StringValue,
    context: "ExecutionContext",
) -> ErrorClassValue:
    return ErrorClassValue(ErrorClass(value.root))


class ResultError(Data):
    """The structured error carried by a ``Result[T]`` err arm."""

    error_class: ErrorClassValue = Field(
        title="Error Class",
        description=(
            "The machine-readable classification of the error, from a closed "
            "vocabulary: timeout, unreachable, rate_limit, validation, "
            "permission, or systemic."
        ),
    )
    name: StringValue = Field(
        title="Error Name",
        description="The short, machine-readable name of the error.",
    )
    message: StringValue = Field(
        title="Message",
        description="The user-facing description of what went wrong.",
    )
    node_id: StringValue = Field(
        title="Node ID",
        description="The identifier of the node that produced the error.",
    )


class _OkRoot(ImmutableBaseModel, Generic[T]):
    """The ok arm of a ``Result[T]``'s root. See ``Result`` for why this shape."""

    tag: Literal["ok"] = "ok"
    ok: T


class _ErrRoot(ImmutableBaseModel):
    """The err arm of a ``Result[T]``'s root. See ``Result`` for why this shape."""

    tag: Literal["err"] = "err"
    err: ResultError


def _item_type(cls: type[Value]) -> ValueType:
    origin, (item_type,) = get_origin_and_args(cls)
    assert issubclass(origin, Result)
    return item_type


class Result(
    Value[Annotated[Union[_OkRoot[T], _ErrRoot], Field(discriminator="tag")]],
    Generic[T],
):
    """
    A registered public value type: ok or err, tagged.

    The err arm carries a structured ``ResultError``: a closed-vocabulary
    ``error_class``, a short ``name``, a user-facing ``message``, and the
    producing node's id as provenance.

    Construct with ``Result[T].ok(value)`` / ``Result[T].err(error)`` rather
    than the RootModel constructor directly, so the tag always matches the
    payload.
    """

    @classmethod
    def ok(cls, value: T) -> Self:
        item_type = _item_type(cls)
        return cls(root=_OkRoot[item_type](ok=value))  # type: ignore[arg-type]

    @classmethod
    def err(cls, error: ResultError) -> Self:
        return cls(root=_ErrRoot(err=error))  # type: ignore[arg-type]

    def is_ok(self) -> bool:
        return self.root.tag == "ok"

    def is_err(self) -> bool:
        return self.root.tag == "err"

    def unwrap_ok(self) -> T:
        root = self.root
        if not isinstance(root, _OkRoot):
            raise ValueError(f"{self!r} is err, not ok")
        return root.ok

    def unwrap_err(self) -> ResultError:
        root = self.root
        if not isinstance(root, _ErrRoot):
            raise ValueError(f"{self!r} is ok, not err")
        return root.err

    @classmethod
    @override
    def to_value_schema(cls) -> "ValueSchema":
        from .schema import ResultValueSchema

        item_type = _item_type(cls)
        return ResultValueSchema(
            type="object",
            value_type=cls.__name__,
            ok=item_type.to_value_schema(),
            err=get_data_schema(ResultError),
        )


def result_value_type(item_type: ValueType) -> type[Result]:
    """Build ``Result[item_type]``. Used by ``ResultValueSchema.build_value_cls``."""
    return Result[item_type]


SourceType = TypeVar("SourceType", bound=Value)
TargetType = TypeVar("TargetType", bound=Value)


@Result.register_generic_cast_to(Result)
def cast_result_to_result(
    source_type: type[Result[SourceType]],
    target_type: type[Result[TargetType]],
) -> Caster[Result[SourceType], Result[TargetType]] | None:
    source_item_type = _item_type(source_type)
    target_item_type = _item_type(target_type)
    if not source_item_type.can_cast_to(target_item_type):
        return None

    async def _cast(
        value: Result[SourceType],
        context: "ExecutionContext",
    ) -> Result[TargetType]:
        if value.is_err():
            return target_type.err(value.unwrap_err())
        casted = await value.unwrap_ok().cast_to(target_item_type, context=context)
        return target_type.ok(casted)  # type: ignore[arg-type]

    return _cast


__all__ = [
    "ErrorClass",
    "ErrorClassValue",
    "Result",
    "ResultError",
    "result_value_type",
]
