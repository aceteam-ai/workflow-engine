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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal, Self, TypeVar, Union

from overrides import override
from pydantic import Field, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from ...utils.model import ImmutableBaseModel
from ..error import ErrorClass, WorkflowException
from ..stakeholder import StakeholderLevel
from .data import Data, get_data_schema
from .json import JSONValue
from .primitives import StringValue
from .value import Caster, Value, ValueType, get_origin_and_args

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from .schema import ValueSchema

T = TypeVar("T", bound=Value)


def _inline_refs(
    node: Any,
    scopes: tuple[Mapping[str, Any], ...] = (),
) -> Any:
    """
    Recursively resolve every ``$ref``/``$defs`` pair found anywhere in
    *node* (a raw JSON-Schema-shaped dict/list, e.g. from
    ``model_json_schema()``) against the nearest enclosing ``$defs``,
    substituting the referenced definition's own (recursively resolved)
    content in place, and drop every ``$defs`` key. The result is
    self-contained: no ``$ref`` remains anywhere.

    ``$defs`` is looked up innermost-first (the same order
    ``ReferenceValueSchema.build_value_cls()`` uses for its ``self.defs``,
    ``*extra_defs``), so a locally-scoped ``$defs`` shadows one further out.

    A sibling key already present next to a ``$ref`` (e.g. a field-level
    ``title``/``description`` override) is applied on top of a *copy* of the
    resolved definition, so it can never mutate, and therefore never leak
    into, another reference to the same shared definition. ``title`` is the
    one exception, and only when the resolved target is itself a record
    schema (has ``properties``): ``DataValueSchema.build_data_cls()`` names
    the class it rebuilds after that title, so a referencing site's title
    (e.g. an outer ``DataValue[D]`` wrapper's own Pydantic-generated title,
    which differs from ``D``'s own title) must not clobber it, unlike every
    other sibling key, which a referencing site's override is meant to
    apply. A record's fields keep their own field-level titles either way,
    since those are themselves referencing sites over leaf schemas, not
    over another record.
    """
    if isinstance(node, list):
        return [_inline_refs(item, scopes) for item in node]
    if not isinstance(node, dict):
        return node

    local_defs = node.get("$defs")
    if local_defs:
        scopes = (local_defs, *scopes)

    ref = node.get("$ref")
    if ref is not None:
        assert isinstance(ref, str) and ref.startswith("#/$defs/")
        name = ref.removeprefix("#/$defs/")
        for defs in scopes:
            if name in defs:
                target = _inline_refs(defs[name], scopes)
                sibling = {
                    k: _inline_refs(v, scopes)
                    for k, v in node.items()
                    if k not in ("$ref", "$defs")
                }
                merged = {**target, **sibling}
                if "properties" in target and "title" in target:
                    merged["title"] = target["title"]
                return merged
        raise KeyError(f"Schema definition for {name!r} not found")

    return {k: _inline_refs(v, scopes) for k, v in node.items() if k != "$defs"}


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


class PropagatedResultError(WorkflowException):
    """
    Re-raises a ``Result[T]`` err arm's ``ResultError`` from inside a
    boundary, carrying its ``error_class``, ``name``, ``message`` and
    ``node_id`` through unchanged.

    Raised by ``unwrap`` (``nodes/result.py``) rather than one of
    ``ResultError``'s own fields being turned back into a fresh exception's
    ``message``/``error_class``/``node_id``: ``Node.__call__`` unconditionally
    stamps a re-raised ``WorkflowException``'s ``node_id`` to the raising
    node's own id (and turns a mismatched pre-set id into an unrelated
    operator error), so there is no channel through the normal exception
    fields for ``unwrap``'s re-raise to carry someone else's ``node_id``.
    ``original`` is the side channel instead: ``result_error_from_exception``
    (``execution/boundary.py``) special-cases this type and hands ``original``
    back unchanged, rather than reconstructing a ``ResultError`` from this
    exception's own (necessarily unwrap-node-scoped) fields. ``unwrap``'s own
    ``on_node_error`` still fires with this exception's own ``node_id``, so
    the ledger still records the hop through ``unwrap``; only the
    materialized value keeps the root cause.

    Constructed at ``level=USER``: ``ResultError.message`` is already
    redaction-gated at the point the original error was materialized, so
    surfacing it again here leaks nothing new.
    """

    def __init__(self, *, original: ResultError):
        super().__init__(
            f"Unwrapped error from node '{original.node_id.root}': "
            f"{original.message.root}",
            level=StakeholderLevel.USER,
            error_class=ErrorClass(original.error_class.root),
        )
        self.original = original


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
        from .schema import ResultValueSchema, validate_value_schema

        item_type = _item_type(cls)
        schema = ResultValueSchema(
            type="object",
            value_type=cls.__name__,
            ok=item_type.to_value_schema(),
            err=get_data_schema(ResultError),
        )
        # get_data_schema(ResultError) (and, for a complex enough item type,
        # item_type.to_value_schema() too) is Pydantic's own $ref/$defs
        # rendering, generated by an independent model_json_schema() call.
        # Inline it away so this schema, and therefore
        # __get_pydantic_json_schema__ below (which returns this unchanged),
        # never contains a $ref: a caller reading either publication of this
        # type sees one shape, not a $ref-based one from one path and an
        # inlined one from the other.
        inlined = _inline_refs(schema.model_dump(mode="json", by_alias=True))
        return validate_value_schema(inlined)

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """
        Emit the published ok/err wire shape (schema/result.md) instead of
        Pydantic's default rendering of the ``_OkRoot`` / ``_ErrRoot``
        discriminated union, so ``model_json_schema()`` agrees with
        ``to_value_schema()`` instead of publishing a second, different
        shape for the same type.

        Deliberately does not compose the ok/err arms by calling ``handler()``
        on pieces of *core_schema*. That works for a single level, but for
        nested ``Result[Result[T]]`` the inner ``Result[T]``'s core schema
        shows up, by the time it would be handed to ``handler()`` here, as an
        inlined model with no metadata attached, so this hook never fires on
        it and the inner arm falls back to Pydantic's default shape. Calling
        ``to_value_schema()`` sidesteps that entirely: it walks the item type
        in plain Python (``item_type.to_value_schema()``), not through
        Pydantic's core-schema tree, so it reaches every nesting depth the
        same way regardless of what Pydantic's own schema generation does
        with the class at that position.

        Returned as-is, with no further transformation: ``to_value_schema()``
        is already fully inlined (see above), so there is no ``$ref``
        anywhere in this output for Pydantic's own ``$ref`` bookkeeping to
        choke on when it scans the assembled document, and this is
        byte-identical to ``to_value_schema().model_dump(mode="json",
        by_alias=True)`` by construction, not by coincidence.
        """
        return cls.to_value_schema().model_dump(mode="json", by_alias=True)


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


@Result.register_generic_cast_to(StringValue)
def _no_cast_result_to_string(
    source_type: type[Result[SourceType]],
    target_type: type[StringValue],
) -> Caster[Result[SourceType], StringValue] | None:
    """
    Shadow ``Value``'s blanket "stringify anything" cast (``primitives.py``)
    so it never reaches a ``Result``. ``_get_casters`` resolves a child
    class's own registrations before a parent's, so registering this here,
    directly on ``Result``, is what makes the blanket cast unreachable for
    every ``Result[T]`` regardless of ``T``; nothing about the registry or
    the blanket casters themselves changes. Without this, a ``Result``-typed
    edge into a ``StringValue`` input passes graph validation and both the ok
    and the err arm arrive downstream as a Pydantic repr instead of the
    caller ever learning the cast was unsound (#232).
    """
    return None


@Result.register_generic_cast_to(JSONValue)
def _no_cast_result_to_json(
    source_type: type[Result[SourceType]],
    target_type: type[JSONValue],
) -> Caster[Result[SourceType], JSONValue] | None:
    """
    Shadow ``Value``'s blanket "JSON-ify anything" cast (``json.py``) for the
    same reason as ``_no_cast_result_to_string`` above. This is unrelated to
    JSON serialization at the HTTP boundary, which goes through
    ``model_dump`` and never touches the cast table (#232).
    """
    return None


__all__ = [
    "ErrorClass",
    "ErrorClassValue",
    "PropagatedResultError",
    "Result",
    "ResultError",
    "result_value_type",
]
