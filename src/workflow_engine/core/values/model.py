# workflow_engine/core/values/model.py

from typing import TYPE_CHECKING, Generic, Type, TypeVar

from pydantic import BaseModel

from .json import JSONValue
from .value import Caster, Value, get_origin_and_args

if TYPE_CHECKING:
    from ..context import Context

M = TypeVar("M", bound=BaseModel)


class ModelValue(Value[M], Generic[M]):
    """Value wrapping a Pydantic BaseModel, validated at edge boundaries."""

    pass


SourceType = TypeVar("SourceType", bound=Value)
TargetType = TypeVar("TargetType", bound=Value)


@JSONValue.register_generic_cast_to(ModelValue)
def cast_json_to_model(
    source_type: Type[JSONValue],
    target_type: Type[ModelValue],
) -> Caster[JSONValue, ModelValue] | None:
    _origin, args = get_origin_and_args(target_type)
    if not args:
        # Unparameterized ModelValue — cannot validate without a model class
        return None

    model_cls = args[0]
    if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
        return None

    def _cast(value: JSONValue, context: "Context") -> ModelValue:
        validated = model_cls.model_validate(value.root)
        return target_type(validated)

    return _cast


@ModelValue.register_generic_cast_to(ModelValue)
def cast_model_to_model(
    source_type: Type[ModelValue],
    target_type: Type[ModelValue],
) -> Caster[ModelValue, ModelValue] | None:
    _source_origin, source_args = get_origin_and_args(source_type)
    _target_origin, target_args = get_origin_and_args(target_type)

    if not source_args or not target_args:
        return None

    source_model_cls = source_args[0]
    target_model_cls = target_args[0]

    if not (isinstance(source_model_cls, type) and issubclass(source_model_cls, BaseModel)):
        return None
    if not (isinstance(target_model_cls, type) and issubclass(target_model_cls, BaseModel)):
        return None

    # Identity shortcut
    if source_model_cls is target_model_cls:

        def _identity(value: ModelValue, context: "Context") -> ModelValue:
            return target_type(value.root)

        return _identity

    # Cross-model cast: serialize then validate
    def _cast(value: ModelValue, context: "Context") -> ModelValue:
        dumped = source_model_cls.model_validate(value.root).model_dump()
        validated = target_model_cls.model_validate(dumped)
        return target_type(validated)

    return _cast


__all__ = [
    "ModelValue",
]
