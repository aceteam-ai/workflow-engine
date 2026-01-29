from collections.abc import MutableMapping
from typing import Any, Generic, Self, TypeVar
from pydantic import BaseModel, ConfigDict, RootModel


_immutable_model_config = ConfigDict(
    frozen=True,
    revalidate_instances="always",
    validate_assignment=True,
)


class _ImmutableMixin:
    """
    A base model that is immutable.
    """

    def __init_subclass__(cls, **kwargs):
        """
        Add the immutable model config to the rest of this class' model config.
        """
        cls.model_config = cls.model_config | _immutable_model_config
        super().__init_subclass__(**kwargs)


class ImmutableBaseModel(BaseModel, _ImmutableMixin):
    def model_update(self, **kwargs: Any) -> Self:
        """
        Returns a new copy of the model with the given kwargs updated, and
        validates the assignment.
        Contrary to common belief, setting revalidate_instances="always" and
        validate_assignment=True is not enough, because model_copy bypasses all
        of that.
        See https://github.com/pydantic/pydantic/issues/418
        """
        assert isinstance(self, BaseModel)
        updated = self.model_copy(update=kwargs)  # type: ignore
        self.__class__.model_validate(updated.__dict__)  # validate the updated model
        return updated

    def _model_mutate(self, **kwargs: Any):
        """
        Mutates the model in place despite its immutability.
        This is obviously dangerous, so the method is protected.
        To make it slightly safer, we validate the assignment via model_update.
        """
        updated = self.model_update(**kwargs)
        assert isinstance(self.__dict__, MutableMapping)
        self.__dict__.update(updated.__dict__)


T = TypeVar("T")


class ImmutableRootModel(RootModel[T], _ImmutableMixin, Generic[T]):
    pass


__all__ = [
    "ImmutableBaseModel",
    "ImmutableRootModel",
]
