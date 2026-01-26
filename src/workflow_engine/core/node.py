# workflow_engine/core/node.py
import asyncio
import logging
import re
import warnings
from collections.abc import Mapping
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    ClassVar,
    Generic,
    Literal,
    Self,
    Type,
    TypeVar,
    Unpack,
    get_origin,
)

from overrides import final
from pydantic import ConfigDict, Field, ValidationError, model_validator

from ..utils.immutable import ImmutableBaseModel
from ..utils.semver import (
    LATEST_SEMANTIC_VERSION,
    SEMANTIC_VERSION_OR_LATEST_PATTERN,
    SEMANTIC_VERSION_PATTERN,
    parse_semantic_version,
)
from .error import NodeException, UserException
from .values import (
    Data,
    DataMapping,
    Value,
    ValueSchema,
    ValueType,
    get_data_fields,
)
from .values.data import Input_contra, Output_co

if TYPE_CHECKING:
    from .context import Context
    from .workflow import Workflow

logger = logging.getLogger(__name__)


class Params(Data):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="allow",
        frozen=True,
    )

    # The base class has extra="allow", so that it can be deserialized into any
    # of its subclasses. However, subclasses should set extra="forbid" to block
    # extra fields.
    def __init_subclass__(cls, **kwargs):
        cls.model_config["extra"] = "forbid"
        super().__init_subclass__(**kwargs)


Params_co = TypeVar("Params_co", bound=Params, covariant=True)
T = TypeVar("T")


@final
class Empty(Params):
    """
    A Data and Params class that is explicitly not allowed to have any
    parameters.
    """

    pass


generic_pattern = re.compile(r"^[a-zA-Z]\w+\[.*\]$")


class NodeTypeInfo(ImmutableBaseModel):
    """
    Information about a node type, in serializable form.
    """

    name: str = Field(
        description="A unique name for the node type, which should be a literal string for concrete subclasses."
    )
    display_name: str = Field(
        description="A human-readable display name for the node, which may or may not be unique."
    )
    description: str | None = Field(
        description="A human-readable description of the node type."
    )
    version: str = Field(
        description="A 3-part version number for the node, following semantic versioning rules (see https://semver.org/).",
        pattern=SEMANTIC_VERSION_PATTERN,
    )
    parameter_schema: ValueSchema = Field(
        default_factory=lambda: Empty.to_value_schema(),
        description="The schema for the parameters of the node type.",
    )
    max_retries: int | None = Field(
        default=None,
        description="Maximum number of retry attempts for this node type. "
        "None means use the execution algorithm's default.",
    )

    @cached_property
    def version_tuple(self) -> tuple[int, int, int]:
        return parse_semantic_version(self.version)

    @classmethod
    def from_parameter_type(
        cls,
        *,
        name: str,
        display_name: str,
        description: str | None = None,
        version: str,
        parameter_type: Type[Params],
        max_retries: int | None = None,
    ) -> Self:
        return cls(
            name=name,
            display_name=display_name,
            description=description,
            version=version,
            parameter_schema=parameter_type.to_value_schema(),
            max_retries=max_retries,
        )


class Node(ImmutableBaseModel, Generic[Input_contra, Output_co, Params_co]):
    """
    A data processing node in a workflow.
    Nodes have three sets of fields:
    - parameter fields must be provided when defining the workflow
    - input fields are provided when executing the workflow
    - output fields are produced by the node if it executes successfully
    """

    # Allow extra fields, such as position or appearance information.
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    # Must be annotated as ClassVar[NodeTypeInfo] when overriding.
    # Does not have a value here, since the base Node class is not meant to be
    # instantiated except very temporarily in dispatches.
    TYPE_INFO: ClassVar[NodeTypeInfo]

    type: str = Field(
        description=(
            "The type of the node, which should be a literal string for discriminating concrete subclasses. "
            "Used to determine which node class to load."
        ),
    )
    version: str = Field(
        pattern=SEMANTIC_VERSION_OR_LATEST_PATTERN,
        description=(
            "A 3-part version number for the node, following semantic versioning rules (see https://semver.org/). "
            "There is no guarantee that outdated versions will load successfully. "
            "If not provided, it will default to the current version of the node type."
        ),
        default=LATEST_SEMANTIC_VERSION,
    )
    id: str = Field(
        description="The ID of the node, which must be unique within the workflow."
    )
    params: Params_co = Field(
        default_factory=Empty,  # type: ignore
        description=(
            "Any parameters for the node which are independent of the workflow inputs. "
            "May affect what inputs are accepted by the node."
        ),
    )

    # --------------------------------------------------------------------------
    # SUBCLASS DISPATCH
    # We use this trick to allow Node.model_validate to deserialize nodes as
    # their registered subclasses, using the type field as a discriminator to
    # select the appropriate subclass.
    # Node subclasses are registered automatically when they are defined.

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]):
        super().__init_subclass__(**kwargs)  # type: ignore

        # NOTE: something about this hack does not work when using
        # `from __future__ import annotations`.
        while generic_pattern.match(cls.__name__) is not None:
            assert cls.__base__ is not None
            cls = cls.__base__
        type_annotation = cls.__annotations__.get("type", None)
        if type_annotation is None or get_origin(type_annotation) is not Literal:
            _registry.register_base(cls)
        else:
            (type_name,) = type_annotation.__args__
            assert isinstance(type_name, str), type_name
            _registry.register(type_name, cls)

    @model_validator(mode="after")  # type: ignore
    def _to_subclass(self):
        """
        Replaces the Node object with an instance of the registered subclass.
        """
        # HACK: This trick only works if the base class can be instantiated, so
        # we cannot make it an ABC even if it has unimplemented methods.
        if _registry.is_base_class(self.__class__):
            cls = _registry.get(self.type)
            if cls is None:
                raise ValueError(f'Node type "{self.type}" is not registered')
            data = self.model_dump()
            # Attempt migration before dispatching to subclass
            data = _migrate_node_data(data, cls)
            return cls.model_validate(data)
        if self.__class__ is Node:
            warnings.warn(
                f"Node validation for node {self} could not find a registered subclass to dispatch to."
            )
        return self

    # --------------------------------------------------------------------------
    # NAMING

    async def display_name(self, context: "Context") -> str:
        """
        A human-readable display name for the node, which is not necessarily
        unique.
        By default, it is the node type's display name, which is a poor default
        at best.
        You should override this method to provide a more meaningful name and
        disambiguate nodes with the same type.

        This method is async in case determining the node name requires some
        asynchronous work, and can use the context.
        """
        return self.TYPE_INFO.display_name

    def with_namespace(self, namespace: str) -> Self:
        """
        Create a copy of this node with a namespaced ID.

        Args:
            namespace: The namespace to prefix the node ID with

        Returns:
            A new Node with ID '{namespace}/{self.id}'
        """
        return self.model_update(id=f"{namespace}/{self.id}")

    # --------------------------------------------------------------------------
    # VERSIONING

    @property
    def version_tuple(self) -> tuple[int, int, int]:
        """
        The major, minor, and patch version numbers of the node version.
        If the node version is not provided, this will crash.
        """
        return parse_semantic_version(self.version)

    @model_validator(mode="after")
    def validate_version(self):
        """
        Sets the node version to the current version of the node type.
        Validates the node version against the TYPE_INFO version.
        """
        # skip validation for the base Node class, which lacks a TYPE_INFO
        if self.__class__ is Node:
            return self

        type_info = self.__class__.TYPE_INFO
        if self.version == LATEST_SEMANTIC_VERSION:
            self._model_mutate(version=type_info.version)
        elif type_info.version_tuple < self.version_tuple:
            raise ValueError(
                f"Node version {self.version} is newer than the latest version ({type_info.version}) supported by this workflow engine instance."
            )
        elif type_info.version_tuple > self.version_tuple:
            # Migration was attempted in _to_subclass but no migration path exists.
            # Issue a warning but allow the node to load (graceful degradation).
            warnings.warn(
                f"Node version {self.version} is older than the latest version ({type_info.version}) supported by this workflow engine instance, and may need to be migrated."
            )
        return self

    # --------------------------------------------------------------------------
    # TYPING

    @property
    def input_type(self) -> Type[Input_contra]:  # type: ignore (contravariant return type)
        # return Empty to spare users from having to specify the input type on
        # nodes that don't have any input fields
        return Empty  # type: ignore

    @property
    def output_type(self) -> Type[Output_co]:
        # return Empty to spare users from having to specify the output type on
        # nodes that don't have any output fields
        return Empty  # type: ignore

    @property
    def input_fields(self) -> Mapping[str, tuple[ValueType, bool]]:  # type: ignore
        return get_data_fields(self.input_type)

    @property
    def output_fields(self) -> Mapping[str, tuple[ValueType, bool]]:
        return get_data_fields(self.output_type)

    @property
    def input_schema(self) -> ValueSchema:
        return self.input_type.to_value_schema()

    @property
    def output_schema(self) -> ValueSchema:
        return self.output_type.to_value_schema()

    # --------------------------------------------------------------------------
    # EXECUTION

    async def _cast_input(
        self,
        input: DataMapping,
        context: "Context",
    ) -> Input_contra:  # type: ignore (contravariant return type)
        allow_extra_input = (
            self.input_type.model_config.get("extra", "forbid") == "allow"
        )

        # Validate all inputs first
        for key, value in input.items():
            if key not in self.input_fields and allow_extra_input:
                continue
            input_type, _ = self.input_fields[key]
            if not value.can_cast_to(input_type):
                raise UserException(
                    f"Input {value} for node {self.id} is invalid: {value} is not assignable to {input_type}"
                )

        # Cast all inputs in parallel
        cast_tasks: list[Awaitable[Value]] = []
        keys: list[str] = []
        for key, value in input.items():
            if key not in self.input_fields and allow_extra_input:
                continue
            input_type, _ = self.input_fields[key]  # type: ignore
            cast_tasks.append(value.cast_to(input_type, context=context))
            keys.append(key)

        casted_values = await asyncio.gather(*cast_tasks)

        # Build the result dictionary
        casted_input: dict[str, Value] = {}
        for key, casted_value in zip(keys, casted_values):
            casted_input[key] = casted_value

        try:
            return self.input_type.model_validate(casted_input)
        except ValidationError as e:
            raise UserException(
                f"Input {casted_input} for node {self.id} is invalid: {e}"
            )

    # @abstractmethod
    async def run(
        self,
        context: "Context",
        input: Input_contra,
    ) -> "Output_co | Workflow":
        """
        Computes the node's outputs based on its inputs.
        Subclasses must implement this method, but it is not marked as abstract
        because the base Node class needs to be instantiable for dispatching.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @final
    async def __call__(
        self,
        context: "Context",
        input: DataMapping,
    ) -> "DataMapping | Workflow":
        """
        Executes the node.
        """
        try:
            logger.info("Starting node %s", self.id)
            output = await context.on_node_start(node=self, input=input)
            if output is not None:
                return output
            try:
                input_obj = await self._cast_input(input, context)
            except ValidationError as e:
                raise UserException(f"Input {input} for node {self.id} is invalid: {e}")
            output_obj = await self.run(context, input_obj)

            from .workflow import Workflow  # lazy to avoid circular import

            if isinstance(output_obj, Workflow):
                output = output_obj
                # TODO: once that workflow eventually finishes running, its
                # output should be the output of this node, and we should call
                # context.on_node_finish.
            else:
                output = await context.on_node_finish(
                    node=self,
                    input=input,
                    output=output_obj.to_dict(),
                )
            logger.info("Finished node %s", self.id)
            return output
        except Exception as e:
            # In subclasses, you don't have to worry about logging the error,
            # since it'll be logged here.
            logger.exception("Error in node %s", self.id)
            e = await context.on_node_error(node=self, input=input, exception=e)
            if isinstance(e, Mapping):
                logger.exception(
                    "Error absorbed by context and replaced with output %s", e
                )
                return e
            else:
                assert isinstance(e, Exception)
                raise NodeException(self.id) from e


def _migrate_node_data(data: dict[str, Any], target_cls: Type["Node"]) -> dict[str, Any]:
    """
    Attempt to migrate node data to the target class's version.

    This function is called during node deserialization, before the data
    is validated by the target class. If migration is needed and a migration
    path exists, the data is transformed. Otherwise, the original data is
    returned (graceful degradation - validation warnings will be issued later).

    Args:
        data: Raw node data dict with 'type', 'version', 'id', 'params', etc.
        target_cls: The concrete Node subclass to migrate to

    Returns:
        Migrated data dict, or original data if no migration needed/available
    """
    # Skip if target class doesn't have TYPE_INFO (shouldn't happen for concrete classes)
    if not hasattr(target_cls, "TYPE_INFO"):
        return data

    current_version = data.get("version", LATEST_SEMANTIC_VERSION)

    # Skip if using "latest" version (will be resolved to current version later)
    if current_version == LATEST_SEMANTIC_VERSION:
        return data

    type_info = target_cls.TYPE_INFO
    target_version = type_info.version

    # Skip if versions match
    if current_version == target_version:
        return data

    try:
        current_tuple = parse_semantic_version(current_version)
    except ValueError:
        # Invalid version format, let validation handle it
        return data

    # Only migrate if current version is older
    if current_tuple >= type_info.version_tuple:
        return data

    # Attempt migration
    # Import here to avoid circular imports
    from .migration import MigrationNotFoundError, migration_runner

    try:
        migrated_data = migration_runner.migrate(data, target_version)
        logger.debug(
            "Migrated node %s from version %s to %s",
            data.get("id"),
            current_version,
            target_version,
        )
        return migrated_data
    except MigrationNotFoundError:
        # No migration path available - return original data
        # Warning will be issued in validate_version
        logger.debug(
            "No migration path found for node %s from version %s to %s",
            data.get("id"),
            current_version,
            target_version,
        )
        return data


class NodeRegistry:
    def __init__(self):
        self.types: dict[str, Type[Node]] = {}
        self.base_classes: list[Type[Node]] = []

    def register(self, type: str, cls: Type[Node]):
        if type in self.types:
            conflict = self.types[type]
            if cls is not conflict:
                raise ValueError(
                    f'Node type "{type}" (class {cls.__name__}) is already registered to a different class ({conflict.__name__})'
                )
        self.types[type] = cls
        logger.debug("Registering class %s as node type %s", cls.__name__, type)

    def get(self, type: str) -> Type[Node]:
        if type not in self.types:
            raise ValueError(f'Node type "{type}" is not registered')
        return self.types[type]

    def register_base(self, cls: Type[Node]):
        if cls not in self.base_classes:
            self.base_classes.append(cls)
            logger.debug("Registering class %s as base node type", cls.__name__)

    def is_base_class(self, cls: Type[Node]) -> bool:
        return cls in self.base_classes


_registry = NodeRegistry()


__all__ = [
    "Empty",
    "Node",
    "NodeTypeInfo",
    "Params",
]
