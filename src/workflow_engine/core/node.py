# workflow_engine/core/node.py
from __future__ import annotations

import asyncio
import logging
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    ClassVar,
    Generic,
    Iterable,
    Literal,
    Self,
    TypeVar,
    Unpack,
    get_origin,
)

from overrides import final, override
from pydantic import ConfigDict, Field, ValidationError, model_validator
from typing_extensions import overload

from ..utils.immutable import ImmutableBaseModel
from ..utils.semver import (
    LATEST_SEMANTIC_VERSION,
    SEMANTIC_VERSION_OR_LATEST_PATTERN,
    SEMANTIC_VERSION_PATTERN,
    parse_semantic_version,
)
from .error import NodeException, ShouldYield, UserException
from .values import (
    Data,
    DataMapping,
    Value,
    ValueSchema,
    ValueType,
    get_data_fields,
)
from .values.data import Input_contra, Output_co, get_data_dict, get_data_schema

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
        default_factory=lambda: get_data_schema(Empty),
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
        parameter_type: type[Params],
        max_retries: int | None = None,
    ) -> Self:
        return cls(
            name=name,
            display_name=display_name,
            description=description,
            version=version,
            parameter_schema=get_data_schema(parameter_type),
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
    # This trick to adds all defined Node subclasses to a defualt NodeRegistry.
    # Once in the registry, we can look up the value of the .type field's
    # annotation to determine the corresponding class.
    # Classes without a .type field annotation, like the Node class itself, are
    # registered as base classes which can be dispatched to any of their
    # subclasses.
    # annotation on the .type field to determine corresponding class.
    # However, this is brittle to the construction of multiple conflicting Node
    # subclasses with the same type name, which can happen when using multiple
    # packages that all share workflow_engine as a dependency.
    # To overcome this, the defualt NodeRegistry is lazy and does not construct
    # its internal mappings until it is actually used.
    # You can always substitute your own manually constructed NodeRegistry to
    # avoid these issues -- in fact, that is the recommended approach even if
    # you do not have collisions.

    @classmethod
    def _concrete_type_name(cls) -> str | None:
        while generic_pattern.match(cls.__name__) is not None:
            assert cls.__base__ is not None
            cls = cls.__base__
        type_annotation = cls.__annotations__.get("type", None)
        if type_annotation is None or get_origin(type_annotation) is not Literal:
            return None
        else:
            (type_name,) = type_annotation.__args__
            assert isinstance(type_name, str), type_name
            return type_name

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]):
        super().__init_subclass__(**kwargs)  # type: ignore

        NodeRegistry.DEFAULT.register_node_class(cls)

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

    @cached_property
    def input_type(self) -> type[Input_contra]:  # type: ignore (contravariant return type)
        """
        The type of the input data for this node.
        This field must always be cached to ensure referential equality on every
        call; otherwise we will be constructing instances of different types.
        """
        # return Empty to spare users from having to specify the input type on
        # nodes that don't have any input fields
        return Empty  # type: ignore

    @cached_property
    def output_type(self) -> type[Output_co]:
        """
        The type of the output data for this node.
        This field must always be cached to ensure referential equality on every
        call; otherwise we will be constructing instances of different types.
        """
        # return Empty to spare users from having to specify the output type on
        # nodes that don't have any output fields
        return Empty  # type: ignore

    @cached_property
    def input_fields(self) -> Mapping[str, tuple[ValueType, bool]]:  # type: ignore
        return get_data_fields(self.input_type)

    @cached_property
    def output_fields(self) -> Mapping[str, tuple[ValueType, bool]]:
        return get_data_fields(self.output_type)

    @cached_property
    def input_schema(self) -> ValueSchema:
        return get_data_schema(self.input_type)

    @cached_property
    def output_schema(self) -> ValueSchema:
        return get_data_schema(self.output_type)

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
            try:
                input_obj = await self._cast_input(input, context)
            except ValidationError as e:
                raise UserException(f"Input {input} for node {self.id} is invalid: {e}")
            output = await context.on_node_start(node=self, input=get_data_dict(input_obj))
            if output is not None:
                return output
            output_obj = await self.run(context, input_obj)

            from .workflow import Workflow  # lazy to avoid circular import

            if isinstance(output_obj, Workflow):
                output = await context.on_node_expand(
                    node=self,
                    input=input,
                    workflow=output_obj,
                )
                # TODO: once that workflow eventually finishes running, its
                # output should be the output of this node, and we should call
                # context.on_node_finish.
            else:
                output = await context.on_node_finish(
                    node=self,
                    input=input,
                    output=get_data_dict(output_obj),
                )
            logger.info("Finished node %s", self.id)
            return output
        except ShouldYield:
            raise
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


def _migrate_node_data(
    data: Mapping[str, Any], target_cls: type[Node]
) -> Mapping[str, Any]:
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


class NodeRegistry(ABC):
    """
    An immutable registry of two types of registrations:
    - Node types: A mapping from names to node classes.
    - Base node classes: A class that should be polymorphically replaced with a
      subtype from this registry. Obviously, `Node` itself is a base node class,
      but you may encounter more.
    """

    DEFAULT: ClassVar[LazyNodeRegistry]

    @abstractmethod
    def get_node_class(self, name: str) -> type[Node]:
        """
        Get a node class by name and version.
        """
        raise NotImplementedError()

    @abstractmethod
    def is_base_node_class(self, base_node_cls: type[Node]) -> bool:
        """
        Check if a class is a base node class.
        """
        raise NotImplementedError()

    @abstractmethod
    def all_concrete_node_classes(self) -> Iterable[tuple[str, type[Node]]]:
        """
        Return all registered concrete node classes, excluding base node classes.
        """
        raise NotImplementedError()

    @abstractmethod
    def all_base_node_classes(self) -> Iterable[type[Node]]:
        """
        Return all registered base node classes.
        """
        raise NotImplementedError()

    def load_node(self, node: Node) -> Node:
        """
        Load an untyped node into its concrete typed form.

        If the node is already a concrete type (not a base class), returns it unchanged.
        Otherwise, looks up the concrete type, applies migrations, and returns a typed instance.

        Args:
            node: An untyped node (base Node class instance)

        Returns:
            A typed node (concrete subclass instance)

        Raises:
            ValueError: If the node type is not registered
        """
        if not self.is_base_node_class(node.__class__):
            return node

        cls = self.get_node_class(node.type)
        if cls is None:
            raise ValueError(f'Node type "{node.type}" is not registered')

        if not issubclass(cls, node.__class__):
            warnings.warn(
                f"Node {node.id} will be reloaded as an instance of {cls.__name__}, which is not a subclass of its current class {node.__class__.__name__}. This is not a bug, just unusual."
            )

        data = node.model_dump()
        # Attempt migration before dispatching to subclass
        data = _migrate_node_data(data, cls)
        return cls.model_validate(data)

    @overload
    @staticmethod
    def builder(*, lazy: Literal[True]) -> LazyNodeRegistry: ...

    @overload
    @staticmethod
    def builder(*, lazy: Literal[False] = False) -> EagerNodeRegistryBuilder: ...

    @staticmethod
    def builder(*, lazy: bool = False):
        return LazyNodeRegistry() if lazy else EagerNodeRegistryBuilder()

    @overload
    def extend(self, *, lazy: Literal[True]) -> LazyNodeRegistry: ...

    @overload
    def extend(self, *, lazy: Literal[False] = False) -> EagerNodeRegistryBuilder: ...

    def extend(self, *, lazy: bool = False):
        """
        Extend the registry with a new builder.
        """
        builder = NodeRegistry.builder(lazy=lazy)
        for _, cls in self.all_concrete_node_classes():
            builder.register_node_class(cls)
        for cls in self.all_base_node_classes():
            builder.register_node_class(cls)
        return builder


class NodeRegistryBuilder(ABC):
    """
    An immutable builder for a mapping from node types to node
    classes, allowing us to replace base node classes with their subtypes by
    looking up their identifiers.
    """

    @abstractmethod
    def register_node_class(self, node_cls: type[Node]) -> Self:
        pass

    @abstractmethod
    def remove_node_class(self, node_cls: type[Node], *, missing_ok: bool = False) -> Self:
        pass

    @abstractmethod
    def build(self) -> NodeRegistry:
        pass


class ImmutableNodeRegistry(NodeRegistry):
    """
    An ImmutableNodeRegistry is a NodeRegistry that is immutable after
    construction; enforced by shallow copying the input data structures.
    """

    def __init__(
        self,
        *,
        node_classes: Mapping[str, type[Node]],
        base_node_classes: Collection[type[Node]],
    ):
        self._node_classes = dict(node_classes)
        self._base_node_classes = tuple(base_node_classes)

    @override
    def get_node_class(self, name: str) -> type[Node]:
        return self._node_classes[name]

    @override
    def is_base_node_class(self, base_node_cls: type[Node]) -> bool:
        return base_node_cls in self._base_node_classes

    @override
    def all_base_node_classes(self) -> Iterable[type[Node]]:
        return self._base_node_classes

    @override
    def all_concrete_node_classes(self) -> Iterable[tuple[str, type[Node]]]:
        return self._node_classes.items()


class EagerNodeRegistryBuilder(NodeRegistryBuilder):
    def __init__(self):
        self._node_classes: dict[str, type[Node]] = {}
        self._base_node_classes: list[type[Node]] = []

    @override
    def register_node_class(self, node_cls: type[Node]) -> Self:
        name = node_cls._concrete_type_name()
        if name is not None and name in self._node_classes:
            raise ValueError(f'Node type "{name}" is already registered')
        if name is None:
            self._base_node_classes.append(node_cls)
        else:
            self._node_classes[name] = node_cls
        return self

    @override
    def remove_node_class(self, node_cls: type[Node], *, missing_ok: bool = False) -> Self:
        name = node_cls._concrete_type_name()
        if name is None:
            if node_cls not in self._base_node_classes:
                if not missing_ok:
                    raise ValueError(f'Base node class "{node_cls.__name__}" is not registered')
            else:
                self._base_node_classes.remove(node_cls)
        else:
            if name not in self._node_classes:
                if not missing_ok:
                    raise ValueError(f'Node type "{name}" is not registered')
            else:
                del self._node_classes[name]
        return self

    @override
    def build(self) -> NodeRegistry:
        return ImmutableNodeRegistry(
            node_classes=self._node_classes,
            base_node_classes=self._base_node_classes,
        )


class LazyNodeRegistry(NodeRegistry, NodeRegistryBuilder):
    """
    A LazyNodeRegistry is both a NodeRegistry and a NodeRegistryBuilder with a
    2-part lifecycle:
    1. Registration phase: acts as a NodeRegistryBuilder until .build() is
       called.
    2. Registry phase: acts as a NodeRegistry.

    Validations are deferred until the registry is frozen; this allows for a
    LazyNodeRegistry to exist as long as it is not used, making it suitable for
    automatic node registration.
    """

    def __init__(self):
        self._registrations: list[type[Node]] = []
        self._removals: dict[type[Node], bool] = {}  # cls -> missing_ok
        self._frozen: bool = False

        # initialized after freeze
        self._node_classes: Mapping[str, type[Node]]
        self._base_node_classes: Collection[type[Node]]

    # REGISTRATION PHASE

    @override
    def register_node_class(self, node_cls: type[Node]) -> Self:
        if self._frozen:
            raise ValueError("Node registry is frozen, cannot register new node types.")
        self._registrations.append(node_cls)
        return self

    @override
    def remove_node_class(self, node_cls: type[Node], *, missing_ok: bool = False) -> Self:
        if self._frozen:
            raise ValueError("Node registry is frozen, cannot remove node types.")
        self._removals[node_cls] = missing_ok
        return self

    @override
    def build(self) -> NodeRegistry:
        if self._frozen:
            return self

        self._frozen = True
        _node_classes = {}
        _base_node_classes = []
        for node_cls in self._registrations:
            name = node_cls._concrete_type_name()
            if name is None:
                if node_cls not in _base_node_classes:
                    _base_node_classes.append(node_cls)
                else:
                    logger.warning(
                        "Node base class %s is already registered, skipping registration",
                        node_cls.__name__,
                    )
            else:
                if name in _node_classes:
                    conflict = _node_classes[name]
                    if node_cls is conflict:
                        logger.warning(
                            "Node type %s is already registered to class %s, skipping registration",
                            name,
                            node_cls.__name__,
                        )
                    else:
                        raise ValueError(
                            f'Node type "{name}" (class {node_cls.__name__}) is already registered to a different class ({conflict.__name__})'
                        )
                else:
                    _node_classes[name] = node_cls

        for node_cls, missing_ok in self._removals.items():
            name = node_cls._concrete_type_name()
            if name is None:
                if node_cls not in _base_node_classes:
                    if not missing_ok:
                        raise ValueError(f'Base node class "{node_cls.__name__}" is not registered')
                else:
                    _base_node_classes.remove(node_cls)
            else:
                if name not in _node_classes:
                    if not missing_ok:
                        raise ValueError(f'Node type "{name}" is not registered')
                else:
                    del _node_classes[name]

        del self._registrations  # just to be extra sure
        del self._removals
        self._node_classes = _node_classes
        self._base_node_classes = tuple(_base_node_classes)

        return self

    # USE PHASE

    @override
    def get_node_class(self, name: str) -> type[Node]:
        self.build()
        if name not in self._node_classes:
            raise ValueError(f'Node type "{name}" is not registered')
        return self._node_classes[name]

    @override
    def is_base_node_class(self, base_node_cls: type[Node]) -> bool:
        self.build()
        return base_node_cls in self._base_node_classes

    @override
    def all_concrete_node_classes(self) -> Iterable[tuple[str, type[Node]]]:
        return self._node_classes.items()

    @override
    def all_base_node_classes(self) -> Iterable[type[Node]]:
        return self._base_node_classes


# Register Node itself as a base node class, which can be dispatched to any of
# its subclasses.
NodeRegistry.DEFAULT = NodeRegistry.builder(lazy=True).register_node_class(Node)


__all__ = [
    "Empty",
    "Node",
    "NodeTypeInfo",
    "Params",
]
