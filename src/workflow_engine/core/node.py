# workflow_engine/core/node.py
from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
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
)

from overrides import final, override
from pydantic import ConfigDict, Field, ValidationError, model_validator
from typing_extensions import overload

from ..utils.asynchronous import gather
from ..utils.model import ImmutableBaseModel
from ..utils.semver import (
    LATEST_SEMANTIC_VERSION,
    SEMANTIC_VERSION_OR_LATEST_PATTERN,
    SEMANTIC_VERSION_PATTERN,
    parse_semantic_version,
)
from .error import NodeException, ShouldYield, WorkflowException
from .values import (
    Data,
    DataMapping,
    Value,
    ValueSchema,
    ValueType,
    get_data_fields,
)
from .values.data import Input_contra, Output, get_data_dict, get_data_schema

if TYPE_CHECKING:
    from .context import ExecutionContext, ValidationContext
    from .io import InputNode, OutputNode
    from .workflow import ValidatedWorkflow, Workflow

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


class NodeTypeInfo(ImmutableBaseModel):
    """
    Information about a node type, in serializable form.
    """

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
        display_name: str,
        description: str | None = None,
        version: str,
        parameter_type: type[Params],
        max_retries: int | None = None,
    ) -> Self:
        return cls(
            display_name=display_name,
            description=description,
            version=version,
            parameter_schema=get_data_schema(parameter_type),
            max_retries=max_retries,
        )


def get_id_with_namespace(id: str, namespace: str) -> str:
    """
    Get the ID of a node with a namespace prefix.
    """
    return f"{namespace}/{id}"


class Node(ImmutableBaseModel, Generic[Input_contra, Output, Params_co]):
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
    # Each concrete Node subclass registers itself on the default lazy registry
    # when the class body is executed, under the name `default_type_name()`.
    #
    # This is brittle when multiple packages define different classes for the
    # same type name; prefer supplying your own NodeRegistry in that case.
    # The default registry stays lazy until first use to avoid eagerly building
    # an invalid mapping.

    @classmethod
    def default_type_name(cls) -> str:
        if cls is Node:
            raise ValueError(
                f"{cls.__name__} is the base class, cannot get a default type name"
            )
        return cls.__name__.removesuffix("Node")

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]):
        super().__init_subclass__(**kwargs)  # type: ignore

        # Skip parameterized generic aliases (e.g. `Node[Data, Data, P]`),
        # which Pydantic synthesizes as intermediate classes when a concrete
        # subclass writes `class FooNode(Node[I, O, P])`. They show up here
        # with bracketed names. Only concrete subclasses should register.
        name = cls.__name__
        if "[" in name and "]" in name and name.count("[") == name.count("]"):
            return

        NodeRegistry.DEFAULT.register(cls)

    # --------------------------------------------------------------------------
    # NAMING

    async def display_name(self, context: "ValidationContext") -> str:
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
        return self.model_update(id=get_id_with_namespace(self.id, namespace))

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

    @classmethod
    def static_input_type(cls) -> type[Input_contra] | None:  # type: ignore (contravariant return type)
        """
        The unchanging input type of the node, or None if the input type is
        dynamic.
        """
        return None

    async def dynamic_input_type(
        self,
        context: "ValidationContext",
    ) -> type[Input_contra]:  # type: ignore (contravariant return type)
        """
        The dynamic input type of the node.
        """
        raise NotImplementedError(
            "All concrete node classes must either provide a static_input_type or implement dynamic_input_type."
        )

    @final
    async def input_type(
        self,
        context: "ValidationContext",
    ) -> type[Input_contra]:  # type: ignore (contravariant return type)
        """
        The type of the input data for this node.
        This field must always be cached to ensure referential equality on every
        call; otherwise we will be constructing instances of different types.
        """
        static_input_type = self.__class__.static_input_type()
        if static_input_type is not None:
            return static_input_type
        return await self.dynamic_input_type(context)

    @classmethod
    def static_output_type(cls) -> type[Output] | None:  # type: ignore (covariant return type)
        """
        The unchanging output type of the node, or None if the output type is
        dynamic.
        """
        return None

    async def dynamic_output_type(
        self,
        context: "ValidationContext",
    ) -> type[Output]:  # type: ignore (covariant return type)
        """
        The dynamic output type of the node.
        """
        raise NotImplementedError(
            "All concrete node classes must either provide a static_output_type or implement dynamic_output_type."
        )

    @final
    async def output_type(
        self,
        context: "ValidationContext",
    ) -> type[Output]:
        """
        The type of the output data for this node.
        This field must always be cached to ensure referential equality on every
        call; otherwise we will be constructing instances of different types.
        """
        static_output_type = self.__class__.static_output_type()
        if static_output_type is not None:
            return static_output_type
        return await self.dynamic_output_type(context)

    # --------------------------------------------------------------------------
    # EXECUTION

    async def _cast_input(
        self,
        input: DataMapping,
        context: "ExecutionContext",
        input_type: type[Input_contra],
    ) -> Input_contra:  # type: ignore (contravariant return type)
        input_fields = get_data_fields(input_type)
        allow_extra_input = input_type.model_config.get("extra", "forbid") == "allow"

        # Validate castability and collect cast tasks
        casted_input: dict[str, Value] = {}
        cast_keys: list[str] = []
        cast_tasks: list[Awaitable[Value]] = []
        for key, value in input.items():
            if key not in input_fields:
                if allow_extra_input:
                    continue
                raise NodeException.for_builder(
                    f"Unknown input field '{key}' for node {self.id}",
                    node=self,
                )
            input_field_type, _ = input_fields[key]
            if not value.can_cast_to(input_field_type):
                raise NodeException.for_user(
                    f"Input {value} for node {self.id} is invalid: {value} is not assignable to {input_field_type}",
                    node=self,
                )

            # avoid asyncio overhead by keeping original value
            if isinstance(value, input_field_type):
                casted_input[key] = value
            else:
                cast_tasks.append(value.cast_to(input_field_type, context=context))
                cast_keys.append(key)

        casted_values = await gather(cast_tasks)

        # Build the result dictionary
        for key, value in zip(cast_keys, casted_values):
            assert key not in casted_input
            casted_input[key] = value

        try:
            return input_type.model_validate(casted_input)
        except ValidationError as e:
            raise NodeException.for_user(
                f"Input {casted_input} for node {self.id} is invalid: {e}",
                node=self,
            ) from e

    # @abstractmethod
    async def run(
        self,
        *,
        context: "ExecutionContext",
        input_type: type[Input_contra],
        output_type: type[Output],
        input: Input_contra,
    ) -> "Output | Workflow":
        """
        Computes the node's outputs based on its inputs.
        Subclasses must implement this method, but it is not marked as abstract
        because the base Node class needs to be instantiable for dispatching.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @final
    async def __call__(
        self,
        *,
        context: "ExecutionContext",
        input_type: type[Input_contra],
        output_type: type[Output],
        input: DataMapping,
    ) -> "DataMapping | ValidatedWorkflow":
        """
        Executes the node.
        """
        casted_input: DataMapping | None = None
        try:
            try:
                logger.info("Starting node %s", self.id)
                try:
                    input_obj = await self._cast_input(input, context, input_type)
                except ValidationError as e:
                    raise NodeException.for_user(
                        f"Input {input} for node {self.id} is invalid: {e}",
                        node=self,
                    ) from e
                casted_input = get_data_dict(input_obj)
                output = await context.on_node_start(
                    node=self,
                    input_type=input_type,
                    output_type=output_type,
                    input=casted_input,
                )

                from .workflow import (
                    ValidatedWorkflow,
                    Workflow,
                )  # lazy to avoid circular import

                if output is None:
                    output = await self.run(
                        context=context,
                        input_type=input_type,
                        output_type=output_type,
                        input=input_obj,
                    )
                    if not isinstance(output, Workflow):
                        output = get_data_dict(output)

                if isinstance(output, Workflow):
                    if not isinstance(output, ValidatedWorkflow):
                        workflow = await output.validate(context.validation_context)
                    else:
                        workflow = output
                    output = await context.on_node_expand(
                        node=self,
                        input_type=input_type,
                        output_type=output_type,
                        input=casted_input,
                        workflow=workflow,
                    )
                    # TODO: once that workflow eventually finishes running, its
                    # output should be the output of this node, and we should call
                    # context.on_node_finish.
                else:
                    output = await context.on_node_finish(
                        node=self,
                        input_type=input_type,
                        output_type=output_type,
                        input=casted_input,
                        output=output,
                    )
                logger.info("Finished node %s", self.id)
                return output
            except ShouldYield:
                raise
            except Exception as e:
                # other errors pass through as NodeExceptions
                if isinstance(e, WorkflowException):
                    raise
                else:
                    raise NodeException.for_operator(
                        f"Unhandled exception in node {self.id}: {e}",
                        node=self,
                    ) from e
        except WorkflowException as e:
            # In subclasses, you don't have to worry about logging the error,
            # since it'll be logged here.
            logger.exception("Error in node %s", self.id)

            # Automatically set or confirm the node ID on WorkflowExceptions
            if e.node_id is None:
                e.node_id = self.id
            else:
                if e.node_id != self.id:
                    # operator-level exception, because the only way such a mismatch can occur is if the node implementation did something illegal
                    raise NodeException.for_operator(
                        f"Node '{self.id}' caught a WorkflowException with mismatched ID '{e.node_id}'.",
                        node=self,
                    ) from e

            context_e = await context.on_node_error(
                node=self,
                input_type=input_type,
                output_type=output_type,
                input=casted_input if casted_input is not None else input,
                exception=e,
            )
            if isinstance(context_e, Mapping):
                logger.exception(
                    "Error absorbed by context and replaced with output %s", context_e
                )
                return context_e
            if not isinstance(context_e, Exception):
                raise NodeException.for_operator(
                    f"Context returned an unexpected type: {type(context_e)}",
                    node=self,
                ) from e
            if not isinstance(context_e, WorkflowException):
                raise NodeException.for_operator(
                    f"Unhandled exception in node {self.id}: {context_e}",
                    node=self,
                ) from context_e
            raise context_e


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


N = TypeVar("N", bound=Node)


class NodeRegistry(ABC):
    """
    An immutable registry of concrete node classes by their names.
    """

    DEFAULT: ClassVar[LazyNodeRegistry]

    @abstractmethod
    def get(self, name: str) -> type[Node] | None:
        """
        Get a node class by name, or None if the name is not registered.
        """
        raise NotImplementedError()

    @abstractmethod
    def items(self) -> Iterable[tuple[str, type[Node]]]:
        """
        Return all registered node classes as (registry name, class) pairs.
        """
        raise NotImplementedError()

    def load(self, node: Node) -> Node:
        """
        Load an untyped node into its concrete typed form.

        If the node is already a concrete type, returns it unchanged.
        Otherwise, looks up the concrete type, applies migrations, and returns a typed instance.

        Args:
            node: An untyped node (base Node class instance)

        Returns:
            A typed node (concrete subclass instance)

        Raises:
            ValueError: If the node type is not registered
        """
        cls = self.get(node.type)
        if cls is None:
            raise ValueError(f'Node type "{node.type}" is not registered')

        if not issubclass(cls, node.__class__):
            warnings.warn(
                f"Node {node.id} will be reloaded as an instance of {cls.__name__}, which is not a subclass of its current class {node.__class__.__name__}. This is not a bug, just unusual."
            )

        if isinstance(node, cls):
            # no transformation needed; node is already an instance of the class
            return node

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
        for _, cls in self.items():
            builder.register(cls)
        return builder

    # NODE CREATION METHODS
    # Instead of invoking a specific class constructor (which prevents us from
    # overriding one node type with another), we use name-based dispatch to
    # create nodes using the following helpers.

    def create_node(
        self,
        name: str | type[N],
        /,
        *,
        id: str,
        params: Mapping[str, Any] | Params | None = None,
        **kwargs: Any,
    ) -> N:
        """
        Create a new node instance by name.
        If a Node type is provided, we will use its default_type_name()
        """
        if isinstance(name, str):
            base_cls = Node
            cls = self.get(name)
        else:
            base_cls = name
            if not issubclass(base_cls, Node):
                raise ValueError(f"Node type {name.__name__} is not a subclass of Node")
            name = name.default_type_name()
            cls = self.get(name)
        if cls is None:
            raise ValueError(f'Node type "{name}" is not registered')
        if not issubclass(cls, base_cls):
            raise ValueError(
                f"Node type {cls.__name__} is not a subclass of {base_cls.__name__}"
            )
        return cls.model_validate(  # pyright: ignore[reportReturnType]
            dict(
                type=name,
                id=id,
                params=({} if params is None else params),
                **kwargs,
            )
        )

    def create_input_node(
        self,
        **fields: ValueType,
    ) -> InputNode:
        """
        Create a new input node instance, using whatever has been registered as
        the "Input" node type.
        """
        from .io import InputNode, SchemaParams

        return self.create_node(
            InputNode,
            id="input",
            params=SchemaParams.from_fields(**fields),
        )

    def create_output_node(
        self,
        **fields: ValueType,
    ) -> OutputNode:
        """
        Create a new output node instance, using whatever has been registered as
        the "Output" node type.
        """
        from .io import OutputNode, SchemaParams

        return self.create_node(
            OutputNode,
            id="output",
            params=SchemaParams.from_fields(**fields),
        )


class NodeRegistryBuilder(ABC):
    """
    A mutable builder for a mapping from node types to node
    classes, allowing us to replace base node classes with their subtypes by
    looking up their identifiers.
    """

    @abstractmethod
    def register(
        self,
        node_cls: type[Node],
        name: str | None = None,
    ) -> Self:
        """
        Register a node class by name.
        If `name` is provided, use it as the registry key.
        Otherwise, use `node_cls.default_type_name()`.
        """
        raise NotImplementedError()

    @abstractmethod
    def get(self, name: str) -> type[Node] | None:
        """
        Returns the node class registered for the given name, or None if not found.
        """
        raise NotImplementedError()

    @abstractmethod
    def _pop(
        self,
        name: str,
    ) -> type[Node]:
        """
        Remove a registered Node class by name and return it.
        Raises KeyError if the node class is not registered.
        """
        raise NotImplementedError()

    def unregister(
        self,
        name: str,
        *,
        expect: type[Node] | None = None,
        missing_ok: bool = False,
    ) -> Self:
        """
        Given a registry name, remove the corresponding node class.

        If ``expect`` is provided, the registered class must be that exact type.
        """
        target = self.get(name)
        if target is None:
            if not missing_ok:
                raise ValueError(f'Node type "{name}" is not registered')
            return self
        if expect is not None and target is not expect:
            raise ValueError(
                f'Node type "{name}" is registered to {target.__name__}, expected {expect.__name__}'
            )
        popped = self._pop(name)
        if popped is not target:
            raise RuntimeError(
                f'Node type "{name}" was registered to a different class ({popped.__name__}) while we were trying to remove it.'
            )
        return self

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
    ):
        self._node_classes = dict(node_classes)

    @override
    def get(self, name: str) -> type[Node] | None:
        return self._node_classes.get(name)

    @override
    def items(self) -> Iterable[tuple[str, type[Node]]]:
        return self._node_classes.items()


class EagerNodeRegistryBuilder(NodeRegistryBuilder):
    def __init__(self):
        self._node_classes: dict[str, type[Node]] = {}

    @override
    def register(
        self,
        node_cls: type[Node],
        name: str | None = None,
    ) -> Self:
        key = node_cls.default_type_name() if name is None else name
        if key in self._node_classes:
            raise ValueError(f'Node type "{key}" is already registered')
        self._node_classes[key] = node_cls
        return self

    @override
    def get(self, name: str) -> type[Node] | None:
        return self._node_classes.get(name)

    @override
    def _pop(self, name: str) -> type[Node]:
        return self._node_classes.pop(name)

    @override
    def build(self) -> NodeRegistry:
        return ImmutableNodeRegistry(
            node_classes=self._node_classes,
        )


class LazyNodeRegistry(NodeRegistryBuilder, NodeRegistry):
    """
    A LazyNodeRegistry is both a NodeRegistry and a NodeRegistryBuilder with a
    2-part lifecycle:
    1. Registration phase: acts as a NodeRegistryBuilder until ``build()`` is
       called or the registry is read via ``get``, ``items``, or ``load``.
    2. Registry phase: exposes an immutable mapping built from registrations.

    Validations are deferred until the registry is frozen; this allows for a
    LazyNodeRegistry to exist as long as it is not used, making it suitable for
    automatic node registration.
    """

    def __init__(self):
        self._registrations: list[tuple[str, type[Node]]] = []
        self._removals: dict[
            str, tuple[type[Node] | None, bool]
        ] = {}  # name -> expected, missing_ok
        self._frozen: bool = False

        # initialized after freeze
        self._node_classes: Mapping[str, type[Node]]

    # REGISTRATION PHASE

    @override
    def register(
        self,
        node_cls: type[Node],
        name: str | None = None,
    ) -> Self:
        if self._frozen:
            raise ValueError("Node registry is frozen, cannot register new node types.")
        key = node_cls.default_type_name() if name is None else name
        self._registrations.append((key, node_cls))
        return self

    @override
    def _pop(self, name: str) -> type[Node]:
        if self._frozen:
            raise ValueError("Cannot remove types from a frozen node registry.")
        for i in range(len(self._registrations) - 1, -1, -1):
            reg_name, cls = self._registrations[i]
            if reg_name == name:
                self._registrations.pop(i)
                return cls
        raise KeyError(name)

    @override
    def unregister(
        self,
        name: str,
        *,
        expect: type[Node] | None = None,
        missing_ok: bool = False,
    ) -> Self:
        if self._frozen:
            raise ValueError("Node registry is frozen, cannot remove node types.")
        self._removals[name] = (expect, missing_ok)
        return self

    @override
    def build(self) -> NodeRegistry:
        if self._frozen:
            return self

        self._frozen = True
        _node_classes: dict[str, type[Node]] = {}
        for name, node_cls in self._registrations:
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

        for name, (expect, missing_ok) in self._removals.items():
            if name not in _node_classes:
                if not missing_ok:
                    raise ValueError(f'Node type "{name}" is not registered')
                continue
            registered = _node_classes[name]
            if expect is not None and registered is not expect:
                raise ValueError(
                    f'Node type "{name}" is registered to {registered.__name__}, expected {expect.__name__}'
                )
            del _node_classes[name]

        del self._registrations  # just to be extra sure
        del self._removals
        self._node_classes = _node_classes

        return self

    # USE PHASE

    @override
    def get(self, name: str) -> type[Node] | None:
        self.build()
        return self._node_classes.get(name)

    @override
    def items(self) -> Iterable[tuple[str, type[Node]]]:
        self.build()
        return self._node_classes.items()


# Register the abstract Node class under "Node" for polymorphic deserialization.
NodeRegistry.DEFAULT = NodeRegistry.builder(lazy=True)


__all__ = [
    "Empty",
    "Node",
    "NodeTypeInfo",
    "Params",
]
