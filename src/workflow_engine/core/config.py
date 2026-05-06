from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Awaitable, Callable, Self, cast

from pydantic import Field

from ..utils.asynchronous import is_coroutine
from ..utils.importing import dynamic_import
from ..utils.model import ImmutableBaseModel, ImmutableRootModel
from ..utils.pattern import MODULE_NAME_PATTERN
from .execution import ExecutionAlgorithm
from .node import ImmutableNodeRegistry, Node, NodeRegistry


class Import(ImmutableRootModel[Annotated[str, Field(pattern=MODULE_NAME_PATTERN)]]):
    @cached_property
    def _module_and_name(self) -> tuple[str, str]:
        module, name = self.root.rsplit(".", 1)
        return module, name

    @cached_property
    def module(self) -> str:
        return self._module_and_name[0]

    @cached_property
    def name(self) -> str:
        return self._module_and_name[1]


class NodeImport(Import):
    @cached_property
    def node_cls(self) -> type[Node]:
        return dynamic_import(
            module=self.module,
            name=self.name,
            validate_subclass=Node,
        )

    @classmethod
    def from_node(cls, node: type[Node]) -> Self:
        return cls(
            root=f"{node.__module__}.{node.__name__}",
        )


class NodesConfig(ImmutableRootModel[Mapping[str, NodeImport]]):
    @classmethod
    def from_nodes(cls, *args: type[Node], **kwargs: type[Node]) -> Self:
        nodes: dict[str, NodeImport] = {}
        for node in args:
            name = node.default_type_name()
            if name in nodes:
                raise ValueError(f"Duplicate node type {name} in args")
            nodes[name] = NodeImport.from_node(node)
        for name, node in kwargs.items():
            if name in nodes:
                raise ValueError(f"Duplicate node type {name} in kwargs")
            nodes[name] = NodeImport.from_node(node)
        return cls(nodes)


class ExecutionAlgorithmImport(Import):
    @cached_property
    def execution_algorithm_factory(
        self,
    ) -> Callable[..., ExecutionAlgorithm | Awaitable[ExecutionAlgorithm]]:
        imported = dynamic_import(
            module=self.module,
            name=self.name,
            validate_predicate=callable,
        )
        return cast(
            Callable[..., ExecutionAlgorithm | Awaitable[ExecutionAlgorithm]],
            imported,
        )

    async def build_execution_algorithm(self, **kwargs) -> ExecutionAlgorithm:
        factory = self.execution_algorithm_factory
        alg = factory(**kwargs)
        if is_coroutine(alg):
            # NOTE: cast needed because is_coroutine isn't smart enough to rule
            # out ExecutionAlgorithm from the type union
            alg = await cast(Awaitable[ExecutionAlgorithm], alg)
        if not isinstance(alg, ExecutionAlgorithm):
            expected_type = ExecutionAlgorithm.__name__
            actual_type = type(alg).__name__
            raise ValueError(
                f"Execution algorithm factory {factory.__name__} returned object of type {actual_type}, which is not an {expected_type} instance"
            )
        return alg  # type: ignore[reportReturn]


class ExecutionAlgorithmConfig(ImmutableBaseModel):
    """
    A configuration for an execution algorithm.
    """

    factory: ExecutionAlgorithmImport = Field(
        description="The import of the execution algorithm factory.",
    )
    config: Mapping[str, Any] = Field(
        description="The configuration for the execution algorithm.",
        default_factory=dict,
    )

    async def build_execution_algorithm(self) -> ExecutionAlgorithm:
        """
        Creates an instance of the execution algorithm.
        We don't want to cache the instance itself because the algorithm object
        may contain state that should not be shared between engines.
        """
        return await self.factory.build_execution_algorithm(**self.config)


class WorkflowEngineConfig(ImmutableBaseModel):
    """
    A configuration for a workflow engine.
    Contains lazy methods for building the engine itself, but this class remains
    separate from the engine to enable easy loading (e.g. editing a config
    without instantiating the engine).
    """

    nodes: NodesConfig = Field(
        description="The configuration for the nodes.",
    )
    execution_algorithm: ExecutionAlgorithmConfig | None = None

    @cached_property
    def node_registry(self) -> NodeRegistry:
        return ImmutableNodeRegistry(
            node_classes={k: v.node_cls for k, v in self.nodes.root.items()}
        )

    async def build_execution_algorithm(self) -> ExecutionAlgorithm | None:
        """
        Creates an instance of the execution algorithm.
        We don't want to cache the instance itself because the algorithm object
        may contain state that should not be shared between engines.
        """
        if self.execution_algorithm is None:
            return None
        return await self.execution_algorithm.build_execution_algorithm()

    @classmethod
    def load(cls, path: Path) -> Self:
        match path.suffix:
            case ".json":
                with open(path, "r") as f:
                    return cls.model_validate_json(f.read())
            case ".yaml":
                with open(path, "r") as f:
                    return cls.model_validate_yaml(f)
            case ".toml":
                with open(path, "r") as f:
                    return cls.model_validate_toml(f)
            case _:
                raise ValueError(f"Unsupported file extension: {path.suffix}")


__all__ = [
    "ExecutionAlgorithmImport",
    "NodeImport",
    "WorkflowEngineConfig",
]
