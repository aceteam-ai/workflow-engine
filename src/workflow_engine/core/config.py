import re
from collections.abc import Mapping
from functools import cached_property
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Annotated, Any, Awaitable, Callable, Literal, Self, cast

from pydantic import Field, model_validator

from ..utils.asynchronous import is_coroutine
from ..utils.importing import dynamic_import
from ..utils.model import ImmutableBaseModel, ImmutableRootModel
from ..utils.pattern import MODULE_NAME_PATTERN
from .execution import ExecutionAlgorithm
from .node import ImmutableNodeRegistry, Node, NodeRegistry

NODES_ENTRY_POINT_GROUP = "aceteam_workflow_engine.nodes"

# engine.yaml `nodes:` key/value grammar.
#
#   "*"                                     — global glob
#   "<prefix>:*"                            — prefixed glob (keyspace isolated)
#   "<Name>" or "<prefix>:<Name>"           — explicit entry
#
# Explicit values are "<distribution>:<entryPointName>" strings; glob values
# are a distribution name or a list of them.
_NODE_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_PREFIX = r"[A-Za-z0-9_][A-Za-z0-9_/\-]*"
_DIST_NAME = r"[A-Za-z0-9][A-Za-z0-9._-]*"

_EXPLICIT_KEY_RE = re.compile(rf"^(?:{_PREFIX}:)?{_NODE_NAME}$")
_PREFIX_GLOB_KEY_RE = re.compile(rf"^({_PREFIX}):\*$")
_EXPLICIT_VALUE_RE = re.compile(rf"^({_DIST_NAME}):({_NODE_NAME})$")


def _normalize_dist(name: str) -> str:
    """PEP 503-style normalization for distribution-name comparison."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _iter_node_entry_points(distribution: str) -> list[EntryPoint]:
    """Return every entry point in the nodes group from the given distribution."""
    target = _normalize_dist(distribution)
    matches: list[EntryPoint] = []
    for ep in entry_points(group=NODES_ENTRY_POINT_GROUP):
        if ep.dist is None:
            continue
        if _normalize_dist(ep.dist.name) == target:
            matches.append(ep)
    return matches


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


class EntryPointRef(
    ImmutableRootModel[Annotated[str, Field(pattern=_EXPLICIT_VALUE_RE.pattern)]]
):
    """An explicit `<distribution>:<entryPointName>` reference to one node."""

    @cached_property
    def _split(self) -> tuple[str, str]:
        dist, ep = self.root.split(":", 1)
        return dist, ep

    @cached_property
    def distribution(self) -> str:
        return self._split[0]

    @cached_property
    def entry_point_name(self) -> str:
        return self._split[1]

    @cached_property
    def node_cls(self) -> type[Node]:
        for ep in _iter_node_entry_points(self.distribution):
            if ep.name == self.entry_point_name:
                loaded = ep.load()
                if not isinstance(loaded, type) or not issubclass(loaded, Node):
                    raise ValueError(
                        f"Entry point {self.root!r} resolved to {loaded!r}, "
                        f"which is not a Node subclass."
                    )
                return loaded
        raise LookupError(
            f"No entry point named {self.entry_point_name!r} in group "
            f"{NODES_ENTRY_POINT_GROUP!r} from distribution {self.distribution!r}. "
            f"Run `wengine install {self.distribution}` to install it."
        )


class GlobValue(ImmutableRootModel[str | list[str]]):
    """Value of a `"*"` or `"prefix:*"` entry: a distribution name or list of names."""

    @cached_property
    def distributions(self) -> tuple[str, ...]:
        if isinstance(self.root, str):
            return (self.root,)
        return tuple(self.root)


class NodesConfig(ImmutableRootModel[Mapping[str, str | list[str]]]):
    """
    The `nodes:` block of `engine.yaml` — recognized name → entry-point ref.

    Three key shapes (see docs/plans/node-distribution.md):

    - `"*"`: global glob. Value is one distribution name, or a list of them.
      Mounts every node each listed distribution exposes under its bare
      entry-point name. A bare name supplied by two glob-mounted distributions
      is a hard error unless an explicit entry disambiguates it.
    - `"<prefix>:*"`: prefixed glob. Value form is the same. Mounts each node
      as `<prefix>:<Name>` — a separate keyspace from bare names.
    - `"<Name>"` or `"<prefix>:<Name>"`: explicit. Value is a
      `"<distribution>:<entryPointName>"` string. Explicit entries override
      any glob that would otherwise supply that name.
    """

    @model_validator(mode="after")
    def _check_grammar(self) -> Self:
        for key, value in self.root.items():
            if key == "*" or _PREFIX_GLOB_KEY_RE.match(key):
                # Glob — value must parse as GlobValue (str | list[str]).
                GlobValue.model_validate(value)
            elif _EXPLICIT_KEY_RE.match(key):
                if not isinstance(value, str):
                    raise ValueError(
                        f"Explicit entry {key!r} must have a string value of "
                        f"the form '<distribution>:<entryPointName>'; "
                        f"got {value!r}."
                    )
                EntryPointRef.model_validate(value)
            else:
                raise ValueError(
                    f"Invalid nodes key {key!r}: must be '*', '<prefix>:*', "
                    f"'<Name>', or '<prefix>:<Name>'."
                )
        return self

    @cached_property
    def explicit_entries(self) -> Mapping[str, EntryPointRef]:
        out: dict[str, EntryPointRef] = {}
        for key, value in self.root.items():
            if key == "*" or _PREFIX_GLOB_KEY_RE.match(key):
                continue
            assert isinstance(value, str)
            out[key] = EntryPointRef(root=value)
        return out

    @cached_property
    def global_glob_distributions(self) -> tuple[str, ...]:
        value = self.root.get("*")
        if value is None:
            return ()
        return GlobValue.model_validate(value).distributions

    @cached_property
    def prefix_glob_distributions(self) -> Mapping[str, tuple[str, ...]]:
        out: dict[str, tuple[str, ...]] = {}
        for key, value in self.root.items():
            m = _PREFIX_GLOB_KEY_RE.match(key)
            if m is None:
                continue
            out[m.group(1)] = GlobValue.model_validate(value).distributions
        return out

    @cached_property
    def node_registry(self) -> NodeRegistry:
        """Resolve the full name → Node-class map (eager; precedence-aware)."""
        resolved: dict[str, type[Node]] = {}

        # 1. Global "*" glob.
        seen: dict[str, str] = {}
        for dist in self.global_glob_distributions:
            for ep in _iter_node_entry_points(dist):
                prior = seen.get(ep.name)
                if prior is not None and prior != dist:
                    raise ValueError(
                        f"Bare-name collision: {ep.name!r} is exposed by both "
                        f"{prior!r} and {dist!r} on the '*' glob. Add an "
                        f"explicit entry for {ep.name!r} or move one "
                        f"distribution to a prefix:* mount."
                    )
                seen[ep.name] = dist
                resolved[ep.name] = EntryPointRef(root=f"{dist}:{ep.name}").node_cls

        # 2. Prefixed globs (own keyspace).
        for prefix, dists in self.prefix_glob_distributions.items():
            prefix_seen: dict[str, str] = {}
            for dist in dists:
                for ep in _iter_node_entry_points(dist):
                    prior = prefix_seen.get(ep.name)
                    if prior is not None and prior != dist:
                        raise ValueError(
                            f"Bare-name collision under prefix {prefix!r}: "
                            f"{ep.name!r} is exposed by both {prior!r} and "
                            f"{dist!r}."
                        )
                    prefix_seen[ep.name] = dist
                    resolved[f"{prefix}:{ep.name}"] = EntryPointRef(
                        root=f"{dist}:{ep.name}"
                    ).node_cls

        # 3. Explicit entries override globs.
        for name, ref in self.explicit_entries.items():
            resolved[name] = ref.node_cls

        return ImmutableNodeRegistry(node_classes=resolved)


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
    A configuration for a workflow engine — the on-disk shape of `engine.yaml`.

    Contains lazy methods for building the engine itself, but this class
    remains separate from the engine to enable easy loading (e.g. editing a
    config without instantiating the engine).
    """

    schema_version: Literal[1] = Field(
        default=1,
        description="The version of the engine.yaml schema this file conforms to.",
    )
    nodes: NodesConfig = Field(
        description="The recognized-name → entry-point-ref map for this engine.",
    )
    execution_algorithm: ExecutionAlgorithmConfig | None = None

    @cached_property
    def node_registry(self) -> NodeRegistry:
        return self.nodes.node_registry

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
            case ".yaml" | ".yml":
                with open(path, "r") as f:
                    return cls.model_validate_yaml(f)
            case ".toml":
                with open(path, "r") as f:
                    return cls.model_validate_toml(f)
            case _:
                raise ValueError(f"Unsupported file extension: {path.suffix}")


__all__ = [
    "NODES_ENTRY_POINT_GROUP",
    "EntryPointRef",
    "ExecutionAlgorithmConfig",
    "ExecutionAlgorithmImport",
    "GlobValue",
    "Import",
    "NodesConfig",
    "WorkflowEngineConfig",
]
