import re
from collections.abc import Mapping, Sequence
from functools import cached_property
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Annotated, Any, Awaitable, Callable, Literal, Self, cast

from packaging.requirements import Requirement
from packaging.utils import NormalizedName, canonicalize_name
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

DistributionName = Annotated[str, Field(pattern=rf"^{_DIST_NAME}$")]
EntryPointRefStr = Annotated[str, Field(pattern=rf"^{_DIST_NAME}:{_NODE_NAME}$")]


class Distribution(ImmutableRootModel[DistributionName]):
    """A distribution name compared by its canonical (PEP 503) form.

    Two spellings of the same project (`Acme.Scrapers` vs `acme-scrapers`) are
    equal and hash alike, so collision checks and set math need no ad-hoc
    normalization. `str(dist)` / `.root` give back the original spelling, which
    is what gets written into and read from `engine.yaml`.
    """

    @classmethod
    def from_requirement(cls, requirement: str) -> "Distribution":
        """Parse a PEP 508 requirement into a `Distribution`.

        Strips any extras and version specifiers (`acme[x]>=1.4` → `acme`), so
        callers holding a raw requirement string don't have to split it first.
        """
        return cls(Requirement(requirement).name)

    @property
    def canonical(self) -> NormalizedName:
        return canonicalize_name(self.root)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Distribution):
            return self.canonical == other.canonical
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.canonical)

    def __str__(self) -> str:
        return self.root


def _iter_node_entry_points(distribution: str) -> Sequence[EntryPoint]:
    """Return every entry point in the nodes group from the given distribution."""
    target = Distribution(distribution)
    matches: list[EntryPoint] = []
    for ep in entry_points(group=NODES_ENTRY_POINT_GROUP):
        if ep.dist is None:
            continue
        if Distribution(ep.dist.name) == target:
            matches.append(ep)
    return matches


def _load_node_class(ep: EntryPoint) -> type[Node]:
    """Import an entry point's target and check it's a Node subclass."""
    loaded = ep.load()
    if not isinstance(loaded, type) or not issubclass(loaded, Node):
        raise ValueError(
            f"Entry point {ep.name!r} resolved to {loaded!r}, "
            f"which is not a Node subclass."
        )
    return loaded


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


class EntryPointRef(ImmutableRootModel[EntryPointRefStr]):
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
        matches = [
            ep
            for ep in _iter_node_entry_points(self.distribution)
            if ep.name == self.entry_point_name
        ]
        if len(matches) == 0:
            raise LookupError(
                f"No entry point named {self.entry_point_name!r} in group "
                f"{NODES_ENTRY_POINT_GROUP!r} from distribution "
                f"{self.distribution!r}. "
                f"Run `wengine install {self.distribution}` to install it."
            )
        if len(matches) > 1:
            raise LookupError(
                f"Ambiguous entry point {self.entry_point_name!r} in group "
                f"{NODES_ENTRY_POINT_GROUP!r} from distribution "
                f"{self.distribution!r}: {len(matches)} matches found."
            )
        return _load_node_class(matches[0])


class GlobValue(ImmutableRootModel[DistributionName | Sequence[DistributionName]]):
    """Value of a `"*"` or `"prefix:*"` entry: a distribution name or list of names."""

    @cached_property
    def distributions(self) -> tuple[str, ...]:
        if isinstance(self.root, str):
            return (self.root,)
        return tuple(self.root)


class NodesConfig(ImmutableRootModel[Mapping[str, str | Sequence[str]]]):
    """
    The `nodes:` block of `engine.yaml` — recognized name → entry-point ref.

    Three key shapes (see docs/node-distribution.md):

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

        # 1. Global "*" glob. Compare distributions by their canonical (PEP 503)
        # name so different spellings of the same distribution (e.g.
        # `Acme_Pkg` vs `acme-pkg`) don't read as a collision.
        seen: dict[str, Distribution] = {}
        for dist in self.global_glob_distributions:
            distribution = Distribution(dist)
            for ep in _iter_node_entry_points(dist):
                prior = seen.get(ep.name)
                if prior is not None and prior != distribution:
                    raise ValueError(
                        f"Bare-name collision: {ep.name!r} is exposed by both "
                        f"{prior.root!r} and {dist!r} on the '*' glob. Add an "
                        f"explicit entry for {ep.name!r} or move one "
                        f"distribution to a prefix:* mount."
                    )
                seen[ep.name] = distribution
                resolved[ep.name] = _load_node_class(ep)

        # 2. Prefixed globs (own keyspace).
        for prefix, dists in self.prefix_glob_distributions.items():
            prefix_seen: dict[str, Distribution] = {}
            for dist in dists:
                distribution = Distribution(dist)
                for ep in _iter_node_entry_points(dist):
                    prior = prefix_seen.get(ep.name)
                    if prior is not None and prior != distribution:
                        raise ValueError(
                            f"Bare-name collision under prefix {prefix!r}: "
                            f"{ep.name!r} is exposed by both {prior.root!r} and "
                            f"{dist!r}."
                        )
                    prefix_seen[ep.name] = distribution
                    resolved[f"{prefix}:{ep.name}"] = _load_node_class(ep)

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
    "Distribution",
    "EntryPointRef",
    "ExecutionAlgorithmConfig",
    "ExecutionAlgorithmImport",
    "GlobValue",
    "Import",
    "NodesConfig",
    "WorkflowEngineConfig",
]
