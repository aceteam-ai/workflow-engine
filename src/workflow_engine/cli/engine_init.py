"""Create a fresh engine project — the `wengine init` machinery.

`wengine init` writes an `engine.yaml` whose name map mounts the builtin nodes,
and (in standalone mode) sets up the `uv` project that backs it. See
docs/plans/node-distribution.md.

The filesystem effects (writing `engine.yaml`, the minimal `pyproject.toml`)
are separated from the `uv` shell-out, which goes through `UvProject` / the
module-private `_run_uv` that tests monkeypatch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from workflow_engine.core.config import _iter_node_entry_points

from .uv_project import ENGINE_YAML_NAMES, UvProject

# The engine's own distribution. Builtins are advertised through the same
# entry-point group as third-party packages (see the trust model in the design
# doc); `wengine init` seeds the `"*"` glob that mounts them.
BUILTIN_DISTRIBUTION = "aceteam-workflow-engine"


class EngineYamlAlreadyExists(FileExistsError):
    """Raised when `wengine init` finds an `engine.yaml` already in the target dir."""


def builtin_entry_point_names() -> tuple[str, ...]:
    """The node entry-point names the engine's own distribution advertises.

    Read from `wengine`'s own environment — the builtins live in the same
    distribution `wengine` itself ships in, so this is the right source even
    before the target project's environment is provisioned.
    """
    return tuple(
        sorted(ep.name for ep in _iter_node_entry_points(BUILTIN_DISTRIBUTION))
    )


def initial_nodes_config() -> Mapping[str, str]:
    """The `nodes:` block a fresh `engine.yaml` starts with.

    One explicit `Name: <dist>:<Name>` entry per builtin. We seed explicit
    entries rather than a `"*"` glob so the name map states exactly what is
    mounted — curating is then deleting lines. The glob form stays valid in the
    schema for anyone who prefers to hand-write it.
    """
    return {
        name: f"{BUILTIN_DISTRIBUTION}:{name}" for name in builtin_entry_point_names()
    }


def render_engine_yaml(nodes: Mapping[str, str | Sequence[str]]) -> str:
    """Render a full `engine.yaml` document for the given `nodes:` block."""
    return yaml.safe_dump(
        {"schema_version": 1, "nodes": dict(nodes)},
        sort_keys=False,
        default_flow_style=False,
    )


def init_engine_project(target_dir: Path) -> Path:
    """Initialize a new engine project rooted at `target_dir`.

    Writes `engine.yaml`, then sets up the backing `uv` project: in standalone
    mode (no enclosing `pyproject.toml`) it writes a minimal one and `uv add`s
    the engine distribution; in embedded mode the host project already owns its
    dependencies, so only `engine.yaml` is written.

    Returns the path to the written `engine.yaml`. Raises
    `EngineYamlAlreadyExists` if `target_dir` already contains one.
    """
    target_dir = target_dir.resolve()
    for name in ENGINE_YAML_NAMES:
        if (target_dir / name).exists():
            raise EngineYamlAlreadyExists(
                f"{target_dir / name} already exists; refusing to overwrite. "
                f"Edit it by hand, or remove it to re-init."
            )

    engine_yaml = target_dir / ENGINE_YAML_NAMES[0]
    engine_yaml.write_text(render_engine_yaml(initial_nodes_config()))

    project = UvProject.locate(target_dir)
    if project.mode == "standalone":
        project.add([BUILTIN_DISTRIBUTION])

    return engine_yaml


__all__ = [
    "BUILTIN_DISTRIBUTION",
    "EngineYamlAlreadyExists",
    "builtin_entry_point_names",
    "init_engine_project",
    "initial_nodes_config",
    "render_engine_yaml",
]
