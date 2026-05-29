"""`wengine uninstall` — remove a mapped node, or a whole distribution.

The inverse of `wengine install` (see `install.py`): it edits `engine.yaml` to
drop a name map, then reconciles `pyproject.toml` to match what is still mapped.

- `wengine uninstall <NodeName>` removes the *explicit* entry for that name. The
  distribution it pointed at may still be reachable through other entries (other
  explicit names, or a glob), in which case the package stays installed and only
  its extras are re-narrowed.
- `wengine uninstall --dist <name>` removes *every* entry referencing a
  distribution at once — explicit entries plus its membership in any glob list.

Reconciliation of the distribution `D` whose mapping shrank:

- If any entry still references `D`, recompute the union of the still-mapped
  nodes' declared extras (a glob keeps every node mapped, so it pins the full
  set) and, if it shrank, rewrite `D`'s `[project.dependencies]` extras to that
  set and `uv sync`. `uv` then drops now-unneeded transitive deps. We edit the
  extras in `pyproject.toml` rather than `uv add D[…]` because `uv add` only
  *merges* extras (it has no reset), and `uv remove`+re-add would wipe `D`'s
  `[tool.uv.sources]` entry — editing the dependency line leaves the source
  mapping untouched and works the same for git/path/workspace/PyPI sources.
- If nothing references `D` anymore, `uv remove D`.

`engine.yaml` is written *first*, then `pyproject.toml` is reconciled: a failed
reconcile then leaves only superfluous extras behind (benign), rather than an
unmapped-but-still-needed node or a removed-but-still-referenced distribution.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from packaging.requirements import Requirement

from ..core.config import Distribution
from ..utils.model import ImmutableBaseModel
from .install import (
    MountedNode,
    NodeEntry,
    extras_union,
    load_nodes_block,
    read_distribution_entry_points,
    read_pyproject_dependencies,
    replace_nodes_block,
)
from .uv_project import UvProject


class UninstallError(Exception):
    """A `wengine uninstall` failed for an operator-actionable reason."""


class UninstallResult(ImmutableBaseModel):
    """What `uninstall` did, for the CLI to report."""

    distribution: Distribution
    removed_distribution: bool  # True if `uv remove` ran; False if it stayed.


# ---------- pure helpers ----------


def _is_glob_key(key: str) -> bool:
    """Whether a `nodes:` key is a glob (`"*"` or `"<prefix>:*"`)."""
    return key == "*" or key.endswith(":*")


def _glob_dists(value: NodeEntry) -> Sequence[str]:
    """The distribution list under a glob value (a name or a list of names)."""
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(v) for v in value)
    raise UninstallError(f"Glob entry has an unexpected value: {value!r}.")


def distribution_for_node(
    nodes: Mapping[str, NodeEntry],
    name: str,
) -> Distribution:
    """The distribution an *explicit* `name` entry points at.

    Raises `UninstallError` if `name` has no explicit entry (it may be supplied
    by a glob, which can only be removed wholesale with `--dist`).
    """
    value = nodes.get(name)
    if not isinstance(value, str):
        raise UninstallError(
            f"No explicit entry for {name!r} in engine.yaml. If it is provided by "
            f"a glob, use --dist <distribution> to remove the whole bundle."
        )
    return Distribution(value.split(":", 1)[0])


def references_distribution(
    nodes: Mapping[str, NodeEntry],
    dist: Distribution,
) -> bool:
    """Whether any `nodes:` entry still references `dist`."""
    for key, value in nodes.items():
        if _is_glob_key(key):
            if any(Distribution(d) == dist for d in _glob_dists(value)):
                return True
        elif isinstance(value, str):
            if Distribution(value.split(":", 1)[0]) == dist:
                return True
    return False


def remove_distribution_entries(
    nodes: Mapping[str, NodeEntry],
    dist: Distribution,
) -> Mapping[str, NodeEntry]:
    """Drop every explicit entry and glob membership referencing `dist`.

    A glob key whose list empties out is dropped entirely.
    """
    out: dict[str, NodeEntry] = {}
    for key, value in nodes.items():
        if _is_glob_key(key):
            kept = [d for d in _glob_dists(value) if Distribution(d) != dist]
            if not kept:
                continue
            out[key] = kept if isinstance(value, list) else kept[0]
        elif isinstance(value, str) and Distribution(value.split(":", 1)[0]) == dist:
            continue
        else:
            out[key] = value
    return out


def remaining_mapped_extras(
    nodes: Mapping[str, NodeEntry],
    dist: Distribution,
    available: Sequence[MountedNode],
) -> Sequence[str]:
    """The extras union of `dist`'s nodes that are still mapped by `nodes`.

    A glob membership mounts every node `dist` exposes, so it pins the full
    extra set; otherwise only the explicitly-mapped entry-point names count.
    """
    for key, value in nodes.items():
        if _is_glob_key(key) and any(
            Distribution(d) == dist for d in _glob_dists(value)
        ):
            return extras_union(available)
    mapped_names = {
        value.split(":", 1)[1]
        for value in nodes.values()
        if isinstance(value, str)
        and ":" in value
        and Distribution(value.split(":", 1)[0]) == dist
    }
    return extras_union([m for m in available if m.entry_point_name in mapped_names])


def narrow_dependency_extras(
    root: Path,
    dist: Distribution,
    extras: Sequence[str],
) -> bool:
    """Rewrite `dist`'s `[project.dependencies]` entry to exactly `extras`.

    Returns whether `pyproject.toml` changed. Edits the dependency string in
    place (preserving its version specifier / direct-URL and the rest of the
    file), so `[tool.uv.sources]` is left untouched. A no-op — and so `False` —
    if `dist` is not a direct dependency or already has exactly `extras`.
    """
    path = root / "pyproject.toml"
    text = path.read_text()
    for dep in read_pyproject_dependencies(root):
        req = Requirement(dep)
        if Distribution(req.name) != dist:
            continue
        if req.extras == set(extras):
            return False
        req.extras = set(extras)
        new_dep = str(req)
        for quote in ('"', "'"):
            quoted = f"{quote}{dep}{quote}"
            if quoted in text:
                path.write_text(text.replace(quoted, f"{quote}{new_dep}{quote}", 1))
                return True
        raise UninstallError(
            f"Could not locate dependency {dep!r} in {path} to re-narrow its extras."
        )
    return False


# ---------- orchestration ----------


def uninstall(
    name: str | None = None,
    *,
    dist: str | None = None,
    start: Path | None = None,
) -> UninstallResult:
    """Remove a node mapping (or a whole distribution) and reconcile `uv`.

    Exactly one of `name` / `dist` must be given. See the module docstring.
    """
    if (name is None) == (dist is None):
        raise UninstallError("Provide either a node name or --dist, but not both.")

    project = UvProject.locate(start or Path.cwd())
    nodes = load_nodes_block(project.engine_yaml)

    if dist is not None:
        target = Distribution(dist)
        if not references_distribution(nodes, target):
            raise UninstallError(f"No engine.yaml entry references {target.root!r}.")
        new_nodes = remove_distribution_entries(nodes, target)
    else:
        assert name is not None
        target = distribution_for_node(nodes, name)
        new_nodes = dict(nodes)
        del new_nodes[name]

    # Write the name map first (see the module docstring on ordering), then
    # reconcile pyproject to match what is still mapped.
    replace_nodes_block(project.engine_yaml, new_nodes)

    if references_distribution(new_nodes, target):
        # Still mapped — re-narrow its extras to the union of what remains.
        available = read_distribution_entry_points(project.root, target)
        new_extras = remaining_mapped_extras(new_nodes, target, available)
        if narrow_dependency_extras(project.root, target, new_extras):
            _run_uv(project.sync, target)
        removed = False
    else:
        _run_uv(lambda: project.remove(target.root), target)
        removed = True

    return UninstallResult(distribution=target, removed_distribution=removed)


def _run_uv(action: Callable[[], None], dist: Distribution) -> None:
    """Run a `uv` mutation, surfacing failures as `UninstallError`.

    `engine.yaml` is written only after this returns, so a failure here leaves
    the name map untouched.
    """
    try:
        action()
    except subprocess.CalledProcessError as e:
        raise UninstallError(f"uv failed while reconciling {dist.root!r}: {e}") from e


__all__ = ["UninstallError", "UninstallResult", "uninstall"]
