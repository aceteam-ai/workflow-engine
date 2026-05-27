"""`wengine install` — add a node-source distribution and map its nodes.

`wengine install <target>` is, in the design doc's words, "deliberately thin":
roughly `uv add <target>` plus an `engine.yaml` edit. The non-trivial parts it
adds are the `engine.yaml` name-map maintenance, the entry-point discovery that
tells it what to write and which extras to request, and the collision checks.

Because `uv` exposes no "read this package's metadata without committing it to
the lockfile" mode (no `uv add --dry-run`, no `uv pip download`), this does
*not* honor the design doc's "abort before touching anything" ideal in full.
The flow is:

1. Parse the target and compute the intended `engine.yaml` change.
2. Pre-check explicit-vs-explicit collisions from `engine.yaml` alone — the
   recognized names of `--only` / `--as` installs are known up front, so this
   aborts before any file write or `uv` call.
3. `uv add <target>` (with any user-typed extras) — installs the package so its
   entry-point metadata becomes readable. **This mutates `pyproject.toml`.**
4. Read the newly-added distribution's entry-point table from the project's
   environment (via `uv run python`, so it works in standalone mode where that
   environment is a separate venv from the one `wengine` runs under).
5. If the mounted nodes declare extras, a second `uv add <dist>[<extras>]`.
6. For a bulk (glob) install, check for bare-name collisions against the other
   glob-mounted distributions; on an unresolved clash, roll back (`uv remove`).
7. Write the `engine.yaml` entries.

The `uv` shell-outs reuse `UvProject` (whose `_run_uv` tests monkeypatch); the
metadata read goes through the module-private `_uv_run_python`.
"""

import io
import json
import re
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

from packaging.requirements import Requirement
from pydantic import ValidationError
from ruamel.yaml import YAML

from ..core.config import Distribution, NodesConfig
from ..utils.iter import only as _only
from ..utils.model import ImmutableBaseModel
from .install_target import InstallTarget, PyPITarget, parse_install_target
from .uv_project import UvProject

_PREFIX_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_/\-]*$")

# Round-trip YAML: preserves comments, key order, and formatting that a
# hand-edited `engine.yaml` carries across a `wengine install` rewrite.
_yaml = YAML()
_yaml.preserve_quotes = True

# A `nodes:` entry is either an explicit `"dist:entryPoint"` string or, for a
# glob key (`"*"` / `"<prefix>:*"`), a list of distribution names. This mirrors
# the value shape `NodesConfig` validates.
NodeEntry = str | list[str]


class InstallError(Exception):
    """A `wengine install` failed for an operator-actionable reason."""


class MountedNode(ImmutableBaseModel):
    """A node entry point being mounted, with the package extras it declares."""

    entry_point_name: str
    extras: Sequence[str]


# ---------- pure helpers ----------


def read_pyproject_dependencies(root: Path) -> list[str]:
    """The `[project.dependencies]` list from `root`'s `pyproject.toml`."""
    path = root / "pyproject.toml"
    if not path.is_file():
        return []
    data = tomllib.loads(path.read_text())
    project = data.get("project", {})
    return list(project.get("dependencies", []))


def added_distribution(before: Sequence[str], after: Sequence[str]) -> Distribution:
    """The distribution newly present in `after` but not `before`.

    Both are `[project.dependencies]` snapshots. Comparison is by canonical
    (PEP 503) name, so a re-spelled version of an existing dependency doesn't
    read as new. Raises `InstallError` if not exactly one name was added.
    """
    before_dists = {Distribution.from_requirement(d) for d in before}
    after_dists = {Distribution.from_requirement(d) for d in after}
    new_dists = after_dists - before_dists
    if len(new_dists) != 1:
        raise InstallError(
            f"Expected exactly one new distribution in pyproject.toml after the "
            f"install, found {sorted(d.root for d in new_dists)}. Project "
            f"dependencies may have been edited concurrently."
        )
    return _only(new_dists)


def with_extras(dist: Distribution, extras: Sequence[str]) -> str:
    """A `uv add` requirement string for `dist` carrying `extras`."""
    req = Requirement(dist.root)
    req.extras |= set(extras)
    return str(req)


def extras_union(nodes: Sequence[MountedNode]) -> Sequence[str]:
    """The sorted union of every mounted node's declared extras."""
    out: set[str] = set()
    for node in nodes:
        out.update(node.extras)
    return tuple(sorted(out))


def resolve_only_mounts(
    available: Sequence[MountedNode],
    only: Sequence[str],
) -> Sequence[MountedNode]:
    """The subset of `available` named by `--only`, preserving `only` order.

    Raises `InstallError` if a requested name isn't exposed by the distribution.
    """
    by_name = {node.entry_point_name: node for node in available}
    out: list[MountedNode] = []
    for name in only:
        node = by_name.get(name)
        if node is None:
            raise InstallError(
                f"Node {name!r} is not exposed by this distribution. "
                f"Available: {', '.join(sorted(by_name)) or '(none)'}."
            )
        out.append(node)
    return out


def plan_explicit_names(only: Sequence[str], as_name: str | None) -> dict[str, str]:
    """Map recognized name → entry-point name for an `--only` install.

    `--as` renames a single `--only` node; without it each node maps to its own
    entry-point name. Raises `InstallError` on misuse of `--as`.
    """
    if as_name is not None and len(only) != 1:
        raise InstallError("--as is only valid with exactly one --only.")
    if as_name is not None:
        return {as_name: _only(only)}
    return {name: name for name in only}


def check_recognized_name_grammar(names: Sequence[str]) -> None:
    """Reject recognized names that aren't valid `engine.yaml` node keys.

    Validates each key through `NodesConfig` (with a placeholder ref value), so
    a bad `--as` name fails before any `uv` call rather than at the post-install
    write. Raises `InstallError` on the first invalid name.
    """
    for name in names:
        try:
            NodesConfig.model_validate({name: "placeholder:Placeholder"})
        except ValidationError as e:
            raise InstallError(
                f"Invalid node name {name!r}: must be a valid '<Name>' or "
                f"'<prefix>:<Name>' key."
            ) from e


# ---------- engine.yaml mutation ----------


def _read_document(engine_yaml: Path) -> Mapping[str, object]:
    """The full `engine.yaml` document as a round-trip mapping (comments kept)."""
    data = _yaml.load(engine_yaml.read_text())
    return data if data is not None else {}


def load_nodes_block(engine_yaml: Path) -> dict[str, NodeEntry]:
    """The validated `nodes:` mapping from `engine.yaml` (for mutation).

    Validates the block against `NodesConfig` — the engine's own model for the
    `nodes:` grammar — so a malformed block (non-mapping, bad key, or a value
    that is neither an entry-point ref nor a glob list) is rejected here rather
    than tripping a helper downstream. Other top-level keys are left untouched.
    """
    raw = dict(_read_document(engine_yaml)).get("nodes") or {}
    try:
        config = NodesConfig.model_validate(raw)
    except ValidationError as e:
        raise InstallError(f"{engine_yaml}: invalid nodes block ({e}).") from e
    return {
        key: value if isinstance(value, str) else [str(v) for v in value]
        for key, value in config.root.items()
    }


def write_nodes_block(engine_yaml: Path, nodes: Mapping[str, NodeEntry]) -> None:
    """Write `nodes` back into `engine.yaml`, preserving the rest of the document.

    The new block is validated through `NodesConfig` first. The write edits the
    round-trip document in place — surviving keys keep their comments and
    formatting — and only then re-serializes, so hand-written annotations on
    untouched entries are not lost.
    """
    try:
        NodesConfig.model_validate(dict(nodes))
    except ValidationError as e:
        raise InstallError(
            f"{engine_yaml}: refusing to write invalid nodes block ({e})."
        ) from e

    document = _read_document(engine_yaml)
    doc = dict(document) if not isinstance(document, dict) else document
    existing = doc.get("nodes")
    if isinstance(existing, dict):
        for key in [k for k in existing if k not in nodes]:
            del existing[key]
        for key, value in nodes.items():
            existing[key] = value
    else:
        doc["nodes"] = dict(nodes)

    buffer = io.StringIO()
    _yaml.dump(doc, buffer)
    engine_yaml.write_text(buffer.getvalue())


def _glob_distributions(nodes: Mapping[str, NodeEntry], key: str) -> list[str]:
    """The distribution list under a glob key (`"*"` or `"<prefix>:*"`)."""
    value = nodes.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def check_explicit_collisions(
    nodes: Mapping[str, NodeEntry],
    recognized_names: Sequence[str],
    target_dist: Distribution,
) -> None:
    """Reject explicit entries that clash with a different distribution.

    An existing *explicit* entry for `name` pointing at another distribution is
    a hard error. (Builtins are explicit entries, so overriding one lands here.)
    An existing entry for the *same* distribution is a no-op. We never silently
    overwrite a mapping — the operator must remove the old entry first.
    """
    for name in recognized_names:
        existing = nodes.get(name)
        if not isinstance(existing, str):
            continue  # absent, or a glob list — not an explicit clash
        if Distribution(existing.split(":", 1)[0]) != target_dist:
            raise InstallError(
                f"{name!r} is already mapped to {existing!r}. Remove that entry "
                f"from engine.yaml first to map it to {target_dist.root!r} instead."
            )


def glob_neighbors(
    nodes: Mapping[str, NodeEntry], target_dist: Distribution
) -> list[Distribution]:
    """The other distributions on the `"*"` glob (excluding `target_dist`)."""
    return [
        dist
        for d in _glob_distributions(nodes, "*")
        if (dist := Distribution(d)) != target_dist
    ]


def check_glob_collision(
    nodes: Mapping[str, NodeEntry],
    target_dist: Distribution,
    ep_names: Sequence[str],
    neighbor_entry_points: Mapping[Distribution, Sequence[str]],
) -> None:
    """Reject a bulk install whose bare names clash with another glob dist.

    `neighbor_entry_points` maps each other `"*"`-mounted distribution to its
    entry-point names (gathered by the caller from the project environment). An
    explicit entry disambiguates a name (explicit beats glob), so names that
    already have an explicit entry are not treated as clashes.
    """
    for other_dist, other_names in neighbor_entry_points.items():
        clashes = {
            n
            for n in set(ep_names) & set(other_names)
            if not isinstance(nodes.get(n), str)
        }
        if clashes:
            raise InstallError(
                f"Bare-name collision on {sorted(clashes)}: both "
                f"{target_dist.root!r} and {other_dist.root!r} expose them on the "
                f"'*' glob. Use --only with an explicit name, or --prefix to "
                f"namespace one bundle."
            )


def merge_explicit(
    nodes: Mapping[str, NodeEntry], dist: Distribution, names: Mapping[str, str]
) -> dict[str, NodeEntry]:
    """Add/overwrite explicit `recognized: dist:entryPoint` entries."""
    out: dict[str, NodeEntry] = dict(nodes)
    for recognized, entry_point in names.items():
        out[recognized] = f"{dist.root}:{entry_point}"
    return out


def merge_glob(
    nodes: Mapping[str, NodeEntry],
    dist: Distribution,
    key: str = "*",
) -> dict[str, NodeEntry]:
    """Append `dist` to the glob list under `key` (`"*"` or `"<prefix>:*"`)."""
    out: dict[str, NodeEntry] = dict(nodes)
    existing = _glob_distributions(out, key)
    if not any(Distribution(d) == dist for d in existing):
        existing.append(dist.root)
    out[key] = existing
    return out


# ---------- I/O: reading the project environment ----------

# Dumps the node entry-point table (name + declared extras) for one distribution
# as JSON. Run under the *project's* interpreter so it sees a standalone venv,
# not the one `wengine` itself runs under — and so it reads the entry-point
# group name from the project's own `workflow_engine`, rather than `wengine`
# reaching across the `core` boundary for a constant it only needs here.
_READ_ENTRY_POINTS = """
import json, sys
from importlib.metadata import entry_points
from packaging.utils import canonicalize_name
from workflow_engine.core.config import NODES_ENTRY_POINT_GROUP

target = canonicalize_name(sys.argv[1])
out = []
for ep in entry_points(group=NODES_ENTRY_POINT_GROUP):
    if ep.dist is not None and canonicalize_name(ep.dist.name) == target:
        out.append({"name": ep.name, "extras": list(ep.extras)})
print(json.dumps(out))
"""


def _uv_run_python(root: Path, code: str, *args: str) -> str:
    """Run `code` under the project's interpreter, returning stdout.

    Tests monkeypatch this. `uv run` executes in the project environment, which
    is the correct one to read installed entry points from in both embedded and
    standalone modes.
    """
    result = subprocess.run(
        ["uv", "run", "--project", str(root), "python", "-c", code, *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def read_distribution_entry_points(
    root: Path,
    dist: Distribution,
) -> Sequence[MountedNode]:
    """The node entry points `dist` advertises, read from the project env."""
    raw = _uv_run_python(root, _READ_ENTRY_POINTS, dist.root)
    return tuple(
        MountedNode(entry_point_name=item["name"], extras=item["extras"])
        for item in json.loads(raw)
    )


# ---------- orchestration ----------


def install(
    target: str,
    *,
    only: Sequence[str] = (),
    as_name: str | None = None,
    prefix: str | None = None,
    start: Path | None = None,
) -> Distribution:
    """Install `target` and map its nodes in `engine.yaml`. Returns the dist.

    See the module docstring for the full flow and its caveats.
    """
    if prefix is not None and only:
        raise InstallError("--prefix cannot be combined with --only.")
    if prefix is not None and not _PREFIX_RE.match(prefix):
        raise InstallError(f"Invalid --prefix {prefix!r}.")
    if as_name is not None and len(only) != 1:
        raise InstallError("--as is only valid with exactly one --only.")

    project = UvProject.locate(start or Path.cwd())
    parsed: InstallTarget = parse_install_target(target)

    nodes = load_nodes_block(project.engine_yaml)

    # Step 2: fast-fail explicit pre-check when the dist name is known up front
    # (PyPI). For git/path/forge the name is only known post-install, so the
    # authoritative explicit check runs after `uv add` below.
    explicit_names = plan_explicit_names(only, as_name) if only else {}
    # The recognized names (notably a free-form `--as`) are known before any
    # `uv` call, so validate their grammar now — otherwise an invalid name only
    # trips `write_nodes_block` after the package is already installed.
    check_recognized_name_grammar(tuple(explicit_names))
    if explicit_names and isinstance(parsed, PyPITarget):
        pre_dist = Distribution.from_requirement(parsed.requirement)
        check_explicit_collisions(nodes, tuple(explicit_names), pre_dist)

    # Step 3: install (with any user-typed extras already in uv_add_args).
    before = read_pyproject_dependencies(project.root)
    project.add(list(parsed.uv_add_args))
    after = read_pyproject_dependencies(project.root)
    dist = added_distribution(before, after)

    try:
        # Step 4: read what the distribution actually exposes.
        available = read_distribution_entry_points(project.root, dist)

        # Steps 5-7 differ by install kind.
        if only:
            mounts = resolve_only_mounts(available, only)
            check_explicit_collisions(nodes, tuple(explicit_names), dist)
            new_nodes = merge_explicit(nodes, dist, explicit_names)
        elif prefix is not None:
            mounts = available  # whole bundle, own keyspace — no glob check
            new_nodes = merge_glob(nodes, dist, f"{prefix}:*")
        else:  # bulk install onto the "*" glob
            mounts = available
            neighbor_eps = {
                neighbor: [
                    m.entry_point_name
                    for m in read_distribution_entry_points(project.root, neighbor)
                ]
                for neighbor in glob_neighbors(nodes, dist)
            }
            check_glob_collision(
                nodes, dist, [m.entry_point_name for m in mounts], neighbor_eps
            )
            new_nodes = merge_glob(nodes, dist)

        # Step 5: re-add with the union of mounted nodes' extras, if any.
        extras = extras_union(mounts)
        if extras:
            project.add([with_extras(dist, extras)])
    except (InstallError, subprocess.CalledProcessError) as e:
        # Roll back the install we can no longer map cleanly. A uv failure
        # mid-flow (e.g. the extras re-add not resolving) lands here too, so
        # the project doesn't keep a dist it never mapped.
        project.remove(dist.root)
        if isinstance(e, InstallError):
            raise
        raise InstallError(
            f"Installed {dist.root!r} but could not finish mapping it ({e}); "
            f"rolled back."
        ) from e

    write_nodes_block(project.engine_yaml, new_nodes)
    return dist


__all__ = [
    "InstallError",
    "install",
]
