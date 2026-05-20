"""Tests for `wengine install` — pure helpers and the install orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from packaging.utils import canonicalize_name

from workflow_engine.cli import install as install_module
from workflow_engine.cli import uv_project as uv_project_module
from workflow_engine.cli.install import (
    InstallError,
    MountedNode,
    added_distribution,
    check_explicit_collisions,
    check_glob_collision,
    distribution_name,
    extras_union,
    glob_neighbors,
    install,
    load_nodes_block,
    merge_explicit,
    merge_glob,
    plan_explicit_names,
    read_pyproject_dependencies,
    resolve_only_mounts,
    with_extras,
    write_nodes_block,
)

# ---------- pure helpers ----------


class TestDistributionName:
    @pytest.mark.parametrize(
        "req,expected",
        [
            ("acme-scrapers", "acme-scrapers"),
            ("acme-scrapers==1.4.0", "acme-scrapers"),
            ("acme-scrapers[screenshot]", "acme-scrapers"),
            ("acme-scrapers[a,b]>=1.4", "acme-scrapers"),
        ],
    )
    def test_parses_name(self, req: str, expected: str):
        assert distribution_name(req) == expected


class TestAddedDistribution:
    def test_single_new(self):
        before = ["aceteam-workflow-engine"]
        after = ["aceteam-workflow-engine", "acme-scrapers>=1.4.0"]
        assert added_distribution(before, after) == "acme-scrapers"

    def test_extras_only_change_is_not_new(self):
        # Re-adding an existing dep with extras must not count as a new dist.
        before = ["acme-scrapers"]
        after = ["acme-scrapers[screenshot]"]
        with pytest.raises(InstallError, match="exactly one new"):
            added_distribution(before, after)

    def test_two_new_is_error(self):
        with pytest.raises(InstallError):
            added_distribution([], ["a", "b"])


class TestWithExtras:
    def test_adds_extra(self):
        assert (
            with_extras("acme-scrapers", ["screenshot"]) == "acme-scrapers[screenshot]"
        )

    def test_merges_with_existing(self):
        out = with_extras("acme-scrapers[a]", ["b"])
        assert distribution_name(out) == "acme-scrapers"
        assert "a" in out and "b" in out

    def test_preserves_version(self):
        assert with_extras("acme-scrapers>=1.4", ["x"]) == "acme-scrapers[x]>=1.4"


class TestExtrasUnion:
    def test_union_sorted(self):
        nodes = [
            MountedNode("A", ("z", "a")),
            MountedNode("B", ("a", "m")),
            MountedNode("C", ()),
        ]
        assert extras_union(nodes) == ("a", "m", "z")


class TestResolveOnlyMounts:
    def test_subset_in_order(self):
        available = [
            MountedNode("A", ()),
            MountedNode("B", ("x",)),
            MountedNode("C", ()),
        ]
        out = resolve_only_mounts(available, ["C", "A"])
        assert [m.entry_point_name for m in out] == ["C", "A"]

    def test_unknown_name_errors(self):
        with pytest.raises(InstallError, match="not exposed"):
            resolve_only_mounts([MountedNode("A", ())], ["Nope"])


class TestPlanExplicitNames:
    def test_only_maps_to_self(self):
        assert plan_explicit_names(["A", "B"], None) == {"A": "A", "B": "B"}

    def test_as_renames_single(self):
        assert plan_explicit_names(["ForEachV2"], "ForEach") == {"ForEach": "ForEachV2"}

    def test_as_with_multiple_only_errors(self):
        with pytest.raises(InstallError, match="--as is only valid"):
            plan_explicit_names(["A", "B"], "X")


class TestCheckExplicitCollisions:
    def test_same_dist_is_noop(self):
        nodes = {"Sum": "aceteam-workflow-engine:Sum"}
        check_explicit_collisions(
            nodes, ["Sum"], "aceteam-workflow-engine", force=False
        )

    def test_different_dist_errors(self):
        nodes = {"ForEach": "aceteam-workflow-engine:ForEach"}
        with pytest.raises(InstallError, match="already mapped"):
            check_explicit_collisions(nodes, ["ForEach"], "acme-iteration", force=False)

    def test_force_bypasses(self):
        nodes = {"ForEach": "aceteam-workflow-engine:ForEach"}
        check_explicit_collisions(nodes, ["ForEach"], "acme-iteration", force=True)

    def test_absent_name_is_fine(self):
        check_explicit_collisions({}, ["New"], "acme", force=False)


class TestGlobCollision:
    def test_no_neighbors_ok(self):
        check_glob_collision({"*": ["acme"]}, "acme", ["A", "B"], {})

    def test_clash_errors(self):
        nodes = {"*": ["acme", "other"]}
        with pytest.raises(InstallError, match="Bare-name collision"):
            check_glob_collision(nodes, "acme", ["A", "B"], {"other": ["B", "C"]})

    def test_explicit_entry_disambiguates(self):
        # B has an explicit entry → not a glob clash.
        nodes = {"*": ["acme", "other"], "B": "acme:B"}
        check_glob_collision(nodes, "acme", ["A", "B"], {"other": ["B"]})

    def test_glob_neighbors_excludes_target(self):
        assert glob_neighbors({"*": ["acme", "other"]}, "acme") == ["other"]


class TestMerge:
    def test_merge_explicit(self):
        out = merge_explicit({}, "acme", {"Foo": "FooNode"})
        assert out["Foo"] == "acme:FooNode"

    def test_merge_glob_appends(self):
        out = merge_glob({"*": ["a"]}, "b")
        assert out["*"] == ["a", "b"]

    def test_merge_glob_idempotent(self):
        out = merge_glob({"*": ["a"]}, "a")
        assert out["*"] == ["a"]

    def test_merge_glob_prefix(self):
        out = merge_glob({}, "legacy", "vendor/legacy:*")
        assert out["vendor/legacy:*"] == ["legacy"]


class TestNodesBlockIO:
    def test_round_trip(self, tmp_path: Path):
        engine_yaml = tmp_path / "engine.yaml"
        engine_yaml.write_text(
            "schema_version: 1\nnodes:\n  Sum: aceteam-workflow-engine:Sum\n"
        )
        nodes = load_nodes_block(engine_yaml)
        nodes["New"] = "acme:New"
        write_nodes_block(engine_yaml, nodes)
        reloaded = yaml.safe_load(engine_yaml.read_text())
        assert reloaded["schema_version"] == 1
        assert reloaded["nodes"]["New"] == "acme:New"


class TestReadPyprojectDependencies:
    def test_reads_list(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "x"\ndependencies = ["a", "b>=1"]\n'
        )
        assert read_pyproject_dependencies(tmp_path) == ["a", "b>=1"]

    def test_missing_file(self, tmp_path: Path):
        assert read_pyproject_dependencies(tmp_path) == []


# ---------- orchestrator (with a uv-simulating fake) ----------


def _write_deps(pyproject: Path, deps: list[str]) -> None:
    body = "".join(f'    "{d}",\n' for d in deps)
    pyproject.write_text(
        f'[project]\nname = "host"\nversion = "0"\ndependencies = [\n{body}]\n'
    )


@pytest.fixture
def project(tmp_path: Path):
    """An embedded engine project (engine.yaml + pyproject side by side)."""
    (tmp_path / "engine.yaml").write_text(
        "schema_version: 1\nnodes:\n  Sum: aceteam-workflow-engine:Sum\n"
    )
    _write_deps(tmp_path / "pyproject.toml", ["aceteam-workflow-engine"])
    return tmp_path


def _patch_uv(
    monkeypatch,
    project: Path,
    entry_points: dict[str, list[MountedNode]],
    resolve_add: dict[str, str] | None = None,
):
    """Simulate `uv add`/`uv remove` mutating pyproject, and entry-point reads.

    `resolve_add` maps a non-PEP-508 add argument (e.g. a `git+https://` URL)
    to the dependency string uv would write — mirroring how uv resolves a URL
    target to a `<dist>` requirement plus a `[tool.uv.sources]` entry.
    """
    resolve_add = resolve_add or {}
    pyproject = project / "pyproject.toml"
    calls: list[list[str]] = []

    def fake_run_uv(args, *, cwd):
        calls.append(list(args))
        deps = read_pyproject_dependencies(project)
        if args[0] == "add":
            req = resolve_add.get(args[-1], args[-1])
            name = canonicalize_name(distribution_name(req))
            deps = [d for d in deps if canonicalize_name(distribution_name(d)) != name]
            deps.append(req)
            _write_deps(pyproject, deps)
        elif args[0] == "remove":
            name = canonicalize_name(args[-1])
            deps = [d for d in deps if canonicalize_name(distribution_name(d)) != name]
            _write_deps(pyproject, deps)

    def fake_uv_run_python(root, code, *args):
        import json

        dist = canonicalize_name(args[-1])
        mounts = entry_points.get(dist, [])
        return json.dumps(
            [{"name": m.entry_point_name, "extras": list(m.extras)} for m in mounts]
        )

    monkeypatch.setattr(uv_project_module, "_run_uv", fake_run_uv)
    monkeypatch.setattr(install_module, "_uv_run_python", fake_uv_run_python)
    return calls


class TestInstallOrchestrator:
    def test_git_target_with_extras(self, project: Path, monkeypatch):
        # A git+ target: uv resolves the URL to a dist name; the --only-driven
        # extra rides into the second uv add against that name.
        url = "git+https://github.com/acme/iteration.git@v2.0.0"
        eps = {"acme-iteration": [MountedNode("ForEachV2", ("browser",))]}
        calls = _patch_uv(
            monkeypatch, project, eps, resolve_add={url: "acme-iteration"}
        )

        dist = install(url, only=["ForEachV2"], as_name="ForEach", start=project)

        assert dist == "acme-iteration"
        assert ["add", url] in calls
        assert ["add", "acme-iteration[browser]"] in calls
        nodes = load_nodes_block(project / "engine.yaml")
        assert nodes["ForEach"] == "acme-iteration:ForEachV2"

    def test_bulk_appends_to_glob(self, project: Path, monkeypatch):
        eps = {
            "acme-scrapers": [
                MountedNode("HttpFetch", ()),
                MountedNode("HtmlExtract", ()),
            ]
        }
        calls = _patch_uv(monkeypatch, project, eps)

        dist = install("acme-scrapers", start=project)

        assert dist == "acme-scrapers"
        nodes = load_nodes_block(project / "engine.yaml")
        assert nodes["*"] == ["acme-scrapers"]
        assert ["add", "acme-scrapers"] in calls

    def test_only_writes_explicit_entry(self, project: Path, monkeypatch):
        eps = {
            "acme-scrapers": [
                MountedNode("HttpFetch", ()),
                MountedNode("Screenshot", ("screenshot",)),
            ]
        }
        _patch_uv(monkeypatch, project, eps)

        install("acme-scrapers", only=["HttpFetch"], start=project)

        nodes = load_nodes_block(project / "engine.yaml")
        assert nodes["HttpFetch"] == "acme-scrapers:HttpFetch"
        assert "*" not in nodes

    def test_only_with_extras_re_adds(self, project: Path, monkeypatch):
        eps = {"acme-scrapers": [MountedNode("Screenshot", ("screenshot",))]}
        calls = _patch_uv(monkeypatch, project, eps)

        install("acme-scrapers", only=["Screenshot"], start=project)

        # Second uv add carries the extra.
        assert ["add", "acme-scrapers[screenshot]"] in calls

    def test_as_rename(self, project: Path, monkeypatch):
        eps = {"acme-iteration": [MountedNode("ForEachV2", ())]}
        _patch_uv(monkeypatch, project, eps)

        install(
            "acme-iteration", only=["ForEachV2"], as_name="NewForEach", start=project
        )

        nodes = load_nodes_block(project / "engine.yaml")
        assert nodes["NewForEach"] == "acme-iteration:ForEachV2"

    def test_explicit_collision_pre_check_aborts_before_uv(
        self, project: Path, monkeypatch
    ):
        # `Sum` already maps to the engine distribution; installing a different
        # dist as `Sum` (PyPI, dist known up front) must fail before any uv call.
        eps = {"other-pkg": [MountedNode("Sum", ())]}
        calls = _patch_uv(monkeypatch, project, eps)

        with pytest.raises(InstallError, match="already mapped"):
            install("other-pkg", only=["Sum"], start=project)
        assert calls == []  # aborted before touching uv

    def test_prefix_mounts_under_namespace(self, project: Path, monkeypatch):
        eps = {"legacy-nodes": [MountedNode("Sum", ()), MountedNode("Foo", ())]}
        _patch_uv(monkeypatch, project, eps)

        install("legacy-nodes", prefix="vendor/legacy", start=project)

        nodes = load_nodes_block(project / "engine.yaml")
        assert nodes["vendor/legacy:*"] == ["legacy-nodes"]

    def test_glob_collision_rolls_back(self, project: Path, monkeypatch):
        # acme already glob-mounted; new bulk install collides on a bare name.
        engine_yaml = project / "engine.yaml"
        engine_yaml.write_text("schema_version: 1\nnodes:\n  '*':\n    - acme\n")
        _write_deps(project / "pyproject.toml", ["aceteam-workflow-engine", "acme"])
        eps = {
            "acme": [MountedNode("Dup", ())],
            "newpkg": [MountedNode("Dup", ())],
        }
        calls = _patch_uv(monkeypatch, project, eps)

        with pytest.raises(InstallError, match="Bare-name collision"):
            install("newpkg", start=project)
        # Rolled back: a uv remove for the just-added dist.
        assert ["remove", "newpkg"] in calls
        # engine.yaml unchanged (newpkg not on the glob).
        assert load_nodes_block(engine_yaml)["*"] == ["acme"]
