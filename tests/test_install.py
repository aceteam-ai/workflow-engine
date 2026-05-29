"""Tests for `wengine install` — pure helpers and the install orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from workflow_engine.cli import install as install_module
from workflow_engine.cli import uv_project as uv_project_module
from workflow_engine.cli.install import (
    InstallError,
    MountedNode,
    added_distribution,
    check_explicit_collisions,
    check_glob_collision,
    check_recognized_name_grammar,
    extras_union,
    glob_neighbors,
    install,
    load_nodes_block,
    merge_explicit,
    merge_glob,
    plan_explicit_names,
    read_pyproject_dependencies,
    replace_nodes_block,
    resolve_only_mounts,
    with_extras,
)
from workflow_engine.core.config import Distribution

# ---------- pure helpers ----------


class TestAddedDistribution:
    def test_single_new(self):
        before = ["aceteam-workflow-engine"]
        after = ["aceteam-workflow-engine", "acme-scrapers>=1.4.0"]
        assert added_distribution(before, after) == Distribution("acme-scrapers")

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
            with_extras(Distribution("acme-scrapers"), ["screenshot"])
            == "acme-scrapers[screenshot]"
        )

    def test_sorts_multiple_extras(self):
        assert with_extras(Distribution("acme"), ["b", "a"]) == "acme[a,b]"


class TestExtrasUnion:
    def test_union_sorted(self):
        nodes = [
            MountedNode(entry_point_name="A", extras=("z", "a")),
            MountedNode(entry_point_name="B", extras=("a", "m")),
            MountedNode(entry_point_name="C", extras=()),
        ]
        assert extras_union(nodes) == ("a", "m", "z")


class TestResolveOnlyMounts:
    def test_subset_in_order(self):
        available = [
            MountedNode(entry_point_name="A", extras=()),
            MountedNode(entry_point_name="B", extras=("x",)),
            MountedNode(entry_point_name="C", extras=()),
        ]
        out = resolve_only_mounts(available, ["C", "A"])
        assert [m.entry_point_name for m in out] == ["C", "A"]

    def test_unknown_name_errors(self):
        with pytest.raises(InstallError, match="not exposed"):
            resolve_only_mounts(
                [MountedNode(entry_point_name="A", extras=())], ["Nope"]
            )


class TestPlanExplicitNames:
    def test_only_maps_to_self(self):
        assert plan_explicit_names(["A", "B"], None) == {"A": "A", "B": "B"}

    def test_as_renames_single(self):
        assert plan_explicit_names(["ForEachV2"], "ForEach") == {"ForEach": "ForEachV2"}

    def test_as_with_multiple_only_errors(self):
        with pytest.raises(InstallError, match="--as is only valid"):
            plan_explicit_names(["A", "B"], "X")


class TestCheckRecognizedNameGrammar:
    def test_accepts_valid_names(self):
        check_recognized_name_grammar(["Foo", "vendor:Bar"])

    def test_rejects_hyphenated_name(self):
        with pytest.raises(InstallError, match="Invalid node name"):
            check_recognized_name_grammar(["my-node"])


class TestCheckExplicitCollisions:
    def test_same_dist_is_noop(self):
        nodes = {"Sum": "aceteam-workflow-engine:Sum"}
        check_explicit_collisions(
            nodes, ["Sum"], Distribution("aceteam-workflow-engine")
        )

    def test_different_dist_errors(self):
        nodes = {"ForEach": "aceteam-workflow-engine:ForEach"}
        with pytest.raises(InstallError, match="already mapped"):
            check_explicit_collisions(
                nodes, ["ForEach"], Distribution("acme-iteration")
            )

    def test_absent_name_is_fine(self):
        check_explicit_collisions({}, ["New"], Distribution("acme"))


class TestGlobCollision:
    def test_no_neighbors_ok(self):
        check_glob_collision({"*": ["acme"]}, Distribution("acme"), ["A", "B"], {})

    def test_clash_errors(self):
        nodes = {"*": ["acme", "other"]}
        with pytest.raises(InstallError, match="Bare-name collision"):
            check_glob_collision(
                nodes,
                Distribution("acme"),
                ["A", "B"],
                {Distribution("other"): ["B", "C"]},
            )

    def test_explicit_entry_disambiguates(self):
        # B has an explicit entry → not a glob clash.
        nodes = {"*": ["acme", "other"], "B": "acme:B"}
        check_glob_collision(
            nodes, Distribution("acme"), ["A", "B"], {Distribution("other"): ["B"]}
        )

    def test_glob_neighbors_excludes_target(self):
        assert glob_neighbors({"*": ["acme", "other"]}, Distribution("acme")) == [
            Distribution("other")
        ]


class TestMerge:
    def test_merge_explicit(self):
        out = merge_explicit({}, Distribution("acme"), {"Foo": "FooNode"})
        assert out["Foo"] == "acme:FooNode"

    def test_merge_glob_appends(self):
        out = merge_glob({"*": ["a"]}, Distribution("b"))
        assert out["*"] == ["a", "b"]

    def test_merge_glob_idempotent(self):
        out = merge_glob({"*": ["a"]}, Distribution("a"))
        assert out["*"] == ["a"]

    def test_merge_glob_prefix(self):
        out = merge_glob({}, Distribution("legacy"), "vendor/legacy:*")
        assert out["vendor/legacy:*"] == ["legacy"]


class TestNodesBlockIO:
    def test_round_trip(self, tmp_path: Path):
        engine_yaml = tmp_path / "engine.yaml"
        engine_yaml.write_text(
            "schema_version: 1\nnodes:\n  Sum: aceteam-workflow-engine:Sum\n"
        )
        nodes = load_nodes_block(engine_yaml)
        nodes["New"] = "acme:New"
        replace_nodes_block(engine_yaml, nodes)
        reloaded = yaml.safe_load(engine_yaml.read_text())
        assert reloaded["schema_version"] == 1
        assert reloaded["nodes"]["New"] == "acme:New"

    def test_replaces_block_dropping_omitted_entries(self, tmp_path: Path):
        # replace_nodes_block writes the *complete* block: a partial mapping
        # drops the entries it omits. Callers must load → mutate → write back.
        engine_yaml = tmp_path / "engine.yaml"
        engine_yaml.write_text(
            "schema_version: 1\n"
            "nodes:\n"
            "  Sum: aceteam-workflow-engine:Sum\n"
            "  Product: aceteam-workflow-engine:Product\n"
        )
        replace_nodes_block(engine_yaml, {"Sum": "aceteam-workflow-engine:Sum"})
        reloaded = yaml.safe_load(engine_yaml.read_text())
        assert "Product" not in reloaded["nodes"]  # omitted → dropped
        assert reloaded["nodes"]["Sum"] == "aceteam-workflow-engine:Sum"

    def test_preserves_comments_and_unknown_keys(self, tmp_path: Path):
        engine_yaml = tmp_path / "engine.yaml"
        engine_yaml.write_text(
            "# top comment\n"
            "schema_version: 1\n"
            "nodes:\n"
            "  Sum: aceteam-workflow-engine:Sum  # inline\n"
            "future_key:\n"
            "  keep: me\n"
        )
        nodes = load_nodes_block(engine_yaml)
        nodes["New"] = "acme:New"
        replace_nodes_block(engine_yaml, nodes)
        text = engine_yaml.read_text()
        assert "# top comment" in text
        assert "# inline" in text  # comment on a surviving entry is kept
        assert "future_key:" in text  # untouched top-level keys round-trip
        assert "New: acme:New" in text

    def test_rejects_non_mapping_nodes(self, tmp_path: Path):
        engine_yaml = tmp_path / "engine.yaml"
        engine_yaml.write_text("schema_version: 1\nnodes:\n  - not\n  - a mapping\n")
        with pytest.raises(InstallError, match="invalid nodes block"):
            load_nodes_block(engine_yaml)


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

    def fake_run_uv(args: list[str], *, cwd):
        calls.append(list(args))
        deps = read_pyproject_dependencies(project)
        if args[0] == "add":
            req = resolve_add.get(args[-1], args[-1])
            name = Distribution.from_requirement(req)
            deps = [d for d in deps if Distribution.from_requirement(d) != name]
            deps.append(req)
            _write_deps(pyproject, deps)
        elif args[0] == "remove":
            name = Distribution(args[-1])
            deps = [d for d in deps if Distribution.from_requirement(d) != name]
            _write_deps(pyproject, deps)

    def fake_uv_run_python(root: Path, code: str, *args: str):
        import json

        target = Distribution(args[-1])
        mounts = next(
            (m for k, m in entry_points.items() if Distribution(k) == target), []
        )
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
        eps = {
            "acme-iteration": [
                MountedNode(entry_point_name="ForEachV2", extras=("browser",))
            ]
        }
        calls = _patch_uv(
            monkeypatch, project, eps, resolve_add={url: "acme-iteration"}
        )

        dist = install(url, only=["ForEachV2"], as_name="ForEach", start=project)

        assert dist == Distribution("acme-iteration")
        assert ["add", url] in calls
        assert ["add", "acme-iteration[browser]"] in calls
        nodes = load_nodes_block(project / "engine.yaml")
        assert nodes["ForEach"] == "acme-iteration:ForEachV2"

    def test_bulk_appends_to_glob(self, project: Path, monkeypatch):
        eps = {
            "acme-scrapers": [
                MountedNode(entry_point_name="HttpFetch", extras=()),
                MountedNode(entry_point_name="HtmlExtract", extras=()),
            ]
        }
        calls = _patch_uv(monkeypatch, project, eps)

        dist = install("acme-scrapers", start=project)

        assert dist == Distribution("acme-scrapers")
        nodes = load_nodes_block(project / "engine.yaml")
        assert nodes["*"] == ["acme-scrapers"]
        assert ["add", "acme-scrapers"] in calls

    def test_only_writes_explicit_entry(self, project: Path, monkeypatch):
        eps = {
            "acme-scrapers": [
                MountedNode(entry_point_name="HttpFetch", extras=()),
                MountedNode(entry_point_name="Screenshot", extras=("screenshot",)),
            ]
        }
        _patch_uv(monkeypatch, project, eps)

        install("acme-scrapers", only=["HttpFetch"], start=project)

        nodes = load_nodes_block(project / "engine.yaml")
        assert nodes["HttpFetch"] == "acme-scrapers:HttpFetch"
        assert "*" not in nodes

    def test_only_with_extras_re_adds(self, project: Path, monkeypatch):
        eps = {
            "acme-scrapers": [
                MountedNode(entry_point_name="Screenshot", extras=("screenshot",))
            ]
        }
        calls = _patch_uv(monkeypatch, project, eps)

        install("acme-scrapers", only=["Screenshot"], start=project)

        # Second uv add carries the extra.
        assert ["add", "acme-scrapers[screenshot]"] in calls

    def test_as_rename(self, project: Path, monkeypatch):
        eps = {"acme-iteration": [MountedNode(entry_point_name="ForEachV2", extras=())]}
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
        eps = {"other-pkg": [MountedNode(entry_point_name="Sum", extras=())]}
        calls = _patch_uv(monkeypatch, project, eps)

        with pytest.raises(InstallError, match="already mapped"):
            install("other-pkg", only=["Sum"], start=project)
        assert calls == []  # aborted before touching uv

    def test_prefix_mounts_under_namespace(self, project: Path, monkeypatch):
        eps = {
            "legacy-nodes": [
                MountedNode(entry_point_name="Sum", extras=()),
                MountedNode(entry_point_name="Foo", extras=()),
            ]
        }
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
            "acme": [MountedNode(entry_point_name="Dup", extras=())],
            "newpkg": [MountedNode(entry_point_name="Dup", extras=())],
        }
        calls = _patch_uv(monkeypatch, project, eps)

        with pytest.raises(InstallError, match="Bare-name collision"):
            install("newpkg", start=project)
        # Rolled back: a uv remove for the just-added dist.
        assert ["remove", "newpkg"] in calls
        # engine.yaml unchanged (newpkg not on the glob).
        assert load_nodes_block(engine_yaml)["*"] == ["acme"]
