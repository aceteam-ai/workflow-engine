"""Tests for `wengine uninstall` — pure helpers and the uninstall orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from workflow_engine.cli import install as install_module
from workflow_engine.cli import uv_project as uv_project_module
from workflow_engine.cli.install import (
    MountedNode,
    load_nodes_block,
    read_pyproject_dependencies,
)
from workflow_engine.cli.uninstall import (
    UninstallError,
    distribution_for_node,
    narrow_dependency_extras,
    references_distribution,
    remaining_mapped_extras,
    remove_distribution_entries,
    uninstall,
)
from workflow_engine.core.config import Distribution

# ---------- pure helpers ----------


class TestDistributionForNode:
    def test_explicit_entry(self):
        nodes = {"Sum": "aceteam-workflow-engine:Sum"}
        assert distribution_for_node(nodes, "Sum") == Distribution(
            "aceteam-workflow-engine"
        )

    def test_glob_only_name_errors(self):
        nodes = {"*": ["acme-scrapers"]}
        with pytest.raises(UninstallError, match="No explicit entry"):
            distribution_for_node(nodes, "HttpFetch")

    def test_missing_name_errors(self):
        with pytest.raises(UninstallError, match="No explicit entry"):
            distribution_for_node({}, "Nope")


class TestReferencesDistribution:
    def test_explicit(self):
        assert references_distribution({"Sum": "acme:Sum"}, Distribution("acme"))

    def test_glob_list(self):
        assert references_distribution({"*": ["acme", "other"]}, Distribution("acme"))

    def test_prefixed_glob_string(self):
        assert references_distribution({"v/legacy:*": "acme"}, Distribution("acme"))

    def test_canonical_name_match(self):
        # PEP 503: "Acme_Scrapers" and "acme-scrapers" are the same distribution.
        assert references_distribution(
            {"Foo": "Acme_Scrapers:Foo"}, Distribution("acme-scrapers")
        )

    def test_absent(self):
        assert not references_distribution({"Sum": "other:Sum"}, Distribution("acme"))


class TestRemoveDistributionEntries:
    def test_drops_explicit_and_glob_membership(self):
        nodes = {
            "Sum": "aceteam-workflow-engine:Sum",
            "Foo": "acme:Foo",
            "*": ["acme", "other"],
        }
        out = remove_distribution_entries(nodes, Distribution("acme"))
        assert out == {"Sum": "aceteam-workflow-engine:Sum", "*": ["other"]}

    def test_drops_emptied_glob_key(self):
        out = remove_distribution_entries({"*": ["acme"]}, Distribution("acme"))
        assert out == {}

    def test_drops_single_string_glob(self):
        out = remove_distribution_entries(
            {"v/legacy:*": "acme", "Sum": "x:Sum"}, Distribution("acme")
        )
        assert out == {"Sum": "x:Sum"}


class TestRemainingMappedExtras:
    def test_explicit_entries_only(self):
        nodes = {"Cap": "acme:Screenshot"}
        available = [
            MountedNode(entry_point_name="Screenshot", extras=("screenshot",)),
            MountedNode(entry_point_name="HttpFetch", extras=()),
        ]
        assert remaining_mapped_extras(nodes, Distribution("acme"), available) == (
            "screenshot",
        )

    def test_glob_pins_full_extra_set(self):
        # A glob mounts every node, so the union spans all of them.
        nodes = {"*": ["acme"]}
        available = [
            MountedNode(entry_point_name="Screenshot", extras=("screenshot",)),
            MountedNode(entry_point_name="Crawl", extras=("browser",)),
        ]
        assert remaining_mapped_extras(nodes, Distribution("acme"), available) == (
            "browser",
            "screenshot",
        )


class TestNarrowDependencyExtras:
    def test_shrinks_extras_preserving_version(self, tmp_path: Path):
        _write_deps(tmp_path / "pyproject.toml", ["acme[a,b]>=1.0"])
        assert narrow_dependency_extras(tmp_path, Distribution("acme"), ["a"]) is True
        assert read_pyproject_dependencies(tmp_path) == ["acme[a]>=1.0"]

    def test_drops_all_extras(self, tmp_path: Path):
        _write_deps(tmp_path / "pyproject.toml", ["acme[a,b]"])
        assert narrow_dependency_extras(tmp_path, Distribution("acme"), []) is True
        assert read_pyproject_dependencies(tmp_path) == ["acme"]

    def test_no_change_when_already_exact(self, tmp_path: Path):
        _write_deps(tmp_path / "pyproject.toml", ["acme[a]"])
        assert narrow_dependency_extras(tmp_path, Distribution("acme"), ["a"]) is False

    def test_no_change_when_not_a_direct_dependency(self, tmp_path: Path):
        _write_deps(tmp_path / "pyproject.toml", ["other"])
        assert narrow_dependency_extras(tmp_path, Distribution("acme"), []) is False

    def test_leaves_uv_sources_untouched(self, tmp_path: Path):
        # The dependency line carries the extras; the source mapping is separate.
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "h"\nversion = "0"\n'
            'dependencies = ["acme[a,b]"]\n\n'
            "[tool.uv.sources]\n"
            'acme = { git = "https://example.com/acme", rev = "v1" }\n'
        )
        assert narrow_dependency_extras(tmp_path, Distribution("acme"), ["a"]) is True
        text = pyproject.read_text()
        assert '"acme[a]"' in text
        assert 'git = "https://example.com/acme"' in text


# ---------- orchestrator (with a uv-simulating fake) ----------


def _write_deps(pyproject: Path, deps: list[str]) -> None:
    body = "".join(f'    "{d}",\n' for d in deps)
    pyproject.write_text(
        f'[project]\nname = "host"\nversion = "0"\ndependencies = [\n{body}]\n'
    )


def _write_engine(project: Path, nodes_yaml: str) -> None:
    (project / "engine.yaml").write_text(f"schema_version: 1\nnodes:\n{nodes_yaml}")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """An embedded engine project (engine.yaml + pyproject side by side)."""
    _write_deps(tmp_path / "pyproject.toml", ["aceteam-workflow-engine"])
    return tmp_path


def _patch_uv(
    monkeypatch,
    project: Path,
    entry_points: dict[str, list[MountedNode]],
) -> list[list[str]]:
    """Simulate `uv add`/`uv remove` mutating pyproject, and entry-point reads."""
    pyproject = project / "pyproject.toml"
    calls: list[list[str]] = []

    def fake_run_uv(args: list[str], *, cwd):
        calls.append(list(args))
        deps = read_pyproject_dependencies(project)
        if args[0] == "add":
            req = args[-1]
            name = Distribution.from_requirement(req)
            deps = [d for d in deps if Distribution.from_requirement(d) != name]
            deps.append(req)
            _write_deps(pyproject, deps)
        elif args[0] == "remove":
            name = Distribution(args[-1])
            deps = [d for d in deps if Distribution.from_requirement(d) != name]
            _write_deps(pyproject, deps)
        # "sync" reconciles the lockfile from pyproject — a no-op for the fake.

    def fake_uv_run_python(root: Path, code: str, *args: str):
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


class TestUninstallOrchestrator:
    def test_name_last_entry_uv_removes(self, project: Path, monkeypatch):
        # Sum is the only entry for acme — uninstalling it removes the package.
        _write_engine(project, "  Sum: acme:Sum\n")
        _write_deps(project / "pyproject.toml", ["aceteam-workflow-engine", "acme"])
        calls = _patch_uv(
            monkeypatch,
            project,
            {"acme": [MountedNode(entry_point_name="Sum", extras=())]},
        )

        result = uninstall("Sum", start=project)

        assert result.removed_distribution is True
        assert result.distribution == Distribution("acme")
        assert ["remove", "acme"] in calls
        assert load_nodes_block(project / "engine.yaml") == {}

    def test_name_still_referenced_renarrows_extras(self, project: Path, monkeypatch):
        # acme exposes Screenshot[screenshot] and HttpFetch[]; both are mapped and
        # pyproject pins [screenshot]. Dropping Screenshot leaves only HttpFetch,
        # which needs no extra, so the dependency is narrowed to bare acme and the
        # lockfile reconciled with `uv sync` (the package stays installed).
        _write_engine(project, "  Cap: acme:Screenshot\n  Fetch: acme:HttpFetch\n")
        _write_deps(
            project / "pyproject.toml", ["aceteam-workflow-engine", "acme[screenshot]"]
        )
        eps = {
            "acme": [
                MountedNode(entry_point_name="Screenshot", extras=("screenshot",)),
                MountedNode(entry_point_name="HttpFetch", extras=()),
            ]
        }
        calls = _patch_uv(monkeypatch, project, eps)

        result = uninstall("Cap", start=project)

        assert result.removed_distribution is False
        assert ["sync"] in calls
        assert all(c[0] != "remove" for c in calls)
        assert "acme" in read_pyproject_dependencies(project)  # narrowed, not removed
        assert read_pyproject_dependencies(project) == [
            "aceteam-workflow-engine",
            "acme",
        ]
        nodes = load_nodes_block(project / "engine.yaml")
        assert "Cap" not in nodes
        assert nodes["Fetch"] == "acme:HttpFetch"

    def test_name_still_referenced_no_extra_change_skips_uv(
        self, project: Path, monkeypatch
    ):
        # Removing one explicit entry while acme stays on the "*" glob keeps every
        # node mapped, so the extra set is unchanged — no uv call should fire.
        _write_engine(project, '  Cap: acme:Screenshot\n  "*":\n    - acme\n')
        _write_deps(
            project / "pyproject.toml", ["aceteam-workflow-engine", "acme[screenshot]"]
        )
        eps = {
            "acme": [MountedNode(entry_point_name="Screenshot", extras=("screenshot",))]
        }
        calls = _patch_uv(monkeypatch, project, eps)

        result = uninstall("Cap", start=project)

        assert result.removed_distribution is False
        assert calls == []  # extras already match (glob pins the full set)
        nodes = load_nodes_block(project / "engine.yaml")
        assert "Cap" not in nodes
        assert nodes["*"] == ["acme"]

    def test_dist_removes_all_entries(self, project: Path, monkeypatch):
        _write_engine(
            project,
            '  Sum: aceteam-workflow-engine:Sum\n  Cap: acme:Screenshot\n  "*":\n    - acme\n    - other\n',
        )
        _write_deps(project / "pyproject.toml", ["aceteam-workflow-engine", "acme"])
        calls = _patch_uv(
            monkeypatch,
            project,
            {"acme": [MountedNode(entry_point_name="Screenshot", extras=())]},
        )

        result = uninstall(dist="acme", start=project)

        assert result.removed_distribution is True
        assert ["remove", "acme"] in calls
        nodes = load_nodes_block(project / "engine.yaml")
        assert nodes == {"Sum": "aceteam-workflow-engine:Sum", "*": ["other"]}

    def test_dist_not_referenced_errors(self, project: Path, monkeypatch):
        _write_engine(project, "  Sum: aceteam-workflow-engine:Sum\n")
        _patch_uv(monkeypatch, project, {})
        with pytest.raises(UninstallError, match=r"No engine\.yaml entry references"):
            uninstall(dist="acme", start=project)

    def test_name_and_dist_both_given_errors(self, project: Path, monkeypatch):
        _write_engine(project, "  Sum: acme:Sum\n")
        _patch_uv(monkeypatch, project, {})
        with pytest.raises(UninstallError, match="either a node name or --dist"):
            uninstall("Sum", dist="acme", start=project)

    def test_neither_given_errors(self, project: Path, monkeypatch):
        _write_engine(project, "  Sum: acme:Sum\n")
        _patch_uv(monkeypatch, project, {})
        with pytest.raises(UninstallError, match="either a node name or --dist"):
            uninstall(start=project)
