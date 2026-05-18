"""Tests for `wengine` uv-project discovery and shell-out."""

from __future__ import annotations

from pathlib import Path

import pytest

from workflow_engine.cli import uv_project as uv_project_module
from workflow_engine.cli.uv_project import (
    EngineYamlNotFound,
    UvProject,
    find_engine_yaml,
    find_pyproject,
)


class TestFindEngineYaml:
    def test_in_start_dir(self, tmp_path: Path):
        (tmp_path / "engine.yaml").write_text("nodes: {}\n")
        assert find_engine_yaml(tmp_path) == (tmp_path / "engine.yaml").resolve()

    def test_yml_variant(self, tmp_path: Path):
        (tmp_path / "engine.yml").write_text("nodes: {}\n")
        assert find_engine_yaml(tmp_path) == (tmp_path / "engine.yml").resolve()

    def test_walks_up(self, tmp_path: Path):
        (tmp_path / "engine.yaml").write_text("nodes: {}\n")
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        assert find_engine_yaml(nested) == (tmp_path / "engine.yaml").resolve()

    def test_yaml_preferred_over_yml(self, tmp_path: Path):
        # If both exist in the same dir, .yaml wins (matches tuple order).
        (tmp_path / "engine.yaml").write_text("nodes: {}\n")
        (tmp_path / "engine.yml").write_text("nodes: {}\n")
        result = find_engine_yaml(tmp_path)
        assert result is not None
        assert result.name == "engine.yaml"

    def test_returns_none_if_missing(self, tmp_path: Path):
        assert find_engine_yaml(tmp_path) is None


class TestFindPyproject:
    def test_in_start_dir(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        assert find_pyproject(tmp_path) == (tmp_path / "pyproject.toml").resolve()

    def test_walks_up(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        assert find_pyproject(nested) == (tmp_path / "pyproject.toml").resolve()

    def test_returns_none_if_missing(self, tmp_path: Path):
        # tmp_path is somewhere under /tmp — no pyproject above it.
        assert find_pyproject(tmp_path) is None


class TestUvProjectLocate:
    def test_embedded_same_dir(self, tmp_path: Path):
        (tmp_path / "engine.yaml").write_text("nodes: {}\n")
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")

        proj = UvProject.locate(tmp_path)
        assert proj.mode == "embedded"
        assert proj.root == tmp_path.resolve()
        assert proj.engine_yaml == (tmp_path / "engine.yaml").resolve()

    def test_embedded_pyproject_above_engine_yaml(self, tmp_path: Path):
        # engine.yaml deeper than pyproject.toml — still embedded, root at
        # the pyproject's dir.
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        subdir = tmp_path / "engine"
        subdir.mkdir()
        (subdir / "engine.yaml").write_text("nodes: {}\n")

        proj = UvProject.locate(subdir)
        assert proj.mode == "embedded"
        assert proj.root == tmp_path.resolve()
        assert proj.engine_yaml == (subdir / "engine.yaml").resolve()

    def test_standalone(self, tmp_path: Path, monkeypatch):
        # engine.yaml with no pyproject.toml anywhere above it. Stub
        # find_pyproject so the result doesn't depend on ambient state
        # (e.g. a pyproject.toml in /home/<user>).
        (tmp_path / "engine.yaml").write_text("nodes: {}\n")
        monkeypatch.setattr(uv_project_module, "find_pyproject", lambda _s: None)

        proj = UvProject.locate(tmp_path)
        assert proj.mode == "standalone"
        assert proj.root == tmp_path.resolve()

    def test_raises_when_no_engine_yaml(self, tmp_path: Path):
        with pytest.raises(EngineYamlNotFound, match=r"No engine\.yaml"):
            UvProject.locate(tmp_path)


class TestEnsurePyproject:
    def test_standalone_writes_minimal(self, tmp_path: Path, monkeypatch):
        (tmp_path / "engine.yaml").write_text("nodes: {}\n")
        # Force standalone regardless of ambient parent pyproject files.
        monkeypatch.setattr(uv_project_module, "find_pyproject", lambda _start: None)

        proj = UvProject.locate(tmp_path)
        assert proj.mode == "standalone"
        proj.ensure_pyproject()

        content = (tmp_path / "pyproject.toml").read_text()
        assert "[project]" in content
        assert 'name = "wengine-nodes"' in content
        assert "dependencies = []" in content

    def test_standalone_does_not_overwrite(self, tmp_path: Path, monkeypatch):
        (tmp_path / "engine.yaml").write_text("nodes: {}\n")
        monkeypatch.setattr(uv_project_module, "find_pyproject", lambda _start: None)

        proj = UvProject.locate(tmp_path)
        # User-written pyproject (locate didn't see it because find_pyproject
        # was stubbed). ensure_pyproject must not clobber it.
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'user-file'\n")
        proj.ensure_pyproject()
        assert "user-file" in (tmp_path / "pyproject.toml").read_text()

    def test_embedded_is_noop(self, tmp_path: Path):
        (tmp_path / "engine.yaml").write_text("nodes: {}\n")
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'host'\n")

        proj = UvProject.locate(tmp_path)
        assert proj.mode == "embedded"
        proj.ensure_pyproject()  # must not touch the host's file
        assert "host" in (tmp_path / "pyproject.toml").read_text()


class _RecordedUvCall:
    def __init__(self, args: list[str], cwd: Path) -> None:
        self.args = args
        self.cwd = cwd


def _patch_run_uv(monkeypatch) -> list[_RecordedUvCall]:
    calls: list[_RecordedUvCall] = []

    def fake(args, *, cwd: Path) -> None:
        calls.append(_RecordedUvCall(list(args), cwd))

    monkeypatch.setattr(uv_project_module, "_run_uv", fake)
    return calls


class TestUvCommands:
    def test_add_passes_args_through(self, tmp_path: Path, monkeypatch):
        (tmp_path / "engine.yaml").write_text("nodes: {}\n")
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        calls = _patch_run_uv(monkeypatch)

        proj = UvProject.locate(tmp_path)
        proj.add(["acme-scrapers==1.4.0"])

        assert len(calls) == 1
        assert calls[0].args == ["add", "acme-scrapers==1.4.0"]
        assert calls[0].cwd == tmp_path.resolve()

    def test_add_in_standalone_ensures_pyproject(self, tmp_path: Path, monkeypatch):
        (tmp_path / "engine.yaml").write_text("nodes: {}\n")
        monkeypatch.setattr(uv_project_module, "find_pyproject", lambda _s: None)
        _patch_run_uv(monkeypatch)

        proj = UvProject.locate(tmp_path)
        assert proj.mode == "standalone"
        assert not (tmp_path / "pyproject.toml").exists()

        proj.add(["acme-scrapers"])
        assert (tmp_path / "pyproject.toml").exists()

    def test_remove(self, tmp_path: Path, monkeypatch):
        (tmp_path / "engine.yaml").write_text("nodes: {}\n")
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        calls = _patch_run_uv(monkeypatch)

        UvProject.locate(tmp_path).remove("acme-scrapers")
        assert calls[0].args == ["remove", "acme-scrapers"]

    def test_sync(self, tmp_path: Path, monkeypatch):
        (tmp_path / "engine.yaml").write_text("nodes: {}\n")
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        calls = _patch_run_uv(monkeypatch)

        UvProject.locate(tmp_path).sync()
        assert calls[0].args == ["sync"]
