"""Tests for `wengine init` — engine.yaml seeding and uv-project setup."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from workflow_engine.cli import uv_project as uv_project_module
from workflow_engine.cli.engine_init import (
    BUILTIN_DISTRIBUTION,
    EngineYamlAlreadyExists,
    builtin_entry_point_names,
    init_engine_project,
    initial_nodes_config,
    render_engine_yaml,
)
from workflow_engine.core.config import WorkflowEngineConfig


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


class TestBuiltinEntryPoints:
    def test_includes_known_builtins(self):
        names = builtin_entry_point_names()
        assert "Input" in names
        assert "Output" in names
        assert "Sum" in names

    def test_sorted(self):
        names = builtin_entry_point_names()
        assert list(names) == sorted(names)


class TestInitialNodesConfig:
    def test_default_is_global_glob(self):
        nodes = initial_nodes_config(explicit=False)
        assert nodes == {"*": [BUILTIN_DISTRIBUTION]}

    def test_explicit_one_entry_per_builtin(self):
        nodes = initial_nodes_config(explicit=True)
        assert "*" not in nodes
        assert nodes["Input"] == f"{BUILTIN_DISTRIBUTION}:Input"
        assert set(nodes) == set(builtin_entry_point_names())


class TestRenderEngineYaml:
    def test_round_trips_through_config(self):
        # The rendered document must parse as a valid engine.yaml.
        text = render_engine_yaml(initial_nodes_config(explicit=False))
        parsed = yaml.safe_load(text)
        assert parsed["schema_version"] == 1
        WorkflowEngineConfig.model_validate(parsed)

    def test_explicit_round_trips(self):
        text = render_engine_yaml(initial_nodes_config(explicit=True))
        WorkflowEngineConfig.model_validate(yaml.safe_load(text))


class TestInitEngineProject:
    def test_standalone_writes_yaml_and_adds_engine(
        self, tmp_path: Path, monkeypatch, confine_is_file_to
    ):
        # No pyproject above tmp_path → standalone.
        confine_is_file_to(tmp_path)
        calls = _patch_run_uv(monkeypatch)

        engine_yaml = init_engine_project(tmp_path, explicit=False)

        assert engine_yaml == (tmp_path / "engine.yaml").resolve()
        assert engine_yaml.is_file()
        # Standalone: a pyproject was written and the engine distribution added.
        assert (tmp_path / "pyproject.toml").is_file()
        assert len(calls) == 1
        assert calls[0].args == ["add", BUILTIN_DISTRIBUTION]
        assert calls[0].cwd == tmp_path.resolve()

    def test_embedded_writes_yaml_only(self, tmp_path: Path, monkeypatch):
        # An enclosing pyproject.toml → embedded; init must not touch uv.
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'host'\n")
        calls = _patch_run_uv(monkeypatch)

        engine_yaml = init_engine_project(tmp_path, explicit=False)

        assert engine_yaml.is_file()
        assert calls == []
        # The host's pyproject is untouched.
        assert "host" in (tmp_path / "pyproject.toml").read_text()

    def test_written_yaml_is_valid_config(
        self, tmp_path: Path, monkeypatch, confine_is_file_to
    ):
        confine_is_file_to(tmp_path)
        _patch_run_uv(monkeypatch)

        engine_yaml = init_engine_project(tmp_path, explicit=True)
        WorkflowEngineConfig.load(engine_yaml)

    def test_refuses_existing_yaml(self, tmp_path: Path, monkeypatch):
        (tmp_path / "engine.yaml").write_text("schema_version: 1\nnodes: {}\n")
        _patch_run_uv(monkeypatch)

        with pytest.raises(EngineYamlAlreadyExists, match="already exists"):
            init_engine_project(tmp_path)

    def test_refuses_existing_yml_variant(self, tmp_path: Path, monkeypatch):
        (tmp_path / "engine.yml").write_text("schema_version: 1\nnodes: {}\n")
        _patch_run_uv(monkeypatch)

        with pytest.raises(EngineYamlAlreadyExists):
            init_engine_project(tmp_path)
