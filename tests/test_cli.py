"""Tests for the wengine CLI."""

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import workflow_engine.nodes  # noqa: F401  — ensure all value types register before any test freezes the value registry
from workflow_engine.cli.main import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """A wengine config that includes the built-in arithmetic + io nodes."""
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "nodes": {
                    "Input": "workflow_engine.core.io.InputNode",
                    "Output": "workflow_engine.core.io.OutputNode",
                    "Sum": "workflow_engine.nodes.arithmetic.SumNode",
                }
            }
        )
    )
    return path


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def _run(runner: CliRunner, *args: str) -> str:
    result = runner.invoke(cli, list(args), catch_exceptions=False)
    assert result.exit_code == 0, f"args={args}\n{result.output}"
    return result.output


# ---------- config ----------


class TestConfig:
    def test_path_prints_default_location(self, runner: CliRunner):
        out = _run(runner, "config", "path")
        assert out.strip().endswith("config.yaml")

    def test_show_prints_file_contents(self, runner: CliRunner, config_path: Path):
        out = _run(runner, "config", "show", "--config", str(config_path))
        assert "workflow_engine.nodes.arithmetic.SumNode" in out

    def test_show_errors_when_missing(self, runner: CliRunner, tmp_path: Path):
        missing = tmp_path / "nope.yaml"
        result = runner.invoke(cli, ["config", "show", "--config", str(missing)])
        assert result.exit_code != 0


# ---------- schema ----------


class TestSchema:
    def test_list_includes_concrete_types_and_excludes_generics(
        self, runner: CliRunner
    ):
        out = _run(runner, "schema", "list")
        names = set(out.split())
        assert {"IntegerValue", "StringValue", "JSONValue", "FileValue"} <= names
        assert "SequenceValue" not in names
        assert "StringMapValue" not in names

    def test_check_resolves_concrete(self, runner: CliRunner):
        out = _run(runner, "schema", "check", '{"x-value-type": "JSONValue"}')
        payload = json.loads(out)
        assert payload["title"] == "JSONValue"

    def test_check_resolves_generic(self, runner: CliRunner):
        out = _run(
            runner,
            "schema",
            "check",
            '{"type": "array", "items": {"x-value-type": "StringValue"}}',
        )
        payload = json.loads(out)
        assert payload["title"] == "SequenceValue[StringValue]"
        assert payload["type"] == "array"

    def test_parse_round_trips_value(self, runner: CliRunner):
        out = _run(runner, "schema", "parse", '{"x-value-type": "IntegerValue"}', "42")
        assert json.loads(out) == 42

    def test_parse_round_trips_array(self, runner: CliRunner):
        out = _run(
            runner,
            "schema",
            "parse",
            '{"type": "array", "items": {"x-value-type": "StringValue"}}',
            '["a", "b"]',
        )
        assert json.loads(out) == ["a", "b"]

    def test_parse_rejects_bad_value(self, runner: CliRunner):
        result = runner.invoke(
            cli,
            ["schema", "parse", '{"x-value-type": "IntegerValue"}', '"not-an-int"'],
        )
        assert result.exit_code != 0


# ---------- node ----------


class TestNode:
    def test_list_uses_config(self, runner: CliRunner, config_path: Path):
        out = _run(runner, "node", "list", "--config", str(config_path))
        names = {line.split("\t", 1)[0] for line in out.strip().splitlines()}
        assert {"Input", "Output", "Sum"} == names

    def test_info_returns_metadata(self, runner: CliRunner, config_path: Path):
        out = _run(runner, "node", "info", "Sum", "--config", str(config_path))
        payload = json.loads(out)
        assert payload["name"] == "Sum"
        assert payload["display_name"] == "Sum"
        assert payload["version"] == "0.4.0"

    def test_info_unknown_node_errors(self, runner: CliRunner, config_path: Path):
        result = runner.invoke(
            cli, ["node", "info", "Nonexistent", "--config", str(config_path)]
        )
        assert result.exit_code != 0

    def test_check_emits_io_schemas(self, runner: CliRunner, config_path: Path):
        out = _run(runner, "node", "check", "Sum", "--config", str(config_path))
        payload = json.loads(out)
        assert payload["ok"] is True
        # SumNode input is a SequenceValue[FloatValue] under "values"
        assert "values" in payload["input_schema"]["properties"]
        # Output is a single "sum" field
        assert "sum" in payload["output_schema"]["properties"]

    def test_run_executes_node(
        self, runner: CliRunner, config_path: Path, base_dir: Path
    ):
        out = _run(
            runner,
            "node",
            "run",
            "Sum",
            "{}",
            '{"values": [1.0, 2.0, 3.5]}',
            "--config",
            str(config_path),
            "--base-dir",
            str(base_dir),
        )
        assert json.loads(out) == {"sum": 6.5}


# ---------- workflow ----------


def _write_sum_workflow(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "input_node": {
                    "type": "Input",
                    "id": "input",
                    "params": {
                        "fields": {
                            "nums": {
                                "type": "array",
                                "items": {"x-value-type": "FloatValue"},
                            }
                        }
                    },
                },
                "output_node": {
                    "type": "Output",
                    "id": "output",
                    "params": {"fields": {"total": {"x-value-type": "FloatValue"}}},
                },
                "inner_nodes": [{"type": "Sum", "id": "summer", "params": {}}],
                "edges": [
                    {
                        "source_id": "input",
                        "source_key": "nums",
                        "target_id": "summer",
                        "target_key": "values",
                    },
                    {
                        "source_id": "summer",
                        "source_key": "sum",
                        "target_id": "output",
                        "target_key": "total",
                    },
                ],
            }
        )
    )


class TestWorkflow:
    def test_init_writes_blank_workflow(self, runner: CliRunner, tmp_path: Path):
        wf = tmp_path / "blank.json"
        _run(runner, "workflow", "init", str(wf))
        payload = json.loads(wf.read_text())
        assert payload["inner_nodes"] == []
        assert payload["edges"] == []
        assert payload["input_node"]["type"] == "Input"

    def test_init_refuses_to_overwrite(self, runner: CliRunner, tmp_path: Path):
        wf = tmp_path / "existing.json"
        wf.write_text("{}")
        result = runner.invoke(cli, ["workflow", "init", str(wf)])
        assert result.exit_code != 0

    def test_init_force_overwrites(self, runner: CliRunner, tmp_path: Path):
        wf = tmp_path / "existing.json"
        wf.write_text("{}")
        _run(runner, "workflow", "init", str(wf), "--force")
        payload = json.loads(wf.read_text())
        assert payload["input_node"]["type"] == "Input"

    def test_init_then_check_validates_blank(self, runner: CliRunner, tmp_path: Path):
        wf = tmp_path / "blank.json"
        _run(runner, "workflow", "init", str(wf))
        out = _run(runner, "workflow", "check", str(wf))
        assert json.loads(out)["ok"] is True

    def test_check_real_workflow(
        self, runner: CliRunner, config_path: Path, tmp_path: Path
    ):
        wf = tmp_path / "sum.json"
        _write_sum_workflow(wf)
        out = _run(runner, "workflow", "check", str(wf), "--config", str(config_path))
        payload = json.loads(out)
        assert payload["ok"] is True
        assert "nums" in payload["input_schema"]["properties"]
        assert "total" in payload["output_schema"]["properties"]

    def test_describe_human_includes_edges(
        self, runner: CliRunner, config_path: Path, tmp_path: Path
    ):
        wf = tmp_path / "sum.json"
        _write_sum_workflow(wf)
        out = _run(
            runner,
            "workflow",
            "describe",
            str(wf),
            "--config",
            str(config_path),
        )
        assert "input.nums" in out
        assert "summer.values" in out
        assert "summer.sum" in out
        assert "output.total" in out

    def test_describe_json_has_full_summary(
        self, runner: CliRunner, config_path: Path, tmp_path: Path
    ):
        wf = tmp_path / "sum.json"
        _write_sum_workflow(wf)
        out = _run(
            runner,
            "workflow",
            "describe",
            str(wf),
            "--json",
            "--config",
            str(config_path),
        )
        payload = json.loads(out)
        ids = {n["id"] for n in payload["nodes"]}
        assert ids == {"input", "summer", "output"}
        assert payload["execution_order"][0] == ["input"]
        assert payload["execution_order"][-1] == ["output"]

    def test_run_executes_workflow(
        self,
        runner: CliRunner,
        config_path: Path,
        base_dir: Path,
        tmp_path: Path,
    ):
        wf = tmp_path / "sum.json"
        _write_sum_workflow(wf)
        out = _run(
            runner,
            "workflow",
            "run",
            str(wf),
            '{"nums": [1.5, 2.5, 4.0]}',
            "--config",
            str(config_path),
            "--base-dir",
            str(base_dir),
        )
        payload = json.loads(out)
        assert payload["output"] == {"total": 8.0}
