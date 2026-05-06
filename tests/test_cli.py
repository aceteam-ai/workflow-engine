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

    def test_invalid_json_input_reports_clean_error(self, runner: CliRunner):
        result = runner.invoke(cli, ["schema", "check", "{not json"])
        assert result.exit_code != 0
        # Should be a click error, not a Python traceback.
        assert "Traceback" not in result.output
        assert "Invalid JSON" in result.output

    def test_check_accepts_config_flag(self, runner: CliRunner, config_path: Path):
        out = _run(
            runner,
            "schema",
            "check",
            '{"x-value-type": "JSONValue"}',
            "--config",
            str(config_path),
        )
        assert json.loads(out)["title"] == "JSONValue"


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


# ---------- workflow edit ----------


@pytest.fixture
def populated_workflow(tmp_path: Path) -> Path:
    """A workflow with input.nums:array<FloatValue> and output.total:FloatValue, no inner nodes."""
    path = tmp_path / "wf.json"
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
                "inner_nodes": [],
                "edges": [],
            }
        )
    )
    return path


class TestWorkflowEdit:
    def test_add_node_appends_to_inner_nodes(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        _run(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
            "--config",
            str(config_path),
        )
        payload = json.loads(populated_workflow.read_text())
        assert [n["id"] for n in payload["inner_nodes"]] == ["summer"]
        assert payload["inner_nodes"][0]["type"] == "Sum"

    def test_add_node_rejects_duplicate_id(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        result = runner.invoke(
            cli,
            [
                "workflow",
                "edit",
                "add-node",
                str(populated_workflow),
                "Sum",
                "input",  # collides with InputNode id
                "--config",
                str(config_path),
            ],
        )
        assert result.exit_code != 0

    def test_remove_node_drops_associated_edges(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        _run(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
            "--config",
            str(config_path),
        )
        _run(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "input.nums",
            "summer.values",
            "--config",
            str(config_path),
        )
        out = _run(
            runner,
            "workflow",
            "edit",
            "remove-node",
            str(populated_workflow),
            "summer",
            "--config",
            str(config_path),
        )
        assert "1 associated edge" in out
        payload = json.loads(populated_workflow.read_text())
        assert payload["inner_nodes"] == []
        assert payload["edges"] == []

    def test_remove_node_refuses_to_remove_input_or_output(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        for nid in ("input", "output"):
            result = runner.invoke(
                cli,
                [
                    "workflow",
                    "edit",
                    "remove-node",
                    str(populated_workflow),
                    nid,
                    "--config",
                    str(config_path),
                ],
            )
            assert result.exit_code != 0

    def test_add_edge_validates_types(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        _run(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
            "--config",
            str(config_path),
        )
        # Wrong direction: SequenceValue[FloatValue] cannot cast to FloatValue
        result = runner.invoke(
            cli,
            [
                "workflow",
                "edit",
                "add-edge",
                str(populated_workflow),
                "input.nums",
                "output.total",
                "--config",
                str(config_path),
            ],
        )
        assert result.exit_code != 0
        # And the file should be untouched
        payload = json.loads(populated_workflow.read_text())
        assert payload["edges"] == []

    def test_remove_edge_works_and_errors_when_missing(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        _run(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
            "--config",
            str(config_path),
        )
        _run(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "input.nums",
            "summer.values",
            "--config",
            str(config_path),
        )
        _run(
            runner,
            "workflow",
            "edit",
            "remove-edge",
            str(populated_workflow),
            "input.nums",
            "summer.values",
            "--config",
            str(config_path),
        )
        # Removing again should fail
        result = runner.invoke(
            cli,
            [
                "workflow",
                "edit",
                "remove-edge",
                str(populated_workflow),
                "input.nums",
                "summer.values",
                "--config",
                str(config_path),
            ],
        )
        assert result.exit_code != 0

    def test_possible_edges_finds_compatible_input(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        _run(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
            "--config",
            str(config_path),
        )
        out = _run(
            runner,
            "workflow",
            "edit",
            "possible-edges",
            str(populated_workflow),
            "summer.values",
            "--config",
            str(config_path),
        )
        assert "input.nums" in out

    def test_possible_edges_finds_compatible_output(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        _run(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
            "--config",
            str(config_path),
        )
        out = _run(
            runner,
            "workflow",
            "edit",
            "possible-edges",
            str(populated_workflow),
            "summer.sum",
            "--config",
            str(config_path),
        )
        assert "output.total" in out

    def test_possible_edges_excludes_already_wired(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        _run(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
            "--config",
            str(config_path),
        )
        _run(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "input.nums",
            "summer.values",
            "--config",
            str(config_path),
        )
        # input.nums (an output handle) is now wired into summer.values; that
        # already-wired target should NOT appear in its candidate list anymore.
        out = _run(
            runner,
            "workflow",
            "edit",
            "possible-edges",
            str(populated_workflow),
            "input.nums",
            "--config",
            str(config_path),
        )
        assert "summer.values" not in out

    def test_full_construction_then_run(
        self,
        runner: CliRunner,
        config_path: Path,
        populated_workflow: Path,
        base_dir: Path,
    ):
        # Build a Sum-pipeline by editing and run it.
        _run(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
            "--config",
            str(config_path),
        )
        _run(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "input.nums",
            "summer.values",
            "--config",
            str(config_path),
        )
        _run(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "summer.sum",
            "output.total",
            "--config",
            str(config_path),
        )
        out = _run(
            runner,
            "workflow",
            "run",
            str(populated_workflow),
            '{"nums": [10.0, 20.0, 30.0]}',
            "--config",
            str(config_path),
            "--base-dir",
            str(base_dir),
        )
        assert json.loads(out)["output"] == {"total": 60.0}

    def test_update_node_modifies_input_fields(
        self, runner: CliRunner, config_path: Path, tmp_path: Path
    ):
        wf = tmp_path / "wf.json"
        _run(runner, "workflow", "init", str(wf))
        _run(
            runner,
            "workflow",
            "edit",
            "update-node",
            str(wf),
            "input",
            '{"fields": {"x": {"x-value-type": "IntegerValue"}}}',
            "--config",
            str(config_path),
        )
        payload = json.loads(wf.read_text())
        assert payload["input_node"]["params"]["fields"] == {
            "x": {"x-value-type": "IntegerValue"}
        }

    def test_update_node_unknown_id_errors(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        result = runner.invoke(
            cli,
            [
                "workflow",
                "edit",
                "update-node",
                str(populated_workflow),
                "ghost",
                "{}",
                "--config",
                str(config_path),
            ],
        )
        assert result.exit_code != 0

    def test_update_node_bad_params_reports_clean_error(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        before = populated_workflow.read_text()
        result = runner.invoke(
            cli,
            [
                "workflow",
                "edit",
                "update-node",
                str(populated_workflow),
                "input",
                '{"fields": "not a dict"}',
                "--config",
                str(config_path),
            ],
        )
        assert result.exit_code != 0
        assert "Traceback" not in result.output
        # File should be untouched
        assert populated_workflow.read_text() == before

    def test_update_node_preserves_inner_node_id(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        # Only relevant to test inner-node update path (different from input/output)
        # Sum has Empty params, so update with {} keeps it valid.
        _run(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
            "--config",
            str(config_path),
        )
        _run(
            runner,
            "workflow",
            "edit",
            "update-node",
            str(populated_workflow),
            "summer",
            "{}",
            "--config",
            str(config_path),
        )
        payload = json.loads(populated_workflow.read_text())
        ids = [n["id"] for n in payload["inner_nodes"]]
        assert ids == ["summer"]

    def test_add_field_to_input(
        self, runner: CliRunner, config_path: Path, tmp_path: Path
    ):
        wf = tmp_path / "wf.json"
        _run(runner, "workflow", "init", str(wf))
        _run(
            runner,
            "workflow",
            "edit",
            "add-field",
            str(wf),
            "input.x",
            '{"x-value-type": "IntegerValue"}',
            "--config",
            str(config_path),
        )
        payload = json.loads(wf.read_text())
        assert payload["input_node"]["params"]["fields"] == {
            "x": {"x-value-type": "IntegerValue"}
        }

    def test_add_field_rejects_duplicate(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        result = runner.invoke(
            cli,
            [
                "workflow",
                "edit",
                "add-field",
                str(populated_workflow),
                "input.nums",
                '{"x-value-type": "IntegerValue"}',
                "--config",
                str(config_path),
            ],
        )
        assert result.exit_code != 0

    def test_update_field_replaces_schema(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        _run(
            runner,
            "workflow",
            "edit",
            "update-field",
            str(populated_workflow),
            "input.nums",
            '{"x-value-type": "IntegerValue"}',
            "--config",
            str(config_path),
        )
        payload = json.loads(populated_workflow.read_text())
        assert payload["input_node"]["params"]["fields"]["nums"] == {
            "x-value-type": "IntegerValue"
        }

    def test_update_field_rejects_unknown(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        result = runner.invoke(
            cli,
            [
                "workflow",
                "edit",
                "update-field",
                str(populated_workflow),
                "input.ghost",
                '{"x-value-type": "IntegerValue"}',
                "--config",
                str(config_path),
            ],
        )
        assert result.exit_code != 0

    def test_remove_field_drops_referencing_edges(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        _run(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
            "--config",
            str(config_path),
        )
        _run(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "input.nums",
            "summer.values",
            "--config",
            str(config_path),
        )
        out = _run(
            runner,
            "workflow",
            "edit",
            "remove-field",
            str(populated_workflow),
            "input.nums",
            "--config",
            str(config_path),
        )
        assert "1 associated edge" in out
        payload = json.loads(populated_workflow.read_text())
        assert payload["edges"] == []
        assert payload["input_node"]["params"]["fields"] == {}

    def test_field_commands_reject_inner_node(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        _run(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
            "--config",
            str(config_path),
        )
        result = runner.invoke(
            cli,
            [
                "workflow",
                "edit",
                "add-field",
                str(populated_workflow),
                "summer.foo",
                '{"x-value-type": "IntegerValue"}',
                "--config",
                str(config_path),
            ],
        )
        assert result.exit_code != 0
        assert "input or output" in result.output

    def test_update_field_drops_now_incompatible_edge(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        # Wire a Sum that consumes input.nums (FloatValue array).
        _run(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
            "--config",
            str(config_path),
        )
        _run(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "input.nums",
            "summer.values",
            "--config",
            str(config_path),
        )
        _run(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "summer.sum",
            "output.total",
            "--config",
            str(config_path),
        )
        # Now break input.nums type — its edge to summer.values should be dropped.
        result = runner.invoke(
            cli,
            [
                "workflow",
                "edit",
                "update-field",
                str(populated_workflow),
                "input.nums",
                '{"x-value-type": "IntegerValue"}',
                "--config",
                str(config_path),
            ],
        )
        assert result.exit_code == 0, result.output
        # ClickException sends warnings to stderr, but CliRunner combines them by default.
        # Check via mix_stderr or by reading workflow state.
        payload = json.loads(populated_workflow.read_text())
        edge_pairs = {
            (e["source_id"], e["source_key"], e["target_id"], e["target_key"])
            for e in payload["edges"]
        }
        assert ("input", "nums", "summer", "values") not in edge_pairs
        assert ("summer", "sum", "output", "total") in edge_pairs

    def test_update_node_drops_now_incompatible_edge(
        self, runner: CliRunner, config_path: Path, populated_workflow: Path
    ):
        # Same as above but use update-node on the input node.
        _run(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
            "--config",
            str(config_path),
        )
        _run(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "input.nums",
            "summer.values",
            "--config",
            str(config_path),
        )
        # Replace input's fields entirely with an IntegerValue 'nums'.
        _run(
            runner,
            "workflow",
            "edit",
            "update-node",
            str(populated_workflow),
            "input",
            '{"fields": {"nums": {"x-value-type": "IntegerValue"}}}',
            "--config",
            str(config_path),
        )
        payload = json.loads(populated_workflow.read_text())
        assert payload["edges"] == []
