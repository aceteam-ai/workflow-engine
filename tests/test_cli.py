"""Tests for the wengine CLI."""

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import workflow_engine.nodes  # noqa: F401  — ensure all value types register before any test freezes the value registry
from workflow_engine.cli.main import cli

# Commands that load a WorkflowEngine (schema, node, workflow except ``init``) call
# ``_build_engine``, which discovers the engine project by walking up from the
# process cwd to the nearest ``engine.yaml``. To keep that resolution reproducible
# across machines and CI, the autouse :func:`engine_project` fixture runs every test
# from a temp dir seeded with the minimal config below (built-in I/O + Sum) instead
# of whatever ``engine.yaml`` happens to live on the developer's machine.
CLI_TEST_CONFIG_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "wengine_test_config.yaml"
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def engine_project(tmp_path_factory, monkeypatch) -> Path:
    """Run each test from a temp dir holding the minimal test ``engine.yaml``.

    The CLI finds its engine project by walking up from the process cwd, so
    chdir-ing into a seeded temp dir is how tests pin a stable node map. Tests
    that need a project-free cwd (``TestInit``, ``TestInstall``) override this by
    entering ``runner.isolated_filesystem()``, which chdirs elsewhere.
    """
    project_dir = tmp_path_factory.mktemp("engine_project")
    (project_dir / "engine.yaml").write_text(CLI_TEST_CONFIG_PATH.read_text())
    monkeypatch.chdir(project_dir)
    return project_dir


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def _run(runner: CliRunner, *args: str) -> str:
    """Invoke the CLI with exactly these argv fragments; must exit 0."""
    result = runner.invoke(cli, list(args), catch_exceptions=False)
    assert result.exit_code == 0, f"args={args}\n{result.output}"
    return result.output


def invoke_cli(runner: CliRunner, *args: str, catch_exceptions: bool = True):
    """Invoke the CLI without asserting on exit code.

    Defaults to ``catch_exceptions=True`` so callers can inspect ``exit_code`` when
    the command raises instead of returning a non-zero exit (e.g. Pydantic errors).
    """
    return runner.invoke(cli, list(args), catch_exceptions=catch_exceptions)


def run_with_default_config(runner: CliRunner, *args: str) -> str:
    """Successful CLI run against the seeded test engine project.

    The node map comes from the autouse :func:`engine_project` fixture, so this is
    just :func:`_run`; the name is kept to flag that the command builds an engine.
    """
    return _run(runner, *args)


def invoke_with_default_config(
    runner: CliRunner, *args: str, catch_exceptions: bool = True
):
    """Like :func:`invoke_cli`, against the seeded test engine project."""
    return invoke_cli(runner, *args, catch_exceptions=catch_exceptions)


# ---------- schema ----------


class TestSchema:
    def test_list_includes_concrete_types_and_excludes_generics(
        self, runner: CliRunner
    ):
        out = run_with_default_config(runner, "schema", "list")
        names = set(out.split())
        assert {"IntegerValue", "StringValue", "JSONValue", "FileValue"} <= names
        assert "SequenceValue" not in names
        assert "StringMapValue" not in names

    def test_check_resolves_concrete(self, runner: CliRunner):
        out = run_with_default_config(
            runner, "schema", "check", '{"x-value-type": "JSONValue"}'
        )
        payload = json.loads(out)
        # `schema check` always emits the expanded form — its purpose is to
        # demystify the compact x-value-type input.
        assert payload["title"] == "JSONValue"
        assert "$defs" in payload

    def test_check_resolves_generic(self, runner: CliRunner):
        out = run_with_default_config(
            runner,
            "schema",
            "check",
            '{"type": "array", "items": {"x-value-type": "StringValue"}}',
        )
        payload = json.loads(out)
        assert payload["title"] == "SequenceValue[StringValue]"
        assert payload["type"] == "array"

    def test_parse_round_trips_value(self, runner: CliRunner):
        out = run_with_default_config(
            runner,
            "schema",
            "parse",
            '{"x-value-type": "IntegerValue"}',
            "42",
        )
        assert json.loads(out) == 42

    def test_parse_round_trips_array(self, runner: CliRunner):
        out = run_with_default_config(
            runner,
            "schema",
            "parse",
            '{"type": "array", "items": {"x-value-type": "StringValue"}}',
            '["a", "b"]',
        )
        assert json.loads(out) == ["a", "b"]

    def test_parse_rejects_bad_value(self, runner: CliRunner):
        result = invoke_with_default_config(
            runner,
            "schema",
            "parse",
            '{"x-value-type": "IntegerValue"}',
            '"not-an-int"',
        )
        assert result.exit_code != 0

    def test_invalid_json_input_reports_clean_error(self, runner: CliRunner):
        result = invoke_with_default_config(runner, "schema", "check", "{not json")
        assert result.exit_code != 0
        # Should be a click error, not a Python traceback.
        assert "Traceback" not in result.output
        assert "Invalid JSON" in result.output

    def test_errors_when_no_engine_yaml(self, runner: CliRunner):
        """A command that builds an engine fails clearly when no engine.yaml is found."""
        with runner.isolated_filesystem():
            result = invoke_cli(runner, "schema", "list")
        assert result.exit_code != 0
        assert "No engine.yaml found" in result.output
        assert "wengine init" in result.output


# ---------- node ----------


class TestNode:
    def test_list_uses_config(self, runner: CliRunner):
        out = run_with_default_config(runner, "node", "list")
        # Each line: "<name>  <display>  <description>" — name is the first token.
        names = {line.split()[0] for line in out.strip().splitlines() if line.strip()}
        assert {"Input", "Output", "Sum"} == names
        # Description should be present for built-in nodes.
        assert "Sums a sequence of numbers." in out

    def test_info_returns_metadata(self, runner: CliRunner):
        out = run_with_default_config(runner, "node", "info", "Sum")
        payload = json.loads(out)
        assert payload["name"] == "Sum"
        assert payload["display_name"] == "Sum"
        assert payload["version"] == "0.4.0"

    def test_info_unknown_node_errors(self, runner: CliRunner):
        result = invoke_with_default_config(runner, "node", "info", "Nonexistent")
        assert result.exit_code != 0

    def test_check_emits_io_schemas(self, runner: CliRunner):
        out = run_with_default_config(runner, "node", "check", "Sum")
        payload = json.loads(out)
        assert payload["ok"] is True
        # SumNode input is a SequenceValue[FloatValue] under "values"
        assert "values" in payload["input_schema"]["properties"]
        # Output is a single "sum" field
        assert "sum" in payload["output_schema"]["properties"]

    def test_run_executes_node(self, runner: CliRunner, base_dir: Path):
        out = run_with_default_config(
            runner,
            "node",
            "run",
            "Sum",
            "{}",
            '{"values": [1.0, 2.0, 3.5]}',
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
        out = run_with_default_config(runner, "workflow", "check", str(wf))
        assert json.loads(out)["ok"] is True

    def test_check_real_workflow(self, runner: CliRunner, tmp_path: Path):
        wf = tmp_path / "sum.json"
        _write_sum_workflow(wf)
        out = run_with_default_config(runner, "workflow", "check", str(wf))
        payload = json.loads(out)
        assert payload["ok"] is True
        assert "nums" in payload["input_schema"]["properties"]
        assert "total" in payload["output_schema"]["properties"]

    def test_describe_human_includes_edges(self, runner: CliRunner, tmp_path: Path):
        wf = tmp_path / "sum.json"
        _write_sum_workflow(wf)
        out = run_with_default_config(runner, "workflow", "describe", str(wf))
        assert "input.nums" in out
        assert "summer.values" in out
        assert "summer.sum" in out
        assert "output.total" in out

    def test_describe_json_has_full_summary(self, runner: CliRunner, tmp_path: Path):
        wf = tmp_path / "sum.json"
        _write_sum_workflow(wf)
        out = run_with_default_config(runner, "workflow", "describe", str(wf), "--json")
        payload = json.loads(out)
        ids = {n["id"] for n in payload["nodes"]}
        assert ids == {"input", "summer", "output"}
        assert payload["execution_order"][0] == ["input"]
        assert payload["execution_order"][-1] == ["output"]

    def test_run_executes_workflow(
        self,
        runner: CliRunner,
        base_dir: Path,
        tmp_path: Path,
    ):
        wf = tmp_path / "sum.json"
        _write_sum_workflow(wf)
        out = run_with_default_config(
            runner,
            "workflow",
            "run",
            str(wf),
            '{"nums": [1.5, 2.5, 4.0]}',
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
        self, runner: CliRunner, populated_workflow: Path
    ):
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
        )
        payload = json.loads(populated_workflow.read_text())
        assert [n["id"] for n in payload["inner_nodes"]] == ["summer"]
        assert payload["inner_nodes"][0]["type"] == "Sum"

    def test_add_node_rejects_duplicate_id(
        self, runner: CliRunner, populated_workflow: Path
    ):
        result = invoke_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "input",  # collides with InputNode id
        )
        assert result.exit_code != 0

    def test_add_node_rejects_input_or_output_types(
        self, runner: CliRunner, populated_workflow: Path
    ):
        for type_name in ("Input", "Output"):
            result = invoke_with_default_config(
                runner,
                "workflow",
                "edit",
                "add-node",
                str(populated_workflow),
                type_name,
                f"second-{type_name.lower()}",
            )
            assert result.exit_code != 0
            assert "inner node" in result.output

    def test_remove_node_drops_associated_edges(
        self, runner: CliRunner, populated_workflow: Path
    ):
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
        )
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "input.nums",
            "summer.values",
        )
        out = run_with_default_config(
            runner, "workflow", "edit", "remove-node", str(populated_workflow), "summer"
        )
        assert "1 associated edge" in out
        payload = json.loads(populated_workflow.read_text())
        assert payload["inner_nodes"] == []
        assert payload["edges"] == []

    def test_remove_node_refuses_to_remove_input_or_output(
        self, runner: CliRunner, populated_workflow: Path
    ):
        for nid in ("input", "output"):
            result = invoke_with_default_config(
                runner,
                "workflow",
                "edit",
                "remove-node",
                str(populated_workflow),
                nid,
            )
            assert result.exit_code != 0

    def test_add_edge_validates_types(
        self, runner: CliRunner, populated_workflow: Path
    ):
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
        )
        # Wrong direction: SequenceValue[FloatValue] cannot cast to FloatValue
        result = invoke_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "input.nums",
            "output.total",
        )
        assert result.exit_code != 0
        # And the file should be untouched
        payload = json.loads(populated_workflow.read_text())
        assert payload["edges"] == []

    def test_remove_edge_works_and_errors_when_missing(
        self, runner: CliRunner, populated_workflow: Path
    ):
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
        )
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "input.nums",
            "summer.values",
        )
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "remove-edge",
            str(populated_workflow),
            "input.nums",
            "summer.values",
        )
        # Removing again should fail
        result = invoke_with_default_config(
            runner,
            "workflow",
            "edit",
            "remove-edge",
            str(populated_workflow),
            "input.nums",
            "summer.values",
        )
        assert result.exit_code != 0

    def test_possible_edges_finds_compatible_input(
        self, runner: CliRunner, populated_workflow: Path
    ):
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
        )
        out = run_with_default_config(
            runner,
            "workflow",
            "edit",
            "possible-edges",
            str(populated_workflow),
            "summer.values",
        )
        assert "input.nums" in out

    def test_possible_edges_finds_compatible_output(
        self, runner: CliRunner, populated_workflow: Path
    ):
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
        )
        out = run_with_default_config(
            runner,
            "workflow",
            "edit",
            "possible-edges",
            str(populated_workflow),
            "summer.sum",
        )
        assert "output.total" in out

    def test_possible_edges_excludes_already_wired(
        self, runner: CliRunner, populated_workflow: Path
    ):
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
        )
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "input.nums",
            "summer.values",
        )
        # input.nums (an output handle) is now wired into summer.values; that
        # already-wired target should NOT appear in its candidate list anymore.
        out = run_with_default_config(
            runner,
            "workflow",
            "edit",
            "possible-edges",
            str(populated_workflow),
            "input.nums",
        )
        assert "summer.values" not in out

    def test_full_construction_then_run(
        self,
        runner: CliRunner,
        populated_workflow: Path,
        base_dir: Path,
    ):
        # Build a Sum-pipeline by editing and run it.
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
        )
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "input.nums",
            "summer.values",
        )
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "summer.sum",
            "output.total",
        )
        out = run_with_default_config(
            runner,
            "workflow",
            "run",
            str(populated_workflow),
            '{"nums": [10.0, 20.0, 30.0]}',
            "--base-dir",
            str(base_dir),
        )
        assert json.loads(out)["output"] == {"total": 60.0}

    def test_update_node_modifies_input_fields(self, runner: CliRunner, tmp_path: Path):
        wf = tmp_path / "wf.json"
        _run(runner, "workflow", "init", str(wf))
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "update-node",
            str(wf),
            "input",
            '{"fields": {"x": {"x-value-type": "IntegerValue"}}}',
        )
        payload = json.loads(wf.read_text())
        assert payload["input_node"]["params"]["fields"] == {
            "x": {"x-value-type": "IntegerValue"}
        }

    def test_update_node_unknown_id_errors(
        self, runner: CliRunner, populated_workflow: Path
    ):
        result = invoke_with_default_config(
            runner,
            "workflow",
            "edit",
            "update-node",
            str(populated_workflow),
            "ghost",
            "{}",
        )
        assert result.exit_code != 0

    def test_update_node_bad_params_reports_clean_error(
        self, runner: CliRunner, populated_workflow: Path
    ):
        before = populated_workflow.read_text()
        result = invoke_with_default_config(
            runner,
            "workflow",
            "edit",
            "update-node",
            str(populated_workflow),
            "input",
            '{"fields": "not a dict"}',
        )
        assert result.exit_code != 0
        assert "Traceback" not in result.output
        # File should be untouched
        assert populated_workflow.read_text() == before

    def test_update_node_preserves_inner_node_id(
        self, runner: CliRunner, populated_workflow: Path
    ):
        # Only relevant to test inner-node update path (different from input/output)
        # Sum has Empty params, so update with {} keeps it valid.
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
        )
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "update-node",
            str(populated_workflow),
            "summer",
            "{}",
        )
        payload = json.loads(populated_workflow.read_text())
        ids = [n["id"] for n in payload["inner_nodes"]]
        assert ids == ["summer"]

    def test_add_field_to_input(self, runner: CliRunner, tmp_path: Path):
        wf = tmp_path / "wf.json"
        _run(runner, "workflow", "init", str(wf))
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-field",
            str(wf),
            "input.x",
            '{"x-value-type": "IntegerValue"}',
        )
        payload = json.loads(wf.read_text())
        assert payload["input_node"]["params"]["fields"] == {
            "x": {"x-value-type": "IntegerValue"}
        }

    def test_add_field_rejects_duplicate(
        self, runner: CliRunner, populated_workflow: Path
    ):
        result = invoke_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-field",
            str(populated_workflow),
            "input.nums",
            '{"x-value-type": "IntegerValue"}',
        )
        assert result.exit_code != 0

    def test_update_field_replaces_schema(
        self, runner: CliRunner, populated_workflow: Path
    ):
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "update-field",
            str(populated_workflow),
            "input.nums",
            '{"x-value-type": "IntegerValue"}',
        )
        payload = json.loads(populated_workflow.read_text())
        assert payload["input_node"]["params"]["fields"]["nums"] == {
            "x-value-type": "IntegerValue"
        }

    def test_update_field_rejects_unknown(
        self, runner: CliRunner, populated_workflow: Path
    ):
        result = invoke_with_default_config(
            runner,
            "workflow",
            "edit",
            "update-field",
            str(populated_workflow),
            "input.ghost",
            '{"x-value-type": "IntegerValue"}',
        )
        assert result.exit_code != 0

    def test_remove_field_drops_referencing_edges(
        self, runner: CliRunner, populated_workflow: Path
    ):
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
        )
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "input.nums",
            "summer.values",
        )
        out = run_with_default_config(
            runner,
            "workflow",
            "edit",
            "remove-field",
            str(populated_workflow),
            "input.nums",
        )
        assert "1 associated edge" in out
        payload = json.loads(populated_workflow.read_text())
        assert payload["edges"] == []
        assert payload["input_node"]["params"]["fields"] == {}

    def test_field_commands_reject_inner_node(
        self, runner: CliRunner, populated_workflow: Path
    ):
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
        )
        result = invoke_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-field",
            str(populated_workflow),
            "summer.foo",
            '{"x-value-type": "IntegerValue"}',
        )
        assert result.exit_code != 0
        assert "input or output" in result.output

    def test_update_field_drops_now_incompatible_edge(
        self, runner: CliRunner, populated_workflow: Path
    ):
        # Wire a Sum that consumes input.nums (FloatValue array).
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
        )
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "input.nums",
            "summer.values",
        )
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "summer.sum",
            "output.total",
        )
        # Now break input.nums type — its edge to summer.values should be dropped.
        result = invoke_with_default_config(
            runner,
            "workflow",
            "edit",
            "update-field",
            str(populated_workflow),
            "input.nums",
            '{"x-value-type": "IntegerValue"}',
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
        self, runner: CliRunner, populated_workflow: Path
    ):
        # Same as above but use update-node on the input node.
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-node",
            str(populated_workflow),
            "Sum",
            "summer",
        )
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "add-edge",
            str(populated_workflow),
            "input.nums",
            "summer.values",
        )
        # Replace input's fields entirely with an IntegerValue 'nums'.
        run_with_default_config(
            runner,
            "workflow",
            "edit",
            "update-node",
            str(populated_workflow),
            "input",
            '{"fields": {"nums": {"x-value-type": "IntegerValue"}}}',
        )
        payload = json.loads(populated_workflow.read_text())
        assert payload["edges"] == []


# ---------- init ----------


class TestInit:
    def test_creates_engine_yaml(
        self, runner: CliRunner, monkeypatch, confine_is_file_to
    ):
        # Standalone init shells out to `uv add`; stub it so the test stays
        # offline and deterministic, while still proving we exercised the
        # standalone path.
        from workflow_engine.cli import uv_project as uv_project_module

        uv_calls = []

        def fake_run_uv(*args, **kwargs):
            uv_calls.append((args, kwargs))
            return None

        monkeypatch.setattr(uv_project_module, "_run_uv", fake_run_uv)

        with runner.isolated_filesystem():
            confine_is_file_to(Path.cwd())
            result = invoke_cli(runner, "init")
            assert result.exit_code == 0, result.output
            assert "Created" in result.output
            assert uv_calls, "expected standalone init to invoke uv"
            assert Path("pyproject.toml").is_file()
            assert Path("engine.yaml").is_file()

    def test_refuses_when_engine_yaml_exists(self, runner: CliRunner, monkeypatch):
        from workflow_engine.cli import uv_project as uv_project_module

        monkeypatch.setattr(uv_project_module, "_run_uv", lambda *a, **k: None)

        with runner.isolated_filesystem():
            Path("engine.yaml").write_text("schema_version: 1\nnodes: {}\n")
            result = invoke_cli(runner, "init")
            assert result.exit_code != 0
            assert "already exists" in result.output


# ---------- install ----------


class TestInstall:
    def test_no_target_syncs(self, runner: CliRunner, monkeypatch):
        from workflow_engine.cli import uv_project as uv_project_module

        calls: list[list[str]] = []
        monkeypatch.setattr(
            uv_project_module, "_run_uv", lambda args, **k: calls.append(list(args))
        )

        with runner.isolated_filesystem():
            Path("engine.yaml").write_text("schema_version: 1\nnodes: {}\n")
            Path("pyproject.toml").write_text('[project]\nname = "x"\nversion = "0"\n')
            result = invoke_cli(runner, "install")
            assert result.exit_code == 0, result.output
            assert ["sync"] in calls

    def test_options_without_target_error(self, runner: CliRunner):
        with runner.isolated_filesystem():
            Path("engine.yaml").write_text("schema_version: 1\nnodes: {}\n")
            Path("pyproject.toml").write_text('[project]\nname = "x"\nversion = "0"\n')
            result = invoke_cli(runner, "install", "--prefix", "p")
            assert result.exit_code != 0
            assert "install target" in result.output

    def test_install_maps_node(self, runner: CliRunner, monkeypatch):
        import json

        from workflow_engine.cli import install as install_module
        from workflow_engine.cli import uv_project as uv_project_module

        def fake_run_uv(args, **k):
            if args[0] == "add":
                Path("pyproject.toml").write_text(
                    f'[project]\nname = "x"\nversion = "0"\n'
                    f'dependencies = ["{args[-1]}"]\n'
                )

        monkeypatch.setattr(uv_project_module, "_run_uv", fake_run_uv)
        monkeypatch.setattr(
            install_module,
            "_uv_run_python",
            lambda root, code, *a: json.dumps([{"name": "Foo", "extras": []}]),
        )

        with runner.isolated_filesystem():
            Path("engine.yaml").write_text("schema_version: 1\nnodes: {}\n")
            Path("pyproject.toml").write_text('[project]\nname = "x"\nversion = "0"\n')
            result = invoke_cli(runner, "install", "acme-pkg", "--only", "Foo")
            assert result.exit_code == 0, result.output
            assert "Installed acme-pkg" in result.output
            nodes = yaml.safe_load(Path("engine.yaml").read_text())["nodes"]
            assert nodes["Foo"] == "acme-pkg:Foo"


# ---------- verify ----------

_BLANK_WORKFLOW = {
    "input_node": {"type": "Input", "id": "input", "params": {"fields": {}}},
    "output_node": {"type": "Output", "id": "output", "params": {"fields": {}}},
    "inner_nodes": [],
    "edges": [],
}


class TestVerify:
    def test_passing_workflow(self, runner: CliRunner, engine_project: Path):
        wf = engine_project / "wf.json"
        wf.write_text(json.dumps(_BLANK_WORKFLOW))
        out = run_with_default_config(runner, "verify", str(wf))
        assert "ok" in out
        assert "1/1 workflows valid." in out

    def test_reports_failure_and_exits_nonzero(
        self, runner: CliRunner, engine_project: Path
    ):
        bad = dict(_BLANK_WORKFLOW)
        bad["inner_nodes"] = [{"type": "Nonexistent", "id": "x", "params": {}}]
        wf = engine_project / "bad.json"
        wf.write_text(json.dumps(bad))
        result = invoke_cli(runner, "verify", str(wf))
        assert result.exit_code != 0
        assert "FAIL" in result.output
        assert "0/1 workflows valid." in result.output

    def test_scans_directory(self, runner: CliRunner, engine_project: Path):
        wf_dir = engine_project / "workflows"
        wf_dir.mkdir()
        (wf_dir / "a.json").write_text(json.dumps(_BLANK_WORKFLOW))
        (wf_dir / "b.json").write_text(json.dumps(_BLANK_WORKFLOW))
        out = run_with_default_config(runner, "verify", str(wf_dir))
        assert "2/2 workflows valid." in out
