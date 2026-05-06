"""
Prototype `wengine` CLI.

Self-contained on purpose: per the design doc in docs/cli.md, the first
iteration lives in a single file even if it duplicates small bits of glue
from elsewhere in the package.
"""

import asyncio
import functools
import json
from pathlib import Path
from typing import Any

import click
import platformdirs
import yaml

from workflow_engine import __version__
from workflow_engine.contexts.local import LocalContext
from workflow_engine.core.config import WorkflowEngineConfig
from workflow_engine.core.context import ValidationContext
from workflow_engine.core.engine import WorkflowEngine
from workflow_engine.core.node import NodeRegistry
from workflow_engine.core.values import ValueRegistry
from workflow_engine.core.values.schema import validate_value_schema
from workflow_engine.core.workflow import Workflow

CONFIG_DIR = Path(platformdirs.user_config_dir("wengine", appauthor=False))
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"


def coro(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


def _load_input(value: str) -> Any:
    """Parse a CLI input argument as JSON, or @file.json, or - for stdin."""
    if value == "-":
        import sys

        source = "stdin"
        try:
            text = sys.stdin.read()
        except OSError as e:
            raise click.ClickException(f"Failed to read from stdin: {e}") from e
    elif value.startswith("@"):
        source = f"file {value[1:]!r}"
        try:
            text = Path(value[1:]).read_text()
        except OSError as e:
            raise click.ClickException(f"Failed to read {value[1:]!r}: {e}") from e
    else:
        source = "inline argument"
        text = value
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON in {source}: {e}") from e


async def _build_engine(config_path: Path | None) -> WorkflowEngine:
    """Build an engine from the given config path, or default registries."""
    if config_path is not None:
        config = WorkflowEngineConfig.load(config_path)
        return await WorkflowEngine.from_config(config)
    if DEFAULT_CONFIG_PATH.exists():
        config = WorkflowEngineConfig.load(DEFAULT_CONFIG_PATH)
        return await WorkflowEngine.from_config(config)
    return WorkflowEngine(
        node_registry=NodeRegistry.DEFAULT,
        value_registry=ValueRegistry.DEFAULT,
    )


config_option = click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a wengine config file (overrides the default location).",
)


@click.group()
@click.version_option(__version__, prog_name="wengine")
def cli():
    """Workflow Engine CLI."""


# ---------- config ----------


@cli.group()
def config():
    """Configuration utilities."""


@config.command("path")
def config_path_cmd():
    """Print the default config file location."""
    click.echo(str(DEFAULT_CONFIG_PATH))


@config.command("show")
@config_option
def config_show(config_path: Path | None):
    """Print the full contents of the config file."""
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise click.ClickException(f"Config file does not exist: {path}")
    click.echo(path.read_text(), nl=False)


# ---------- schema ----------


@cli.group()
def schema():
    """Inspect registered value types."""


@schema.command("list")
@config_option
@coro
async def schema_list(config_path: Path | None):
    """List all registered value types."""
    engine = await _build_engine(config_path)
    for name, _ in engine.value_registry.all_value_classes():
        click.echo(name)


@schema.command("check")
@click.argument("schema_arg", metavar="SCHEMA")
@config_option
@coro
async def schema_check(schema_arg: str, config_path: Path | None):
    """Validate an aliased value schema and print the fully-resolved schema.

    SCHEMA is a JSON literal, @file.json, or - for stdin. Examples:

      wengine schema check '{"x-value-type": "JSONValue"}'
      wengine schema check '{"type": "array", "items": {"x-value-type": "StringValue"}}'
    """
    # Building the engine surfaces config errors early; schema resolution
    # itself still uses ValueRegistry.DEFAULT today.
    await _build_engine(config_path)
    raw = _load_input(schema_arg)
    schema_obj = validate_value_schema(raw)
    cls = schema_obj.to_value_cls()
    click.echo(json.dumps(cls.model_json_schema(), indent=2, default=str))


@schema.command("parse")
@click.argument("schema_arg", metavar="SCHEMA")
@click.argument("value_arg", metavar="VALUE")
@config_option
@coro
async def schema_parse(schema_arg: str, value_arg: str, config_path: Path | None):
    """Parse VALUE as the given SCHEMA. Prints the validated value as JSON."""
    await _build_engine(config_path)
    raw_schema = _load_input(schema_arg)
    raw_value = _load_input(value_arg)
    schema_obj = validate_value_schema(raw_schema)
    cls = schema_obj.to_value_cls()
    instance = cls.model_validate(raw_value)
    click.echo(instance.model_dump_json(indent=2))


# ---------- node ----------


@cli.group()
def node():
    """Inspect and run nodes."""


@node.command("list")
@config_option
@coro
async def node_list(config_path: Path | None):
    """List registered node types."""
    engine = await _build_engine(config_path)
    for name, cls in engine.node_registry.items():
        info = getattr(cls, "TYPE_INFO", None)
        display = getattr(info, "display_name", name) if info else name
        click.echo(f"{name}\t{display}")


@node.command("info")
@click.argument("name")
@config_option
@coro
async def node_info(name: str, config_path: Path | None):
    """Show detailed info about a node type."""
    engine = await _build_engine(config_path)
    cls = engine.node_registry.get(name)
    if cls is None:
        raise click.ClickException(f"Unknown node type: {name}")
    info = getattr(cls, "TYPE_INFO", None)
    payload: dict[str, Any] = {"name": name}
    if info is not None:
        payload["display_name"] = getattr(info, "display_name", None)
        payload["description"] = getattr(info, "description", None)
        payload["version"] = getattr(info, "version", None)
        param_schema = getattr(info, "parameter_schema", None)
        if param_schema is not None:
            try:
                payload["parameter_schema"] = param_schema.model_dump(mode="json")
            except AttributeError:
                payload["parameter_schema"] = str(param_schema)
    click.echo(json.dumps(payload, indent=2, default=str))


@node.command("check")
@click.argument("name")
@click.argument("params_arg", metavar="PARAMS", default="{}")
@config_option
@coro
async def node_check(name: str, params_arg: str, config_path: Path | None):
    """Validate a node and print its input and output JSON schemas."""
    engine = await _build_engine(config_path)
    params = _load_input(params_arg)
    instance = engine.create_node(name, id="check", params=params)
    ctx = ValidationContext(
        node_registry=engine.node_registry,
        value_registry=engine.value_registry,
    )
    in_t = await instance.input_type(ctx)
    out_t = await instance.output_type(ctx)
    click.echo(
        json.dumps(
            {
                "ok": True,
                "input_schema": in_t.model_json_schema(),
                "output_schema": out_t.model_json_schema(),
            },
            indent=2,
            default=str,
        )
    )


@node.command("run")
@click.argument("name")
@click.argument("params_arg", metavar="PARAMS")
@click.argument("input_arg", metavar="INPUT")
@click.option(
    "--base-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("./local"),
    help="Base directory for the LocalContext run files.",
)
@config_option
@coro
async def node_run(
    name: str,
    params_arg: str,
    input_arg: str,
    base_dir: Path,
    config_path: Path | None,
):
    """Run a single node. PARAMS and INPUT are JSON, @file.json, or - for stdin."""
    engine = await _build_engine(config_path)
    params = _load_input(params_arg)
    raw_input = _load_input(input_arg)
    instance = engine.create_node(name, id=name, params=params)
    val_ctx = ValidationContext(
        node_registry=engine.node_registry,
        value_registry=engine.value_registry,
    )
    in_t = await instance.input_type(val_ctx)
    out_t = await instance.output_type(val_ctx)
    exec_ctx = LocalContext(base_dir=str(base_dir))
    exec_ctx.validation_context = val_ctx
    validated_input = in_t.model_validate(raw_input)
    output = await instance(
        context=exec_ctx,
        input_type=in_t,
        output_type=out_t,
        input={k: getattr(validated_input, k) for k in in_t.model_fields},
    )
    if hasattr(output, "model_dump_json"):
        click.echo(output.model_dump_json(indent=2))
    else:
        click.echo(
            json.dumps(
                output,
                indent=2,
                default=lambda o: o.model_dump(mode="json")
                if hasattr(o, "model_dump")
                else str(o),
            )
        )


# ---------- workflow ----------


@cli.group()
def workflow():
    """Workflow operations."""


def _load_workflow(path: Path) -> Workflow:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
        return Workflow.model_validate(data)
    return Workflow.model_validate_json(text)


@workflow.command("check")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@config_option
@coro
async def workflow_check(path: Path, config_path: Path | None):
    """Validate a workflow and print its input/output schemas."""
    engine = await _build_engine(config_path)
    wf = _load_workflow(path)
    validated = await engine.validate(wf)
    out = {
        "ok": True,
        "input_schema": validated.input_type.model_json_schema(),
        "output_schema": validated.output_type.model_json_schema(),
    }
    click.echo(json.dumps(out, indent=2, default=str))


@workflow.command("run")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("input_arg", metavar="INPUT")
@click.option(
    "--base-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("./local"),
    help="Base directory for the LocalContext run files.",
)
@config_option
@coro
async def workflow_run(
    path: Path,
    input_arg: str,
    base_dir: Path,
    config_path: Path | None,
):
    """Run a workflow. INPUT is JSON, @file.json, or - for stdin."""
    engine = await _build_engine(config_path)
    wf = _load_workflow(path)
    input_data = _load_input(input_arg)
    context = LocalContext(base_dir=str(base_dir))
    result = await engine.execute(context=context, workflow=wf, input=input_data)
    click.echo(result.model_dump_json(indent=2))


@workflow.command("describe")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@config_option
@coro
async def workflow_describe(path: Path, as_json: bool, config_path: Path | None):
    """Print a structured summary of the workflow without executing it."""
    import networkx as nx

    engine = await _build_engine(config_path)
    wf = _load_workflow(path)
    validated = await engine.validate(wf)

    nodes_summary = [
        {
            "id": n.id,
            "type": n.type,
            "params": n.params.model_dump(mode="json") if n.params else {},
        }
        for n in wf.nodes
    ]
    edges_summary = [
        {
            "from": f"{e.source_id}.{'.'.join(e.source_key_path)}",
            "to": f"{e.target_id}.{e.target_key}",
        }
        for e in wf.edges
    ]
    # Parallelizable groups: nodes at the same topological "generation" can run concurrently.
    groups = [list(layer) for layer in nx.topological_generations(wf.nx_graph)]

    summary: dict[str, Any] = {
        "nodes": nodes_summary,
        "edges": edges_summary,
        "input_schema": validated.input_type.model_json_schema(),
        "output_schema": validated.output_type.model_json_schema(),
        "execution_order": groups,
    }

    if as_json:
        click.echo(json.dumps(summary, indent=2, default=str))
        return

    # Human-readable rendering
    click.echo(f"Nodes ({len(nodes_summary)}):")
    for n in nodes_summary:
        click.echo(f"  {n['id']} : {n['type']}")
        if n["params"]:
            for k, v in n["params"].items():
                click.echo(f"      {k} = {json.dumps(v)}")
    click.echo(f"\nEdges ({len(edges_summary)}):")
    for e in edges_summary:
        click.echo(f"  {e['from']}  ->  {e['to']}")
    click.echo("\nExecution order (parallel groups):")
    for i, group in enumerate(groups):
        click.echo(f"  [{i}] {', '.join(group)}")
    click.echo("\nInput schema title:  " + summary["input_schema"].get("title", "?"))
    click.echo("Output schema title: " + summary["output_schema"].get("title", "?"))
    click.echo("\nUse --json for the full machine-readable schema output.")


@workflow.command("init")
@click.argument("path", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--force", is_flag=True, help="Overwrite an existing file at PATH.")
def workflow_init(path: Path, force: bool):
    """Create a blank workflow at PATH (no template support yet)."""
    if path.exists() and not force:
        raise click.ClickException(f"{path} already exists (use --force to overwrite).")
    blank: dict[str, Any] = {
        "input_node": {"type": "Input", "id": "input", "params": {"fields": {}}},
        "output_node": {"type": "Output", "id": "output", "params": {"fields": {}}},
        "inner_nodes": [],
        "edges": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in (".yaml", ".yml"):
        path.write_text(yaml.safe_dump(blank, sort_keys=False))
    else:
        path.write_text(json.dumps(blank, indent=2) + "\n")
    click.echo(f"Wrote blank workflow to {path}")


def main():
    cli()


if __name__ == "__main__":
    main()
