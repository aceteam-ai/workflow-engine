---
name: wengine
description: Use the `wengine` CLI to explore, compose, validate, and run typed workflows from the aceteam-workflow-engine package. Trigger when the user asks about workflows, nodes, value schemas; when they want to scrape/transform data via composable nodes; or whenever a `wengine.yaml` / `engine.yaml` is present in the working directory.
---

# wengine — Typed Workflow CLI

`wengine` is a CLI for `aceteam-workflow-engine`. It lets you (a) execute single nodes for ad-hoc work, (b) build and validate DAGs of nodes ("workflows") with full type checking, and (c) run those workflows against typed inputs.

## Bundled script

This skill bundles a POSIX `sh` launcher that bootstraps `uv` (if needed), installs the pinned `aceteam-workflow-engine` wheel under `~/.wengine/runtime`, and runs the real CLI. You do not need a global `pip install` or `wengine` on `PATH`.

Per the [agentskills.io convention for scripts](https://agentskills.io/skill-creation/using-scripts), paths are **relative to this skill directory** (the folder that contains `SKILL.md`). The agent should run commands from that directory.

| Script                   | Role                                                                                                |
| ------------------------ | --------------------------------------------------------------------------------------------------- |
| **`scripts/wengine.sh`** | CLI entrypoint; pass the same arguments you would pass to `wengine` (subcommands, flags, `--help`). |

Example:

```sh
sh scripts/wengine.sh --help
sh scripts/wengine.sh config path
```

If the environment already has `wengine` on `PATH` from a normal install, you may use `wengine` directly instead of `sh scripts/wengine.sh` in the examples below.

## Invoking this skill

This is a playbook, not a parameterized template. The user supplies the task in plain language ("use the wengine skill to prototype a workflow that scrapes pricing data from these URLs"), and you carry it out by following the steps below — substituting the user's goal for the abstract task. You don't need to ask for arguments; pull every concrete detail you need from the user prompt and the surrounding context.

Throughout this document, **`wengine`** in command examples means the CLI itself—use `sh scripts/wengine.sh` from the skill directory when no global `wengine` is available (the common case for agents).

## The agent loop

The standard agent loop is:

1. **Explore** with `wengine node run <Name> <params> <input>` to learn what each node produces on real data, writing artifacts to disk that the next node can consume.
2. **Compose** with `wengine workflow init` + `wengine workflow edit ...` to assemble a saved workflow from steps that worked.
3. **Run** with `wengine workflow run <path> <input>` to reproduce the pipeline on demand.

## Setup

`wengine` needs a config file that lists the node types it can resolve. The config is YAML and lives at the path printed by `wengine config path` (typically `~/.config/wengine/config.yaml`). Override per-invocation with `--config <path>`.

```sh
# See the default location
wengine config path

# View the active config
wengine config show
```

A minimal config looks like:

```yaml
nodes:
  Input: workflow_engine.core.io.InputNode
  Output: workflow_engine.core.io.OutputNode
  Sum: workflow_engine.nodes.arithmetic.SumNode
  # ... add more node imports here, mapping a short alias to a fully-qualified Python class
```

If no config is found, `wengine` falls back to the in-process default registry (mostly just `Input`/`Output`), which is rarely useful — most real work needs a config.

## Value types and schemas

The engine has a typed value system. Every workflow input, every node port, and every output has a `Value` type.

- **List concrete value types**: `wengine schema list`
- **Inspect a node**: `wengine node info <name>` (params schema), `wengine node check <name> [params]` (full input/output schemas)

To reference a value type from a workflow or node param, wrap its name in `x-value-type`:

```json
{ "x-value-type": "JSONValue" }
```

For generic types, use the standard JSON Schema shape with `x-value-type` on the inner Value:

```json
{"type": "array", "items": {"x-value-type": "StringValue"}}
{"type": "object", "additionalProperties": {"x-value-type": "IntegerValue"}}
```

For object schemas with named fields, use `properties` + `required`:

```json
{
  "type": "object",
  "title": "Person",
  "required": ["name", "age"],
  "properties": {
    "name": { "x-value-type": "StringValue", "title": "Full Name" },
    "age": { "x-value-type": "IntegerValue", "title": "Age" }
  }
}
```

**Always include `required`** for any field that isn't truly optional. Most `Value` types are non-nullable; an absent value defaults to `null` and fails validation.

### Type casting between Values

The engine has a registry of casts between `Value` types. When a workflow runs, the engine **implicitly casts** between non-identical-but-compatible types at edge boundaries — you do not need an intermediate "convert" node. For example, an `IntegerValue` source can flow into a `FloatValue` input, and a `StringValue` source can flow into a `JSONValue` input (assuming the relevant casts are registered).

Two implications for agents:

- `wengine workflow edit possible-edges` **already includes castable matches**, not just exact-type matches. If a target appears in the list, the edge is wireable as-is.
- If `add-edge` rejects an edge with a "not assignable to" error, no cast path exists between those types. You need an intermediate node that bridges them, or a different source field. `possible-edges` from the _target_ side will tell you which sources are reachable.

## Reading schemas

`node check`, `workflow check`, and `workflow describe --json` emit a **compact** JSON schema by default. Sub-schemas for concrete `Value` types collapse to `{"x-value-type": "<Name>"}`, generics keep their JSON Schema shape with `x-value-type` on the inner Value, and there are no `$defs` to chase. This is the form to read.

Pass `--expanded` to any of those commands to get the full Pydantic schema (`$defs`, `$ref`, structural titles) when you need a strict JSON Schema for an external validator.

`schema check` is the one exception: it always emits the expanded form, because its purpose is to demystify a compact `x-value-type` blob into its concrete shape.

When reading either form:

- `required` lists mandatory fields; `additionalProperties: false` rejects extras.
- In the compact form, each property carries its `title`/`description` directly.
- In the expanded form, follow each property's `$ref` into `$defs` and read the def's `title` (e.g. `"SequenceValue[FloatValue]"`) for the canonical Value type. `$defs` keys are mangled identifiers — read the title, not the key.

To validate a value against a schema: `wengine schema parse <schema> <value>`.

## The explore → compose → run loop

### 1. Explore: run individual nodes

```sh
wengine node run <Name> <params-json> <input-json>
```

Inputs accept three forms anywhere `<params>` or `<input>` appears: an inline JSON literal, `@path/to/file.json`, or `-` to read from stdin. Outputs from `node run` are written to a `LocalContext` directory (default `./local/<uuid>/`); use `--base-dir` to relocate.

Example:

```sh
wengine node run Sum '{}' '{"values": [1.5, 2.5, 4.0]}'
# prints {"sum": 8.0} and writes intermediate files under ./local/<uuid>/
```

#### Discovering nodes and their shapes

Before running, three commands make a node legible:

- `wengine node list` — every registered node with its alias, display name, and description (one row per node). The first column is the alias to use everywhere else.
- `wengine node info <Name>` — full metadata for one node: display name, description, version, and the **parameter schema** (what goes in the `<params>` argument). Read this _first_ whenever you need to construct params.
- `wengine node check <Name> <params>` — validates the node with concrete params and prints its **resolved input and output schemas** (compact form by default). This is critical for nodes whose I/O shape is dynamic — e.g. variadic `Add`, where setting `num_arguments=3` causes input fields `a`, `b`, `c` to appear. Always run `node check` after settling on params and before constructing inputs.

### 2. Compose: build a saved workflow

```sh
wengine workflow init my-flow.json
```

Then assemble:

| Subcommand                                               | Purpose                                             |
| -------------------------------------------------------- | --------------------------------------------------- |
| `workflow edit add-field <path> <id>.<name> <schema>`    | Add a field to the input or output node             |
| `workflow edit update-field <path> <id>.<name> <schema>` | Change an existing field's schema                   |
| `workflow edit remove-field <path> <id>.<name>`          | Drop a field (and any edges that reference it)      |
| `workflow edit add-node <path> <Name> <id> [params]`     | Add an inner node                                   |
| `workflow edit update-node <path> <id> <params>`         | Replace a node's params (works on input/output too) |
| `workflow edit remove-node <path> <id>`                  | Remove an inner node and all touching edges         |
| `workflow edit add-edge <path> <src> <tgt>`              | Connect two handles, formatted as `nodeId.handle`   |
| `workflow edit remove-edge <path> <src> <tgt>`           | Disconnect a specific edge                          |
| `workflow edit possible-edges <path> <id>.<handle>`      | List handles compatible with the given handle       |

Each edit reloads the file, applies the change, runs full validation, and **only writes back on success** — failed edits leave the file untouched.

When `update-node` / `update-field` change a schema such that an existing edge becomes type-incompatible, the edge is **auto-dropped with a warning to stderr** (rather than failing the edit). Watch for those warnings.

`possible-edges` is the fastest way to discover what to wire next:

```sh
# After adding a Sum node named "summer", what can feed its values input?
wengine workflow edit possible-edges my-flow.json summer.values
# -> input.nums         (any compatible source on another node)
```

### 3. Inspect and run

```sh
wengine workflow describe my-flow.json            # human-readable summary
wengine workflow describe my-flow.json --json     # machine-readable
wengine workflow check my-flow.json               # validate + emit input/output schemas
wengine workflow run   my-flow.json '<input>'     # execute
```

`describe` shows nodes, edges, parallelizable execution generations, and the workflow's overall input/output schema titles. Use `--json` when programmatically inspecting; outputs include the full expanded schemas.

## Common gotchas

- **Empty registry surprise**: if `wengine node list` only shows `Input`/`Output`, you forgot a `--config` (or the default config doesn't include the nodes you need). Set the config first.
- **`required` on object schemas**: leaving fields out of `required` makes them default to `null`, which fails validation for non-nullable Values.
- **Edge handle dotted notation**: `add-edge`, `remove-edge`, `possible-edges`, and field commands all use `nodeId.handle`. Field paths can be nested at the engine level, but the CLI's `add-edge` only supports single-segment keys; for nested source paths edit JSON directly.
- **Run outputs are on disk**: `node run` and `workflow run` write to `./local/<uuid>/` by default. The `<uuid>` is regenerated each run — use `--base-dir` to anchor a stable location, or chain by feeding `@./local/<known-id>/output/...` as the next input.

## Reference

- Docs: `docs/cli.md` in the `aceteam-workflow-engine` repo.
- Engine concepts: `CLAUDE.md` in the same repo (Workflow / Node / Edge / Value definitions).
- All commands accept `--help` for full flag listings.
