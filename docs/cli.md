# Workflow Engine `wengine` CLI

## `wengine config`

Config is stored as a YAML file at a default location determined by `platformdirs`. Users edit it directly — the CLI does not provide per-key edit operations.

### `wengine config path`

Prints the default config file location, so users know where to edit it.

### `wengine config show`

Prints the full contents of the config file (default location, or the file given via `--config`).

All commands that build an engine (everything under `schema`, `node`, and the read/run subcommands of `workflow`) accept an optional `--config=<path>` flag to override the default config location. `workflow init` does not — it only writes a file.

## `wengine schema`

### `wengine schema list`

Lists all registered value types.

Only **concrete** value types are listed — generic base types like `SequenceValue` and `StringMapValue` are intentionally excluded because they aren't directly instantiable on their own.

To reference a concrete type from a workflow or node parameter, wrap its name in `x-value-type`:

```json
{"x-value-type": "JSONValue"}
```

For generic types, compose them with the standard JSON Schema shape and put `x-value-type` on the inner Value:

```json
{"type": "array", "items": {"x-value-type": "StringValue"}}
{"type": "object", "additionalProperties": {"x-value-type": "IntegerValue"}}
```

#### Composite object schemas

Objects with heterogeneous fields use the standard `properties` form. Each property is itself a schema (either an `x-value-type` reference or a generic composition). The engine synthesizes a `DataValue[<Title>]` class from the schema:

```json
{
  "type": "object",
  "title": "Person",
  "description": "A user record.",
  "required": ["name", "age", "tags", "meta"],
  "properties": {
    "name": {"x-value-type": "StringValue", "title": "Full Name", "description": "The full legal name."},
    "age":  {"x-value-type": "IntegerValue", "title": "Age", "description": "Age in years."},
    "tags": {"type": "array", "items": {"x-value-type": "StringValue"}, "title": "Tags", "description": "Free-form tags."},
    "meta": {"x-value-type": "JSONValue", "title": "Metadata", "description": "Arbitrary metadata."}
  }
}
```

#### `title` and `description`

Both are surfaced to end users (and to agents constructing inputs), so write them for a non-technical audience.

- **At the top level of an object schema**, `title` becomes the synthesized class name (`DataValue[<title>]`) — keep it a valid identifier (e.g. `"Person"`, not `"A Person"`).
- **On a property**, `title` is a short human label (title case, e.g. `"Full Name"`) and `description` is a noun phrase starting with "The" (e.g. `"The full legal name."`).
- On generic composites (`array`, `object`), `title`/`description` apply to the *outer* shape; the inner item schema gets its own `title`/`description`.

#### `required` is strongly recommended

JSON Schema treats absent properties as optional, but most `Value` types (e.g. `StringValue`, `IntegerValue`) are non-nullable — an absent value defaults to `null` and will fail validation. **Always list every property that isn't truly optional in `required`.** Omitting `required` for a non-nullable property will produce a class that can never validate successfully.

Use `wengine schema check <schema>` to expand any of these into a fully-resolved schema, and `wengine schema parse <schema> <value>` to check whether a given JSON value matches.

#### Reading expanded schemas

Commands that emit schemas (`schema check`, `node check`, `workflow check`) return **fully-expanded JSON Schemas** rather than the compact `x-value-type` form. They follow standard JSON Schema with a few conventions worth knowing:

- **Top-level `title`** is the synthesized class name (e.g. `SumNodeInput`, `DataValue[Person]`). It is descriptive, not addressable — don't rely on it for type identity.
- **Properties use `$ref`** into the schema's `$defs` block. To read a property's type, look up its `$ref` target in `$defs`.
- **Inside `$defs`**, each entry's `title` is the canonical Value type name. Concrete types appear as their bare name (e.g. `"StringValue"`); generic types appear as `"<Generic>[<Arg>]"` (e.g. `"SequenceValue[FloatValue]"`).
- **`$defs` keys** are mangled identifiers (e.g. `SequenceValue_FloatValue_`) — those are internal addresses, not user-facing names. Always read the def's `title`, not its key.
- **`required`** on each object lists fields that must be supplied. Anything not in `required` is optional with default `null`.
- **`additionalProperties: false`** is set on synthesized object schemas — extra fields will be rejected.

Example: `wengine node check Sum` returns an `input_schema` whose `values` property `$ref`s `#/$defs/SequenceValue_FloatValue_`. Looking up that def yields `{"title": "SequenceValue[FloatValue]", "type": "array", "items": {"$ref": "#/$defs/FloatValue"}}` — i.e. an array of `FloatValue`. To construct an input for this node, use the compact form `{"values": [1.0, 2.5, 3.0]}`; the engine will validate it against the expanded schema.

### `wengine schema check <schema>`

Uses the engine's validation powers to validate an aliased JSON schema type (e.g. `{"x-value-type": "JSONValue"}`, `{"type": "array", "items": {"x-value-type": "StringValue"}}`) and return a full schema.

### `wengine schema parse <schema> <value>`

Checks whether the given JSON string value can be deserialized as something matching the given schema.

## `wengine node`

### `wengine node list`

Lists all available node types according to the config.

### `wengine node info <name>`

Provides detailed information about the single node type.

### `wengine node check <name> <params>`

Validate a single node. On success, prints the node's resolved input and output schemas (so callers know what `node run` will accept and produce).

### `wengine node run <name> <params> <input>`

Run a single node.

## `wengine workflow`

### `wengine workflow init <path>`

Create a blank workflow and store it at `path`. Pass `--force` to overwrite an existing file. (Template support is planned but not yet implemented.)

### `wengine workflow edit <path> <subcommand>`

Apply an edit to the workflow stored at `path`. Each edit reloads the file, applies the change, runs full workflow validation, and saves the result back to `path` only if validation succeeds. On failure the file is left untouched.

#### `wengine workflow edit <path> add-node <name> <id> [params]`

Append a new node of type `<name>` with id `<id>` to `inner_nodes`. `params` is a JSON literal, `@file.json`, or `-` for stdin (defaults to `{}`).

#### `wengine workflow edit <path> remove-node <id>`

Remove an inner node and any edges that touch it. Refuses to remove the workflow's input or output node.

#### `wengine workflow edit <path> add-edge <source> <target>`

Add an edge `source -> target`, where each side is formatted as `nodeId.handle`. Type compatibility is enforced by the validation step.

#### `wengine workflow edit <path> remove-edge <source> <target>`

Remove the edge `source -> target`. Errors if no matching edge exists.

#### `wengine workflow edit <path> possible-edges <nodeId>.<handle>`

For the given handle (an output or input on the named node), list all compatible counterparts on other nodes. Uses `Value.can_cast_to` to determine compatibility. Already-wired inputs are excluded (each input takes one source). Doesn't modify the workflow.

### `wengine workflow check <path>`

Loads the workflow at `path`, validates it, and prints its overall input and output schemas. (A future iteration will accept an inline JSON payload that is validated and saved back to `path` if successful.)

### `wengine workflow describe <path>`

Print a structured summary of the workflow stored at `path` without executing it. Intended as the primary read-only inspection command for both humans and agents — avoids re-parsing the workflow JSON to answer common questions.

Current output includes:

- **Nodes**: for each node, its `id`, type name, and parameter values.
- **Edges**: list of `sourceNodeId.handle -> targetNodeId.handle` connections.
- **Inferred I/O**: the workflow's overall input and output schemas.
- **Execution order**: topological generations of node IDs (each generation can run in parallel).

Future work (not yet emitted):

- **Metadata**: workflow name, version, and top-level annotations.
- **Warnings**: dangling handles, unreachable nodes, or nodes whose params reference inputs that aren't provided by the workflow input schema. Non-fatal — `check` is the command that fails on these.
- Resolved-parameter views that distinguish references to upstream outputs from literals.

Supports `--json` for machine-readable output (stable shape, suitable for agent consumption) and a default human-readable rendering.

### `wengine workflow run <path> <input>`

Run a workflow by loading it from `path`.
