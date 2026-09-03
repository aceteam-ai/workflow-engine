# `Result[T]` wire shape

This is the published serialization contract for `Result[T]` (#200), the
tagged ok/err value type described in [discussion
#198](https://github.com/aceteam-ai/workflow-engine/discussions/198). It is
published ahead of `attempt` (#201) and the eliminators (#202) so that hosts
consuming this engine can align field names before those land.

A `schema/` directory is the eventual home for the full interchange schema
(#206). This file covers only the `Result[T]` shape; it is not that full
schema.

## Value shape

A serialized `Result[T]` instance is always an object with exactly these
three keys. `tag` says which arm is populated; the other two keys are always
present, and exactly one of them is non-null, matching `tag`.

| Key   | Type                          | Present when   |
| ----- | ----------------------------- | -------------- |
| `tag` | `"ok" \| "err"`                | always         |
| `ok`  | `T`'s serialized value, or `null` | `tag == "ok"`  |
| `err` | a `ResultError` object, or `null` | `tag == "err"` |

### `ok` example

```json
{"tag": "ok", "ok": 3.14, "err": null}
```

### `err` example

```json
{
  "tag": "err",
  "ok": null,
  "err": {
    "error_class": "timeout",
    "name": "FetchTimeout",
    "message": "The upstream service did not respond in time.",
    "node_id": "fetch-1"
  }
}
```

### Nesting: `Result[Result[T]]`

Each level keeps its own `tag`. Nothing is lost or collapsed going in or out:

```json
{
  "tag": "ok",
  "ok": {
    "tag": "err",
    "ok": null,
    "err": {
      "error_class": "timeout",
      "name": "FetchTimeout",
      "message": "The upstream service did not respond in time.",
      "node_id": "fetch-1"
    }
  },
  "err": null
}
```

## `ResultError`: the err arm

| Field         | Type     | Description                                                                                   |
| ------------- | -------- | ----------------------------------------------------------------------------------------------- |
| `error_class` | `string` | Closed vocabulary: `"timeout"`, `"unreachable"`, `"rate_limit"`, `"validation"`, `"permission"`, `"systemic"`. |
| `name`        | `string` | Short, machine-readable name of the error.                                                      |
| `message`     | `string` | User-facing description of what went wrong.                                                     |
| `node_id`     | `string` | The id of the node that produced the error (provenance).                                        |

`error_class` is a single, closed field so that a host's retry policy
(`attempt(retries=n)`), circuit breakers, and run ledger can all key off of it
instead of string-matching `message` or juggling three separate vocabularies.

## Type schema shape

`Result[T].to_value_schema()` publishes a dedicated shape rather than the
generic `properties`/`required` shape an ordinary record would produce, so
that a receiving engine reconstructs a real `Result[T]` — with its ok/err
identity intact — rather than a plain 3-field record. `x-value-type` follows
this engine's usual convention (e.g. `"Result[FloatValue]"`).

```json
{
  "type": "object",
  "x-value-type": "Result[FloatValue]",
  "ok": {
    "type": "number",
    "x-value-type": "FloatValue",
    "title": "FloatValue"
  },
  "err": {
    "type": "object",
    "title": "ResultError",
    "properties": {
      "error_class": {
        "$ref": "#/$defs/ErrorClassValue",
        "title": "Error Class",
        "description": "The machine-readable classification of the error, from a closed vocabulary: timeout, unreachable, rate_limit, validation, permission, or systemic."
      },
      "name": {
        "$ref": "#/$defs/StringValue",
        "title": "Error Name",
        "description": "The short, machine-readable name of the error."
      },
      "message": {
        "$ref": "#/$defs/StringValue",
        "title": "Message",
        "description": "The user-facing description of what went wrong."
      },
      "node_id": {
        "$ref": "#/$defs/StringValue",
        "title": "Node ID",
        "description": "The identifier of the node that produced the error."
      }
    },
    "additionalProperties": false,
    "required": ["error_class", "name", "message", "node_id"],
    "$defs": {
      "ErrorClass": {
        "type": "string",
        "enum": ["timeout", "unreachable", "rate_limit", "validation", "permission", "systemic"]
      },
      "ErrorClassValue": {"$ref": "#/$defs/ErrorClass"},
      "StringValue": {"type": "string"}
    }
  }
}
```

`ok` here is `T`'s own value schema (recursively, for nested `Result[Result[T]]`);
`err` is always this same fixed `ResultError` shape.

## What is deliberately not here

This document covers the `Result[T]` value type only. `attempt`, the
eliminator vocabulary (`partition`, `unwrap_or`, `all_ok`, `first_error`),
and the hints channel are separate, later contracts — see #199 for the
overall sequencing.

See also: [`docs/values.md`](../docs/values.md#result-values) for the
Python-facing API (`Result[T].ok(...)`, `.err(...)`, `.is_ok()`, `.is_err()`,
casting behavior, and gather-side typing).
