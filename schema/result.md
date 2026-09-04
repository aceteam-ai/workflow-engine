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

A serialized `Result[T]` instance is a tagged object: `tag` says which arm is
populated, and exactly the matching payload key is present. This is a
discriminated union of two shapes, not a single object with two optional
sibling fields that default to `null`. That distinction is load-bearing: a
sibling-fields shape would use `null` as the "this arm is absent" sentinel,
which is indistinguishable from a populated `ok` payload that is itself
`null` (e.g. `T = NullValue`). Routing on `tag` first means the payload is
only ever validated against its own type, never against a sentinel.

| Shape | Keys present    |
| ----- | ---------------- |
| ok    | `tag: "ok"`, `ok: <T's serialized value>` |
| err   | `tag: "err"`, `err: <ResultError object>` |

### `ok` example

```json
{"tag": "ok", "ok": 3.14}
```

### `err` example

```json
{
  "tag": "err",
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
    "err": {
      "error_class": "timeout",
      "name": "FetchTimeout",
      "message": "The upstream service did not respond in time.",
      "node_id": "fetch-1"
    }
  }
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
that a receiving engine reconstructs a real `Result[T]`, with its ok/err
identity intact, rather than a plain 3-field record. `x-value-type` follows
this engine's usual convention (e.g. `"Result[FloatValue]"`).

This is the exact, unabridged output of `Result[FloatValue].to_value_schema().model_dump(by_alias=True)`:

```json
{
  "type": "object",
  "ok": {
    "title": "FloatValue",
    "type": "number",
    "x-value-type": "FloatValue"
  },
  "err": {
    "$defs": {
      "ErrorClass": {
        "title": "ErrorClass",
        "description": "The closed-vocabulary, machine-readable classification of a ``Result`` err arm. A single field lets callers key retry policy, circuit breakers, and a run ledger off of one vocabulary instead of three.",
        "type": "string",
        "enum": ["timeout", "unreachable", "rate_limit", "validation", "permission", "systemic"]
      },
      "ErrorClassValue": {
        "title": "ErrorClassValue",
        "$ref": "#/$defs/ErrorClass"
      },
      "StringValue": {
        "title": "StringValue",
        "type": "string"
      }
    },
    "title": "ResultError",
    "description": "The structured error carried by a ``Result[T]`` err arm.",
    "type": "object",
    "properties": {
      "error_class": {
        "title": "Error Class",
        "description": "The machine-readable classification of the error, from a closed vocabulary: timeout, unreachable, rate_limit, validation, permission, or systemic.",
        "$ref": "#/$defs/ErrorClassValue"
      },
      "name": {
        "title": "Error Name",
        "description": "The short, machine-readable name of the error.",
        "$ref": "#/$defs/StringValue"
      },
      "message": {
        "title": "Message",
        "description": "The user-facing description of what went wrong.",
        "$ref": "#/$defs/StringValue"
      },
      "node_id": {
        "title": "Node ID",
        "description": "The identifier of the node that produced the error.",
        "$ref": "#/$defs/StringValue"
      }
    },
    "additionalProperties": false,
    "required": ["error_class", "name", "message", "node_id"]
  },
  "x-value-type": "Result[FloatValue]"
}
```

(The `ErrorClass` `description` above has embedded newlines collapsed to spaces for
readability in this document; the real output preserves them literally.)

`ok` here is `T`'s own value schema (recursively, for nested `Result[Result[T]]`);
`err` is always this same fixed `ResultError` shape.

## What is deliberately not here

This document covers the `Result[T]` value type only. `attempt`, the
eliminator vocabulary (`partition`, `unwrap_or`, `all_ok`, `first_error`),
and the hints channel are separate, later contracts. See #199 for the
overall sequencing.

See also: [`docs/values.md`](../docs/values.md#result-values) for the
Python-facing API (`Result[T].ok(...)`, `.err(...)`, `.is_ok()`, `.is_err()`,
casting behavior, and gather-side typing).
