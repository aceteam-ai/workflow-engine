"""Publish portable graph, value-type metadata, and value-instance JSON Schemas."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic.json_schema import JsonSchemaValue, models_json_schema

import workflow_engine.nodes as builtin_nodes
from workflow_engine import (
    InputNode,
    IntegerValue,
    Node,
    OutputNode,
    Result,
    SequenceValue,
    Value,
    Workflow,
    WorkflowEngine,
)
from workflow_engine.core.edge import Edge
from workflow_engine.core.hints import Hints
from workflow_engine.core.values.schema import (
    BaseValueSchema,
    StringMapValueSchema,
    UnionValueSchema,
    ValueSchemaValue,
)
from workflow_engine.core.values.value import (
    ValueRegistry,
    _NoDocstringGenerateJsonSchema,
)

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schema"
DIALECT = "https://json-schema.org/draft/2020-12/schema"
PUBLIC_BASE = (
    "https://raw.githubusercontent.com/aceteam-ai/workflow-engine/main/schema/"
)


class InterchangeJsonSchema(_NoDocstringGenerateJsonSchema):
    """Project runtime models onto their portable serialized representation.

    This does not change runtime acceptance or the existing ValueSchema protocol.
    Portability requires resolved versions and reconstructible type metadata;
    parsing in Python is intentionally more permissive than this export profile.
    """

    def model_schema(self, schema: Any) -> JsonSchemaValue:
        result = super().model_schema(schema)
        cls = schema["cls"]
        properties = result.get("properties", {})
        if issubclass(cls, BaseValueSchema):
            # BaseValueSchema's wrap serializer renames this field on the wire.
            properties["x-value-type"] = properties.pop("value_type")
            # Definitions may include plain Pydantic models and schema-language
            # models (including Any/{}), not just reconstructible Value types.
            # Their references/meaning are checked by the receiving engine.
            properties["$defs"] = {
                "type": "object",
                "additionalProperties": {"type": "object"},
            }
            if cls is BaseValueSchema:
                # The runtime catch-all parses arbitrary metadata but cannot
                # reconstruct it. Only the explicit registry shorthand is portable.
                result["additionalProperties"] = False
                result["required"] = ["x-value-type"]
                properties["x-value-type"] = {"type": "string", "minLength": 1}
            if cls is StringMapValueSchema:
                # A malformed record or Result must not pass as an open map.
                result["not"] = {
                    "anyOf": [
                        {"required": [key]} for key in ("properties", "ok", "err")
                    ]
                }
            if cls is UnionValueSchema:
                properties["anyOf"]["minItems"] = 1
        if issubclass(cls, Node):
            properties["version"] = {
                "type": "string",
                "pattern": r"^\d+\.\d+\.\d+$",
                "description": "The resolved node version. Export must pin latest before sharing.",
            }
            result["required"] = sorted(set(result.get("required", [])) | {"version"})
            parameter_type = cls.model_fields["params"].annotation
            if isinstance(parameter_type, type) and any(
                field.is_required() for field in parameter_type.model_fields.values()
            ):
                result["required"] = sorted(set(result["required"]) | {"params"})
            if cls is Node:
                # The abstract Params model is empty/closed. Concrete node
                # parameters are dispatched below; extension nodes stay open.
                properties["params"] = {"type": "object"}
        if cls is Edge:
            properties["source_key"]["anyOf"][0]["minLength"] = 1
            properties["source_key"]["anyOf"][1]["minItems"] = 1
            properties["target_key"]["minLength"] = 1
        return result


def project_result_instances(schema: Any) -> Any:
    """Translate Result's engine metadata into actual tagged-value assertions.

    Never interpret defaults/examples/const/enum as schemas: those are data and
    may themselves contain Result-looking dictionaries or literal $ref strings.
    """
    if isinstance(schema, list):
        return [project_result_instances(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    projected = {
        key: value
        if key in {"default", "examples", "const", "enum"}
        else project_result_instances(value)
        for key, value in schema.items()
    }
    if (
        str(projected.get("x-value-type", "")).startswith("Result[")
        and "ok" in projected
        and "err" in projected
    ):
        arms = {arm: projected.pop(arm) for arm in ("ok", "err")}
        projected["oneOf"] = [
            {
                "type": "object",
                "properties": {"tag": {"const": arm, "type": "string"}, arm: payload},
                "required": ["tag", arm],
                "additionalProperties": False,
            }
            for arm, payload in arms.items()
        ]
    return projected


def referenced_definitions(schema: Any) -> set[str]:
    if isinstance(schema, list):
        return set().union(*(referenced_definitions(item) for item in schema))
    if not isinstance(schema, dict):
        return set()
    references: set[str] = set()
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if not ref.startswith("#/$defs/"):
            raise ValueError(f"Interchange schemas must use local definitions: {ref}")
        references.add(ref.removeprefix("#/$defs/"))
    for key, value in schema.items():
        if key not in {"default", "examples", "const", "enum"}:
            references.update(referenced_definitions(value))
    return references


def document(
    name: str, title: str, root: JsonSchemaValue, definitions: JsonSchemaValue
) -> JsonSchemaValue:
    """Bundle only reachable shared definitions; every document works offline."""
    included: dict[str, Any] = {}
    pending = referenced_definitions(root)
    while pending:
        key = pending.pop()
        if key in included:
            continue
        included[key] = deepcopy(definitions[key])
        pending.update(referenced_definitions(included[key]) - included.keys())
    return {
        "$schema": DIALECT,
        "$id": PUBLIC_BASE + name,
        "title": title,
        "$comment": "Generated by scripts/generate_interchange_schema.py. See schema/README.md for the portability contract and validation limits.",
        **deepcopy(root),
        **({"$defs": included} if included else {}),
    }


def public_node_classes() -> list[type[Node]]:
    """Use checked-out source exports, independent of installed entry points."""
    classes = {InputNode, OutputNode}
    for name in builtin_nodes.__all__:
        candidate = getattr(builtin_nodes, name)
        if isinstance(candidate, type) and issubclass(candidate, Node):
            classes.add(candidate)
    return sorted(classes, key=lambda cls: cls.default_type_name())


def generate_documents() -> dict[str, JsonSchemaValue]:
    nodes = public_node_classes()
    values = dict(ValueRegistry.DEFAULT.all_value_classes())
    # Tests may register helper Values; only production types belong in this
    # public catalogue. Generation in CI runs in a clean subprocess as well.
    values = {
        name: cls
        for name, cls in values.items()
        if cls.__module__.startswith("workflow_engine.")
    }
    classes = list(
        dict.fromkeys(
            [
                Workflow,
                Node,
                Hints,
                ValueSchemaValue,
                Result[Value],
                *nodes,
                *values.values(),
            ]
        )
    )
    roots, bundle = models_json_schema(
        [(cls, "serialization") for cls in classes],
        schema_generator=InterchangeJsonSchema,
    )
    definitions = project_result_instances(bundle["$defs"])
    node_root = definitions[
        roots[(Node, "serialization")]["$ref"].removeprefix("#/$defs/")
    ]
    # Check current built-in parameter contracts, while preserving the open
    # node namespace. Older/extension versions still require receiver validation.
    node_root["allOf"] = [
        {
            "if": {
                "properties": {
                    "type": {"const": cls.default_type_name()},
                    "version": {"const": cls.TYPE_INFO.version},
                },
                "required": ["type", "version"],
            },
            "then": roots[(cls, "serialization")],
        }
        for cls in nodes
    ]
    specifications = {
        "workflow.schema.json": (
            "Portable workflow graph",
            roots[(Workflow, "serialization")],
        ),
        "value-type.schema.json": (
            "Engine value-type metadata",
            roots[(ValueSchemaValue, "serialization")],
        ),
        "result.schema.json": (
            "Tagged Result value with an unconstrained ok payload",
            roots[(Result[Value], "serialization")],
        ),
        "hints.schema.json": (
            "Erasable host annotations",
            roots[(Hints, "serialization")],
        ),
        "values.schema.json": (
            "Registered Value instances (select a named definition)",
            {
                "anyOf": [
                    roots[(cls, "serialization")] for _, cls in sorted(values.items())
                ]
            },
        ),
    }
    documents = {
        filename: document(filename, title, root, definitions)
        for filename, (title, root) in specifications.items()
    }
    return documents


def generate_examples() -> dict[str, Any]:
    """Build examples from current node versions and serialized defaults."""
    engine = WorkflowEngine()

    def identity(value_type: type[Value]) -> Workflow:
        return Workflow(
            input_node=engine.create_input_node(value=value_type),
            inner_nodes=[],
            output_node=engine.create_output_node(value=value_type),
            edges=[
                Edge(
                    source_id="input",
                    source_key="value",
                    target_id="output",
                    target_key="value",
                )
            ],
        )

    each = engine.create_node(
        builtin_nodes.ForEachNode,
        id="each",
        params={"workflow": identity(IntegerValue)},
        hints=Hints(
            max_concurrency=2,
            **{"example.future_hint": {"preference": "small-batches"}},
        ),
    )
    foreach = Workflow(
        input_node=engine.create_input_node(sequence=SequenceValue[IntegerValue]),
        inner_nodes=[each],
        output_node=engine.create_output_node(sequence=SequenceValue[IntegerValue]),
        edges=[
            Edge(
                source_id="input",
                source_key="sequence",
                target_id="each",
                target_key="sequence",
            ),
            Edge(
                source_id="each",
                source_key="sequence",
                target_id="output",
                target_key="sequence",
            ),
        ],
    )
    return {
        "examples/foreach-workflow.json": foreach.model_dump(mode="json"),
        "examples/result-workflow.json": identity(Result[IntegerValue]).model_dump(
            mode="json"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when a published schema differs from the checked-out runtime.",
    )
    parser.add_argument("--output-dir", type=Path, default=SCHEMA_DIR)
    args = parser.parse_args()
    if not args.check:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    stale = []
    for name, schema in {**generate_documents(), **generate_examples()}.items():
        path = args.output_dir / name
        rendered = (
            json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
        if args.check:
            if not path.exists() or path.read_text() != rendered:
                stale.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered)
            print(f"Generated {path}")
    if stale:
        parser.exit(
            1,
            f"Stale interchange schemas: {', '.join(stale)}. Run uv run python scripts/generate_interchange_schema.py.\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
