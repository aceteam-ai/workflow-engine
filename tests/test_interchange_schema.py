"""Portable documents validate real exports and reject malformed wire shapes."""

import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest
from jsonschema import Draft202012Validator

from workflow_engine import (
    FloatValue,
    IntegerValue,
    NullValue,
    Result,
    StringValue,
    Workflow,
    WorkflowEngine,
    WorkflowExecutionResultStatus,
)
from workflow_engine.contexts import InMemoryExecutionContext
from workflow_engine.core.values.value import ValueRegistry

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schema"


def load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate_interchange_schema",
        ROOT / "scripts" / "generate_interchange_schema.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load_generator()
pytestmark = pytest.mark.unit


def schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text())


def example(name: str) -> dict:
    return json.loads((SCHEMA_DIR / "examples" / f"{name}-workflow.json").read_text())


@pytest.mark.parametrize(
    "name", ["workflow", "value-type", "result", "hints", "values"]
)
def test_published_documents_are_valid_and_every_reference_is_local(name):
    document = schema(name)
    Draft202012Validator.check_schema(document)
    assert (
        generator.referenced_definitions(document) <= document.get("$defs", {}).keys()
    )


def test_generated_documents_are_current_and_stale_files_fail(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_interchange_schema.py"),
            "--check",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for path in SCHEMA_DIR.glob("*.schema.json"):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    stale = tmp_path / "workflow.schema.json"
    stale.write_text(stale.read_text().replace('"minItems": 1', '"minItems": 0'))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_interchange_schema.py"),
            "--check",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Stale interchange schemas: workflow.schema.json" in result.stderr


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("result", {"value": {"tag": "ok", "ok": 7}}),
        (
            "result",
            {
                "value": {
                    "tag": "err",
                    "err": {
                        "error_class": "timeout",
                        "name": "FutureError",
                        "message": "Try later.",
                        "node_id": "upstream",
                    },
                }
            },
        ),
        ("foreach", {"sequence": [1, 2, 3]}),
    ],
)
async def test_examples_validate_round_trip_and_execute(name, data, algorithm):
    wire = example(name)
    Draft202012Validator(schema("workflow")).validate(wire)
    workflow = Workflow.model_validate(wire)
    engine = WorkflowEngine(execution_algorithm=algorithm)
    validated = await engine.validate(workflow)
    assert validated.model_dump(mode="json") == wire
    result = await engine.execute(
        context=InMemoryExecutionContext(), workflow=workflow, input=data
    )
    assert result.status is WorkflowExecutionResultStatus.SUCCESS
    assert result.output is not None
    assert {
        key: value.model_dump(mode="json") for key, value in result.output.items()
    } == data


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-edges",
        "empty-source",
        "missing-version",
        "latest-version",
        "bad-hint",
        "stored-workflow-id",
        "missing-params",
    ],
)
def test_malformed_workflow_exports_are_rejected(mutation):
    wire = example("foreach")
    if mutation == "missing-edges":
        del wire["edges"]
    elif mutation == "empty-source":
        wire["edges"][0]["source_key"] = []
    elif mutation == "missing-version":
        del wire["inner_nodes"][0]["version"]
    elif mutation == "latest-version":
        wire["inner_nodes"][0]["version"] = "latest"
    elif mutation == "missing-params":
        del wire["inner_nodes"][0]["params"]
    elif mutation == "bad-hint":
        wire["inner_nodes"][0]["hints"]["max_concurrency"] = 0
    else:
        wire["inner_nodes"][0]["params"]["workflow"] = "host:stored-version-42"
    assert not Draft202012Validator(schema("workflow")).is_valid(wire)


async def test_extension_node_namespace_is_open_but_does_not_prove_availability():
    wire = example("foreach")
    wire["inner_nodes"][0]["type"] = "org.example.CustomEach"
    wire["inner_nodes"][0]["params"] = {"custom": 3}
    Draft202012Validator(schema("workflow")).validate(wire)
    with pytest.raises(ValueError, match="not registered"):
        await WorkflowEngine().validate(Workflow.model_validate(wire))


@pytest.mark.parametrize(
    "value",
    [
        {"tag": "ok", "ok": None},
        {"tag": "ok", "ok": {"tag": "ok", "ok": 1}},
        {
            "tag": "err",
            "err": {
                "error_class": "systemic",
                "name": "UnknownName",
                "message": "Failed.",
                "node_id": "root",
            },
        },
    ],
)
def test_result_instance_schema_accepts_null_nested_and_open_error_names(value):
    Draft202012Validator(schema("result")).validate(value)


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"tag": "ok"},
        {"tag": "err"},
        {"tag": "unknown", "ok": 1},
        {"tag": "ok", "ok": 1, "err": {}},
        {
            "tag": "err",
            "err": {
                "error_class": "unknown",
                "name": "X",
                "message": "x",
                "node_id": "n",
            },
        },
    ],
)
def test_result_instance_schema_rejects_bad_tags_payloads_and_classes(value):
    assert not Draft202012Validator(schema("result")).is_valid(value)


@pytest.mark.parametrize(
    "value",
    [
        Result[IntegerValue].ok(IntegerValue(7)),
        Result[NullValue].ok(NullValue(None)),
        Result[Result[IntegerValue]].ok(Result[IntegerValue].ok(IntegerValue(7))),
    ],
)
def test_specialized_result_projection_validates_the_serialized_payload(value):
    document = generator.project_result_instances(type(value).model_json_schema())
    Draft202012Validator.check_schema(document)
    validator = Draft202012Validator(document)
    validator.validate(value.model_dump(mode="json"))
    assert not validator.is_valid({"tag": "ok", "ok": "wrong type"})
    assert not validator.is_valid({"tag": "ok", "ok": {"tag": "unknown", "ok": 7}})


def test_value_type_metadata_accepts_emitted_types_and_preserves_wire_alias():
    validator = Draft202012Validator(schema("value-type"))
    for _, cls in ValueRegistry.DEFAULT.all_value_classes():
        if cls.__module__.startswith("workflow_engine."):
            validator.validate(cls.to_value_schema().model_dump(mode="json"))
    validator.validate(Result[IntegerValue].to_value_schema().model_dump(mode="json"))
    validator.validate({"type": "string", "title": "IntegerValue"})
    validator.validate({"x-value-type": "IntegerValue"})
    assert not validator.is_valid({"value_type": "IntegerValue"})


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"anyOf": []},
        {
            "type": "object",
            "ok": {"type": "integer"},
            "x-value-type": "Result[IntegerValue]",
        },
    ],
)
def test_incomplete_type_metadata_is_not_treated_as_an_open_value(value):
    assert not Draft202012Validator(schema("value-type")).is_valid(value)


@pytest.mark.parametrize(
    "value", [IntegerValue(7), FloatValue(1.25), StringValue("hi"), NullValue(None)]
)
def test_named_value_definitions_validate_actual_instances(value):
    document = schema("values")
    document["$ref"] = f"#/$defs/{type(value).__name__}"
    Draft202012Validator(document).validate(value.model_dump(mode="json"))


def test_parameter_schemas_share_the_recursive_workflow_definition():
    definitions = schema("workflow")["$defs"]
    assert (
        definitions["ForEachParams"]["properties"]["workflow"]["$ref"]
        == "#/$defs/WorkflowValue"
    )
    assert definitions["WorkflowValue"]["$ref"] == "#/$defs/Workflow"
    assert "$defs" not in definitions["ForEachParams"]


def test_result_projection_does_not_rewrite_example_or_default_values():
    value = {
        "x-value-type": "Result[IntegerValue]",
        "ok": {"type": "integer"},
        "err": {},
        "$ref": "literal data",
    }
    document = {"default": deepcopy(value), "examples": [deepcopy(value)]}
    assert generator.project_result_instances(document) == document
    assert generator.referenced_definitions(document) == set()
