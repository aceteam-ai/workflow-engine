"""Resource preflight never needs valid executable nodes or a valid graph."""

import asyncio
from copy import deepcopy
from typing import ClassVar

import pytest
from pydantic import ConfigDict, Field, ValidationError, model_validator

from workflow_engine import (
    Empty,
    IntegerValue,
    Node,
    NodeRegistry,
    NodeTypeInfo,
    Params,
    ResourceCheck,
    ResourceRequest,
    ResourceValidationOptions,
    ResourceValidationReport,
    ValidationContext,
    Value,
    Workflow,
    WorkflowEngine,
    validate_resources,
)
from workflow_engine.core.values.schema import validate_value_schema

pytestmark = pytest.mark.unit


class ResourceAgentId(Value[str]):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra={"x-resource-type": "agent"}
    )


def forbidden_factory():
    raise AssertionError("default factory was invoked")


class ResourceProbeParams(Params):
    agent: ResourceAgentId
    strict_unrelated: IntegerValue
    generated: ResourceAgentId = Field(default_factory=forbidden_factory)


class ResourceProbeNode(Node[Empty, Empty, ResourceProbeParams]):
    TYPE_INFO = NodeTypeInfo.from_parameter_type(
        display_name="Resource probe",
        version="1.0.0",
        parameter_type=ResourceProbeParams,
    )

    @model_validator(mode="after")
    def never_construct(self):
        raise AssertionError("strict node construction was invoked")

    async def run(self, **kwargs):
        raise AssertionError("node run was invoked")

    async def dynamic_input_type(self, context):
        raise AssertionError("dynamic input inference was invoked")

    async def dynamic_output_type(self, context):
        raise AssertionError("dynamic output inference was invoked")


class ResourceBareNode(Node[Empty, Empty, Empty]):
    TYPE_INFO = NodeTypeInfo.from_parameter_type(
        display_name="No resources", version="1.0.0", parameter_type=Empty
    )


class FakeResolver:
    def __init__(self, statuses=None):
        self.statuses = statuses or {}
        self.calls = []

    async def check_resources(self, requests):
        self.calls.append(tuple(requests))
        return {
            r.request_id: ResourceCheck(
                status=self.statuses.get(r.resource_id, "available")
            )
            for r in requests
        }


def registry(probe: type[Node] = ResourceProbeNode):
    builder = NodeRegistry.builder()
    builder.register(ResourceBareNode, name="Bare")
    builder.register(probe, name="Probe")
    return builder.build()


def context(resolver=None):
    return ValidationContext(node_registry=registry(), resource_resolver=resolver)


def node(params, *, id="probe", **kwargs):
    return {"id": id, "type": "Probe", "params": params, **kwargs}


def document(*nodes):
    return {
        "input_node": {"id": "input", "type": "Bare"},
        "inner_nodes": list(nodes),
        "output_node": {"id": "output", "type": "Bare"},
    }


def declare(monkeypatch, schema):
    monkeypatch.setattr(
        ResourceProbeNode,
        "TYPE_INFO",
        ResourceProbeNode.TYPE_INFO.model_update(
            parameter_schema=validate_value_schema(schema)
        ),
    )


def record(properties, required=None, **extra):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        **extra,
    }


def resource(kind="agent", /, **extra):
    return {"type": "string", "x-resource-type": kind, **extra}


@pytest.mark.parametrize(
    "value", [{}, {"agent": None}, {"agent": ""}, {"agent": "   "}, {"agent": 3}]
)
async def test_incomplete_raw_node_never_constructs_executes_or_evaluates_factories(
    value,
):
    resolver = FakeResolver()
    raw = document(node(value))
    raw["edges"] = [{"source_id": "missing", "target_id": "missing"}]
    before = deepcopy(raw)
    report = await validate_resources(raw, context(resolver))
    assert report.issues[0].code == (
        "missing_resource" if value.get("agent") is None else "invalid_resource_id"
    )
    assert report.issues[0].node_id == "probe"
    assert report.issues[0].param_path == ("agent",)
    assert report.issues[1].code == "unsupported_default"
    assert not report.complete and not report.valid
    assert resolver.calls == [] and raw == before


@pytest.mark.parametrize(
    ("status", "code", "complete", "checked"),
    [
        ("available", None, True, 1),
        ("missing", "missing_resource", True, 1),
        ("forbidden", "resource_forbidden", True, 1),
        ("unavailable", "resource_unavailable", False, 0),
        ("unsupported", "unsupported_resource_type", False, 0),
    ],
)
async def test_provider_status_contract(monkeypatch, status, code, complete, checked):
    declare(monkeypatch, record({"agent": resource()}))
    resolver = FakeResolver({"secret-id": status})
    report = await validate_resources(
        document(node({"agent": "secret-id"})), context(resolver)
    )
    assert report.complete is complete
    assert report.checked_references == checked
    assert report.valid is (status == "available")
    assert [i.code for i in report.issues] == ([] if code is None else [code])
    assert "secret-id" not in report.model_dump_json()
    assert resolver.calls[0][0].resource_id == "secret-id"
    assert report == ResourceValidationReport.model_validate_json(
        report.model_dump_json()
    )


async def test_no_resource_declarations_needs_no_provider_and_accepts_workflow_model():
    report = await validate_resources(document(), context())
    assert report.valid and report.checked_references == 0
    workflow = Workflow(
        input_node=NodeRegistry.DEFAULT.create_input_node(),
        output_node=NodeRegistry.DEFAULT.create_output_node(),
        inner_nodes=[],
        edges=[],
    )
    assert (await validate_resources(workflow, ValidationContext())).valid


async def test_no_provider_is_incomplete(monkeypatch):
    declare(monkeypatch, record({"agent": resource()}))
    report = await validate_resources(document(node({"agent": "a"})), context())
    assert not report.complete and report.issues[0].code == "resource_unavailable"


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"0": {"status": "available"}, "extra": {"status": "available"}},
        {"0": {"status": "mystery"}},
        {"0": {"status": "available", "unexpected": True}},
        None,
        [],
    ],
)
async def test_malformed_provider_response_cannot_report_success(monkeypatch, response):
    declare(monkeypatch, record({"agent": resource()}))

    class Resolver:
        async def check_resources(self, requests):
            return response

    report = await validate_resources(
        document(node({"agent": "a"})), context(Resolver())
    )
    assert not report.complete and report.checked_references == 0
    assert [i.code for i in report.issues] == ["provider_protocol_error"]


async def test_provider_exception_does_not_leak_secrets(monkeypatch):
    declare(monkeypatch, record({"agent": resource()}))

    class Resolver:
        async def check_resources(self, requests):
            raise RuntimeError("secret connection credentials")

    report = await validate_resources(
        document(node({"agent": "a"})), context(Resolver())
    )
    assert not report.complete
    assert "secret" not in report.model_dump_json()


async def test_required_optional_overrides_and_literal_defaults(monkeypatch):
    declare(
        monkeypatch,
        record(
            {
                "absent": resource(),
                "null": resource(),
                "optional": resource(),
                "default": resource(default="D"),
                "override": resource(**{"x-resource-required": True}),
                "waived": resource(**{"x-resource-required": False}),
            },
            required=["absent", "null", "waived"],
        ),
    )
    resolver = FakeResolver()
    report = await validate_resources(
        document(node({"null": None, "optional": None})), context(resolver)
    )
    assert report.complete
    assert [i.param_path for i in report.issues] == [
        ("absent",),
        ("null",),
        ("override",),
    ]
    assert [r.resource_id for batch in resolver.calls for r in batch] == ["D"]


async def test_nested_refs_arrays_maps_overrides_and_deduplication(monkeypatch):
    declare(
        monkeypatch,
        record(
            {
                "a": {"$ref": "#/$defs/Resource", "x-resource-type": "document"},
                "b": {"type": "array", "items": {"$ref": "#/$defs/Resource"}},
                "c": {"type": "object", "additionalProperties": resource()},
                "d": record({"inner": resource()}),
            },
            **{"$defs": {"Resource": resource()}},
        ),
    )
    resolver = FakeResolver({"a": "missing", "b": "forbidden"})
    report = await validate_resources(
        document(
            node(
                {
                    "a": "a",
                    "b": ["a", "b"],
                    "c": {"key.with/slash": "a"},
                    "d": {"inner": "a"},
                }
            )
        ),
        context(resolver),
    )
    assert report.complete and report.checked_references == 3
    assert [i.param_path for i in report.issues] == [
        ("a",),
        ("b", 0),
        ("b", 1),
        ("c", "key.with/slash"),
        ("d", "inner"),
    ]
    assert report.issues[0].resource_type == "document"
    assert sum(map(len, resolver.calls)) == 3


async def test_missing_required_parent_and_optional_parent(monkeypatch):
    declare(
        monkeypatch,
        record(
            {
                "needed": record({"agent": resource()}),
                "optional": record({"agent": resource()}),
            },
            required=["needed"],
        ),
    )
    report = await validate_resources(document(node({})), context())
    assert (
        report.complete
        and len(report.issues) == 1
        and report.issues[0].param_path == ("needed",)
    )


async def test_nullable_and_tagged_union(monkeypatch):
    union = {
        "oneOf": [
            record({"kind": {"const": "agent", "type": "string"}, "id": resource()}),
            record(
                {"kind": {"const": "doc", "type": "string"}, "id": resource("document")}
            ),
        ],
        "discriminator": {"propertyName": "kind"},
    }
    declare(
        monkeypatch,
        record(
            {"nullable": {"anyOf": [resource(), {"type": "null"}]}, "selected": union}
        ),
    )
    resolver = FakeResolver()
    report = await validate_resources(
        document(node({"nullable": "a", "selected": {"kind": "doc", "id": "d"}})),
        context(resolver),
    )
    assert report.valid
    assert [r.resource_type for r in resolver.calls[0]] == ["agent", "document"]


@pytest.mark.parametrize(
    "schema",
    [
        {"anyOf": [resource(), resource("document")]},
        resource(allOf=[{"x-resource-type": "document"}]),
        {"type": "array", "x-resource-type": "agent", "items": {"type": "string"}},
        {"$ref": "https://example.invalid/resource"},
        {"type": "object", "patternProperties": {".*": resource()}},
        resource(**{"x-resource-required": "yes"}),
    ],
)
async def test_unsupported_or_ambiguous_schema_has_explicit_incomplete_coverage(
    monkeypatch, schema
):
    declare(monkeypatch, record({"r": schema}))
    report = await validate_resources(
        document(node({"r": "a"})), context(FakeResolver())
    )
    assert not report.complete and not report.valid
    assert report.issues[0].code in {
        "unsupported_resource_schema",
        "ambiguous_resource_schema",
    }


async def test_constraint_only_conjunction(monkeypatch):
    declare(
        monkeypatch,
        record({"r": resource(allOf=[{"minLength": 2}, {"x-resource-type": "agent"}])}),
    )
    assert (
        await validate_resources(document(node({"r": "abc"})), context(FakeResolver()))
    ).valid


async def test_recursive_records_and_shared_definitions(monkeypatch):
    branch = record(
        {"agent": resource(), "next": {"$ref": "#/$defs/Branch"}}, required=["agent"]
    )
    declare(
        monkeypatch,
        record(
            {"a": {"$ref": "#/$defs/Branch"}, "b": {"$ref": "#/$defs/Branch"}},
            **{"$defs": {"Branch": branch}},
        ),
    )
    resolver = FakeResolver()
    report = await validate_resources(
        document(
            node({"a": {"agent": "a", "next": {"agent": "b"}}, "b": {"agent": "c"}})
        ),
        context(resolver),
    )
    assert report.valid and report.checked_references == 3


async def test_reference_chain_cycle_reports_incomplete(monkeypatch):
    declare(
        monkeypatch,
        record(
            {"r": {"$ref": "#/$defs/Cycle"}},
            **{"$defs": {"Cycle": {"$ref": "#/$defs/Cycle"}}},
        ),
    )
    report = await validate_resources(document(node({"r": "a"})), context())
    assert (
        not report.complete and report.issues[0].code == "unsupported_resource_schema"
    )


async def test_inline_workflows_have_document_paths_not_runtime_ids(monkeypatch):
    declare(
        monkeypatch,
        record(
            {"agent": resource(), "child": {"x-value-type": "WorkflowValue"}},
            required=["agent"],
        ),
    )
    resolver = FakeResolver({"child": "missing"})
    raw = document(
        node(
            {
                "agent": "parent",
                "child": document(node({"agent": "child"}, id="nested")),
            }
        )
    )
    report = await validate_resources(raw, context(resolver))
    assert report.complete and report.checked_references == 2
    assert report.issues[0].node_id == "nested"
    assert report.issues[0].node_path == (
        "inner_nodes",
        0,
        "params",
        "child",
        "inner_nodes",
        0,
    )


@pytest.mark.parametrize("version", ["0.9.0", "2.0.0", "bad", None])
async def test_unsupported_versions_are_not_migrated(monkeypatch, version):
    declare(monkeypatch, record({"agent": resource()}))
    report = await validate_resources(
        document(node({"agent": "a"}, version=version)), context()
    )
    assert not report.complete and report.issues[0].code == "unsupported_node_version"


@pytest.mark.parametrize("version", ["latest", "1.0.0"])
async def test_current_versions_are_supported(monkeypatch, version):
    declare(monkeypatch, record({"agent": resource()}))
    assert (
        await validate_resources(
            document(node({"agent": "a"}, version=version)), context(FakeResolver())
        )
    ).valid


async def test_invalid_envelopes_do_not_hide_valid_sibling_diagnostics(monkeypatch):
    declare(monkeypatch, record({"agent": resource()}))
    raw = document(
        None,
        node({}, id="duplicate"),
        node({}, id="duplicate"),
        {"params": {}},
        node([], id="bad-params"),
        node({}, id=""),
    )
    report = await validate_resources(raw, context())
    assert not report.complete
    assert sum(i.code == "missing_resource" for i in report.issues) == 3
    assert any(i.code == "duplicate_node_id" for i in report.issues)
    assert any(i.code == "unknown_node_type" for i in report.issues)
    assert any(i.node_id is None for i in report.issues)


@pytest.mark.parametrize("raw", [{}, {"inner_nodes": "bad"}, None])
async def test_malformed_workflow_is_a_report(raw):
    report = await validate_resources(raw, context())
    assert not report.complete and not report.valid
    assert report.issues[0].code in {"invalid_workflow", "invalid_node"}


async def test_engine_context_factory_and_explicit_registry_isolation(monkeypatch):
    declare(monkeypatch, record({"agent": resource()}))

    class Engine(WorkflowEngine):
        async def _get_validation_context(self):
            return context(FakeResolver())

    raw = document(node({"agent": "a"}))
    assert (await Engine().validate_resources(raw)).valid
    different = ValidationContext(node_registry=registry(ResourceBareNode))
    # The explicit registry maps Probe to a class with no resource declaration.
    report = await Engine(node_registry=registry()).validate_resources(
        raw, context=different
    )
    assert report.valid and report.checked_references == 0


async def test_bounded_concurrency_determinism_and_no_cross_call_cache(monkeypatch):
    declare(monkeypatch, record({"ids": {"type": "array", "items": resource()}}))

    class Resolver(FakeResolver):
        active = 0
        peak = 0

        async def check_resources(self, requests):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0.005 if requests[0].resource_id == "0" else 0)
                return await super().check_resources(requests)
            finally:
                self.active -= 1

    resolver = Resolver({str(i): "missing" for i in range(7)})
    raw = document(node({"ids": [str(i) for i in range(7)] + ["0"]}))
    reports = [
        await validate_resources(
            raw,
            context(resolver),
            options=ResourceValidationOptions(batch_size=1, max_concurrency=c),
        )
        for c in (1, 3)
    ]
    assert reports[0] == reports[1]
    assert [i.param_path[-1] for i in reports[0].issues] == list(range(8))
    assert resolver.peak == 3 and len(resolver.calls) == 14


async def test_timeout_and_cancellation_clean_up_provider_tasks(monkeypatch):
    declare(monkeypatch, record({"ids": {"type": "array", "items": resource()}}))
    started, stopped = asyncio.Event(), asyncio.Event()

    class Resolver:
        async def check_resources(self, requests):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    raw = document(node({"ids": ["a", "b"]}))
    report = await validate_resources(
        raw, context(Resolver()), options=ResourceValidationOptions(timeout=0.01)
    )
    assert stopped.is_set() and not report.complete
    assert all(i.code == "resource_unavailable" for i in report.issues)
    started.clear()
    stopped.clear()
    task = asyncio.create_task(validate_resources(raw, context(Resolver())))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()


@pytest.mark.parametrize(
    "limits", [{"max_references": 1}, {"max_visits": 5}, {"max_depth": 2}]
)
async def test_budget_exhaustion_is_explicit(monkeypatch, limits):
    declare(monkeypatch, record({"ids": {"type": "array", "items": resource()}}))
    report = await validate_resources(
        document(node({"ids": ["a", "b", "c"]})),
        context(FakeResolver()),
        options=ResourceValidationOptions(**limits),
    )
    assert not report.complete and not report.valid
    assert any(i.code == "validation_limit" for i in report.issues)


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 0},
        {"max_concurrency": -1},
        {"timeout": float("inf")},
        {"max_depth": True},
    ],
)
def test_invalid_options_fail_before_provider_execution(options):
    with pytest.raises(ValidationError):
        ResourceValidationOptions(**options)


def test_models_are_frozen_and_reports_have_standalone_schema():
    request = ResourceRequest(request_id="0", resource_type="agent", resource_id="a")
    with pytest.raises(ValidationError):
        request.resource_id = "b"
    assert ResourceValidationReport.model_json_schema()["$defs"][
        "ResourceValidationIssue"
    ]["properties"]["code"]["enum"]


async def test_schema_parameters_are_metadata_not_resource_values(monkeypatch):
    declare(monkeypatch, record({"schema": {"x-value-type": "ValueSchemaValue"}}))
    report = await validate_resources(document(node({"schema": resource()})), context())
    assert report.valid and report.checked_references == 0


async def test_shared_resource_free_schema_dag_does_not_expand_exponentially(
    monkeypatch,
):
    definitions = {"end": {"type": "string"}}
    previous = "end"
    for i in range(30):
        name = f"Level{i}"
        definitions[name] = record(
            {"a": {"$ref": f"#/$defs/{previous}"}, "b": {"$ref": f"#/$defs/{previous}"}}
        )
        previous = name
    declare(
        monkeypatch,
        record({"tree": {"$ref": f"#/$defs/{previous}"}}, **{"$defs": definitions}),
    )
    assert (await validate_resources(document(node({})), context())).valid


async def test_nested_nullable_resource_and_absent_optional_union(monkeypatch):
    declare(
        monkeypatch,
        record(
            {
                "nested": {
                    "anyOf": [
                        {"anyOf": [resource(), {"type": "null"}]},
                        {"type": "null"},
                    ]
                },
                "optional": {"anyOf": [resource(), resource("document")]},
            },
            required=["nested"],
        ),
    )
    report = await validate_resources(
        document(node({"nested": "a"})), context(FakeResolver())
    )
    assert report.valid and report.checked_references == 1


async def test_union_properties_named_like_keywords_preserve_ambiguity(monkeypatch):
    declare(
        monkeypatch,
        record(
            {
                "choice": {
                    "anyOf": [
                        record({"default": resource()}),
                        record({"default": resource("document")}),
                    ]
                }
            }
        ),
    )
    report = await validate_resources(
        document(node({"choice": {"default": "a"}})), context(FakeResolver())
    )
    assert not report.complete and report.issues[0].code == "ambiguous_resource_schema"


async def test_union_branch_resource_required_override_is_not_skipped(monkeypatch):
    declare(
        monkeypatch,
        record(
            {
                "agent": {
                    "anyOf": [
                        resource(**{"x-resource-required": True}),
                        {"type": "null"},
                    ]
                }
            },
            required=[],
        ),
    )
    report = await validate_resources(document(node({})), context())
    assert not report.valid
    assert [issue.code for issue in report.issues] == ["missing_resource"]


async def test_union_literal_default_selects_tagged_branch_before_traversal(
    monkeypatch,
):
    union = {
        "oneOf": [
            record({"kind": {"const": "agent", "type": "string"}, "id": resource()}),
            record(
                {"kind": {"const": "doc", "type": "string"}, "id": resource("document")}
            ),
        ],
        "discriminator": {"propertyName": "kind"},
        "default": {"kind": "agent", "id": "a"},
    }
    declare(monkeypatch, record({"selected": union}))
    resolver = FakeResolver()
    report = await validate_resources(document(node({})), context(resolver))
    assert report.valid and report.checked_references == 1
    assert resolver.calls[0][0].resource_type == "agent"


async def test_nullable_branch_literal_default_is_inspected(monkeypatch):
    declare(
        monkeypatch,
        record(
            {"agent": {"anyOf": [resource(default="gone"), {"type": "null"}]}},
            required=[],
        ),
    )
    report = await validate_resources(
        document(node({})), context(FakeResolver({"gone": "missing"}))
    )
    assert report.checked_references == 1
    assert [issue.code for issue in report.issues] == ["missing_resource"]


@pytest.mark.parametrize(
    "keyword",
    ["unevaluatedProperties", "unevaluatedItems", "propertyNames", "contentSchema"],
)
async def test_unsupported_resource_applicators_never_report_complete(
    monkeypatch, keyword
):
    declare(monkeypatch, record({"r": {"type": "object", keyword: resource()}}))
    report = await validate_resources(
        document(node({"r": {"a": "id"}})), context(FakeResolver())
    )
    assert not report.complete and not report.valid
    assert report.issues[0].code == "unsupported_resource_schema"


async def test_map_key_inspection_honors_total_visit_budget(monkeypatch):
    from collections.abc import Mapping

    class LargeMap(Mapping):
        inspected = 0

        def __len__(self):
            return 1_000

        def __getitem__(self, key):
            return "id"

        def __iter__(self):
            for i in range(len(self)):
                self.inspected += 1
                yield f"key-{i}"

    values = LargeMap()
    declare(
        monkeypatch,
        record({"ids": {"type": "object", "additionalProperties": resource()}}),
    )
    report = await validate_resources(
        document(node({"ids": values})),
        context(FakeResolver()),
        options=ResourceValidationOptions(max_visits=8),
    )
    assert not report.complete
    assert values.inspected <= 8


async def test_lazy_malformed_provider_mapping_does_not_escape_or_leak(monkeypatch):
    from collections.abc import Mapping

    class BrokenResponse(Mapping):
        def __len__(self):
            return 1

        def __getitem__(self, key):
            raise RuntimeError("secret connection credentials")

        def __iter__(self):
            return iter(["0"])

    class Resolver:
        async def check_resources(self, requests):
            return BrokenResponse()

    declare(monkeypatch, record({"agent": resource()}))
    report = await validate_resources(
        document(node({"agent": "a"})), context(Resolver())
    )
    assert not report.complete
    assert report.issues[0].code == "provider_protocol_error"
    assert "secret" not in report.model_dump_json()


@pytest.mark.parametrize("identity", ["ValueSchemaValue", "WorkflowValue"])
async def test_metadata_identity_does_not_hide_explicit_invalid_resource_declaration(
    monkeypatch, identity
):
    declare(
        monkeypatch,
        record({"r": {"x-value-type": identity, "x-resource-type": "agent"}}),
    )
    report = await validate_resources(
        document(node({"r": document()})), context(FakeResolver())
    )
    assert not report.complete and not report.valid
    assert report.issues[0].code == "unsupported_resource_schema"


async def test_optional_shared_union_inspection_does_not_expand_exponentially(
    monkeypatch,
):
    from workflow_engine.core import resources

    definitions = {"end": resource()}
    previous = "end"
    for i in range(30):
        name = f"Level{i}"
        definitions[name] = {
            "anyOf": [{"$ref": f"#/$defs/{previous}"}, {"$ref": f"#/$defs/{previous}"}]
        }
        previous = name
    declare(
        monkeypatch,
        record(
            {"optional": {"$ref": f"#/$defs/{previous}"}},
            required=[],
            **{"$defs": definitions},
        ),
    )
    resolve = resources._resolve
    calls = 0

    def bounded_resolve(schema, scopes):
        nonlocal calls
        calls += 1
        assert calls < 200, "Shared branch resolution exceeded linear inspection"
        return resolve(schema, scopes)

    monkeypatch.setattr(resources, "_resolve", bounded_resolve)
    assert (await validate_resources(document(node({})), context())).valid
