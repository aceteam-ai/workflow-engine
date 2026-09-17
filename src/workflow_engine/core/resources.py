"""Optional, read-only resource preflight for incomplete workflow documents."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, get_args

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from ..utils.model import ImmutableBaseModel
from .values import Value

if TYPE_CHECKING:
    from .context import ValidationContext
    from .workflow import Workflow

ResourceStatus = Literal[
    "available", "missing", "forbidden", "unavailable", "unsupported"
]
ResourceIssueCode = Literal[
    "missing_resource",
    "invalid_resource_id",
    "resource_forbidden",
    "resource_unavailable",
    "unsupported_resource_type",
    "provider_protocol_error",
    "invalid_workflow",
    "invalid_node",
    "duplicate_node_id",
    "unknown_node_type",
    "unsupported_node_version",
    "unsupported_resource_schema",
    "ambiguous_resource_schema",
    "unsupported_default",
    "validation_limit",
]
Path = tuple[StrictStr | Annotated[StrictInt, Field(ge=0)], ...]


class ResourceRequest(ImmutableBaseModel):
    request_id: str
    resource_type: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)


class ResourceCheck(ImmutableBaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: ResourceStatus
    message: str | None = None


class ResourceResolver(Protocol):
    async def check_resources(
        self, requests: Sequence[ResourceRequest]
    ) -> Mapping[str, ResourceCheck]: ...


class ResourceValidationIssue(ImmutableBaseModel):
    node_id: str | None
    node_path: Path
    param_path: Path = ()
    resource_type: str | None = None
    code: ResourceIssueCode
    message: str


class ResourceValidationReport(ImmutableBaseModel):
    issues: tuple[ResourceValidationIssue, ...] = ()
    checked_references: int = Field(default=0, ge=0)
    complete: bool = True

    @property
    def valid(self) -> bool:
        """Resource preflight succeeded; this does not assert graph validity."""
        return self.complete and not self.issues


class ResourceValidationOptions(ImmutableBaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    batch_size: int = Field(default=100, gt=0)
    max_concurrency: int = Field(default=4, gt=0)
    timeout: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    max_references: int = Field(default=10_000, gt=0)
    max_depth: int = Field(default=64, gt=0)
    # Also bound repeated IDs, resource-free documents, and diagnostic storage.
    max_visits: int = Field(default=100_000, gt=0)


_MISSING = object()
_SCHEMA_CHILDREN = (
    "properties",
    "items",
    "additionalProperties",
    "anyOf",
    "oneOf",
    "allOf",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
    "contains",
    "patternProperties",
    "dependentSchemas",
    "unevaluatedProperties",
    "unevaluatedItems",
    "propertyNames",
    "contentSchema",
)
_CONSTRAINTS = {
    "title",
    "description",
    "default",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "enum",
    "const",
    "$comment",
}
_UNSUPPORTED = {
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
    "contains",
    "patternProperties",
    "dependentSchemas",
    "unevaluatedProperties",
    "unevaluatedItems",
    "propertyNames",
    "contentSchema",
}


@dataclass(frozen=True)
class _Location:
    node_id: str | None
    node_path: tuple[str | int, ...]
    param_path: tuple[str | int, ...] = ()

    def child(self, key: str | int) -> _Location:
        return _Location(self.node_id, self.node_path, (*self.param_path, key))


class _SchemaError(ValueError):
    def __init__(self, code: ResourceIssueCode, message: str):
        self.code: ResourceIssueCode = code
        self.message = message


def _resolve(schema: Mapping[str, Any], scopes: tuple[Mapping[str, Any], ...]):
    """Resolve local reference chains only; leave recursive child shapes intact."""
    seen: set[int] = set()
    siblings: dict[str, Any] = {}
    while True:
        if id(schema) in seen:
            raise _SchemaError(
                "unsupported_resource_schema", "Cyclic resource schema reference."
            )
        seen.add(id(schema))
        if "$defs" in schema:
            if not isinstance(schema["$defs"], Mapping):
                raise _SchemaError(
                    "unsupported_resource_schema",
                    "Invalid resource schema definitions.",
                )
            scopes = (schema["$defs"], *scopes)
        if "$ref" not in schema:
            return {**schema, **siblings}, scopes
        ref = schema["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
            raise _SchemaError(
                "unsupported_resource_schema",
                "Only local resource schema references are supported.",
            )
        name = ref.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
        target = next((scope[name] for scope in scopes if name in scope), None)
        if not isinstance(target, Mapping):
            raise _SchemaError(
                "unsupported_resource_schema",
                "Resource schema definition is unavailable.",
            )
        siblings = {
            **{k: v for k, v in schema.items() if k not in {"$ref", "$defs"}},
            **siblings,
        }
        schema = target


def _relevant(
    schema: Any,
    scopes: tuple[Mapping[str, Any], ...],
    seen: set[int] | None = None,
    depth: int = 0,
) -> bool:
    """Find declarations without following unrelated metadata or unused $defs."""
    if seen is None:
        seen = set()
    if not isinstance(schema, Mapping) or id(schema) in seen:
        return False
    if depth > 64 or len(seen) >= 10_000:
        raise _SchemaError(
            "validation_limit", "Resource schema inspection limit exceeded."
        )
    seen.add(id(schema))
    if "x-resource-type" in schema:
        return True
    if schema.get("x-value-type") == "ValueSchemaValue":
        return False  # A schema parameter describes data; it is not that data.
    if schema.get("x-value-type") == "WorkflowValue":
        return True
    try:
        resolved, scopes = _resolve(schema, scopes)
    except _SchemaError:
        return True
    if "x-resource-type" in resolved:
        return True
    if resolved.get("x-value-type") == "ValueSchemaValue":
        return False
    if resolved.get("x-value-type") == "WorkflowValue":
        return True
    for key in _SCHEMA_CHILDREN:
        child = resolved.get(key)
        if key in {
            "properties",
            "patternProperties",
            "dependentSchemas",
        } and isinstance(child, Mapping):
            children = child.values()
        elif isinstance(child, list):
            children = child
        else:
            children = (child,)
        if any(_relevant(item, scopes, seen, depth + 1) for item in children):
            return True
    return False


def _omission_needs_inspection(
    schema: Mapping[str, Any],
    scopes: tuple[Mapping[str, Any], ...],
    override: Any = None,
    depth: int = 0,
    seen: set[tuple[int, bool]] | None = None,
) -> bool:
    """Look through root composition before omitting an optional parameter.

    Child record fields do not make an absent optional parent required. Root
    branch defaults and resource-required overrides, however, affect this very
    parameter and must be resolved before deciding to omit it.
    """
    if seen is None:
        seen = set()
    key = (id(schema), override is False)
    if key in seen:
        return False
    if depth > 64 or len(seen) >= 10_000:
        raise _SchemaError("validation_limit", "Resource union nesting limit exceeded.")
    seen.add(key)
    schema, scopes = _resolve(schema, scopes)
    if "default" in schema:
        return True
    if override is None:
        override = schema.get("x-resource-required")
    if override is not None and override is not False:
        return True
    for keyword in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(keyword, [])
        if not isinstance(branches, list):
            return True  # Normalization reports the malformed declaration.
        for branch in branches:
            if not isinstance(branch, Mapping) or _omission_needs_inspection(
                branch, scopes, override, depth + 1, seen
            ):
                return True
    return False


def _branch_shape(value: Any) -> Any:
    """Ignore scalar restrictions, preserving property names and defaults."""
    if isinstance(value, Mapping):
        shape = {}
        for key, item in value.items():
            if key in _CONSTRAINTS and key != "default":
                continue
            if key in {"properties", "$defs"} and isinstance(item, Mapping):
                shape[key] = {
                    name: _branch_shape(child) for name, child in item.items()
                }
            else:
                shape[key] = _branch_shape(item)
        return shape
    if isinstance(value, list):
        return [_branch_shape(v) for v in value]
    return value


def _root_annotation(annotation: Any) -> Any:
    seen = set()
    while isinstance(annotation, type) and issubclass(annotation, Value):
        if annotation in seen:
            return None
        seen.add(annotation)
        annotation = annotation.model_fields["root"].annotation
    return annotation


def _model_fields(annotation: Any) -> Mapping[str, Any]:
    annotation = _root_annotation(annotation)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.model_fields
    return {}


class _Preflight:
    def __init__(self, context: ValidationContext, options: ResourceValidationOptions):
        self.context, self.options = context, options
        self.issues: list[tuple[int, ResourceValidationIssue]] = []
        self.requests: dict[tuple[str, str], ResourceRequest] = {}
        self.locations: dict[str, list[tuple[int, _Location]]] = {}
        self.complete = True
        self.checked = 0
        self.visits = 0
        self.stopped = False

    def issue(
        self,
        location: _Location,
        code: ResourceIssueCode,
        message: str,
        *,
        resource_type: str | None = None,
        incomplete: bool = True,
        order: int | None = None,
    ):
        self.complete &= not incomplete
        self.issues.append(
            (
                self.visits if order is None else order,
                ResourceValidationIssue(
                    node_id=location.node_id,
                    node_path=location.node_path,
                    param_path=location.param_path,
                    resource_type=resource_type,
                    code=code,
                    message=message,
                ),
            )
        )

    def visit(self, location: _Location, depth: int) -> bool:
        if self.stopped:
            return False
        self.visits += 1
        if self.visits > self.options.max_visits:
            self.issue(
                location, "validation_limit", "Resource traversal visit limit exceeded."
            )
            self.stopped = True
            return False
        if depth > self.options.max_depth:
            self.issue(
                location, "validation_limit", "Resource traversal depth limit exceeded."
            )
            return False
        return True

    def workflow(self, document: Any, path: tuple[str | int, ...] = (), depth: int = 0):
        location = _Location(None, path)
        if not self.visit(location, depth):
            return
        if not isinstance(document, Mapping):
            self.issue(location, "invalid_workflow", "Expected a workflow document.")
            return
        seen: set[str] = set()
        for key in ("input_node", "inner_nodes", "output_node"):
            value = document.get(key, _MISSING)
            if key == "inner_nodes":
                if not isinstance(value, (list, tuple)):
                    self.issue(
                        _Location(None, (*path, key)),
                        "invalid_workflow",
                        "Expected a node list.",
                    )
                    continue
                entries = ((node, (*path, key, i)) for i, node in enumerate(value))
            else:
                entries = ((value, (*path, key)),)
            for node, node_path in entries:
                if self.stopped:
                    return
                self.node(node, node_path, seen, depth + 1)

    def node(self, node: Any, path: tuple[str | int, ...], seen: set[str], depth: int):
        location = _Location(None, path)
        if not self.visit(location, depth):
            return
        if not isinstance(node, Mapping):
            self.issue(location, "invalid_node", "Expected a node envelope.")
            return
        node_id = node.get("id")
        if isinstance(node_id, str) and node_id:
            location = _Location(node_id, path)
            if node_id in seen:
                self.issue(
                    location,
                    "duplicate_node_id",
                    "Node ID is duplicated in this workflow.",
                )
            seen.add(node_id)
        else:
            self.issue(location, "invalid_node", "Expected a nonempty node ID.")
        node_type = node.get("type")
        cls = (
            self.context.node_registry.get(node_type)
            if isinstance(node_type, str)
            else None
        )
        if cls is None:
            self.issue(
                location,
                "unknown_node_type",
                "Node type is unavailable in this registry.",
            )
            return
        if node.get("version", "latest") not in ("latest", cls.TYPE_INFO.version):
            self.issue(
                location,
                "unsupported_node_version",
                "Resource preflight requires the current node version or latest.",
            )
            return
        params = node.get("params", {})
        if not isinstance(params, Mapping):
            self.issue(location, "invalid_node", "Expected a parameter mapping.")
            return
        schema = cls.TYPE_INFO.parameter_schema.model_dump(mode="json")
        self.walk(
            schema,
            params,
            (),
            location,
            depth + 1,
            annotation=cls.model_fields["params"].annotation,
        )

    def normalize(
        self,
        schema: Mapping[str, Any],
        value: Any,
        scopes: tuple[Mapping[str, Any], ...],
        depth: int = 0,
    ):
        if depth > 64:
            raise _SchemaError(
                "validation_limit", "Resource union nesting limit exceeded."
            )
        schema, scopes = _resolve(schema, scopes)
        if "allOf" in schema:
            clauses = schema.pop("allOf")
            if not isinstance(clauses, list):
                raise _SchemaError(
                    "unsupported_resource_schema",
                    "Expected resource constraint objects.",
                )
            for clause in clauses:
                if not isinstance(clause, Mapping) or any(
                    k not in _CONSTRAINTS and not k.startswith("x-") for k in clause
                ):
                    raise _SchemaError(
                        "unsupported_resource_schema",
                        "Only constraint-only resource conjunctions are supported.",
                    )
                for key, item in clause.items():
                    if (
                        key.startswith("x-resource-")
                        and key in schema
                        and schema[key] != item
                    ):
                        raise _SchemaError(
                            "ambiguous_resource_schema",
                            "Conflicting resource declarations.",
                        )
                    schema.setdefault(key, item)
        if value is _MISSING and "default" in schema:
            value = schema["default"]
        for keyword in ("anyOf", "oneOf"):
            if keyword not in schema:
                continue
            branches = schema[keyword]
            if not isinstance(branches, list) or not all(
                isinstance(b, Mapping) for b in branches
            ):
                raise _SchemaError(
                    "unsupported_resource_schema", "Expected resource union branches."
                )
            normalized = [_resolve(branch, scopes) for branch in branches]
            normalized = [
                (branch, scope)
                for branch, scope in normalized
                if branch.get("type") != "null"
            ]
            discriminator = schema.get("discriminator", {})
            prop = (
                discriminator.get("propertyName")
                if isinstance(discriminator, Mapping)
                else None
            )
            if isinstance(prop, str) and isinstance(value, Mapping) and prop in value:
                normalized = [
                    (branch, scope)
                    for branch, scope in normalized
                    if branch.get("properties", {}).get(prop, {}).get("const", _MISSING)
                    == value[prop]
                ]
            if not normalized or any(
                _branch_shape(b) != _branch_shape(normalized[0][0])
                for b, _ in normalized[1:]
            ):
                raise _SchemaError(
                    "ambiguous_resource_schema",
                    "Cannot select an unambiguous resource branch.",
                )
            branch, branch_scopes = normalized[0]
            branch, scopes = self.normalize(branch, value, branch_scopes, depth + 1)
            schema = {
                **branch,
                **{
                    k: v
                    for k, v in schema.items()
                    if k not in {keyword, "discriminator"}
                },
            }
        return schema, scopes

    def walk(
        self,
        raw: Any,
        value: Any,
        scopes: tuple[Mapping[str, Any], ...],
        location: _Location,
        depth: int,
        *,
        required: bool = True,
        annotation: Any = None,
        factory: bool = False,
    ):
        if not self.visit(location, depth):
            return
        try:
            if not _relevant(raw, scopes):
                return
            raw, scopes = _resolve(raw, scopes)
            if (
                raw.get("x-resource-required", required) is False
                and (value is None or value is _MISSING)
                and "default" not in raw
                and not factory
                and not _omission_needs_inspection(raw, scopes)
            ):
                return
            schema, scopes = self.normalize(raw, value, scopes)
        except _SchemaError as error:
            self.issue(location, error.code, error.message)
            return
        resource_type = schema.get("x-resource-type")
        if "x-resource-type" in schema and (
            not isinstance(resource_type, str) or not resource_type.strip()
        ):
            self.issue(
                location,
                "unsupported_resource_schema",
                "Expected a nonempty resource type.",
            )
            return
        if resource_type is not None and schema.get("x-value-type") in {
            "WorkflowValue",
            "ValueSchemaValue",
        }:
            self.issue(
                location,
                "unsupported_resource_schema",
                "Resource declarations require a string identifier schema.",
                resource_type=resource_type,
            )
            return
        override = schema.get("x-resource-required", required)
        if not isinstance(override, bool):
            self.issue(
                location,
                "unsupported_resource_schema",
                "Resource requiredness must be Boolean.",
            )
            return
        required = override
        if value is _MISSING:
            if "default" in schema:
                value = schema["default"]
            elif factory:
                self.issue(
                    location,
                    "unsupported_default",
                    "Save an explicit resource value instead of evaluating a default factory.",
                    resource_type=resource_type,
                )
                return
        if value is _MISSING or value is None:
            if required:
                self.issue(
                    location,
                    "missing_resource",
                    "A required resource or its containing parameter is missing.",
                    resource_type=resource_type,
                    incomplete=False,
                )
            return
        if schema.get("x-value-type") == "WorkflowValue":
            self.workflow(
                value, (*location.node_path, "params", *location.param_path), depth + 1
            )
            return
        if any(key in schema for key in _UNSUPPORTED):
            self.issue(
                location,
                "unsupported_resource_schema",
                "Unsupported resource-bearing schema construct.",
            )
            return
        if resource_type is not None:
            if (
                schema.get("type") not in (None, "string")
                or "properties" in schema
                or "items" in schema
            ):
                self.issue(
                    location,
                    "unsupported_resource_schema",
                    "Resource declarations must describe string IDs.",
                    resource_type=resource_type,
                )
            elif not isinstance(value, str) or not value.strip():
                self.issue(
                    location,
                    "invalid_resource_id",
                    "Expected a nonempty string resource ID.",
                    resource_type=resource_type,
                    incomplete=False,
                )
            else:
                self.request(resource_type, value, location)
            return
        properties = schema.get("properties", {})
        if "properties" in schema or isinstance(
            schema.get("additionalProperties"), Mapping
        ):
            if not isinstance(value, Mapping) or not isinstance(properties, Mapping):
                self.issue(
                    location,
                    "unsupported_resource_schema",
                    "Expected an object containing resource parameters.",
                )
                return
            # Bound raw key inspection and sorting, not only subsequent values.
            if len(value) > self.options.max_visits - self.visits:
                self.issue(
                    location,
                    "validation_limit",
                    "Resource mapping inspection limit exceeded.",
                )
                self.stopped = True
                return
            data_keys: set[str] = set()
            for key in value:
                if not self.visit(location, depth):
                    return
                if not isinstance(key, str):
                    self.issue(
                        location,
                        "unsupported_resource_schema",
                        "Expected string resource parameter keys.",
                    )
                    return
                data_keys.add(key)
            fields = _model_fields(annotation)
            required_fields = schema.get("required", [])
            for name, child in sorted(properties.items()):
                field = next(
                    (
                        f
                        for key, f in fields.items()
                        if name in (key, f.alias, f.validation_alias)
                    ),
                    None,
                )
                self.walk(
                    child,
                    value.get(name, _MISSING),
                    scopes,
                    location.child(name),
                    depth + 1,
                    required=name in required_fields,
                    annotation=field.annotation if field else None,
                    factory=field is not None and field.default_factory is not None,
                )
                if self.stopped:
                    return
            additional = schema.get("additionalProperties")
            if isinstance(additional, Mapping):
                args = get_args(_root_annotation(annotation))
                for name in sorted(data_keys - properties.keys()):
                    self.walk(
                        additional,
                        value[name],
                        scopes,
                        location.child(name),
                        depth + 1,
                        annotation=args[1] if len(args) > 1 else None,
                    )
                    if self.stopped:
                        return
        elif "items" in schema:
            if not isinstance(value, (list, tuple)):
                self.issue(
                    location,
                    "unsupported_resource_schema",
                    "Expected a resource collection.",
                )
                return
            args = get_args(_root_annotation(annotation))
            for index, item in enumerate(value):
                self.walk(
                    schema["items"],
                    item,
                    scopes,
                    location.child(index),
                    depth + 1,
                    annotation=args[0] if args else None,
                )
                if self.stopped:
                    return

    def request(self, kind: str, value: str, location: _Location):
        key = (kind, value)
        if key not in self.requests:
            if len(self.requests) >= self.options.max_references:
                self.issue(
                    location,
                    "validation_limit",
                    "Distinct resource reference limit exceeded.",
                )
                self.stopped = True
                return
            self.requests[key] = ResourceRequest(
                request_id=str(len(self.requests)),
                resource_type=kind,
                resource_id=value,
            )
        request = self.requests[key]
        self.locations.setdefault(request.request_id, []).append(
            (self.visits, location)
        )

    async def check(self):
        requests = tuple(self.requests.values())
        cursor = 0

        async def worker():
            nonlocal cursor
            while cursor < len(requests):
                batch = requests[cursor : cursor + self.options.batch_size]
                cursor += len(batch)
                await self.batch(batch)

        tasks = [
            asyncio.create_task(worker())
            for _ in range(min(self.options.max_concurrency, len(requests)))
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def batch(self, requests: Sequence[ResourceRequest]):
        resolver = self.context.resource_resolver
        code: ResourceIssueCode | None = None
        checks: dict[str, ResourceCheck] = {}
        if resolver is None:
            code = "resource_unavailable"
        else:
            try:
                async with asyncio.timeout(self.options.timeout):
                    response = await resolver.check_resources(requests)
            except Exception:
                code = "resource_unavailable"
            else:
                try:
                    if (
                        not isinstance(response, Mapping)
                        or len(response) != len(requests)
                        or set(response) != {r.request_id for r in requests}
                    ):
                        raise ValueError("Mismatched response IDs")
                    checks = {
                        key: ResourceCheck.model_validate(
                            value.model_dump()
                            if isinstance(value, ResourceCheck)
                            else value
                        )
                        for key, value in response.items()
                    }
                except Exception:
                    # A Mapping may read lazily; contain access failures too,
                    # without exposing backend exception text.
                    code = "provider_protocol_error"
        for request in requests:
            if code is not None:
                for order, location in self.locations[request.request_id]:
                    self.issue(
                        location,
                        code,
                        "Resource checks are unavailable."
                        if code == "resource_unavailable"
                        else "Resource provider returned an invalid response.",
                        resource_type=request.resource_type,
                        order=order,
                    )
                continue
            check = checks[request.request_id]
            if check.status in {"available", "missing", "forbidden"}:
                self.checked += 1
            if check.status == "available":
                continue
            codes: dict[str, ResourceIssueCode] = {
                "missing": "missing_resource",
                "forbidden": "resource_forbidden",
                "unavailable": "resource_unavailable",
                "unsupported": "unsupported_resource_type",
            }
            messages = {
                "missing": "Resource does not exist.",
                "forbidden": "Resource is not accessible.",
                "unavailable": "Resource checks are unavailable.",
                "unsupported": "Resource type is not supported by this provider.",
            }
            for order, location in self.locations[request.request_id]:
                self.issue(
                    location,
                    codes[check.status],
                    check.message or messages[check.status],
                    resource_type=request.resource_type,
                    incomplete=check.status in {"unavailable", "unsupported"},
                    order=order,
                )


async def validate_resources(
    workflow: Workflow | Mapping[str, Any],
    context: ValidationContext,
    *,
    options: ResourceValidationOptions | None = None,
) -> ResourceValidationReport:
    """Inspect declared resources without constructing nodes or executing a graph.

    Malformed/incomplete documents produce diagnostics. Provider failures make
    coverage incomplete; cancellation propagates. No report guarantees that a
    structurally invalid graph can run or that resources cannot later change.
    """
    from .workflow import Workflow

    document = (
        workflow.model_dump(mode="json") if isinstance(workflow, Workflow) else workflow
    )
    preflight = _Preflight(context, options or ResourceValidationOptions())
    preflight.workflow(document)
    await preflight.check()
    return ResourceValidationReport(
        issues=tuple(
            issue for _, issue in sorted(preflight.issues, key=lambda pair: pair[0])
        ),
        checked_references=preflight.checked,
        complete=preflight.complete,
    )
