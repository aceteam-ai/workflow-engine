"""Serializable replacement provenance and the executor outcome protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from ..utils.model import ImmutableBaseModel
from .node import Node
from .values import Data, DataMapping, ValueSchema, get_data_dict
from .values.data import get_data_schema

if TYPE_CHECKING:
    from .context import ExecutionContext


def fingerprint(value: object) -> str:
    """Hash canonical JSON; checkpoint identities never depend on Python objects."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


class ReplacementRetry(ImmutableBaseModel):
    attempt: int = Field(ge=0)
    next_retry_at: datetime | None = None


class ReplacementFrame(ImmutableBaseModel):
    """Versioned host checkpoint. Store before dispatch; replay by explicit relation.

    Hosts supply ``replacement`` from ``on_node_start`` and this frame from
    ``get_node_replacement_frame``. Inputs and contracts are checked again by
    the engine. Cached completed outputs remain the ordinary context's concern.
    """

    format_version: Literal[1] = 1
    logical_node_id: str
    delegator_id: str
    replacement_id: str
    original_label: str
    hop: int = Field(ge=0)
    max_replacement_hops: int = Field(gt=0)
    delegator: Node
    replacement: Node
    input_schema: ValueSchema
    output_schema: ValueSchema
    replacement_input_schema: ValueSchema
    replacement_output_schema: ValueSchema
    input_fingerprint: str
    contract_fingerprint: str
    completed_slots: tuple[str, ...] = ()
    retries: dict[str, ReplacementRetry] = Field(default_factory=dict)
    status: Literal["pending", "completed", "failed"] = "pending"
    failure_node_id: str | None = None
    output_json: str | None = None

    def replay(
        self,
        *,
        node: Node,
        input_type: type[Data],
        output_type: type[Data],
        input: DataMapping,
    ) -> DataMapping | Node:
        """Validate a saved selection or completed pin before returning a start override."""
        from .error import NodeReplacementException

        if (
            node.id != self.delegator_id
            or node.without_hints().model_dump(mode="json")
            != self.delegator.without_hints().model_dump(mode="json")
            or fingerprint(
                {key: value.model_dump(mode="json") for key, value in input.items()}
            )
            != self.input_fingerprint
            or get_data_schema(input_type) != self.input_schema
            or get_data_schema(output_type) != self.output_schema
        ):
            raise NodeReplacementException(
                "Replacement checkpoint input, node or schema mismatch.", node=node
            )
        if self.status == "failed":
            raise NodeReplacementException(
                "Cannot resume a terminal failed replacement frame.", node=node
            )
        if self.status == "completed":
            if self.output_json is None:
                raise NodeReplacementException(
                    "Completed replacement checkpoint has no output.", node=node
                )
            return get_data_dict(output_type.model_validate_json(self.output_json))
        return self.replacement


@dataclass(frozen=True)
class NodeReplacement:
    """Outcome understood by replacement-aware execution algorithms.

    The input is the caller's already-cast, defaulted input, never an upstream
    shortcut. The executor owns normalization, validation and child admission.
    """

    replacement: Node
    input: DataMapping


async def adapt_data(
    values: DataMapping, target: type[Data], *, node: Node, context: ExecutionContext
) -> DataMapping:
    """Project, cast and instantiate a contract, retaining concrete Value validation."""
    from .error import NodeReplacementException
    from .values import get_data_dict, get_data_fields

    try:
        projected = {}
        for name, (value_type, _) in get_data_fields(target).items():
            if name in values:
                value = values[name]
                projected[name] = (
                    value
                    if isinstance(value, value_type)
                    else await value.cast_to(value_type, context=context)
                )
        return get_data_dict(target.model_validate(projected))
    except Exception as exc:
        raise NodeReplacementException(
            f"Replacement value adaptation to {target.__name__} failed: {exc}",
            node=node,
        ) from exc
