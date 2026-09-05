# workflow_engine/core/hints.py
from typing import ClassVar

from pydantic import ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer

from ..utils.model import ImmutableBaseModel


class Hints(ImmutableBaseModel):
    """
    Host-facing annotations attached to a single node.

    A hint is information a host *may* honor, clamp, or ignore. Ignoring every
    hint on every node must never change a workflow's result; that property is
    what keeps this channel out of the engine's semantics and is enforced by
    construction rather than convention, since nothing under ``execution/`` or
    ``nodes/`` reads this field. See ``schema/hints.md`` for the published
    contract, including why a host-specific reference (e.g. a machine pin)
    does *not* belong here.

    Unlike ``Params``, hints are not part of a node's typed input/output
    contract: a node's ``input_type`` and ``output_type`` never depend on its
    hints, so adding, removing, or changing a hint can't change what a
    workflow computes.

    Unknown keys round-trip rather than being rejected (``extra="allow"``),
    since a hint's entire point is that not understanding it is safe. A key
    that only resolves inside one host is an environment reference, not a
    hint, and does not belong here regardless of whether it is recognized.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="allow",
        frozen=True,
    )

    max_concurrency: int | None = Field(
        default=None,
        ge=1,
        title="Max Concurrency",
        description=(
            "A suggested upper bound on how many of this node's parallel "
            "branches (for example, the items of a For Each) a host should "
            "run at once. A host may clamp this lower to protect its own "
            "resources, run fewer branches at a time, or ignore it entirely "
            "and run unbounded; the workflow's result does not depend on how "
            "many branches were in flight at once."
        ),
    )

    @model_serializer(mode="wrap")
    def _serialize_omit_unset_max_concurrency(
        self, handler: SerializerFunctionWrapHandler
    ):
        """
        Omit ``max_concurrency`` when it is unset, so a node that hasn't
        opted into this hint doesn't carry a ``{"max_concurrency": null}``
        key it never asked for.

        Any other key survives unconditionally, including ones this engine
        doesn't recognize (``extra="allow"``): a hint set by a newer host
        must still round-trip even though we can't interpret it.
        """
        data = handler(self)
        if data.get("max_concurrency") is None:
            data.pop("max_concurrency", None)
        return data


__all__ = [
    "Hints",
]
