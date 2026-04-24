# workflow_engine/core/values/extraction.py

from pydantic import Field

from ...utils.model import ImmutableBaseModel
from .model import ModelValue


class Entity(ImmutableBaseModel):
    """An extracted entity from a document."""

    id: str
    text: str
    type: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Relation(ImmutableBaseModel):
    """A relation between two entities."""

    id: str
    type: str
    subject_id: str
    object_id: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ExtractionResult(ImmutableBaseModel):
    """Result of an entity/relation extraction from a document."""

    document_id: str
    chunk_id: str | None = None
    source_text: str
    schema_id: str
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


ExtractionResultValue = ModelValue[ExtractionResult]


__all__ = [
    "Entity",
    "ExtractionResult",
    "ExtractionResultValue",
    "Relation",
]
