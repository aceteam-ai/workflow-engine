import pytest
from pydantic import BaseModel, ValidationError

from workflow_engine.contexts.in_memory import InMemoryContext
from workflow_engine.core.values import (
    JSONValue,
    ModelValue,
    Value,
)
from workflow_engine.core.values.extraction import (
    Entity,
    ExtractionResult,
    ExtractionResultValue,
    Relation,
)


@pytest.fixture
def context():
    return InMemoryContext()


# --- Simple test models ---


class PersonModel(BaseModel):
    name: str
    age: int


class EmployeeModel(BaseModel):
    name: str
    age: int
    department: str


PersonValue = ModelValue[PersonModel]
EmployeeValue = ModelValue[EmployeeModel]


# --- ModelValue creation ---


@pytest.mark.unit
def test_model_value_creation():
    person = PersonModel(name="Alice", age=30)
    val = PersonValue(person)
    assert val.root == person
    assert val.root.name == "Alice"
    assert val.root.age == 30
    assert isinstance(val, ModelValue)
    assert isinstance(val, Value)


# --- JSONValue -> ModelValue ---


@pytest.mark.unit
@pytest.mark.asyncio
async def test_json_to_model_value_cast(context):
    json_val = JSONValue({"name": "Bob", "age": 25})
    result = await json_val.cast_to(PersonValue, context=context)
    assert isinstance(result.root, PersonModel)
    assert result.root.name == "Bob"
    assert result.root.age == 25


@pytest.mark.unit
@pytest.mark.asyncio
async def test_json_to_model_value_cast_invalid(context):
    json_val = JSONValue({"name": "Bob"})  # missing required 'age'
    with pytest.raises(ValidationError):
        await json_val.cast_to(PersonValue, context=context)


@pytest.mark.unit
def test_json_cannot_cast_to_unparameterized_model_value():
    assert not JSONValue.can_cast_to(ModelValue)


# --- ModelValue -> JSONValue ---


@pytest.mark.unit
@pytest.mark.asyncio
async def test_model_value_to_json_cast(context):
    person = PersonModel(name="Carol", age=40)
    val = PersonValue(person)
    json_val = await val.cast_to(JSONValue, context=context)
    assert isinstance(json_val, JSONValue)
    assert json_val.root == {"name": "Carol", "age": 40}


# --- ModelValue -> ModelValue (same type) ---


@pytest.mark.unit
@pytest.mark.asyncio
async def test_model_value_same_type_cast(context):
    person = PersonModel(name="Dave", age=35)
    val = PersonValue(person)
    result = await val.cast_to(PersonValue, context=context)
    assert result.root.name == "Dave"
    assert result.root.age == 35


# --- ModelValue[A] -> ModelValue[B] (compatible) ---


@pytest.mark.unit
@pytest.mark.asyncio
async def test_model_value_cross_model_cast_compatible(context):
    """EmployeeModel has a superset of PersonModel fields, so Employee -> Person works."""
    employee = EmployeeModel(name="Eve", age=28, department="Engineering")
    val = EmployeeValue(employee)
    result = await val.cast_to(PersonValue, context=context)
    assert isinstance(result.root, PersonModel)
    assert result.root.name == "Eve"
    assert result.root.age == 28


# --- ModelValue[A] -> ModelValue[B] (incompatible) ---


@pytest.mark.unit
@pytest.mark.asyncio
async def test_model_value_cross_model_cast_incompatible(context):
    """PersonModel -> EmployeeModel fails because 'department' is missing."""
    person = PersonModel(name="Frank", age=50)
    val = PersonValue(person)
    with pytest.raises(ValidationError):
        await val.cast_to(EmployeeValue, context=context)


# --- can_cast_to checks ---


@pytest.mark.unit
def test_can_cast_to_checks():
    assert JSONValue.can_cast_to(PersonValue)
    assert PersonValue.can_cast_to(JSONValue)
    assert PersonValue.can_cast_to(PersonValue)
    assert PersonValue.can_cast_to(EmployeeValue)  # generic caster exists, validation at runtime
    assert EmployeeValue.can_cast_to(PersonValue)


# --- ExtractionResultValue ---


@pytest.mark.unit
def test_extraction_result_value_creation():
    result = ExtractionResult(
        document_id="doc-1",
        source_text="Alice works at Acme Corp.",
        schema_id="schema-v1",
        entities=[
            Entity(id="e1", text="Alice", type="PERSON", confidence=0.95),
            Entity(id="e2", text="Acme Corp.", type="ORGANIZATION"),
        ],
        relations=[
            Relation(
                id="r1",
                type="WORKS_AT",
                subject_id="e1",
                object_id="e2",
                confidence=0.9,
            ),
        ],
    )
    val = ExtractionResultValue(result)
    assert val.root.document_id == "doc-1"
    assert len(val.root.entities) == 2
    assert len(val.root.relations) == 1
    assert val.root.entities[0].confidence == 0.95
    assert val.root.entities[1].confidence is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extraction_result_value_json_roundtrip(context):
    result = ExtractionResult(
        document_id="doc-2",
        chunk_id="chunk-1",
        source_text="Bob is CEO of WidgetCo.",
        schema_id="schema-v2",
        entities=[
            Entity(id="e1", text="Bob", type="PERSON"),
            Entity(id="e2", text="WidgetCo", type="ORGANIZATION"),
        ],
        relations=[
            Relation(id="r1", type="CEO_OF", subject_id="e1", object_id="e2"),
        ],
    )
    val = ExtractionResultValue(result)

    # ModelValue -> JSONValue
    json_val = await val.cast_to(JSONValue, context=context)
    assert isinstance(json_val, JSONValue)
    assert isinstance(json_val.root, dict)
    assert json_val.root["document_id"] == "doc-2"

    # JSONValue -> ModelValue
    restored = await json_val.cast_to(ExtractionResultValue, context=context)
    assert isinstance(restored.root, ExtractionResult)
    assert restored.root.document_id == "doc-2"
    assert restored.root.chunk_id == "chunk-1"
    assert len(restored.root.entities) == 2
    assert restored.root.relations[0].type == "CEO_OF"
