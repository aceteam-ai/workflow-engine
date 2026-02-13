from hashlib import md5

import pytest

from workflow_engine import (
    Context,
    File,
    IntegerValue,
    JSONValue,
    SequenceValue,
    StringMapValue,
    StringValue,
)
from workflow_engine.contexts.in_memory import InMemoryContext
from workflow_engine.files import (
    CSVFileValue,
    JSONFileValue,
    JSONLinesFileValue,
    MarkdownFileValue,
    TextFileValue,
)


@pytest.fixture
def context():
    """Create a test context for value casting operations."""
    return InMemoryContext()


@pytest.mark.unit
async def test_cast_jsonlines_to_sequence(context: Context):
    """Test that JSONLinesFileValue can be cast to a SequenceValue."""
    jsonl_file = JSONLinesFileValue.from_path("input.jsonl")
    contents = [{"a": 1}, {"b": 2}, {"c": 3}]
    contents_str = '{"a": 1}\n{"b": 2}\n{"c": 3}'

    await jsonl_file.write_data(context, contents)

    assert (await jsonl_file.read_text(context)) == contents_str

    data = await SequenceValue[StringMapValue[IntegerValue]].cast_from(
        jsonl_file,
        context=context,
    )
    assert data == contents

    json_files = await SequenceValue[JSONFileValue].cast_from(
        jsonl_file,
        context=context,
    )
    assert json_files == SequenceValue[JSONFileValue](
        # md5 hashes of the data
        [
            JSONFileValue(File(path=f"{md5(b"{'a': 1}").hexdigest()}.json")),
            JSONFileValue(File(path=f"{md5(b"{'b': 2}").hexdigest()}.json")),
            JSONFileValue(File(path=f"{md5(b"{'c': 3}").hexdigest()}.json")),
        ]
    )
    assert (await json_files[0].read_data(context)) == {"a": 1}
    assert (await json_files[1].read_data(context)) == {"b": 2}
    assert (await json_files[2].read_data(context)) == {"c": 3}


@pytest.mark.unit
async def test_cast_text_file_to_and_from_string(context: Context):
    """Test that TextFileValue can be cast to and from StringValue."""
    content = "Hello, world!"
    text_file = TextFileValue(File(path="test.txt"))
    text_file = await text_file.write_text(context, content)

    # TextFileValue -> StringValue
    str_val = await text_file.cast_to(StringValue, context=context)
    assert isinstance(str_val, StringValue)
    assert str_val.root == content

    # StringValue -> TextFileValue
    str_val = StringValue("New text content")
    text_file_from_str = await str_val.cast_to(TextFileValue, context=context)
    assert isinstance(text_file_from_str, TextFileValue)
    assert text_file_from_str.path.endswith(".txt")
    assert (await text_file_from_str.read_text(context)) == "New text content"


@pytest.mark.unit
async def test_cast_markdown_file_to_and_from_string(context: Context):
    """Test that MarkdownFileValue can be cast to and from StringValue."""
    content = "# Heading\n\nSome **bold** markdown content."
    md_file = MarkdownFileValue(File(path="doc.md"))
    md_file = await md_file.write_text(context, content)

    # MarkdownFileValue -> StringValue
    str_val = await md_file.cast_to(StringValue, context=context)
    assert isinstance(str_val, StringValue)
    assert str_val.root == content

    # StringValue -> MarkdownFileValue
    str_val = StringValue("# Another doc\n\nWith _italic_ text.")
    md_from_str = await str_val.cast_to(MarkdownFileValue, context=context)
    assert isinstance(md_from_str, MarkdownFileValue)
    assert md_from_str.path.endswith(".md")
    assert (await md_from_str.read_text(context)) == str_val.root


@pytest.mark.unit
async def test_csv_write_and_read_data(context: Context):
    """Test that CSVFileValue can write and read data via InMemoryContext."""
    csv_file = CSVFileValue(File(path="data.csv"))
    data = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ]
    csv_file = await csv_file.write_data(context, data)

    read_back = await csv_file.read_data(context)
    # DictReader returns string values
    assert read_back == (
        {"name": "Alice", "age": "30"},
        {"name": "Bob", "age": "25"},
    )


@pytest.mark.unit
async def test_csv_cast_json_to_csv(context: Context):
    """Test that JSONValue can be cast to CSVFileValue."""
    # Sequence of mappings
    json_val = JSONValue([{"x": 1, "y": 2}, {"x": 3, "y": 4}])
    csv_file = await json_val.cast_to(CSVFileValue, context=context)
    assert isinstance(csv_file, CSVFileValue)
    data = await csv_file.read_data(context)
    assert data == (
        {"x": "1", "y": "2"},
        {"x": "3", "y": "4"},
    )

    # Single mapping as one row
    json_val = JSONValue({"a": 10, "b": 20})
    csv_file = await json_val.cast_to(CSVFileValue, context=context)
    data = await csv_file.read_data(context)
    assert data == ({"a": "10", "b": "20"},)


@pytest.mark.unit
async def test_csv_cast_string_map_to_csv(context: Context):
    """Test that StringMapValue can be cast to CSVFileValue via JSON."""
    str_map = StringMapValue[IntegerValue](
        {"foo": IntegerValue(1), "bar": IntegerValue(2)}
    )
    csv_file = await str_map.cast_to(CSVFileValue, context=context)
    assert isinstance(csv_file, CSVFileValue)
    data = await csv_file.read_data(context)
    assert data == ({"foo": "1", "bar": "2"},)


@pytest.mark.unit
async def test_csv_cast_sequence_to_csv(context: Context):
    """Test that SequenceValue of mappings can be cast to CSVFileValue."""
    seq = SequenceValue[StringMapValue[IntegerValue]](
        [
            StringMapValue[IntegerValue]({"a": IntegerValue(1), "b": IntegerValue(2)}),
            StringMapValue[IntegerValue]({"a": IntegerValue(3), "b": IntegerValue(4)}),
        ]
    )
    csv_file = await seq.cast_to(CSVFileValue, context=context)
    assert isinstance(csv_file, CSVFileValue)
    data = await csv_file.read_data(context)
    assert data == (
        {"a": "1", "b": "2"},
        {"a": "3", "b": "4"},
    )


@pytest.mark.unit
async def test_csv_cast_csv_to_json(context: Context):
    """Test that CSVFileValue can be cast to JSONValue."""
    csv_file = CSVFileValue(File(path="test.csv"))
    data = [{"col1": "val1", "col2": "val2"}]
    csv_file = await csv_file.write_data(context, data)

    json_val = await csv_file.cast_to(JSONValue, context=context)
    assert isinstance(json_val, JSONValue)
    # read_data returns a tuple; JSONValue preserves that
    assert json_val.root == ({"col1": "val1", "col2": "val2"},)


@pytest.mark.unit
async def test_csv_cast_csv_to_sequence(context: Context):
    """Test that CSVFileValue can be cast to SequenceValue."""
    csv_file = CSVFileValue(File(path="test.csv"))
    data = [{"name": "X", "id": "1"}, {"name": "Y", "id": "2"}]
    csv_file = await csv_file.write_data(context, data)

    seq = await SequenceValue[JSONValue].cast_from(csv_file, context=context)
    assert len(seq) == 2
    assert seq[0].root == {"name": "X", "id": "1"}
    assert seq[1].root == {"name": "Y", "id": "2"}
