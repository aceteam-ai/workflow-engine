import pytest

from workflow_engine.utils.hash import json_digest


@pytest.mark.unit
def test_json_digest_stable():
    """Hash must not change across refactors — if this fails, digest values in
    schema.py will silently differ from ones produced by older code."""
    data = {"name": "test", "values": [1, 2, 3], "meta": {"active": True}}
    assert json_digest(data) == "70c34029df6c2a201891cdc2f92cb4921dc7dbc5043b47b50813aad2aa4c4dc1"


@pytest.mark.unit
def test_json_digest_key_order_is_irrelevant():
    """Key insertion order must not affect the digest."""
    a = {"z": 1, "a": 2}
    b = {"a": 2, "z": 1}
    assert json_digest(a) == json_digest(b)
