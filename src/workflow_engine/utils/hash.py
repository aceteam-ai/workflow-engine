import hashlib
import json
from typing import Any


def json_digest(data: dict[str, Any]) -> str:
    """Return a stable SHA-256 hex digest of a JSON-serialisable dict."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
