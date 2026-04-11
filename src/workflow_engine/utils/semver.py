# workflow_engine/utils/semver.py

import re

SEMANTIC_VERSION_PATTERN = r"^(\d+)\.(\d+)\.(\d+)$"
_SEMANTIC_VERSION_REGEX = re.compile(SEMANTIC_VERSION_PATTERN)


SEMANTIC_VERSION_OR_LATEST_PATTERN = r"^(\d+)\.(\d+)\.(\d+)$|^latest$"
LATEST_SEMANTIC_VERSION = "latest"


def parse_semantic_version(version: str) -> tuple[int, int, int]:
    match = _SEMANTIC_VERSION_REGEX.match(version)
    if match is None:
        raise ValueError(f"Invalid semantic version: {version}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


__all__ = [
    "LATEST_SEMANTIC_VERSION",
    "SEMANTIC_VERSION_OR_LATEST_PATTERN",
    "SEMANTIC_VERSION_PATTERN",
    "parse_semantic_version",
]
