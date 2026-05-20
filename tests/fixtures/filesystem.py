"""Filesystem-related test fixtures.

Wired into the test session via `pytest_plugins` in the top-level
`tests/conftest.py`, so every fixture here is available project-wide as if it
were defined in `conftest.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest


@pytest.fixture
def confine_is_file_to(monkeypatch) -> Callable[[Path], None]:
    """Confine `Path.is_file()` to a subtree, hiding everything outside it.

    CLI helpers that walk up the filesystem looking for marker files
    (`engine.yaml`, `pyproject.toml`, ...) ascend all the way to the root, so
    a stray file in a real parent directory (e.g. `/tmp` or a CI home dir) can
    make "not found" assertions flaky. Call the returned function with a root
    (typically `tmp_path`): paths at or below it delegate to the real
    `is_file`, everything else reports `False`.
    """
    real_is_file = Path.is_file

    def confine(root: Path) -> None:
        resolved_root = root.resolve()

        def fake(self: Path) -> bool:
            resolved = self.resolve()
            if resolved != resolved_root and resolved_root not in resolved.parents:
                return False
            return real_is_file(self)

        monkeypatch.setattr(Path, "is_file", fake)

    return confine
