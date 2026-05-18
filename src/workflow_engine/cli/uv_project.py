"""Locate the `uv` project that backs an `engine.yaml`, and drive `uv` against it.

Two modes:

- **Embedded**: `engine.yaml` lives inside an existing `uv`/Python project (a
  `pyproject.toml` somewhere at or above `engine.yaml`'s directory). `wengine
  install` mutates the host project's `pyproject.toml` and `uv.lock`.
- **Standalone**: no enclosing `pyproject.toml`. `wengine` owns a minimal
  `pyproject.toml` written next to `engine.yaml` whose sole purpose is to
  pin the operator's chosen node-source packages.

Discovery walks up from a starting directory looking for `engine.yaml` (or
`engine.yml`), then walks up again from that directory looking for
`pyproject.toml` to decide the mode.

Subprocess invocation goes through the module-private `_run_uv`, which tests
monkeypatch.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

ENGINE_YAML_NAMES: tuple[str, ...] = ("engine.yaml", "engine.yml")

UvMode = Literal["embedded", "standalone"]


_MINIMAL_PYPROJECT = """\
[project]
name = "wengine-nodes"
version = "0.0.0"
description = "Node-source packages managed by wengine."
requires-python = ">=3.12"
dependencies = []
"""


class EngineYamlNotFound(FileNotFoundError):
    """Raised when no `engine.yaml` is found by walking up from a start dir."""


def find_engine_yaml(start: Path) -> Path | None:
    """Walk up from `start` looking for `engine.yaml` / `engine.yml`.

    Returns the first match found, or `None` if the filesystem root is
    reached without one.
    """
    current = start.resolve()
    while True:
        for name in ENGINE_YAML_NAMES:
            candidate = current / name
            if candidate.is_file():
                return candidate
        if current.parent == current:
            return None
        current = current.parent


def find_pyproject(start: Path) -> Path | None:
    """Walk up from `start` looking for `pyproject.toml`."""
    current = start.resolve()
    while True:
        candidate = current / "pyproject.toml"
        if candidate.is_file():
            return candidate
        if current.parent == current:
            return None
        current = current.parent


class UvProject:
    """A `uv` project rooted at a `pyproject.toml`, with awareness of the
    `engine.yaml` it backs and whether that pairing is embedded or standalone."""

    root: Path
    """Directory containing `pyproject.toml`."""
    mode: UvMode
    engine_yaml: Path

    def __init__(self, *, root: Path, mode: UvMode, engine_yaml: Path) -> None:
        self.root = root
        self.mode = mode
        self.engine_yaml = engine_yaml

    @classmethod
    def locate(cls, start: Path) -> UvProject:
        """Locate the uv project backing the `engine.yaml` discovered from `start`.

        Walks up for `engine.yaml`, then walks up from that directory for
        `pyproject.toml`. If found, embedded mode rooted at the pyproject's
        directory; otherwise standalone mode rooted alongside `engine.yaml`.

        Raises `EngineYamlNotFound` if no `engine.yaml` is reachable.
        """
        engine_yaml = find_engine_yaml(start)
        if engine_yaml is None:
            raise EngineYamlNotFound(
                f"No engine.yaml found at or above {start!s}. "
                f"Run `wengine init` to create one."
            )
        pyproject = find_pyproject(engine_yaml.parent)
        if pyproject is not None:
            return cls(root=pyproject.parent, mode="embedded", engine_yaml=engine_yaml)
        return cls(root=engine_yaml.parent, mode="standalone", engine_yaml=engine_yaml)

    def ensure_pyproject(self) -> None:
        """Write a minimal `pyproject.toml` in standalone mode if missing.

        No-op in embedded mode — the host project already owns its file.
        """
        if self.mode == "embedded":
            return
        path = self.root / "pyproject.toml"
        if path.exists():
            return
        path.write_text(_MINIMAL_PYPROJECT)

    def add(self, args: Sequence[str]) -> None:
        """Run `uv add <args>` in this project."""
        self.ensure_pyproject()
        _run_uv(["add", *args], cwd=self.root)

    def remove(self, name: str) -> None:
        """Run `uv remove <name>` in this project."""
        _run_uv(["remove", name], cwd=self.root)

    def sync(self) -> None:
        """Run `uv sync` in this project."""
        _run_uv(["sync"], cwd=self.root)


def _run_uv(args: Sequence[str], *, cwd: Path) -> None:
    """Invoke `uv` with the given args in `cwd`. Tests monkeypatch this.

    stdout/stderr are inherited so users see uv's progress live. A non-zero
    exit raises `subprocess.CalledProcessError`.
    """
    subprocess.run(["uv", *args], cwd=cwd, check=True)


__all__ = [
    "ENGINE_YAML_NAMES",
    "EngineYamlNotFound",
    "UvMode",
    "UvProject",
    "find_engine_yaml",
    "find_pyproject",
]
