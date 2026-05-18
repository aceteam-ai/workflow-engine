"""Parse `wengine install` target strings.

Operators name node-source packages with several shorthands on top of plain pip
requirements:

- `acme-scrapers` / `acme-scrapers==1.4.0` / `acme-scrapers[screenshot]` —
  any PyPI (or configured-index) requirement.
- `git+https://host/owner/repo@<ref>[#subdirectory=...]` — a raw `git+` URL
  handed straight to `uv add`. Needs system `git`.
- `./path` (or `../path` / `/path`) or `path:<anything>` — an editable
  local install.
- `github:owner/repo[@ref][#subdirectory=...]` / `gitlab:...` — forge
  shorthand: `wengine` resolves the ref to a commit SHA over HTTPS and points
  `uv` at a pinned tarball URL. No system `git` required.
- `owner/repo` (bare) — shorthand for `github:owner/repo`.

`parse_install_target` does no I/O. `resolve_forge_ref` performs HTTPS GETs
via the module-private `_http_get`, which tests monkeypatch.
"""

from __future__ import annotations

import json
import re
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Literal
from urllib.parse import quote

from workflow_engine.utils.model import ImmutableBaseModel

ForgeName = Literal["github", "gitlab"]

_FORGE_TARBALL_URL_TEMPLATES: Mapping[ForgeName, str] = {
    "github": "https://codeload.github.com/{owner}/{repo}/tar.gz/{sha}",
    # GitLab's archive endpoint: the trailing filename is cosmetic but required
    # by the route, and conventionally `<repo>-<sha>.tar.gz`.
    "gitlab": "https://gitlab.com/{owner}/{repo}/-/archive/{sha}/{repo}-{sha}.tar.gz",
}


class BaseInstallTarget(ImmutableBaseModel, ABC):
    """Common base for parsed `wengine install` targets."""

    @cached_property
    @abstractmethod
    def uv_add_args(self) -> Sequence[str]:
        """The argv tail to splice after `uv add`."""


class PyPITarget(BaseInstallTarget):
    """A PEP 508-style requirement, passed straight through to `uv add`."""

    requirement: str

    @cached_property
    def uv_add_args(self) -> Sequence[str]:
        return (self.requirement,)


class GitTarget(BaseInstallTarget):
    """A raw `git+` URL, passed straight through to `uv add`."""

    url: str

    @cached_property
    def uv_add_args(self) -> Sequence[str]:
        return (self.url,)


class PathTarget(BaseInstallTarget):
    """A local path, installed editable."""

    path: str

    @cached_property
    def uv_add_args(self) -> Sequence[str]:
        return ("--editable", self.path)


class ForgeTarget(BaseInstallTarget):
    """A `github:` / `gitlab:` shorthand needing HTTPS ref resolution.

    `uv_add_args` resolves the ref to a commit SHA over HTTPS, then hands
    `uv add` a pinned tarball URL. uv downloads the archive and reads its
    metadata to discover the distribution name. Reading the property issues
    network requests — the result is cached on the instance after the first
    read.

    Only the public `github.com` / `gitlab.com` hosts are supported. For
    self-hosted instances or GitHub Enterprise, use a `git+https://` URL.
    """

    forge: ForgeName
    owner: str
    repo: str
    ref: str | None = None
    """The git ref to install at. `None` means resolve the default branch."""
    subdirectory: str | None = None
    """Optional subdirectory within the repo containing the package."""

    def tarball_url(self, sha: str) -> str:
        return _FORGE_TARBALL_URL_TEMPLATES[self.forge].format(
            owner=self.owner,
            repo=self.repo,
            sha=sha,
        )

    @cached_property
    def uv_add_args(self) -> Sequence[str]:
        sha = resolve_forge_ref(self)
        return (self.tarball_url(sha),)


InstallTarget = PyPITarget | GitTarget | PathTarget | ForgeTarget


# Forge shorthand: optional `<forge>:` prefix, owner/repo, optional @ref,
# optional #subdirectory=...
_FORGE_SHORTHAND_RE = re.compile(
    r"""
    ^
    (?:(?P<forge>github|gitlab):)?
    (?P<owner>[A-Za-z0-9][A-Za-z0-9._-]*)
    /
    (?P<repo>[A-Za-z0-9][A-Za-z0-9._-]*)
    (?:@(?P<ref>[^#]+))?
    (?:\#subdirectory=(?P<subdir>.+))?
    $
    """,
    re.VERBOSE,
)


def parse_install_target(s: str) -> InstallTarget:
    """Parse a `wengine install` target string.

    Raises `ValueError` on empty input. Anything that doesn't look like a
    git/path/forge shorthand is treated as a PyPI requirement and handed to
    `uv add` as-is — `uv` produces the real error if the requirement is
    malformed.
    """
    if not s:
        raise ValueError("empty install target")

    if s.startswith("git+"):
        return GitTarget(url=s)

    if s.startswith("path:"):
        return PathTarget(path=s[len("path:") :])
    if s.startswith(("./", "../", "/")):
        return PathTarget(path=s)

    # Explicit forge shorthand (`github:` / `gitlab:`) or a bare `owner/repo`
    # which defaults to GitHub. PyPI requirements never contain a `/`, so a
    # slash is the disambiguator. Anything with brackets or version specifiers
    # falls through to PyPI even if it has a slash.

    if (match := _FORGE_SHORTHAND_RE.match(s)) is not None:
        forge = match.group("forge") or "github"
        return ForgeTarget(
            forge=forge,  # type: ignore[arg-type]
            owner=match.group("owner"),
            repo=match.group("repo"),
            ref=match.group("ref"),
            subdirectory=match.group("subdir"),
        )

    return PyPITarget(requirement=s)


def _http_get(url: str) -> bytes:
    """HTTPS GET returning the response body. Tests monkeypatch this."""
    req = urllib.request.Request(url, headers={"User-Agent": "wengine-install"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def resolve_forge_ref(target: ForgeTarget) -> str:
    """Resolve a `ForgeTarget`'s ref to a commit SHA over HTTPS.

    When `target.ref` is `None`, the repository's default branch is fetched
    first and then resolved. Returns the commit SHA as a string.
    """
    if target.forge == "github":
        return _resolve_github(target)
    if target.forge == "gitlab":
        return _resolve_gitlab(target)
    raise AssertionError(f"unreachable forge: {target.forge!r}")


def _resolve_github(target: ForgeTarget) -> str:
    base = f"https://api.github.com/repos/{target.owner}/{target.repo}"
    ref = target.ref
    if ref is None:
        repo_info = json.loads(_http_get(base))
        ref = repo_info["default_branch"]
    commit = json.loads(_http_get(f"{base}/commits/{quote(ref, safe='')}"))
    return commit["sha"]


def _resolve_gitlab(target: ForgeTarget) -> str:
    project = quote(f"{target.owner}/{target.repo}", safe="")
    base = f"https://gitlab.com/api/v4/projects/{project}"
    ref = target.ref
    if ref is None:
        project_info = json.loads(_http_get(base))
        ref = project_info["default_branch"]
    commit = json.loads(_http_get(f"{base}/repository/commits/{quote(ref, safe='')}"))
    return commit["id"]


__all__ = [
    "BaseInstallTarget",
    "ForgeName",
    "ForgeTarget",
    "GitTarget",
    "InstallTarget",
    "PathTarget",
    "PyPITarget",
    "parse_install_target",
    "resolve_forge_ref",
]
