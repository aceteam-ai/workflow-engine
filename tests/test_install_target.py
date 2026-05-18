"""Tests for the `wengine install` target parser and forge ref resolver."""

from __future__ import annotations

import json

import pytest

from workflow_engine.cli import install_target as install_target_module
from workflow_engine.cli.install_target import (
    ForgeTarget,
    GitTarget,
    PathTarget,
    PyPITarget,
    parse_install_target,
    resolve_forge_ref,
)


class TestParsePypi:
    def test_bare_name(self):
        t = parse_install_target("acme-scrapers")
        assert isinstance(t, PyPITarget)
        assert t.requirement == "acme-scrapers"
        assert t.uv_add_args == ("acme-scrapers",)

    def test_with_version_specifier(self):
        t = parse_install_target("acme-scrapers==1.4.0")
        assert isinstance(t, PyPITarget)
        assert t.requirement == "acme-scrapers==1.4.0"

    def test_with_extras(self):
        t = parse_install_target("acme-scrapers[screenshot]")
        assert isinstance(t, PyPITarget)
        assert t.requirement == "acme-scrapers[screenshot]"

    def test_with_extras_and_version(self):
        t = parse_install_target("acme-scrapers[screenshot,crawl-site]>=1.4")
        assert isinstance(t, PyPITarget)
        assert t.requirement == "acme-scrapers[screenshot,crawl-site]>=1.4"


class TestParseGit:
    def test_https(self):
        t = parse_install_target("git+https://gitlab.com/acme/iteration.git@main")
        assert isinstance(t, GitTarget)
        assert t.url == "git+https://gitlab.com/acme/iteration.git@main"
        assert t.uv_add_args == ("git+https://gitlab.com/acme/iteration.git@main",)

    def test_with_subdirectory(self):
        url = "git+https://github.com/acme/big-monorepo@main#subdirectory=tools/nodes"
        t = parse_install_target(url)
        assert isinstance(t, GitTarget)
        assert t.url == url


class TestParsePath:
    def test_dot_slash(self):
        t = parse_install_target("./vendor/internal-nodes")
        assert isinstance(t, PathTarget)
        assert t.path == "./vendor/internal-nodes"
        assert t.uv_add_args == ("--editable", "./vendor/internal-nodes")

    def test_parent(self):
        t = parse_install_target("../sibling")
        assert isinstance(t, PathTarget)
        assert t.path == "../sibling"

    def test_absolute(self):
        t = parse_install_target("/opt/internal-nodes")
        assert isinstance(t, PathTarget)
        assert t.path == "/opt/internal-nodes"

    def test_path_prefix(self):
        t = parse_install_target("path:./vendor/internal-nodes")
        assert isinstance(t, PathTarget)
        assert t.path == "./vendor/internal-nodes"

    def test_path_prefix_strips_only_prefix(self):
        # `path:` is just a disambiguator; the remainder is used verbatim.
        t = parse_install_target("path:relative/dir")
        assert isinstance(t, PathTarget)
        assert t.path == "relative/dir"


class TestParseForge:
    def test_github_explicit_with_ref(self):
        t = parse_install_target("github:acme/iteration@v2.0.0")
        assert isinstance(t, ForgeTarget)
        assert t.forge == "github"
        assert t.owner == "acme"
        assert t.repo == "iteration"
        assert t.ref == "v2.0.0"
        assert t.subdirectory is None

    def test_github_default_branch(self):
        t = parse_install_target("github:acme/iteration")
        assert isinstance(t, ForgeTarget)
        assert t.ref is None

    def test_gitlab_explicit(self):
        t = parse_install_target("gitlab:acme/scrapers@1.4.0")
        assert isinstance(t, ForgeTarget)
        assert t.forge == "gitlab"
        assert t.owner == "acme"
        assert t.repo == "scrapers"
        assert t.ref == "1.4.0"

    def test_bare_owner_repo_defaults_to_github(self):
        t = parse_install_target("acme/iteration")
        assert isinstance(t, ForgeTarget)
        assert t.forge == "github"
        assert t.ref is None

    def test_with_subdirectory(self):
        t = parse_install_target(
            "github:acme/big-monorepo@main#subdirectory=tools/nodes"
        )
        assert isinstance(t, ForgeTarget)
        assert t.subdirectory == "tools/nodes"
        assert t.ref == "main"

    def test_ref_can_contain_slash(self):
        # Branch names like `release/v1` should round-trip.
        t = parse_install_target("github:acme/iteration@release/v1")
        assert isinstance(t, ForgeTarget)
        assert t.ref == "release/v1"


class TestParseRejection:
    def test_empty(self):
        with pytest.raises(ValueError, match="empty"):
            parse_install_target("")

    def test_pypi_with_slash_falls_through(self):
        # `foo/bar[extras]` looks pypi-ish (brackets); regex won't match
        # because of the trailing `[`, so it falls through to PyPI and `uv add`
        # is left to complain.
        t = parse_install_target("foo/bar[extras]")
        assert isinstance(t, PyPITarget)


class TestForgeTarballUrl:
    def test_github(self):
        t = ForgeTarget(forge="github", owner="acme", repo="iteration")
        assert (
            t.tarball_url("abc123")
            == "https://codeload.github.com/acme/iteration/tar.gz/abc123"
        )

    def test_gitlab(self):
        t = ForgeTarget(forge="gitlab", owner="acme", repo="scrapers")
        assert (
            t.tarball_url("deadbeef")
            == "https://gitlab.com/acme/scrapers/-/archive/deadbeef/scrapers-deadbeef.tar.gz"
        )


class TestForgeUvAddArgs:
    def test_resolves_and_returns_tarball_url(self, monkeypatch):
        url = "https://api.github.com/repos/acme/iteration/commits/v2.0.0"
        _patch_http_get(monkeypatch, {url: {"sha": "abc123"}})

        t = ForgeTarget(forge="github", owner="acme", repo="iteration", ref="v2.0.0")
        assert t.uv_add_args == (
            "https://codeload.github.com/acme/iteration/tar.gz/abc123",
        )


def _patch_http_get(monkeypatch, responses: dict[str, dict]) -> list[str]:
    """Patch `_http_get` to return canned JSON responses; record calls."""
    calls: list[str] = []

    def fake(url: str) -> bytes:
        calls.append(url)
        return json.dumps(responses[url]).encode()

    monkeypatch.setattr(install_target_module, "_http_get", fake)
    return calls


class TestResolveForgeRef:
    def test_github_explicit_ref(self, monkeypatch):
        url = "https://api.github.com/repos/acme/iteration/commits/v2.0.0"
        calls = _patch_http_get(monkeypatch, {url: {"sha": "3f2a9c1d8e0b"}})

        target = ForgeTarget(
            forge="github", owner="acme", repo="iteration", ref="v2.0.0"
        )
        assert resolve_forge_ref(target) == "3f2a9c1d8e0b"
        assert calls == [url]

    def test_github_default_branch(self, monkeypatch):
        _patch_http_get(
            monkeypatch,
            {
                "https://api.github.com/repos/acme/iteration": {
                    "default_branch": "trunk"
                },
                "https://api.github.com/repos/acme/iteration/commits/trunk": {
                    "sha": "deadbeef"
                },
            },
        )

        target = ForgeTarget(forge="github", owner="acme", repo="iteration")
        assert resolve_forge_ref(target) == "deadbeef"

    def test_github_url_encodes_ref(self, monkeypatch):
        # `/` in the ref must be encoded so it doesn't get treated as a path
        # segment in the API URL.
        url = "https://api.github.com/repos/acme/iteration/commits/release%2Fv1"
        calls = _patch_http_get(monkeypatch, {url: {"sha": "abc"}})

        target = ForgeTarget(
            forge="github", owner="acme", repo="iteration", ref="release/v1"
        )
        resolve_forge_ref(target)
        assert calls == [url]

    def test_gitlab_explicit_ref(self, monkeypatch):
        url = (
            "https://gitlab.com/api/v4/projects/acme%2Fscrapers"
            "/repository/commits/1.4.0"
        )
        _patch_http_get(monkeypatch, {url: {"id": "cafef00d"}})

        target = ForgeTarget(forge="gitlab", owner="acme", repo="scrapers", ref="1.4.0")
        assert resolve_forge_ref(target) == "cafef00d"

    def test_gitlab_default_branch(self, monkeypatch):
        _patch_http_get(
            monkeypatch,
            {
                "https://gitlab.com/api/v4/projects/acme%2Fscrapers": {
                    "default_branch": "main"
                },
                "https://gitlab.com/api/v4/projects/acme%2Fscrapers/repository/commits/main": {
                    "id": "feedface"
                },
            },
        )

        target = ForgeTarget(forge="gitlab", owner="acme", repo="scrapers")
        assert resolve_forge_ref(target) == "feedface"
