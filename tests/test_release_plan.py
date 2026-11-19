"""Release gating without network access or publishing."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "release_plan.py"
SPEC = importlib.util.spec_from_file_location("release_plan", SCRIPT)
assert SPEC and SPEC.loader
release_plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_plan)


@pytest.fixture
def release_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "src/workflow_engine").mkdir(parents=True)
    (tmp_path / "plugins/wengine/.claude-plugin").mkdir(parents=True)
    (tmp_path / "plugins/wengine/skills/wengine/scripts").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "aceteam-workflow-engine"\nversion = "2.0.0rc19"\n'
    )
    (tmp_path / "src/workflow_engine/__init__.py").write_text(
        '__version__ = "2.0.0rc19"\n'
    )
    (tmp_path / "plugins/wengine/.claude-plugin/plugin.json").write_text(
        json.dumps({"version": "2.0.0rc19"})
    )
    (tmp_path / "plugins/wengine/skills/wengine/scripts/wengine.sh").write_text(
        'WENGINE_PINNED_VERSION="2.0.0rc19"\n'
    )
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "aceteam-workflow-engine"\nversion = "2.0.0rc19"\n'
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "## [Unreleased]\n\n## [2.0.0rc20] - 2026-11-19\n\nFixed append.\n"
        "\n## [2.0.0rc19] - 2026-09-26\n\nPrevious release.\n"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_version_sites_and_changelog_fail_closed(release_tree: Path) -> None:
    assert release_plan.version_sites() == "2.0.0rc19"
    (release_tree / "plugins/wengine/.claude-plugin/plugin.json").write_text(
        '{"version": "2.0.0rc18"}'
    )
    with pytest.raises(SystemExit, match="Version sites disagree"):
        release_plan.version_sites()
    with pytest.raises(SystemExit, match=r"lacks ## \[2.0.0rc21\]"):
        release_plan.changelog_section("2.0.0rc21")


def test_main_push_without_version_bump_is_noop(
    release_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        release_plan.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, (release_tree / "pyproject.toml").read_text(), ""
        ),
    )
    assert release_plan.plan(
        "push", "a" * 40, "b" * 40, "aceteam-ai/workflow-engine", "test"
    ) == {
        "run": "false",
        "published": "false",
        "version": "",
    }


def test_existing_release_is_noop_only_at_this_commit(
    release_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(release_plan, "checked_tag", lambda version, sha: True)
    monkeypatch.setattr(release_plan, "pypi_metadata", lambda version: {"urls": []})
    monkeypatch.setattr(release_plan, "request_json", lambda url, token: {"id": 1})
    assert release_plan.plan(
        "workflow_dispatch", "", "b" * 40, "aceteam-ai/workflow-engine", "test"
    ) == {"run": "false", "published": "true", "version": "2.0.0rc19"}

    def conflicting_tag(version: str, sha: str) -> bool:
        raise SystemExit("Existing tag points to another commit")

    monkeypatch.setattr(release_plan, "checked_tag", conflicting_tag)
    with pytest.raises(SystemExit, match="another commit"):
        release_plan.plan(
            "workflow_dispatch", "", "b" * 40, "aceteam-ai/workflow-engine", "test"
        )


def test_bump_requires_changelog_and_updates_four_sources(release_tree: Path) -> None:
    with pytest.raises(SystemExit, match=r"CHANGELOG\.md lacks"):
        release_plan.bump("2.0.0rc21")
    assert release_plan.version_sites() == "2.0.0rc19"
    assert release_plan.bump("--rc") == "2.0.0rc20"
    assert "2.0.0rc20" in (release_tree / "pyproject.toml").read_text()
    assert "2.0.0rc20" in (release_tree / "src/workflow_engine/__init__.py").read_text()
    assert (
        "2.0.0rc20"
        in (release_tree / "plugins/wengine/.claude-plugin/plugin.json").read_text()
    )
    assert (
        "2.0.0rc20"
        in (
            release_tree / "plugins/wengine/skills/wengine/scripts/wengine.sh"
        ).read_text()
    )
    assert release_plan.version_order("2.0.0") > release_plan.version_order("2.0.0rc20")


@pytest.mark.parametrize(
    ("flag", "target"),
    [
        ("--rc", "2.0.0rc20"),
        ("--patch", "2.0.1"),
        ("--minor", "2.1.0"),
        ("--major", "3.0.0"),
    ],
)
def test_bump_script_accepts_flag_forms(
    release_tree: Path, monkeypatch: pytest.MonkeyPatch, flag: str, target: str
) -> None:
    scripts = release_tree / "scripts"
    scripts.mkdir()
    shutil.copy2(SCRIPT, scripts / SCRIPT.name)
    shutil.copy2(SCRIPT.parent / "bump-version.sh", scripts / "bump-version.sh")
    if target != "2.0.0rc20":
        with (release_tree / "CHANGELOG.md").open("a") as changelog:
            changelog.write(f"\n## [{target}] - 2026-11-19\n\nRelease note.\n")

    # Stub only the lock refresh. The script and release_plan.py run as real CLIs.
    bin_dir = release_tree / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env python3\n"
        "import re\n"
        "import sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1:] != ['lock']:\n"
        "    raise SystemExit(2)\n"
        'version = re.search(r\'^version = "([^"]+)"$\', '
        "Path('pyproject.toml').read_text(), re.M).group(1)\n"
        "lock = Path('uv.lock')\n"
        'lock.write_text(re.sub(r\'(?m)^version = "[^"]+"$\', '
        "f'version = \"{version}\"', lock.read_text(), count=1))\n"
    )
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    result = subprocess.run(
        ["bash", str(scripts / "bump-version.sh"), flag],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert release_plan.version_sites() == target


def test_published_artifacts_must_match_local_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version = "2.0.0rc20"
    wheel = tmp_path / f"aceteam_workflow_engine-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("workflow_engine/__init__.py", "")
        archive.writestr("aceteam_workflow_engine-2.0.0rc20.dist-info/METADATA", "")
    sdist = tmp_path / f"aceteam_workflow_engine-{version}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name in ["workflow_engine/__init__.py", "PKG-INFO"]:
            data = b"x"
            info = tarfile.TarInfo(f"aceteam_workflow_engine-{version}/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    urls = [
        {
            "filename": path.name,
            "digests": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            "url": f"https://files.pythonhosted.org/{path.name}",
        }
        for path in [wheel, sdist]
    ]
    monkeypatch.setattr(release_plan, "pypi_metadata", lambda _: {"urls": urls})
    release_plan.verify_dist(version, tmp_path, published=True)
    published_archive = sdist.read_bytes()
    with tarfile.open(sdist, "w:gz") as archive:
        for name in ["workflow_engine/__init__.py", "PKG-INFO"]:
            data = b"x"
            info = tarfile.TarInfo(f"aceteam_workflow_engine-{version}/{name}")
            info.size = len(data)
            info.mtime = 42
            archive.addfile(info, io.BytesIO(data))
    assert sdist.read_bytes() != published_archive

    class Response(io.BytesIO):
        url = f"https://files.pythonhosted.org/{sdist.name}"

    monkeypatch.setattr(
        release_plan.urllib.request,
        "urlopen",
        lambda url, timeout: Response(published_archive),
    )
    release_plan.verify_dist(version, tmp_path, published=True)
    urls[0]["digests"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="wheel does not match"):
        release_plan.verify_dist(version, tmp_path, published=True)
