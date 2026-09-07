"""Tests for scripts/ci/check_no_ai_attribution.py.

The guard is a standalone script (not part of the installed package), so we
import it directly by path rather than through the `workflow_engine` package.
"""

import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ci"
    / "check_no_ai_attribution.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_no_ai_attribution", _SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


guard = _load_module()


@pytest.mark.unit
def test_empty_input_passes() -> None:
    assert guard.find_violations("") == []


@pytest.mark.unit
def test_clean_commit_message_passes() -> None:
    text = "fix: correct off-by-one error in topological sort\n\nCloses #123."
    assert guard.find_violations(text) == []


@pytest.mark.unit
def test_legitimate_human_coauthor_is_not_caught() -> None:
    text = (
        "feat: add retry policy to node execution\n\n"
        "Co-Authored-By: Jane Doe <jane@example.com>"
    )
    assert guard.find_violations(text) == []


@pytest.mark.unit
def test_prose_mentioning_claude_is_not_caught() -> None:
    text = "docs: note that this engine works well with Claude and other models via Anthropic's API"
    assert guard.find_violations(text) == []


@pytest.mark.unit
def test_session_link_is_caught() -> None:
    text = "fix: patch validation bug\n\nSee https://claude.ai/code/session/abc123 for context."
    violations = guard.find_violations(text)
    assert len(violations) == 1
    assert violations[0].pattern_name == "Claude Code session link"


@pytest.mark.unit
def test_claude_session_trailer_is_caught() -> None:
    text = "fix: patch validation bug\n\nClaude-Session: https://example.com/s/abc123"
    violations = guard.find_violations(text)
    assert len(violations) == 1
    assert violations[0].pattern_name == "Claude-Session trailer"


@pytest.mark.unit
def test_coauthored_by_claude_is_caught() -> None:
    text = "fix: patch validation bug\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
    violations = guard.find_violations(text)
    assert len(violations) == 1
    assert (
        violations[0].pattern_name
        == "Co-Authored-By trailer naming Claude or Anthropic"
    )


@pytest.mark.unit
def test_coauthored_by_anthropic_is_caught() -> None:
    text = (
        "fix: patch validation bug\n\nCo-Authored-By: Anthropic Bot <bot@anthropic.com>"
    )
    violations = guard.find_violations(text)
    assert len(violations) == 1
    assert (
        violations[0].pattern_name
        == "Co-Authored-By trailer naming Claude or Anthropic"
    )


@pytest.mark.unit
def test_generated_with_claude_code_footer_is_caught() -> None:
    text = "fix: patch validation bug\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)"
    violations = guard.find_violations(text)
    names = {v.pattern_name for v in violations}
    assert "AI attribution footer" in names


@pytest.mark.unit
def test_claude_code_link_is_caught() -> None:
    text = "fix: patch validation bug\n\nMore info: https://claude.com/claude-code"
    violations = guard.find_violations(text)
    assert len(violations) == 1
    assert violations[0].pattern_name == "Claude Code product link"


@pytest.mark.unit
def test_matching_is_case_insensitive() -> None:
    text = "fix: patch bug\n\nCO-AUTHORED-BY: CLAUDE <noreply@ANTHROPIC.com>"
    violations = guard.find_violations(text)
    assert len(violations) == 1


@pytest.mark.unit
def test_main_reads_stdin_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    text = "Co-Authored-By: Claude <noreply@anthropic.com>"
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))

    exit_code = guard.main(["--label", "commit deadbeef", "--kind", "commit"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "commit deadbeef" in captured.err
    assert "Co-Authored-By: Claude <noreply@anthropic.com>" in captured.err
    assert "git rebase" in captured.err
    assert "this repository is public and carries no AI attribution" in captured.err


@pytest.mark.unit
def test_main_reads_stdin_and_passes_when_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "fix: correct off-by-one error in topological sort"
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))

    exit_code = guard.main(["--label", "commit cafebabe", "--kind", "commit"])

    assert exit_code == 0


@pytest.mark.unit
def test_main_reads_from_file_path(tmp_path: Path) -> None:
    path = tmp_path / "pr_body.txt"
    path.write_text("Generated with [Claude Code](https://claude.com/claude-code)")

    exit_code = guard.main(
        ["--label", "pull request description", "--kind", "description", str(path)]
    )

    assert exit_code == 1
