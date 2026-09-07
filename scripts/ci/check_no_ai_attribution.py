"""Guard against AI attribution reaching a commit message or PR description.

This repository is public and carries no AI attribution, by policy. This
script makes that policy checkable in CI instead of relying on convention:
it scans a single piece of text (a commit message, or a pull request
description) for a fixed set of forbidden patterns and reports every match.

Usage:
    # From stdin (used by CI, and the simplest for scripting):
    echo "$TEXT" | uv run python scripts/ci/check_no_ai_attribution.py --label "commit abc123"

    # From a file:
    uv run python scripts/ci/check_no_ai_attribution.py --label "pull request description" path/to/file.txt

Exit status is non-zero if any forbidden pattern is found.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

REASON = "this repository is public and carries no AI attribution"


@dataclass(frozen=True)
class Pattern:
    name: str
    regex: re.Pattern[str]


@dataclass(frozen=True)
class Violation:
    pattern_name: str
    line: str


# Each pattern is matched line-by-line, case-insensitively, against the
# supplied text. Patterns target trailers, links, and footers specifically,
# not the words "Claude" or "Anthropic" on their own, so that prose mentioning
# either in a normal sentence, or a legitimate human Co-Authored-By trailer,
# is left alone.
PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        name="Claude Code session link",
        regex=re.compile(r"claude\.ai/code", re.IGNORECASE),
    ),
    Pattern(
        name="Claude-Session trailer",
        regex=re.compile(r"^\s*claude-session\s*:", re.IGNORECASE),
    ),
    Pattern(
        name="Co-Authored-By trailer naming Claude or Anthropic",
        regex=re.compile(
            r"^\s*co-authored-by\s*:.*\b(claude|anthropic)\b", re.IGNORECASE
        ),
    ),
    Pattern(
        # Named without the literal forbidden phrase so this pattern's own
        # name, echoed back in a failure message, cannot itself trip the
        # guard when that message is quoted (e.g. in a PR description).
        name="AI attribution footer",
        regex=re.compile(
            r"generated\s+with\s*\[?\s*claude\s+code\s*\]?", re.IGNORECASE
        ),
    ),
    Pattern(
        # Same reasoning: avoid the literal dotted domain in the name.
        name="Claude Code product link",
        regex=re.compile(r"claude\.com/claude-code", re.IGNORECASE),
    ),
)


def find_violations(text: str) -> list[Violation]:
    """Return every forbidden-pattern match found in `text`, line by line."""
    violations: list[Violation] = []
    for line in text.splitlines():
        for pattern in PATTERNS:
            if pattern.regex.search(line):
                violations.append(
                    Violation(pattern_name=pattern.name, line=line.strip())
                )
    return violations


def format_violation(*, label: str, kind: str, violation: Violation) -> str:
    if kind == "commit":
        fix = "Reword the commit message with `git rebase`."
    else:
        fix = "Edit the pull request description to remove it."
    return (
        f"AI attribution guard: forbidden pattern found in {label}.\n"
        f"  Pattern: {violation.pattern_name}\n"
        f"  Matching line: {violation.line}\n"
        f"  Fix: {fix}\n"
        f"  Reason: {REASON}."
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="File to read the text from. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--label",
        default="input",
        help='What to call the source in failure messages, e.g. "commit <sha>" '
        'or "pull request description".',
    )
    parser.add_argument(
        "--kind",
        choices=("commit", "description"),
        default="commit",
        help="Selects the fix instructions in failure messages.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.path is not None:
        with open(args.path, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    violations = find_violations(text)
    if not violations:
        return 0

    for violation in violations:
        print(
            format_violation(label=args.label, kind=args.kind, violation=violation),
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
