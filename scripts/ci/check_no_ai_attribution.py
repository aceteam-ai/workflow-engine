"""Guard against AI attribution reaching a commit or a pull request.

This repository is public and carries no AI attribution, by policy. This
script makes that policy checkable instead of relying on convention. It has
two modes:

- Text mode (``--kind commit`` or ``--kind description``): scans a block of
  text, such as a full commit message, a pull request description, or a pull
  request title, for a fixed set of forbidden patterns (trailers, links,
  footers) and reports every match.
- Identity mode (``--kind identity``): scans a git identity value, such as
  ``%an``/``%ae``/``%cn``/``%ce`` from ``git log``, for the same vendor names
  used above, one value per line.

It is invoked from both the CI workflow and a local ``commit-msg`` git hook,
so a bad message is rejected before the commit object exists, not only after
it reaches a pull request.

Usage:
    # From stdin (used by CI, and the simplest for scripting):
    echo "$TEXT" | uv run python scripts/ci/check_no_ai_attribution.py --label "commit abc123"

    # From a file (used by the commit-msg hook, which is handed a path):
    uv run python scripts/ci/check_no_ai_attribution.py --label "commit message" path/to/COMMIT_EDITMSG

    # Identity check, one value per invocation:
    git log -1 --format=%ae "$sha" | uv run python scripts/ci/check_no_ai_attribution.py --kind identity --label "commit $sha author email"

Exit status is non-zero if any forbidden pattern is found.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

REASON = "this repository is public and carries no AI attribution"

# Names of AI coding assistants and vendors whose attribution should never
# reach a commit or pull request in this repository, whether in a trailer or
# in the author/committer identity. Adding a vendor is a one-line change:
# append its name here and every pattern below that references
# VENDOR_ALTERNATION picks it up automatically.
#
# Some of these are also plausible human names or words (Devin is a common
# first name, Cursor and Gemini are ordinary words). That is a known,
# accepted tradeoff: a human contributor named Devin, or a commit that
# legitimately mentions a product called Cursor in a trailer position, would
# be rejected and need a reword. We are not attempting to distinguish those
# cases, only to keep the common accidental-leak case out of a public repo.
#
# This also does not attempt to catch deliberate evasion, such as homoglyphs
# (e.g. a Cyrillic "a" in "Claude") or a trailer folded across a line
# continuation. Both require intent to construct, and defending against them
# is a different, much harder problem than catching an accidental paste.
# Out of scope by decision, not by oversight.
VENDOR_NAMES: tuple[str, ...] = (
    "claude",
    "anthropic",
    "copilot",
    "codex",
    "openai",
    "cursor",
    "devin",
    "gemini",
    "chatgpt",
)

_VENDOR_ALTERNATION = "|".join(re.escape(name) for name in VENDOR_NAMES)


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
#
# Pattern names are deliberately written without the literal forbidden
# substring they detect, so that a failure message which quotes a pattern's
# own name (for example when a PR description quotes prior CI output) cannot
# itself trip the guard on a second pass.
PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        name="Claude session or share link",
        regex=re.compile(r"claude\.ai/", re.IGNORECASE),
    ),
    Pattern(
        name="Claude-Session trailer",
        regex=re.compile(r"^\s*claude-session\s*:", re.IGNORECASE),
    ),
    Pattern(
        name="Co-Authored-By trailer naming an AI assistant",
        regex=re.compile(
            rf"^\s*co-authored-by\s*:.*\b({_VENDOR_ALTERNATION})\b", re.IGNORECASE
        ),
    ),
    Pattern(
        name="Signed-off-by trailer naming an AI assistant",
        regex=re.compile(
            rf"^\s*signed-off-by\s*:.*\b({_VENDOR_ALTERNATION})\b", re.IGNORECASE
        ),
    ),
    Pattern(
        name="Authored-by trailer naming an AI assistant",
        regex=re.compile(
            rf"^\s*authored-by\s*:.*\b({_VENDOR_ALTERNATION})\b", re.IGNORECASE
        ),
    ),
    Pattern(
        name="AI attribution footer",
        regex=re.compile(
            r"generated\s+(?:with|by)\s*\[?\s*claude(?:\s+code)?\b", re.IGNORECASE
        ),
    ),
    Pattern(
        name="Claude Code product link",
        regex=re.compile(r"claude\.com/claude-code", re.IGNORECASE),
    ),
)

# Used only in identity mode: a bare vendor-name match against a git
# author/committer name or email field. Deliberately not anchored to a
# trailer prefix, since a name or email field has no such structure.
_IDENTITY_VENDOR_RE = re.compile(rf"\b({_VENDOR_ALTERNATION})\b", re.IGNORECASE)


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


def find_identity_violations(text: str) -> list[Violation]:
    """Return every vendor-name match found in a git identity value.

    `text` is expected to be one or more identity fields, one per line (for
    example `%an`, `%ae`, `%cn`, `%ce` from `git log --format`).
    """
    violations: list[Violation] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _IDENTITY_VENDOR_RE.search(stripped):
            violations.append(
                Violation(
                    pattern_name="commit identity naming an AI assistant",
                    line=stripped,
                )
            )
    return violations


def format_violation(*, label: str, kind: str, violation: Violation) -> str:
    if kind == "commit":
        fix = "Reword the commit message with `git rebase`."
    elif kind == "identity":
        fix = (
            "Fix the git author/committer identity (`git config user.name` / "
            "`user.email`) and rewrite history with `git rebase`."
        )
    else:
        fix = "Edit the pull request description or title to remove it."
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
        choices=("commit", "description", "identity"),
        default="commit",
        help="Selects the check performed and the fix instructions in "
        "failure messages. 'identity' checks git author/committer name and "
        "email values instead of trailer/link/footer patterns.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.path is not None:
        with open(args.path, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    if args.kind == "identity":
        violations = find_identity_violations(text)
    else:
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
