#!/usr/bin/env python3
"""
Lint LaTeX math in Markdown files for GitHub rendering compatibility.

ROOT CAUSE
----------
GitHub's Markdown parser (cmark-gfm) applies CommonMark backslash-escape
processing to the content of $$...$$  math blocks before the math renderer
(KaTeX) ever sees it.  All ASCII punctuation is escapable in CommonMark
(! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \\ ] ^ _ { | } ~), so
any LaTeX command whose delimiter is one of those characters will have its
backslash silently stripped.

The critical failure mode is \\|  (double vertical bar / norm):
  \\|  →  |    (bare pipe triggers GitHub's pipe-table parser)
The surrounding $$…$$ block is then re-parsed as a Markdown paragraph
containing a pipe table, causing the entire formula to show as raw text.

Secondary issues:
  \\!  → !   (renders an exclamation mark before the next bracket; ugly)
  \\{  → {   (display brace becomes a LaTeX grouping brace; invisible)
  \\}  → }   (same)
  \\,  → ,   (spacing command becomes literal comma)
  t^*  → t*  (Markdown emphasis eats the asterisk)

FIXES
-----
  \\|              → \\Vert (or \\lVert / \\rVert for explicit left/right)
  \\!              → (remove)
  \\{  \\}         → \\lbrace  \\rbrace
  \\,  \\;  \\:    → (remove or accept cosmetic difference)
  ^*   ^+          → ^{\\ast}  ^{+}

USAGE
-----
  python lint_math.py [file1.md file2.md ...]   # check specific files
  python lint_math.py                            # check all *.md in tree

Exit code: 0 if no errors, 1 if any ERROR-level issues found.
"""

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Pattern definitions
# Each entry: (regex, severity, human-readable message)
# Severity: 'ERROR' causes non-zero exit; 'WARN' is advisory only.
# ---------------------------------------------------------------------------

CHECKS = [
    # -----------------------------------------------------------------------
    # ERRORs — these reliably break GitHub rendering
    # -----------------------------------------------------------------------
    (
        r'\\Vert|\\lVert|\\rVert',   # sentinel: already fixed — skip
        None, None,
    ),
    (
        r'\\\|',
        'ERROR',
        r'\\| will be stripped to bare |, which triggers the pipe-table parser '
        r'and breaks the math block. Use \\Vert (or \\lVert / \\rVert).',
    ),
    (
        r'\^[\*\+](?![{a-zA-Z0-9])',
        'ERROR',
        r'^* and ^+ without braces: * and + are CommonMark special chars. '
        r'Use ^{\\ast} or ^{+}.',
    ),
    # -----------------------------------------------------------------------
    # WARNings — degrade rendering quality but rarely cause full failure
    # -----------------------------------------------------------------------
    (
        r'\\!',
        'WARN',
        r'\\! (negative thin space) is a CommonMark escape; it will be '
        r'stripped to a literal !. Remove it — spacing is fine without it.',
    ),
    (
        r'\\[{]',
        'WARN',
        r'\\{ will be stripped to { (a grouping char, not a displayed brace). '
        r'Use \\lbrace to display a curly brace.',
    ),
    (
        r'\\[}]',
        'WARN',
        r'\\} will be stripped to } (a grouping char, not a displayed brace). '
        r'Use \\rbrace to display a curly brace.',
    ),
    (
        r'\\[,;:]',
        'WARN',
        r'\\,  \\;  \\: spacing commands are CommonMark escape sequences and '
        r'will be stripped to literal punctuation. Remove them.',
    ),
]

# Strip the sentinel entry (first tuple with severity=None)
CHECKS = [(pat, sev, msg) for pat, sev, msg in CHECKS if sev is not None]


# ---------------------------------------------------------------------------
# Math span extraction
# ---------------------------------------------------------------------------

def extract_math_spans(text: str):
    """
    Yield (line_number, math_content) for every math span in *text*.

    Handles:
      - Block math:  $$ ... $$ (single-line or multi-line)
      - Inline math: $ ... $   (single line only)

    Line numbers are 1-based and refer to the line where the span starts.
    """
    # Block math first (greedy over newlines)
    for m in re.finditer(r'\$\$(.*?)\$\$', text, re.DOTALL):
        line_no = text[:m.start()].count('\n') + 1
        yield line_no, m.group(1)

    # Collect block-math byte ranges to avoid double-counting
    block_ranges = set()
    for m in re.finditer(r'\$\$(.*?)\$\$', text, re.DOTALL):
        block_ranges.update(range(m.start(), m.end()))

    # Inline math: $ ... $ not already inside a block-math span
    for m in re.finditer(r'(?<!\$)\$([^$\n]{1,200}?)\$(?!\$)', text):
        if m.start() in block_ranges:
            continue
        line_no = text[:m.start()].count('\n') + 1
        yield line_no, m.group(1)


# ---------------------------------------------------------------------------
# Main lint logic
# ---------------------------------------------------------------------------

Issue = tuple  # (filepath, line_no, severity, message, snippet)


def lint_file(filepath: str) -> list[Issue]:
    text = Path(filepath).read_text(encoding='utf-8')
    issues: list[Issue] = []

    for line_no, math in extract_math_spans(text):
        for pattern, severity, message in CHECKS:
            if re.search(pattern, math):
                snippet = math.strip().replace('\n', ' ')[:100]
                issues.append((filepath, line_no, severity, message, snippet))
                # one report per check per span (avoid duplicate noise)

    return issues


def main() -> int:
    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        paths = [str(p) for p in Path('.').rglob('*.md')
                 if '.git' not in p.parts]

    all_issues: list[Issue] = []
    for path in paths:
        try:
            all_issues.extend(lint_file(path))
        except Exception as exc:
            print(f'ERROR reading {path}: {exc}', file=sys.stderr)

    if not all_issues:
        print('No math linting issues found.')
        return 0

    # Group by file for readable output
    by_file: dict[str, list[Issue]] = {}
    for issue in all_issues:
        by_file.setdefault(issue[0], []).append(issue)

    total_errors = 0
    total_warns = 0

    for filepath, issues in sorted(by_file.items()):
        for fp, line, severity, message, snippet in issues:
            # GitHub Actions annotation format
            annotation = 'error' if severity == 'ERROR' else 'warning'
            print(f'::{annotation} file={fp},line={line}::{severity}: {message}')
            print(f'  in: {snippet!r}')
            if severity == 'ERROR':
                total_errors += 1
            else:
                total_warns += 1

    print(
        f'\n{total_errors + total_warns} issues: '
        f'{total_errors} error(s), {total_warns} warning(s).'
    )
    return 1 if total_errors > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
