#!/usr/bin/env python3
"""
Notation-consistency check for the diffusion-rl-survey.

The shared notation lives in `papers/NOTATION.md`. Individual pages no longer
repeat a "> Notation:" header — instead this script asks an LLM to verify that
each page's math stays consistent with NOTATION.md: i.e. it never reuses a
canonical symbol with a different meaning, and any symbol it does introduce that
is *not* in NOTATION.md is defined locally on the page.

Provider-agnostic: it talks to any OpenAI-compatible chat/completions endpoint
(OpenAI, DeepSeek, MiniMax, MiMo, Gemini's OpenAI-compat layer, …) via three env
vars. Without a key the check is skipped and a notice is written, so CI never
hard-fails just because the secret is absent.

    LLM_API_KEY    bearer token (required to actually run the check)
    LLM_BASE_URL   default https://api.openai.com/v1
    LLM_MODEL      default gpt-4o-mini

Stdlib only (urllib, json, re) so it runs in CI with no extra deps.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTATION = ROOT / "papers" / "NOTATION.md"
# Pages that carry real math and should track the shared notation.
SCAN_DIRS = ("papers/policy_gradient", "papers/direct_preference")


def default_targets() -> list[Path]:
    out: list[Path] = []
    for d in SCAN_DIRS:
        out += sorted((ROOT / d).glob("*.md"))
    return out


def ask_llm(notation: str, page_name: str, page_text: str) -> list[dict] | None:
    """Return a list of {symbol, problem} issues, or None if the call fails."""
    key = os.environ.get("LLM_API_KEY")
    if not key:
        return None
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    system = (
        "You are a meticulous technical copy-editor checking mathematical "
        "notation consistency across a survey of RL algorithms for diffusion / "
        "flow-matching models. You are given the canonical notation file and one "
        "page. Flag ONLY genuine inconsistencies: (a) a symbol that NOTATION.md "
        "defines being used on the page with a clearly different meaning, or (b) a "
        "symbol used on the page that is neither in NOTATION.md nor defined "
        "locally on the page itself. Pages are allowed to introduce extra local "
        "symbols as long as they define them inline. Do not nitpick wording, "
        "formatting, or stylistic differences. Reply with ONLY a compact JSON "
        'object: {"issues": [{"symbol": "...", "problem": "<=25 words"}]}. '
        "Empty list if the page is consistent."
    )
    user = (
        f"=== NOTATION.md (canonical) ===\n{notation}\n\n"
        f"=== PAGE: {page_name} ===\n{page_text}"
    )
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
        text = data["choices"][0]["message"]["content"]
        parsed = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        return parsed.get("issues", [])
    except Exception as exc:  # noqa: BLE001 — advisory check, never hard-fail here
        print(f"[warn] LLM check failed for {page_name}: {exc}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="notation_report.md")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any inconsistency is found (default: advisory, exit 0)")
    ap.add_argument("files", nargs="*", help="pages to check (default: the math-heavy paper dirs)")
    args = ap.parse_args()

    targets = [Path(f) for f in args.files] if args.files else default_targets()
    notation = NOTATION.read_text(encoding="utf-8")

    lines = ["## Notation consistency check", ""]
    if not os.environ.get("LLM_API_KEY"):
        lines += [
            "_Skipped — no `LLM_API_KEY` set. Add the secret (and optionally "
            "`LLM_BASE_URL` / `LLM_MODEL` repo variables) to enable the AI check._",
        ]
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("LLM_API_KEY not set; notation check skipped.")
        return 0

    total = 0
    for page in targets:
        issues = ask_llm(notation, page.name, page.read_text(encoding="utf-8"))
        if not issues:
            continue
        total += len(issues)
        lines.append(f"### `{page.relative_to(ROOT)}`")
        for it in issues:
            lines.append(f"- **{it.get('symbol', '?')}** — {it.get('problem', '?')}")
        lines.append("")

    if total == 0:
        lines.append("No notation inconsistencies found against `NOTATION.md`.")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Checked {len(targets)} page(s); {total} potential inconsistency(ies). Report: {args.out}")
    return 1 if (args.strict and total) else 0


if __name__ == "__main__":
    sys.exit(main())
