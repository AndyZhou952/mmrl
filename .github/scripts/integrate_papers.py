#!/usr/bin/env python3
"""
Agentic monthly integrator for the diffusion-rl-survey.

Pipeline (run after, or instead of, fetch_papers.py):
  1. Reuse the arXiv sweep + dedup from fetch_papers.py to get fresh candidates.
  2. For each candidate, use an **OpenAI-compatible** LLM (same env as
     fetch_papers.py: LLM_API_KEY / LLM_BASE_URL / LLM_MODEL) to:
       a. CLASSIFY per INTEGRATION_GUIDE §1 (scope) and §2 (paradigm +
          dedicated-page-vs-academia). The guide text is fed to the model so the
          decision *is* the guideline. Default to academia.md when unsure.
       b. GENERATE either an academia.md entry (§8) or a full dedicated page (§4),
          plus the §7 cross-file insertions (INDEX row, README tree line,
          READING_GUIDE row). The model controls *what*; this script controls
          *where* (fixed anchors), and the math linter + human PR review are the
          safety net.
  3. Splice the changes in, run the math linter (--fix), and write
     integration_summary.md for the PR body.

This script **only edits files**; committing and opening the PR is done by the
workflow (so the bot never pushes to a protected branch directly).

No LLM_API_KEY  → exits 2 (the workflow then falls back to opening a triage issue).
Stdlib only (urllib, json, re) plus imports from fetch_papers.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

# Reuse the sweep's fetch + dedup so there is a single source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_papers import arxiv_search, tracked_ids, QUERY_TERMS, GEN_RE, RL_RE  # noqa: E402

import datetime as _dt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GUIDE = (ROOT / "INTEGRATION_GUIDE.md").read_text(encoding="utf-8")
NOTATION = (ROOT / "papers/NOTATION.md").read_text(encoding="utf-8")
PARADIGM_DIR = {"policy_gradient": "papers/policy_gradient", "direct_preference": "papers/direct_preference"}


# --------------------------------------------------------------------------- #
# OpenAI-compatible chat helper
# --------------------------------------------------------------------------- #

def chat(system: str, user: str, max_tokens: int = 4000) -> str:
    key = os.environ["LLM_API_KEY"]
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"]


def chat_json(system: str, user: str, max_tokens: int = 4000) -> dict:
    """Chat call that must return a single JSON object (tolerates code fences)."""
    text = chat(system, user, max_tokens)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in model reply: {text[:200]}")
    return json.loads(m.group(0))


# --------------------------------------------------------------------------- #
# arXiv full-text (best effort) for richer dedicated-page context
# --------------------------------------------------------------------------- #

def arxiv_fulltext(arxiv_id: str, limit: int = 16000) -> str:
    for url in (f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
                f"https://arxiv.org/abs/{arxiv_id}"):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                html = resp.read().decode("utf-8", "ignore")
            text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            if len(text) > 500:
                return text[:limit]
        except Exception:  # noqa: BLE001
            continue
    return ""


# --------------------------------------------------------------------------- #
# Classification + generation prompts (guideline is the spec)
# --------------------------------------------------------------------------- #

CLASSIFY_SYS = (
    "You are a maintainer of a survey repository of RL algorithms for image/video "
    "diffusion & flow-matching generation. You apply the repository's INTEGRATION_GUIDE "
    "exactly. Be conservative: when unsure about scope, exclude; when unsure about "
    "dedicated-vs-academia, choose academia (the guide says so)."
)


def classify(cand: dict) -> dict:
    user = (
        "INTEGRATION_GUIDE.md (authoritative):\n\n" + GUIDE + "\n\n"
        "Candidate paper:\n"
        f"- arXiv: {cand['id']}\n- Title: {cand['title']}\n- Abstract: {cand['abstract']}\n\n"
        "Decide per §1 (scope) and §2 (paradigm + dedicated-file-vs-academia). "
        "Reply with ONLY this JSON:\n"
        '{"in_scope": true/false, "placement": "academia"|"dedicated", '
        '"paradigm": "policy_gradient"|"direct_preference", '
        '"short_name": "lowercase_underscore", "reason": "<=25 words citing the guide rule"}'
    )
    return chat_json(CLASSIFY_SYS, user, max_tokens=400)


def gen_academia(cand: dict, decision: dict) -> dict:
    user = (
        "INTEGRATION_GUIDE.md (authoritative — follow §8 exactly):\n\n" + GUIDE + "\n\n"
        "NOTATION.md (use these symbols; flow-matching-centric):\n\n" + NOTATION[:6000] + "\n\n"
        f"Paper: arXiv {cand['id']} ({cand['date']}) — {cand['title']}\n"
        f"Abstract: {cand['abstract']}\n"
        f"Paradigm: {decision['paradigm']}\n\n"
        "Produce ONLY this JSON (no prose):\n"
        '{"master_row": "| YYYY-MM | [NNNN.NNNNN](https://arxiv.org/abs/NNNN.NNNNN) | ShortName | '
        'Policy Gradient|Direct Preference | <=8-word problem |", '
        '"entry_md": "<the full ### detailed entry markdown per §8.2, GitHub-math-safe: '
        'use \\\\Vert{} not \\\\|, \\\\lbrace/\\\\rbrace not \\\\{ \\\\}, no \\\\, or \\\\; spacing>"}'
    )
    return chat_json("You write strictly guideline-conformant survey entries.", user, max_tokens=2000)


def gen_dedicated(cand: dict, decision: dict, example: str) -> dict:
    fulltext = arxiv_fulltext(cand["id"])
    user = (
        "INTEGRATION_GUIDE.md (authoritative — follow §4 section order, §4.6 math rules, "
        "§4.7 boxed loss, §4.8 incl. a condensed VeRL-Omni snippet ONLY if the loss is "
        "registered in diffusion_algos.py, and add a **Result** block per §4.5):\n\n" + GUIDE + "\n\n"
        "NOTATION.md:\n\n" + NOTATION[:6000] + "\n\n"
        "EXAMPLE of a conformant page (match this structure and tone):\n\n" + example[:6000] + "\n\n"
        f"Paper: arXiv {cand['id']} ({cand['date']}) — {cand['title']}\n"
        f"Abstract: {cand['abstract']}\n"
        f"Paradigm: {decision['paradigm']} → directory papers/{decision['paradigm']}/\n"
        f"Paper full text (truncated, best-effort): {fulltext}\n\n"
        "Produce ONLY this JSON (GitHub-math-safe LaTeX: \\\\Vert{} not \\\\|, \\\\lbrace/\\\\rbrace, "
        "no \\\\, \\\\; \\\\: spacing; blank line before every $$):\n"
        '{"short_name": "lowercase_underscore", '
        '"page_md": "<full markdown page per §4, starting with the # title line>", '
        '"index_row": "| **ShortName** | Full title | [NNNN.NNNNN](url) | YYYY-MM-DD | Venue | '
        '[paradigm/file.md](paradigm/file.md) |", '
        '"readme_tree_line": "    \\u2502   \\u251c\\u2500\\u2500 file.md   \\u2190 ShortName (Mon YYYY) \\u2014 one line", '
        '"reading_guide_row": "| **ShortName** | <keeps> | <changes> |"}'
    )
    return chat_json("You write strictly guideline-conformant survey pages.", user, max_tokens=6000)


# --------------------------------------------------------------------------- #
# Splicing helpers (code controls WHERE)
# --------------------------------------------------------------------------- #

def insert_academia(master_row: str, paradigm: str, entry_md: str) -> None:
    p = ROOT / "papers/academia.md"
    text = p.read_text(encoding="utf-8")
    # 1) master index table: append row at the end of the *contiguous* table block
    #    under "## Master Index" (don't scan to EOF — future entries may add tables).
    lines = text.split("\n")
    mi_start = next(i for i, ln in enumerate(lines) if ln.startswith("## Master Index"))
    first_row = next(i for i in range(mi_start, len(lines)) if lines[i].strip().startswith("|"))
    last_tbl = first_row
    while last_tbl + 1 < len(lines) and lines[last_tbl + 1].strip().startswith("|"):
        last_tbl += 1
    lines.insert(last_tbl + 1, master_row.strip())
    text = "\n".join(lines)
    # 2) detailed entry: append at the end of the matching section.
    header = ("## Policy Gradient Paradigm Advances" if paradigm == "policy_gradient"
              else "## Direct Preference Paradigm Advances")
    start = text.index(header)
    # next top-level section after the header (e.g. the other family, or Cross-Cutting Notes)
    nxt = re.search(r"\n## ", text[start + len(header):])
    insert_at = (start + len(header) + nxt.start()) if nxt else len(text)
    block = "\n\n---\n\n" + entry_md.strip() + "\n"
    text = text[:insert_at] + block + text[insert_at:]
    p.write_text(text, encoding="utf-8")


def insert_dedicated(decision: dict, gen: dict) -> str:
    short = gen["short_name"]
    paradigm = decision["paradigm"]
    rel = f"{PARADIGM_DIR[paradigm]}/{short}.md"
    (ROOT / rel).write_text(gen["page_md"].rstrip() + "\n", encoding="utf-8")

    # INDEX.md master table: append row at the end of the *contiguous* table block
    # under "## Master Table" (INDEX has other pipe tables later, so don't scan to EOF).
    idx = ROOT / "papers/INDEX.md"
    t = idx.read_text(encoding="utf-8")
    ml = t.split("\n")
    mt_start = ml.index("## Master Table")
    first_row = next(i for i in range(mt_start, len(ml)) if ml[i].strip().startswith("|"))
    last_row = first_row
    while last_row + 1 < len(ml) and ml[last_row + 1].strip().startswith("|"):
        last_row += 1
    ml.insert(last_row + 1, gen["index_row"].strip())
    idx.write_text("\n".join(ml), encoding="utf-8")

    # README.md directory tree: add the line under the right subtree (best-effort).
    readme = ROOT / "README.md"
    r = readme.read_text(encoding="utf-8")
    anchor = "policy_gradient/" if paradigm == "policy_gradient" else "direct_preference/"
    rlines = r.split("\n")
    for i, ln in enumerate(rlines):
        if anchor in ln and "←" in ln:
            rlines.insert(i + 1, gen["readme_tree_line"])
            break
    readme.write_text("\n".join(rlines), encoding="utf-8")

    # READING_GUIDE.md Quick Reference table: append row at end of the last table.
    rg = ROOT / "papers/READING_GUIDE.md"
    g = rg.read_text(encoding="utf-8")
    glines = g.split("\n")
    last_tbl = max(i for i, ln in enumerate(glines) if ln.strip().startswith("|"))
    glines.insert(last_tbl + 1, gen["reading_guide_row"].strip())
    rg.write_text("\n".join(glines), encoding="utf-8")
    return rel


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=35)
    ap.add_argument("--max", type=int, default=5, help="max papers to integrate per run")
    ap.add_argument("--summary", default="integration_summary.md")
    args = ap.parse_args()

    if not os.environ.get("LLM_API_KEY"):
        print("LLM_API_KEY not set — skipping integration (workflow will open a triage issue).",
              file=sys.stderr)
        return 2

    since = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=args.days)
    known = tracked_ids()
    seen: dict[str, dict] = {}
    for term in QUERY_TERMS:
        try:
            for c in arxiv_search(term, since):
                blob = c["title"] + c["abstract"]
                if c["id"] in known or c["id"] in seen:
                    continue
                if GEN_RE.search(blob) and RL_RE.search(blob):
                    seen[c["id"]] = c
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] query '{term}' failed: {exc}", file=sys.stderr)

    cands = sorted(seen.values(), key=lambda c: c["date"], reverse=True)[: args.max]
    example_pg = (ROOT / "papers/policy_gradient/flow_grpo.md").read_text(encoding="utf-8")
    example_dp = (ROOT / "papers/direct_preference/awm.md").read_text(encoding="utf-8")

    log: list[str] = []
    added = 0
    for c in cands:
        try:
            d = classify(c)
        except Exception as exc:  # noqa: BLE001
            log.append(f"- `{c['id']}` {c['title']} — **classify failed**: {exc}")
            continue
        if not d.get("in_scope"):
            log.append(f"- `{c['id']}` {c['title']} — **skipped** (out of scope: {d.get('reason','')})")
            continue
        try:
            if d["placement"] == "dedicated":
                example = example_pg if d["paradigm"] == "policy_gradient" else example_dp
                gen = gen_dedicated(c, d, example)
                rel = insert_dedicated(d, gen)
                added += 1
                log.append(f"- `{c['id']}` {c['title']} — **dedicated page** `{rel}` "
                           f"({d['paradigm']}). _{d.get('reason','')}_ ⚠ review §4/§7 carefully.")
            else:
                gen = gen_academia(c, d)
                insert_academia(gen["master_row"], d["paradigm"], gen["entry_md"])
                added += 1
                log.append(f"- `{c['id']}` {c['title']} — **academia.md** "
                           f"({d['paradigm']}). _{d.get('reason','')}_")
        except Exception as exc:  # noqa: BLE001
            log.append(f"- `{c['id']}` {c['title']} — **generation/splice failed**: {exc}")

    summary = [
        f"## Automated paper integration — {_dt.datetime.now(_dt.timezone.utc):%Y-%m}",
        "",
        f"Integrated **{added}** paper(s) from the last {args.days} days, following "
        "`INTEGRATION_GUIDE.md`. **This is a draft for human review — verify against the "
        "§10 pre-commit checklist before merging.**",
        "",
        "### Decisions",
        *log,
        "",
        "### Reviewer checklist (INTEGRATION_GUIDE §10)",
        "- [ ] Scope (§1) and paradigm (§2) classification correct",
        "- [ ] Notation matches NOTATION.md; boxed objective present (dedicated pages)",
        "- [ ] Condensed VeRL-Omni snippet only if the loss is registered; diffs marked",
        "- [ ] **Result** block per Problem section (§4.5); numbers verified against the paper",
        "- [ ] Cross-references/links resolve; INDEX/README/READING_GUIDE updated (§7)",
        "- [ ] Math lint passes",
    ]
    Path(args.summary).write_text("\n".join(summary), encoding="utf-8")
    print(f"Integrated {added} paper(s); summary → {args.summary}")
    # Exit 0 with changes, 3 with none (workflow uses this to decide PR vs issue).
    return 0 if added else 3


if __name__ == "__main__":
    sys.exit(main())
