#!/usr/bin/env python3
"""
Monthly paper sweep for the diffusion-rl-survey.

Queries the arXiv API for recent papers on RL fine-tuning of image/video
diffusion / flow-matching models, de-duplicates against the arXiv IDs already
tracked in papers/INDEX.md and papers/academia.md, and writes a Markdown
candidate list (for a GitHub issue) to the path given by --out (default:
candidates.md).

If an LLM key is configured, each candidate is given a one-line "problem
statement" and a suggested paradigm (Policy Gradient / Direct Preference) via an
**OpenAI-compatible** chat/completions call, so any provider works — OpenAI,
DeepSeek, Google Gemini (OpenAI-compat endpoint), MiMo, Anthropic (OpenAI-compat
endpoint), OpenRouter, a self-hosted vLLM, etc. Configured by env:

    LLM_API_KEY   the key (required to enable the AI step; if unset → skipped)
    LLM_BASE_URL  OpenAI-compatible base URL (default: https://api.openai.com/v1)
    LLM_MODEL     model id (default: gpt-4o-mini)

Without the key the AI step is skipped and the raw arXiv abstract snippet is used
instead — the script always produces a usable list.

Stdlib only (urllib, xml, re, json) so it runs in CI with no extra deps.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

ARXIV_API = "http://export.arxiv.org/api/query"

# Search scope: RL alignment of image/video diffusion & flow-matching models.
QUERY_TERMS = [
    "GRPO", "flow matching reinforcement learning", "diffusion reinforcement learning",
    "reward fine-tuning diffusion", "preference optimization diffusion",
    "policy optimization image generation", "RLHF text-to-image", "RLHF text-to-video",
]
CATEGORIES = ("cs.CV", "cs.LG", "cs.AI")

# Heuristic relevance filter: title/abstract must mention generation AND an RL signal.
GEN_RE = re.compile(r"\b(image|video|text-to-image|text-to-video|t2i|t2v|diffusion|flow[- ]match)", re.I)
RL_RE = re.compile(r"\b(reinforc|grpo|ppo|dpo|reward|preference|policy gradient|alignment|rlhf)", re.I)


def tracked_ids() -> set[str]:
    """Collect arXiv IDs already present in INDEX.md and academia.md."""
    ids: set[str] = set()
    root = Path(__file__).resolve().parents[2]
    for rel in ("papers/INDEX.md", "papers/academia.md"):
        p = root / rel
        if p.exists():
            ids.update(re.findall(r"(\d{4}\.\d{4,5})", p.read_text(encoding="utf-8")))
    return ids


def arxiv_search(term: str, since: datetime, max_results: int = 30) -> list[dict]:
    cat = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    search = f'(abs:"{term}") AND ({cat})'
    params = {
        "search_query": search,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": str(max_results),
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        raw = resp.read()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(raw).findall("a:entry", ns):
        published = e.find("a:published", ns).text.strip()
        dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        if dt < since:
            continue
        arxiv_id = e.find("a:id", ns).text.strip().rsplit("/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        out.append({
            "id": arxiv_id,
            "title": " ".join(e.find("a:title", ns).text.split()),
            "abstract": " ".join(e.find("a:summary", ns).text.split()),
            "date": dt.date().isoformat(),
        })
    return out


def ai_annotate(cand: dict) -> dict | None:
    """Optional: draft a one-line problem + paradigm via any OpenAI-compatible API.

    Provider is chosen entirely by env vars (LLM_API_KEY / LLM_BASE_URL /
    LLM_MODEL), so the same code path works for OpenAI, DeepSeek, Gemini's
    OpenAI-compat endpoint, MiMo, Anthropic's OpenAI-compat endpoint, OpenRouter,
    self-hosted vLLM, etc. Best-effort: any failure returns None and the caller
    falls back to the raw abstract.
    """
    key = os.environ.get("LLM_API_KEY")
    if not key:
        return None
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    prompt = (
        "You are triaging a paper for a survey of RL algorithms for image/video "
        "diffusion/flow generation. Given the title and abstract, reply with ONLY a "
        "compact JSON object: {\"problem\": \"<=15 words\", \"paradigm\": "
        "\"Policy Gradient|Direct Preference|unsure\", \"relevant\": true|false}. "
        "Policy Gradient = PPO-clip over the trajectory; Direct Preference = "
        "preference/MSE loss on final samples.\n\n"
        f"Title: {cand['title']}\nAbstract: {cand['abstract']}"
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 200,
    }).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        text = data["choices"][0]["message"]["content"]
        return json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception as exc:  # noqa: BLE001 — AI step is best-effort
        print(f"[warn] AI annotation failed: {exc}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=35, help="look-back window")
    ap.add_argument("--out", default="candidates.md")
    args = ap.parse_args()

    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    known = tracked_ids()

    seen: dict[str, dict] = {}
    for term in QUERY_TERMS:
        try:
            for c in arxiv_search(term, since):
                if c["id"] in known or c["id"] in seen:
                    continue
                if not (GEN_RE.search(c["title"] + c["abstract"]) and RL_RE.search(c["title"] + c["abstract"])):
                    continue
                seen[c["id"]] = c
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] query '{term}' failed: {exc}", file=sys.stderr)
        time.sleep(3)  # be polite to the arXiv API

    cands = sorted(seen.values(), key=lambda c: c["date"], reverse=True)

    lines = [
        f"## Monthly paper sweep — {datetime.now(timezone.utc):%Y-%m}",
        "",
        f"{len(cands)} candidate(s) from the last {args.days} days not yet in "
        "`INDEX.md` / `academia.md`. Triage each per `INTEGRATION_GUIDE.md` §1 "
        "(scope) and §2 (paradigm + dedicated-file-vs-academia).",
        "",
    ]
    for c in cands:
        ann = ai_annotate(c)
        lines.append(f"### [{c['id']}](https://arxiv.org/abs/{c['id']}) — {c['title']}")
        lines.append(f"*submitted {c['date']}*")
        if ann:
            lines.append(
                f"- **Problem (AI draft)**: {ann.get('problem','?')}  "
                f"\n- **Paradigm (AI draft)**: {ann.get('paradigm','?')}  "
                f"\n- **Likely relevant**: {ann.get('relevant','?')}"
            )
        else:
            lines.append(f"> {c['abstract'][:300]}…")
        lines.append("")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(cands)} candidates to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
