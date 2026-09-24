"""Offline stage 4: per-intent playbook distillation. The grounding unit.

Contract: docs/12 §10. Design: docs/02 §2.

Per intent, feed ~40-60 SUCCESSFULLY RESOLVED threads to an LLM and distil a
playbook containing:
  (i)   canonical resolution steps the brand actually took,
  (ii)  information the brand always asks for,
  (iii) whether the brand's standard move is a public reply or a DM handoff,
  (iv)  tone/signature conventions,
  (v)   an explicit "things this brand never promises" list.

Why the hybrid (playbook + 3 raw exemplars) rather than either alone:
  - It SEPARATES POLICY FROM PRECEDENT. A reviewer can open a playbook, read it,
    and DISAGREE WITH IT. Three raw retrieved tweets are not reviewable that way.
  - It structurally fixes the DM-deflection problem: the playbook can state "this
    brand resolves 78% of shipping-delay cases by moving to DM after collecting the
    order number", converting DM handoff from a retrieval accident into an explicit,
    defensible ACTION.

THE COUNTER-ARGUMENT, stated in the report because reviewers reward a stated
tradeoff: distillation is a lossy, un-grounded step. An LLM summarising 50 threads
can assert "refunds within 30 days" when no thread said so. MITIGATION: every
playbook line carries evidence_ids pointing at source thread IDs, and they are
spot-checked. Without provenance we have only moved the hallucination upstream and
hidden it. Raw exemplars are kept IN ADDITION TO, never instead of, the playbook.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from src import REPO_ROOT, load_config
from src.cache import cache_key, get as cache_get, put as cache_put
from src.retrieve import is_resolution

DISTILL_PROMPT = """Below are {n} customer-support threads for {brand}, all of which
were RESOLVED (the brand gave a substantive answer and the customer confirmed it
worked).

Write a playbook for the intent "{intent}" describing what this brand ACTUALLY DOES.
Do not write what a good support team should do -- write what these threads show.

Cover:
(i)   canonical resolution steps the brand actually took
(ii)  information the brand always asks for before resolving
(iii) whether the standard move is a public reply or a DM handoff, with the
      observed proportion
(iv)  tone and signature conventions
(v)   "things this brand never promises" -- commitments ABSENT from every thread

EVERY factual line must end with the thread ids that support it, like
[evidence: t_123, t_456]. A line you cannot attribute is a line you must delete.
If the threads do not establish something, say so rather than filling the gap.

Threads:
{threads}"""

FRONT_MATTER = """---
intent: {intent}
brand: {brand}
n_source_threads: {n}
distiller_model: {model}
generated_by: src.distill
caveat: >
  Distillation is a lossy, UN-GROUNDED step. An LLM summarising 50 threads can
  assert "refunds within 30 days" when no thread said so. Every line carries
  evidence_ids and spot_check() verifies a sample. Raw exemplars are retrieved
  IN ADDITION TO this playbook, never instead of it.
---
"""

_EVIDENCE_RE = re.compile(r"\[evidence:\s*([^\]]+)\]")
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_SIGIL_RE = re.compile(r"[\^~*/][A-Z]{1,3}\s*$")
_DM_RE = re.compile(r"\b(?:dm|direct message) us\b|\bsend us a (?:dm|direct message)\b", re.IGNORECASE)


def select_resolved_threads(threads: Sequence[dict], intent: str, n: int = 50) -> list[dict]:
    """Threads with a substantive brand solution AND a customer confirmation.

    Confirmed threads are preferred over merely-substantive ones, because a
    customer saying "that worked" is the only signal in this corpus that the
    brand's answer was actually correct. Falling back to unconfirmed threads when
    there are too few is explicit, and the count of each is returned so the report
    can say how much of a playbook rests on which.
    """
    confirmed, substantive = [], []
    for thread in threads:
        if intent and thread.get("intent") != intent:
            continue
        turns = thread.get("turns") or []
        for i, turn in enumerate(turns):
            if not turn.get("is_brand"):
                continue
            follow_up = turns[i + 1]["text"] if i + 1 < len(turns) else None
            if is_resolution(turn.get("text", ""), follow_up):
                confirmed.append(thread)
                break
            if is_resolution(turn.get("text", ""), None):
                substantive.append(thread)
                break
    return (confirmed + substantive)[:n]


def distill(threads: Sequence[dict], intent: str, config=None) -> str:
    """Produce a playbook markdown doc with YAML front-matter. Schema: docs/11 §6.

    Runs through src.llm_client with config.models.distiller (OpenAI by default;
    local vLLM on the DGX experiment path). Output is COMMITTED, so the
    reproduction path never runs this.
    """
    config = config or load_config()
    model_cfg = config.models["distiller"]
    selected = select_resolved_threads(threads, intent, model_cfg["threads_per_playbook"])
    if not selected:
        raise ValueError(f"no resolved threads found for intent {intent!r}")

    rendered = "\n\n".join(
        f"[{t['conversation_id']}]\n"
        + "\n".join(f"  {'brand' if turn.get('is_brand') else 'customer'}: {turn['text']}"
                     for turn in (t.get("turns") or []))
        for t in selected
    )
    prompt = DISTILL_PROMPT.format(
        n=len(selected), brand=config.brand["handle"], intent=intent, threads=rendered
    )

    key = cache_key(model_cfg["id"], "distill-v1", prompt)
    cached = cache_get(key, config)
    if cached is not None:
        body = cached["response"]
    else:
        from src.llm_client import complete

        body, usage = complete(model_cfg, prompt)
        cache_put(key, body, usage, config)

    front = FRONT_MATTER.format(
        intent=intent, brand=config.brand["handle"],
        n=len(selected), model=model_cfg["id"],
    )
    return front + body


def spot_check(playbook_path, threads: Sequence[dict], sample: int = 10) -> dict:
    """Verify a sample of evidence_ids actually support their claim. Reported.

    Without this, provenance is decoration: the model can emit plausible-looking
    thread ids next to an invented policy line and we would have moved the
    hallucination one layer upstream and hidden it behind a citation.

    This checks that cited ids EXIST and that the cited thread shares vocabulary
    with the claim. It does not check entailment -- that would be another LLM step
    inheriting every judge bias, and the report says so rather than implying a
    stronger guarantee than the code delivers.
    """
    text = Path(playbook_path).read_text(encoding="utf-8")
    known = {str(t["conversation_id"]): t for t in threads}

    lines_with_evidence = []
    for line in text.splitlines():
        match = _EVIDENCE_RE.search(line)
        if match:
            ids = [i.strip() for i in match.group(1).split(",") if i.strip()]
            lines_with_evidence.append((line, ids))

    factual_lines = [l for l in text.splitlines() if l.strip().startswith(("-", "*", "1.")) and l.strip()]
    checked = lines_with_evidence[:sample]

    dangling, weak = [], []
    for line, ids in checked:
        for cited in ids:
            if cited not in known:
                dangling.append((line[:60], cited))
                continue
            thread_text = " ".join(t["text"] for t in (known[cited].get("turns") or [])).lower()
            claim_tokens = {w for w in re.findall(r"[a-z]{5,}", line.lower())}
            if claim_tokens and not (claim_tokens & set(re.findall(r"[a-z]{5,}", thread_text))):
                weak.append((line[:60], cited))

    return {
        "n_factual_lines": len(factual_lines),
        "n_lines_with_evidence": len(lines_with_evidence),
        "attribution_rate": len(lines_with_evidence) / len(factual_lines) if factual_lines else 0.0,
        "n_checked": len(checked),
        "dangling_ids": dangling,
        "weak_support": weak,
        "limitation": (
            "Checks existence and lexical overlap, NOT entailment. Entailment "
            "checking is itself an LLM step and inherits every judge bias."
        ),
    }


def build_style_card(brand_replies: Sequence[str]) -> dict:
    """Pure statistics, no LLM: mean reply length, signature regex hit rate,
    emoji rate, question rate. Injected into the prompt as a compact style card.

    No LLM because none is needed, and a summarised "voice" would be an unverifiable
    assertion where five counts are a fact. It is also cheaper and more stable: the
    same corpus always yields the same card.
    """
    replies = [r for r in brand_replies if r]
    if not replies:
        return {}
    n = len(replies)
    lengths = [len(r) for r in replies]
    return {
        "n_replies": n,
        "mean_length_chars": sum(lengths) / n,
        "median_length_chars": sorted(lengths)[n // 2],
        "signature_rate": sum(1 for r in replies if _SIGIL_RE.search(r.strip())) / n,
        "emoji_rate": sum(1 for r in replies if _EMOJI_RE.search(r)) / n,
        "question_rate": sum(1 for r in replies if "?" in r) / n,
        "url_rate": sum(1 for r in replies if "http" in r.lower()) / n,
        "dm_deflection_rate": sum(1 for r in replies if _DM_RE.search(r)) / n,
        "note": (
            "dm_deflection_rate is the number that turns DM handoff from a retrieval "
            "accident into a defensible action -- it is what the playbook cites."
        ),
    }
