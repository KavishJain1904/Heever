"""End-to-end online pipeline. `make eval-full` enters here.

Contract: docs/12 §13. Architecture: docs/02 §10.

Per inbound message, in order:
  [A] guards.redact / scan_injection / scan_safety
  [B] classifier -> (intent, confidence)
  [C] retrieve.retrieve -> playbook + 3 exemplars
  [D] generate.generate (schema-constrained)
  [E] pydantic validation + 1 bounded repair retry
  [F] grounding.tripwire + verify_against_evidence
  [G] policy.apply_ladder
  [H] route: action + escalation_code + route_to + priority
  [I] append a DecisionRecord to decisions.jsonl

Order matters: [F] runs BEFORE [G] because the grounding result is rung 6's input.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Optional, Sequence

from src import REPO_ROOT, load_config, load_taxonomy
from src import guards, grounding, policy
from src.generate import build_prompt, generate
from src.policy import PolicyContext, build_reason, route_for
from src.retrieve import RetrievalIndex, retrieve
from src.schemas import (
    Action, CommitmentSlots, Decision, DecisionRecord, PolicyLayer, Stratum,
)

FILTER_INTENTS = frozenset({"non_support"})
RETRYABLE = frozenset({429, 500, 502, 503, 504})


def _playbook_for(intent: str) -> str:
    path = REPO_ROOT / "playbooks" / f"{intent}.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def process_one(
    message: str,
    config=None,
    classifier=None,
    index: Optional[RetrievalIndex] = None,
    style_card: Optional[dict] = None,
    message_id: str = "",
    conversation_id: str = "",
    stratum: Optional[str] = None,
    thread_turns: int = 1,
    days_since_last_contact: Optional[float] = None,
) -> DecisionRecord:
    """Run one message through A-I. Returns a DecisionRecord.

    [F] runs BEFORE [G] because the grounding result is rung 6's input. Getting
    that order wrong would let an ungrounded draft reach AUTO_SEND -- the ladder
    would have nothing to veto.
    """
    config = config or load_config()
    started = time.perf_counter()

    # [A] ingest guard
    redacted, pii_types = guards.redact(message)
    safety_flagged = guards.scan_safety(message)
    injection_flagged = guards.scan_injection(message)

    # [B] classify
    intent, confidence = classifier(redacted) if classifier else ("other_unclear", 0.0)

    # Filter classes are dropped here, before the ladder. The Action enum has no
    # FILTER member by design, and inventing an action for a message we decided not
    # to reply to would be a silent contract violation.
    if intent in FILTER_INTENTS:
        return _filtered_record(
            message_id, conversation_id, stratum, redacted, intent, confidence,
            time.perf_counter() - started,
        )

    # [C] retrieve
    exemplars = retrieve(redacted, intent, config.brand["handle"], config, index=index)
    playbook = _playbook_for(intent)
    # Absolute similarity of the closest precedent. Not rrf_score: that is a fused
    # RANK score capped at 2/(k+1) ~= 0.033, and gating it against tau_retrieval=0.35
    # escalated 191/200 messages on the first real run.
    top1 = max(
        (e.dense_score if e.dense_score is not None else e.rrf_score for e in exemplars),
        default=0.0,
    )

    # [D][E] generate + bounded repair
    draft: Optional[str] = None
    raw_output = ""
    validation_errors: list[str] = []
    repair_attempts = 0
    usage: dict = {}
    cache_hit = False
    decision_from_model: Optional[Decision] = None

    # Skip generation entirely when rungs 1-4 will veto regardless: a draft that
    # cannot be sent is pure cost. Correctness is unaffected because the ladder is
    # re-run below over the real context.
    pre_context = PolicyContext(
        text=redacted, intent=intent, intent_confidence=confidence,
        pii_types=pii_types, safety_flagged=safety_flagged,
        injection_flagged=injection_flagged, top1_retrieval_score=top1,
        evidence_ids=[e.evidence_id for e in exemplars],
        thread_turns=thread_turns, days_since_last_contact=days_since_last_contact,
    )
    pre_action, _, pre_layer = policy.apply_ladder(pre_context, config)

    # Evaluation mode (config.generation.draft_on_escalate): also draft for messages
    # the ladder already escalated, so reply quality can be judged on a uniform
    # sample instead of only on the survivors of every gate. The draft is recorded
    # in raw_model_output and never sent: final_draft below is still None on ESCALATE.
    eval_draft = bool(config.generation.get("draft_on_escalate")) and pre_layer is not PolicyLayer.SAFETY_VETO
    if (pre_layer is PolicyLayer.DEFAULT and pre_action is not Action.ESCALATE) or eval_draft:
        try:
            prompt = build_prompt(redacted, playbook, exemplars, style_card or {})
            decision_from_model, raw_output, usage, cache_hit, validation_errors, repair_attempts = \
                generate(prompt, config, exemplars=exemplars)
            draft = decision_from_model.reply_draft
        except Exception as exc:  # validation exhausted, or provider error
            validation_errors.append(str(exc))

    # [F] grounding -- BEFORE the ladder
    grounding_passed: Optional[bool] = None
    grounding_reasons: list[str] = []
    commitments = CommitmentSlots()
    if draft:
        commitments = grounding.extract_commitments(draft)
        grounding_passed, grounding_reasons = grounding.validate_draft(draft, exemplars, playbook)

    # [G] policy ladder
    context = PolicyContext(
        text=redacted, intent=intent, intent_confidence=confidence,
        pii_types=pii_types, safety_flagged=safety_flagged,
        injection_flagged=injection_flagged, top1_retrieval_score=top1,
        evidence_ids=[e.evidence_id for e in exemplars],
        thread_turns=thread_turns, days_since_last_contact=days_since_last_contact,
        draft=draft, grounding_passed=grounding_passed,
        grounding_reasons=grounding_reasons,
    )
    action, code, layer = policy.apply_ladder(context, config)

    # [H] route
    route_to, priority = route_for(code, config)
    reason = build_reason(code, context, config)
    if action is Action.ESCALATE and not reason:
        # Rung 7 escalating because the intent's playbook says a person owns it
        # (account access, billing, ...). No gate fired, so there is no code, but
        # the brief requires every escalation to carry a stated reason.
        reason = (
            f"Classified as {intent} ({confidence:.2f}). This intent is always handled "
            f"by a person under taxonomy/intents.yaml default_action: escalate."
        )

    # A discarded draft must not be carried on the record as if it were sendable.
    final_draft = draft if action in (Action.AUTO_SEND, Action.DM_HANDOFF, Action.REQUEST_INFO) else None

    decision = Decision(
        intent=intent,
        intent_confidence=confidence,
        action=action,
        reply_draft=final_draft,
        escalation_code=code,
        escalation_reason=reason if action is Action.ESCALATE else None,
        route_to=route_to,
        priority=priority,
        evidence_ids=[e.evidence_id for e in exemplars] or ["none"],
    )

    # [I] decision record
    return DecisionRecord(
        message_id=message_id,
        conversation_id=conversation_id,
        stratum=Stratum(stratum) if stratum else None,
        input_text_redacted=redacted,
        predicted_intent=intent,
        intent_confidence=confidence,
        evidence=exemplars,
        playbook_id=intent if playbook else None,
        raw_model_output=raw_output,
        validation_errors=validation_errors,
        repair_attempts=repair_attempts,
        commitments=commitments,
        grounding_passed=bool(grounding_passed) if grounding_passed is not None else True,
        policy_layer_fired=layer,
        decision=decision,
        latency_ms=(time.perf_counter() - started) * 1000,
        tokens_in=usage.get("input_tokens", 0),
        tokens_out=usage.get("output_tokens", 0),
        cost_usd=_cost(usage, config),
        cache_hit=cache_hit,
    )


def _filtered_record(message_id, conversation_id, stratum, redacted, intent, confidence, elapsed):
    """A non-support message: dropped, with a record so the drop is auditable."""
    return DecisionRecord(
        message_id=message_id, conversation_id=conversation_id,
        stratum=Stratum(stratum) if stratum else None,
        input_text_redacted=redacted, predicted_intent=intent,
        intent_confidence=confidence, evidence=[], playbook_id=None,
        raw_model_output="", validation_errors=[], repair_attempts=0,
        commitments=CommitmentSlots(), grounding_passed=True,
        policy_layer_fired=PolicyLayer.DEFAULT,
        decision=Decision(
            intent=intent, intent_confidence=confidence, action=Action.ESCALATE,
            escalation_reason=(
                "Classified as non_support (not a support request). Filtered from "
                "the reply path; no draft was generated."
            ),
            evidence_ids=["filtered"],
        ),
        latency_ms=elapsed * 1000, tokens_in=0, tokens_out=0,
        cost_usd=0.0, cache_hit=True,
    )


# Published per-million-token prices for the configured generator. Kept here rather
# than in config.yaml because it is a vendor fact, not a policy a reviewer would
# want to argue with.
# Standard (non-batch) list prices, USD per 1M tokens, read off developers.openai.com
# and ai.google.dev pricing pages on 2026-09-16. An unknown model costs 0.0 here,
# which under-reports -- so add a row before switching the generator.
_PRICES = {
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5.4-mini": (0.75, 4.50),   # checked 2026-09-16, developers.openai.com/api/docs/pricing
    "gpt-5.4": (2.50, 15.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
}


def _cost(usage: dict, config) -> float:
    model_id = config.models["generator"]["id"]
    price_in, price_out = _PRICES.get(model_id, (0.0, 0.0))
    return (
        usage.get("input_tokens", 0) / 1e6 * price_in
        + usage.get("output_tokens", 0) / 1e6 * price_out
    )


async def _run_async(messages: Sequence[dict], config, **kwargs) -> list[DecisionRecord]:
    semaphore = asyncio.Semaphore(int(config.runtime["concurrency"]))

    async def one(item: dict) -> Optional[DecisionRecord]:
        async with semaphore:
            for attempt in range(int(config.runtime["max_retries"]) + 1):
                try:
                    return await asyncio.to_thread(
                        process_one, item["text"], config,
                        message_id=item.get("message_id", ""),
                        conversation_id=item.get("conversation_id", ""),
                        stratum=item.get("stratum"),
                        thread_turns=item.get("thread_turns", 1),
                        days_since_last_contact=item.get("days_since_last_contact"),
                        **kwargs,
                    )
                except Exception as exc:
                    status = getattr(exc, "status_code", None)
                    # NEVER retry a 400: the request is malformed and will be
                    # malformed again. Retrying it burns quota to reproduce a bug.
                    if status not in RETRYABLE or attempt >= int(config.runtime["max_retries"]):
                        print(f"[{item.get('message_id')}] giving up: {exc}")
                        return None
                    base = float(config.runtime["backoff_base_seconds"])
                    # Jitter, not bare exponential: without it every retry in a
                    # burst lands at the same instant and re-triggers the 429.
                    delay = base * (2 ** attempt) * (0.5 + random.random())
                    await asyncio.sleep(delay)
            return None

    results = await asyncio.gather(*(one(m) for m in messages))
    return [r for r in results if r is not None]


def run(messages: Sequence[dict], config=None, **kwargs) -> list[DecisionRecord]:
    """Bounded concurrency via asyncio.Semaphore(config.runtime.concurrency).

    Retry with exponential backoff + jitter on 429/5xx ONLY -- never retry a 400.
    Emits run_report.json: tokens in/out, $/message, p50/p95 latency, cache hit rate.

    The semaphore is a concurrency bound, not a rate limiter. We are not building a
    rate limiter; the provider has one and it communicates via 429.
    """
    config = config or load_config()
    started = time.perf_counter()
    records = asyncio.run(_run_async(messages, config, **kwargs))

    decisions_path = REPO_ROOT / config.runtime["decisions_log"]
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(decisions_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")

    _write_run_report(records, config, time.perf_counter() - started)
    print(f"wrote {len(records)} decisions -> {decisions_path}")
    return records


def _write_run_report(records: Sequence[DecisionRecord], config, elapsed: float) -> None:
    import numpy as np

    latencies = sorted(r.latency_ms for r in records) or [0.0]
    layer_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for record in records:
        layer_counts[record.policy_layer_fired.value] = layer_counts.get(record.policy_layer_fired.value, 0) + 1
        action_counts[record.decision.action.value] = action_counts.get(record.decision.action.value, 0) + 1

    report = {
        "n_messages": len(records),
        "wall_clock_seconds": elapsed,
        "tokens_in": sum(r.tokens_in for r in records),
        "tokens_out": sum(r.tokens_out for r in records),
        "cost_usd_total": sum(r.cost_usd for r in records),
        "cost_usd_per_message": (
            sum(r.cost_usd for r in records) / len(records) if records else 0.0
        ),
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "cache_hit_rate": (
            sum(1 for r in records if r.cache_hit) / len(records) if records else 0.0
        ),
        "repair_rate": (
            sum(1 for r in records if r.repair_attempts) / len(records) if records else 0.0
        ),
        # The escalation breakdown that makes the system operationally legible:
        # "41% of escalations are NO_PRECEDENT" is a knowledge-base gap, not a
        # model problem, and only the layer counts can tell you that.
        "policy_layer_counts": layer_counts,
        "action_counts": action_counts,
    }
    path = REPO_ROOT / config.runtime["run_report"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="src.pipeline")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)

    golden_path = REPO_ROOT / config.evaluation["golden_set"]
    if not golden_path.exists():
        print(
            f"{golden_path} not found. Tier 2 regenerates predictions from the golden "
            f"set; `make rejoin` hydrates it from your own twcs.csv first.\n"
            f"`make eval` (Tier 1) reproduces every headline number without it."
        )
        return 1

    import csv

    with open(golden_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]

    run([
        {
            "message_id": r["message_id"],
            "conversation_id": r.get("conversation_id", ""),
            "text": r.get("text", ""),
            "stratum": r.get("stratum"),
        }
        for r in rows
    ], config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
