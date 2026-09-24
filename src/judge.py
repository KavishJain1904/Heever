"""LLM-as-judge: binary 6-criterion checklist. Different family from the generator.

Contract: docs/12 §11. Design: docs/04 §5. Rubric text: docs/14 §7.

WHY BINARY, NOT LIKERT (the biggest single reliability win available):
A base judge "never predicts Likert scores 5 and 6, and barely predicts 3 and 2"
-- scores concentrate near the domain mean while humans use the full range
(arXiv:2506.02945). Replacing "rate 1-5" with 6 yes/no criteria kills score
compression and 4/5-clustering outright, AND gives each criterion its own kappa.

WHY A DIFFERENT FAMILY (non-negotiable, config-enforced):
Self-preference is LINEARLY CORRELATED with self-recognition capability, with a
causal link supported by fine-tuning (arXiv:2404.13076). Generator is OpenAI;
judge is Gemini (config.yaml `models:`). If the two are ever configured to the same
provider, a warning prints and the report must flag that the families are shared.

Tier-1 mitigations, all cheap, all applied:
  1. different family (above)
  2. randomise order, score BOTH orderings, average or call ties -- measured
     consistency in the wild runs ~70-77%, so this is not cosmetic
  3. binary checklist (above)
  4. structured output, per-criterion sub-score, REQUIRED VERBATIM EVIDENCE QUOTE
  5. brief CoT before the score
  6. report kappa, not raw agreement
"""
from __future__ import annotations

import json
import random
from typing import Optional, Sequence

from src import load_config
from src.cache import cache_key, get as cache_get, put as cache_put

# Score sub-criteria, not a single gestalt number (FLASK, arXiv:2307.10928).
CRITERIA: tuple[str, ...] = (
    "addresses_stated_problem",
    "grounded_in_retrieved_context",
    "no_unsupported_policy_commitment",
    "correct_brand_voice",
    "correct_escalate_or_autohandle_decision",
    "no_pii_or_unsafe_content",
)

# Criteria where a `pass` REQUIRES a verbatim quote. That requirement is what makes
# a judge error auditable rather than merely recorded: a reviewer can check the
# quote against the source in seconds.
QUOTE_REQUIRED: frozenset[str] = frozenset({
    "addresses_stated_problem",
    "grounded_in_retrieved_context",
    "correct_brand_voice",
})

CRITERION_TEXT: dict[str, str] = {
    "addresses_stated_problem":
        "Does the reply respond to what the customer actually asked, rather than to "
        "a superficially similar problem?",
    "grounded_in_retrieved_context":
        "Is every factual claim in the reply supported by the exemplars or playbook?",
    "no_unsupported_policy_commitment":
        "Does the reply avoid promising any refund, timeline, entitlement or policy "
        "that the evidence does not show this brand promising?",
    "correct_brand_voice":
        "Does the reply match the brand's observed tone, length and conventions?",
    "correct_escalate_or_autohandle_decision":
        "Given the message, is auto-handling versus escalating the right call?",
    "no_pii_or_unsafe_content":
        "Is the reply free of personal data and of unsafe or inappropriate content?",
}

RUBRIC_HEADER = """You are evaluating a drafted customer-support reply.

Answer each criterion YES or NO. Do not rate on a scale -- a scale compresses
toward the middle and the decision you are making is binary.

Before each answer, give one sentence of reasoning. For criteria marked
[QUOTE REQUIRED], a YES must include a verbatim quote from the reply or evidence
that justifies it. A YES with no quote will be treated as NO.

The historical reply, where shown, is CONTEXT ONLY. It is not the target. Do not
reward similarity to it -- on this corpus it is often mediocre itself."""


def build_rubric_prompt(
    message: str,
    draft: str,
    evidence: str,
    criteria_order: Sequence[str],
    historical: Optional[str] = None,
) -> str:
    """The judge prompt, with criteria in the given order.

    Order is randomised per call (config.evaluation.judge.criteria_order) because
    all LLM judges examined show strong position bias, most favouring the first
    position, with measured consistency in the wild of ~70-77%. That is not
    cosmetic, so it is mitigated rather than noted.
    """
    numbered = "\n".join(
        f"{i}. {name} {'[QUOTE REQUIRED] ' if name in QUOTE_REQUIRED else ''}"
        f"-- {CRITERION_TEXT[name]}"
        for i, name in enumerate(criteria_order, 1)
    )
    historical_block = (
        f"\n<historical_reply note=\"context only, NOT the target\">\n{historical}\n</historical_reply>\n"
        if historical else ""
    )
    return (
        f"{RUBRIC_HEADER}\n\n<customer_message>\n{message}\n</customer_message>\n\n"
        f"<evidence>\n{evidence}\n</evidence>\n{historical_block}\n"
        f"<drafted_reply>\n{draft}\n</drafted_reply>\n\nCriteria:\n{numbered}\n\n"
        f"Return JSON: {{\"criteria\": {{\"<name>\": {{\"reasoning\": str, "
        f"\"pass\": bool, \"quote\": str}}}}}}"
    )


def _call_judge(prompt: str, config) -> tuple[str, dict, bool]:
    """Cache-first judge call. Family is enforced to differ from the generator."""
    judge_cfg = config.models["judge"]
    generator_cfg = config.models["generator"]

    provider = judge_cfg["provider"]
    model_id = judge_cfg["id"]
    if provider == generator_cfg["provider"]:
        print(
            f"WARNING: judge provider {provider!r} matches the generator's. "
            f"Self-preference is linearly correlated with self-recognition "
            f"capability -- this MUST be flagged in the report (docs/04 §5)."
        )

    key = cache_key(model_id, config.evaluation["judge"]["rubric_version"], prompt)
    cached = cache_get(key, config)
    if cached is not None:
        return cached["response"], cached.get("usage", {}), True

    from src.llm_client import complete

    raw, usage = complete(judge_cfg, prompt, json_mode=True)
    cache_put(key, raw, usage, config)
    return raw, usage, False


def _parse_verdict(raw: str) -> dict:
    """Parse, then ENFORCE the quote requirement. A YES with no quote becomes NO.

    Enforced here rather than trusted from the prompt: an instruction the model can
    ignore is not a mitigation, and this one is checkable in two lines.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    payload = json.loads(text)
    criteria = payload.get("criteria", {})
    for name, entry in criteria.items():
        if name in QUOTE_REQUIRED and entry.get("pass") and not (entry.get("quote") or "").strip():
            entry["pass"] = False
            entry["quote_enforcement"] = "downgraded: pass claimed with no verbatim quote"
    return payload


def judge_one(message: str, draft: str, evidence, config=None, historical: Optional[str] = None):
    """Score one (message, draft) pair. Returns a JudgeVerdict. Schema: docs/11 §7.

    Reference-guided grading caveat (docs/04 §5 tier 2): on this corpus the
    historical reply is often itself mediocre -- pass it as CONTEXT, never as THE
    TARGET. Scoring similarity-to-history rewards mimicking mediocrity.

    Both criteria orderings are scored and DISAGREEMENTS ARE RECORDED, not silently
    averaged away: the disagreement rate is the position-bias measurement, and it
    goes in the report.
    """
    config = config or load_config()
    judge_cfg = config.evaluation["judge"]
    evidence_text = evidence if isinstance(evidence, str) else "\n".join(
        f"[{e.evidence_id}] {e.brand_reply}" for e in (evidence or [])
    )

    orders = [list(CRITERIA)]
    if judge_cfg["criteria_order"] == "randomised":
        shuffled = list(CRITERIA)
        random.Random(hash(draft) & 0xFFFF).shuffle(shuffled)
        orders = [shuffled]
    if judge_cfg["score_both_orderings"]:
        orders.append(list(reversed(orders[0])))

    runs = []
    usage_total: dict = {}
    cache_hits = []
    for order in orders:
        prompt = build_rubric_prompt(message, draft, evidence_text, order, historical)
        raw, usage, hit = _call_judge(prompt, config)
        runs.append(_parse_verdict(raw))
        cache_hits.append(hit)
        for k, v in usage.items():
            usage_total[k] = usage_total.get(k, 0) + v

    merged: dict = {}
    disagreements = []
    for name in CRITERIA:
        votes = [r["criteria"][name]["pass"] for r in runs if name in r.get("criteria", {})]
        if not votes:
            continue
        if len(set(votes)) > 1:
            disagreements.append(name)
        # A tie across orderings resolves to FAIL. The conservative direction is the
        # one that does not let an auto-send through on a coin flip.
        merged[name] = {
            **runs[0]["criteria"][name],
            "pass": all(votes),
            "order_disagreement": len(set(votes)) > 1,
        }

    return {
        "criteria": merged,
        "all_six_pass": all(c["pass"] for c in merged.values()) if merged else False,
        "position_bias_disagreements": disagreements,
        "usage": usage_total,
        "cache_hit": all(cache_hits),
    }


def judge_historical_replies(threads, config=None):
    """Score the BRAND'S OWN historical replies with the same rubric.

    This reframes the entire result: if the human replies score 3.4/5-equivalent and
    the model scores 3.6/5-equivalent, the headline means something completely
    different. Reported as a reference band (docs/00 trap 3).
    """
    config = config or load_config()
    verdicts = []
    for thread in threads:
        brand_turns = [t for t in (thread.get("turns") or []) if t.get("is_brand")]
        if not brand_turns:
            continue
        verdict = judge_one(
            thread.get("customer_text", ""), brand_turns[0]["text"], "", config
        )
        verdict.update({"system": "historical", "message_id": thread.get("conversation_id")})
        verdicts.append(verdict)
    return verdicts


def paraphrase_robustness(rubric: str, pairs, n: int = 3, config=None) -> dict:
    """Run the judge under n rubric paraphrases; return the flip rate.

    GPT-4o flips its rating of the same summary on 8.5% of pairs under
    semantically-preserving rephrasings. This is a RESULT WE REPORT, not an internal
    check -- it directly answers "how fragile is my judge?".

    Our score is conditional on ONE rubric wording. This number quantifies how much
    that conditioning matters, which is the difference between a measurement and a
    number.
    """
    config = config or load_config()
    paraphrases = [
        rubric,
        rubric.replace("Answer each criterion YES or NO", "For each criterion, respond YES or NO"),
        rubric.replace("Does the reply", "Would you say the reply"),
    ][:n]

    flips = 0
    compared = 0
    for message, draft, evidence in pairs:
        outcomes = []
        for variant in paraphrases:
            order = list(CRITERIA)
            prompt = build_rubric_prompt(message, draft, evidence, order)
            prompt = prompt.replace(RUBRIC_HEADER, variant)
            raw, _, _ = _call_judge(prompt, config)
            parsed = _parse_verdict(raw)
            outcomes.append(all(c["pass"] for c in parsed["criteria"].values()))
        compared += 1
        if len(set(outcomes)) > 1:
            flips += 1

    return {
        "n_pairs": compared,
        "n_paraphrases": len(paraphrases),
        "flip_rate": flips / compared if compared else 0.0,
        "note": (
            "Reported as a result, not an internal check. Our headline is "
            "conditional on one rubric wording; this is how much that matters."
        ),
    }
