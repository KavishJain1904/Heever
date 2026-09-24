"""Stage G (online): the escalation policy ladder. Pure Python, no LLM.

Contract: docs/12 §7. Design: docs/02 §6. Tests: docs/15 §2 (one per rung).

THE INVARIANT that makes this defensible in a live review:
    Deterministic guardrails can VETO automation but can NEVER AUTHORIZE it.
Layers 1-6 may only move the decision toward ESCALATE. Only layer 7 may return
AUTO_SEND or DM_HANDOFF.

Evaluate in order, SHORT-CIRCUIT on the first hit, and RECORD WHICH LAYER FIRED.
Every decision is then traceable to the rule that produced it -- which is the whole
argument against a single "should I escalate?" LLM call.

  1. SAFETY VETO        self-harm, threats, abuse -> ESCALATE, P0. No confidence overrides.
  2. COMPLIANCE VETO    legal/regulatory, fraud/account-security, PII in the public
                        tweet, amount > config.thresholds.auto_refund_ceiling_usd.
  3. NO-PRECEDENT GATE  top1_retrieval_score < tau_retrieval, or intent is OTHER.
                        Nothing similar ever happened -> do not invent a precedent.
  4. CONFIDENCE GATE    intent_confidence < tau_intent.
  5. CONTEXT GATE       repeat contact / thread >= N turns / missing required slot
                        (the last -> REQUEST_INFO, not ESCALATE).
  6. POST-GEN VALIDATION draft exists but fails the grounding check.
  7. DEFAULT            AUTO_SEND, or DM_HANDOFF per the intent's playbook.

Future work, stated honestly: a learned escalation classifier trained on "did a
human eventually take this over?" is the right long-term answer. We do not have
that label and do not pretend to.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src import load_taxonomy
from src.schemas import Action, EscalationCode, PolicyLayer

LEGAL_MARKERS: tuple[str, ...] = ("lawyer", "sue", "solicitor", "gdpr", "chargeback",
                                  "ombudsman", "small claims", "legal action")
SECURITY_MARKERS: tuple[str, ...] = ("hacked", "unauthorized charge", "unauthorised charge",
                                     "someone else", "stolen", "fraud")
CURRENCY_PATTERN = r"[$£€]\s?\d[\d,]*(?:\.\d{2})?"

# "sue" must not match "issue", "pursue", "sued" is fine but "tissue" is not.
# Every marker is matched on word boundaries for exactly this reason -- a substring
# scan here produces a LEGAL_THREAT on "I have an issue with my order", which is
# the single most common inbound sentence in the corpus.
_LEGAL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in LEGAL_MARKERS) + r")\b", re.IGNORECASE
)
_SECURITY_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in SECURITY_MARKERS) + r")\b", re.IGNORECASE
)
_CURRENCY_RE = re.compile(CURRENCY_PATTERN)

OTHER_INTENT = "other_unclear"


@dataclass
class PolicyContext:
    """Everything the ladder is allowed to look at. Deliberately a closed set.

    The ladder is pure: it reads this object and config, and touches nothing else.
    That is what makes every rung independently unit-testable without constructing
    a pipeline, and it is why the rungs take a context rather than a DecisionRecord
    -- a DecisionRecord contains the ladder's own output.
    """

    text: str                                   # post-redaction inbound message
    intent: str
    intent_confidence: float

    # Stage A (guards)
    pii_types: list[str] = field(default_factory=list)
    safety_flagged: bool = False
    injection_flagged: bool = False

    # Stage E (retrieval)
    top1_retrieval_score: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)

    # Thread context
    thread_turns: int = 1
    days_since_last_contact: Optional[float] = None
    missing_required_slot: Optional[str] = None

    # Stage F (grounding). None means "no draft was produced", which is NOT a
    # failure -- rung 6 can only fire on a draft that exists and failed.
    draft: Optional[str] = None
    grounding_passed: Optional[bool] = None
    grounding_reasons: list[str] = field(default_factory=list)


def _money_amounts(text: str) -> list[float]:
    """Every currency amount in the text, as floats. Empty is the common case."""
    amounts: list[float] = []
    for raw in _CURRENCY_RE.findall(text or ""):
        cleaned = re.sub(r"[^\d.]", "", raw)
        if cleaned:
            try:
                amounts.append(float(cleaned))
            except ValueError:
                continue
    return amounts


def safety_veto(context: PolicyContext) -> bool:
    """Rung 1. Self-harm, threats, abuse. NO confidence score may override this."""
    return bool(context.safety_flagged)


def compliance_veto(context: PolicyContext) -> Optional[EscalationCode]:
    """Rung 2. Returns the specific code, or None.

    Returns a code rather than a bool because the four sub-cases route to four
    different queues -- collapsing them to a bool would force the caller to
    re-derive which one fired, and a re-derivation that drifts from this function
    is a silent misrouting bug.

    Order within the rung is by severity of getting it wrong: an account takeover
    mishandled is worse than a legal threat mishandled is worse than a PII echo,
    and all three are worse than a refund above the ceiling.
    """
    text = context.text or ""
    if _SECURITY_RE.search(text):
        return EscalationCode.ACCOUNT_SECURITY
    if _LEGAL_RE.search(text):
        return EscalationCode.LEGAL_THREAT
    if context.injection_flagged:
        return EscalationCode.SUSPECTED_INJECTION
    if context.pii_types:
        return EscalationCode.PII_IN_PUBLIC_TWEET
    return None


def compliance_money_veto(context: PolicyContext, config) -> bool:
    """The fourth compliance sub-case, split out because it needs config.

    STRICTLY greater than the ceiling. An amount exactly equal to
    `auto_refund_ceiling_usd` does not fire: the ceiling is the largest value the
    system may auto-handle, not the smallest it may not. The boundary is pinned by
    a test because off-by-one on a money threshold is the kind of bug that is
    invisible until it is expensive.
    """
    ceiling = float(config.thresholds["auto_refund_ceiling_usd"])
    return any(amount > ceiling for amount in _money_amounts(context.text))


def no_precedent_gate(context: PolicyContext, config) -> bool:
    """Rung 3. Nothing similar ever happened -> do not invent a precedent.

    This is the rung worth explaining aloud in a review. The other gates ask "am I
    confident?"; this one asks "does this brand have any history of handling this
    at all?". A confident classification with no precedent behind it is exactly the
    condition under which a generator invents policy.
    """
    if context.intent == OTHER_INTENT:
        return True
    tau = float(config.thresholds["tau_retrieval"])
    if not context.evidence_ids:
        return True
    return context.top1_retrieval_score < tau


def confidence_gate(context: PolicyContext, config) -> bool:
    """Rung 4. Threshold is READ FROM CONFIG -- config-as-policy is the demonstration."""
    return context.intent_confidence < float(config.thresholds["tau_intent"])


def context_gate(context: PolicyContext, config) -> Optional[tuple[Action, Optional[EscalationCode]]]:
    """Rung 5. Returns (action, code) or None.

    Three distinct conditions with two distinct outcomes, which is why this rung
    returns a pair rather than a bool:

      - a missing required slot is REQUEST_INFO, not ESCALATE. Asking the customer
        for their order number is the correct, cheap, non-human-consuming move;
        escalating it burns an agent on a question the bot could have asked.
      - repeat contact and long threads are REPEAT_CONTACT escalations. Both mean
        the same thing operationally -- this is not first contact, and whatever the
        automated path had to offer has already been tried and did not work.
    """
    if context.missing_required_slot:
        return (Action.REQUEST_INFO, None)

    window = float(config.thresholds["repeat_contact_window_days"])
    if (
        context.days_since_last_contact is not None
        and context.days_since_last_contact <= window
    ):
        return (Action.ESCALATE, EscalationCode.REPEAT_CONTACT)

    if context.thread_turns >= int(config.thresholds["max_thread_turns_auto"]):
        return (Action.ESCALATE, EscalationCode.REPEAT_CONTACT)

    return None


def post_generation_gate(context: PolicyContext) -> bool:
    """Rung 6. Fires only on a draft that EXISTS and FAILED.

    `grounding_passed is None` means no draft was produced (the message was already
    heading for escalation, or generation was skipped). That is not a grounding
    failure and must not be reported as one -- conflating the two would inflate the
    UNGROUNDED_COMMITMENT count with messages that were never at risk, and that
    count is a headline safety number.
    """
    return context.draft is not None and context.grounding_passed is False


def apply_ladder(context: PolicyContext, config) -> tuple[Action, Optional[EscalationCode], PolicyLayer]:
    """Run the seven rungs in order. Returns (Action, EscalationCode|None, PolicyLayer).

    The caller maps EscalationCode -> route_to/priority via config.policy.routes,
    and emits BOTH forms of the reason: the code is what dashboards aggregate, the
    text is what the human reads in three seconds. `route_for` and `build_reason`
    below do that mapping.
    """
    # 1. SAFETY VETO -- first, unconditionally, and with no confidence override.
    if safety_veto(context):
        return (Action.ESCALATE, EscalationCode.SAFETY_RISK, PolicyLayer.SAFETY_VETO)

    # 2. COMPLIANCE VETO
    code = compliance_veto(context)
    if code is not None:
        return (Action.ESCALATE, code, PolicyLayer.COMPLIANCE_VETO)
    if compliance_money_veto(context, config):
        return (
            Action.ESCALATE,
            EscalationCode.MONEY_ABOVE_THRESHOLD,
            PolicyLayer.COMPLIANCE_VETO,
        )

    # 3. NO-PRECEDENT GATE
    if no_precedent_gate(context, config):
        return (Action.ESCALATE, EscalationCode.NO_PRECEDENT, PolicyLayer.NO_PRECEDENT_GATE)

    # 4. CONFIDENCE GATE
    if confidence_gate(context, config):
        return (Action.ESCALATE, EscalationCode.LOW_CONFIDENCE, PolicyLayer.CONFIDENCE_GATE)

    # 5. CONTEXT GATE
    ctx_result = context_gate(context, config)
    if ctx_result is not None:
        action, ctx_code = ctx_result
        return (action, ctx_code, PolicyLayer.CONTEXT_GATE)

    # 6. POST-GENERATION VALIDATION
    if post_generation_gate(context):
        return (
            Action.ESCALATE,
            EscalationCode.UNGROUNDED_COMMITMENT,
            PolicyLayer.POST_GENERATION_VALIDATION,
        )

    # 7. DEFAULT -- the ONLY rung permitted to authorize automation.
    return (_playbook_action(context.intent), None, PolicyLayer.DEFAULT)


def _playbook_action(intent: str) -> Action:
    """The intent's declared default_action from taxonomy/intents.yaml.

    An intent whose playbook says `escalate` (billing, refunds, account access)
    escalates even on a clean run through rungs 1-6. That is intentional and is the
    taxonomy expressing brand policy, not the ladder failing to find a problem.
    """
    for entry in load_taxonomy()["intents"]:
        if entry["name"] != intent:
            continue
        declared = entry.get("default_action", "escalate")
        if declared == "filter":
            raise ValueError(
                f"intent {intent!r} is a FILTER class and must be dropped in "
                f"pipeline.process_one before the ladder runs. Choosing an action "
                f"for a message we already decided not to reply to would be a silent "
                f"contract violation -- see taxonomy scope_classes."
            )
        return Action(declared)
    raise ValueError(f"intent {intent!r} has no entry in the taxonomy")


def route_for(code: Optional[EscalationCode], config) -> tuple[Optional[str], Optional[str]]:
    """EscalationCode -> (route_to, priority) via config.policy.routes.

    Mirrors Hiver's skill-based routing. Every EscalationCode member must be a key
    here; `test_every_escalation_code_has_a_route` asserts the enum and the config
    cannot drift apart.
    """
    if code is None:
        return (None, None)
    routes = config.policy["routes"]
    if code.value not in routes:
        raise KeyError(
            f"escalation code {code.value} has no route in config.policy.routes"
        )
    entry = routes[code.value]
    return (entry["route_to"], entry["priority"])


_REASON_TEMPLATES: dict[EscalationCode, str] = {
    EscalationCode.SAFETY_RISK:
        "Message contains safety-risk language. Routed to a human immediately; no "
        "automated reply was drafted.",
    EscalationCode.LEGAL_THREAT:
        "Customer invokes legal or regulatory language ({markers}). Legal-sensitive "
        "messages are never auto-handled.",
    EscalationCode.ACCOUNT_SECURITY:
        "Message reports possible account compromise ({markers}). Account-security "
        "cases are never auto-handled at any confidence.",
    EscalationCode.SUSPECTED_INJECTION:
        "Message contains text resembling an instruction to the assistant rather "
        "than a support request. Routed for review; the reply was not sent.",
    EscalationCode.PII_IN_PUBLIC_TWEET:
        "Customer posted {markers} in a public tweet. Replying publicly risks "
        "echoing it; a human should move this to DM.",
    EscalationCode.MONEY_ABOVE_THRESHOLD:
        "Customer requests {amount}; the auto-handling ceiling is ${ceiling:.0f}. "
        "{precedent}",
    EscalationCode.NO_PRECEDENT:
        "No sufficiently similar resolved case exists in this brand's history "
        "(top match scored {score:.2f} against a {tau:.2f} floor). Rather than "
        "invent a precedent, this goes to a human.",
    EscalationCode.LOW_CONFIDENCE:
        "Intent classified as {intent} at {conf:.0%} confidence, below the {tau:.0%} "
        "bar for auto-handling.",
    EscalationCode.REPEAT_CONTACT:
        "Not first contact ({detail}). The automated path has already been tried "
        "here and did not resolve it.",
    EscalationCode.UNGROUNDED_COMMITMENT:
        "The drafted reply made a commitment not supported by any retrieved "
        "precedent ({detail}). The draft was discarded, not sent.",
}


def build_reason(
    code: Optional[EscalationCode], context: PolicyContext, config
) -> Optional[str]:
    """The human-readable half of the reason. The code is the machine-readable half.

    The text is what a support agent reads in three seconds when the ticket lands
    in their queue, so it names the specific trigger and cites evidence ids where
    they exist. The brief asks for escalation "with a stated reason"; a bare enum
    value is not a stated reason.
    """
    if code is None:
        return None

    template = _REASON_TEMPLATES[code]
    ids = ", ".join(context.evidence_ids[:2])

    if code is EscalationCode.MONEY_ABOVE_THRESHOLD:
        amounts = _money_amounts(context.text)
        largest = max(amounts) if amounts else 0.0
        precedent = (
            f"Similar cases ({ids}) were resolved by a billing specialist, not a "
            f"first-line agent."
            if ids
            else "No first-line precedent exists for an amount of this size."
        )
        return template.format(
            amount=f"${largest:,.2f}".rstrip("0").rstrip("."),
            ceiling=float(config.thresholds["auto_refund_ceiling_usd"]),
            precedent=precedent,
        )

    if code is EscalationCode.NO_PRECEDENT:
        return template.format(
            score=context.top1_retrieval_score,
            tau=float(config.thresholds["tau_retrieval"]),
        )

    if code is EscalationCode.LOW_CONFIDENCE:
        return template.format(
            intent=context.intent,
            conf=context.intent_confidence,
            tau=float(config.thresholds["tau_intent"]),
        )

    if code is EscalationCode.REPEAT_CONTACT:
        if context.days_since_last_contact is not None:
            detail = f"previous contact {context.days_since_last_contact:.0f} days ago"
        else:
            detail = f"thread is {context.thread_turns} turns deep"
        return template.format(detail=detail)

    if code is EscalationCode.UNGROUNDED_COMMITMENT:
        detail = "; ".join(context.grounding_reasons[:2]) or "unsupported commitment"
        return template.format(detail=detail)

    if code in (EscalationCode.LEGAL_THREAT, EscalationCode.ACCOUNT_SECURITY):
        pattern = _LEGAL_RE if code is EscalationCode.LEGAL_THREAT else _SECURITY_RE
        markers = ", ".join(sorted({m.lower() for m in pattern.findall(context.text)}))
        if not markers:
            markers = "regulatory language"
        return template.format(markers=markers)

    if code is EscalationCode.PII_IN_PUBLIC_TWEET:
        return template.format(markers=", ".join(context.pii_types) or "personal data")

    return template
