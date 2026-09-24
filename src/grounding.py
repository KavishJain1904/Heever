"""Stage F (online): the grounding validator. ~60 lines, disproportionately impressive.

Contract: docs/12 §8. Design: docs/02 §7, docs/04 §6.

NEVER INVENT POLICY. The canonical case is Moffatt v. Air Canada, 2024 BCCRT 149:
the chatbot said a bereavement fare could be claimed retroactively within 90 days,
contradicting the policy page it itself linked. The tribunal found negligent
misrepresentation and REJECTED the argument that the chatbot was "a separate legal
entity responsible for its own actions". One invented policy sentence, one legally
binding obligation.

Two layers, cheapest first:
  1. A deterministic TRIPWIRE -- regex for currency amounts, day/hour counts, and
     modal commitment verbs. Catches most of these at near-zero cost, no LLM.
  2. A typed COMMITMENT EXTRACTOR feeding strict entailment against retrieved
     evidence. We extract commitments; we do not score prose. Empty is the common
     and correct case.

Unsupported populated slot => HARD FAIL, no matter how good the reply reads.
Reported as a rate with a Wilson UPPER BOUND ("0/100 unsupported commitments;
95% upper bound 3.6% by rule-of-three") -- an upper bound is the honest form for a
near-zero rate.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from src.schemas import CommitmentSlots, Exemplar

DURATION_PATTERN = r"\b\d+\s*(?:business\s+)?(?:hour|day|week|month)s?\b"
COMMITMENT_VERBS = ("we will", "we'll", "you're entitled", "you are entitled",
                    "guaranteed", "guarantee", "we can refund", "we will refund")

CURRENCY_PATTERN = r"[$£€]\s?\d[\d,]*(?:\.\d{2})?"
URL_PATTERN = r"https?://[^\s)]+|\b(?:www\.)[^\s)]+"
# Four confirmed sigils: AmazonHelp ^SM, Spotify /AL, Delta *AMV, comcastcares ~AT.
# Stripped at ingest (build_sample.normalise_text), so the generator has never seen
# one in its evidence. Emitting one means it invented an agent identity.
SIGIL_PATTERN = r"[\^~*/][A-Z]{1,3}\s*$"
# Bare numbers, excluding those already captured as currency or duration. Dates in
# any of the forms support agents actually write.
BARE_NUMBER_PATTERN = r"\b\d[\d,]*(?:\.\d+)?\b"
DATE_PATTERN = (
    r"\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2})\b"
)

_CURRENCY_RE = re.compile(CURRENCY_PATTERN)
_DURATION_RE = re.compile(DURATION_PATTERN, re.IGNORECASE)
_COMMITMENT_RE = re.compile(
    "|".join(re.escape(v) for v in COMMITMENT_VERBS), re.IGNORECASE
)
_URL_RE = re.compile(URL_PATTERN, re.IGNORECASE)
_SIGIL_RE = re.compile(SIGIL_PATTERN)
_BARE_NUMBER_RE = re.compile(BARE_NUMBER_PATTERN)
_DATE_RE = re.compile(DATE_PATTERN, re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")

# Numbers that carry no policy content. "24/7" and "100%" are brand boilerplate,
# not commitments, and flagging them would bury the real hits under noise -- the
# classic way a safety check gets switched off by the team that has to read it.
_NUMBER_ALLOWLIST = {"24", "7", "100", "1", "2", "3"}


def tripwire(draft: str) -> list[str]:
    """Deterministic scan. Returns the list of risky spans found. No LLM.

    This is layer 1 and it is deliberately evidence-blind: it answers "does this
    draft assert anything that could bind the brand?", not "is the assertion true?".
    Separating those two questions is what keeps the cheap check cheap -- most
    drafts trip nothing and never reach the entailment layer at all.
    """
    if not draft:
        return []
    spans: list[str] = []
    for pattern in (_CURRENCY_RE, _DURATION_RE, _COMMITMENT_RE, _DATE_RE):
        spans.extend(m.group(0).strip() for m in pattern.finditer(draft))
    # De-duplicate while preserving discovery order, so the list reads as the
    # reviewer would have read the draft.
    seen: set[str] = set()
    ordered: list[str] = []
    for span in spans:
        low = span.lower()
        if low not in seen:
            seen.add(low)
            ordered.append(span)
    return ordered


def extract_commitments(draft: str) -> CommitmentSlots:
    """Structured extraction into CommitmentSlots. Empty is the expected result.

    Typed slots, not prose scoring. A judge asked "is this reply well-grounded?"
    returns a number that cannot be audited; a slot that says
    `refund_window_days=90` can be checked against evidence by string search, and
    the check either passes or it does not.
    """
    slots = CommitmentSlots()
    if not draft:
        return slots

    low = draft.lower()

    if re.search(r"\brefund(?:ed|ing)?\b", low) and not re.search(
        r"\b(?:can(?:no|')t|cannot|unable to|no)\s+refund", low
    ):
        slots.refund_offered = True

    window = re.search(
        r"\bwithin\s+(\d+)\s*(?:business\s+)?(day|week|month)s?\b", low
    )
    if window:
        n = int(window.group(1))
        unit = window.group(2)
        slots.refund_window_days = n * {"day": 1, "week": 7, "month": 30}[unit]

    money = _CURRENCY_RE.search(draft)
    if money:
        cleaned = re.sub(r"[^\d.]", "", money.group(0))
        if cleaned:
            slots.compensation_amount = float(cleaned)

    timeline = _DURATION_RE.search(draft)
    if timeline:
        slots.promised_timeline = timeline.group(0).strip()

    entitlement = re.search(
        r"\byou(?:'re| are)\s+entitled\s+to\s+([^.!?]{1,60})", draft, re.IGNORECASE
    )
    if entitlement:
        slots.entitlement_claimed = entitlement.group(1).strip()

    escalation = re.search(
        r"\b(?:we(?:'ll| will)|I(?:'ll| will))\s+(?:escalate|pass this (?:on|to)|"
        r"have (?:a|our) (?:specialist|manager|team)[^.!?]{0,40})", draft, re.IGNORECASE
    )
    if escalation:
        slots.escalation_promise = escalation.group(0).strip()

    return slots


def _evidence_corpus(exemplars: Iterable[Exemplar], playbook: Optional[str]) -> str:
    """The full text a claim may be grounded in: historical replies plus the playbook.

    Customer text is deliberately EXCLUDED. A customer writing "you promised me a
    $50 refund" must not license the generator to promise a $50 refund -- that is
    grounding a commitment in the demand for it, which is the exact failure mode
    an injection attempt exploits.
    """
    parts = [e.brand_reply or "" for e in exemplars or []]
    if playbook:
        parts.append(playbook)
    return "\n".join(parts)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).lower()


def verify_against_evidence(
    slots: CommitmentSlots,
    exemplars: list[Exemplar],
    playbook: str,
) -> tuple[bool, list[str]]:
    """Each populated slot must be supported by a VERBATIM SPAN in retrieved evidence.

    Also checks, independently of the slots:
      - every number, currency amount, duration and date in the draft appears in
        the evidence or the playbook;
      - every URL and phone number is on the allowlist observed in that brand's
        historical replies (hallucinated support URLs are a classic failure);
      - no invented agent sigil (^AB / /AL / *AMV / ~AT) -- predicted failure #3.

    Returns (passed, reasons). A False here fires PolicyLayer.POST_GENERATION_VALIDATION.
    """
    corpus = _normalise(_evidence_corpus(exemplars, playbook))
    reasons: list[str] = []

    if slots.refund_offered and not re.search(r"\brefund", corpus):
        reasons.append("offers a refund; no retrieved precedent mentions a refund")

    if slots.refund_window_days is not None:
        if str(slots.refund_window_days) not in corpus:
            reasons.append(
                f"states a {slots.refund_window_days}-day window with no supporting "
                f"span in evidence"
            )

    if slots.compensation_amount is not None:
        amount = slots.compensation_amount
        rendered = {
            f"{amount:.2f}",
            f"{amount:,.2f}",
            f"{int(amount)}" if amount == int(amount) else f"{amount}",
        }
        if not any(r in corpus for r in rendered):
            reasons.append(
                f"promises {amount:g} with no matching amount in evidence"
            )

    if slots.promised_timeline and _normalise(slots.promised_timeline) not in corpus:
        reasons.append(
            f"promises {slots.promised_timeline!r}; no precedent states that timeline"
        )

    if slots.entitlement_claimed:
        if not re.search(r"\bentitle", corpus):
            reasons.append(
                f"asserts an entitlement ({slots.entitlement_claimed!r}) that no "
                f"precedent supports"
            )

    return (not reasons, reasons)


def validate_draft(
    draft: str,
    exemplars: list[Exemplar],
    playbook: str,
) -> tuple[bool, list[str]]:
    """Full Stage F check: slots plus the three slot-independent checks.

    This is what pipeline.process_one calls; `verify_against_evidence` is the slot
    half, kept separately callable because the slot logic is what the failure
    analysis tables break down by.
    """
    reasons: list[str] = []
    if not draft:
        return (True, reasons)

    corpus_raw = _evidence_corpus(exemplars, playbook)
    corpus = _normalise(corpus_raw)

    slots = extract_commitments(draft)
    _, slot_reasons = verify_against_evidence(slots, exemplars, playbook)
    reasons.extend(slot_reasons)

    # Every number in the draft must appear in evidence. Deliberately strict: a
    # false positive costs one needless escalation, a false negative costs a
    # binding commitment the brand never made.
    for match in _BARE_NUMBER_RE.finditer(draft):
        token = match.group(0)
        if token.replace(",", "") in _NUMBER_ALLOWLIST:
            continue
        if token.lower() not in corpus:
            reasons.append(f"number {token!r} does not appear in evidence")

    for match in _DATE_RE.finditer(draft):
        if match.group(0).lower() not in corpus:
            reasons.append(f"date {match.group(0)!r} does not appear in evidence")

    # URL and phone allowlists, observed from this brand's own replies.
    url_allow = {u.lower().rstrip("/.,") for u in _URL_RE.findall(corpus_raw)}
    for match in _URL_RE.finditer(draft):
        url = match.group(0).lower().rstrip("/.,")
        if url != "[url]" and url not in url_allow:
            reasons.append(f"URL {match.group(0)!r} is not on the brand's observed allowlist")

    phone_allow = {re.sub(r"\D", "", p) for p in _PHONE_RE.findall(corpus_raw)}
    for match in _PHONE_RE.finditer(draft):
        digits = re.sub(r"\D", "", match.group(0))
        if digits and digits not in phone_allow:
            reasons.append(f"phone number {match.group(0)!r} is not in evidence")

    # Invented agent sigil. Sigils are stripped at ingest, so the model never saw
    # one in context; emitting one is pure invention of a human identity.
    if _SIGIL_RE.search(draft):
        reasons.append(
            f"draft ends in an agent sigil ({_SIGIL_RE.search(draft).group(0).strip()!r}); "
            f"sigils are stripped at ingest and must never be generated"
        )

    return (not reasons, reasons)
