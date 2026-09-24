"""Stage A (online): ingest guard -- PII redaction, injection scan, safety scan.

Contract: docs/12 §5. Design: docs/02 §7.

TWCS is only PARTIALLY anonymized. Emails and phones are masked upstream to
__email__ / __phone__, but free text still contains order numbers, addresses and
account identifiers customers typed themselves. Run the PII pass on INGEST and
again on EGRESS -- never let the generator echo a PII token into a public reply.
That is the strongest single argument for DM_HANDOFF.
"""
from __future__ import annotations

import re

# Documented regex set. Presidio is the production answer and is named as such in
# the report; a dependency-light repo with a documented regex set plus that note is
# the defensible take-home choice (docs/02 §7).
PII_PATTERNS: dict[str, str] = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "phone": r"(?:\+?\d[\d\s().-]{7,}\d)",
    "order_number": r"\b\d{3}-\d{7}-\d{7}\b",
    "card_like": r"\b(?:\d[ -]*?){13,16}\b",
}

# Application order matters and is NOT dict-insertion order by accident.
# `order_number` and `card_like` both match runs of digits and separators; the
# more specific pattern must consume the span first or an Amazon order id is
# reported as a card number. `email` precedes `phone` because a phone-shaped run
# can sit inside an email local part.
_PII_ORDER: tuple[str, ...] = ("email", "order_number", "card_like", "phone")

# Upstream TWCS masks. Their presence means PII WAS there and has already been
# removed -- so they are not re-reported as findings, but they are recognised so
# the redactor never double-masks them into `__<__EMAIL__>__` soup.
_PRE_MASKED = re.compile(r"__(?:email|phone)__", re.IGNORECASE)

# Injection detection is a ROUTING SIGNAL, not a filter. Treating attack detection
# as routing rather than blocking is safer and more honest about detector
# fallibility -- a missed detection still goes through the rest of the ladder.
INJECTION_MARKERS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous",
    "system prompt",
    "you are now",
    "disregard the above",
    "new instructions",
)

# Self-harm and crisis language. A hit routes to trust_and_safety at P0 and is the
# one rung that no confidence score may override. Recall is deliberately favoured
# over precision here: the cost of a false positive is one needless human review,
# the cost of a false negative is a person in crisis receiving an automated reply
# about their subscription.
SELF_HARM_MARKERS: tuple[str, ...] = (
    "kill myself", "killing myself", "end my life", "ending my life",
    "want to die", "wanna die", "better off dead", "suicidal", "suicide",
    "self harm", "self-harm", "hurt myself", "hurting myself",
    "no reason to live", "don't want to be here anymore",
)

THREAT_MARKERS: tuple[str, ...] = (
    "i'll kill you", "ill kill you", "kill you", "hunt you down", "find you and",
    "burn down", "shoot up", "bomb", "come to your office", "you're dead",
    "watch your back", "i will hurt", "make you pay",
)

# Abuse is scoped to slurs and sustained hostility directed at the agent. Ordinary
# profanity in a support tweet is NOT abuse -- "my app is f***ing broken" is a
# frustrated customer with a real bug, and escalating it wastes a human and insults
# the customer. Only second-person-directed profanity counts.
ABUSE_PATTERNS: tuple[str, ...] = (
    r"\b(?:fuck|screw)\s+(?:you|off|yourself)\b",
    r"\byou(?:'re|\s+are|\s+r)\s+(?:a\s+)?(?:fucking\s+)?(?:idiot|moron|useless|"
    r"retard|scum|trash|garbage|worthless)\b",
    r"\b(?:stupid|useless|incompetent)\s+(?:fucking\s+)?(?:bitch|bastard|idiots?)\b",
)

_INJECTION_RE = re.compile("|".join(re.escape(m) for m in INJECTION_MARKERS), re.IGNORECASE)
_SELF_HARM_RE = re.compile("|".join(re.escape(m) for m in SELF_HARM_MARKERS), re.IGNORECASE)
_THREAT_RE = re.compile("|".join(re.escape(m) for m in THREAT_MARKERS), re.IGNORECASE)
_ABUSE_RE = re.compile("|".join(ABUSE_PATTERNS), re.IGNORECASE)

# Compiled once. `card_like` is intentionally applied to a copy with pre-masked
# tokens removed so `__phone__` cannot be scavenged for digits.
_COMPILED: dict[str, re.Pattern[str]] = {
    name: re.compile(PII_PATTERNS[name]) for name in _PII_ORDER
}


def redact(text: str) -> tuple[str, list[str]]:
    """Return (redacted_text, list_of_pii_types_found).

    Types are returned in `_PII_ORDER`, deduplicated -- callers aggregate these
    into dashboard counts, so a stable order keeps the counts comparable across
    runs. The redacted text substitutes a typed placeholder rather than a generic
    one: `[ORDER_NUMBER]` tells the generator that an order number existed without
    telling it the number, which is exactly the information a DM_HANDOFF reply
    needs ("could you DM us your order number?").
    """
    if not text:
        return "", []

    found: list[str] = []
    redacted = text
    for name in _PII_ORDER:
        pattern = _COMPILED[name]

        def _sub(match: re.Match[str], _name: str = name) -> str:
            # Never redact inside an upstream mask.
            if _PRE_MASKED.fullmatch(match.group(0)):
                return match.group(0)
            if _name not in found:
                found.append(_name)
            return f"[{_name.upper()}]"

        redacted = pattern.sub(_sub, redacted)

    return redacted, found


def scan_injection(text: str) -> bool:
    """True if the message looks like an indirect prompt-injection attempt.

    Customer text is untrusted input flowing into a prompt -- the canonical
    indirect-injection setup (Greshake et al., AISec '23, arXiv:2302.12173), and
    LLM01, the #1 risk, for the second consecutive edition of the OWASP Top 10 for
    LLM Applications (2025).

    The other two defences are structural, not in this function:
      - customer text goes in a delimited, explicitly-labelled-untrusted block, and
        the system instruction asserts it is DATA TO CLASSIFY, NEVER INSTRUCTIONS;
      - the action space is a four-value enum, so the model cannot emit an action
        that does not exist (Beurer-Kellner et al., arXiv:2506.08837).

    This is a keyword detector and will be evaded by anything beyond the most
    casual attempt. It is reported in the failure analysis as such. Its value is
    that a hit ROUTES rather than blocks, so evasion degrades to the normal ladder
    rather than to a bypass.
    """
    if not text:
        return False
    return _INJECTION_RE.search(text) is not None


def scan_safety(text: str) -> bool:
    """True if the message contains self-harm, threats or abuse. Feeds the SAFETY VETO."""
    if not text:
        return False
    return bool(
        _SELF_HARM_RE.search(text)
        or _THREAT_RE.search(text)
        or _ABUSE_RE.search(text)
    )


def safety_categories(text: str) -> list[str]:
    """Which safety markers fired. Not on the ladder's path -- the veto only needs a
    bool -- but the handoff payload carries this so the human picking the ticket up
    knows whether they are opening a crisis message or an angry one before they
    read it.
    """
    if not text:
        return []
    cats: list[str] = []
    if _SELF_HARM_RE.search(text):
        cats.append("self_harm")
    if _THREAT_RE.search(text):
        cats.append("threat")
    if _ABUSE_RE.search(text):
        cats.append("abuse")
    return cats
