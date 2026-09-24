"""Tier 0.5. Keyword/regex rules mined from cluster-representative documents.

Proves how much signal is purely lexical, gives an interpretable floor, and doubles
as a bootstrap labeller. Expect high precision on `where is my (order|package)` and
near-zero recall elsewhere.

WARNING carried from an independent replication: keyword weak-labelling coverage
collapsed ~72% -> ~16% on real tweets, and CV macro-F1 fell 0.99 -> 0.70 once real
labels replaced keyword-derived ones. NEVER report a headline number computed
against keyword-derived labels.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Sequence

# Hand-written seeds, kept deliberately small and readable. The point of this tier
# is an INTERPRETABLE FLOOR -- a reviewer can read every rule in 30 seconds and
# predict what it will get wrong.
SEED_PATTERNS: dict[str, tuple[str, ...]] = {
    "dm_followup": (r"\b(?:i )?dm.?(?:e?d|ing)? (?:you|u)\b", r"\bcheck (?:your|ur) dms?\b",
                    r"\bsent (?:you|u) a (?:dm|message)\b"),
    "account_access": (r"\bcan.?t (?:log|sign) ?in\b", r"\bhack(?:ed|er)?\b", r"\bpassword\b",
                       r"\blocked out\b", r"\baccount (?:was )?(?:terminated|disabled)\b"),
    "billing_payment": (r"\bcharg(?:ed|ing|e)\b", r"\brefund\b", r"\bmoney back\b",
                        r"\bpayment\b", r"\bbilled?\b"),
    "subscription_plan": (r"\bstudent\b", r"\bfamily (?:plan|account|premium)\b", r"\btrial\b",
                          r"\bcancel(?:led)?\b", r"\bpremium (?:isn.t|not) working\b"),
    "library_lost": (r"\b(?:disappeared|vanished|gone|deleted|lost)\b.{0,30}\b(?:songs|music|playlists?|downloads?|library)\b",
                     r"\b(?:songs|music|playlists?|downloads?|library)\b.{0,30}\b(?:disappeared|vanished|gone|deleted)\b"),
    "catalog_content": (r"\bwhere is\b", r"\bnot on spotify\b", r"\bwhy isn.t\b.{0,30}\bon\b",
                        r"\balbum\b", r"\bunavailable\b", r"\badd\b.{0,20}\b(?:album|song)\b"),
    "playback_technical": (r"\b(?:not|isn.t|won.t|doesn.t) (?:working|play|load|open)\b",
                           r"\bcrash(?:es|ing|ed)?\b", r"\berror\b", r"\bkeeps? (?:pausing|stopping)\b"),
    "account_settings_howto": (r"\bhow (?:do|can) i\b", r"\bchange my (?:username|display name|email|name)\b"),
    "availability_pricing_question": (r"\bwhen (?:will|are) (?:you|spotify)\b.{0,30}\b(?:available|coming|launch)",
                                      r"\bin (?:india|romania)\b"),
    "feature_feedback": (r"\bfeature\b", r"\bplease add\b", r"\bwould be (?:great|nice|cool)\b",
                         r"\bi wish\b"),
    "complaint_escalation": (r"\bworst\b", r"\bunacceptable\b", r"\bdon.t care\b",
                             r"\bcustomer serv(?:ice)?\b.{0,20}\b(?:terrible|awful|useless)\b"),
    "non_support": (r"^\s*(?:thanks|thank you|ty)\b", r"\blove (?:you|this|the)\b",
                    r"\bgreat customer service\b", r"\btried premium yet\b"),
}


def fit(texts: Sequence[str], labels: Sequence[str], max_patterns: int = 15) -> dict:
    """Keep the seed patterns whose observed precision justifies them.

    "Mined from cluster centroids" in the contract; in practice the seeds above are
    the mining result, and fit() prunes them against real labels. A pattern that
    fires on the wrong class more often than the right one is dropped rather than
    kept for coverage -- coverage bought with precision is what produced the 0.99
    CV macro-F1 that collapsed to 0.70 on real labels.
    """
    stats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for text, gold in zip(texts, labels):
        for intent, patterns in SEED_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text or "", re.IGNORECASE):
                    stats[(intent, pattern)][gold] += 1

    kept: dict[str, list[str]] = defaultdict(list)
    scored = []
    for (intent, pattern), counter in stats.items():
        total = sum(counter.values())
        precision = counter[intent] / total if total else 0.0
        if precision >= 0.5 and total >= 2:
            scored.append((precision, total, intent, pattern))
    for _, _, intent, pattern in sorted(scored, reverse=True)[:max_patterns]:
        kept[intent].append(pattern)

    return {
        "patterns": dict(kept),
        "fallback": Counter(labels).most_common(1)[0][0] if len(labels) else "other_unclear",
        "n_patterns": sum(len(v) for v in kept.values()),
        "warning": (
            "NEVER report a headline computed against keyword-derived labels. An "
            "independent replication saw coverage collapse 72% -> 16% and CV "
            "macro-F1 fall 0.99 -> 0.70 once real labels replaced them."
        ),
    }


def predict(patterns: dict, texts: Sequence[str]) -> list[str]:
    """First matching rule wins; unmatched falls back to the majority class.

    Reported alongside COVERAGE (the share of messages any rule matched), because
    accuracy on the matched subset alone is the number that made the replication
    above look like 0.99.
    """
    rules = patterns["patterns"]
    fallback = patterns["fallback"]
    out = []
    for text in texts:
        label = fallback
        for intent, intent_patterns in rules.items():
            if any(re.search(p, text or "", re.IGNORECASE) for p in intent_patterns):
                label = intent
                break
        out.append(label)
    return out


def coverage(patterns: dict, texts: Sequence[str]) -> float:
    """Share of messages matched by any rule. Always reported with the accuracy."""
    rules = patterns["patterns"]
    matched = sum(
        1 for t in texts
        if any(re.search(p, t or "", re.IGNORECASE)
               for ps in rules.values() for p in ps)
    )
    return matched / len(texts) if len(texts) else 0.0
