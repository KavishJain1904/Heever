"""Tier 0. Establishes what accuracy MEANS under imbalance.

Report accuracy AND macro-F1 for these: a majority predictor gets macro-F1 near
zero, and that contrast is precisely the point. If the majority class is 30%, a
55%-accurate model is not obviously good.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Sequence

DM_US_REPLY = "Sorry to hear that! Please DM us so we can help."


def majority_class(y_train: Sequence[str], X_test: Sequence) -> list[str]:
    """Predict the training majority class for everything.

    Report accuracy AND macro-F1 side by side: this scores respectably on accuracy
    and near-zero on macro-F1, and that gap is the entire reason macro-F1 is the
    headline metric rather than accuracy.
    """
    if not len(y_train):
        raise ValueError("majority_class needs a non-empty training set")
    label = Counter(y_train).most_common(1)[0][0]
    return [label] * len(X_test)


def random_by_prior(y_train: Sequence[str], X_test: Sequence, seed: int = 42) -> list[str]:
    """Sample from the training class prior. Seeded, so the baseline is reproducible.

    Strictly harder to beat than uniform-random on macro-F1, because it gets rare
    classes occasionally right by construction. Using uniform-random here would be
    quietly rigging the comparison in our own favour.
    """
    counts = Counter(y_train)
    labels = sorted(counts)
    weights = [counts[l] for l in labels]
    rng = random.Random(seed)
    return rng.choices(labels, weights=weights, k=len(X_test))


def constant_dm_us(X_test: Sequence) -> list[str]:
    """The DEGENERATE CONTROL. Always emits "Sorry to hear that! Please DM us".

    Not a classifier -- a reply baseline, and the most important one in the repo.
    A system that always says this scores extremely well on any reply-similarity
    metric and is worthless. We report: the corpus frequency of the pattern, our
    system's rate of producing it, and this baseline's score, side by side.
    Present in every reply-quality table, permanently (docs/04 §8 item 6).
    """
    return [DM_US_REPLY] * len(X_test)


def dm_us_corpus_frequency(brand_replies: Sequence[str]) -> dict:
    """How often the brand itself sends essentially this reply.

    The number that makes the degenerate control interpretable: if the brand says
    some version of "DM us" in 40% of replies, then a model that learned to say it
    has learned the single most common thing in its training distribution, and a
    reply-quality metric that rewards it is measuring mimicry.
    """
    import re

    pattern = re.compile(r"\b(?:dm|direct message|message) us\b|\bsend us a (?:dm|direct message)\b",
                         re.IGNORECASE)
    hits = sum(1 for r in brand_replies if pattern.search(r or ""))
    return {
        "n": len(brand_replies),
        "hits": hits,
        "rate": hits / len(brand_replies) if brand_replies else 0.0,
    }
