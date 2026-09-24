"""Tier 4. LLM zero-/few-shot with the frozen taxonomy as a constrained enum menu.

Every response is cached (src/cache.py), so this baseline reproduces exactly,
offline, with no API key.

Two upgrades with published support:
  - retrieval-augmented few-shot (KATE, arXiv:2101.06804);
  - PARTIAL-LABEL-SPACE RETRIEVAL -- show the model only the plausible subset of
    labels, which set few-shot SOTA on three intent datasets with no fine-tuning
    (Milios et al., GenBench 2023, arXiv:2309.10954).
Self-consistency buys a point or two at 3-5x cost; skip unless budget remains.

FORMAT NOTE: "Let Me Speak Freely?" (arXiv:2408.02442) found format restriction
degrades REASONING -- but also observed JSON-mode HELPING on classification,
because constraining the output space reduces answer-selection errors. For
single-label intent, constrained decoding to a label enum is the right call.

Cost per 1000 messages at ~460 in / ~20 out tokens (list prices checked on the
provider pricing pages, 2026-09-16): gpt-5-nano ($0.05 / $0.40 per 1M) ~= $0.03,
plus reasoning tokens, which bill as output. TF-IDF: $0 and sub-second.
That three-order-of-magnitude gap is the argument for TnT-LLM's distillation
pattern -- lightweight classifiers trained on LLM annotations match or beat using
the LLM directly, with far better scalability and transparency.
"""
from __future__ import annotations

import json
from typing import Optional, Sequence

from src import load_config
from src.cache import cache_key, get as cache_get, put as cache_put

PROMPT = """Classify this customer support message into exactly one intent.

Intents:
{menu}

Message: {text}

Return JSON: {{"intent": "<one of the above>", "confidence": <0.0-1.0>}}"""


def select_partial_label_space(text: str, taxonomy: dict, k: int = 5,
                               embeddings=None, label_vectors=None) -> list[str]:
    """Show the model only the plausible subset of labels.

    Partial-label-space retrieval set few-shot SOTA on three intent datasets with no
    fine-tuning (Milios et al., GenBench 2023). The mechanism is prosaic: a menu of
    11 labels spends context on 6 options the model was never going to pick, and
    lengthens the string it has to disambiguate within.

    Falls back to the full menu when no vectors are supplied -- degrading to the
    correct-but-slower option, never to a silently truncated menu.
    """
    names = [i["name"] for i in taxonomy["intents"]]
    if embeddings is None or label_vectors is None:
        return names
    import numpy as np

    sims = np.asarray(label_vectors) @ np.asarray(embeddings).ravel()
    top = np.argsort(-sims)[:k]
    selected = [names[i] for i in top]
    # other_unclear must always be on the menu or the model cannot decline.
    if "other_unclear" not in selected:
        selected.append("other_unclear")
    return selected


def classify(texts: Sequence[str], taxonomy: dict, config=None,
             label_subsets: Optional[Sequence[Sequence[str]]] = None) -> list[dict]:
    """Zero-shot with the frozen taxonomy as a constrained enum menu.

    Every response is cached, so this baseline reproduces exactly, offline, with no
    API key. Definitions are included in the menu because the taxonomy's own
    `not_this` boundaries are what make the classes separable -- a bare list of
    names asks the model to guess our precedence rule.
    """
    config = config or load_config()
    model_cfg = config.models["generator"]
    by_name = {i["name"]: i for i in taxonomy["intents"]}

    out = []
    for idx, text in enumerate(texts):
        names = list(label_subsets[idx]) if label_subsets else list(by_name)
        menu = "\n".join(
            f"- {n}: {(by_name[n].get('definition') or '').strip()}" for n in names
        )
        prompt = PROMPT.format(menu=menu, text=text)
        key = cache_key(model_cfg["id"], "baseline-llm-v1", prompt)

        cached = cache_get(key, config)
        if cached is not None:
            raw = cached["response"]
        else:
            from src.llm_client import complete

            # No tight max_tokens: on a reasoning model the thinking is billed
            # against the same budget, and a 128-token cap returns an empty answer.
            raw, usage = complete(model_cfg, prompt, json_mode=True)
            cache_put(key, raw, usage, config)

        try:
            payload = json.loads(raw.strip().strip("`").lstrip("json"))
            intent = payload["intent"] if payload["intent"] in by_name else "other_unclear"
            out.append({"intent": intent, "confidence": float(payload.get("confidence", 0.5))})
        except (json.JSONDecodeError, KeyError, ValueError):
            # An unparseable classification is other_unclear at zero confidence,
            # not a crash and not a guess -- it routes to a human, which is the
            # correct handling of "the classifier failed".
            out.append({"intent": "other_unclear", "confidence": 0.0})
    return out


def calibrate_on_banking77(config=None) -> dict:
    """Harness calibration ONLY. Never presented as evidence about the Twitter system.

    Run OUR EXACT ladder on Banking77 at N=8 or 16 per class and check it reproduces
    the published ordering and magnitudes. If SetFit lands near 77.9% at 8 shots,
    the implementation is sound and the Twitter numbers are trustworthy. If it lands
    at 45%, there is a bug.

    Domain mismatch is severe and unfixable -- transfer learning from Banking77 will
    not help and is not attempted (docs/01 §10).
    """
    return {
        "status": "not_run",
        "published_anchor": {"setfit_8shot_accuracy": 0.779, "source": "arXiv:2209.11055"},
        "interpretation": (
            "A pass means the HARNESS is sound. It says nothing about performance on "
            "Twitter support text and must never be quoted as though it did."
        ),
        "how_to_run": (
            "load PolyAI/banking77, sample N per class, run baselines.tfidf.tune, "
            "baselines.embed_lr.cross_validate and baselines.setfit.train, and "
            "compare the ORDERING against the published numbers."
        ),
    }
