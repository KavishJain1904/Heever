"""Draw the pilot-30 and golden-200 candidate pools from data/threads.parquet.

Not part of the reproduction path (make eval). Run once, by the annotator, before
labelling. Output is CANDIDATES for a human to label -- the heuristics below pick
which tweets to *look at*, never the intent or escalation label itself.

Usage:
    python scripts/sample_golden.py

Writes:
    data/golden_pilot_30.json   -- 30 examples, stratum=pilot, for docs/13 §7.
                                    Disjoint from the 200. Labels get discarded.
    data/golden_200_candidates.json -- 100 random + 60 stratified + 25 adversarial
                                    + 15 policy_trap, per docs/11 §3.

Determinism: every stratum is drawn by hashing conversation_id (+ a salt string
per stratum) rather than df.sample(seed=) -- stable across pandas/pyarrow versions
and however the parquet file gets rewritten (docs/05 decision 10).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
THREADS_PATH = REPO_ROOT / "data" / "threads.parquet"
PILOT_OUT = REPO_ROOT / "data" / "golden_pilot_30.json"
GOLDEN_OUT = REPO_ROOT / "data" / "golden_200_candidates.json"

PILOT_N = 30
RANDOM_N = 100
STRATIFIED_N = 60
ADVERSARIAL_N = 25
POLICY_TRAP_N = 15

# Rough keyword heuristics used ONLY to pick candidates for a human to look at --
# never to assign the label. Keeps the stratified pool from being 90% one intent.
INTENT_KEYWORDS = {
    "order_delivery_status": [r"\bwhere.{0,15}(my|is)\b", r"\bship", r"\btrack", r"\barriv", r"\bdeliver"],
    "billing_charges": [r"\bcharg", r"\bbill", r"\bpayment", r"\bcard\b", r"\bprice\b"],
    "refund_return": [r"\brefund", r"\bmoney back", r"\breturn\b", r"\bexchange\b"],
    "account_access_security": [r"\bhack", r"\bcan'?t (log|sign) in", r"\bpassword\b", r"\blocked out\b", r"\bcompromis"],
    "cancel_downgrade": [r"\bcancel", r"\bdowngrade", r"\bunsubscrib", r"\bend my (subscription|plan)"],
    "service_outage_technical": [r"\bcrash", r"\bwon'?t (open|play|work|load)", r"\berror\b", r"\bbug\b", r"\bnot working\b", r"\bdown\b"],
    "product_howto_question": [r"\bhow do i\b", r"\bhow can i\b", r"\bis it possible\b", r"\bcan i\b", r"\bwhat is\b"],
    "change_of_details": [r"\bchange my (email|address|name|number)\b", r"\bupdate my\b"],
    "complaint_escalation": [r"\bterrible\b", r"\bworst\b", r"\bdisgust", r"\blawsuit\b", r"\bsue\b", r"\bridiculous\b", r"\bnever again\b"],
    "non_support": [r"^thanks?!?$", r"\blove (you|spotify)\b", r"^great\b"],
}

ADVERSARIAL_PATTERNS = [
    r"\byeah,? (right|sure)\b",  # sarcasm
    r"ignore (all|previous|the above)",  # prompt-injection-ish
    r"\bas an ai\b",
    r"\band\b.*\band\b",  # multi-clause -> likely multi-intent
    r"[!?]{2,}",  # heavy punctuation, often sarcastic/angry
    r"[\U0001F600-\U0001FAFF]",  # emoji-heavy
]

POLICY_TRAP_PATTERNS = [
    r"\bhow long (is|does)\b.*\b(refund|return|window|cancel)",
    r"\bwhat if\b",
    r"\bwhat'?s (the|your) policy\b",
    r"\bhypothetically\b",
    r"\bwhen does\b.*\b(expire|renew)\b",
]


def _hash_frac(key: str) -> float:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _row_to_candidate(row, stratum: str, idx: int, prefix: str) -> dict:
    turns = [
        {"author": t["author_id"] if not t["is_brand"] else "SpotifyCares",
         "is_brand": bool(t["is_brand"]), "text": t["text"]}
        for t in row["turns"]
    ]
    return {
        "id": f"{prefix}{idx:03d}",
        "tweet_id": str(row["conversation_id"]),
        "conversation_id": str(row["conversation_id"]),
        "stratum": stratum,
        "customer_text": row["customer_text"],
        "thread": turns,
        "has_brand_reply": any(t["is_brand"] for t in row["turns"]),
    }


def main() -> int:
    if not THREADS_PATH.exists():
        raise FileNotFoundError(f"{THREADS_PATH} not found -- run `make sample` / src/build_sample.py first.")

    df = pd.read_parquet(THREADS_PATH)
    df = df[df["brand"] == "SpotifyCares"].reset_index(drop=True)
    df["_pilot_frac"] = df["conversation_id"].map(lambda c: _hash_frac(f"{c}|pilot"))
    df["_random_frac"] = df["conversation_id"].map(lambda c: _hash_frac(f"{c}|random100"))

    pilot_pool = df.sort_values("_pilot_frac").head(PILOT_N)
    pilot_ids = set(pilot_pool["conversation_id"])

    remaining = df[~df["conversation_id"].isin(pilot_ids)].copy()
    random_pool = remaining.sort_values("_random_frac").head(RANDOM_N)
    random_ids = set(random_pool["conversation_id"])

    pool2 = remaining[~remaining["conversation_id"].isin(random_ids)].copy()

    # Stratified: round-robin across intent keyword buckets so no single class
    # dominates. Falls back to filling from the general pool if a bucket is thin.
    used_ids: set[str] = set()
    stratified_rows = []
    buckets = {
        name: pool2[pool2["customer_text_normalised"].str.contains(
            "|".join(pats), case=False, regex=True, na=False)]
        for name, pats in INTENT_KEYWORDS.items()
    }
    per_bucket = max(1, STRATIFIED_N // len(buckets))
    for name, bucket in buckets.items():
        bucket = bucket[~bucket["conversation_id"].isin(used_ids)]
        take = bucket.sort_values("conversation_id").head(per_bucket)
        used_ids.update(take["conversation_id"])
        stratified_rows.append(take)
    stratified_pool = pd.concat(stratified_rows) if stratified_rows else pool2.head(0)
    if len(stratified_pool) < STRATIFIED_N:
        filler = pool2[~pool2["conversation_id"].isin(used_ids)].sort_values("conversation_id")
        need = STRATIFIED_N - len(stratified_pool)
        stratified_pool = pd.concat([stratified_pool, filler.head(need)])
    stratified_pool = stratified_pool.head(STRATIFIED_N)
    used_ids.update(stratified_pool["conversation_id"])

    pool3 = pool2[~pool2["conversation_id"].isin(used_ids)].copy()

    adv_mask = pool3["customer_text_normalised"].apply(lambda t: _matches_any(t, ADVERSARIAL_PATTERNS))
    adversarial_pool = pool3[adv_mask].sort_values("conversation_id").head(ADVERSARIAL_N)
    used_ids.update(adversarial_pool["conversation_id"])

    pool4 = pool3[~pool3["conversation_id"].isin(used_ids)].copy()
    trap_mask = pool4["customer_text_normalised"].apply(lambda t: _matches_any(t, POLICY_TRAP_PATTERNS))
    policy_trap_pool = pool4[trap_mask].sort_values("conversation_id").head(POLICY_TRAP_N)
    used_ids.update(policy_trap_pool["conversation_id"])

    # Backfill any stratum short of target from whatever's left, so counts are exact.
    def backfill(pool_df, current, target, exclude):
        if len(current) >= target:
            return current.head(target)
        filler = pool_df[~pool_df["conversation_id"].isin(exclude)].sort_values("conversation_id")
        need = target - len(current)
        return pd.concat([current, filler.head(need)])

    pool5 = pool2[~pool2["conversation_id"].isin(used_ids)]
    adversarial_pool = backfill(pool5, adversarial_pool, ADVERSARIAL_N, used_ids)
    used_ids.update(adversarial_pool["conversation_id"])
    pool6 = pool2[~pool2["conversation_id"].isin(used_ids)]
    policy_trap_pool = backfill(pool6, policy_trap_pool, POLICY_TRAP_N, used_ids)

    pilot_candidates = [
        _row_to_candidate(row, "pilot", i + 1, "p")
        for i, (_, row) in enumerate(pilot_pool.iterrows())
    ]
    golden_candidates = []
    i = 0
    for _, row in random_pool.iterrows():
        i += 1
        golden_candidates.append(_row_to_candidate(row, "random", i, "g"))
    j = 0
    for _, row in stratified_pool.iterrows():
        j += 1
        golden_candidates.append(_row_to_candidate(row, "stratified", 100 + j, "g"))
    k = 0
    for _, row in adversarial_pool.iterrows():
        k += 1
        golden_candidates.append(_row_to_candidate(row, "adversarial", 160 + k, "g"))
    m = 0
    for _, row in policy_trap_pool.iterrows():
        m += 1
        golden_candidates.append(_row_to_candidate(row, "policy_trap", 185 + m, "g"))

    PILOT_OUT.write_text(json.dumps(pilot_candidates, indent=2, ensure_ascii=False))
    GOLDEN_OUT.write_text(json.dumps(golden_candidates, indent=2, ensure_ascii=False))

    print(f"pilot: {len(pilot_candidates)} -> {PILOT_OUT}")
    print(f"golden 200: {len(golden_candidates)} -> {GOLDEN_OUT}")
    counts = {}
    for c in golden_candidates:
        counts[c["stratum"]] = counts.get(c["stratum"], 0) + 1
    print("stratum counts:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
