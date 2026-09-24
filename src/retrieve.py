"""Stage C (online): hybrid retrieval with brand+intent filtering.

Contract: docs/12 §4. Design: docs/02 §3.

Pipeline: BM25 || dense -> RRF(k=60) -> [optional rerank] -> MMR(lambda=0.7)
          filtered by {brand, intent} -> 3 exemplars.

Deliberately ~100 lines of our own code. No FAISS, no Chroma, no LangChain: at
100k texts a 100k x 384 float32 matmul is ~150MB and sub-100ms, and every one of
these functions must be defensible line-by-line in a live review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from src.featurize import tokenize
from src.schemas import Exemplar

# A first reply that ENDS IN A QUESTION and offers no step is a diagnostic probe,
# not a resolution. 65.9% of first replies are these. Indexing them teaches the
# system to ask questions forever.
_QUESTION_ONLY_RE = re.compile(r"\?\s*$")
_CONFIRMATION_RE = re.compile(
    r"\b(?:thank(?:s| you)|thx|that worked|it works|works now|sorted|fixed|"
    r"perfect|great,? thanks|cheers)\b",
    re.IGNORECASE,
)
_RESOLUTION_MARKERS = (
    "try", "go to", "head to", "tap", "click", "reinstall", "log out", "sign out",
    "settings", "we've", "we have", "refunded", "issued", "updated", "reset",
)


def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence], k: int = 60) -> list[tuple]:
    """Fuse heterogeneous scorers: sum of 1/(k + rank). Cormack et al., SIGIR 2009.

    RRF needs NO score normalization across scorers -- that is why it is the
    industry default and why it is right for mixing BM25 with cosine. k=60 is the
    canonical constant from the paper.

    The absence of normalisation is the whole point: BM25 scores are unbounded and
    corpus-dependent, cosine sits in [-1, 1]. Any attempt to put them on a common
    scale requires a fitted mapping that then has to be re-fitted whenever the
    corpus changes. Rank position has no such problem.

    Returns [(doc_id, fused_score)] sorted descending. Ties break on doc_id so the
    output is deterministic across runs.
    """
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))


def mmr(
    candidates: Sequence[int],
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    lambda_: float = 0.7,
    top_k: int = 3,
) -> list[int]:
    """Maximal Marginal Relevance. Carbonell & Goldstein, SIGIR 1998.

    Stops the 3 exemplars from being three near-identical "sorry about that, DM us!"
    tweets -- which is exactly what unfiltered top-3 returns on this corpus.

        MMR = argmax [ lambda * sim(d, q) - (1 - lambda) * max sim(d, d_selected) ]

    lambda=0.7 leans toward relevance; the diversity term only breaks in when two
    candidates are genuinely near-duplicates, which on this corpus is often.
    """
    if not len(candidates):
        return []
    candidates = list(candidates)
    query_vec = np.asarray(query_vec, dtype=np.float32).ravel()

    relevance = {i: float(doc_vecs[i] @ query_vec) for i in candidates}
    selected: list[int] = []
    remaining = list(candidates)

    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda i: (relevance[i], -i))
        else:
            selected_mat = doc_vecs[selected]

            def score(i: int) -> float:
                redundancy = float(np.max(selected_mat @ doc_vecs[i]))
                return lambda_ * relevance[i] - (1 - lambda_) * redundancy

            best = max(remaining, key=lambda i: (score(i), -i))
        selected.append(best)
        remaining.remove(best)
    return selected


def is_resolution(reply_text: str, follow_up_text: Optional[str] = None) -> bool:
    """Does this brand reply RESOLVE, or merely diagnose?

    The decision that matters most in this module (docs/12 §4). 65.9% of first
    replies are diagnostic questions. A precedent corpus built from 1-hop pairs
    teaches the generator that the correct response to any problem is another
    question -- which is a fluent, plausible, useless agent.

    Resolution requires a substantive step AND (where available) a customer
    confirmation. Tracing that for Spotify yielded ~11,944 clean precedents from
    43,265 replies. That 27.6% ratio is itself a finding worth reporting.
    """
    text = (reply_text or "").strip()
    if not text:
        return False
    if _QUESTION_ONLY_RE.search(text) and not any(
        m in text.lower() for m in _RESOLUTION_MARKERS
    ):
        return False
    if not any(m in text.lower() for m in _RESOLUTION_MARKERS):
        return False
    if follow_up_text is not None:
        return bool(_CONFIRMATION_RE.search(follow_up_text))
    return True


def extract_resolution_precedents(threads: Sequence[dict]) -> list[dict]:
    """Trace each thread to a substantive brand solution plus customer confirmation.

    Returns [{evidence_id, conversation_id, customer_text, brand_reply, ...}].
    This is what gets indexed -- NOT (customer, first_reply) pairs.
    """
    precedents = []
    for thread in threads:
        turns = thread.get("turns") or []
        customer_text = thread.get("customer_text") or ""
        for i, turn in enumerate(turns):
            if not turn.get("is_brand"):
                continue
            follow_up = turns[i + 1]["text"] if i + 1 < len(turns) else None
            if not is_resolution(turn.get("text", ""), follow_up):
                continue
            precedents.append({
                "evidence_id": turn["tweet_id"],
                "conversation_id": thread["conversation_id"],
                "customer_text": customer_text,
                "brand_reply": turn["text"],
                "brand_reply_normalised": turn.get("text_normalised", turn["text"]),
                "intent": thread.get("intent"),
                "brand": thread.get("brand"),
                "confirmed": follow_up is not None and bool(_CONFIRMATION_RE.search(follow_up or "")),
            })
            break  # one precedent per thread: the first genuine resolution
    return precedents


@dataclass
class RetrievalIndex:
    """The in-memory index. A vector DB here would be a dependency to justify in a
    live code review for zero measured benefit."""

    records: list[dict]
    embeddings: np.ndarray
    bm25: object

    @property
    def size(self) -> int:
        return len(self.records)

    def candidate_mask(self, brand: str, intent: Optional[str]) -> np.ndarray:
        """Hard brand filter, soft intent filter.

        `brand` is HARD: cross-brand retrieval is actively harmful -- Apple's tone
        must not leak into a Delta reply, and the whole premise is "grounded in how
        THIS brand historically resolved similar issues".
        """
        mask = np.array(
            [r.get("brand") == brand for r in self.records], dtype=bool
        )
        if intent is not None:
            mask &= np.array([r.get("intent") == intent for r in self.records], dtype=bool)
        return mask


def retrieve(
    query: str,
    intent: str,
    brand: str,
    config,
    index: Optional[RetrievalIndex] = None,
    query_vec: Optional[np.ndarray] = None,
) -> list[Exemplar]:
    """Two-stage: classify -> restrict pool to that intent -> retrieve.

    `brand` is a HARD filter. Cross-brand retrieval is actively harmful -- Apple's
    tone must not leak into a Delta reply.
    `intent` is a soft filter: keep config.retrieval.unfiltered_tail cross-intent
    hits so a misclassification is not fatal.

    IMPORTANT -- what gets indexed (docs/01 §5): index RESOLUTION precedents, not
    1-hop (customer -> first reply) pairs. 65.9% of first replies are diagnostic
    questions, not resolutions; index those and the RAG learns to ask questions
    forever. Tracing to a substantive solution PLUS a customer confirmation yielded
    only ~11,944 clean precedents for Spotify out of 43,265 replies.
    """
    if index is None or index.size == 0:
        return []

    cfg = config.retrieval
    n_cands = int(cfg["candidates_per_scorer"])

    in_intent = np.flatnonzero(index.candidate_mask(brand, intent if cfg["filter_by_intent"] else None))
    # The unfiltered tail: a few cross-intent hits from the same brand, so a
    # misclassification degrades the exemplars rather than emptying them.
    brand_only = np.flatnonzero(index.candidate_mask(brand, None))
    tail = [i for i in brand_only if i not in set(in_intent.tolist())][: int(cfg["unfiltered_tail"])]
    pool = np.array(sorted(set(in_intent.tolist()) | set(tail)), dtype=int)
    if pool.size == 0:
        return []

    # --- lexical ---
    bm25_scores = np.asarray(index.bm25.get_scores(tokenize(query)), dtype=np.float32)
    pooled_bm25 = sorted(pool.tolist(), key=lambda i: -bm25_scores[i])[:n_cands]

    # --- dense ---
    if query_vec is None:
        from src.featurize import embed
        query_vec = embed([query], cfg["encoder"])[0]
    query_vec = np.asarray(query_vec, dtype=np.float32).ravel()
    dense_scores = index.embeddings[pool] @ query_vec
    pooled_dense = [int(pool[j]) for j in np.argsort(-dense_scores)[:n_cands]]

    fused = reciprocal_rank_fusion([pooled_bm25, pooled_dense], k=int(cfg["rrf_k"]))
    if cfg["rerank"]["enabled"]:
        fused = rerank(fused, query, cfg["rerank"]["model"], index)

    bm25_rank = {doc: r for r, doc in enumerate(pooled_bm25, 1)}
    dense_rank = {doc: r for r, doc in enumerate(pooled_dense, 1)}
    fused_score = dict(fused)

    chosen = mmr(
        [doc for doc, _ in fused],
        query_vec,
        index.embeddings,
        lambda_=float(cfg["mmr_lambda"]),
        top_k=int(cfg["top_k_exemplars"]),
    )

    exemplars = []
    for i in chosen:
        record = index.records[i]
        exemplars.append(Exemplar(
            evidence_id=str(record["evidence_id"]),
            conversation_id=str(record["conversation_id"]),
            customer_text=record["customer_text"],
            brand_reply=record["brand_reply"],
            intent=record.get("intent") or "",
            bm25_rank=bm25_rank.get(i),
            dense_rank=dense_rank.get(i),
            rrf_score=float(fused_score.get(i, 0.0)),
            dense_score=float(index.embeddings[i] @ query_vec),
            selected_by_mmr=True,
        ))
    return exemplars


def rerank(candidates: Sequence[tuple], query: str, model_name: str, index: RetrievalIndex) -> list[tuple]:
    """Cross-encoder rerank. OFF by default; the ablation is reported either way.

    Cross-encoders typically add 5-20% nDCG@10, but the same literature reports
    bge-reranker-base DEGRADING nDCG by -0.3% to -3.1% on corpora far from its
    training distribution. "I tried it, measured it, it did not help on my corpus,
    here is the number" beats both using it blindly and not trying it.

    Twitter support text is about as far from a reranker's training distribution as
    text gets -- 280 chars, @-handles, no punctuation discipline -- so the
    degradation case is the one to expect here, not the improvement case.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    doc_ids = [doc for doc, _ in candidates]
    pairs = [(query, index.records[i]["customer_text"]) for i in doc_ids]
    scores = model.predict(pairs)
    return sorted(zip(doc_ids, (float(s) for s in scores)), key=lambda kv: -kv[1])


def retrieval_metrics(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 3) -> dict:
    """recall@k, MRR, nDCG@k.

    ALWAYS reported alongside groundedness, never instead of it (docs/14 §8):
    faithfulness stays high when retrieval fails, because the generator answers
    coherently from whatever partial context it got. A groundedness score without
    a retrieval score can mask a retriever that missed the document entirely.
    """
    relevant = set(relevant_ids)
    top = list(retrieved_ids)[:k]
    hits = [1.0 if d in relevant else 0.0 for d in top]

    mrr = 0.0
    for rank, d in enumerate(top, 1):
        if d in relevant:
            mrr = 1.0 / rank
            break

    dcg = sum(h / np.log2(r + 1) for r, h in enumerate(hits, 1))
    ideal = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return {
        "recall_at_k": sum(hits) / len(relevant) if relevant else 0.0,
        "mrr": mrr,
        "ndcg_at_k": dcg / ideal if ideal else 0.0,
        "k": k,
    }
