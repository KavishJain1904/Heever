"""Stage 3: embeddings and BM25 index. Cached and committed.

Contract: docs/12 §3.

Chunking is a non-issue -- tweets are <=280 chars, one tweet = one unit. Stated
explicitly because it shows we know WHY the usual RAG machinery does not apply
rather than cargo-culting it.

Embeddings are committed as float16 (a few MB) so `make eval-full` never re-embeds.
MiniLM is 80MB and runs in minutes on laptop CPU -- a DGX does not help here.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9#]+")
_WS_RE = re.compile(r"\s+")


def embed(texts: Sequence[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """Encode with sentence-transformers. Returns float32 (N, 384).

    Note on alternatives: bge/gte/e5 are stronger on MTEB but REQUIRE the right
    query/passage prefixes ("query:"/"passage:" for e5). Getting the prefix wrong
    silently costs more than the model gains. MiniLM needs no prefix.

    Vectors are L2-normalised on the way out, so downstream cosine similarity is a
    plain dot product and MMR does not have to renormalise per comparison.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 2000,
    )
    return vectors.astype(np.float32)


def save_embeddings(array: np.ndarray, path: str | Path, dtype: str = "float16") -> Path:
    """Persist as float16 to keep the committed artifact small.

    float16 costs ~3 decimal digits of precision on a unit-norm vector, which is
    well below the gap between adjacent neighbours in this corpus. It halves a
    100k x 384 artifact from 150MB to 75MB, which is the difference between an
    artifact that can be committed and one that cannot.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(dtype))
    return path


def load_embeddings(path: str | Path) -> np.ndarray:
    """Read back as float32. Cosine on float16 accumulates visible error at 100k rows."""
    return np.load(Path(path)).astype(np.float32)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, '#' retained.

    Hashtags carry real signal here -- '#Error503' and '#spotifydown' are exactly
    the exact-match tokens BM25 exists to catch, and stripping '#' merges them with
    unrelated prose.
    """
    return _TOKEN_RE.findall((text or "").lower())


def build_bm25(texts: Sequence[str], k1: float, b: float):
    """Build the lexical index.

    BM25 is LOAD-BEARING here, not a formality: support text is full of exact
    tokens embeddings smear -- order numbers, error codes, "iOS 17.2", #Error503.
    A dense-only retriever maps "iOS 17.2" and "iOS 16.4" to nearly the same point,
    which is precisely wrong when the version number is the bug.
    """
    from rank_bm25 import BM25Okapi

    return BM25Okapi([tokenize(t) for t in texts], k1=k1, b=b)


def dedupe_near_identical(texts: Sequence[str]) -> tuple[list[int], dict]:
    """Hash normalised text to drop brand macros and bot output.

    ~0.75% exact text dupes in a 500k sample. This is the retrieval-leakage hazard
    -- not duplicate rows, of which there are zero.

    Returns (kept_indices, stats). The leakage concern is specific: a macro reply
    appearing 400 times will be retrieved for almost any query, and if one copy is
    in the golden set the system is graded on a precedent it also retrieved.
    """
    seen: dict[str, int] = {}
    kept: list[int] = []
    duplicates = 0
    for i, text in enumerate(texts):
        normalised = _WS_RE.sub(" ", (text or "").strip().lower())
        digest = hashlib.md5(normalised.encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        seen[digest] = i
        kept.append(i)
    return kept, {
        "n_input": len(texts),
        "n_kept": len(kept),
        "n_duplicates": duplicates,
        "duplicate_rate": duplicates / len(texts) if texts else 0.0,
    }
