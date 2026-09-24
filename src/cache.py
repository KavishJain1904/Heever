"""Response cache. The reproducibility guarantee.

Contract: docs/12 §9.

Keyed by sha256(model_id + prompt_template_version + input_text) -- see
config.cache.key_fields. Committed to the repo as artifacts/llm_cache.jsonl.

This single ~24-line component fixes four problems at once (docs/00 trap 5):
slow reruns, cost, non-determinism, and the reviewer needing an API key.

CACHE-MISS BEHAVIOUR IS EXPLICIT AND LOUD: if the key is unset and the entry is
missing, raise. A silent skip turns a broken reproduction into a passing one
(docs/05 decision 18).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

from src import REPO_ROOT, load_config

_LOCK = threading.Lock()
_INDEX: dict[str, dict[str, Any]] | None = None
_INDEX_PATH: Path | None = None

# NUL is the separator because it cannot occur in any of the three key fields.
# Joining on a printable character (":", "|") makes the key forgeable by input:
# a tweet containing "|v1|" could collide with a different (model, version) pair.
_SEP = "\x00"


class CacheMissWithoutKey(RuntimeError):
    """Raised on a miss when no API key is configured. Never caught internally."""


def cache_key(model_id: str, prompt_template_version: str, input_text: str) -> str:
    """sha256 hex digest over the three key fields, joined by a NUL separator.

    Bumping `prompt_template_version` invalidates every entry by design -- a prompt
    edit that silently reused cached completions would be the worst kind of
    reproducibility bug, because the numbers would still be internally consistent.
    """
    payload = _SEP.join((model_id, prompt_template_version, input_text))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(config=None) -> Path:
    config = config or load_config()
    return REPO_ROOT / config.cache["path"]


def _has_api_key() -> bool:
    """True if any provider credential is present in the environment (or .env)."""
    from src import load_env

    load_env()
    return any(
        os.environ.get(var)
        for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "VLLM_BASE_URL")
    )


def _load_index(config=None) -> dict[str, dict[str, Any]]:
    """Read the JSONL once into memory. Later lines win on duplicate keys.

    The file is append-only, so a re-run that legitimately supersedes an entry
    appends rather than rewrites; last-write-wins is what makes that correct.
    """
    global _INDEX, _INDEX_PATH
    path = _cache_path(config)
    with _LOCK:
        if _INDEX is not None and _INDEX_PATH == path:
            return _INDEX
        index: dict[str, dict[str, Any]] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"{path}:{lineno} is not valid JSON -- the cache is a "
                            f"committed artifact and a corrupt line invalidates the "
                            f"offline reproduction: {exc}"
                        ) from exc
                    index[entry["key"]] = entry
        _INDEX = index
        _INDEX_PATH = path
        return index


def get(key: str, config=None) -> Optional[dict[str, Any]]:
    """Return the cached response, or None.

    Returns None only when a miss is RECOVERABLE -- i.e. an API key exists and the
    caller can go and fetch the thing. On a miss with no key and
    `config.cache.on_miss_without_key == "fail"`, this raises instead: returning
    None there would let a reviewer's `make eval` skip the uncached items and print
    a complete-looking table computed on a subset. That is the one failure a grader
    cannot see from the outside, so it is made impossible from the inside.
    """
    config = config or load_config()
    entry = _load_index(config).get(key)
    if entry is not None:
        return entry
    if config.cache["on_miss_without_key"] == "fail" and not _has_api_key():
        raise CacheMissWithoutKey(
            f"cache miss for key {key[:12]}... and no API key is configured.\n"
            f"  cache: {_cache_path(config)}\n"
            f"  This is deliberate: silently skipping uncached items would produce a "
            f"headline table computed on a subset of the golden set.\n"
            f"  Either set OPENAI_API_KEY / GEMINI_API_KEY to regenerate, or run `make eval` "
            f"(Tier 1), which needs no LLM call at all."
        )
    return None


def put(key: str, response: Any, usage: dict, config=None) -> None:
    """Append one JSONL line. Never rewrites; the file is append-only and sorted on commit."""
    config = config or load_config()
    path = _cache_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "key": key,
        "response": response,
        "usage": dict(usage or {}),
    }
    line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if _INDEX is not None and _INDEX_PATH == path:
            _INDEX[key] = entry


def sort_for_commit(config=None) -> int:
    """Rewrite the cache sorted by key, one entry per key. Run before committing.

    An append-only file accumulates duplicate keys across runs and lands in a diff
    in arrival order, which makes review impossible. Sorting collapses it to the
    reviewable form. Returns the number of entries written.
    """
    config = config or load_config()
    path = _cache_path(config)
    index = _load_index(config)
    if not path.exists():
        return 0
    with open(path, "w", encoding="utf-8") as fh:
        for key in sorted(index):
            fh.write(json.dumps(index[key], ensure_ascii=False, sort_keys=True) + "\n")
    return len(index)


def reset_index() -> None:
    """Drop the in-memory index. Tests use this after writing a temp cache."""
    global _INDEX, _INDEX_PATH
    with _LOCK:
        _INDEX = None
        _INDEX_PATH = None
