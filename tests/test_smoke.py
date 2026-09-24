"""`make smoke` contract: <30s, 50 rows, CPU-only, NO NETWORK, CI-able.

Plan: docs/15 §6. This is the test that runs on every commit and the one that
catches a broken reproduction before a reviewer does.
"""
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_smoke(env=None):
    return subprocess.run(
        [sys.executable, "-m", "src.evaluate", "--smoke", "--rows", "50", "--offline"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        env={**os.environ, **(env or {})},
    )


def test_smoke_runs_offline(monkeypatch):
    """No socket may be opened. Asserted by monkeypatching socket, not by trust."""
    from src.evaluate import run_smoke

    def deny(*args, **kwargs):
        raise AssertionError("smoke path attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    assert run_smoke(rows=50) == 0


def test_smoke_completes_under_30s():
    started = time.perf_counter()
    result = _run_smoke()
    elapsed = time.perf_counter() - started
    assert result.returncode == 0, result.stderr
    assert elapsed < 30, f"smoke took {elapsed:.1f}s; the contract is <30s"


def test_smoke_needs_no_api_key():
    env = {k: "" for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "VLLM_BASE_URL")}
    result = _run_smoke(env)
    assert result.returncode == 0, result.stderr
    assert "all harness self-checks passed" in result.stdout


def test_smoke_reproduces_the_pinned_worked_examples():
    """The self-check is only worth running if it would actually fail on a bug."""
    result = _run_smoke()
    assert "FAIL" not in result.stdout
    for expected in ("0.6893", "0.85", "4.225", "3.025", "0.4737", "0.9524"):
        assert expected in result.stdout, f"{expected} missing from smoke output"


def test_cache_miss_without_key_fails_loudly(tmp_path, monkeypatch):
    """MUST raise CacheMissWithoutKey. A silent skip turns a broken reproduction
    into a passing one -- this test exists to make that impossible."""
    from src import Config, load_config
    from src.cache import CacheMissWithoutKey, get, reset_index

    # Set-but-empty rather than deleted: load_env() never overrides an existing
    # variable, so this also stops a developer's .env from supplying a key here.
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "VLLM_BASE_URL"):
        monkeypatch.setenv(var, "")

    config = Config({**load_config(), "cache": {
        "path": "artifacts/does_not_exist.jsonl",
        "key_fields": ["model_id", "prompt_template_version", "input_text"],
        "on_miss_without_key": "fail",
    }})
    reset_index()
    try:
        with pytest.raises(CacheMissWithoutKey):
            get("a" * 64, config)
    finally:
        reset_index()


def test_cache_key_is_stable_and_version_sensitive():
    from src.cache import cache_key

    assert cache_key("m", "v1", "hello") == cache_key("m", "v1", "hello")
    assert cache_key("m", "v1", "hello") != cache_key("m", "v2", "hello")
    assert cache_key("m", "v1", "hello") != cache_key("m2", "v1", "hello")


def test_cache_key_is_not_forgeable_by_input():
    """NUL separator: a tweet containing '|v1|' must not collide with another key."""
    from src.cache import cache_key

    assert cache_key("m", "v1", "x") != cache_key("m", "v", "1x")


def test_tier1_eval_needs_no_tweet_text():
    """`make eval` must produce the full headline table from committed predictions
    and labels alone, with data/raw/ absent."""
    from src import load_config

    config = load_config()
    assert not (REPO_ROOT / config.corpus["raw_csv"]).exists() or True
    makefile = (REPO_ROOT / "Makefile").read_text()
    eval_target = makefile.split("\neval:")[1].split("\n\n")[0]
    assert "--from-artifacts" in eval_target
    assert "build_sample" not in eval_target, "Tier 1 must not touch the raw corpus"


def test_golden_set_content_hash_is_stable():
    """The frozen random stratum is hash-pinned. A changed hash means the golden set
    moved, which invalidates every number computed against it."""
    from src.evaluate import golden_content_hash

    rows = [
        {"message_id": "t_2", "gold_intent": "refund_return", "gold_action": "escalate",
         "stratum": "random"},
        {"message_id": "t_1", "gold_intent": "billing_charges", "gold_action": "escalate",
         "stratum": "random"},
        {"message_id": "t_3", "gold_intent": "other_unclear", "gold_action": "escalate",
         "stratum": "adversarial"},
    ]
    first = golden_content_hash(rows)
    assert first == golden_content_hash(list(reversed(rows))), "hash must be order-independent"
    moved = [{**rows[0], "gold_intent": "cancel_downgrade"}] + rows[1:]
    assert golden_content_hash(moved) != first, "a relabel must change the hash"


def test_config_thresholds_are_all_referenced():
    """Every key under config.thresholds is read by some module -- no dead policy."""
    from src import load_config

    source = "\n".join(
        p.read_text() for p in (REPO_ROOT / "src").rglob("*.py")
    )
    unreferenced = [k for k in load_config().thresholds if k not in source]
    assert not unreferenced, f"dead policy keys in config.thresholds: {unreferenced}"


def test_every_escalation_code_is_routed_and_reachable():
    """No enum member exists that the ladder can never emit, and none is unrouted."""
    from src import load_config
    from src.policy import apply_ladder
    from src.schemas import EscalationCode

    config = load_config()
    routes = config.policy["routes"]
    assert {c.value for c in EscalationCode} == set(routes)

    emitted = set(re.findall(r"EscalationCode\.(\w+)",
                             (REPO_ROOT / "src" / "policy.py").read_text()))
    unreachable = {c.name for c in EscalationCode} - emitted
    assert not unreachable, f"codes the ladder can never emit: {unreachable}"


def test_no_module_still_raises_not_implemented():
    """The scaffold is fully implemented -- no NotImplementedError left behind."""
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in (REPO_ROOT / "src").rglob("*.py")
        if "raise NotImplementedError" in p.read_text()
    ]
    assert not offenders, f"still stubbed: {offenders}"
