"""src/llm_client.py: provider routing, request shape, and the cache-first contract.

The SDKs are replaced with fakes in sys.modules, so these tests never open a
socket and never need a key. They pin what WE send; whether the provider accepts
it is checked by the one-message live smoke run, not here.
"""
import sys
import types

import pytest

from src import llm_client


class _Recorder:
    def __init__(self, result):
        self.calls = []
        self.result = result

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


@pytest.fixture
def fake_openai(monkeypatch):
    usage = types.SimpleNamespace(input_tokens=11, output_tokens=7)
    create = _Recorder(types.SimpleNamespace(output_text='{"ok": true}', usage=usage))

    class OpenAI:
        def __init__(self, *a, **k):
            self.responses = types.SimpleNamespace(create=create)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=OpenAI))
    return create


@pytest.fixture
def fake_gemini(monkeypatch):
    meta = types.SimpleNamespace(prompt_token_count=20, candidates_token_count=5, thoughts_token_count=0)
    generate = _Recorder(types.SimpleNamespace(text='{"criteria": {}}', usage_metadata=meta))

    class Client:
        def __init__(self, *a, **k):
            self.models = types.SimpleNamespace(generate_content=generate)

    genai_types = types.SimpleNamespace(
        GenerateContentConfig=lambda **kw: kw,
        ThinkingConfig=lambda **kw: kw,
    )
    genai = types.SimpleNamespace(Client=Client, types=genai_types)
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types)
    return generate


def test_openai_schema_call_is_strict_and_sends_no_sampling_params(fake_openai):
    schema = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": ["string", "null"]}},
              "required": ["a"], "additionalProperties": False}
    cfg = {"provider": "openai", "id": "gpt-x", "max_tokens": 100, "reasoning_effort": "minimal"}

    raw, usage = llm_client.complete(cfg, "hi", system="sys", json_schema=schema)

    sent = fake_openai.calls[0]
    assert raw == '{"ok": true}' and usage == {"input_tokens": 11, "output_tokens": 7}
    assert sent["instructions"] == "sys" and sent["max_output_tokens"] == 100
    fmt = sent["text"]["format"]
    assert fmt["strict"] is True and fmt["schema"]["required"] == ["a", "b"]
    assert sent["reasoning"] == {"effort": "minimal"}
    assert not {"temperature", "top_p", "top_k"} & set(sent)
    # the caller's schema must not be mutated by the strict rewrite
    assert schema["required"] == ["a"]


def test_openai_omits_reasoning_when_not_configured(fake_openai):
    llm_client.complete({"provider": "openai", "id": "gpt-4.1-nano"}, "hi", json_mode=True)
    sent = fake_openai.calls[0]
    assert "reasoning" not in sent
    assert sent["text"] == {"format": {"type": "json_object"}}
    assert sent["max_output_tokens"] == llm_client.DEFAULT_MAX_TOKENS


def test_gemini_json_mode_and_thinking_budget(fake_gemini):
    cfg = {"provider": "gemini", "id": "gemini-x", "max_tokens": 256, "thinking_budget": 0}
    raw, usage = llm_client.complete(cfg, "judge this", json_mode=True)

    sent = fake_gemini.calls[0]
    assert sent["model"] == "gemini-x" and sent["contents"] == "judge this"
    assert sent["config"]["response_mime_type"] == "application/json"
    assert sent["config"]["thinking_config"] == {"thinking_budget": 0}
    assert usage["input_tokens"] == 20 and usage["output_tokens"] == 5


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        llm_client.complete({"provider": "nope", "id": "m"}, "hi")


def test_strict_schema_recurses_into_nested_objects():
    schema = {"type": "object", "properties": {"inner": {
        "type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}}}}
    out = llm_client.strict_schema(schema)
    assert out["required"] == ["inner"]
    assert out["properties"]["inner"]["required"] == ["x", "y"]
    assert out["properties"]["inner"]["additionalProperties"] is False


def test_cache_hit_never_reaches_a_provider(tmp_path, monkeypatch):
    """The reproducibility contract: a cached call makes no provider call at all."""
    from src import Config, load_config
    from src import cache
    from src.judge import _call_judge

    def boom(*a, **k):
        raise AssertionError("provider called on a cache hit")

    monkeypatch.setattr(llm_client, "complete", boom)
    base = load_config()
    config = Config({**base, "cache": {**base["cache"], "path": str(tmp_path / "c.jsonl")}})
    cache.reset_index()
    try:
        prompt = "p"
        key = cache.cache_key(config.models["judge"]["id"], config.evaluation["judge"]["rubric_version"], prompt)
        cache.put(key, '{"criteria": {}}', {"input_tokens": 1}, config)
        raw, usage, hit = _call_judge(prompt, config)
        assert hit is True and raw == '{"criteria": {}}'
    finally:
        cache.reset_index()
