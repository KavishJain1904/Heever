"""One place that talks to model providers. Every LLM call in src/ goes through here.

Callers stay cache-first (src/cache.py): they build the key, check the cache, and
only call `complete()` on a miss. This module never touches the cache itself, so
the reproducibility guarantee lives in exactly one file.

Providers, chosen per role in config.yaml `models:`:
  openai     -- generator, LLM baseline, taxonomy naming, playbook distiller
  gemini     -- judge. A different family from the generator by design: self-
                preference is linearly correlated with self-recognition
                (arXiv:2404.13076), docs/04 §5.
  anthropic  -- the original generator; kept so the Claude path still runs
  local-vllm -- open-weight judge/distiller on the DGX experiment path

No sampling parameters are sent. Determinism comes from the committed cache, not
from temperature=0, which reasoning models reject anyway (docs/10 §6).
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Optional

DEFAULT_MAX_TOKENS = 4096


def complete(
    model_cfg: dict,
    prompt: str,
    *,
    system: Optional[str] = None,
    json_schema: Optional[dict] = None,
    json_mode: bool = False,
    max_tokens: Optional[int] = None,
) -> tuple[str, dict]:
    """One uncached call. Returns (raw_text, usage).

    `json_schema` asks for constrained decoding against that schema; `json_mode`
    only asks for syntactically valid JSON. Callers still validate what comes back.
    """
    from src import load_env

    load_env()
    provider = model_cfg["provider"]
    max_tokens = max_tokens or model_cfg.get("max_tokens") or DEFAULT_MAX_TOKENS
    if provider == "openai":
        return _openai(model_cfg, prompt, system, json_schema, json_mode, max_tokens)
    if provider == "gemini":
        return _gemini(model_cfg, prompt, system, json_schema, json_mode, max_tokens)
    if provider == "anthropic":
        return _anthropic(model_cfg, prompt, system, json_schema, max_tokens)
    if provider == "local-vllm":
        return _vllm(model_cfg, prompt, system, json_mode, max_tokens)
    raise ValueError(f"unknown provider {provider!r} for model {model_cfg.get('id')!r}")


def strict_schema(schema: dict) -> dict:
    """OpenAI strict mode wants every property listed in `required`.

    Optional fields are already expressed as nullable types (["string", "null"]),
    so listing them as required changes nothing about what a valid answer is -- it
    only makes the model emit an explicit null instead of omitting the key.
    """
    out = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(out)
    return out


def _openai(model_cfg, prompt, system, json_schema, json_mode, max_tokens):
    from openai import OpenAI

    kwargs: dict[str, Any] = {
        "model": model_cfg["id"],
        "input": prompt,
        "max_output_tokens": max_tokens,
    }
    if system:
        kwargs["instructions"] = system
    if json_schema is not None:
        kwargs["text"] = {"format": {
            "type": "json_schema", "name": "output", "strict": True,
            "schema": strict_schema(json_schema),
        }}
    elif json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}
    # Reasoning tokens are billed as output and count against max_output_tokens,
    # so an unset effort on a reasoning model can spend the whole budget thinking
    # and return nothing. Only sent when configured: non-reasoning models reject it.
    if model_cfg.get("reasoning_effort"):
        kwargs["reasoning"] = {"effort": model_cfg["reasoning_effort"]}

    response = OpenAI().responses.create(**kwargs)
    usage = getattr(response, "usage", None)
    return response.output_text, {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
    }


def _gemini(model_cfg, prompt, system, json_schema, json_mode, max_tokens):
    from google import genai
    from google.genai import types

    config_kwargs: dict[str, Any] = {"max_output_tokens": max_tokens}
    if system:
        config_kwargs["system_instruction"] = system
    if json_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_json_schema"] = json_schema
    elif json_mode:
        config_kwargs["response_mime_type"] = "application/json"
    if model_cfg.get("thinking_budget") is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=model_cfg["thinking_budget"]
        )

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=model_cfg["id"], contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    meta = getattr(response, "usage_metadata", None)
    return response.text or "", {
        "input_tokens": getattr(meta, "prompt_token_count", 0) or 0,
        "output_tokens": getattr(meta, "candidates_token_count", 0) or 0,
        "thinking_tokens": getattr(meta, "thoughts_token_count", 0) or 0,
    }


def _anthropic(model_cfg, prompt, system, json_schema, max_tokens):
    import anthropic

    kwargs: dict[str, Any] = {
        "model": model_cfg["id"],
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    if json_schema is not None:
        kwargs["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
    response = anthropic.Anthropic().messages.create(**kwargs)
    raw = "".join(b.text for b in response.content if b.type == "text")
    return raw, {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def _vllm(model_cfg, prompt, system, json_mode, max_tokens):
    import urllib.request

    base = os.environ.get("VLLM_BASE_URL", "http://localhost:8000").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    payload: dict[str, Any] = {"model": model_cfg["id"], "messages": messages, "max_tokens": max_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        f"{base}/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read())
    return body["choices"][0]["message"]["content"], body.get("usage", {})
