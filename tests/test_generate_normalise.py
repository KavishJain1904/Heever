"""The generator's one deterministic clean-up: stray escalation_reason on non-escalate."""
import json

from src import generate as gen


def _payload(**overrides):
    base = {
        "intent": "playback_technical", "intent_confidence": 0.8, "action": "auto_send",
        "reply_draft": "Try logging out and back in.", "evidence_ids": ["t_1"],
        "escalation_reason": "none needed",
    }
    return {**base, **overrides}


def test_stray_reason_dropped_and_logged():
    notes = []
    out = gen._normalise_payload(_payload(), notes)
    assert out["escalation_reason"] is None
    assert notes == ["normalised: dropped escalation_reason on non-escalate action"]


def test_escalate_keeps_its_reason():
    notes = []
    out = gen._normalise_payload(_payload(action="escalate", escalation_reason="refund over ceiling"), notes)
    assert out["escalation_reason"] == "refund over ceiling"
    assert notes == []


def test_generate_accepts_stray_reason_without_a_repair_call(monkeypatch, config):
    calls = []

    def fake_call(system, user, cfg, model_key="generator"):
        calls.append(user)
        return json.dumps(_payload()), {"input_tokens": 1, "output_tokens": 1}, False

    monkeypatch.setattr(gen, "_call_model", fake_call)
    decision, _, _, _, errors, attempts = gen.generate(("sys", "user"), config)
    assert decision.reply_draft == "Try logging out and back in."
    assert attempts == 0 and len(calls) == 1
    assert errors == ["normalised: dropped escalation_reason on non-escalate action"]
