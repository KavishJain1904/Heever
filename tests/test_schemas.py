"""Output-contract tests. Plan: docs/15 §4."""
import pytest
from pydantic import ValidationError

from src import intent_names
from src.schemas import Action, Decision, EscalationCode


def _valid(**overrides):
    base = dict(
        intent="playback_technical", intent_confidence=0.9,
        action=Action.AUTO_SEND, reply_draft="Try reinstalling the app.",
        evidence_ids=["t_1"],
    )
    base.update(overrides)
    return base


def test_action_enum_is_closed():
    """Exactly four actions. The constrained action space is the core injection
    defence -- the model cannot 'issue a refund' because no such action exists."""
    assert {a.value for a in Action} == {
        "auto_send", "request_info", "dm_handoff", "escalate"
    }
    with pytest.raises(ValueError):
        Action("issue_refund")


def test_evidence_ids_must_be_non_empty():
    with pytest.raises(ValidationError):
        Decision(**_valid(evidence_ids=[]))


def test_escalation_reason_present_iff_escalate():
    with pytest.raises(ValidationError):
        Decision(**_valid(action=Action.ESCALATE, reply_draft=None,
                          escalation_code=EscalationCode.LOW_CONFIDENCE))
    with pytest.raises(ValidationError):
        Decision(**_valid(escalation_reason="not an escalation"))
    ok = Decision(**_valid(action=Action.ESCALATE, reply_draft=None,
                           escalation_code=EscalationCode.LOW_CONFIDENCE,
                           escalation_reason="confidence below threshold"))
    assert ok.escalation_reason


def test_reply_draft_capped_at_280_chars():
    with pytest.raises(ValidationError):
        Decision(**_valid(reply_draft="x" * 281))
    assert Decision(**_valid(reply_draft="x" * 280))


def test_intent_must_be_in_frozen_taxonomy():
    with pytest.raises(ValidationError):
        Decision(**_valid(intent="refund_but_spelled_wrong"))
    for name in intent_names():
        assert Decision(**_valid(intent=name, action=Action.ESCALATE, reply_draft=None,
                                 escalation_reason="x"))


def test_confidence_is_bounded():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValidationError):
            Decision(**_valid(intent_confidence=bad))


def test_repair_retry_is_bounded_and_logged():
    """max_repair_retries from config; every repair appears in the decision record.
    An unbounded self-repair loop is a cost bomb."""
    from src import load_config
    from src.schemas import DecisionRecord

    config = load_config()
    assert config.generation["max_repair_retries"] == 1
    assert "repair_attempts" in DecisionRecord.model_fields
    assert "validation_errors" in DecisionRecord.model_fields
