"""Pydantic models for every structured object crossing a module boundary.

Contract: docs/11-data-contracts.md. This module is the single source of truth --
if a field is not here, it is not in the contract.

Nothing in this module does I/O. Nothing in this module calls an LLM.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator  # noqa: F401


class Action(str, Enum):
    """The complete action space. Constrained by design.

    The model *cannot* "issue a refund" because these four values are the only
    emittable actions -- the core of the constrained-action-space defence against
    prompt injection (docs/02 §7).

    DM_HANDOFF is a first-class action, not a failure: on a public channel, moving
    to DM is the correct path whenever the next step needs PII (docs/05 decision 7).
    """

    AUTO_SEND = "auto_send"
    REQUEST_INFO = "request_info"
    DM_HANDOFF = "dm_handoff"
    ESCALATE = "escalate"


class EscalationCode(str, Enum):
    """Machine-readable escalation reasons. Keys into config.yaml `policy.routes`.

    The code is what dashboards aggregate ("41% of escalations are NO_PRECEDENT ->
    that is a knowledge-base gap"). The human-readable reason string is separate.
    """

    SAFETY_RISK = "SAFETY_RISK"
    LEGAL_THREAT = "LEGAL_THREAT"
    ACCOUNT_SECURITY = "ACCOUNT_SECURITY"
    SUSPECTED_INJECTION = "SUSPECTED_INJECTION"
    PII_IN_PUBLIC_TWEET = "PII_IN_PUBLIC_TWEET"
    MONEY_ABOVE_THRESHOLD = "MONEY_ABOVE_THRESHOLD"
    NO_PRECEDENT = "NO_PRECEDENT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    REPEAT_CONTACT = "REPEAT_CONTACT"
    UNGROUNDED_COMMITMENT = "UNGROUNDED_COMMITMENT"


class PolicyLayer(str, Enum):
    """Which rung of the ladder produced the decision. Always recorded.

    Deterministic guardrails can VETO automation but never AUTHORIZE it: layers
    1-6 may only downgrade toward ESCALATE. docs/02 §6, docs/12 §7.
    """

    SAFETY_VETO = "safety_veto"
    COMPLIANCE_VETO = "compliance_veto"
    NO_PRECEDENT_GATE = "no_precedent_gate"
    CONFIDENCE_GATE = "confidence_gate"
    CONTEXT_GATE = "context_gate"
    POST_GENERATION_VALIDATION = "post_generation_validation"
    DEFAULT = "default"


class Stratum(str, Enum):
    """Golden-set stratum. Tagged at insertion time, never inferred later.

    Only RANDOM is headline-eligible. The other three diagnose; they cannot make
    performance claims (docs/04 §1).
    """

    RANDOM = "random"
    STRATIFIED = "stratified"
    ADVERSARIAL = "adversarial"
    POLICY_TRAP = "policy_trap"


class Exemplar(BaseModel):
    """One retrieved historical (customer message -> brand reply) precedent."""

    evidence_id: str
    conversation_id: str
    customer_text: str
    brand_reply: str
    intent: str
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    rrf_score: float
    # Cosine similarity to the query. The no-precedent gate reads THIS, not
    # rrf_score: RRF is rank-based and tops out at 2/(k+1) ~= 0.033, so it says
    # nothing about whether a close precedent exists at all.
    dense_score: Optional[float] = None
    selected_by_mmr: bool


class CommitmentSlots(BaseModel):
    """Typed commitments extracted from a draft. Empty is the common, correct case.

    Every populated slot must be verified by strict entailment against retrieved
    evidence. Unsupported => hard fail, regardless of how good the reply reads.
    This is measured as its own metric, not as a judge sub-score (docs/04 §6).
    """

    refund_offered: bool = False
    refund_window_days: Optional[int] = None
    compensation_amount: Optional[float] = None
    promised_timeline: Optional[str] = None
    entitlement_claimed: Optional[str] = None
    escalation_promise: Optional[str] = None


class Decision(BaseModel):
    """The agent's output for one inbound message. The output contract.

    Established early precisely so every later component is testable against it
    (docs/02 §11 build step 2).
    """

    intent: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    action: Action
    reply_draft: Optional[str] = Field(default=None, max_length=280)
    escalation_code: Optional[EscalationCode] = None
    escalation_reason: Optional[str] = None
    route_to: Optional[str] = None
    priority: Optional[str] = None
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("intent")
    @classmethod
    def _intent_in_taxonomy(cls, v: str) -> str:
        """Business rule the JSON schema cannot express: closed enum from intents.yaml.

        Read at validation time rather than baked into an Enum at import time: the
        taxonomy is a data file that induction rewrites (docs/16 T2.1), and an Enum
        would silently freeze whatever happened to be on disk when the module was
        first imported.
        """
        from src import intent_names

        allowed = intent_names()
        if v not in allowed:
            raise ValueError(
                f"intent {v!r} is not in the frozen taxonomy; allowed: {sorted(allowed)}"
            )
        return v

    @model_validator(mode="after")
    def _reason_iff_escalate(self):
        """escalation_reason present iff action == ESCALATE.

        A MODEL validator, not a field validator: pydantic skips field validators
        for optional fields that were never supplied, so the field-level version
        silently passed the exact case that matters most -- an escalation emitted
        with no reason at all. The brief asks for escalation "with a stated
        reason", and the omitted-field case is how that requirement actually gets
        violated in practice.

        Both directions are enforced. A reason attached to an AUTO_SEND is the
        quieter bug: it reads fine in a log and corrupts every "why did we
        escalate?" aggregation downstream.
        """
        is_escalation = self.action == Action.ESCALATE
        has_reason = self.escalation_reason is not None and self.escalation_reason.strip() != ""
        if is_escalation and not has_reason:
            raise ValueError("escalation_reason is required when action == escalate")
        if not is_escalation and has_reason:
            raise ValueError(
                f"escalation_reason must be absent when action == {self.action.value}"
            )
        return self


class DecisionRecord(BaseModel):
    """One JSONL line per message. The strongest single artifact in the repo.

    Makes every decision auditable and is what you point at in a live review
    (docs/02 §8). Schema: docs/11 §5.
    """

    message_id: str
    conversation_id: str
    stratum: Optional[Stratum] = None
    input_text_redacted: str
    predicted_intent: str
    intent_confidence: float
    evidence: list[Exemplar]
    playbook_id: Optional[str]
    raw_model_output: str
    validation_errors: list[str]
    repair_attempts: int
    commitments: CommitmentSlots
    grounding_passed: bool
    policy_layer_fired: PolicyLayer
    decision: Decision
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    cache_hit: bool
