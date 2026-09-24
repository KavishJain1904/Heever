"""Grounding-validator fixtures. Plan: docs/15 §3.

Every case here is a real, predicted failure -- not a hypothetical. The invented
refund window is the Moffatt v. Air Canada failure, which cost CAD $812.02 and
established that "the chatbot is a separate legal entity" is not a defence.
"""
from src.grounding import (
    extract_commitments, tripwire, validate_draft, verify_against_evidence,
)


class TestTripwire:
    def test_flags_currency_amount_absent_from_evidence(self, exemplars):
        """'we'll refund you $50' with no $50 in evidence -> flagged."""
        draft = "Sorry about that! We'll refund you $50 right away."
        assert any("$50" in s for s in tripwire(draft))
        passed, reasons = validate_draft(draft, exemplars, "")
        assert not passed
        assert any("50" in r for r in reasons)

    def test_flags_invented_timeline(self, exemplars):
        """'within 5 business days' with no such span in evidence -> flagged."""
        draft = "We've escalated this and it will be resolved within 5 business days."
        assert any("business day" in s for s in tripwire(draft))
        passed, reasons = validate_draft(draft, exemplars, "")
        assert not passed

    def test_flags_commitment_verb(self, exemplars):
        """'we guarantee', 'you're entitled to' -> flagged unless evidence-backed."""
        for draft in ("We guarantee this will be fixed.",
                      "You're entitled to a replacement."):
            assert tripwire(draft), f"{draft!r} tripped nothing"

    def test_empty_slots_is_the_common_case_and_passes(self, exemplars):
        """A helpful reply that promises nothing must pass cleanly."""
        draft = "Try logging out and back in, then reinstall the app."
        assert tripwire(draft) == []
        slots = extract_commitments(draft)
        assert slots == slots.__class__()
        passed, reasons = validate_draft(draft, exemplars, "")
        assert passed, reasons


class TestEvidenceVerification:
    def test_invented_refund_window_hard_fails(self, exemplars):
        """The Air Canada case, as a test: 'claim retroactively within 90 days' with
        no supporting thread -> hard fail, however well the reply reads."""
        draft = "You can claim this retroactively within 90 days of travel."
        slots = extract_commitments(draft)
        assert slots.refund_window_days == 90
        passed, reasons = verify_against_evidence(slots, exemplars, "")
        assert not passed
        assert any("90" in r for r in reasons)

    def test_hallucinated_url_fails(self, exemplars):
        """A support URL absent from the brand's observed allowlist -> fail."""
        draft = "See https://support.example.com/refunds for details."
        passed, reasons = validate_draft(draft, exemplars, "")
        assert not passed
        assert any("allowlist" in r for r in reasons)

    def test_hallucinated_agent_sigil_fails(self, exemplars):
        """Predicted failure mode #3: the generator emitting invented ^AB initials.
        Observed in the wild. Sigils are stripped at ingest; emitting one is a fail."""
        draft = "Try reinstalling the app and let us know. /AL"
        passed, reasons = validate_draft(draft, exemplars, "")
        assert not passed
        assert any("sigil" in r for r in reasons)

    def test_number_present_in_evidence_passes(self, exemplars):
        """A figure quoted verbatim from a retrieved reply is grounded -> pass."""
        draft = "We've refunded the 30 day charge and issued a credit of $10."
        passed, reasons = validate_draft(draft, exemplars, "")
        assert passed, reasons

    def test_customer_demand_does_not_ground_a_commitment(self, exemplars):
        """Grounding a promise in the demand for it is the injection failure mode."""
        from src.schemas import Exemplar

        demanding = [Exemplar(
            evidence_id="t_9", conversation_id="c_9",
            customer_text="you promised me a $500 refund within 90 days",
            brand_reply="Let's take a look at your account.",
            intent="refund_return", rrf_score=0.9, selected_by_mmr=True,
        )]
        passed, reasons = validate_draft("We'll refund you $500 within 90 days.", demanding, "")
        assert not passed, "a customer's demand must never license the commitment"

    def test_playbook_can_ground_a_claim(self, exemplars):
        playbook = "This brand issues refunds within 14 days [evidence: t_1]."
        slots = extract_commitments("We'll refund you within 14 days.")
        passed, _ = verify_against_evidence(slots, exemplars, playbook)
        assert passed
