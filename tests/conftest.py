"""Shared fixtures. Deliberately thin -- each test constructs the context it needs."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import load_config  # noqa: E402
from src.policy import PolicyContext  # noqa: E402
from src.schemas import Exemplar  # noqa: E402


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture
def clean_context():
    """A context that reaches rung 7 untouched. Every rung test perturbs ONE field.

    Starting from a clean context and changing one thing is what makes a failure
    diagnostic: if the test breaks, the field you changed is the cause.
    """
    return PolicyContext(
        text="my playlist won't load on the app",
        intent="playback_technical",
        intent_confidence=0.95,
        top1_retrieval_score=0.80,
        evidence_ids=["t_1", "t_2"],
        thread_turns=1,
    )


@pytest.fixture
def exemplars():
    return [
        Exemplar(
            evidence_id="t_1", conversation_id="c_1",
            customer_text="app won't play anything",
            brand_reply="Try logging out and back in, then reinstall the app. That fixes it in most cases.",
            intent="service_outage_technical", rrf_score=0.8, selected_by_mmr=True,
        ),
        Exemplar(
            evidence_id="t_2", conversation_id="c_2",
            customer_text="playback keeps stopping",
            brand_reply="We've refunded the 30 day charge and issued a credit of $10 to your account.",
            intent="service_outage_technical", rrf_score=0.6, selected_by_mmr=True,
        ),
    ]
