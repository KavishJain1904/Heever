"""Tier 3. SetFit (arXiv:2209.11055). Experiment path -- not required to reproduce.

The right tool at 200 labels: contrastive fine-tuning of a sentence transformer on
generated pairs, then a logistic head. No prompts, no verbalizers.

HONEST NOTE ON HARDWARE: SetFit trains in minutes on a single consumer GPU or even
CPU. The DGX does not help; SetFit helps. We use it because it is the right tool,
not because the hardware was available (docs/02 §9).

Published anchors: 77.9% on Banking77 at 8 shots/class; on RAFT, SetFit-RoBERTa
(355M) scores 71.3 vs T-Few (11B) 75.8, GPT-3 (175B) 62.7, human baseline 73.5.
"""
from __future__ import annotations

from typing import Sequence


def train(texts: Sequence[str], labels: Sequence[str], base_model: str, seed: int = 42):
    """Contrastive fine-tuning of a sentence transformer, then a logistic head.

    Trains in minutes on CPU. Stated plainly in the report: the DGX does not help
    here, SetFit helps. Reaching for the biggest available hardware when the right
    tool runs on a laptop is a judgement error, not a resourcing win.
    """
    try:
        from setfit import SetFitModel, TrainingArguments, Trainer
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(
            "SetFit is on the experiment path only; install via "
            "requirements-dgx.txt. It is not needed to reproduce the headline."
        ) from exc

    model = SetFitModel.from_pretrained(base_model)
    dataset = Dataset.from_dict({"text": list(texts), "label": list(labels)})
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(batch_size=16, num_epochs=1, seed=seed),
    )
    trainer.train()
    return model


def predict(model, texts: Sequence[str]) -> list[str]:
    return [str(p) for p in model.predict(list(texts))]
