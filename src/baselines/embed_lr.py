"""Tier 2. Frozen MiniLM embeddings + logistic regression, and nearest-centroid.

Reuses the cached embeddings, so this costs one extra .fit(). It is the honest
"what do embeddings buy over bag-of-words" experiment.

LR on 384-d with 200 labels is well-conditioned. Nearest-centroid works with 3
examples per class and needs no training at all.

Imbalance handling at n~200 (docs/03 §5):
  - stratified 5-fold, report MEAN +/- STD, never a single 80/20 split -- the std
    is the honest signal at this n, and an unstratified split can leave a class
    absent from test entirely;
  - CLASS WEIGHTS over resampling (one low-resource ablation: removing class
    weights cost 8.2 macro-F1 points);
  - per-class threshold tuning rather than argmax, TUNED INSIDE THE CV LOOP, NEVER
    ON TEST.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def fit_lr(embeddings, labels: Sequence[str], class_weight: str = "balanced", seed: int = 42):
    """Logistic regression on frozen MiniLM vectors.

    class_weight='balanced' is not a default to leave alone -- removing it cost 8.2
    macro-F1 points in one low-resource ablation, which at this n is the difference
    between "embeddings beat TF-IDF" and the opposite conclusion.
    """
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        max_iter=2000, class_weight=class_weight, random_state=seed, multi_class="auto"
    )
    model.fit(np.asarray(embeddings), list(labels))
    return model


def fit_nearest_centroid(embeddings, labels: Sequence[str]):
    """Class centroids in embedding space. No training, works at 3 examples/class.

    Included because it is the cheapest possible use of the embeddings, so if LR
    does not beat it, the LR is not buying anything.
    """
    from sklearn.neighbors import NearestCentroid

    model = NearestCentroid()
    model.fit(np.asarray(embeddings), list(labels))
    return model


def cross_validate(embeddings, labels: Sequence[str], cv: int = 5, seed: int = 42) -> dict:
    """Stratified 5-fold, MEAN +/- STD. Never a single 80/20 split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    scores = cross_val_score(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        np.asarray(embeddings), list(labels),
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed),
        scoring="f1_macro",
    )
    return {
        "macro_f1_mean": float(scores.mean()),
        "macro_f1_std": float(scores.std()),
        "folds": [float(s) for s in scores],
    }


def tune_per_class_thresholds(probs, y_true: Sequence[str], labels: Sequence[str]) -> dict:
    """Per-class decision thresholds instead of argmax.

    TUNED INSIDE THE CV LOOP, NEVER ON TEST. Tuning a threshold on the test set is
    the most common way a low-resource result becomes unreproducible, and it is
    invisible in the reported number -- which is why this function takes
    validation-fold probabilities and the caller is responsible for never handing
    it test data.
    """
    probs = np.asarray(probs)
    thresholds = {}
    for j, label in enumerate(labels):
        binary = np.array([1 if y == label else 0 for y in y_true])
        best_f1, best_t = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            pred = (probs[:, j] >= t).astype(int)
            tp = int(((pred == 1) & (binary == 1)).sum())
            fp = int(((pred == 1) & (binary == 0)).sum())
            fn = int(((pred == 0) & (binary == 1)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[label] = best_t
    return thresholds
