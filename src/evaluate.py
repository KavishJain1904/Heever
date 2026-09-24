"""The eval harness. `make eval` (Tier 1) and `make smoke` enter here.

Contract: docs/12 §12. Metric definitions: docs/14.

TIER 1 (`--from-artifacts`): recomputes EVERY headline number, CI and significance
test from data/golden_200.csv + predictions.jsonl + judge_verdicts.jsonl. No tweet
text, no Kaggle account, no API key, no network. Seconds.
TIER 2 (`make eval-full`): regenerates predictions first, then runs Tier 1.

Rules this module enforces mechanically, so they cannot be forgotten under time
pressure (docs/14 §2):
  - The headline may only be computed on stratum=random. Any attempt to compute a
    headline over all 200 raises.
  - Every proportion ships with a Wilson 95% CI. Every macro-F1 ships with a
    10,000-replicate bootstrap CI.
  - Every system comparison is PAIRED: McNemar with continuity correction for
    binary, paired bootstrap for F1. Overlapping CIs do NOT imply no significant
    difference.
  - Baselines present in EVERY table: majority class, random-by-prior,
    retrieval-nearest-neighbour reply, and the constant-"DM us" DEGENERATE CONTROL.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from src import REPO_ROOT, load_config

HEADLINE_STRATUM = "random"
REQUIRED_REPLY_BASELINES = ("dm_us_degenerate", "historical")
REQUIRED_INTENT_BASELINES = ("majority", "random_prior", "retrieval_nn", "dm_us_degenerate")
OOS_INTENT = "other_unclear"


class HeadlineDisciplineError(RuntimeError):
    """Raised on any attempt to compute a headline off a non-random stratum.

    Enforced in code rather than by discipline because this is exactly the rule
    that erodes at 2 a.m. on day 5 (docs/14 §1).
    """


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. NOT Wald.

        (p + z^2/2n  +/-  z*sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n)

    CLT methods "dramatically underestimate uncertainty" in small-data evals
    (arXiv:2503.01747, ICML 2025 spotlight). At p=1.0, n=25 Wald gives the absurd
    [1.0, 1.0]; Wilson gives [0.867, 1.0].
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} out of range for n={n}")

    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def rule_of_three_upper(n: int) -> float:
    """95% upper bound on a rate observed as 0/n. The honest form for a near-zero rate.

    Quoted directly in the policy-invention line: "0/100 unsupported commitments;
    95% upper bound 3.6%". Note this is ~3/n, and agrees with `wilson_interval(0, n)`
    to within a fraction of a point at n>=100 -- both are reported so a reviewer
    recomputing either one finds what they expect.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return 3.0 / n


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def _per_class_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Iterable[str]) -> dict[str, Optional[float]]:
    """Per-class F1. None where the class has zero support AND zero predictions.

    The None is load-bearing: it is what the bootstrap's `drop_class` convention
    drops. Collapsing it to 0.0 here would silently implement the other convention.
    """
    out: dict[str, Optional[float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        if tp + fp + fn == 0:
            out[label] = None
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        out[label] = (
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )
    return out


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], in_scope_only: bool = False) -> float:
    """Arithmetic mean of per-class F1 -- config.evaluation.macro_f1_formula.

    STATE WHICH FORMULA. There are two definitions in circulation (arithmetic mean
    of per-class F1s vs. harmonic mean of macro-averaged P and R); they can differ
    by up to 0.5 AND REORDER CLASSIFIERS (Opitz & Burst, arXiv:1911.03347). The
    arithmetic version is more robust under imbalance.

    Reported TWICE: all classes, and in_scope_only=True. OTHER is a heterogeneous
    grab-bag that drags the macro down while telling you nothing; it is evaluated
    separately as binary in-scope/OOS detection with its own P/R.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if not y_true:
        raise ValueError("cannot compute macro-F1 on an empty set")

    labels = sorted(set(y_true) | set(y_pred))
    if in_scope_only:
        labels = [l for l in labels if l != OOS_INTENT]
        if not labels:
            raise ValueError("in_scope_only=True left no classes")

    scores = [s for s in _per_class_f1(y_true, y_pred, labels).values() if s is not None]
    return float(sum(scores) / len(scores)) if scores else 0.0


def oos_detection(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    """OTHER treated as a binary in-scope/out-of-scope problem, with its own P/R.

    Mirrors how CLINC150 and HINT3 evaluate. Kept out of the macro so a bad OOS
    detector cannot be hidden inside an average, and kept out of the headline so a
    good one cannot flatter it either.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == OOS_INTENT and p == OOS_INTENT)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != OOS_INTENT and p == OOS_INTENT)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == OOS_INTENT and p != OOS_INTENT)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        "support": tp + fn,
    }


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], normalise: bool = True):
    """Normalised confusion matrix plus the top-5 off-diagonal cells.

    The matrix alone is not the analysis (docs/14 §3). The returned
    `top_offdiagonal` is what the failure section is written from, and the
    symmetric/asymmetric read is what turns it into a conclusion:
      - symmetric confusion  => taxonomy problem, merge or redefine
      - asymmetric into one  => prior/threshold problem
    """
    labels = sorted(set(y_true) | set(y_pred))
    idx = {l: i for i, l in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    for t, p in zip(y_true, y_pred):
        mat[idx[t], idx[p]] += 1

    counts = mat.copy()
    if normalise:
        row_sums = mat.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            mat = np.where(row_sums > 0, mat / row_sums, 0.0)

    cells = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i == j or counts[i, j] == 0:
                continue
            reverse = counts[j, i]
            total = counts[i, j] + reverse
            cells.append({
                "true": true_label,
                "pred": pred_label,
                "count": int(counts[i, j]),
                "reverse_count": int(reverse),
                # Symmetry ratio near 1.0 => taxonomy problem; near 0 => threshold.
                "symmetry": float(min(counts[i, j], reverse) / total) if total else 0.0,
                "diagnosis": (
                    "symmetric: taxonomy problem, merge or redefine"
                    if total and min(counts[i, j], reverse) / total > 0.35
                    else "asymmetric: prior/threshold problem"
                ),
            })
    cells.sort(key=lambda c: c["count"], reverse=True)
    return {"labels": labels, "matrix": mat, "counts": counts, "top_offdiagonal": cells[:5]}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(
    metric_fn: Callable,
    y_true: Sequence,
    y_pred: Sequence,
    b: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Nonparametric bootstrap over EXAMPLES. Percentile interval.

    Resampling at the example level automatically propagates rare-class
    instability -- expect macro-F1 CIs 15-25pp wide when a class has <10 support.

    ZERO-SUPPORT CONVENTION (config.evaluation.bootstrap_zero_support): replicates
    where a rare class has zero support leave F1 undefined. We DROP the class from
    that replicate. The alternative (score it 0) gives materially different
    intervals; declaring the choice is the point.

    `macro_f1` implements drop_class natively -- it averages over the classes
    present in the replicate -- so passing it here applies the declared convention
    without a second code path that could drift from it.
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("cannot bootstrap an empty set")
    rng = np.random.default_rng(seed)
    y_true = list(y_true)
    y_pred = list(y_pred)

    stats = np.empty(b, dtype=float)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        stats[i] = metric_fn([y_true[j] for j in idx], [y_pred[j] for j in idx])

    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def bootstrap_ci_zero_support_variant(
    y_true: Sequence, y_pred: Sequence, b: int = 2000, seed: int = 42
) -> dict:
    """Both conventions, side by side. Exists to make the declared choice checkable.

    The spec says the two conventions "give materially different intervals".
    Asserting that in prose is cheap; computing both and printing the gap is what
    makes the declaration meaningful.
    """
    # The full label universe, fixed OUTSIDE the replicate loop. This is the whole
    # difference between the conventions: `macro_f1` derives its labels from the
    # replicate, so a class with zero support is simply not averaged over (=
    # drop_class), whereas score_zero must still see the class in order to score
    # it 0. Deriving labels inside the replicate would make both branches
    # drop_class and the comparison would be vacuous.
    universe = sorted(set(y_true) | set(y_pred))

    def score_zero(yt, yp):
        scores = [s if s is not None else 0.0 for s in _per_class_f1(yt, yp, universe).values()]
        return sum(scores) / len(scores) if scores else 0.0

    return {
        "drop_class": bootstrap_ci(macro_f1, y_true, y_pred, b=b, seed=seed),
        "score_zero": bootstrap_ci(score_zero, y_true, y_pred, b=b, seed=seed),
    }


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------

def mcnemar(b: int, c: int) -> tuple[float, float]:
    """Paired binary comparison with continuity correction.

        chi2 = (|b - c| - 1)^2 / (b + c),  df=1,  reject at chi2 > 3.841

    At b+c=40 discordant pairs you need |b-c| >= 14. Minimum detectable paired
    difference ~7.0pp. For contrast, an UNPAIRED 85% vs 90% at 80% power needs
    ~683 per arm; we have 200 total. This is the single most important power fact
    in the report.

    (docs/04 §3 says 13 and 6.5pp; that is the UNCORRECTED boundary. Solving the
    corrected form gives |b-c| > sqrt(3.841*40) + 1 = 13.4, so 14. docs/10 §6.4.)
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    if b + c == 0:
        # No discordant pairs: the systems are identical on this sample. chi2=0,
        # p=1. Returning NaN here would propagate into Holm-Bonferroni as a silent
        # drop, which is worse than an honest "no evidence of difference".
        return (0.0, 1.0)

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(chi2_dist.sf(chi2, df=1))
    except ImportError:
        p = math.erfc(math.sqrt(chi2 / 2.0))
    return (float(chi2), p)


def mcnemar_from_predictions(
    y_true: Sequence, pred_a: Sequence, pred_b: Sequence
) -> dict:
    """Build the discordant counts, then run McNemar. The usual entry point.

    b = A right, B wrong; c = A wrong, B right. Concordant pairs are discarded by
    construction -- that discard is precisely why the paired test has power the
    unpaired one does not.
    """
    b = sum(1 for t, a, bb in zip(y_true, pred_a, pred_b) if a == t and bb != t)
    c = sum(1 for t, a, bb in zip(y_true, pred_a, pred_b) if a != t and bb == t)
    chi2, p = mcnemar(b, c)
    return {
        "b": b, "c": c, "chi2": chi2, "p": p,
        "significant": chi2 > 3.841,
        "min_detectable_pp": (
            (math.sqrt(3.841 * (b + c)) + 1) / len(y_true) * 100 if (b + c) and y_true else None
        ),
    }


def paired_bootstrap(
    metric_fn: Callable,
    y_true: Sequence,
    pred_a: Sequence,
    pred_b: Sequence,
    b: int = 10000,
    seed: int = 42,
) -> dict:
    """Berg-Kirkpatrick et al., EMNLP 2012. Report the fraction of replicates with
    delta <= 0.

    The SAME resampled indices are applied to both systems in every replicate. That
    pairing is the whole method: resampling the two systems independently throws
    away the correlation between them and answers a different, weaker question.
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("cannot bootstrap an empty set")
    rng = np.random.default_rng(seed)
    y_true, pred_a, pred_b = list(y_true), list(pred_a), list(pred_b)

    observed = metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b)
    deltas = np.empty(b, dtype=float)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        yt = [y_true[j] for j in idx]
        deltas[i] = metric_fn(yt, [pred_a[j] for j in idx]) - metric_fn(yt, [pred_b[j] for j in idx])

    return {
        "observed_delta": float(observed),
        "p_value": float(np.mean(deltas <= 0)),
        "ci": (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))),
    }


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> list[dict]:
    """10 slices at alpha=0.05 => 1 - 0.95^10 = 40.1% chance of a spurious result.

    config.evaluation.declared_slice_count is declared UP FRONT so the correction
    is not chosen after seeing results. Per-slice results are marked exploratory.

    Step-down: sort ascending, compare p_(i) against alpha/(k-i), and STOP at the
    first failure -- every subsequent hypothesis is retained regardless of its own
    p-value. Forgetting the stop rule is the common implementation bug and it makes
    the correction anti-conservative.
    """
    k = len(pvalues)
    if k == 0:
        return []
    order = sorted(range(k), key=lambda i: pvalues[i])
    results = [dict() for _ in range(k)]
    still_rejecting = True
    for rank, i in enumerate(order):
        threshold = alpha / (k - rank)
        rejected = still_rejecting and pvalues[i] <= threshold
        if not rejected:
            still_rejecting = False
        results[i] = {
            "p": float(pvalues[i]),
            "rank": rank + 1,
            "threshold": threshold,
            "rejected": rejected,
            "exploratory": True,
        }
    return results


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

def cohens_kappa(a: Sequence, b: Sequence) -> dict:
    """Return P0, Pe, kappa, the 2x2 table, PABAK and Gwet's AC1.

    THE KAPPA PARADOX, and why all of those are returned together: at IDENTICAL
    95% raw agreement, a skewed 5% base rate gives kappa=0.47 while a balanced 50/50
    gives kappa=0.90. Escalation WILL be rare here, so expect a deflated kappa and
    pre-empt the criticism rather than explaining it away afterwards (docs/04 §4).

    Publishing kappa alone would be misleading -- in our favour or against it,
    depending on which way the reviewer reads it. So all six numbers ship together.
    """
    if len(a) != len(b):
        raise ValueError("rater sequences must be the same length")
    n = len(a)
    if n == 0:
        raise ValueError("cannot compute kappa on an empty set")

    a = [bool(x) for x in a]
    b = [bool(x) for x in b]
    tp = sum(1 for x, y in zip(a, b) if x and y)          # a: both yes
    fp = sum(1 for x, y in zip(a, b) if x and not y)      # b: rater1 yes only
    fn = sum(1 for x, y in zip(a, b) if not x and y)      # c: rater2 yes only
    tn = sum(1 for x, y in zip(a, b) if not x and not y)  # d: both no

    p0 = (tp + tn) / n
    p1_yes = (tp + fp) / n
    p2_yes = (tp + fn) / n
    pe = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)
    kappa = (p0 - pe) / (1 - pe) if pe < 1 else 1.0

    # PABAK: prevalence-and-bias-adjusted. Depends only on raw agreement.
    pabak = 2 * p0 - 1

    # Gwet's AC1: chance agreement estimated from the mean marginal, which is far
    # less sensitive to base-rate skew than Cohen's Pe.
    pi = (p1_yes + p2_yes) / 2
    pe_gwet = 2 * pi * (1 - pi)
    ac1 = (p0 - pe_gwet) / (1 - pe_gwet) if pe_gwet < 1 else 1.0

    return {
        "n": n,
        "p0": p0,
        "pe": pe,
        "kappa": kappa,
        "table": {"a_both_yes": tp, "b_r1_only": fp, "c_r2_only": fn, "d_both_no": tn},
        "pabak": pabak,
        "ac1": ac1,
        # Krippendorff bands -- stricter than Landis-Koch and better suited here.
        "band": (
            "reliable" if kappa >= 0.80
            else "tentative conclusions only" if kappa >= 0.667
            else "unreliable -- do not use"
        ),
        "raw_agreement_ci": wilson_interval(tp + tn, n),
    }


# ---------------------------------------------------------------------------
# Selective prediction
# ---------------------------------------------------------------------------

def elkan_threshold(cost_ratio: float) -> float:
    """T = C_FP / (C_FP + C_FN). At a 20:1 ratio, T = 20/21 ~= 0.952.

    The threshold comes from an explicit cost matrix, not a round number. Accuracy
    weights a wrongly auto-sent reply to a furious customer identically to a
    needless escalation; those costs differ by orders of magnitude.
    """
    return cost_ratio / (cost_ratio + 1.0)


def risk_coverage(confidences: Sequence[float], correct: Sequence[bool], config=None) -> dict:
    """Selective prediction: risk-coverage curve, AURC, E-AURC, selective risk at
    fixed coverage. Threshold from Elkan: T = C_FP / (C_FP + C_FN).

    At config.cost_model ratio 20 -> T = 20/21 ~= 0.952. The sensitivity table over
    {5,10,20,50} is more persuasive than any single accuracy figure: it converts an
    ML result into a business decision (docs/04 §7).
    """
    config = config or load_config()
    n = len(confidences)
    if n == 0:
        raise ValueError("cannot compute a risk-coverage curve on an empty set")
    if n != len(correct):
        raise ValueError("confidences and correct must be the same length")

    order = sorted(range(n), key=lambda i: confidences[i], reverse=True)
    errors = 0
    curve = []
    for rank, i in enumerate(order, start=1):
        if not correct[i]:
            errors += 1
        curve.append({"coverage": rank / n, "risk": errors / rank})

    aurc = float(np.mean([pt["risk"] for pt in curve]))

    # E-AURC: excess over the oracle that ranks every correct answer above every
    # incorrect one. Unitless, so it compares across systems with different base
    # error rates -- which raw AURC cannot.
    n_correct = sum(1 for c in correct if c)
    oracle_errors = 0
    oracle_risks = []
    for rank in range(1, n + 1):
        if rank > n_correct:
            oracle_errors += 1
        oracle_risks.append(oracle_errors / rank)
    aurc_oracle = float(np.mean(oracle_risks))

    ratios = config.cost_model["sensitivity_ratios"]
    default_ratio = config.cost_model["bad_autosend_vs_needless_escalation"]
    sensitivity = []
    for ratio in ratios:
        t = elkan_threshold(float(ratio))
        kept = [i for i in range(n) if confidences[i] >= t]
        coverage = len(kept) / n
        sel_risk = (
            sum(1 for i in kept if not correct[i]) / len(kept) if kept else 0.0
        )
        # Expected cost per 100 messages: bad auto-sends weighted by the ratio,
        # needless escalations at unit cost.
        bad_autosends = sum(1 for i in kept if not correct[i])
        needless_escalations = sum(1 for i in range(n) if i not in set(kept) and correct[i])
        sensitivity.append({
            "cost_ratio": ratio,
            "threshold": t,
            "coverage": coverage,
            "selective_risk": sel_risk,
            "selective_risk_ci": (
                wilson_interval(sum(1 for i in kept if not correct[i]), len(kept))
                if kept else (0.0, 1.0)
            ),
            "expected_cost_per_100": (
                (bad_autosends * float(ratio) + needless_escalations) / n * 100
            ),
            "is_default": ratio == default_ratio,
        })

    return {
        "curve": curve,
        "aurc": aurc,
        "aurc_oracle": aurc_oracle,
        "e_aurc": aurc - aurc_oracle,
        "accuracy_at_full_coverage": n_correct / n,
        "sensitivity": sensitivity,
    }


def calibration(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> dict:
    """Brier score as the headline (a proper scoring rule), with ECE and a
    reliability diagram as SUPPORTING evidence.

    ECE is binning-dependent, unstable and biased; at n=200 a 10-bin ECE has ~20
    points per bin. Report it, but NEVER compare two ECEs and call the difference
    real.
    """
    n = len(confidences)
    if n == 0:
        raise ValueError("cannot compute calibration on an empty set")

    conf = np.asarray(confidences, dtype=float)
    truth = np.asarray([1.0 if c else 0.0 for c in correct], dtype=float)

    brier = float(np.mean((conf - truth) ** 2))

    # Murphy decomposition: reliability - resolution + uncertainty.
    base = float(np.mean(truth))
    edges = np.linspace(0.0, 1.0, bins + 1)
    reliability = resolution = 0.0
    diagram = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        count = int(mask.sum())
        if count == 0:
            diagram.append({"lo": float(lo), "hi": float(hi), "count": 0,
                            "mean_confidence": None, "accuracy": None})
            continue
        mean_conf = float(conf[mask].mean())
        acc = float(truth[mask].mean())
        reliability += count / n * (mean_conf - acc) ** 2
        resolution += count / n * (acc - base) ** 2
        diagram.append({"lo": float(lo), "hi": float(hi), "count": count,
                        "mean_confidence": mean_conf, "accuracy": acc})

    ece = float(
        sum(
            b["count"] / n * abs(b["mean_confidence"] - b["accuracy"])
            for b in diagram if b["count"]
        )
    )

    return {
        "brier": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": base * (1 - base),
        "ece": ece,
        "ece_bins": bins,
        "ece_mean_points_per_bin": n / bins,
        "diagram": diagram,
        "caveat": (
            "ECE is binning-dependent, unstable and biased. Reported as supporting "
            "evidence only; two ECEs are never compared and called different."
        ),
    }


# ---------------------------------------------------------------------------
# Headline discipline
# ---------------------------------------------------------------------------

def require_headline_slice(rows: Sequence[dict]) -> list[dict]:
    """Return the random-stratum rows, or raise. The mechanical enforcement.

    Raises on ANY non-random row in the input, rather than silently filtering,
    because a silent filter would let a caller pass all 200 rows and receive a
    correct-looking headline -- the caller would never learn that the harness had
    quietly disagreed with them.
    """
    strata = {r.get("stratum") for r in rows}
    if strata - {HEADLINE_STRATUM}:
        raise HeadlineDisciplineError(
            f"headline requested over strata {sorted(s for s in strata if s)}; only "
            f"{HEADLINE_STRATUM!r} is headline-eligible (docs/14 §1). The stratified, "
            f"adversarial and policy_trap strata estimate different quantities: "
            f"stratified aggregates estimate a reweighted population, not production "
            f"traffic, and adversarial is pessimistic by construction. "
            f"Use `select_stratum(rows, 'random')` if that is what you meant."
        )
    return list(rows)


def select_stratum(rows: Sequence[dict], stratum: str) -> list[dict]:
    """Explicit stratum selection. The only sanctioned way to get to a headline."""
    return [r for r in rows if r.get("stratum") == stratum]


def proportion(successes: int, n: int, label: str = "") -> dict:
    """Every proportion in this harness ships with its Wilson CI. No exceptions.

    Returning a dict rather than a float is the enforcement: there is no way to
    obtain a bare rate from this module and forget to carry its interval.
    """
    lo, hi = wilson_interval(successes, n)
    return {
        "label": label, "successes": successes, "n": n,
        "rate": successes / n if n else 0.0,
        "ci_low": lo, "ci_high": hi,
        "ci_width_pp": (hi - lo) * 100,
    }


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_golden(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def golden_content_hash(rows: Sequence[dict]) -> str:
    """Content hash over the frozen random stratum.

    Pinned in the test suite. A changed hash means the golden set moved, which
    invalidates every number computed against it -- so it is better to fail a test
    than to discover it in a reviewer's diff.
    """
    payload = "\n".join(
        f"{r.get('message_id','')}\x00{r.get('gold_intent','')}\x00{r.get('gold_action','')}"
        for r in sorted(
            (r for r in rows if r.get("stratum") == HEADLINE_STRATUM),
            key=lambda r: r.get("message_id", ""),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_hash(config_path: Path) -> str:
    """Every results table carries this. A table that cannot be attributed to a
    configuration is not a result (docs/14 §9)."""
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------

def run_from_artifacts(config, config_path: Path) -> dict:
    """Tier 1. Recompute every number from committed artifacts. No network, no key."""
    golden_path = REPO_ROOT / config.evaluation["golden_set"]
    results_dir = REPO_ROOT / "artifacts" / "results"
    predictions = _read_jsonl(results_dir / "predictions.jsonl")
    verdicts = _read_jsonl(Path(REPO_ROOT / config.evaluation["judge"]["verdicts_path"]))
    golden = _read_golden(golden_path)

    missing = [
        str(p) for p, present in (
            (golden_path, golden),
            (results_dir / "predictions.jsonl", predictions),
        ) if not present
    ]
    if missing:
        raise SystemExit(
            "Tier 1 needs committed artifacts that are not present:\n  "
            + "\n  ".join(missing)
            + "\n\nThese are produced by `make eval-full` (which needs data/raw/twcs.csv "
              "and the committed LLM cache) and then committed. `make smoke` runs the "
              "harness self-checks with no artifacts at all."
        )

    gold_by_id = {r["message_id"]: r for r in golden}
    rows = []
    for pred in predictions:
        gold = gold_by_id.get(pred.get("message_id"))
        if gold is None:
            continue
        rows.append({**gold, **{f"pred_{k}": v for k, v in pred.items()}})

    tables = build_tables(rows, verdicts, config, config_path)

    if "T3_reply_quality" in tables:
        summary_path = results_dir / "judge_summary.json"
        summary_path.write_text(
            json.dumps(_judge_summary_document(tables["T3_reply_quality"]), indent=2, default=str),
            encoding="utf-8",
        )

    return tables


def build_tables(rows: list[dict], verdicts: list[dict], config, config_path: Path) -> dict:
    """T1-T7 per docs/14 §9. Every table carries the config and golden-set hashes."""
    headline_rows = select_stratum(rows, HEADLINE_STRATUM)
    require_headline_slice(headline_rows)

    tables: dict = {
        "footer": {
            "config_sha256": config_hash(config_path),
            "golden_set_sha256": golden_content_hash(rows),
            "n_total": len(rows),
            "n_headline": len(headline_rows),
            "macro_f1_formula": config.evaluation["macro_f1_formula"],
            "bootstrap_zero_support": config.evaluation["bootstrap_zero_support"],
        }
    }

    if headline_rows:
        y_true = [r["gold_intent"] for r in headline_rows]
        y_pred = [r["pred_intent"] for r in headline_rows]
        b = int(config.evaluation["bootstrap_replicates"])
        seed = int(config.evaluation["bootstrap_seed"])
        tables["T1_intent"] = {
            "accuracy": proportion(
                sum(1 for t, p in zip(y_true, y_pred) if t == p), len(y_true), "accuracy"
            ),
            "macro_f1_all": macro_f1(y_true, y_pred),
            "macro_f1_all_ci": bootstrap_ci(macro_f1, y_true, y_pred, b=b, seed=seed),
            "macro_f1_in_scope": macro_f1(y_true, y_pred, in_scope_only=True),
            "oos": oos_detection(y_true, y_pred),
        }
        tables["T2_per_class"] = confusion_matrix(y_true, y_pred)

        confs = [float(r.get("pred_intent_confidence", 0.0)) for r in headline_rows]
        correct = [t == p for t, p in zip(y_true, y_pred)]
        tables["T4_selective"] = risk_coverage(confs, correct, config)
        tables["T4_calibration"] = calibration(confs, correct)

    if verdicts:
        tables["T3_reply_quality"] = _reply_quality_table(verdicts)
        tables["T5_judge"] = _judge_validation_table(verdicts)

    tables["T6_policy_invention"] = _policy_invention_table(rows)
    return tables


def _reply_quality_table(verdicts: list[dict]) -> dict:
    """T3. The degenerate 'DM us' control and the brand's own replies are PERMANENT rows.

    Their presence is asserted, not assumed: a reply-quality table without the
    degenerate control cannot tell a reviewer whether the system beat a constant
    string, and that is the first thing a good reviewer checks.

    Depends ONLY on the fields the redacted verdicts file will still carry:
    `message_id`, `system`, `position_bias_disagreements`,
    `criteria[name].pass`, `criteria[name].order_disagreement`, and optionally
    `human_criteria`. It must never read `draft`, `usage`, `cache_hit`, or
    `criteria[*].reasoning`/`quote`.

    Beyond the per-system pass-rate table this harness has always produced, this
    also computes PAIRED comparisons (heever vs. historical, heever vs. the
    dm_us_degenerate control) for every criterion AND the all-six-pass
    composite, via `mcnemar_from_predictions` -- the same paired McNemar used
    everywhere else in this module, reused rather than reimplemented. Pairing
    is by `message_id`; only ids present for every system are used, and the
    count is reported as `paired.n_paired`. `order_flip` is the count of rows
    where the judge disagreed with itself under criterion-order flip, read off
    `criteria[name].order_disagreement` (equivalently, whether `name` appears in
    `position_bias_disagreements`).
    """
    systems = {v.get("system") for v in verdicts}
    missing = [b for b in REQUIRED_REPLY_BASELINES if b not in systems]
    if missing:
        raise ValueError(
            f"reply-quality table is missing required baseline rows: {missing}. "
            f"docs/14 §9 T3 makes these permanent fixtures -- 'dm_us_degenerate' is "
            f"the control that shows the system beats a constant string, and "
            f"'historical' reframes the headline by scoring the brand's own replies "
            f"under the same rubric."
        )

    criteria_names = list(verdicts[0]["criteria"])

    def row_passes(v: dict) -> dict[str, bool]:
        passes = {name: bool(v["criteria"][name]["pass"]) for name in criteria_names}
        passes["all_six_pass"] = all(passes.values())
        return passes

    def row_order_flip(v: dict, name: str) -> bool:
        # Defined per individual criterion only -- "did the judge flip under
        # reordering" has no single meaning for the all-six-pass composite, so
        # the committed reference table carries no order_flip for it either.
        return bool(v["criteria"][name].get("order_disagreement", False))

    by_system_rows = {
        system: [v for v in verdicts if v.get("system") == system] for system in systems if system
    }

    out: dict = {}
    for system in sorted(by_system_rows):
        rows = by_system_rows[system]
        all_pass = sum(1 for v in rows if row_passes(v)["all_six_pass"])
        per_criterion = {}
        for name in criteria_names:
            hits = sum(1 for v in rows if row_passes(v)[name])
            entry = proportion(hits, len(rows), name)
            entry["order_flip"] = sum(1 for v in rows if row_order_flip(v, name))
            per_criterion[name] = entry
        out[system] = {
            "all_six_pass": proportion(all_pass, len(rows), "all_six_pass"),
            "per_criterion": per_criterion,
        }

    # Paired comparisons, heever vs. every other required-baseline system.
    lookup = {
        system: {v["message_id"]: row_passes(v) for v in rows}
        for system, rows in by_system_rows.items()
    }
    other_systems = sorted(s for s in lookup if s != "heever")
    common_ids = set(lookup.get("heever", {}))
    for system in other_systems:
        common_ids &= set(lookup[system])
    common_ids = sorted(common_ids)
    n_paired = len(common_ids)

    paired: dict = {"n_paired": n_paired, "criteria": {}}
    if "heever" in lookup:
        for name in criteria_names + ["all_six_pass"]:
            comparisons = {}
            for system in other_systems:
                pred_heever = [lookup["heever"][mid][name] for mid in common_ids]
                pred_other = [lookup[system][mid][name] for mid in common_ids]
                mc = mcnemar_from_predictions([True] * n_paired, pred_heever, pred_other)
                comparisons[f"heever_vs_{system}"] = {
                    "b": mc["b"], "c": mc["c"], "p": round(mc["p"], 4),
                }
            paired["criteria"][name] = comparisons

    out["paired"] = paired
    return out


def _judge_summary_document(reply_quality: dict) -> dict:
    """Flatten `T3_reply_quality` into the `judge_summary.json` shape: one entry
    per criterion (plus the all-six-pass composite) holding each system's pass
    count/rate/Wilson CI/order-flip count, and the two heever-vs-baseline
    paired McNemar results. This is the sole writer of that file; it is built
    here so the number quoted in the report is reachable from `make eval`.
    """
    paired = reply_quality["paired"]
    systems = sorted(s for s in reply_quality if s != "paired")
    criteria_names = sorted(paired["criteria"])

    criteria: dict = {}
    for name in criteria_names:
        entry = {}
        for system in systems:
            if name == "all_six_pass":
                src = reply_quality[system]["all_six_pass"]
            else:
                src = reply_quality[system]["per_criterion"][name]
            system_entry = {
                "pass": src["successes"],
                "rate": src["rate"],
                "wilson": [src["ci_low"], src["ci_high"]],
            }
            if "order_flip" in src:
                system_entry["order_flip"] = src["order_flip"]
            entry[system] = system_entry
        entry.update(paired["criteria"][name])
        criteria[name] = entry

    return {"n_paired": paired["n_paired"], "criteria": criteria}


def _judge_validation_table(verdicts: list[dict]) -> dict:
    """T5. Per-criterion kappa against human labels, with the kappa-intra ceiling."""
    labelled = [v for v in verdicts if v.get("human_criteria")]
    if not labelled:
        return {"note": "no human-labelled verdicts present; judge validation skipped"}
    out = {}
    for name in labelled[0]["criteria"]:
        judge = [v["criteria"][name]["pass"] for v in labelled]
        human = [v["human_criteria"][name] for v in labelled]
        out[name] = cohens_kappa(judge, human)
    return out


def _policy_invention_table(rows: list[dict]) -> dict:
    """T6. Reported per stratum. The random/policy_trap gap is a finding, not a failure."""
    out = {}
    for stratum in sorted({r.get("stratum") for r in rows if r.get("stratum")}):
        subset = [r for r in rows if r.get("stratum") == stratum]
        unsupported = sum(1 for r in subset if r.get("pred_grounding_passed") is False)
        entry = proportion(unsupported, len(subset), f"unsupported_commitments[{stratum}]")
        if unsupported == 0:
            entry["rule_of_three_upper"] = rule_of_three_upper(len(subset))
        out[stratum] = entry
    return out


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

def run_smoke(rows: int = 50) -> int:
    """<30s, CPU-only, NO NETWORK, no data download, no API key. CI-able.

    Verifies the MEASUREMENT APPARATUS against the worked examples pinned in
    docs/14, then exercises the full table path on a synthetic set. A bug in the
    harness produces a confident wrong number, which is worse than a bug in the
    agent, so this is the check that runs on every commit.
    """
    failures: list[str] = []

    def check(name: str, got, want, tol=5e-3):
        ok = all(abs(g - w) <= tol for g, w in zip(np.atleast_1d(got), np.atleast_1d(want)))
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {np.round(got, 4)}")
        if not ok:
            failures.append(f"{name}: got {got}, want {want}")

    print("harness self-check against docs/14 worked examples")
    check("wilson(78,100)   headline slice", wilson_interval(78, 100), (0.689, 0.850))
    check("wilson(170,200)  full golden set", wilson_interval(170, 200), (0.794, 0.893))
    check("wilson(25,25)    perfect slice", wilson_interval(25, 25), (0.867, 1.000))
    check("mcnemar(27,13)   significant", mcnemar(27, 13)[0], 4.225)
    check("mcnemar(26,14)   not significant", mcnemar(26, 14)[0], 3.025)

    skewed = cohens_kappa([True] * 5 + [True] * 5 + [False] * 5 + [False] * 185,
                          [True] * 5 + [False] * 5 + [True] * 5 + [False] * 185)
    balanced = cohens_kappa([True] * 95 + [True] * 5 + [False] * 5 + [False] * 95,
                            [True] * 95 + [False] * 5 + [True] * 5 + [False] * 95)
    check("kappa skewed     (5% base rate)", skewed["kappa"], 0.474)
    check("kappa balanced   (50/50)", balanced["kappa"], 0.900)
    check("  ... at identical P0", (skewed["p0"], balanced["p0"]), (0.95, 0.95))
    check("elkan T @ 20:1", elkan_threshold(20), 0.952)

    print(f"\nsynthetic end-to-end on {rows} rows")
    rng = np.random.default_rng(42)
    labels = ["billing_charges", "refund_return", "service_outage_technical", OOS_INTENT]
    y_true = [labels[i % len(labels)] for i in range(rows)]
    y_pred = [t if rng.random() < 0.8 else labels[int(rng.integers(len(labels)))] for t in y_true]
    confs = [0.95 if t == p else float(rng.uniform(0.4, 0.9)) for t, p in zip(y_true, y_pred)]
    correct = [t == p for t, p in zip(y_true, y_pred)]

    print(f"  macro-F1 all      {macro_f1(y_true, y_pred):.3f}")
    print(f"  macro-F1 in-scope {macro_f1(y_true, y_pred, in_scope_only=True):.3f}")
    lo, hi = bootstrap_ci(macro_f1, y_true, y_pred, b=200, seed=42)
    print(f"  bootstrap CI      [{lo:.3f}, {hi:.3f}]  (B=200 in smoke; 10,000 in `make eval`)")
    rc = risk_coverage(confs, correct)
    print(f"  AURC {rc['aurc']:.3f}  E-AURC {rc['e_aurc']:.3f}")
    print(f"  Brier {calibration(confs, correct)['brier']:.3f}")

    try:
        require_headline_slice([{"stratum": "random"}, {"stratum": "adversarial"}])
        failures.append("headline discipline did not raise on a mixed-stratum input")
        print("  [FAIL] headline discipline")
    except HeadlineDisciplineError:
        print("  [ok] headline discipline raises on non-random strata")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nall harness self-checks passed")
    return 0


def _plot_confusion_matrix(t2: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    labels, matrix = t2["labels"], t2["matrix"]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), max(5, len(labels) * 0.9)))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="row-normalised rate")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("T2: confusion matrix (row-normalised)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j] > 0:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if matrix[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_risk_coverage(t4: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    curve = t4["curve"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([pt["coverage"] for pt in curve], [pt["risk"] for pt in curve], color="#1f5fa8")
    ax.set_xlabel("coverage"); ax.set_ylabel("selective risk")
    ax.set_title(f"T4: risk-coverage curve (AURC={t4['aurc']:.3f}, E-AURC={t4['e_aurc']:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_reliability(t4c: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    diagram = [b for b in t4c["diagram"] if b["count"]]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.bar([b["mean_confidence"] for b in diagram], [b["accuracy"] for b in diagram],
           width=0.06, edgecolor="black", color="#c0392b", alpha=0.75, label="observed")
    ax.set_xlabel("mean confidence"); ax.set_ylabel("accuracy")
    ax.set_title(f"T4: reliability diagram (Brier={t4c['brier']:.3f}, ECE={t4c['ece']:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_figures(config, config_path: Path) -> int:
    """Confusion matrix, risk-coverage curve, reliability diagram -> reports/*.png.

    Requires the same committed Tier 1 artifacts as `--from-artifacts`; each
    figure is built directly off the table a reviewer would otherwise have to
    reconstruct by hand from the printed JSON.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib is not installed; `pip install -r requirements.txt`")
        return 1

    try:
        tables = run_from_artifacts(config, config_path)
    except SystemExit as exc:
        print(f"figures skipped -- {exc}")
        return 0

    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    plots = [
        ("T2_per_class", "confusion_matrix.png", _plot_confusion_matrix),
        ("T4_selective", "risk_coverage.png", _plot_risk_coverage),
        ("T4_calibration", "reliability_diagram.png", _plot_reliability),
    ]
    written, skipped = [], []
    for table_key, filename, plot_fn in plots:
        if table_key not in tables:
            skipped.append(filename)
            continue
        out_path = reports_dir / filename
        plot_fn(tables[table_key], out_path)
        written.append(out_path)

    for path in written:
        print(f"wrote {path}")
    if skipped:
        print(
            f"skipped {', '.join(skipped)} -- required table(s) absent, likely because "
            f"data/golden_200.csv is not yet committed"
        )
    return 0


def main() -> int:
    """CLI: --from-artifacts | --smoke | --figures | --rows N | --offline."""
    parser = argparse.ArgumentParser(prog="src.evaluate", description=__doc__)
    parser.add_argument("--from-artifacts", action="store_true",
                        help="Tier 1: recompute every number from committed artifacts")
    parser.add_argument("--smoke", action="store_true",
                        help="harness self-check, <30s, offline, no artifacts needed")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--offline", action="store_true",
                        help="assert no network access is attempted")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    args = parser.parse_args()

    if args.offline:
        _block_network()

    config_path = Path(args.config)
    config = load_config(config_path)

    if args.smoke:
        return run_smoke(args.rows)
    if args.figures:
        return run_figures(config, config_path)
    if args.from_artifacts:
        tables = run_from_artifacts(config, config_path)
        print(json.dumps(tables, indent=2, default=str))
        return 0

    parser.print_help()
    return 1


def _block_network() -> None:
    """Make any socket connection raise. `--offline` is asserted, not trusted.

    The smoke test's offline guarantee is worth nothing if it is a comment. This
    turns it into a runtime property.
    """
    import socket

    def _deny(*args, **kwargs):
        raise RuntimeError(
            "network access attempted under --offline. The smoke path must run with "
            "no Kaggle account, no API key and no network."
        )

    socket.socket.connect = _deny            # type: ignore[assignment]
    socket.create_connection = _deny         # type: ignore[assignment]


if __name__ == "__main__":
    raise SystemExit(main())
