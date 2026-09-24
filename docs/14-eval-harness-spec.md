# Evaluation harness spec

Metric definitions bound to code paths. `docs/04` argues *why*; this document says
*exactly what is computed*, with the conventions declared up front so they cannot be
chosen after seeing the results.

**The harness is the deliverable.** A bug here is worse than a bug in the agent: it
produces a confident wrong number.

---

## 1. The golden set and what each stratum may claim

| Stratum | n | Estimates | Headline-eligible |
|---|---:|---|---|
| `random` | 100 | Production performance, unbiased | **yes — and only this** |
| `stratified` | 60 | Per-intent behaviour; floors rare classes at ~8–10 | no |
| `adversarial` | 25 | Behaviour under profanity, sarcasm, multi-intent, non-English, already-escalated | no — **pessimistic by construction** |
| `policy_trap` | 15 | Policy-invention rate under bait | no |

The four designs estimate **different quantities**. Stratified aggregates estimate a
*reweighted* population, not your traffic. Adversarial is maximally informative per label
and cannot be quoted as performance.

`src.evaluate` **raises** on any attempt to compute a headline over anything but
`stratum == random`. Enforced in code, not by discipline, because this is exactly the
rule that erodes at 2 a.m. on day 5.

---

## 2. Universal rules

1. Every proportion ships a **Wilson 95% CI**. Never Wald.
2. Every macro-F1 ships a **10,000-replicate bootstrap CI**, seeded.
3. Every system comparison is **paired** — McNemar (binary) or paired bootstrap (F1).
   **Overlapping CIs do not imply no significant difference.**
4. Four baselines appear in every table: `majority`, `random_prior`, `retrieval_nn`,
   and `dm_us_degenerate`.
5. Every metric is reported **twice**: all 200, and the
   `confidence == high AND ambiguous == false` subset.

---

## 3. Classification metrics

**Headline: macro-F1.** Accuracy, weighted-F1 and a full per-class P/R/F1/support table
accompany it. Micro-F1 equals accuracy in single-label multi-class and is dominated by
the largest class; weighted-F1 hides minority failure by construction.

**Declare the formula.** There are two "macro-F1" definitions in circulation — the
arithmetic mean of per-class F1s, and the harmonic mean of macro-averaged P and R. They
can differ by up to 0.5 **and reorder classifiers**. We use the **arithmetic mean of
per-class F1** (`config.evaluation.macro_f1_formula`), which is more robust under
imbalance. This is stated in the README, not buried here.

**Report macro-F1 twice — all classes, and in-scope only.** `other_unclear` is a
heterogeneous grab-bag: its F1 will be low, it drags the macro down, and it tells you
nothing about the in-scope classes. It is evaluated **separately** as a binary
in-scope/OOS detection problem with its own precision and recall — mirroring how
CLINC150 and HINT3 evaluate.

**Error analysis.** Normalised confusion matrix. For each of the top 5 off-diagonal
cells: three verbatim tweets.
- **Symmetric confusion ⇒ taxonomy problem.** Merge or redefine the classes.
- **Asymmetric confusion into one class ⇒ prior/threshold problem.**
That distinction is the analysis; the matrix alone is not.

---

## 4. Uncertainty

### Wilson score interval

```
        p̂ + z²/2n  ±  z·√( p̂(1−p̂)/n + z²/4n² )
CI  =  ─────────────────────────────────────────
                    1 + z²/n
```

Pinned worked examples (`tests/test_eval.py::TestWilson`):

| Case | p̂ | n | Wilson 95% CI | Width |
|---|---:|---:|---|---:|
| **Headline slice** | 0.78 | **100** | **[0.689, 0.850]** | **±8pp** |
| Full golden set | 0.85 | 200 | [0.794, 0.893] | ±5pp |
| Per-intent slice | 0.84 | 25 | [0.654, 0.936] | ±14pp |
| Perfect slice | 1.00 | 25 | [0.867, 1.000] | — |

> **Correction to `docs/04 §8` item 8.** That item says *"±~5pp on the headline at
> n=200"*. The ±5pp figure is correct for n=200 and **wrong for the headline**, which
> lives on the frozen n=100 random stratum and carries **±8pp**. The honesty checklist
> must quote ±8pp. Understating your own uncertainty is the one error that section
> cannot afford, and a grader who recomputes the interval will check exactly this.

Read the headline row twice: **78% is statistically indistinguishable from 69% and from
85%.** Any claimed improvement smaller than ~16pp is inside the noise of a single
measurement on that slice.

At n=200 with zero errors Wilson gives [0.981, 1.0], consistent with the **rule of
three**: with 0 failures in n trials the 95% upper bound on the failure rate is ≈ 3/n.
That is the form used for the policy-invention rate.

**Per-intent numbers at n=200 are nearly meaningless for rare intents. Say so rather
than tabling them silently** — a ±14pp interval on a per-class cell carries essentially
no information, and presenting it without the interval implies otherwise.

### Bootstrap CIs for macro-F1

No clean closed form (a ratio of correlated counts). Nonparametric bootstrap over
**examples**, B = 10,000, percentile interval, `seed=42`. Resampling at the example level
automatically propagates rare-class instability — expect macro-F1 CIs **15–25pp wide**
when a class has <10 support.

**Declared convention** (`config.evaluation.bootstrap_zero_support: drop_class`):
replicates where a rare class has zero support leave F1 undefined. We **drop the class
from that replicate**. The alternative — scoring it 0 — gives materially different
intervals. Declaring the choice up front is the point; choosing it after seeing both is
the failure.

---

## 5. Significance

### McNemar, paired binary

```
χ² = (|b − c| − 1)² / (b + c),   df = 1,   reject at χ² > 3.841
```

At b+c = 40 discordant pairs you need **|b − c| ≥ 14**: b=27,c=13 → χ²=4.23
(significant); b=26,c=14 → χ²=3.02 (not). **Minimum detectable paired difference = 7.0pp**
(14/200).

> **Correction to `docs/04 §3`.** That section states *"you need roughly |b − c| ≥ 13"*
> and *"minimum detectable paired difference ≈ 6.5pp"*, and gives b=26,c=14 → 3.60. The
> 3.60 is the **uncorrected** statistic, (26−14)²/40; with the continuity correction the
> same formula it states gives (12−1)²/40 = **3.02**. Solving the corrected form,
> |b−c| must exceed √(3.841·40) + 1 = 13.4, so the threshold is **14, not 13**, and the
> minimum detectable difference is **7.0pp, not 6.5pp**. `docs/04`'s own worked example
> (b=27,c=13) has |b−c| = 14, which contradicts its stated threshold of 13.
>
> The direction matters: the research claims it can detect a *smaller* difference than it
> actually can. Both of its significance verdicts still hold — only the boundary moves.

For contrast, an **unpaired** comparison of 85% vs 90% at 80% power needs
n ≈ 683 **per arm**. We have 200 total. **This is the single most important power fact in
the report**, and it is the reason every comparison is paired and every "improvement"
below ~6.5pp is reported as undetectable rather than as a win.

### Paired bootstrap, for macro-F1

Resample examples, recompute Δmetric, report the fraction of replicates with Δ ≤ 0.

### Multiple comparisons

10 slices at α = 0.05 ⇒ **1 − 0.95¹⁰ = 40.1%** chance of at least one spurious
"significant" result.

`config.evaluation.declared_slice_count: 10` is fixed **before** results are seen.
Holm–Bonferroni: the smallest p must beat 0.005 at k=10. The honest minimum, always met:
declare how many slices were examined and mark per-slice results **exploratory**.

---

## 6. Escalation as selective prediction

**Metrics:** risk–coverage curve; **AURC** and **E-AURC** (excess over the oracle —
unitless, so it compares across systems with different base error rates); **selective
risk at fixed coverage** (the most operationally legible single number);
precision/recall of the escalation trigger treating "should escalate" as positive;
accuracy at 100% coverage as the trivial baseline.

**The threshold comes from an explicit cost matrix, not from a round number.** Accuracy
weights a wrongly auto-sent reply to a furious customer identically to a needless
escalation; those costs differ by orders of magnitude. Elkan's threshold:

```
T = C_FP / (C_FP + C_FN)
```

At `config.cost_model.bad_autosend_vs_needless_escalation: 20`, **T = 20/21 ≈ 0.952** —
auto-handle only at ≥95% confidence.

**Ship the sensitivity table** over ratios {5, 10, 20, 50}, with coverage and expected
cost at each:

| Cost ratio | T | Coverage | Selective risk | Expected cost |
|---:|---:|---:|---:|---:|
| 5 | 0.833 | | | |
| 10 | 0.909 | | | |
| **20** | **0.952** | | | |
| 50 | 0.980 | | | |

The assumed ratio is an **explicit, arguable assumption**, stated as such. That table is
more persuasive than any single accuracy figure, because it converts an ML result into a
business decision — and it invites the reviewer to substitute their own ratio.

### Calibration

Calibration matters more than raw accuracy here, because escalation is a downstream
decision on the confidence score.

- **Brier score is the headline** — a proper scoring rule, decomposable into reliability
  + resolution + uncertainty.
- **ECE and a reliability diagram are supporting evidence.** ECE is binning-dependent,
  unstable and biased; internal compensation within a bin makes it optimistic. At n=200
  a 10-bin ECE has ~20 points per bin. **Report it, but never compare two ECEs and call
  the difference real.**
- **Do not trust LLM self-reported confidence.** The literature is mixed and on net
  negative: RLHF increases verbalized overconfidence, and reward models systematically
  favour high-confidence responses regardless of quality. Prefer logprob-derived scores
  where exposed. **Measure your own calibration on your own 200; do not inherit a claim.**
- **Compare confidence sources on one risk–coverage plot**: token logprob,
  self-consistency at k=5 (answer-agreement rate — usually the strongest cheap signal),
  ensemble disagreement, and a trained correctness predictor over features (retrieval
  score, thread length, policy-keyword presence). **That single plot is a genuinely
  differentiating deliverable.**

Present the operational form: *"at a 0.8 threshold we auto-handle 62% of traffic at 94%
accuracy."*

---

## 7. The judge

### Rubric — six binary criteria

Not a 1–5 Likert. Binary kills score compression and 4/5-clustering outright, and gives
**each criterion its own κ**. This is the biggest single reliability win available.

1. `addresses_stated_problem`
2. `grounded_in_retrieved_context`
3. `no_unsupported_policy_commitment`
4. `correct_brand_voice`
5. `correct_escalate_or_autohandle_decision`
6. `no_pii_or_unsafe_content`

Each returns `{pass: bool, quote: str}`. **A `pass` on a content criterion requires a
non-empty verbatim quote** — that is what makes judge errors auditable rather than
merely recorded.

**Headline composite:** all six pass. A reply is good or it is not; averaging six binaries
into a score re-introduces exactly the compression the binary rubric removes.

### Mitigations applied (all of Tier 1)

| # | Mitigation | Why |
|---|---|---|
| 1 | **Judge family ≠ generator family** | Self-preference is *linearly correlated with self-recognition capability*, with a causal link supported by fine-tuning. Clearest causal evidence of any judge bias; one config line. |
| 2 | **Randomise order, score both orderings** | All LLM judges examined exhibit strong position bias, most favouring the first position. Measured consistency in the wild ~70–77% — not cosmetic. |
| 3 | **Binary checklist** | Kills score compression (a base judge "never predicts 5 and 6, and barely predicts 3 and 2"). |
| 4 | **Structured output + required verbatim quote** | Forces grounding; makes errors auditable. |
| 5 | **Brief CoT before the score** | Nearly free, consistent gains. |
| 6 | **Report κ, not raw agreement** | See below. |

Tier 2, budget permitting: few-shot anchoring with human-scored exemplars (**which must
then be excluded from the eval set**); a 3-judge panel from disjoint families — a panel
of three small models outperforms a single large judge with less intra-model bias at
~7× lower cost.

**Reference-guided grading caveat.** On this corpus the historical reply is often itself
mediocre. Pass it as **context, never as the target**. Scoring similarity-to-history
rewards mimicking mediocrity; scoring "better-than-history" rewards verbosity. Say which
you did.

### Judge validation deliverable

Per criterion, against the human labels:

| Criterion | Raw agr. | Wilson CI | **Cohen's κ** | 2×2 (a,b,c,d) | Direction | vs κ_intra ceiling |
|---|---|---|---|---|---|---|

Plus: the **3-paraphrase flip rate**, reported as a result. GPT-4o flips its rating of
the same summary on 8.5% of pairs under semantically-preserving rubric rephrasings; our
score is conditional on one rubric wording and we quantify how much that matters.

**Report κ as the headline, not exact-match.** κ deflation between exact-match and
Cohen's κ is **universal: 33–41 percentage points on MT-Bench**. Raw agreement would look
~35pp better than the honest number. The Minimum Viable Validation Protocol is exactly
this: report Cohen's κ alongside any exact-match figure and treat the chance-corrected
number as the headline.

**Bands.** Krippendorff's thresholds, which are stricter than Landis–Koch and better
suited here: **α ≥ 0.80 reliable; 0.667–0.80 tentative conclusions only; < 0.667
unreliable, do not use.** Wild κ values for context: human–human typically 0.50–0.80;
strong judges 0.65–0.75 on subjective tasks.

### The kappa paradox — pre-empted, not explained away afterwards

Escalation is rare in this corpus, so κ **will** be deflated by the base rate. At an
identical **95% raw agreement**:

- **Skewed** (5% escalation): a=5,b=5,c=5,d=185 → P₀=0.95, Pₑ=0.905 → **κ = 0.47**
- **Balanced** (50/50): a=95,b=5,c=5,d=95 → P₀=0.95, Pₑ=0.50 → **κ = 0.90**

Identical agreement; κ differs by 0.43 purely from base rate. So `cohens_kappa()` returns
**P₀, Pₑ, κ, the full 2×2 table, PABAK and Gwet's AC1** together, and all of them are
published. Reporting κ alone here would be misleading in our own favour or against it,
depending on which way the reviewer reads it.

**The ceiling.** κ_intra from `docs/13 §9` is reported **before** the judge's κ. If
κ_intra = 0.83 and the judge reaches 0.71, that is the honest framing. A judge κ
exceeding κ_intra is a red flag, not a triumph.

**What we cannot claim.** The Alternative Annotator Test gives a statistical procedure
for justifying LLM replacement of human annotators, but it assumes **multiple** human
annotators. We cite it as **the standard we cannot fully meet**, not one we passed.

---

## 8. Groundedness and policy invention

**Policy invention gets its own metric, not a judge sub-score.** See `docs/12 §8` for the
mechanism and the Air Canada rationale.

Reported as: *"0/100 unsupported policy commitments on the random slice; 95% upper bound
3.6% by the rule of three."* An upper bound is the honest form for a near-zero rate.

The **policy-trap stratum is reported separately** and will be far worse. **That gap is
a finding, not an embarrassment** — it is the measurement that shows the random-slice
number is not the whole story.

**Limits stated alongside, every time:**
- **Faithfulness ≠ correctness.** A true claim absent from retrieved chunks is flagged; a
  wrong claim entailed by a bad chunk passes.
- **Faithfulness stays high when retrieval fails** — the generator answers coherently
  from partial context, so a high score can mask a retriever that missed the document.
  **Always report retrieval metrics (recall@k, MRR, nDCG@k) alongside, never instead.**
- Cherry-picking and sycophancy survive faithfulness checks.
- Claim decomposition is itself an LLM step and inherits every judge bias.
- nDCG on relevance labels you invented is circular.

---

## 9. Output tables

`make eval` prints these, in this order, to stdout and to
`artifacts/results/tables/*.md`.

**T1 — Intent classification** (all 200; and high-confidence subset)
`system | accuracy [CI] | macro-F1 all [CI] | macro-F1 in-scope [CI] | OOS P/R | vs TF-IDF: Δ, McNemar χ², p`

**T2 — Per-class** (stratified stratum; marked *diagnostic, not headline*)
`intent | support | P | R | F1 | most-confused-with`

**T3 — Reply quality** (random stratum only)
`system | all-six-pass [Wilson CI] | per-criterion pass rates | judge κ vs human`
Rows always include `dm_us_degenerate` and `historical` (the brand's own replies).

**T4 — Escalation / selective prediction** (random stratum only)
`cost ratio | T | coverage | selective risk | AURC | E-AURC | escalate P/R`

**T5 — Judge validation**
As §7. Includes κ_intra ceiling and the paraphrase flip rate.

**T6 — Policy invention**
`stratum | n | unsupported commitments | rate | 95% upper bound`

**T7 — Ablations**
`rerank on/off | playbook on/off | intent-filter on/off | Δ macro-F1 [paired bootstrap]`

Every table carries `config_sha256` and the golden-set content hash in its footer. A
results table that cannot be attributed to a configuration is not a result.

---

## 10. The headline

Not accuracy. This shape:

> **"On a frozen, uniformly-sampled 100-tweet slice, the agent produced a reply meeting
> all six rubric criteria in 78% of cases (95% Wilson CI 69–85%), with 0 unsupported
> policy commitments (95% upper bound 3.6%), at 62% auto-handle coverage under a 20:1
> cost ratio. The judge agrees with my labels at κ = 0.71 on the binary auto-handle
> criterion, against an intra-annotator ceiling of κ = 0.83."**

Every clause carries its own uncertainty, names its sample, and states the ceiling.

The honesty checklist (`docs/04 §8`, with item 8 corrected to ±8pp) runs immediately
beneath it. On an assignment where *"the proof is worth more than the system"*, **the
checklist is the proof** — and items 6, 12 and 2 (the DM-us degenerate baseline, the
offline/online gap, and label leakage) are the three that most separate a serious
submission from a confident one.

---

## 11. CI

`make smoke` runs on every commit. The golden set and harness are baked in, so a
regression in any headline number is visible in the diff of
`artifacts/results/tables/*.md` rather than discovered on submission day.
