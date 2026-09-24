# Evaluation design — the core deliverable

Research pass 4. The brief says *"the proof is worth more than the system."* This document is
the proof. Everything else in the repo is the object under test.

> **Verify before quoting.** Every figure below came from search summaries of papers, not
> full-text reads. Statistics I computed directly (Wilson intervals, McNemar, the kappa paradox
> worked example) are exact. Check any paper number against the PDF before it enters the report.

## 1. Golden set — 200 examples, 4 tagged strata

The four sampling designs estimate **different quantities**:

- **Pure random** — unbiased estimate of production performance, but at n=200 with a long-tail
  intent distribution you get ~3 "billing dispute" and 60 "where is my order". Useless for
  diagnosis.
- **Stratified by intent** — good for diagnosis and stable per-intent numbers, but every
  aggregate estimates a *reweighted* population, not your traffic.
- **Adversarial oversampling** — maximally informative per label, but **pessimistic by
  construction** and cannot be quoted as performance.
- **Hybrid with a held-out random slice** — the only design that supports both "here is what
  breaks" and "here is what production looks like".

**The unbiased random stratum is non-negotiable.** Any headline of the form "handles X% of
tickets correctly" is a claim about the distribution of tickets you will actually see. Reserve
100 examples by **simple random sampling** from the deduplicated corpus, freeze them with a
content hash, and **report the headline only off that slice.**

| Stratum | n | Purpose |
|---|---:|---|
| **Random (frozen)** | **100** | **Unbiased. The headline number lives here and nowhere else.** |
| Intent-stratified | 60 | Diagnosis; floors rare classes at ~8–10 each |
| Adversarial | 25 | Profanity, sarcasm, multi-intent, non-English, no-resolution threads, already-escalated customers |
| Policy-trap | 15 | Tweets that bait a policy answer ("how long do I have to return this?") |

Tag every row with its stratum **at insertion time** so metrics compute per-stratum and
aggregate. Per-example fields: `id, stratum, intent, auto_handle_vs_escalate,
quality_of_historical_reply(1-5), confidence ∈ {high,med,low}, ambiguous: bool, note,
guideline_version, label_date`.

## 2. Label quality with exactly one annotator

You cannot compute inter-annotator agreement alone. You **can** produce credible evidence:

1. **Write guidelines before labelling, version them, ship them.** Research on sentiment
   annotation found detailed instructions alone do *not* guarantee consistency — but undocumented
   criteria guarantee inconsistency and make the taxonomy unfalsifiable. Include a decision tree,
   ≥2 worked examples per class, and an explicit tie-break rule.
2. **Pilot round.** Label 25–30, revise the guidelines, **discard those labels**, start over.
   **The diff between guideline v1 and v2 is itself evidence of rigour** — ship it.
3. **Intra-annotator (test–retest) agreement — the solo annotator's substitute for IAA.**
   *Consistency is Key* (arXiv:2301.10684) argues intra-annotator agreement should be reported
   alongside IAA to measure label *stability*, and notes it is rarely reported in NLP.
   **Method: re-label a random 40 after ≥48 hours, blind, in shuffled order; report κ_intra.**
   **If κ_intra = 0.65, no downstream judge can credibly exceed that — it is your measurement
   ceiling, and saying so is one of the strongest moves available in the whole report.**
   (Caveat honestly: test–retest has its own biases — fatigue, memory of the first pass.)
4. **Adjudication log.** For every hard case, record the competing readings and the rule applied,
   then fold the rule back into the guidelines.
5. **Report metrics twice** — all 200, and the high-confidence/non-ambiguous subset. **If the gap
   is large, your label noise is doing real work and you must say so.**

## 3. Statistical power — the numbers that discipline every claim

### Wilson score interval (use this, not Wald)

```
        p̂ + z²/2n  ±  z·√( p̂(1−p̂)/n + z²/4n² )
CI  =  ─────────────────────────────────────────
                    1 + z²/n
```

**Worked example A — headline accuracy, p̂ = 0.85, n = 200, z = 1.96.**
centre = (0.85 + 0.009604)/1.019208 = 0.8434; half-width = 0.0495.
→ **95% CI ≈ [0.794, 0.893] — a width of ~10 percentage points.**

Read that twice. **"85% accuracy" is statistically indistinguishable from 80% and from 89%.**
Any improvement you claim smaller than ~10pp is inside the noise of a single measurement.

**Worked example B — a per-intent slice, p̂ = 0.84 (21/25).**
→ **95% CI ≈ [0.654, 0.936], width ~28pp.** This number carries essentially no information.
**Per-intent numbers at n=200 are nearly meaningless for rare intents. Say so rather than
tabling them silently.**

**Worked example C — a perfect slice, p̂ = 1.0, n = 25.** Wald gives the absurd [1.0, 1.0];
**Wilson gives [0.867, 1.0]**. At n=200 with zero errors Wilson gives [0.981, 1.0] — consistent
with the **rule of three**: with 0 failures in n trials the 95% upper bound on the failure rate
is ≈ **3/n**.

Support: *Position: Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints*
(arXiv:2503.01747, ICML 2025 spotlight) finds CLT methods "dramatically underestimate
uncertainty" in small-data evals and recommends **Wilson or Bayesian intervals**. Framing:
Evan Miller, *Adding Error Bars to Evals* (arXiv:2411.00640).

### Macro-F1 CIs

No clean closed form (ratio of correlated counts). Use a **nonparametric bootstrap over
examples**, B = 10,000, percentile or BCa. Resampling at the *example* level automatically
propagates rare-class instability — expect macro-F1 CIs **15–25pp wide** when a class has <10
support. **Bootstrap replicates where a rare class has zero support leave F1 undefined; decide
and document the convention** (drop the class from that replicate, or score it 0) — the two
choices give materially different intervals.

### Paired comparison — McNemar

Comparing two systems on the *same* 200 examples is paired. **Overlapping CIs do not imply no
significant difference.**

```
χ² = (|b − c| − 1)² / (b + c),   df = 1,   reject at χ² > 3.841
```

**Worked example.** With b + c = 40 discordant pairs, you need roughly **|b − c| ≥ 13**:
b=27,c=13 → χ² = 4.23 (significant); b=26,c=14 → 3.60 (not). **Minimum detectable paired
difference ≈ 6.5pp.**

For contrast, an **unpaired** comparison of 85% vs 90% at 80% power needs
n ≈ 7.849·[0.85·0.15 + 0.90·0.10]/0.05² ≈ **683 per arm.** You have 200 total. **This is the
single most important power fact in the report.**

For macro-F1, use the **paired bootstrap** (Berg-Kirkpatrick et al., EMNLP 2012): resample,
recompute Δmetric, report the fraction of replicates with Δ ≤ 0. Decision guide: Dror et al.,
*The Hitchhiker's Guide to Testing Statistical Significance in NLP* (ACL 2018). Canonical
warning against single-number comparisons: Reimers & Gurevych, arXiv:1803.09578.

### Multiple comparisons

10 slices at α = 0.05 ⇒ **1 − 0.95¹⁰ = 40.1%** chance of at least one spurious "significant"
result. Holm–Bonferroni (smallest p must beat 0.005 at k=10) or Benjamini–Hochberg FDR. **The
honest minimum: declare how many slices you looked at and mark per-slice results exploratory.**

## 4. Agreement metrics

| Task | Metric | Why |
|---|---|---|
| Binary auto-handle vs escalate | **Cohen's κ** + raw agreement + the 2×2 table + per-class P/R of *escalate* | Chance-corrected, but unstable under skew — publish the table |
| 1–5 ordinal reply quality | **Quadratic-weighted κ** or **Krippendorff's α (ordinal)**, + Spearman ρ + exact/±1 agreement | Penalises 1↔5 more than 4↔5; ρ captures ranking utility under scale shift |
| Multi-class intent | **Cohen's κ** / **α (nominal)** + confusion matrix + per-class F1 | The matrix is what tells you which two intents collapse |

**Bands.** Landis & Koch (widely used, widely criticised as arbitrary): <0.20 slight, 0.21–0.40
fair, 0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.00 almost perfect. **Krippendorff's own
thresholds are stricter and better suited here: α ≥ 0.80 reliable; 0.667–0.80 tentative
conclusions only; < 0.667 unreliable, do not use.**

### The kappa paradox, instantiated on this exact problem

Feinstein & Cicchetti (1990) established: asymmetric marginals produce **higher** κ than balanced
marginals at identical observed agreement. Suppose n = 200 and **95% raw agreement in both cases**:

- **Skewed** (escalation rare, 5% base rate): a=5, b=5, c=5, d=185.
  P₀ = 0.95, Pₑ = 0.05² + 0.95² = 0.905 → **κ = 0.045/0.095 = 0.47** ("moderate").
- **Balanced** (50/50): a=95, b=5, c=5, d=95.
  P₀ = 0.95, Pₑ = 0.50 → **κ = 0.45/0.50 = 0.90** ("almost perfect").

**Identical raw agreement; κ differs by 0.43 purely from base rate.** Escalation *will* be rare
in this corpus, so **expect a deflated κ and pre-empt the criticism**: report P₀, Pₑ, κ, the full
2×2 table, and **PABAK or Gwet's AC1 as a sensitivity check.**

### What "good enough" actually looks like

- **Zheng et al. (arXiv:2306.05685, NeurIPS 2023)**: GPT-4 achieves **over 80% agreement** with
  human preferences — "the same level of agreement between humans". **But that is raw agreement,
  not chance-corrected.**
- **Reliability without Validity (arXiv:2606.19544)** — 21 judges, 9 providers, 118 runs, ~541k
  judgments — reports **κ deflation between exact-match and Cohen's κ is universal: 33–41
  percentage points on MT-Bench.** Documents a "consistency–bias paradox": test–retest >0.95
  coexisting with position bias >0.10. Its **Minimum Viable Validation Protocol: report Cohen's κ
  alongside any exact-match figure and treat the chance-corrected number as the headline.**
  **This is the single most quotable citation for this report.**
- Wild κ values: human–human typically **0.50–0.80**; strong judges **0.65–0.75** on subjective
  tasks, **>0.90** on near-objective. Production ship bar: **κ ≥ 0.80 strong, 0.60–0.80
  substantial, <0.60 means the rubric needs work.**
- **Calderon, Reichart & Dror, *The Alternative Annotator Test*** (arXiv:2501.10970, ACL 2025)
  gives a statistical procedure for justifying LLM replacement of humans. It assumes **multiple**
  human annotators — **cite it as the standard you cannot fully meet, not one you passed.**

## 5. LLM-as-judge — failure modes and what to do

### The literature worth citing

- **Zheng et al. 2023** — foundational. Position bias ("all LLM judges examined exhibit strong
  position bias, most favouring the first position"), verbosity bias, self-enhancement bias.
  Two mitigations that still matter most: **CoT grading** and **reference-guided grading** — the
  latter **cut GPT-4's failure rate on math from 70% to 15%.**
- **G-Eval** (arXiv:2303.16634) — CoT + form-filling; Spearman **0.514** with humans on
  summarisation. Origin of the probability-weighted-score tie-break.
- **Prometheus 2** (arXiv:2405.01535) — open evaluator LM with **custom rubrics**. Lesson:
  fine-grained criterion-specific rubrics beat generic "helpfulness".
- **FLASK** (arXiv:2307.10928, ICLR 2024) — decomposes coarse scoring into **12 skills**.
  Principle: **score sub-criteria, not a single gestalt number.**
- **JudgeBench** (arXiv:2410.12784, ICLR 2025) — existing judge benchmarks measure alignment with
  human *preference*, a poor proxy for **correctness**. Directly relevant: **a support reply can
  be preferred and still be wrong about policy.**
- **Replacing Judges with Juries** (arXiv:2404.18796) — a **PoLL** of three small models from
  *disjoint* families outperforms a single large judge across six datasets, with less intra-model
  bias, at **~7× lower cost**.
- **LLM Evaluators Recognize and Favor Their Own Generations** (arXiv:2404.13076) — self-preference
  is **linearly correlated with self-recognition capability**, with a causal link supported by
  fine-tuning. **This is the hard evidence that judge and generator must not share a family.**
- **JudgeSense** (arXiv:2604.23478) — **GPT-4o flips its 1–5 coherence rating of the same summary
  on 8.5% of pairs under semantically-preserving rephrasings of the prompt.**
- **Quantitative LLM Judges** (arXiv:2506.02945) — score compression: a base judge "never predicts
  Likert scores 5 and 6, and barely predicts 3 and 2"; scores concentrate near the domain mean
  while humans use the full range.
- **How to Correctly Report LLM-as-a-Judge Evaluations** (arXiv:2511.21140) — derives a
  **plug-in bias-corrected estimator** and a CI accounting for uncertainty from *both* the test
  set and the calibration set. **This is exactly your situation** (200 human labels calibrating a
  judge run over thousands of tweets). Code: `github.com/UW-Madison-Lee-Lab/LLM-judge-reporting`.
- **Prediction-Powered Inference** (*AutoEval Done Right*, arXiv:2403.07008; Stratified PPI,
  arXiv:2406.04291) — combines a **small gold set** with a **large LLM-judged set** for provably
  unbiased estimates with **tighter** CIs than the gold set alone.
- **Large Language Model Hacking** (arXiv:2509.08825) — 37 tasks, 18 models, 13M labels:
  **empirical "LLM hacking" risk of 31% (best models) to 50% (smaller models)** that plausible
  implementation choices flip a statistical conclusion. *"With just a handful of prompt
  paraphrases, virtually anything can be presented as statistically significant."*

### Mitigations ranked by reliability gain per unit effort

**Tier 1 — all cheap, do all of them:**
1. **Different model family for judge than generator.** One config change; removes the bias with
   the clearest causal evidence.
2. **Randomise order, score both orderings, average or call ties.** Measured consistency in the
   wild runs ~70–77%, so this is not cosmetic.
3. **Binary rubric checklist instead of a raw Likert.** Replace "rate 1–5" with 5–8 yes/no
   criteria. **Kills score compression and 4/5-clustering outright, and each binary criterion
   gets its own κ. This is the biggest single reliability win available.**
4. **Structured output with per-criterion sub-scores and a required verbatim evidence quote.**
   Forces grounding; makes judge errors auditable.
5. **CoT before the score.** Nearly free, consistent gains.
6. **Report κ, not raw agreement.**

**Tier 2 — if budget allows:** reference-guided grading (**caveat: on this corpus the historical
reply is often itself mediocre — use it as *context*, not as *the target***); few-shot anchoring
with your own human-scored exemplars, one per scale point (**those exemplars must then be
excluded from the eval set**); a **3-judge PoLL** from different families.

**Tier 3:** pairwise over absolute when comparing two systems; self-consistency at k=5 (the
spread is a free uncertainty signal); **prompt-perturbation robustness — run the judge under 3
rubric paraphrases and report the flip rate.** That last one is a *result you report*, not just
an internal check, and directly answers "how fragile is my judge?".

## 6. Groundedness — and the one failure that matters most

**Approaches:** NLI/entailment over decomposed atomic claims (DeBERTa cross-encoders; see CLATTER
arXiv:2506.05243, Luna arXiv:2406.00975); **RAGAS** (arXiv:2309.15217) faithfulness / answer
relevance / context precision-recall; retrieval metrics recall@k, MRR, nDCG@k.

**State these limits:**
- **Faithfulness ≠ correctness.** A true claim absent from retrieved chunks is flagged; a wrong
  claim entailed by a bad chunk passes.
- **Faithfulness stays high when retrieval fails** — the generator answers coherently from
  partial context, so a high score can mask a retriever that missed the document. **Always report
  retrieval metrics alongside, never instead.**
- **Cherry-picking and sycophancy survive faithfulness checks.**
- Claim decomposition is itself an LLM step and inherits every judge bias.
- nDCG on relevance labels you invented is circular.
- Production pattern: a **cascade** — cheap heuristic → NLI classifier → LLM judge only on
  borderline scores.

### Inventing policy deserves its own metric, not a sub-score

**Moffatt v. Air Canada (2024 BCCRT 149)** is the canonical case: the chatbot said a bereavement
discount could be claimed retroactively within 90 days; the real policy — on a page the bot
itself linked — forbade it. Negligent misrepresentation; the tribunal **rejected the argument
that the chatbot was "a separate legal entity responsible for its own actions."** *One invented
policy sentence, one legally binding obligation.*

**How to measure it:**
1. **Extract commitments, don't score prose.** A structured extractor emits typed slots:
   `refund_offered, refund_window_days, compensation_amount, promised_timeline,
   entitlement_claimed, escalation_promise`. **Empty is the common and correct case.**
2. **Verify each populated slot by strict entailment against retrieved evidence** — must be
   supported by a verbatim span in a retrieved historical brand reply. **Unsupported ⇒ hard
   fail**, no matter how good the reply reads.
3. **Report as a rate with a Wilson upper bound** — "0/100 unsupported policy commitments on the
   random slice; **95% upper bound 3.6%** by rule-of-three". An upper bound is the honest form
   for a near-zero rate.
4. **Report the policy-trap stratum separately.** It will be far worse than the random slice.
   **That gap is a finding, not an embarrassment.**
5. **A deterministic tripwire** — regex for currency amounts, day/hour counts, and modal
   commitment verbs ("we will", "you're entitled to", "guaranteed") — catches most of these at
   near-zero cost with no LLM in the loop.

## 7. Escalation as selective prediction

**Metrics:** risk–coverage curve; **AURC** and **E-AURC** (excess over the oracle — unitless,
comparable across systems with different base error rates); **selective risk at fixed coverage**
(the most operationally legible single number); precision/recall of the escalation trigger
treating "should escalate" as positive; accuracy at 100% coverage as the trivial baseline.
Caveat: *Entropy Alone is Insufficient for Safe Selective Prediction* (arXiv:2603.21172).

### Setting the threshold from an explicit cost matrix

Accuracy weights a wrongly auto-sent reply to a furious customer identically to a needless
escalation. Those costs differ by orders of magnitude. **Elkan's threshold** (IJCAI 2001):

```
T = C_FP / (C_FP + C_FN)
```

If a bad auto-send costs **20×** a needless escalation, **T = 20/21 ≈ 0.952** — auto-handle only
at ≥95% confidence. **Put the assumed cost ratio in the report as an explicit, arguable
assumption, and show a sensitivity table of coverage and expected cost across 5×, 10×, 20×,
50×.** That table is more persuasive than any single accuracy figure, and it converts an ML
result into a business decision.

### Calibration

- **Brier score as headline** (a proper scoring rule, decomposable into reliability + resolution
  + uncertainty), with **ECE and a reliability diagram as supporting evidence**.
- **ECE is binning-dependent, unstable and biased** (arXiv:2109.03480); internal compensation
  within a bin makes it optimistic. At n=200, a 10-bin ECE has ~20 points per bin — **report it,
  but never compare two ECEs and call the difference real.**
- **Is verbalized LLM confidence calibrated? Mixed, and on net no.** *Just Ask for Calibration*
  (EMNLP 2023) found RLHF'd models' verbalized confidences **better** calibrated than their
  conditional probabilities, cutting ECE ~50% relative. But Xiong et al. and *Taming
  Overconfidence in LLMs* (arXiv:2410.09724) show **RLHF increases verbalized overconfidence**
  and reward models systematically favour high-confidence responses regardless of quality.
  **Conclusion: measure your own calibration on your own 200; don't inherit a claim.**
- **Alternatives:** token logprob; **self-consistency at k=5** (answer-agreement rate — usually
  the strongest cheap signal); ensemble disagreement; a trained correctness predictor over
  features (retrieval score, thread length, sentiment, policy-keyword presence). **Compare them
  all on one risk–coverage plot — that plot is a genuinely differentiating deliverable.**

## 8. "What is misleading about my headline number?" — the mandatory section

Write these as declarative admissions, each with the **direction** and, where possible, the
**magnitude** of the bias.

1. **Circularity of taxonomy and golden set.** I defined the taxonomy, labelled the golden set
   against my own taxonomy, and built the system around it. Ambiguous cases were resolved by the
   person who benefits from them being resolvable. Inflates all intent numbers by an unknown amount.
2. **Label leakage from LLM-assisted labelling.** If an LLM proposed labels I confirmed, my
   "human" labels are partly model labels, and evaluating a model against them measures
   agreement-with-a-model. Anchoring alone inflates agreement. LLM-hacking risk: **31–50%**.
3. **Solo annotation.** No IAA exists. **Intra-annotator κ = [X] is a ceiling on any judge–human
   agreement I report**; a second annotator would lower it further.
4. **Base rate / class imbalance.** Majority-class already scores X%. Report majority and
   random-by-prior baselines beside the headline; report macro-F1 and per-class recall.
5. **Survivorship bias.** The dataset contains only tweets that **received a public reply**.
   Ignored complaints, DM-only conversations, and threads that moved to phone or email are
   absent. **The corpus is the subset of cases the brand chose to answer publicly — systematically
   easier and more PR-friendly than the true inbound distribution.**
6. **The "DM us" degenerate reply.** A system that always emits "Sorry to hear that! Please DM
   us" scores extremely well on reply-similarity and is worthless. **Report the corpus frequency
   of the pattern, your system's rate of producing it, and the similarity score of a
   constant-"DM us" baseline as an explicit degenerate control.** *How NOT To Evaluate Your
   Dialogue System* (arXiv:1603.08023, EMNLP 2016) showed BLEU-style metrics "correlate very
   weakly with human judgements in the non-technical Twitter domain, and not at all in the
   technical Ubuntu domain" — **that paper is about this exact domain; cite it and drop
   reply-overlap metrics from the headline.**
7. **Judge/generator family sharing** — self-preference scales with self-recognition.
8. **Small-n CIs.** ±~5pp on the headline at n=200; ±~14pp on a 25-example slice; ~683 per arm
   needed for an unpaired 5pp difference; nothing below ~6.5pp detectable on a paired comparison.
9. **Multiple comparisons** — 40% chance of a spurious result at k=10 slices.
10. **Single-brand / single-platform.** Twitter replies are short, public, tone-driven and
    heavily templated — nothing like an email or phone queue.
11. **Historical replies are a weak gold standard.** The agent's reply was often generic, wrong,
    or a deflection. Scoring similarity-to-history **rewards mediocrity**; scoring
    "better-than-history" rewards verbosity. **Say which you did.**
12. **Offline eval measures none of the business outcome** — no resolution rate, CSAT, handle
    time, repeat-contact, churn. Industry audits find **15–25% of "deflected" tickets were
    deflected with incorrect or incomplete answers**, and "60% resolved by AI" routinely means
    "60% where the customer stopped replying". **My numbers are upper bounds on a proxy.**
13. **Judge prompt fragility** — ~8.5% rating flips under semantically-preserving rubric
    rephrasings. My score is conditional on one rubric wording.
14. **Chance-uncorrected agreement is inflated** — κ deflation of **33–41pp** vs exact-match is
    universal. Raw agreement would have looked ~35pp better than the honest number.
15. **Contamination.** The dataset is public and old enough to be in pretraining corpora. **I
    cannot rule out that the model has seen the actual historical replies I score against.**
16. **Frozen snapshot.** Policies, products and tone guidelines changed after collection; a reply
    matching history may be wrong today.

## 9. The recommended headline

Not accuracy. This:

> **"On a frozen, uniformly-sampled 100-tweet slice, the agent produced a reply meeting all six
> rubric criteria in 78% of cases (95% Wilson CI 69–85%), with 0 unsupported policy commitments
> (95% upper bound 3.6%), at 62% auto-handle coverage under a 20:1 cost ratio. The judge agrees
> with my labels at κ = 0.71 on the binary auto-handle criterion, against an intra-annotator
> ceiling of κ = 0.83."**

Every clause carries its own uncertainty, names its sample, and states the ceiling. Run the §8
checklist immediately beneath it.

**On an assignment where "the proof is worth more than the system", the checklist *is* the
proof** — and items **6, 12 and 2** (the DM-us degenerate baseline, the offline/online gap, and
label leakage) are the three that most separate a serious submission from a confident one.

## 10. Harness summary

- Every number ships with a **Wilson 95% CI** (proportions) or a **10,000-replicate bootstrap CI**
  (macro-F1, means).
- Every system comparison is **paired** — McNemar with continuity correction for binary, paired
  bootstrap for F1.
- Baselines present in **every** table: majority class, random-by-prior, retrieval-nearest-neighbour
  reply, and the **constant-"DM us" degenerate control**.
- Judge: different family from the generator; **binary 6-criterion checklist** — *addresses the
  stated problem · grounded in retrieved context · no unsupported policy commitment · correct
  brand voice · correct escalate/auto-handle decision · no PII or unsafe content*; structured
  JSON; one verbatim evidence quote per criterion; brief CoT; 3 human-scored few-shot exemplars;
  randomised criterion order.
- Judge validation deliverable: per criterion — raw agreement, **Cohen's κ**, the 2×2 table, a
  Wilson CI on the agreement rate, direction of error (lenient/strict), κ against Krippendorff's
  bands and against your intra-annotator ceiling, plus the **3-paraphrase flip rate**.
- Scaling the judge beyond the golden set: use **PPI** or the plug-in bias-corrected estimator so
  the corpus-wide number carries a valid CI rather than being an uncorrected judge score.
- Bake the golden set + harness into CI so a regression is visible on every commit.
