# Work breakdown

Ordered, dependency-annotated, with a done-when assertion per task. Mapped onto the
five-day plan from `docs/05 §3`.

**Day 2 is the long pole and the one that slips. Protect it.** It slips for a specific
reason: labelling cannot start until the taxonomy is frozen *and* the guidelines are
written. `docs/13` now exists, which removes half of that blocker; T2.1–T2.3 remove the
rest and are therefore the critical path.

Legend: **⛔** blocks the report · **🔬** experiment path, not on the reproduction path.

---

## Day 1 — data

| # | Task | Depends on | Done when |
|---|---|---|---|
| T1.1 | Download `twcs.csv`; record sha256 into `config.corpus.raw_sha256` | — | Row count reads **2,811,774** and the checksum is committed |
| T1.2 ⛔ | **Read the Kaggle license string yourself and quote it** | T1.1 | The verbatim string is in `docs/01 §7`, replacing the "unverified" marker. **Do not assert a license you have not read.** |
| T1.3 | Implement `build_sample.load_raw` + `build_threads` | — | `threads.parquet` written; all §1 invariants from `docs/11` assert clean |
| T1.4 | Verify the four silent-bug defences | T1.3 | A test row with a null parent, a dangling parent, a split `N/M` reply, and a numeric-handle inbound tweet all round-trip correctly |
| T1.5 ⛔ | **Hand-read 100 random SpotifyCares threads** | T1.3 | A written note in `reports/brand_verification.md` either confirming or **overturning** the brand choice |
| T1.6 | Deterministic conversation-level sample → `sample_ids.txt` | T1.3 | Re-running on a different machine produces a byte-identical file |

**T1.5 is the highest-leverage task in the whole plan.** Every published deflection
figure — including all three sources in `docs/01 §5` — is a regex heuristic. Nobody has
human-verified in-thread resolution rates for any brand in this corpus. One afternoon of
reading beats every number in the research docs, and it is the difference between *"I
picked SpotifyCares because three repos' regexes agree"* and *"I picked it, then read 100
threads to check."* If the reading overturns the choice, switching to `hulu_support` on
day 1 costs almost nothing; discovering it on day 4 costs everything.

---

## Day 2 — taxonomy and labels (the long pole)

| # | Task | Depends on | Done when |
|---|---|---|---|
| T2.1 🔬 | Taxonomy induction: MiniLM → UMAP → k-means → LLM naming | T1.3 | Candidate labels with definitions and boundary cases |
| T2.2 | Two merge/split rounds → 8–12 in-scope intents | T2.1 | `intents.yaml` has `status: frozen`, `frozen_at` set, `config.taxonomy.frozen: true` |
| T2.3 | **Pilot: label 30 (outside the 200) → revise `docs/13` to v2 → discard the 30** | T2.2 | `docs/13-annotation-guidelines-v2.md` exists; the v1→v2 diff is committed; §10's known defects are resolved |
| T2.4 ⛔ | Label all 200 against v2, stratum-tagged at insertion | T2.3 | `golden_200.csv` complete; stratum counts exactly 100/60/25/15; content hash pinned |
| T2.5 | Coverage check | T2.4 | `other_unclear` rate on the random 100 is **≤15–20%**. Above that, a class is missing — **go back to T2.2**, do not proceed |

T2.3 is non-negotiable and is the step most likely to be skipped under pressure. Its
output — the v1→v2 diff — is cheap, is evidence of rigour, and costs an hour. Skipping it
saves an hour and removes the only documentation that the guidelines were ever tested.

T2.5 is a genuine stop-and-return gate, not a checkpoint. Labelling 200 rows against a
taxonomy with a missing class produces 200 rows that have to be redone.

---

## Day 3 — baselines, retrieval, policy

| # | Task | Depends on | Done when |
|---|---|---|---|
| T3.1 | Baseline ladder tiers 0, 0.5, 1, 2 | T2.4 | `make eval` prints T1 with Wilson CIs on every cell |
| T3.2 | Tune TF-IDF properly (stratified 5-fold, mean ± std) | T3.1 | Tuning is in the repo and reproducible — **an untuned "simple baseline" is a strawman and the easiest thing for a grader to spot** |
| T3.3 | `featurize` + retrieval (BM25 ‖ dense → RRF → MMR, brand+intent filtered) | T1.3 | Retrieval returns 3 non-duplicate exemplars on a hand-checked query |
| T3.4 | Build the resolution-precedent index (not 1-hop pairs) | T3.3 | Precedent count recorded and reported — expect **far fewer than the raw reply count** |
| T3.5 | Policy ladder + **one unit test per rung** | — | All 18 `test_policy.py` cases pass |
| T3.6 | Decision record + `config.yaml` wiring | T3.5 | `decisions.jsonl` validates against `docs/11 §5` |

T3.4's ratio (clean precedents ÷ total replies) is a reportable finding in its own right.
The research puts it near 11,944 / 43,265 for Spotify; recompute it and report what you
get, because **65.9% of first replies are diagnostic questions, not resolutions**, and a
system indexing those learns to ask questions forever.

---

## Day 4 — judge, grounding, calibration

| # | Task | Depends on | Done when |
|---|---|---|---|
| T4.1 ⛔ | **Blind re-label 40 after ≥48h → κ_intra** | T2.4 + 48h | `reports/kappa_intra.json` has κ for intent and for auto/escalate |
| T4.2 | Judge implementation, six binary criteria, both orderings | T2.4 | `judge_verdicts.jsonl` validates; every content-criterion pass carries a verbatim quote |
| T4.3 ⛔ | Judge-vs-human agreement, per criterion | T4.1, T4.2 | T5 populated: raw agreement, Wilson CI, **κ**, 2×2, direction, vs the κ_intra ceiling |
| T4.4 | 3-paraphrase flip rate | T4.2 | A number, reported as a **result** |
| T4.5 | Judge the **brand's own historical replies** | T4.2 | The reference band exists — this is what reframes the headline |
| T4.6 | Grounding validator + commitment extractor | T3.3 | All 12 `test_grounding.py` cases pass |
| T4.7 | Risk–coverage, calibration, Elkan sensitivity table | T3.1 | T4 populated across ratios {5, 10, 20, 50} |
| T4.8 | Error analysis pass: **read real outputs** | T3.1 | Top-5 off-diagonal cells identified with three verbatim tweets each |

**T4.1 must start the clock on day 2.** The ≥48-hour wait is the binding constraint on
the entire schedule: if the 200 are finished late on day 3, κ_intra cannot exist before
day 5. Finish T2.4 on day 2 and the wait costs nothing.

T4.5 is cheap and disproportionately valuable. If the brand's own replies pass 3.4 of 6
criteria and the model passes 3.6, the entire meaning of the headline changes — and that
reframing is worth more than several points of model quality.

T4.8 is where the five failure modes come from. **Real examples, not categories** — the
brief asks specifically for verbatim examples, which requires actually reading outputs.
Candidates to confirm or refute (`docs/05 §4`): mid-thread fragment misclassification;
`praise_thanks` over-prediction; agent-signature hallucination; policy invention on the
trap stratum; multi-intent collapse. **Predictions are not findings** — each needs a real
example, a hypothesis, and a proposed test that would confirm or refute it.

---

## Day 5 — report and hardening

| # | Task | Depends on | Done when |
|---|---|---|---|
| T5.1 ⛔ | Verify Hiver's current AI branding and customer count | — | Quoted from `hiverhq.com` **on the day you submit**. Naming a deprecated feature reads as stale research |
| T5.2 ⛔ | Verify each incident citation against primary sources | — | Air Canada, DPD, Chevrolet, Klarna — dates, names, amounts exact |
| T5.3 | Commit the LLM cache, sorted by key | T3.1 | `make eval-full` runs with no API key set |
| T5.4 | Report ≤6 pages | all | Framing · ≥2 baselines · 5 failure modes · **the misleading section** · next week |
| T5.5 | Decision log, 10–15 **non-obvious** entries | all | See §"Decision-log additions" below |
| T5.6 | **Time the reproduction with a stopwatch on a clean machine** | T5.3 | `git clone && make setup && make eval` under 15 minutes, measured, not estimated |
| T5.7 | Honesty checklist, with item 8 corrected to **±8pp** | T4.3 | All 16 items, each with direction and where possible magnitude |

T5.6 means what it says. A reviewer will clone, run one command, and start a timer. If
that path needs a Kaggle login, an API key with credit, a GPU, or a 20-minute embedding
job, points are lost before a word of the report is read.

---

## Blocking verification items (carried from `docs/05 §7`)

These four need network access and human reading; none is done, and each blocks a
specific claim.

| Item | Blocks | Task |
|---|---|---|
| **Kaggle license string — unread, the page was blocked** | Any licensing claim in the README | T1.2 |
| **Deflection numbers are all regex heuristics, never human-verified** | The entire brand-selection argument | T1.5 |
| **Hiver AI branding (Harvey vs. "Hiver AI" / AI Agents) and customer count** | The framing section | T5.1 |
| **Incident citations: Air Canada, DPD, Chevrolet, Klarna** | The escalation and injection arguments | T5.2 |

Each is quoting someone else's fact. Getting one wrong costs more credibility than the
fact buys.

---

## Decision-log additions

`docs/05 §5` seeds 18. These implementation docs generate three more, all non-obvious:

19. **The reproducibility guarantee is the response cache, not `temperature=0`** —
    because sampling parameters no longer exist on current models. The cache is a
    strictly stronger guarantee than greedy decoding ever was, which was never
    bit-reproducible across serving-stack changes either.
20. **The licensing constraint and the 15-minute constraint are in direct tension**;
    resolved by making the *statistical* claim reproducible without the data (Tier 1) and
    the *generative* claim reproducible with it (Tier 2). The README states which tier
    proves what rather than implying Tier 1 is end-to-end.
21. **The headline confidence interval is ±8pp, not ±5pp**, because the headline lives on
    the n=100 random stratum, not the full n=200 golden set. Caught by recomputing
    Wilson rather than by re-reading the research — and understating your own uncertainty
    is the one error the honesty section cannot afford.
22. **Two of the research's own statistical figures were wrong in the same direction** —
    the headline CI (±5pp quoted, ±8pp actual) and the McNemar boundary (|b−c| ≥ 13 and
    6.5pp quoted, ≥ 14 and 7.0pp actual, with an uncorrected χ² presented under a
    corrected formula). Both overstate what the harness can resolve. Found by recomputing
    every quoted statistic rather than by re-reading, which is why the harness's own
    worked examples are pinned as unit tests against hand-computed values.

23. **The judge stays a different family even without the DGX** — OpenAI generates,
    Gemini judges. The DGX was the plan for the independent judge, but on inspection the
    account's 40 GB quota was already full and every GPU was occupied. Falling back to a
    same-family judge would have re-introduced the one bias with causal evidence behind
    it; a second API provider keeps the separation for a few dollars. What is lost is the
    privacy and pinned-weights argument for open models, stated as such in the report.
24. **All provider calls go through one module, behind the cache.** `src/llm_client.py`
    is the only file that imports an SDK; callers stay cache-first. Swapping a role's
    provider is a config edit, and a cache hit provably never reaches a provider
    (`tests/test_llm_client.py::test_cache_hit_never_reaches_a_provider`).

---

## Slip protocol

If day 2 runs long, cut in this order. The ordering is derived from the inferred rubric
weights in `docs/00` — cut from the 15%-weighted "working system" before the
30%-weighted "evaluation rigour".

1. SetFit (tier 3) — the ladder is complete without it.
2. The reranking ablation — state it as untested future work rather than half-doing it.
3. Playbook distillation — exemplar-only retrieval is an explicitly valid fallback.
4. The 3-judge panel — the single different-family judge already carries the argument.

**Never cut:** the unbiased random stratum · κ_intra · the degenerate "DM us" control ·
the honesty checklist · the sub-15-minute reproduction. Those are the assignment.
