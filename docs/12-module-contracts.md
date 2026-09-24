# Module contracts

One section per file in `src/`. Signature, behaviour, invariants, edge cases, and the
research decision each implements. An implementer should not need to reopen `01`–`04`.

Every threshold named here exists as a key in `config.yaml`. Nothing that a reviewer
might argue with is a constant in code — that is the demonstration of Hiver's *"You
decide when AI steps in… and set the bar for what needs human review."*

---

## 1. `build_sample.py` — raw CSV → threads

`load_raw(csv_path) -> LazyFrame` · `build_threads(lf, brand) -> DataFrame` ·
`normalise_text(str) -> str` · `deterministic_sample(ids, percent, algo) -> set` ·
`rejoin(sample_ids, golden, raw_csv) -> DataFrame`

Reads every column as `str` (`infer_schema_length=0`), streams, never materialises the
493 MB file.

**Four silent-correctness bugs this module exists to avoid:**

1. **`in_response_to_tweet_id` as float64.** NaN is present, so a naive `read_csv`
   infers float and silently corrupts IDs. Read as string, cast to nullable `Int64`.
2. **`created_at` is not ISO.** `Tue Oct 31 22:10:47 +0000 2017` needs an explicit
   `format=`.
3. **Split replies stored out of order.** Real case: three AmazonHelp tweets at the
   *identical* timestamp `06:23:00` marked `1/3^AP 2/3^AP 3/3^AP`, with 3/3 stored
   first. Sort by `created_at`, tie-break on `tweet_id`, then reorder any run sharing a
   timestamp by its parsed `N/M` marker.
4. **The dual-identity trap.** Customers write `@115821`; brand replies say
   `@AmazonHelp`. Filtering inbound rows on the brand's handle returns **nothing**.
   Recover the brand for an inbound tweet from the `author_id` of the row that replies
   to it — the reply edge — never from text matching.

**Thread reconstruction:** build `parent → children` from `in_response_to_tweet_id`
(one clean int per row) rather than `response_tweet_id` (comma-separated list). A tweet
is a root if its parent is null **or** its parent id is absent from the file. BFS from
each root; `conversation_id := root tweet_id`.

**Branch policy** (222,426 rows have multiple children): prefer the branch ending with a
brand reply → longest → lowest terminal `tweet_id`. Stated because flattening a tree
without a policy is a silent bug.

**`normalise_text`:** strip `[\^~*/][A-Z]{1,3}\s*$` (four confirmed sigils —
AmazonHelp `^SM`, Spotify `/AL`, Delta `*AMV`, comcastcares `~AT`), replace URLs with
`[URL]` (41.3% of AmazonHelp tweets carry a live `t.co` link, unmasked), collapse
whitespace. Without the sigil strip you get similarity clusters keyed on **agent
identity** and a generator that hallucinates initials — an observed failure.

**Scope filter:** `starters_only: true`. 62.5% of inbound tweets are thread starters;
the remaining 37.5% are context-free fragments like *"iPhone 8, iOS 11"*. Training the
classifier on those is training it on noise, and evaluating on them single-turn is
dishonest.

**`deterministic_sample`:** hash the **root id**, not the tweet id. Tweet-level sampling
leaks turns of the same thread across train and eval.

---

## 2. `induce_taxonomy.py` — taxonomy derivation

`select_k(embeddings) -> int` · `name_clusters(clusters, texts)` ·
`refine(labels, target_range, rounds)` · `inspect_outlier_bin(hdbscan_labels, texts)`

Not on the reproduction path; `make labels` only; requires an API key.

**`select_k` uses stability across seeds, not silhouette or elbow.** Both mislead on
UMAP-projected embeddings — silhouette rewards compact spherical clusters, which UMAP
manufactures. Method: cluster twice on 80% bootstraps, measure ARI between partitions.

**UMAP params are fixed and seeded:** `n_neighbors=30` (not the default 15 — noisy short
text), `n_components=5`, `metric='cosine'`, `random_state=42`. Without `random_state`
the run is not reproducible.

**`inspect_outlier_bin`:** read the HDBSCAN noise cluster before discarding it. On
multi-domain short text it has held >74% of the dataset. For *induction* that is a
feature — it is a preview of the `OTHER` class.

**`refine` runs twice** with an explicit merge/split/drop instruction and a stated target
of 8–12 intents. This is TnT-LLM's iterative-refinement step. Granularity discipline:
Banking77's 77 intents are the cautionary tale — ≥14% of its training utterances may be
mislabelled, and removing suspect ones raised accuracy 88.2% → 92.4%. Fine granularity
with semantic overlap manufactures label noise.

**Prior work to not repeat:** BERTopic is an exploratory lens, not the taxonomy
generator — on tweets its c-TF-IDF labels degenerate into brand names and support
boilerplate ("DM", "sorry", "team"). And unsupervised clustering alone did not produce a
usable taxonomy in any prior applied project on this corpus; the LLM naming plus human
merge/split rounds are what make it usable.

---

## 3. `featurize.py` — embeddings and lexical index

`embed(texts, model, batch_size)` · `save_embeddings(arr, path, dtype)` ·
`build_bm25(texts, k1, b)` · `dedupe_near_identical(texts)`

Chunking is a non-issue — tweets are ≤280 chars, one tweet is one unit. Say so
explicitly in the report: it shows you know *why* the usual RAG machinery does not apply
rather than cargo-culting it.

MiniLM (`all-MiniLM-L6-v2`, 384-d, ~80 MB) needs no query/passage prefix. `bge`/`gte`/`e5`
are stronger on MTEB but **require** the right prefixes (`query:`/`passage:` for e5);
getting the prefix wrong silently costs more than the model gains.

**BM25 is load-bearing, not a formality:** support text is full of exact tokens that
embeddings smear — order numbers, error codes, `iOS 17.2`, `#Error503`, SKUs.

`dedupe_near_identical` hashes normalised text. ~0.75% exact text dupes in a 500k sample
— this is the retrieval-leakage hazard. (Duplicate *rows* are not: the source has zero.)

---

## 4. `retrieve.py` — hybrid retrieval

`reciprocal_rank_fusion(lists, k=60)` · `mmr(cands, qv, dv, λ=0.7, top_k=3)` ·
`retrieve(query, intent, brand, config)` · `rerank(cands, query, model)`

**RRF** = `Σ 1/(k + rank)`, k=60 canonical. Needs no score normalisation across
heterogeneous scorers — that is why it is the industry default and why it is right for
mixing BM25 with cosine.

**MMR** at λ=0.7 stops the three exemplars from being three near-identical
*"sorry about that, DM us!"* tweets — which is exactly what unfiltered top-3 returns on
this corpus. ~15 lines.

**Filters.** `brand` is a **hard** filter: cross-brand retrieval is actively harmful —
Apple's tone must not leak into a Delta reply. `intent` is soft: keep
`config.retrieval.unfiltered_tail` cross-intent hits so a misclassification is not fatal.

**What gets indexed — the decision that matters most here.** Index *resolution
precedents*, not 1-hop (customer → first reply) pairs. **65.9% of first replies are
diagnostic questions, not resolutions**; index those and the RAG learns to ask questions
forever. Tracing multi-turn trees to a substantive solution *plus* a customer
confirmation ("thanks, it worked") yielded only ~11,944 clean precedents for Spotify out
of 43,265 replies. That ratio is a finding worth reporting.

**`rerank` is off by default and ablated either way.** Cross-encoders typically add
5–20% nDCG@10, but the same literature reports `bge-reranker-base` *degrading* nDCG by
−0.3% to −3.1% on corpora far from its training distribution. *"I tried it, measured it,
it didn't help on my corpus, here's the number"* beats both using it blindly and not
trying it.

**No FAISS, no Chroma.** At 100k texts a 100k×384 float32 matmul is ~150 MB and
sub-100 ms. A vector DB here is a dependency you must justify in a live code review for
zero measured benefit.

---

## 5. `guards.py` — ingest guard

`redact(text) -> (str, list[str])` · `scan_injection(text) -> bool` ·
`scan_safety(text) -> bool`

TWCS is only *partially* anonymised. Emails and phones are masked upstream; free text
still contains order numbers, addresses and account identifiers customers typed
themselves. Run the PII pass **on ingest and again on egress**. Never let the generator
echo a PII token into a public reply — that is the strongest argument for `DM_HANDOFF`.

Presidio is the production answer and is named as such in the report; a documented regex
set is the defensible take-home choice for a dependency-light repo.

**Injection detection is a routing signal, not a filter.** A hit sets
`SUSPECTED_INJECTION` and routes to trust-and-safety; it does not block. Treating attack
detection as routing rather than blocking is safer and more honest about detector
fallibility — a missed detection still passes through the rest of the ladder.

The other two defences are structural and live elsewhere:
- customer text sits in a delimited, explicitly-labelled-untrusted block, with the system
  instruction asserting it is **data to classify, never instructions to follow**
  (`generate.build_prompt`);
- the action space is a four-value enum, so the model *cannot* emit an action that does
  not exist (`schemas.Action`). This is the core of the constrained-action-space pattern,
  and it is what the Chevrolet-of-Watsonville "$1 Tahoe, legally binding, no takesies
  backsies" incident is a demonstration of.

---

## 6. `generate.py` — constrained generation

`build_prompt(message, playbook, exemplars, style_card) -> str` ·
`generate(prompt, config)` · `repair(raw, error, config)`

Structured output binding: `output_config={"format": {...}}` or `messages.parse()`.
**Not** `output_format`; **not** "respond in JSON"; **no** assistant prefill (removed,
400). Plain JSON schema beats tool calling for a single-shot classify-and-draft — do not
add an agent loop you do not need. See `docs/10 §6.2`.

There is **no `temperature` parameter** — sampling params are removed on current models
and return a 400. Determinism comes from the cache. See `docs/10 §6.1`.

`repair` runs at most `config.generation.max_repair_retries` (=1) and **every repair is
logged** to the decision record. An unbounded self-repair loop is a cost bomb and a bad
look.

Pydantic `@field_validator`s carry the business rules the JSON schema cannot express:
`evidence_ids` non-empty, `escalation_reason` present iff `action == "escalate"`, intent
within the frozen taxonomy, no URL absent from evidence.

---

## 7. `policy.py` — the escalation ladder

`apply_ladder(context, config) -> (Action, EscalationCode|None, PolicyLayer)` plus one
predicate per rung.

**The invariant that makes this defensible live:** deterministic guardrails can **veto**
automation but can **never authorize** it. Rungs 1–6 may only move the decision toward
`ESCALATE` (or `REQUEST_INFO`). Only rung 7 may return `AUTO_SEND` or `DM_HANDOFF`.
`tests/test_policy.py::test_guardrails_never_authorize_automation` is a property test
over randomised contexts.

Evaluate in order, **short-circuit on the first hit**, and **record which layer fired**.
Every decision is then traceable to the rule that produced it — which is the entire
argument against a single "should I escalate?" LLM call, and the thing a reviewer will
probe hardest.

| # | Rung | Fires when | Code |
|---|---|---|---|
| 1 | `SAFETY_VETO` | self-harm, threats, abuse | `SAFETY_RISK`, P0. **No confidence overrides.** |
| 2 | `COMPLIANCE_VETO` | legal/regulatory markers; fraud/account-security; PII in the public tweet; amount > `auto_refund_ceiling_usd` | `LEGAL_THREAT` / `ACCOUNT_SECURITY` / `PII_IN_PUBLIC_TWEET` / `MONEY_ABOVE_THRESHOLD` |
| 3 | `NO_PRECEDENT_GATE` | `top1_rrf < tau_retrieval`, or intent is `other_unclear` | `NO_PRECEDENT` |
| 4 | `CONFIDENCE_GATE` | `intent_confidence < tau_intent` | `LOW_CONFIDENCE` |
| 5 | `CONTEXT_GATE` | repeat contact within `repeat_contact_window_days`; thread ≥ `max_thread_turns_auto`; missing required slot | `REPEAT_CONTACT`, or → `REQUEST_INFO` for a missing slot |
| 6 | `POST_GENERATION_VALIDATION` | draft fails the grounding check | `UNGROUNDED_COMMITMENT` |
| 7 | `DEFAULT` | — | → `AUTO_SEND`, or `DM_HANDOFF` per the intent's playbook |

Rung 3 is the one worth explaining aloud: *nothing similar ever happened, so do not
invent a precedent.*

**The reason string is emitted in both forms.** The **code** is what dashboards
aggregate ("41% of escalations are `NO_PRECEDENT` → that's a knowledge-base gap"). The
**text** is what the human reads in three seconds:

```json
{"escalation_code": "MONEY_ABOVE_THRESHOLD",
 "escalation_reason": "Customer requests a $340 refund; auto-handling ceiling is $50. Two similar cases (t_88421, t_10233) were resolved by a billing specialist, not a first-line agent.",
 "route_to": "billing_specialist", "priority": "P2",
 "evidence_ids": ["t_88421", "t_10233"]}
```

`route_to` maps from `config.policy.routes`, mirroring Hiver's skill-based routing. The
handoff payload includes a thread summary — Hiver pitches AI Summarizer explicitly as
the escalation-handoff artifact.

**Future work, stated honestly:** a learned escalation classifier trained on *"did a
human eventually take this over?"* is the right long-term answer. We do not have that
label and do not pretend to.

---

## 8. `grounding.py` — the grounding validator

`tripwire(draft) -> list[str]` · `extract_commitments(draft) -> CommitmentSlots` ·
`verify_against_evidence(slots, exemplars, playbook) -> (bool, list[str])`

~60 lines, disproportionately impressive relative to its size.

**Two layers, cheapest first.** The deterministic tripwire (regex for currency amounts,
day/hour counts, modal commitment verbs) catches most invented policy at near-zero cost
with no LLM in the loop. The typed extractor then feeds strict entailment.

**Extract commitments; do not score prose.** Typed slots — `refund_offered`,
`refund_window_days`, `compensation_amount`, `promised_timeline`,
`entitlement_claimed`, `escalation_promise`. **Empty is the common and correct case.**

Each populated slot must be supported by a **verbatim span** in a retrieved historical
brand reply. **Unsupported ⇒ hard fail**, no matter how good the reply reads.
Independently, also verified: every number/amount/duration/date in the draft appears in
evidence or the playbook; every URL and phone number is on the allowlist observed in
that brand's own replies (hallucinated support URLs are a classic failure); no invented
agent sigil.

**Why this gets its own metric rather than a judge sub-score.** *Moffatt v. Air Canada*,
2024 BCCRT 149: the chatbot told a customer a bereavement discount could be claimed
retroactively within 90 days, contradicting the policy page it itself linked. The
tribunal found negligent misrepresentation and **rejected the argument that the chatbot
was "a separate legal entity responsible for its own actions"**. Damages CAD $812.02.
One invented policy sentence, one legally binding obligation. That is not a sub-score.

Reported as a rate with a **Wilson upper bound** — *"0/100 unsupported policy
commitments on the random slice; 95% upper bound 3.6% by the rule of three"*. An upper
bound is the honest form for a near-zero rate. The policy-trap stratum is reported
**separately** and will be far worse; **that gap is a finding, not an embarrassment**.

---

## 9. `cache.py` — the reproducibility guarantee

`cache_key(model_id, version, text) -> str` · `get(key)` · `put(key, response, usage)`
· `CacheMissWithoutKey`

Key = `sha256` over the three fields joined by NUL. Append-only JSONL, sorted before
commit so diffs are reviewable.

Fixes four problems at once: slow reruns, cost, non-determinism, and the reviewer
needing an API key. Bumping `prompt_template_version` invalidates the cache by design.

**Miss behaviour is loud.** No key + no entry ⇒ raise. Never caught internally. A silent
skip turns a broken reproduction into a passing one, and that is precisely the failure a
grader cannot detect from the outside.

---

## 10. `distill.py` — playbooks and style cards

`select_resolved_threads(threads, intent, n=50)` · `distill(threads, intent, config)` ·
`spot_check(path, threads)` · `build_style_card(replies)`

The grounding unit is the **hybrid**: distilled per-intent playbook **plus** three
nearest raw exemplars.

Why the playbook: it **separates policy from precedent**. A reviewer can open
`playbooks/billing_charges.md`, read it, and *disagree with it*. Three retrieved tweets
are not reviewable that way. It also structurally fixes the DM-deflection problem — the
playbook can state *"this brand resolves 78% of shipping-delay cases by moving to DM
after collecting the order number"*, converting DM handoff from a retrieval accident
into an explicit, defensible **action**. And it is cheap at inference: one text block
beats stuffing ten threads.

**The counter-argument, stated in the report because reviewers reward a stated
tradeoff:** distillation is a lossy, un-grounded step. An LLM summarising 50 threads can
assert *"refunds within 30 days"* when no thread said so. Mitigation: every playbook
line carries `evidence_ids`, and `spot_check` verifies a sample. Without provenance we
have moved the hallucination one layer upstream and hidden it. Raw exemplars are kept
**in addition to**, never instead of, the playbook.

`build_style_card` is pure statistics — no LLM. Mean reply length, signature rate, emoji
rate, question rate, URL rate.

---

## 11. `judge.py` — LLM-as-judge

`judge_one(message, draft, evidence, config)` · `judge_historical_replies(threads, config)` ·
`paraphrase_robustness(rubric, pairs, n=3)`

Six binary criteria, not a 1–5 Likert. **This is the biggest single reliability win
available.** A base judge *"never predicts Likert scores 5 and 6, and barely predicts 3
and 2"* — scores concentrate near the domain mean while humans use the full range.
Binary kills score compression and 4/5-clustering outright, and gives each criterion its
own κ.

Judge family ≠ generator family, enforced in config. Self-preference is linearly
correlated with self-recognition capability, with a causal link supported by
fine-tuning — this is the bias with the clearest causal evidence and the cheapest fix.

`judge_historical_replies` scores **the brand's own replies** under the same rubric.
This reframes the entire result: if the human replies pass 3.4 criteria on average and
the model passes 3.6, the headline means something completely different. The historical
reply is **context, never the target** — scoring similarity-to-history rewards mimicking
mediocrity.

`paraphrase_robustness` is a **reported result**, not an internal check. It answers
*"how fragile is my judge?"* with a number.

---

## 12. `evaluate.py` — the harness

Public: `wilson_interval`, `macro_f1`, `bootstrap_ci`, `mcnemar`, `paired_bootstrap`,
`holm_bonferroni`, `cohens_kappa`, `risk_coverage`, `calibration`, `main`.

Formulas, conventions and table layouts: **`docs/14`**.

Four rules are enforced *mechanically* here so they cannot be forgotten under deadline:

1. A headline computed off anything but `stratum == random` **raises**.
2. Every proportion ships with a Wilson 95% CI; every macro-F1 with a 10,000-replicate
   bootstrap CI.
3. Every system comparison is **paired** — McNemar with continuity correction for
   binary, paired bootstrap for F1. Overlapping CIs do **not** imply no significant
   difference.
4. Four baselines appear in every table: majority, random-by-prior,
   retrieval-nearest-neighbour reply, and the constant-"DM us" degenerate control.

---

## 13. `pipeline.py` — online orchestration

`process_one(message, config) -> DecisionRecord` · `run(messages, config)`

Stages A→I in the order given in `docs/10 §3`. `[F]` before `[G]`: the grounding result
is rung 6's input.

Bounded concurrency via `asyncio.Semaphore(config.runtime.concurrency)` — not a rate
limiter, we are not building one. Retry with exponential backoff **and jitter** on
429/5xx **only**; never retry a 400. Emits `run_report.json`.

---

## 14. `baselines/` — the ladder

Every baseline must be **genuinely tuned**. The brief asks for "a trivial one and a
simple one", and a rigged simple baseline is the easiest thing for a grader to spot.

| Tier | Module | Notes |
|---|---|---|
| 0 | `trivial.py` | majority; random-by-prior. Report accuracy **and** macro-F1 — a majority predictor gets macro-F1 near zero, and that contrast is the point. |
| 0 | `trivial.constant_dm_us` | The **degenerate control**. Always *"Sorry to hear that! Please DM us"*. Permanent fixture of every reply-quality table. |
| 0.5 | `rules.py` | 5–15 patterns from cluster centroids. Interpretable floor. **Never report a headline computed against keyword-derived labels** — an independent run saw coverage collapse 72% → 16% and CV macro-F1 fall 0.99 → 0.70 once real labels replaced them. |
| 1 | `tfidf.py` | word 1–2 gram **+ char_wb 3–5 gram**. **Character n-grams are the single most important detail for this data** — resilient to misspellings and abbreviations because they don't depend on whitespace tokenisation. Tune `C`, `min_df`, `sublinear_tf`, `class_weight='balanced'`. **This is the baseline the fancier models must beat.** |
| 2 | `embed_lr.py` | Frozen MiniLM + LR, and nearest-centroid. The honest "what do embeddings buy over bag-of-words" experiment, at the cost of one extra `.fit()`. |
| 3 | `setfit.py` | Right tool at 200 labels. **Trains in minutes on CPU — the DGX does not help; SetFit helps.** |
| 4 | `llm.py` | Frozen taxonomy as a constrained enum menu; every response cached. Partial-label-space retrieval set few-shot SOTA on three intent datasets with no fine-tuning. |

**Imbalance handling at n≈200** applies across tiers 1–3: stratified 5-fold with
**mean ± std reported, never a single 80/20 split** (the std is the honest signal at this
n, and an unstratified split can leave a class absent from test); class weights over
resampling (removing them cost 8.2 macro-F1 points in one low-resource ablation);
per-class thresholds tuned **inside** the CV loop, never on test.

**Banking77 is harness calibration only.** Run the *exact same ladder* at 8 or 16 shots
and check it reproduces the published ordering — if SetFit lands near 77.9%, the
implementation is sound and the Twitter numbers are trustworthy; if it lands at 45%,
there is a bug. **Never present a Banking77 number as evidence about the Twitter
system.** Domain mismatch is severe and unfixable; transfer learning from it is not
attempted.

**Augmentation rule, if used at all:** augmented data may enter *training only*; every
reported metric is computed on human-labelled test data only; the augmenting model must
not be the judge family; augmented and non-augmented numbers are reported side by side.
