# Implementation spec — the build contract

Docs `00`–`05` are research: what to build and why. This document and `11`–`16` are
the build: an engineer should be able to execute from the `1x` tier **without
reopening the research**. Where the two disagree, the `1x` tier wins and says why.

---

## 1. What is being built, in one paragraph

An offline pipeline that takes a first-turn inbound customer tweet addressed to
SpotifyCares, classifies it into a frozen intent taxonomy, retrieves the brand's own
historical resolution precedents, drafts a reply grounded in them, and returns one of
four actions — `auto_send`, `request_info`, `dm_handoff`, `escalate` — with a
machine-readable reason. Wrapped around it is an evaluation harness that measures the
whole thing against 200 hand-labelled examples, with confidence intervals on every
number, paired significance tests on every comparison, and a validated LLM judge whose
agreement with the human labeller is itself reported as a coefficient.

The second half is the deliverable. The brief says *"the proof is worth more than the
system"*, so ~60% of the effort goes to the measurement apparatus and the agent is
treated as the thing being measured.

---

## 2. Repo layout

```
README.md                    headline table + the 3-command reproduction
Makefile                     the target contract (§5)
config.yaml                  EVERY threshold. No policy constant lives in code.
requirements.txt             pinned, CPU-only
requirements-dgx.txt         experiment path only, never on the repro path
taxonomy/intents.yaml        frozen: name, definition, positive, not_this
data/
  sample_ids.txt             deterministic subsample, committed
  golden_200.csv             tweet_id + labels + stratum tags. THE real deliverable.
  raw/                       gitignored; fetch script + checksum
artifacts/
  embeddings_minilm.npy      committed, float16
  llm_cache.jsonl            committed, keyed by prompt hash
  playbooks/*.md             distilled, each line carrying evidence_ids
  results/
    predictions.jsonl        committed — Tier-1 eval reads this
    judge_verdicts.jsonl     committed — Tier-1 eval reads this
    decisions.jsonl          full audit trail
    run_report.json          tokens, cost, latency, cache hit rate
src/
  schemas.py                 the output contract (Pydantic)
  build_sample.py  induce_taxonomy.py  featurize.py
  guards.py  retrieve.py  generate.py  grounding.py  policy.py  cache.py
  distill.py  judge.py  evaluate.py  pipeline.py
  baselines/{trivial,rules,tfidf,embed_lr,setfit,llm}.py
tests/                       test_policy, test_grounding, test_schemas, test_eval, test_smoke
reports/                     confusion_matrix.png, error_analysis.md, taxonomy_notes.md
```

Per-module contracts: `docs/12`. On-disk schemas: `docs/11`.

---

## 3. Dataflow

**Offline (build-time, GPU-optional, output committed):**

```
twcs.csv
  └─ build_sample: thread assembly → brand filter via reply edge → sigil strip
     → URL tokenise → deterministic conversation-level sample   → threads.parquet
       ├─ induce_taxonomy: MiniLM → UMAP → k-means → LLM naming
       │                   → 2 merge/split rounds                → intents.yaml (frozen)
       ├─ distill: 40–60 resolved threads per intent, with evidence_ids
       │                                                          → playbooks/*.md
       ├─ distill.build_style_card: pure statistics, no LLM       → style_cards.json
       └─ featurize: MiniLM embed + BM25 index          → embeddings_minilm.npy
```

**Online (per message, CPU, no GPU anywhere):**

```
[A] guards      PII redact · injection scan · safety scan
[B] classifier  → (intent, confidence)
[C] retrieve    BM25 ‖ dense → RRF(k=60) → [rerank?] → MMR(λ=0.7)
                filtered {brand, intent} → 3 exemplars + playbook
[D] generate    schema-constrained → Decision
[E] validate    Pydantic + 1 bounded repair retry
[F] grounding   numbers · dates · URLs · commitments ⊆ evidence
[G] policy      7-rung ladder, short-circuit, record which layer fired
[H] route       action + escalation_code + route_to + priority
[I] record      one DecisionRecord line → decisions.jsonl
```

`[F]` runs **before** `[G]` because the grounding result is rung 6's input.

Escalations still carry the thread summary and the best-effort draft — the human
starts from something, per Hiver's Tier-3 *"groundwork already done."*

---

## 4. Build order

From `docs/02 §11`, unchanged, because the ordering argument still holds: everything
downstream is measured against the labels, and the output contract must exist before
anything can be tested against it.

1. Thread assembly + taxonomy + 200 hand labels — least glamorous, highest leverage.
2. Decision record + `config.yaml` — establishes the output contract.
3. Hybrid retrieval with brand/intent filtering — requirement (b).
4. Policy ladder with unit tests — requirement (c), and the part probed hardest live.
5. Schema-constrained generation + Pydantic + grounding validator.
6. Playbook distillation — high value, but the system works without it; exemplar-only
   retrieval is a valid fallback.
7. Eval harness + `run_report.json`.
8. Only then: reranking ablation, SetFit, LLM judge, DGX experiments — each
   documented, optional, and **measured**.

---

## 5. The two-tier reproduction contract

This resolves a live contradiction in the research. `docs/01 §7` says commit
`tweet_id` + labels and never tweet text (the TweetSumm precedent, which is the
defensible licensing position). `docs/03 §7` says a reviewer must reproduce the
headline results in under 15 minutes. Both cannot hold if reproducing the headline
requires a 493 MB Kaggle download behind a login.

**Resolution: split the claim from the regeneration.**

| | `make eval` — **Tier 1** | `make eval-full` — **Tier 2** |
|---|---|---|
| Reads | `golden_200.csv`, `predictions.jsonl`, `judge_verdicts.jsonl` | everything above, plus `data/raw/twcs.csv` |
| Needs Kaggle account | **no** | yes |
| Needs API key | **no** | no (committed cache), unless a cache miss |
| Needs network | **no** | no |
| Runtime | seconds | minutes |
| Produces | **every headline number, CI, and significance test in the report** | `predictions.jsonl`, then all of Tier 1 |

Tier 1 recomputes the statistics; it does not re-run the model. That is exactly the
claim being made — *"these labels and these predictions yield these numbers"* — and it
is checkable without any tweet text. Tier 2 proves the predictions came from the
pipeline rather than from thin air.

**What each tier may and may not claim.** Tier 1 verifies the *arithmetic and the
statistical discipline*. It cannot detect a prediction file that does not correspond to
the pipeline. Tier 2 closes that gap. The README states this distinction plainly rather
than implying Tier 1 is an end-to-end reproduction; overclaiming here would be the same
category of error the honesty section is about.

This split is itself a decision-log entry: *"the licensing constraint and the
15-minute constraint are in direct tension; I resolved it by making the statistical
claim reproducible without the data and the generative claim reproducible with it."*

`make smoke` is a third, smaller path: <30 s, 50 rows, CPU, no network, no key,
CI-able on every commit.

---

## 6. Four corrections to the research

The research was written before these were checked. The `1x` docs and the scaffold
follow the corrected versions.

### 6.1 `temperature=0` does not exist on the current models

`docs/02 §8` prescribes *"Determinism — `temperature=0`, fixed seed where supported"*.
Sampling parameters (`temperature`, `top_p`, `top_k`) are **removed on current Claude
models and return a 400**. There is no temperature knob to set to zero.

This does not weaken the reproducibility story — it *relocates* it, onto a mechanism the
research already mandates for three other reasons. The guarantee is the committed
response cache keyed by `sha256(model_id + prompt_template_version + input_text)`. On a
cache hit the output is byte-identical by construction. On a cache miss it is not
reproducible at all, and the README says so rather than implying otherwise.

Decision-log entry: *"the reproducibility guarantee is the cache, not a sampling
parameter, because the sampling parameter no longer exists."* Note this is a
strictly stronger guarantee than `temperature=0` ever was — greedy decoding was never
bit-reproducible across serving-stack changes either.

### 6.2 Structured output binding

`docs/02 §5` is right in principle (*prefer native JSON-schema / constrained decoding
over "respond in JSON"*) and unspecific about the call. The current binding:

- `output_config={"format": {...}}` on `messages.create()`, or `client.messages.parse()`
  for automatic schema validation.
- **Not** the deprecated `output_format` parameter.
- **Not** a "respond in JSON" instruction.
- Assistant prefill is removed (400), so no prefill-based JSON forcing either.

Plain JSON schema beats tool calling for a single-shot classify-and-draft; we do not add
an agent loop we do not need.

### 6.3 The headline CI is ±8pp, not ±5pp

`docs/04 §9` correctly places the headline on the frozen **n=100** random stratum:
Wilson at p̂ = 0.78, n = 100 gives `[0.689, 0.850]` — **±8 percentage points**.

But `docs/04 §8` item 8 says *"±~5pp on the headline at n=200."* The ±5pp figure is
correct **for n=200** (p̂ = 0.85 → `[0.794, 0.893]`) and wrong for a claim that lives on
n=100. The honesty checklist must quote ±8pp. Understating uncertainty is the one error
that section cannot afford, and a grader who recomputes the interval will check exactly
this.

`docs/14 §4` carries the corrected wording; `tests/test_eval.py::test_headline_slice_width`
pins it.

### 6.4 The minimum detectable paired difference is 7.0pp, not 6.5pp

`docs/04 §3` states the continuity-corrected McNemar formula, then reports *"you need
roughly |b − c| ≥ 13"* and a minimum detectable difference of *"≈ 6.5pp"*, with
b=26,c=14 → 3.60.

3.60 is the **uncorrected** statistic. Applying the formula as stated,
(|26−14| − 1)²/40 = **3.02**. Solving the corrected form against the 3.841 threshold
gives |b−c| > √(3.841·40) + 1 = 13.4, so the boundary is **14**, and the minimum
detectable paired difference is **14/200 = 7.0pp**.

Its own worked example (b=27, c=13, |b−c| = 14) contradicts the threshold of 13 it
states.

Both significance verdicts in `docs/04` survive — only the boundary moves. But the error
runs in the same direction as §6.3: it claims the harness can resolve a smaller
difference than it can. `docs/14 §5` carries the corrected values and
`tests/test_eval.py::TestMcNemar` pins both sides.

---

## 7. Model assignment

**Revised 2026-09-16: API path.** The DGX turned out unusable for this build (40 GB home
quota already full, all 8 V100s occupied), and the available keys are OpenAI and Gemini.
Every call goes through `src/llm_client.py`, so moving a role back to vLLM is a two-line
config change.

| Role | Model | Why |
|---|---|---|
| Generator, LLM baseline | `gpt-5-nano` (OpenAI), `reasoning_effort: minimal` | $0.05 / $0.40 per 1M tokens (list price, 2026-09-16). Reasoning tokens bill as output and share the output budget, so effort is pinned low for a 280-char draft. Structured output via strict JSON schema. |
| Judge | `gemini-2.5-flash` (Google), paid tier | **Different family from the generator, by design.** Self-preference is linearly correlated with self-recognition capability (arXiv:2404.13076) — the bias with the clearest causal evidence, and a provider choice removes it. Paid tier because free-tier content is used to improve Google's products. |
| Playbook distiller, taxonomy naming | `gpt-5-nano` (OpenAI), `reasoning_effort: low` | Low volume on the API path (one call per intent, a few per induction round). |
| Judge fallback | `Qwen/Qwen2.5-14B-Instruct` via local vLLM | Only if DGX storage frees up. Adds a third family and the privacy/cost framing below. |

The original assignment (Claude generator, Qwen judge and distiller on the DGX) is kept
below as the design intent; the reasoning about family separation carries over unchanged.

Escalation path if the fallback is used: report the judge-vs-human κ anyway, and add a
line to the honesty checklist stating the families are shared and the direction of the
resulting bias (inflation).

The local-model path is framed on three grounds, which is what a helpdesk company with
enterprise customers actually cares about: **privacy** (support transcripts never leave
the perimeter), **cost** (large-scale offline scoring is free), and **reproducibility**
(pinned open weights do not silently change under you the way a hosted model version
can). That converts "I had a DGX" into "I made a defensible infrastructure choice".

**The DGX is never on the reviewer's path.** A GPU-dependent repo fails the 15-minute
reproduction outright, and no model quality compensates for that.

---

## 8. Engineering that stays achievable

Build, because each is high-signal and low-cost:

- **Response cache** (`src/cache.py`, ~24 lines) — reruns free, demos instant.
- **Token + cost accounting** → `run_report.json`: tokens in/out, $/message, p50/p95
  latency, cache hit rate. Reviewers notice immediately.
- **Retry with exponential backoff + jitter on 429/5xx only.** Never retry a 400.
- **Bounded concurrency** — `asyncio.Semaphore(n)`, `n` in config. Not a rate limiter.
- **One decision record per message** — input, redacted input, intent + confidence,
  `evidence_ids` + scores, playbook used, raw model output, validation results, **which
  policy layer fired**, final action, latency, tokens, cost, cache hit. This single
  artifact is the strongest thing in the repo: it makes every decision auditable, and
  it is what you point at in a live review.

Do not build: LangChain / LlamaIndex, a vector DB, Docker Compose, a web UI,
multi-agent loops, a fine-tuned generator, unbounded self-repair loops. The premise of
the interview is *"we will ask you to explain and modify your own code live"* — every
abstraction you cannot defend line-by-line is a liability, and RRF, MMR, the policy
ladder and the grounding validator are ~200 lines you wrote yourself.

The answer if asked: *"at this scale the framework's abstractions cost more than they
save, and I wanted the decision logic readable."*

---

## 9. Non-goals, carried forward

From `docs/05 §6`, because the brief explicitly asks what you chose **not** to build:

- No multi-turn dialogue management. Scope is first-turn inbound messages.
- No fine-tuned generator. SetFit for the classifier only.
- No live Twitter integration, no web UI, no vector database, no agent framework.
- No multi-brand generalisation claim.
- No BLEU/ROUGE reply-similarity headline — those metrics correlate weakly-to-not-at-all
  with human judgement in exactly this domain, and the "DM us" degenerate scores well
  on them.
- No claim of resolution, CSAT, or deflection improvement. Offline eval cannot measure
  those; saying so is the honest position.
