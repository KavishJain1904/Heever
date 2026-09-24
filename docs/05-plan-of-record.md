# Plan of record

Synthesis of the four research passes into decisions. Where the research left a genuine choice
open, the call is made here and the reason recorded.

## 1. The eight decisions that shape everything

| # | Decision | Why |
|---|---|---|
| 1 | **Brand: SpotifyCares** (fallback: hulu_support) | Only candidate clearing all five bars. Crucially, **its historical answers are still true today** — "clear cache, reinstall, log out of all devices" is a fact about the world. Amazon's answers depend on a private order DB; the airlines' on ephemeral schedules; **AppleSupport deflects 52–64% of replies to DM**, making "ground the reply in history" vacuous. |
| 2 | **Treat this as an evaluation-engineering assignment** | The brief says the proof outweighs the system, and makes the honesty section mandatory. Budget ~60% of effort on measurement. |
| 3 | **Golden set = 100 random (frozen) + 60 stratified + 25 adversarial + 15 policy-trap** | The headline may only be quoted off the unbiased random slice. The other strata diagnose; they cannot make performance claims. |
| 4 | **Judge = binary 6-criterion checklist, different model family from the generator** | Kills Likert score-compression, gives each criterion its own κ, and removes self-preference bias — the bias with the clearest causal evidence. |
| 5 | **Grounding unit = distilled per-intent playbook + 3 retrieved exemplars, playbook lines carry `evidence_ids`** | Separates reviewable *policy* from raw *precedent*. Without provenance you have only moved the hallucination upstream and hidden it. |
| 6 | **Escalation = layered policy ladder; deterministic guardrails can veto automation but never authorize it** | Every layer independently unit-testable; every decision traceable to the rule that produced it. A single "should I escalate?" LLM call is undefendable in a live review. |
| 7 | **`DM_HANDOFF` is a first-class action, not a failure** | On a public channel, moving to DM is the correct path whenever the next step needs PII. Reframes the corpus's most common reply as a privacy-motivated policy decision. |
| 8 | **No LangChain/LlamaIndex; dual-path repo (CPU-cached default, DGX optional)** | "We will ask you to explain and modify your own code live." A GPU-dependent repo fails the 15-minute reproduction outright. |

## 2. On the DGX

**Use it for exactly two things, and neither is on the reviewer's path:**

1. **A second, independent open-weight judge** (different family from the generator). This is the
   highest-value use — it converts a methodological weakness into a strength, and scoring
   thousands of pairs is free locally where it costs real money via API.
2. **Offline playbook distillation at scale** — the one step with genuine volume. 8–14B instruct
   models via vLLM are the sweet spot; 70B is feasible and unnecessary.

**Do not use it for:** embedding (MiniLM is 80MB, minutes on CPU), retrieval (a NumPy matmul),
or fine-tuning the generator. **SetFit is the right tool at 200 labels, but it trains in minutes
on CPU — the DGX does not help; SetFit helps.**

Frame the local-model path on **privacy** (support transcripts never leave the perimeter),
**cost**, and **reproducibility** (pinned open weights don't silently change under you). That
converts "I had a DGX" into "I made a defensible infrastructure choice" — and it is exactly what
a helpdesk company with enterprise customers cares about.

## 3. Day plan

| Day | Work | Done when |
|---|---|---|
| **1** | Download; thread reconstruction with a **stated branch policy**; brand filter via the reply edge (not text matching); `^AB` sigil stripping; URL tokenising; **hand-read 100 random SpotifyCares threads** to verify the deflection numbers | A clean `spotify_threads.parquet` + a written note confirming or overturning the brand choice |
| **2** | Taxonomy induction (MiniLM → UMAP → k-means → LLM naming → 2 merge/split rounds); write annotation guidelines v1; **30-example pilot; revise to v2; discard pilot labels**; label all 200 | `taxonomy/intents.yaml` frozen + `golden_200.csv` with stratum tags |
| **3** | Baseline ladder (majority, random-by-prior, rules, TF-IDF+char n-grams, embed+LR); retrieval (BM25 + dense + RRF k=60 + MMR λ=0.7, filtered by brand+intent); policy ladder with one unit test per rule | `make eval` prints a baseline table with Wilson CIs |
| **4** | **Blind re-label 40 → κ_intra (the ceiling)**; judge rubric; judge-vs-human agreement per criterion; 3-paraphrase flip rate; grounding validator + commitment-slot extractor; risk–coverage curve | Judge validation table + the ceiling number |
| **5** | Report, decision log, failure analysis (5 modes, verbatim examples, hypotheses), cache commit, **time the reproduction with a stopwatch** | `git clone && make setup && make eval` under 15 min on a clean machine |

**Day 2 is the long pole and the one that slips. Protect it.**

## 4. Failure-mode candidates to look for on day 4

Predictions to confirm or refute against real outputs — the report needs 5 with verbatim examples:

1. **Mid-thread fragments misclassified.** 37.5% of inbound tweets are follow-ups like *"iPhone 8,
   iOS 11"* — context-free, and the classifier will guess.
2. **`praise_thanks` over-prediction.** Observed in an independent real-data run; it is a large,
   lexically obvious class that attracts probability mass.
3. **Agent-signature hallucination** — the generator emitting invented `^AB` initials. Observed in
   the wild. Strip sigils at ingest or this is guaranteed.
4. **Policy invention on the policy-trap stratum** — refund windows and timelines no thread
   supports.
5. **Multi-intent collapse** — "charged me twice, cancel my account" forced into one label by the
   precedence rule, then judged wrong for addressing only one.

## 5. Decision-log seeds (the brief wants 10–15 non-obvious ones)

Beyond the eight above:

9. Sample at the **conversation level** (hash the root id), never the tweet level — tweet-level
   sampling leaks turns of the same thread across train/eval.
10. Deterministic **hash-based** subsampling rather than `df.sample(seed=)` — stable across chunk
    boundaries, library versions and machines.
11. Recover brand for inbound tweets **via the reply edge**, because customers address numeric
    anonymised handles while replies use the brand name — naive filtering returns nothing.
12. Commit **`tweet_id` + labels**, never tweet text, with a rejoin script — following the
    **TweetSumm** precedent, which is the defensible answer on redistribution.
13. **Single-label with an explicit precedence rule**, plus a *measurement* of how often the
    second annotation pass disagrees — the ambiguity rate is reported as a ceiling, not hidden.
14. Keep **`NON_SUPPORT` separate from `OTHER`** — one is a filter decision, the other an
    escalation decision.
15. **Reranking behind a config flag, off by default, with the ablation reported.** Off-the-shelf
    rerankers have been measured *degrading* nDCG on out-of-distribution corpora.
16. **Report macro-F1 twice** — all classes, and in-scope only — because `OTHER` is a
    heterogeneous grab-bag that drags the macro down while telling you nothing.
17. State **which macro-F1 formula** is used; the two definitions can differ by up to 0.5 and
    reorder classifiers.
18. **Fail loudly on an LLM cache miss with no API key**, rather than silently skipping — a
    silent skip turns a broken reproduction into a passing one.

## 6. Non-goals — the brief explicitly asks what you chose not to build

- **No multi-turn dialogue management.** Scope is first-turn inbound messages. 62.5% of inbound
  tweets are thread starters; the rest need conversational state that a single-turn evaluation
  cannot honestly measure.
- **No fine-tuned generator.** SetFit for the classifier only.
- **No live Twitter integration, no web UI, no vector database, no agent framework.**
- **No multi-brand generalisation claim.** One brand, stated as a limitation.
- **No BLEU/ROUGE reply-similarity headline** — the 2016 dialogue-evaluation literature shows
  these correlate weakly-to-not-at-all with human judgement on exactly this domain, and the
  "DM us" degenerate would score well on them.
- **No claim of resolution, CSAT, or deflection improvement.** Offline eval cannot measure those;
  saying so is the honest position.

## 7. Open items to resolve before writing the report

- **Read the Kaggle license string yourself.** It could not be retrieved during research. Do not
  assert a license you have not read.
- **Hand-verify the deflection numbers** for SpotifyCares and hulu_support on 100 random threads
  each. Every published figure is a regex heuristic; nobody has human-verified resolution rates.
  One afternoon of reading beats every number in these documents.
- **Verify Hiver's current AI branding** (Harvey vs. "Hiver AI" / AI Agents) and the customer
  count on the day you submit. Naming a deprecated feature reads as stale research.
- **Verify each incident citation** against primary sources before it enters the report —
  Air Canada, DPD, Chevrolet, Klarna.
