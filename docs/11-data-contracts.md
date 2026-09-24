# Data contracts

Every artifact that crosses a module boundary or lands on disk. Field, type,
nullability, invariant. `src/schemas.py` is the executable form of §4–§7; this
document is the reference and covers the file-level artifacts the Pydantic models
do not.

**Rule:** if a field is not in this document, it is not in the contract. A module may
not read a field it did not declare here.

---

## 1. `data/threads.parquet` — reconstructed conversations

Produced by `src.build_sample.build_threads`. Gitignored (derived from raw data).

| Column | Type | Null | Notes |
|---|---|---|---|
| `tweet_id` | `Int64` | no | Kaggle's *anonymised* surrogate key. **Not a Twitter snowflake ID.** Range 1–2,987,950 over 2,811,774 rows — ~176k IDs referenced or reserved but absent. Never assume `id+1`. |
| `conversation_id` | `Int64` | no | `:= root tweet_id`. Deterministic and reproducible. |
| `author_id` | `str` | no | Anonymised. Brand handles appear as names (`SpotifyCares`); customers as numerics (`115821`). |
| `inbound` | `bool` | no | `True` = customer→brand. |
| `created_at` | `datetime[tz]` | no | Parsed with `format="%a %b %d %H:%M:%S %z %Y"`. **Not ISO** — a default parse silently fails or misreads. |
| `text` | `str` | no | As received. Emails/phones masked upstream to `__email__`/`__phone__`; **URLs are not masked**. |
| `text_normalised` | `str` | no | Sigils stripped, URLs → `[URL]`, whitespace collapsed. This is what gets embedded and indexed. |
| `in_response_to_tweet_id` | `Int64` | **yes** | **Read as string, cast to nullable `Int64`.** A naive `read_csv` infers `float64` because NaN is present, and silently corrupts every ID. |
| `turn_index` | `Int32` | no | 0-based position after branch resolution and split-reply reordering. |
| `is_root` | `bool` | no | Parent is null **or** parent id is absent from the file (dangling reference — 654 orphan roots for AmazonHelp alone). |
| `brand` | `str` | no | **Recovered via the reply edge, never by text matching.** Customers address numeric handles while replies say the brand name; filtering on `author_id == 'SpotifyCares'` misses every inbound tweet. |
| `split_part` / `split_total` | `Int32` | yes | Parsed from an `N/M` marker. Brand replies split across tweets share an *identical* timestamp and are stored out of order (3/3 first); sorting by `created_at` alone scrambles them. |

**Invariants** (assert at write time):
- `conversation_id` is present in the set of `tweet_id` values, or is flagged `orphan_root`.
- Exactly one row per conversation has `is_root == True`.
- `turn_index` is contiguous from 0 within each conversation.
- `brand != ""` — the ~20,794 conversations under an empty brand string are excluded.
- No duplicate `tweet_id` (the source has zero; a duplicate means a join bug).

**Branch policy** (stated, because flattening a tree without one is a silent bug —
222,426 rows have multiple children): prefer the branch ending with a brand reply, then
the longest branch, then the lowest terminal `tweet_id`.

---

## 2. `data/sample_ids.txt` — the frozen subsample

One `conversation_id` per line, sorted ascending, LF-terminated. Committed.

Produced by hashing the **root id** (`md5(str(root_id))[:8] % 100 < 5`). Conversation-level,
never tweet-level: tweet-level sampling leaks turns of the same thread across train and
eval. Deliberately not `df.sample(seed=)`, which is not stable across chunk boundaries,
library versions or machines.

First line is a `#` comment carrying the generating parameters and the `twcs.csv`
sha256, so a mismatch is detectable rather than silent.

---

## 3. `data/golden_200.csv` — the golden set

**The real deliverable.** Committed. Contains `tweet_id` + labels, **never tweet text**
— following the TweetSumm precedent (Findings of EMNLP 2021), which ships IDs +
annotations and a processor requiring the user's own `twcs.csv`.

| Column | Type | Null | Notes |
|---|---|---|---|
| `id` | `str` | no | Stable row key, `g0001`–`g0200`. |
| `tweet_id` | `Int64` | no | Rejoin key. |
| `conversation_id` | `Int64` | no | |
| `stratum` | enum | no | `random` \| `stratified` \| `adversarial` \| `policy_trap`. **Tagged at insertion time, never inferred later.** |
| `intent` | enum | no | From the frozen `intents.yaml`. |
| `auto_handle_vs_escalate` | enum | no | `auto_handle` \| `escalate`. The human's decision, independent of the model. |
| `quality_of_historical_reply` | `Int8` | yes | 1–5. Null where the thread has no brand reply. Feeds the reference band — if the brand's own replies score 3.4 and the model scores 3.6, that reframes the whole result. |
| `confidence` | enum | no | `high` \| `med` \| `low`. The annotator's own confidence. |
| `ambiguous` | `bool` | no | True where >1 intent genuinely applies. |
| `note` | `str` | yes | Free text. Adjudication reasoning for hard cases. |
| `guideline_version` | `str` | no | e.g. `v2`. Which version of `docs/13` produced this label. |
| `label_date` | `date` | no | Enables the ≥48-hour blind re-label protocol. |

**Stratum counts are fixed:** random 100, stratified 60, adversarial 25, policy_trap 15.

**Content hash.** The `random` stratum is hash-pinned in `artifacts/results/golden_hash.txt`
(sha256 over sorted `tweet_id`). A changed hash means the frozen slice moved, which
invalidates every number computed against it. `tests/test_smoke.py::test_golden_set_content_hash_is_stable`
enforces this.

**Only `stratum == random` is headline-eligible.** Enforced mechanically in
`src.evaluate`, not by discipline.

### 3.1 `data/golden_relabel_40.csv`

Same schema plus `original_label` and `relabel_date`. 40 rows re-labelled blind, in
shuffled order, ≥48 hours later. Produces κ_intra — the measurement ceiling. If
κ_intra = 0.65, no downstream judge can credibly exceed that, and saying so is one of
the strongest moves available in the report.

---

## 4. `artifacts/results/predictions.jsonl`

One line per (system, golden example). Committed — this is what Tier-1 `make eval` reads.

```json
{"system": "tfidf", "id": "g0001", "stratum": "random",
 "predicted_intent": "service_outage_technical", "intent_confidence": 0.81,
 "predicted_action": "auto_send", "reply_draft": "...", "evidence_ids": ["t_88421"],
 "policy_layer_fired": "default", "grounding_passed": true,
 "latency_ms": 412.0, "cache_hit": true}
```

`system` ∈ {`majority`, `random_prior`, `rules`, `tfidf`, `embed_lr`, `setfit`,
`llm_fewshot`, `agent`, `dm_us_degenerate`, `retrieval_nn`}.

**Invariant:** every `system` has exactly 200 rows, and the `id` sets are identical
across systems. Comparisons are paired; a ragged file silently breaks McNemar.

---

## 5. `artifacts/results/decisions.jsonl` — the audit trail

One line per processed message. `src.schemas.DecisionRecord`. The strongest single
artifact in the repo: it makes every decision auditable and it is what you point at in a
live review.

Fields: `message_id`, `conversation_id`, `stratum`, `input_text_redacted`,
`predicted_intent`, `intent_confidence`, `evidence[]` (each with `evidence_id`,
`rrf_score`, `bm25_rank`, `dense_rank`, `selected_by_mmr`), `playbook_id`,
`raw_model_output`, `validation_errors[]`, `repair_attempts`, `commitments{}`,
`grounding_passed`, `policy_layer_fired`, `decision{}`, `latency_ms`, `tokens_in`,
`tokens_out`, `cost_usd`, `cache_hit`.

**Invariants:**
- `policy_layer_fired` is never null, on any path including `default`.
- `repair_attempts <= config.generation.max_repair_retries`.
- `decision.escalation_reason` is present iff `decision.action == "escalate"`.
- `len(decision.evidence_ids) >= 1` always.
- `input_text_redacted` never contains a PII pattern match (checked on egress too).

---

## 6. `playbooks/<intent>.md` — distilled policy

Markdown with YAML front-matter. Committed. Lives at repo root (`playbooks/`, not under
`artifacts/`), matching `src/pipeline.py`. **Every claim carries provenance** — without
`evidence_ids` we have only moved the hallucination one layer upstream and hidden it.

Playbooks are not load-bearing: `_playbook_for()` in `src/pipeline.py:41` returns an empty
string when the file for an intent is absent, and `src/generate.py:131` substitutes
"(no playbook for this intent)" in the prompt, so the pipeline runs correctly without them.

```yaml
---
intent: service_outage_technical
brand: SpotifyCares
distilled_from_n_threads: 50
distiller_model: Qwen/Qwen2.5-14B-Instruct
distilled_at: 2026-09-15
spot_check_sample: 10
spot_check_supported: 9
---
```

Body sections, each line ending with `[evidence: t_88421, t_10233]`:

1. **Canonical resolution steps** the brand actually took.
2. **Information the brand always asks for.**
3. **Standard move** — public reply or DM handoff, with the observed rate.
4. **Tone and signature conventions.**
5. **Things this brand never promises.** ← the section that prevents invented policy.

A reviewer can open this file, read it, and *disagree with it*. Three raw retrieved
tweets are not reviewable that way. That is the entire argument for the playbook over
exemplars alone — and the exemplars are kept **in addition to**, never instead of it.

---

## 7. `artifacts/results/judge_verdicts.jsonl`

Committed. One line per (system, example, ordering).

```json
{"system": "agent", "id": "g0001", "ordering": "forward", "judge_model": "Qwen/Qwen2.5-14B-Instruct",
 "rubric_version": "v1", "criteria": {
   "addresses_stated_problem": {"pass": true, "quote": "your app crashes on launch"},
   "grounded_in_retrieved_context": {"pass": true, "quote": "try reinstalling"},
   "no_unsupported_policy_commitment": {"pass": true, "quote": ""},
   "correct_brand_voice": {"pass": true, "quote": ""},
   "correct_escalate_or_autohandle_decision": {"pass": true, "quote": ""},
   "no_pii_or_unsafe_content": {"pass": true, "quote": ""}},
 "all_six_pass": true, "cot": "...", "paraphrase_variant": 0}
```

**Invariants:**
- Every example appears under **both** orderings (`forward`, `reverse`). Order is
  randomised and both are scored; measured judge order-consistency in the wild runs
  ~70–77%, so this is not cosmetic.
- Every `pass: true` on a content criterion carries a non-empty verbatim `quote`.
  Forcing a quote is what makes judge errors auditable.
- `paraphrase_variant ∈ {0,1,2}` — variant 0 is the canonical rubric; 1 and 2 feed the
  flip rate, which is a **reported result**, not an internal check.
- The historical brand replies are judged under the same schema with
  `system: "historical"`, producing the reference band.

---

## 8. `artifacts/llm_cache.jsonl`

Committed. Append-only; sorted by key before commit so diffs are reviewable.

```json
{"key": "<sha256 hex>", "model_id": "claude-haiku-4-5",
 "prompt_template_version": "v1", "response": "...",
 "usage": {"input_tokens": 462, "output_tokens": 19}, "created_at": "..."}
```

Key = `sha256(model_id + "\0" + prompt_template_version + "\0" + input_text)`.

**Miss behaviour is explicit and loud:** if the key is unset and the entry is missing,
raise `CacheMissWithoutKey`. A silent skip turns a broken reproduction into a passing one.
Bumping `prompt_template_version` invalidates the whole cache by design — that is what
it is for.

---

## 9. `artifacts/results/run_report.json`

```json
{"run_id": "...", "config_sha256": "...", "n_messages": 200,
 "tokens_in": 92400, "tokens_out": 3800, "cost_usd": 0.112,
 "cost_per_message_usd": 0.00056,
 "latency_p50_ms": 380, "latency_p95_ms": 910,
 "cache_hit_rate": 1.0, "repair_rate": 0.015,
 "action_distribution": {"auto_send": 0.41, "dm_handoff": 0.21,
                         "request_info": 0.09, "escalate": 0.29},
 "policy_layer_distribution": {"safety_veto": 0.005, "...": 0.0}}
```

`config_sha256` pins which thresholds produced the run. A results table without it
cannot be attributed to a configuration.

---

## 10. `artifacts/embeddings_minilm.npy` and `style_cards.json`

Embeddings: `float16`, shape `(N, 384)`, row order matching `text_ids.txt` written
alongside. float16 keeps the committed artifact to a few MB; it is cast to float32 for
the matmul.

Style cards: per brand, pure statistics with no LLM involved —
`{"mean_reply_chars": 118, "signature_rate": 0.83, "emoji_rate": 0.21,
"question_rate": 0.66, "url_rate": 0.14}`. Injected as a compact style card, which is
cheaper and more reliable than hoping the model infers voice from three exemplars.
