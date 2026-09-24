# Heever: AI support agent for Twitter customer support

Build for the Hiver SDE Intern take-home: classify incoming customer messages into a
data-derived intent taxonomy, draft a reply grounded in how the brand historically resolved
similar issues, and decide auto-handle vs. escalate with a stated reason.

**The report is [`REPORT.md`](REPORT.md).** It carries the headline, the baseline comparison,
the five failure modes, the decision log, and the list of what is misleading about its own
numbers. Read the blockquote at the top of it first: the golden set was not hand-labelled, and
that changes how every number should be read.

**Status:** implemented against the specs, and now exercised end to end on real data.
`docs/0x` is the research, `docs/1x` is the build contract, and `src/` now implements it.
96 unit tests pass and `make smoke` reproduces every worked example pinned in `docs/14`
(Wilson intervals, the McNemar boundary, the kappa paradox, Elkan's threshold).

**What is NOT yet done**, stated plainly because the gap matters more than the code:

| | |
|---|---|
| Corpus | `data/raw/twcs.csv` downloaded locally; sha256 pinned in `config.yaml`, 2,811,774 records verified; license read (CC BY-NC-SA 4.0, `docs/01 §7`). Threads rebuilt locally into `data/threads.parquet`, which stays gitignored because it embeds tweet text; `make rejoin` regenerates it |
| Models | OpenAI generator (gpt-5-nano) and OpenAI judge (gpt-5.4-mini) through `src/llm_client.py`. Exercised live on 2026-09-16. Judge shares a vendor with the generator; see REPORT.md limitation 4 |
| Taxonomy | `taxonomy/intents.yaml` frozen 2026-09-16: 11 SpotifyCares intents drawn from 160 real tweets. Clustering induction was not run |
| Golden set | **Not hand-labelled.** 200 rows pre-labelled by gpt-5.4 and used as the reference as-is; the deadline ran out before the human review pass. `data/golden_200.csv` stamps `reviewed: false` and `label_source: gpt-5.4-prelabel` on every row. The labelling tool is built and working (`label_golden_200.html`), it was just never used. REPORT.md limitation 1 states what this costs |
| Pipeline | Run repeatedly against the full 200-message set: predict has run four times (`predict_run2.log` to `predict_run4.log` plus the run behind the committed `predictions.jsonl`, all in `artifacts/results/`), judge has run twice (`judge_run.log`, `judge_run2.log`). The first pass surfaced and fixed a unit bug in the no-precedent gate (REPORT.md decision 10) |
| Playbooks | `playbooks/*.md` not distilled. The pipeline does not depend on them: see `docs/11 §6` |
| LLM cache | `artifacts/llm_cache.jsonl` holds 1,832 entries (2.36 MB); `artifacts/results/run_report.json` reports `cache_hit_rate: 1.0` for the latest predict run |
| Judge validation | **Not measured.** The brief asks for evidence of judge-human agreement. `make judge-tool` builds the blind scoring page and `judge-import` computes the kappa, but no human has scored the 30 drafts, so every judge number is unvalidated |
| Every number | Computed and in REPORT.md, with no `[[TBD]]` left. But they measure agreement between gpt-5-nano and gpt-5.4, not accuracy: see the golden-set row above |

The deterministic core (policy ladder, grounding validator, eval harness, schemas,
guards, cache) is fully exercised by tests and runs today, and the data-dependent and
LLM-dependent paths have now run against real data. Every number in REPORT.md is
computed. The thing to hold onto while reading them is that the reference labels they
are scored against are a model's, not a person's.

[`docs/17-implementation-findings.md`](docs/17-implementation-findings.md) has the
module-by-module verification status, plus three contract defects the build surfaced
in `docs/11 §5`, `docs/14 §4` and the `non_support` action mapping.

## Documents

| Doc | Contents |
|---|---|
| [`ASSIGNMENT.md`](docs/ASSIGNMENT.md) | **The brief, verbatim.** Source of truth for what was asked. Read this first. |
| [`00-assignment-deconstruction.md`](docs/00-assignment-deconstruction.md) | What the brief actually grades. Inferred rubric, deliverables checklist, six traps. |
| [`01-dataset-and-brand-selection.md`](docs/01-dataset-and-brand-selection.md) | Corpus audit (2.8M rows, 108 brands), schema traps, measured DM-deflection by brand, licensing, download mechanics, prior work. |
| [`02-system-design-and-hiver-framing.md`](docs/02-system-design-and-hiver-framing.md) | Hiver's product vocabulary and three-tier framework, grounding-unit choice, retrieval stack, the escalation policy ladder, safety/injection, the DGX question. |
| [`03-taxonomy-and-baselines.md`](docs/03-taxonomy-and-baselines.md) | Taxonomy induction recipe, taxonomy design rules, the five-rung baseline ladder, classifier metrics, reproducibility layout. |
| [`04-evaluation-design.md`](docs/04-evaluation-design.md) | **The core deliverable.** Golden-set strata, solo-annotator label quality, statistical power, agreement metrics, LLM-judge biases and mitigations, groundedness, selective prediction, and the mandatory honesty checklist. |
| [`05-plan-of-record.md`](docs/05-plan-of-record.md) | The decisions, the day plan, decision-log seeds, and the explicit non-goals. |

### Implementation tier

An engineer should be able to execute from these without reopening the research.

| Doc | Contents |
|---|---|
| [`10-implementation-spec.md`](docs/10-implementation-spec.md) | The build contract. Repo layout, dataflow, build order, the two-tier reproduction contract, model assignment, and **three corrections to the research**. |
| [`11-data-contracts.md`](docs/11-data-contracts.md) | Every on-disk artifact, field by field, with invariants. |
| [`12-module-contracts.md`](docs/12-module-contracts.md) | Per-module spec for every file in `src/`: signature, behaviour, edge cases, and the decision each implements. |
| [`13-annotation-guidelines-v1.md`](docs/13-annotation-guidelines-v1.md) | **A shipped deliverable.** The labelling manual: decision tree, precedence rule, pilot protocol, and the κ_intra protocol. |
| [`14-eval-harness-spec.md`](docs/14-eval-harness-spec.md) | Metric formulas bound to code paths, declared conventions, the judge rubric, and every output table's columns. |
| [`15-test-plan.md`](docs/15-test-plan.md) | 51 cases: one per policy rung, real grounding-failure fixtures, hand-computed statistical values, and the offline-reproduction guarantee. |
| [`16-work-breakdown.md`](docs/16-work-breakdown.md) | Ordered tasks with done-when assertions, the four blocking verification items, and the slip protocol. |
| [`17-implementation-findings.md`](docs/17-implementation-findings.md) | **Written during the build.** Three contract defects the specs did not anticipate, and, more importantly, exactly which modules are verified and which have never seen real data. |

## Reproduction

```
git clone https://github.com/SirjanSingh/Heever.git && cd Heever
make setup
make eval          # Tier 1: every headline number, CI and significance test,
                   # from committed predictions + labels + judge verdicts.
                   # No Kaggle account. No API key. No network. Seconds.
make figures       # confusion matrix, risk-coverage curve, reliability diagram
```

Verified on 2026-09-16 from a clean clone with no API key and no `twcs.csv`:
`make eval` exits 0 in under 20 seconds and reproduces the headline exactly.

`make eval-full` regenerates the predictions from source and needs your own
`twcs.csv`. We commit `tweet_id` plus labels and never tweet text, following the
TweetSumm precedent. `make smoke` is a 30-second CI-able subset. What each tier may and
may not claim is spelled out in [`docs/10 §5`](docs/10-implementation-spec.md).

## The three claims this build rests on

1. **The brand choice is a measurement, not a preference.** AppleSupport , the popular pick,
   deflects 52 to 64% of its public replies to DM, which makes "ground the reply in how the brand
   historically resolved this" nearly vacuous. SpotifyCares resolves in-thread, is >99.7%
   English, and its historical answers are still true today.
2. **The headline number may only be quoted off a frozen, uniformly-sampled slice.** Everything
   else diagnoses; it does not make performance claims.
3. **The honesty section is the deliverable.** On an assignment where "the proof is worth more
   than the system", the list of what is misleading about the headline *is* the proof.
