# 2026-09-16: first real run, judge fixes, report draft

Applications close around 2026-09-17, so this session optimised for getting real numbers out of
the pipeline, not for completing the full plan in docs/16.

## What changed and why

- **Taxonomy frozen (v2, 13 intents).** The draft was a cross-industry prior (orders, returns,
  deliveries) that does not fit a streaming service. Redrawn from 160 random SpotifyCares
  opening tweets. New: `dm_followup`, `library_lost`, `catalog_content`, `feature_feedback`.
  Clustering induction was skipped for time.
- **Golden-set pre-labelling.** `scripts/prelabel.py` (gpt-5.4, a different tier from the
  gpt-5-nano classifier, to limit anchoring leakage). Labelling moved from the HTML tool to
  `data/golden_200_labels.xlsx` (`scripts/build_label_sheet.py`) at Sirjan's request.
  Confidence is never pre-filled.
- **`scripts/run_experiment.py`**: predict, golden import, baselines (train on non-random
  strata, test on random 100), tune (tau_intent on non-random only), judge, judge-sheet,
  judge-import, errors. Precedent index holds out all golden and pilot conversations.
- **Bug: no-precedent gate unit mismatch.** Compared RRF score (max ~0.033) with tau 0.35,
  escalating 191/200. Now gates on top exemplar cosine (`Exemplar.dense_score`).
- **Bug: default-rung escalation had no reason**, crashing one message (g022). Fixed in
  `pipeline.py`.
- **Generator normalisation.** 47/73 validation failures were a stray `escalation_reason` on
  non-escalate actions. Dropped in code and logged; repair rate 29% -> 6%.
- **Bug: judge ran without evidence.** Grounding failed 92/93. Judge now sees retrieved
  precedents, the brand style card and the proposed handling. Broken run kept locally as
  `judge_verdicts_run1_no_evidence.jsonl`.
- **Judge moved to OpenAI (gpt-5.4-mini)**; only an OpenAI key was available. Limitation stated.
- **`draft_on_escalate: true`** (eval only) so reply quality is judged on a uniform sample.
- **REPORT.md** drafted: framing, label-free reply-quality table, 3 failure modes, 13
  misleading-number items, 15-entry decision log.

## Results so far (no labels needed)

Random 100, per criterion, Heever / historical / constant "DM us": addresses problem 75/28/17,
grounded 21/21/90, no unsupported commitment 75/91/100, brand voice 37/64/67, all six 14/8/12.
The composite cannot separate real drafts from a constant string. Drafts average 204 chars vs
the brand's 130; sign-off 2% vs 73%. Every message is still escalated because gpt-5-nano's
self-reported confidence sits at 0.5 to 0.65, under tau_intent 0.75.

## Verification

- `pytest`: 91 passed.
- Labelling HTML tool exercised in the browser pane over http (keys, reviewed flag, export).
- Pipeline run end to end on 200 messages four times; judge twice.

## Not committed (on purpose)

Tweet-text-bearing files (threads.parquet, candidates/prelabels JSON, xlsx sheets, decisions,
predictions, judge verdicts, LLM cache) and run logs. See `.gitignore`. Decide before submission
whether predictions/verdicts get committed in text-free form for `make eval` Tier 1.

## Open for next session

1. Sirjan hand-labels `data/golden_200_labels.xlsx` and `data/judge_human_30.xlsx`.
2. `run_experiment.py golden data/golden_200_labels.xlsx`, then `tune`, `baselines`, `errors`,
   `judge-import`.
3. Fill every `[[TBD]]` in REPORT.md; failure modes 4 and 5 from labelled errors.
4. Make `make eval` Tier 1 work from committed, text-free artifacts; time the reproduction.
5. C: has ~499 MB free.

---

# 2026-09-16, session 2: close-out

Picked up the five items above. Items 2 to 5 are done. **Item 1 was not done and will not be**,
which is the defining decision of this session.

## The labelling decision

Sirjan had no time left and asked me to label the golden set myself. I declined that specific
framing: REPORT.md claimed in three places that he had reviewed every row, and filling the
`confidence` field on his behalf would have made those claims false to a hiring reviewer about
his own work. Offered three honest alternatives (audit-sample 40 rows, all-LLM stated plainly,
or a smaller genuinely-hand-labelled set). **He chose all-LLM, stated plainly.**

So `cmd_golden_from_prelabels` loads gpt-5.4's pre-labels as the reference and stamps
`reviewed: false`, `confidence: model` and `guideline_version: v2-prelabelled-NO-HUMAN-REVIEW`
on all 200 rows. The artifact declares its own provenance rather than relying on the prose. The
report leads with a blockquote saying no human reviewed any label, and limitation 1 states that
every number measures gpt-5-nano reproducing gpt-5.4, not accuracy.

Worth noting for the future: the `confidence` gate designed in session 1 did exactly its job.
It refused all 200 rows rather than silently passing model labels off as reviewed, which forced
the degraded path to be named explicitly. That gate is the reason this failure is legible.

## What changed and why

- **Tier 1 reproduction fixed** (item 4, and a brief requirement that was failing). A fresh
  clone could not run `make eval` because it read three gitignored or absent files. Verified
  `predictions.jsonl` carries no corpus text (its long fields are the model's own drafts) and
  un-ignored it. The judge verdicts do leak text in two places, not one: the `historical` arm's
  `draft` is a real brand tweet, and `criteria[*].reasoning`/`quote` cite the *customer* message
  on all three arms. New `scripts/redact_artifacts.py` strips them to the pass/fail data
  `evaluate.py` actually reads, and fails loudly on an unknown key so a future judge field
  cannot leak silently. `config.yaml` now splits `verdicts_path` (redacted, committed) from
  `verdicts_path_full` (local, for producers).
- **`judge_summary.json` had no producer.** Its numbers were already quoted in REPORT.md §3 and
  no tracked code wrote the file; `_reply_quality_table` computed Wilson intervals but no paired
  McNemar, so the p-values traced to nothing. Extended it to pair by `message_id` and compute
  McNemar plus order-flip, and `make eval` now writes the file. The recomputed values match the
  committed ones exactly, so the published numbers were right, they just had no provenance.
- **Bug: the tuner counted drafts the pipeline would veto.** `simulate` re-scored rows at a
  lower tau without re-applying the grounding validator, which sits at rung 6, below the
  confidence gate at rung 4. A row that died at rung 4 never reached grounding, so re-simulating
  it credited 4 phantom auto-sends at tau 0.62 and 7 at tau 0.55. Would have put an inflated
  automation rate in the report. Folded into decision-log item 9 rather than adding a 16th entry,
  since the brief caps that log at 15.
- **`make figures` was a stub** that printed a message and produced nothing while the Makefile
  advertised three plots. Now renders confusion matrix, risk-coverage and reliability diagram.
- **Blind judge scorer built** (`scripts/build_judge_tool.py`, `make judge-tool`). Verified the
  page embeds only `message_id`, `customer_text` and `draft`: no judge verdict, no system label,
  so a scorer cannot be anchored. `judge-import` now accepts its CSV as well as the xlsx.
- **`openpyxl` was missing from requirements**, so `make setup && make golden` failed on a clean
  environment.
- **Documented a representational defect:** three `non_support` rows are counted as escalations.
  It is a filter class dropped before the ladder, but `Action` is a closed four-value enum with
  no `FILTER` member by design, so the drop is recorded as `escalate`. The
  `evidence_ids: ["filtered"]` marker survives in `decisions.jsonl` but is stripped from
  `predictions.jsonl`, which is what the evaluation reads. Now limitation 14.

## Results (items 2 and 3)

Random 100, against gpt-5.4 reference labels. Intent agreement **0.56** (0.46 to 0.65) vs 0.39
tuned TF-IDF SVM (p=0.02), 0.35 MiniLM-NN (p=0.003), 0.25 keyword rules, 0.17 majority.
Macro-F1 0.44. Out-of-scope detection is **0.00** on precision, recall and F1 across all 5
`other_unclear` rows: the classifier never once abstained. Action head 32%, identical to the
always-escalate baseline, because in this configuration it is that baseline.

**The threshold sweep is the finding worth keeping.** Tuning on the non-random strata picks tau
0.30 at an apparently safe 2.4% bad-auto-send share. On the held-out random 100 that same
threshold gives 10%, over double the 4.76% Elkan ceiling. Reading down the held-out column,
every threshold that automates even one message breaches the ceiling, and every threshold that
respects it automates nothing. The tuning slice was fooled because its safe readings rested on a
single bad auto-send out of 42. So the 0% automation rate the system shipped with was correct by
accident: the gate was unreachable, not calibrated. Failure mode 4 has the arithmetic, tau 0.75
sits above the 0.74 ceiling across every auto-sendable intent, while Elkan wants 0.9524.

## Verification

- `pytest`: 96 passed (was 91; the judge-summary work added 5).
- `make smoke` passes, network-blocked.
- **Fresh-clone test, the requirement that was failing:** cloned to a temp dir with
  `OPENAI_API_KEY` unset, no `twcs.csv`, no network. `make eval` exits 0 in **3 seconds** and
  reproduces the headline exactly. Well inside the brief's 15-minute cap.
- Independently confirmed no tweet text is committed: longest non-draft string across both
  committed artifacts is 39 chars, a criterion name. `redact_artifacts.py --check` passes.
- REPORT.md: 5.9 pages against the 6-page cap, zero `[[TBD]]`, zero em dashes.

## Still open

1. **Judge-vs-human agreement is unmeasured**, an explicit brief requirement. The tool is built
   and blind, `judge-import` computes the kappa, but nobody has scored the 30 drafts. Roughly 15
   minutes if there is ever time. README and REPORT both name it as a gap.
2. **`master` is six commits behind** and the submission links the repo.
3. Sirjan still has to submit the Notion form in `docs/ASSIGNMENT.md`.
4. Playbooks never distilled; taxonomy clustering induction never run. Neither is load-bearing.
