# Hiver SDE Intern take-home — what is actually being graded

Source: assignment brief (Google Doc), received 2026-09-10.
Submission form: https://intelligent-bar-256.notion.site/39492cbf0da2800682cfc78a600a745f

## The one sentence that determines everything

> "What we are testing: whether you can turn a messy real-world dataset into a working
> AI system and prove it works. **The proof is worth more than the system.**"

Read literally, this is not an ML-modelling assignment. It is an **evaluation-engineering**
assignment with an ML system attached as the object under test. Most candidates will
invert this — they will spend 80% of their time on prompt/RAG quality and ship a thin
`eval.py` with an accuracy number. The differentiated submission spends 60%+ of its
effort on the measurement apparatus and treats the agent as the thing being measured.

Corroborating signals in the brief:
- The golden set has a **hand-labelled** requirement with a **sampling + labelling note**.
  They want to see that you understand sampling bias, not that you can label 200 rows.
- "evidence of how well your judge agrees with a human" — they know LLM-judges are
  unreliable and want to see you know it too, with a number.
- A **mandatory** section titled "What is misleading about my headline number?" This is
  the single highest-signal item in the entire brief. It is an intellectual-honesty test.
  A candidate who writes a weak version of this section fails the test the brief is
  actually running.
- "Results vs. at least two baselines (a trivial one and a simple one)" — they want to
  know whether your LLM system beats TF-IDF, and they suspect it might not by much on
  intent classification.
- "Failure analysis: top 5 failure modes with real examples and hypotheses" — real
  examples, not categories. Requires you to have actually read your model's outputs.
- "Decision log — 10-15 **non-obvious** decisions" — the word non-obvious is doing work.
  "I used Python" is not a decision. "I excluded threads where the brand's only reply is
  a DM deflection, because including them inflates every similarity metric" is.
- "We will ask you to explain and modify your own code live." — every abstraction you
  cannot defend line-by-line is a liability. This is a hard argument against dragging in
  a heavyweight framework.
- "We will not run your code on the full dataset — a subsample is expected and
  encouraged." — they are explicitly removing scale as a scoring dimension. Do not spend
  compute proving you can process 3M rows.
- "README must let us reproduce your headline results in under 15 minutes." — a
  reviewer will clone, run one command, and start a timer. If that path requires a
  Kaggle login, an API key with credit, a GPU, or a 20-minute embedding job, you have
  already lost points before they read a word of the report.

## Inferred rubric (my reconstruction, ~weights)

| Dimension | Weight | What a top answer looks like |
|---|---|---|
| Evaluation rigour | 30% | Stratified golden set with an unbiased random stratum; CIs on every headline number; judge-vs-human agreement reported as a coefficient, not a vibe |
| Intellectual honesty | 20% | The "misleading" section names real, specific, self-inflicted threats to validity — including ones the grader had not thought of |
| Failure analysis | 15% | 5 modes, each with a verbatim example, a hypothesis, and a proposed test that would confirm/refute it |
| Working system | 15% | Runs, is reproducible in <15 min, produces the three required outputs, code is legible and defensible live |
| Framing + scoping | 10% | Explicit definition of "good" for the chosen brand; an explicit **not-building** list |
| Baselines | 10% | Trivial + simple baselines that are genuinely tuned, not strawmen |

## Explicitly requested deliverables — checklist

- [ ] Repo, runnable pipeline, README reproducing headline results in <15 min
- [ ] Golden eval set, 150–250 hand-labelled examples, with sampling + labelling note
- [ ] Eval harness: automated metrics + LLM-as-judge rubric + judge/human agreement evidence
- [ ] Report (≤6 pages or a README section):
  - [ ] Problem framing: what "good" means for this brand; **what you chose not to build**
  - [ ] Results vs. ≥2 baselines (trivial + simple)
  - [ ] Failure analysis: top 5 modes, real examples, hypotheses
  - [ ] **"What is misleading about my headline number?"** (mandatory)
  - [ ] What you'd do next with one more week
- [ ] Decision log: 10–15 non-obvious decisions + why
- [ ] Citations for anything borrowed

## Traps worth naming up front

1. **The DM-deflection trap.** On Twitter support, a large share of brand replies are
   "Sorry to hear that — please DM us." If those stay in the corpus, a system that
   always outputs "Please DM us" scores extremely well on any reply-similarity metric
   while being worthless. Any headline reply-quality number that does not address this
   is fraudulent. This is simultaneously a trap and the best single item for the
   "misleading" section.
2. **The judge/generator collusion trap.** Using the same model family to generate and
   to judge inflates scores (self-enhancement bias). Cheap fix: different family for the
   judge, plus report the human-agreement coefficient.
3. **The historical-reply-as-ground-truth trap.** The brand's actual historical reply is
   not "correct" — it is just what a rushed human typed. Scoring similarity-to-history
   rewards mimicking mediocrity. Needs to be stated, and ideally measured (have the
   judge score the *historical* replies too and report that as a reference band — if the
   human replies score 3.4/5 and the model scores 3.6/5, that reframes the whole result).
4. **The accuracy-on-imbalanced-intents trap.** If 40% of messages are one intent,
   majority-class accuracy is 40% and looks respectable. Macro-F1 with per-class support
   is the honest report; accuracy is the misleading headline.
5. **The 15-minute trap.** Live LLM calls in the reproduction path make the run slow,
   costly, non-deterministic and dependent on the reviewer having a key. A committed
   response cache keyed by prompt hash fixes all four at once, and is itself a good
   decision-log entry.
6. **The framework trap.** "Explain and modify your own code live" punishes LangChain
   abstractions you would have to read the source of on the spot.

## Time-budget shape (assuming ~5 focused days)

- Day 1: data acquisition, brand selection with evidence, corpus construction, EDA.
- Day 2: taxonomy induction, annotation guidelines, golden-set labelling (the long pole).
- Day 3: baselines + agent v1 + eval harness skeleton.
- Day 4: judge design, judge/human agreement study, error analysis pass.
- Day 5: report, decision log, reproduction hardening, cache commit, README timing test.

The labelling day is the one that slips. Budget it first, not last.
