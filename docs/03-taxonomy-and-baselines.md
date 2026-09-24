# Intent taxonomy induction, the baseline ladder, and reproducibility

Research pass 3.

## 1. Taxonomy induction — the recipe

Embeddings are not the bottleneck at this scale. `all-MiniLM-L6-v2` (384-d, 6 layers, ~80MB)
embeds tens of thousands of short texts in minutes on laptop CPU; `all-mpnet-base-v2` is ~5×
slower for a modest gain. BERTopic's own FAQ still recommends MiniLM as the English default.
The `bge`/`gte`/`e5` families are stronger on MTEB retrieval but **require the right
query/passage prefixes** (`e5` needs `query:`/`passage:`, `bge` a query instruction) — getting
the prefix wrong silently costs more than the model gains.

API embeddings are ~free here (`text-embedding-3-small` at $0.02/M tokens ⇒ 50k tweets ≈
**$0.06**), so **cost is not the deciding factor — reproducibility is.** A local
sentence-transformer a grader can run offline beats an API embedding requiring a key.

### Clustering notes that matter

- UMAP before density clustering (density methods degrade in high dimensions). `n_neighbors=30`
  rather than the default 15 for noisy short text; **always set `random_state`** or runs are
  not reproducible.
- **Choosing k: silhouette and elbow both mislead on UMAP-projected embeddings** — silhouette
  rewards compact spherical clusters, which UMAP manufactures. Defensible alternative:
  **stability across seeds** (cluster twice on 80% bootstraps, measure ARI between partitions).
- **The HDBSCAN noise cluster is the main failure mode.** On multi-domain short text the outlier
  bin has been observed holding **>74% of the dataset** (arXiv:2212.08459). Mitigations: lower
  `min_samples`, `.reduce_outliers()`, or just use k-means (no outliers, more robust on short
  docs). For *induction*, the noise cluster is a feature — **it is a preview of your `OTHER`
  class. Read it before discarding it.**
- **BERTopic is an exploratory lens, not your taxonomy generator.** On tweets, c-TF-IDF labels
  degenerate into brand names and support boilerplate ("DM", "sorry", "team") — topics are real,
  labels are useless as intents.

### LLM-assisted induction — the current literature

- **TnT-LLM** (Microsoft, arXiv:2403.12173, KDD 2024) — the reference. Phase 1: zero-shot
  multi-stage reasoning where the LLM produces and *iteratively refines* a taxonomy over batches
  (initialise → update → review, explicitly analogised to SGD over minibatches). Phase 2: the
  LLM labels data to train a lightweight classifier that is what actually ships. **Headline
  finding to quote: lightweight classifiers trained on LLM annotations match or beat using the
  LLM directly, with far better scalability and transparency.** They evaluate the taxonomy on
  coverage, label accuracy, and relevance-to-instruction. **This is almost exactly the
  architecture the assignment wants.**
- **ClusterLLM** (arXiv:2305.14871, EMNLP 2023) — LLM as a cheap oracle on hard *triplets*
  ("is A closer to B or C?") and on pairwise questions to pick **clustering granularity**.
  ~$0.6 per dataset. The granularity trick answers "how many intents?".
- **GoalEx** (arXiv:2305.13749, EMNLP 2023) — Propose–Assign–Select with **goal conditioning**.
  "Cluster by what the customer wants the company to do" gives a very different taxonomy than
  "cluster by product". Use the former.
- **Open/new intent discovery**: DeepAligned (arXiv:2012.08987), USNID (arXiv:2304.07699, TKDE).
- **Closest to this domain: Dial-In LLM** (arXiv:2412.09049, EMNLP 2025) — LLM-in-the-loop
  intent clustering for *customer-service dialogues*, >95% agreement with human judgment on
  semantic coherence and cluster naming, on 100k+ real service calls with 1,507 human-annotated
  clusters. Also IntentGPT, SPILL, LUMI, NILC.

### The afternoon recipe

1. Filter to **inbound first-turn** tweets; dedupe near-identical text (hash normalised text) —
   brand macros and bots otherwise dominate clusters.
2. Deterministic seeded subsample of 20k.
3. Embed with MiniLM, cache `.npy`.
4. UMAP(n_neighbors=30, n_components=5, metric='cosine', random_state=42) → k-means at
   k ∈ {15, 25, 40}; one HDBSCAN run just to see the outlier mass.
5. Per cluster: 15 centroid-nearest + 5 random members → LLM names it, writes a one-sentence
   definition, lists 3 boundary cases. (TnT-LLM phase 1 + GoalEx propose, in one call.)
6. Feed all candidate labels back with an explicit **merge/split/drop** instruction and a stated
   target of 8–12 intents. **Twice.** This is the iterative-refinement step.
7. Freeze as `taxonomy/intents.yaml` with `name`, `definition`, `positive examples`,
   **`not-this` negative examples**. That file is what you defend in the interview *and* the
   prompt for the LLM baseline.
8. Label a fresh random 200 by hand against the frozen taxonomy; measure `OTHER` rate and
   self-agreement.

~2–4 hours wall clock, single-digit dollars.

## 2. Designing the taxonomy

**Granularity: 8–12 intents.** Banking77's 77 intents are the cautionary tale — *Label Errors in
BANKING77* (Insights@ACL 2022) found **>1,400 (14%) of 10,003 training utterances may be
mislabelled**, and removing suspect utterances raised accuracy 88.2%→92.4% and F1 87.8→92.0.
Fine granularity with semantic overlap manufactures label noise. At 10 classes, ~200 labels
gives 15–25 per class.

**Mutual exclusivity vs. multi-label.** Real tweets carry two intents ("charged me twice, cancel
my account"). Two honest options: (a) single-label with a documented **precedence rule** —
*"label the action the customer is asking the company to take; if two, label the one that
determines routing"* — or (b) true multi-label. (a) is defensible and cheaper. **The interview
answer is that you measured the ambiguity**: count how often your second-pass annotation
disagrees with your first, and report it as an upper bound on achievable accuracy.

**Two extra classes, kept separate.** `NON_SUPPORT` (praise, jokes, marketing, bots, contests)
is a **filter** decision — drop the message. `OTHER` is an **escalation** decision — a human
should look. **Merging them is a mistake.** The OOS literature backs treating this as
first-class: CLINC150 ships 22,500 in-scope + 1,200 explicitly out-of-scope queries; HINT3
(arXiv:2009.13833) showed that on *real* chatbot traffic all four commercial NLU platforms plus
BERT saturate at inadequate levels with a hard tradeoff between in-scope accuracy and OOS
rejection.

**Four cheap validation tests:** (1) **coverage** — if >15–20% of a random 200 lands in `OTHER`,
a class is missing; (2) **agreement** — κ below ~0.6 on a class means the *definition* is broken,
not the annotator; (3) **balance** — a class with <3% support cannot be evaluated at n=200; merge
it or state it as unmeasurable; (4) **confusability** — symmetric error trading between two
classes is a merge signal.

**Proposed taxonomy** (converges across industry and published taxonomies): order/delivery
status · billing & charges · refund/return · account access & security · cancel/downgrade ·
service outage/technical fault · product or how-to question · change of details ·
complaint/escalation · `PRAISE_OR_NON_SUPPORT` · `OTHER`.

## 3. The baseline ladder

**Tier 0 — trivial.** Majority class; random-stratified-by-prior. Establishes what accuracy
*means* under imbalance. If the majority class is 30%, a 55%-accurate model is not obviously
good. **Report accuracy AND macro-F1 for these** — a majority predictor gets macro-F1 near zero,
and that contrast is precisely the point.

**Tier 0.5 — keyword/regex rules.** 5–15 patterns mined from cluster-representative docs.
Proves how much signal is purely lexical, gives an interpretable floor, doubles as a bootstrap
labeller. Expect high precision on `where is my (order|package|delivery)` and near-zero recall
elsewhere.

**Tier 1 — TF-IDF + LinearSVC.** `FeatureUnion` of word 1–2 grams **and char_wb 3–5 grams**.
**Character n-grams are the single most important detail for this data** — resilient to
misspellings, abbreviations and derivations because they don't depend on whitespace
tokenisation. Tune `C`, `min_df`, `sublinear_tf`, `class_weight='balanced'` over stratified
5-fold. Trains in seconds; **this is the baseline the fancier models must beat.**

**Tier 2 — frozen embeddings + logistic regression, and nearest-centroid/kNN.** Reuses the
cached MiniLM embeddings. LR on 384-d with 200 labels is well-conditioned; nearest-centroid
works with 3 examples per class and needs no training. This is the honest "what do embeddings
buy over bag-of-words" experiment, and it costs one extra `.fit()`.

**Tier 3 — SetFit.** (arXiv:2209.11055) The right tool at 200 labels: contrastive fine-tuning of
a sentence transformer on generated pairs, then a logistic head — no prompts, no verbalizers.
Published figures: **77.9% on Banking77 at 8 shots/class**; on RAFT, SetFit-RoBERTa (355M) scores
**71.3** vs T-Few (11B) 75.8, GPT-3 (175B) 62.7, **human baseline 73.5**; training
**~$0.025 / 30s on a V100** vs ~$0.70 / 11min for T-Few 3B. A follow-up (*Making LLMs Worth
Every Penny*, arXiv:2311.06102) found SetFit at ~20 examples/class comparable to full-data
training on Banking77.

**Tier 4 — LLM zero-/few-shot.** Prompt with the frozen taxonomy YAML as a **constrained enum
menu**, structured JSON out, cache every response. Two upgrades with published support:
**retrieval-augmented few-shot** (KATE, arXiv:2101.06804) and **partial-label-space retrieval**
— show the model only the plausible subset of labels, which set few-shot SOTA on three intent
datasets with no fine-tuning (Milios et al., GenBench 2023, arXiv:2309.10954). Self-consistency
buys a point or two at 3–5× cost; skip unless budget remains.

### Where LLMs win and lose — the citable summary

Bucher & Martini (arXiv:2406.08660) compared GPT-3.5/GPT-4/Claude Opus zero-shot against
fine-tuned RoBERTa/DeBERTa-v3/ELECTRA/XLNet on four tasks and found **fine-tuning superior in
all cases**. But in the *extreme* low-data regime the ordering flips: Loukas et al. found ICL at
1–3 examples/class outperforming fine-tuned MLMs on Banking77. **At ~200 labels you are in the
crossover zone — which is exactly why you run both and report the crossover as a finding.**

**Cost per 1000 messages** (~60 input tokens/tweet, ~400-token prompt, ~20 out): Haiku 4.5
≈ **$0.56** (~$0.28 batched); Sonnet 5 ≈ **$1.12**; TF-IDF+LR: **$0** and sub-second. That
three-order-of-magnitude gap is the argument for TnT-LLM's distillation pattern.

**JSON caveat, worth citing:** *Let Me Speak Freely?* (arXiv:2408.02442) found format
restriction degrades *reasoning* — but also observed JSON-mode **helping on classification**,
because constraining the output space reduces answer-selection errors. For single-label intent,
constrained decoding to a label enum is the right call.

## 4. Evaluating the classifier

**Macro-F1 is the headline**, with accuracy, weighted-F1 and a full per-class P/R/F1/support
table. Micro-F1 equals accuracy in single-label multi-class and is dominated by the largest
class; weighted-F1 hides minority failure by construction.

**Name this trap in the README:** there are **two different "macro-F1" formulas** — arithmetic
mean of per-class F1s, and harmonic mean of macro-averaged P and R — which **can differ by up to
0.5 and even reorder classifiers** (Opitz & Burst, arXiv:1911.03347). The arithmetic version is
more robust under imbalance. **State which you use.**

**`OTHER` distorts macro-F1.** It is a heterogeneous grab-bag; its F1 will be low and drags the
macro down while telling you little. **Report macro-F1 twice** — all classes, and in-scope only
— with `OTHER` handled separately as a binary in-scope/OOS detection problem with its own P/R.
This mirrors how CLINC150 and HINT3 evaluate.

**Error analysis:** normalised confusion matrix; for each of the top 5 off-diagonal cells, three
verbatim tweets. **Symmetric confusion ⇒ taxonomy problem** (merge or redefine). **Asymmetric
confusion into one class ⇒ prior/threshold problem.**

**Calibration matters more than raw accuracy** because escalation is a downstream decision on the
confidence score. Fit temperature or Platt scaling on a held-out slice; report ECE + reliability
diagram. **Do not trust LLM self-reported confidence** — verbalized confidence is generally
overconfident. Prefer logprob-derived scores where exposed. Present a **coverage–accuracy
curve**: "at a 0.8 threshold we auto-handle 62% of traffic at 94% accuracy."

## 5. Imbalance at n≈200

- **Stratified splits, always** — an unstratified split can leave a class absent from test. Use
  stratified 5-fold and **report mean ± std, not a single 80/20 split**; the std is the honest
  signal at this n.
- **Class weights over resampling.** One low-resource ablation showed removing class weights
  degrading macro-F1 0.8032 → 0.7214 (−8.2 points).
- **Per-class threshold tuning** rather than argmax — reported gains of 0.87–5.98 macro-F1
  points single-label. **Tune inside the CV loop, never on test.**
- **LLM paraphrase augmentation, carefully.** Generating training data with the model family you
  later evaluate with creates a closed loop (see *Preference Leakage*, arXiv:2502.01534).
  **Rule to state in the README:** augmented data may enter *training only*; every reported
  metric is computed on human-labelled test data only; any LLM judge must be a different family
  from the generator; **report augmented and non-augmented numbers side by side.**

## 6. Banking77 as harness calibration only

Full-data SOTA is mid-90s (a ModernBERT-base fine-tune reports 0.9399 accuracy / 0.9401
macro-F1). Few-shot: SetFit 77.9% at 8 shots/class.

**Never present a Banking77 number as evidence about the Twitter system.** Use it as a
**methodology calibration set**: run your exact ladder on Banking77 at N=8 or 16 per class and
check it reproduces the published ordering and magnitudes. **If SetFit lands near 77.9% at 8
shots, your implementation is sound and your Twitter numbers are trustworthy. If it lands at
45%, you have a bug.** "I validated the harness on a public benchmark, then applied it to the
target domain" is a strong interview answer. Cite the 14% label-error finding as the reason not
to chase the last 2 points.

## 7. What "reproduce in 15 minutes" forces

1. **No training in the critical path.** `make eval` loads artifacts and computes metrics.
   TF-IDF/LR/SetFit-head fitting takes seconds and can stay; embedding 50k tweets and 1000 LLM
   calls cannot.
2. **Everything expensive is cached and committed** — float16 embeddings (a few MB) and an
   **LLM response cache keyed by `sha256(model + prompt_template_version + input_text)`**.
   Cache-hit means the LLM baseline reproduces exactly, offline, with no API key. **Document the
   cache-miss behaviour explicitly**: if the key is unset and the entry is missing, **fail
   loudly** rather than silently skipping.
3. **A deterministic seeded subsample.** Commit `data/sample_ids.txt` so a grader who downloads
   the 493MB CSV lands on identical rows. Commit the ~200 gold labels as a small CSV — **that is
   the real deliverable.**
4. **Pinned deps and two modes** — `make smoke` (30s, 50 rows, offline, CI-able) and `make eval`.

```
README.md                 # headline table + 3-command repro
Makefile
requirements.txt          # fully pinned
config.yaml               # every threshold
taxonomy/intents.yaml     # frozen: name, definition, examples, not-this
data/
  sample_ids.txt          # seeded deterministic subsample (committed)
  golden_200.csv          # gold labels + stratum tags (committed)
  raw/                    # gitignored; fetch script + checksum
artifacts/
  embeddings_minilm.npy   # committed, float16
  llm_cache.jsonl         # committed, keyed by prompt hash
  playbooks/*.md          # distilled, with evidence_ids
  results/*.json
src/
  build_sample.py  induce_taxonomy.py  featurize.py
  baselines/{trivial,rules,tfidf,embed_lr,setfit,llm}.py
  retrieve.py  policy.py  generate.py  judge.py  evaluate.py
reports/
  confusion_matrix.png  error_analysis.md  taxonomy_notes.md
tests/test_policy.py  tests/test_smoke.py
```

```
make setup     # pip install -r requirements.txt (pinned)
make smoke     # <30s, 50 rows, CPU-only, no network — CI-able
make eval      # <5 min CPU: cached embeddings + LLM cache, prints headline table
make figures   # confusion matrix, risk-coverage curve, reliability diagram
make labels    # (optional) re-derive taxonomy — needs API key, documented as NOT required
```

README's first section is literally `git clone && make setup && make eval` with the expected
output table pasted below it, then a "what costs money and what doesn't" note.

## 8. Predicted results (state as predictions, then measure)

rules ≈ 0.25–0.35 macro-F1 · TF-IDF ≈ 0.55–0.65 · embeddings+LR ≈ 0.60–0.70 · SetFit ≈
0.65–0.75 · LLM few-shot ≈ 0.70–0.80. **The interesting finding will be *which* classes each
wins on** — the LLM on rare ones, TF-IDF on lexically distinctive ones. That decomposition is
worth more than the aggregate.
