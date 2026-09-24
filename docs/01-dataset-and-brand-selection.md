# Dataset, brand selection, and licensing

Research pass 1. Every number here is sourced; anything unverified is marked. **Recompute
the handful you actually cite in the report** — the corroborating sources are third-party
take-home repos, not authorities.

## 1. Corpus shape (high confidence — 3+ independent full-file computations agree exactly)

One file, `twcs.csv`.

| Metric | Value |
|---|---|
| Rows | **2,811,774** (Kaggle's "~3M" is marketing) |
| Size | 516,508,641 B ≈ **492.6 MB** uncompressed |
| Inbound (customer→brand) / outbound | 1,537,843 (54.7%) / 1,273,931 (45.3%) |
| Unique `author_id` | 702,777 |
| Distinct brand handles | **108** |
| Duplicate `tweet_id` / duplicate rows | **0 / 0** |
| Thread roots (null parent) | **794,335** (28.3%) |
| Rows with multiple comma-separated children | **222,426** |
| Date range | 2008-05-08 → 2017-12-03, bulk Oct–Dec 2017 |

Cross-check that raises confidence: the HF derivative `TNE-AI/customer-support-on-twitter-conversation`
reports **794,335 conversations** — exactly the root count. Independent confirmation that
"root = null parent" is the canonical threading rule.

## 2. Schema

| Column | Parse as | Semantics (Kaggle, verbatim) |
|---|---|---|
| `tweet_id` | `int64` | "A unique, **anonymized** ID." Surrogate key, **not** a Twitter snowflake ID. |
| `author_id` | `str` | "A unique, anonymized user ID. **@s in the dataset have been replaced with their associated anonymized user ID.**" |
| `inbound` | `bool` | `True` = customer→brand. |
| `created_at` | datetime | `Tue Oct 31 22:10:47 +0000 2017` — **not ISO**, needs explicit `format=`. |
| `text` | `str` | "Sensitive information like phone numbers and email addresses are replaced with mask values like `__email__`." |
| `response_tweet_id` | `list[int]` | Children, comma-separated. |
| `in_response_to_tweet_id` | `Int64` | Parent. **Naïve `read_csv` infers `float64`** (NaN present) and silently corrupts IDs. Cast to nullable `Int64` or `str`. |

There is **no** `brand`, `conversation_id`, `language`, `sentiment` or `resolution` column.
Brand = `author_id` where `inbound == False`.

### Thread reconstruction recipe

1. Build `parent → children` from `in_response_to_tweet_id` — more reliable than
   `response_tweet_id` (one clean int per row, no list parsing).
2. A tweet is a root if its parent is null **or its parent id is absent from the file**
   (dangling reference — 654 orphan roots for AmazonHelp alone).
3. BFS from each root; `conversation_id := root tweet_id` (deterministic, reproducible).
4. Sort turns by `created_at`, tie-break on `tweet_id`.

## 3. Gotchas that are silent correctness bugs

- **Non-contiguous IDs.** Range 1–2,987,950 over 2,811,774 rows — ~176k IDs referenced or
  reserved but absent. Never assume `id+1`.
- **Multi-branch threads.** 222,426 rows have multiple children (19,381 branching
  AmazonHelp conversations). Flattening a tree to a list *without a stated branch policy*
  is a silent bug. Policy to state: prefer the branch ending with the brand, then longest,
  deterministic tie-break.
- **Brand replies split across tweets.** Real AmazonHelp case: three tweets at the
  *identical* timestamp `06:23:00`, marked `1/3^AP  2/3^AP  3/3^AP`, **stored out of order**
  (3/3 first). Sorting by `created_at` alone scrambles them — parse the `N/M` marker.
- **`^AB` agent-initials signatures.** Confirmed across brands with **four different sigils**:
  AmazonHelp `^SM ^JE ^AP`, Spotify `/AL`, Delta `*AMV`, comcastcares `~AT`. Strip with
  `[\^~*/][A-Z]{1,3}\s*$` before embedding, or you get similarity clusters keyed on *agent
  identity* and a generator that hallucinates initials (observed in a real run).
- **The dual-identity trap.** Customer tweets mention *numeric* handles (`@115821`,
  `@115850`, `@116935` are all Amazon-family) while brand replies say `@AmazonHelp`.
  Filtering only on `author_id == 'SpotifyCares'` **misses every inbound tweet**. Recover the
  brand for inbound rows **via the reply edge**, never via text matching.
- **URLs are NOT masked** — real `t.co` shortlinks survive (41.3% of AmazonHelp tweets carry
  a link). Replace with a `[URL]` token or the embedder keys on random hashes.
- **Non-English contamination.** Verified real row under AmazonHelp:
  `ご購入いただきありがとうございました。ぜひご活用くださいませ！😉 EK`. AmazonHelp measures
  **77.5% English**.
- **Near-duplicate templated replies** (~0.75% exact text dupes in a 500k sample) are the
  retrieval-leakage hazard — not duplicate rows, of which there are none.

## 4. Brand distribution (outbound tweets; identical across four independent repos)

| Brand | Outbound | Brand | Outbound |
|---|---:|---|---:|
| AmazonHelp | **169,840** | comcastcares | 33,031 |
| AppleSupport | **106,860** | British_Airways | 29,361 |
| Uber_Support | 56,270 | SouthwestAir | 28,977 |
| SpotifyCares | 43,265 | VirginTrains | 27,817 |
| Delta | 42,253 | Ask_Spectrum | 25,860 |
| Tesco | 38,573 | XboxSupport | 24,557 |
| AmericanAir | 36,764 | sprintcare | 22,381 |
| TMobileHelp | 34,317 | hulu_support | ~14.6k convs |

There are also **~20,794 conversations under an empty brand string** — a real bucket you
must explicitly exclude.

## 5. The DM-deflection measurement — the decision-relevant number

Share of brand replies that are essentially *"please DM us"*. Three independent repos;
absolute magnitudes differ by regex but **the ranking is stable**. Treat as **ordinal, not
cardinal**.

| Brand | src A | src B | src C |
|---|---:|---:|---:|
| AmazonHelp | 0.64% | 0.7% | 1.2% |
| hulu_support | — | — | **1.0%** |
| Delta | 16.3% | 16.4% | — |
| AmericanAir | — | 16.8% | 15.9% |
| XboxSupport | — | 20.9% | — |
| Tesco | — | 26.8% | — |
| SpotifyCares | 31.3% | 30.8% | 35.5% |
| Uber_Support | 35.4% | 35.4% | — |
| **AppleSupport** | **52.5%** | **52.5%** | **64.2%** |
| **TMobileHelp** | — | **81.8%** | — |

**This inverts the naïve intuition.** AppleSupport — the brand most candidates pick, and the
one the 2018 Hardalov paper used — is a *deflection machine*: half to two-thirds of its public
replies push the conversation off-platform, so "draft a reply grounded in how the brand
historically resolved this" degenerates to "say DM us". TMobileHelp (~82%) is disqualifying.

### Two further traps, both measured

- **The 1-hop retrieval trap.** In (customer → immediate reply) pairs, **65.9% of first
  replies are diagnostic questions**, not resolutions. Index 1-hop pairs and your RAG learns
  to ask questions forever. Tracing multi-turn trees to a substantive solution *plus* a
  customer confirmation ("thanks, it worked") yielded only **11,944 clean resolution
  precedents** for Spotify out of 43,265 replies.
- **Starter vs. follow-up.** 62.5% of Spotify inbound tweets are thread starters; 37.5% are
  mid-thread fragments like *"iPhone 8, iOS 11"* or *"thanks"*. Scope the agent to starters
  or the intent classifier trains on context-free noise.

## 6. Brand comparison and recommendation

| Brand | Volume | Depth | Intent diversity | In-thread resolution | Language | Rank |
|---|---|---|---|---|---|---|
| **SpotifyCares** | 43k / ~28k convs | median 2, 38% >2 turns | genuinely multi-topic | **Good** — 42.6% "pure text fix", real remediation steps | **>99.7% EN** | **1** |
| hulu_support | ~14.6k convs | 3.2 avg, 39.8% >2 | streaming taxonomy | **Best in dataset** — 1.0% DM, 99.7% substantive | 89.3% EN | **2** |
| AmazonHelp | 170k / ~82k convs | **best**: 4.8 avg turns | widest | **Deceptive** — 0.64% DM but 41.2% link-only; resolution depends on a private order DB | **77.5% EN** ❌ | 3 |
| AmericanAir / Delta | 37k / 42k | 2.0–2.2 | narrow (~5 intents) | needs PNR + live schedule — **ephemeral facts**, worst case for grounding | >99.8% EN | 4 |
| AppleSupport | 107k | 3.0 avg | broad | **Poor** — 52–64% DM | >99.8% EN | 5 |
| Uber_Support | 56k | 2.0 | rigid templates | **Poor** — 35% DM, 48% link-only | >99.9% EN | 6 |
| TMobileHelp | 34k | 2.0 | narrow | **Disqualifying** — 81.8% DM | 89.3% EN | ✗ |

### Recommendation: **SpotifyCares**

The only candidate that clears all five bars at once:

1. **Volume is sufficient but not excessive** — ~28k conversations is far more than a
   150–250 golden set needs and small enough to embed on a laptop.
2. **Resolutions are genuinely in-thread and reproducible.** "Clear cache, reinstall, check
   OS version, log out of all devices" is a *fact about the world* a retrieval corpus can
   legitimately ground. Contrast AmazonHelp ("where is order 113-…?" — private DB) and the
   airlines ("is DL102 delayed?" — ephemeral). **This is the crux: pick a brand whose
   historical answers are still true today.**
3. **Highest observable direct-reply rate among the top 5 — 79.08%.**
4. **>99.7% English** — no language-filtering step to defend in review.
5. **The auto-handle/escalate boundary is crisp and principled**: playback / app bug /
   catalog / UI → auto; account-security and billing → always escalate. That is exactly the
   story the brief asks for.

**Runner-up: hulu_support** — best DM/substantive ratio in the entire dataset (1.0% / 99.7%),
same clean streaming taxonomy, half the volume, and almost nobody picks it. Pick AmazonHelp
only if you explicitly reframe the task as *deflection quality* rather than *resolution
quality*, and say out loud that 41% of its "resolutions" are a link.

## 7. Licensing — what you may commit

**Key structural fact:** `tweet_id` values run 1–2,987,950 and Kaggle calls them *"anonymized"*.
**These are surrogate keys, not Twitter snowflake IDs** — so the ID-only rehydration workflow
that X's developer terms contemplate is *impossible* here. Nobody can rehydrate from this
dataset; @handles are stripped and emails/phones masked upstream.

X's terms historically: redistribute **IDs only**, and *"you should not make hydrated X
content publicly available (for example… in a public GitHub repository)."*

**Verified 2026-09-16: the Kaggle license label is `CC BY-NC-SA 4.0`.** Read from Kaggle's own
dataset metadata (`kaggle.com/api/v1/datasets/view/thoughtvector/customer-support-on-twitter`,
field `licenseName`), because the HTML page renders client-side and returns no license text.
The dataset description adds, verbatim: *"For commercial applications and use of full dataset,
please contact stuart@thoughtvector.io"*. So: attribution required, non-commercial only, and
derivatives under the same license. A take-home evaluation is non-commercial use; the rules
below keep the repo inside the share-alike term by never redistributing the text itself.

Safe practice under that license:

1. **Never commit `twcs.csv` or a large slice.** Already `.gitignore`d in this repo.
2. Commit a **download script + checksum**, not data.
3. For the golden set, commit **`tweet_id` + your labels** plus a rejoin script.
   **Precedent to cite: TweetSumm** (Findings of EMNLP 2021) does exactly this — ships
   `tweet_id` + offsets + annotations and a processor requiring the user's own `twcs.csv`.
   Citing that in the README is a strong, defensible move.
4. For illustrative rows in the README, ≤20–50 tweets as fair-use illustration with a note,
   preferring **brand-authored** tweets (corporate speech, no individual privacy interest)
   over customer tweets.

## 8. Download and deterministic subsampling

```bash
pip install kaggle                        # needs ~/.kaggle/kaggle.json, chmod 600
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw
unzip -o data/raw/customer-support-on-twitter.zip -d data/raw    # -> data/raw/twcs.csv
```

Unofficial HF mirrors, **verify row count + checksum before trusting**: `SunidhiSriram/twcs`
(reported straight mirror), `TNE-AI/customer-support-on-twitter-conversation` (794,335
pre-threaded conversations with a `company` label, ~207 MB parquet — convenient for brand EDA).

```python
# polars, streaming, never materializes the file
import polars as pl
(pl.scan_csv("data/raw/twcs.csv", infer_schema_length=0)   # all-str, dodges the float64 id bug
   .filter(pl.col("author_id") == "SpotifyCares")
   .sink_parquet("data/spotify.parquet"))

# deterministic hash sample — stable across chunk boundaries, library versions, machines
import hashlib
keep = lambda i: int(hashlib.md5(str(i).encode()).hexdigest()[:8], 16) % 100 < 5   # exact 5%
```

**Sample at the conversation level (hash the root id), never the tweet level.** Tweet-level
sampling leaks turns of the same thread across train/eval.

## 9. Prior work — and what not to repeat

- **Oraby et al., "How May I Help You?" (IUI 2017, arXiv:1709.05413)** — 25 fine-grained
  dialogue acts for Twitter customer service, SVM-HMM turn prediction, plus prediction of
  satisfaction/frustration/resolution. **Read for the taxonomy; do not re-derive dialogue acts.**
- **Hardalov et al. (AIMSA 2018, arXiv:1809.00303)** — retrieval vs. seq2seq vs. Transformer
  on this exact dataset, focused on Apple. **Avoid repeating:** end-to-end seq2seq reply
  generation scored with BLEU/ROUGE is 2018 framing and weak in 2026.
- **TweetSumm (Findings of EMNLP 2021)** — 1,100 dialogs reconstructed *from this dataset* with
  ~6,500 human summaries. Directly reusable: human-verified thread reconstructions to
  sanity-check yours against, and the licensing template above.
- **mtaruno/eve-bot** — the best-known applied project (AppleSupport, Doc2Vec + K-Means/DBSCAN/LDA,
  BiLSTM). The honest lesson: **unsupervised clustering alone did not produce a usable
  taxonomy** — the author had to hand-craft mutually exclusive intents.
- **Independent replication of that lesson:** a real-data run reports **silhouette 0.036** at
  k=16 on real tweets (vs ~0.40 on synthetic), keyword weak-labelling coverage collapsing
  ~72% → **~16%**, and CV macro-F1 falling 0.99 → 0.70 with the classifier over-predicting
  `praise_thanks`. **Never report a headline number computed against keyword-derived labels.**
- **Avoid repeating** the popular Kaggle notebooks: VADER sentiment, word clouds, generic LDA,
  response-time-by-brand bar charts.

**Intents that converge across all sources** (a strong prior): order/delivery status · billing
& subscription · account access & security · technical/app bug · returns/refunds/cancellation ·
product or policy question · complaint/feedback · praise/thanks · other-unclear. Note
`praise_thanks` and `other_unclear` are **large** classes in reality and are exactly what
wrecks a naïve accuracy number.

## 10. Banking77 — honest assessment

**13,083 utterances / 77 intents**, 10,003 train / 3,080 test, `text` + `label`, CC-BY-4.0.
Reference numbers: BERT-Fixed 87.19 · USE 92.81 · **USE+ConveRT 93.36** (Casanueva et al. 2020,
arXiv:2003.04807); 93.83 full-data / **85.95 at 10-shot** (Mehri & Eric). *Label Errors in
BANKING77* (Insights@ACL 2022) documents real annotation errors — 92.4% on a trimmed version.
Treat any ">98% by prompting an LLM" claim with suspicion given label noise and likely
pretraining contamination.

**Domain mismatch is severe and unfixable** — single-domain banking vs. 108 consumer brands;
clean annotation-written sentences vs. noisy tweets with emoji, sigils, sarcasm and split
messages; 77 fine-grained intents vs. your 6–10 coarse ones. **Transfer learning from
Banking77 will not help and should not be attempted.**

**Genuinely useful:**
1. **Methodology calibration.** Run *your exact* stack on it. Landing near ~93% proves the
   machinery is sound, so a low Twitter number is a data/taxonomy problem, not a bug. Cheap,
   powerful "is my code right?" control.
2. **Measure the LLM's few-shot intent ability against a labelled ceiling.** Lets you claim
   *"the model does 77-way intent at ~9x%; the Twitter gap is taxonomy ambiguity, not model
   capability"* — far more sophisticated than a bare Twitter number.
3. **Label-noise humility** — cite the label-errors paper to justify why your golden set has
   an adjudication story.

**Useless or misleading:** fine-tuning on it then applying to tweets; mixing rows into your
training set; mapping the 77 labels onto your taxonomy; quoting 93.36% as if it predicts your
performance; using its rows as few-shot exemplars (wrong register, drags outputs toward
banking vocabulary).

## 11. Strategic note: this assignment has a public corpus of prior submissions

The corroborating sources for §1–§5 are **other candidates' public GitHub repos for this exact
Hiver assignment** — at least ten of them, several visibly AI-assisted, most picking
AppleSupport or AmazonHelp, most shipping the same EDA → k-means → LLM-prompt → accuracy-number
arc. Two implications:

1. **Assume the graders have seen this pattern many times.** Differentiation comes from the
   evaluation rigour and the honesty section, exactly as the brief says.
2. **Do not converge on the median submission.** Choosing SpotifyCares or hulu_support on
   *measured deflection grounds*, and saying why AppleSupport is a trap, is itself a
   decision-log entry that most submissions cannot make.

## 12. Confidence

**High** — corpus shape, brand outbound counts, column semantics, the gotchas (all seen in
quoted real rows).
**Medium** — DM-deflection *ordering* (ordinal only); per-brand language purity; conversation
counts (method-dependent, 81k–83k for AmazonHelp); the 65.9% and 62.5/37.5 splits (single
source each, methodologically sound).
**Low / verify yourself** — compressed zip size;
HF mirror fidelity; **which brands truly resolve in-thread** (every measurement is a regex
heuristic; nobody has human-verified this — **read 100 random threads for your top two
candidates by hand**, one afternoon that beats every number in this report).
