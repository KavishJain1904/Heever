# Target contract. See docs/10 §5 for the two-tier reproduction guarantee.
#
#   make setup      install pinned deps
#   make smoke      <30s, 50 rows, CPU, NO NETWORK, no data download -- CI-able
#   make eval       TIER 1: headline table from committed artifacts. Offline, no
#                   Kaggle account, no API key, seconds. This is the 15-minute path.
#   make eval-full  TIER 2: regenerate predictions from source. Needs the Kaggle
#                   rejoin (data/raw/twcs.csv) and the committed LLM cache.
#   make figures    confusion matrix, risk-coverage curve, reliability diagram
#   make labels     re-derive the taxonomy. NEEDS AN API KEY. Not required to reproduce.
#   make redact     strip tweet text from judge_verdicts.jsonl -> the committed
#                   .redacted.jsonl that make eval actually reads
#   make test       unit tests
#   make judge-tool build judge_human_30.html -- blind human-scoring page for the
#                   30 judge-validation drafts (gitignored, embeds tweet text)

PY ?= python
.DEFAULT_GOAL := help

.PHONY: help setup smoke eval eval-full figures labels redact test rejoin clean

help:
	@grep -E '^#   make' Makefile | sed 's/^#   //'

setup:
	$(PY) -m pip install -r requirements.txt

smoke:
	$(PY) -m src.evaluate --smoke --rows 50 --offline

# Tier 1. Recomputes every metric, CI and significance test from
# data/golden_200.csv + artifacts/results/predictions.jsonl + judge_verdicts.jsonl.
# Never touches tweet text. Never touches the network.
eval:
	$(PY) -m src.evaluate --from-artifacts --config config.yaml

# Tier 2. Regenerates predictions.jsonl end-to-end, then runs Tier 1 on the result.
eval-full: rejoin
	$(PY) -m src.pipeline --config config.yaml
	$(MAKE) eval

# Rehydrates tweet text by joining committed tweet_ids against the reviewer's own
# twcs.csv. Follows the TweetSumm precedent -- docs/01 §7.
rejoin:
	$(PY) -m src.build_sample --rejoin --config config.yaml

figures:
	$(PY) -m src.evaluate --figures --config config.yaml

labels:
	$(PY) -m src.induce_taxonomy --config config.yaml

# Rebuilds artifacts/results/judge_verdicts.redacted.jsonl (the file make eval
# reads and the only judge_verdicts* file that gets committed) from the full,
# gitignored judge_verdicts.jsonl. Never touches the network.
redact:
	$(PY) scripts/redact_artifacts.py

test:
	$(PY) -m pytest -q

# Experiment stages (scripts/run_experiment.py). Each writes artifacts/results/*.
# predict needs data/raw/twcs.csv-derived data/threads.parquet and an API key on a
# cache miss; everything after `golden` needs data/golden_200.csv.
predict:
	HF_HUB_OFFLINE=1 $(PY) scripts/run_experiment.py predict

golden:
	$(PY) scripts/run_experiment.py golden data/golden_200_labels.xlsx

baselines:
	$(PY) scripts/run_experiment.py baselines

tune:
	$(PY) scripts/run_experiment.py tune

judge:
	$(PY) scripts/run_experiment.py judge

judge-tool:
	$(PY) scripts/build_judge_tool.py

.PHONY: predict golden baselines tune judge judge-tool

clean:
	rm -rf artifacts/results/*.json reports/*.png .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
