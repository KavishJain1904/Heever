"""Strip tweet text out of judge_verdicts.jsonl before it can be committed.

The repo never commits tweet TEXT (CC BY-NC-SA; commit ids + labels only,
TweetSumm precedent -- see .gitignore). src/evaluate.py's Tier 1 path only ever
reads `system`, `message_id`, `criteria[*]["pass"]`, `position_bias_disagreements`
and `human_criteria` off a verdict, so everything else is display-only and safe
to drop.

    python scripts/redact_artifacts.py            # verdicts -> verdicts.redacted
    python scripts/redact_artifacts.py --check     # scan committed artifacts for leaks

Fails loudly (non-zero exit) on any top-level or per-criterion key it doesn't
recognise, so a new judge field that happens to carry text breaks this script
instead of silently reaching a commit.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "artifacts" / "results"
VERDICTS_IN = RESULTS / "judge_verdicts.jsonl"
VERDICTS_OUT = RESULTS / "judge_verdicts.redacted.jsonl"
PREDICTIONS = RESULTS / "predictions.jsonl"
GOLDEN_CSV = REPO_ROOT / "data" / "golden_200.csv"

# Allowlists. Any key not in these raises -- see module docstring.
TOP_LEVEL_ALLOWED = {
    "message_id", "system", "position_bias_disagreements", "human_criteria",
    "criteria",
    # dropped, but recognised as expected/benign rather than "unknown":
    "draft", "usage", "cache_hit", "all_six_pass",
}
TOP_LEVEL_KEPT = {"message_id", "system", "position_bias_disagreements", "human_criteria"}
CRITERION_ALLOWED = {"pass", "order_disagreement", "reasoning", "quote"}
CRITERION_KEPT = {"pass", "order_disagreement"}

CHECK_LONG_STRING_LIMIT = 200
CHECK_ALLOWED_LONG_FIELDS = {"reply_draft", "candidate_draft"}


def redact_row(row: dict) -> dict:
    unknown = set(row) - TOP_LEVEL_ALLOWED
    if unknown:
        raise SystemExit(
            f"redact_artifacts: unknown top-level key(s) {sorted(unknown)} in judge_verdicts.jsonl "
            "-- add them to TOP_LEVEL_ALLOWED (and decide keep/drop) before redacting.")

    out = {k: row[k] for k in TOP_LEVEL_KEPT if k in row}

    criteria = {}
    for name, c in row.get("criteria", {}).items():
        unknown_c = set(c) - CRITERION_ALLOWED
        if unknown_c:
            raise SystemExit(
                f"redact_artifacts: unknown criterion key(s) {sorted(unknown_c)} in "
                f"criteria['{name}'] -- add them to CRITERION_ALLOWED before redacting.")
        criteria[name] = {k: c[k] for k in CRITERION_KEPT if k in c}
    out["criteria"] = criteria
    return out


def redact(in_path: Path = VERDICTS_IN, out_path: Path = VERDICTS_OUT) -> int:
    if not in_path.exists():
        raise SystemExit(f"redact_artifacts: {in_path} not found -- run `make judge` first.")
    rows = [json.loads(line) for line in open(in_path, encoding="utf-8") if line.strip()]
    redacted = [redact_row(r) for r in rows]
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in redacted:
            fh.write(json.dumps(r) + "\n")
    print(f"redacted {len(redacted)} verdicts: {in_path} -> {out_path}")
    return len(redacted)


def _iter_long_strings(obj, path: str, limit: int):
    """Yield (path, string) for every str value longer than `limit`."""
    if isinstance(obj, str):
        if len(obj) > limit:
            yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_long_strings(v, f"{path}.{k}" if path else str(k), limit)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _iter_long_strings(v, f"{path}[{i}]", limit)


def check() -> int:
    problems: list[str] = []

    if PREDICTIONS.exists():
        for i, line in enumerate(open(PREDICTIONS, encoding="utf-8")):
            if not line.strip():
                continue
            row = json.loads(line)
            for key, value in row.items():
                if key in CHECK_ALLOWED_LONG_FIELDS:
                    continue
                for path, s in _iter_long_strings(value, key, CHECK_LONG_STRING_LIMIT):
                    problems.append(f"predictions.jsonl:{i} field '{path}' is {len(s)} chars: {s[:80]!r}...")
    else:
        print(f"note: {PREDICTIONS} not found, skipping")

    if GOLDEN_CSV.exists():
        with open(GOLDEN_CSV, encoding="utf-8", newline="") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                note = (row.get("note") or "").strip()
                if note:
                    problems.append(f"golden_200.csv:{i} non-empty 'note' column: {note[:80]!r}...")
                for key, value in row.items():
                    if key == "note" or key in CHECK_ALLOWED_LONG_FIELDS:
                        continue
                    if value and len(value) > CHECK_LONG_STRING_LIMIT:
                        problems.append(f"golden_200.csv:{i} field '{key}' is {len(value)} chars: {value[:80]!r}...")
    else:
        print(f"note: {GOLDEN_CSV} not found (not labelled yet), skipping")

    if problems:
        print(f"FOUND {len(problems)} POSSIBLE LEAK(S):")
        for p in problems:
            print(f"  {p}")
        return 1
    print("check ok: no long strings outside allowed fields, no non-empty 'note' cells")
    return 0


def blank_note_column(path: Path = GOLDEN_CSV) -> int:
    """Blank the `note` column in-place, if the CSV exists and has one. Returns
    the number of rows that were changed."""
    if not path.exists():
        return 0
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if "note" not in fieldnames:
        return 0
    changed = 0
    for row in rows:
        if (row.get("note") or "").strip():
            row["note"] = ""
            changed += 1
    if changed:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="scan predictions.jsonl / golden_200.csv for leaked text instead of redacting")
    parser.add_argument("--blank-notes", action="store_true",
                        help="blank any non-empty 'note' cells in data/golden_200.csv in place")
    args = parser.parse_args()

    if args.blank_notes:
        n = blank_note_column()
        print(f"blanked 'note' on {n} row(s)")
        return 0
    if args.check:
        return check()
    redact()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
