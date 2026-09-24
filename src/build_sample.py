"""Stage 1: raw twcs.csv -> data/threads.parquet, plus golden-set rejoin.

Contract: docs/12 §1. Schema: docs/11 §1.

Four silent-correctness bugs this module exists to avoid (docs/01 §2-3):
  1. `in_response_to_tweet_id` inferred as float64 by a naive read_csv (NaN present)
     silently corrupts IDs. Read all columns as str, cast to nullable Int64.
  2. `created_at` is "Tue Oct 31 22:10:47 +0000 2017" -- NOT ISO. Needs explicit
     format="%a %b %d %H:%M:%S %z %Y".
  3. Brand replies split across tweets share an identical timestamp and are stored
     out of order ("3/3" first). Sorting by created_at alone scrambles them --
     parse the N/M marker.
  4. Customers address numeric anonymised handles (@115821) while replies say
     @AmazonHelp. Filtering on author_id text MISSES EVERY INBOUND TWEET. Recover
     the brand for inbound rows via the reply edge, never via text matching.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

CREATED_AT_FORMAT = "%a %b %d %H:%M:%S %z %Y"

# ^AB agent-initials signatures, four confirmed sigils across brands:
# AmazonHelp ^SM, Spotify /AL, Delta *AMV, comcastcares ~AT.
# Strip before embedding or you get similarity clusters keyed on AGENT IDENTITY,
# and a generator that hallucinates initials (predicted failure mode #3).
SIGIL_PATTERN = r"[\^~*/][A-Z]{1,3}\s*$"

# URLs are NOT masked upstream -- real t.co shortlinks survive. Replace or the
# embedder keys on random hashes.
URL_TOKEN = "[URL]"

# Split-reply marker, e.g. "1/3^AP".
SPLIT_MARKER_PATTERN = r"^(\d+)/(\d+)\b"

_SIGIL_RE = re.compile(SIGIL_PATTERN)
_SPLIT_RE = re.compile(SPLIT_MARKER_PATTERN)
_URL_RE = re.compile(r"https?://\S+|\bwww\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_WS_RE = re.compile(r"\s+")

RAW_COLUMNS = (
    "tweet_id", "author_id", "inbound", "created_at", "text",
    "response_tweet_id", "in_response_to_tweet_id",
)


def load_raw(csv_path: Path):
    """Stream twcs.csv with every column as str. Returns a polars LazyFrame.

    infer_schema_length=0 dodges the float64 id bug. Never materializes the file.

    Why this matters concretely: `in_response_to_tweet_id` is empty for every
    thread root, so a schema-inferring reader sees nulls, picks float64, and
    rewrites 1234567890123456789 as 1.2345678901234568e18. The ids still LOOK like
    ids. The joins just quietly stop matching.
    """
    import polars as pl

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download the Kaggle corpus "
            f"(thoughtvector/customer-support-on-twitter) and place twcs.csv there. "
            f"Tier 1 (`make eval`) does not need this file."
        )
    return pl.scan_csv(
        csv_path,
        infer_schema_length=0,      # every column as Utf8; no silent float coercion
        low_memory=True,
    )


def _parse_created_at(lazy_frame):
    """Explicit format. The default ISO parser returns null on every row here."""
    import polars as pl

    return lazy_frame.with_columns(
        pl.col("created_at")
        .str.to_datetime(format=CREATED_AT_FORMAT, strict=False)
        .alias("created_at_dt")
    )


def _split_marker(text: str) -> Optional[tuple[int, int]]:
    """Parse a '1/3' split-reply marker. None when absent, which is the common case."""
    match = _SPLIT_RE.match((text or "").strip())
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)))


def _order_turns(turns: list[dict]) -> list[dict]:
    """Sort by created_at, tie-break on tweet_id, then repair split-reply runs.

    Bug 3, concretely: three AmazonHelp tweets carry the identical timestamp
    06:23:00 and the markers 1/3^AP 2/3^AP 3/3^AP, with 3/3 stored first. Sorting
    by timestamp alone is stable but preserves the wrong order, so the reply reads
    backwards and any precedent distilled from it is nonsense.
    """
    ordered = sorted(turns, key=lambda t: (t["created_at_dt"], int(t["tweet_id"])))

    # Re-sort each run sharing (author, timestamp) by its N/M marker.
    out: list[dict] = []
    i = 0
    while i < len(ordered):
        j = i + 1
        while (
            j < len(ordered)
            and ordered[j]["created_at_dt"] == ordered[i]["created_at_dt"]
            and ordered[j]["author_id"] == ordered[i]["author_id"]
        ):
            j += 1
        run = ordered[i:j]
        if len(run) > 1 and any(_split_marker(t["text"]) for t in run):
            run = sorted(
                run,
                key=lambda t: (_split_marker(t["text"]) or (10**6, 0))[0],
            )
        out.extend(run)
        i = j
    return out


def _branch_score(branch: list[dict], brand_author_ids: set[str]) -> tuple:
    """Branch policy, as a sort key. Highest wins.

    Stated explicitly because flattening a tree to a list without a policy is a
    silent bug, and 222,426 rows have multiple children. Preference order:
      1. ends with a brand reply  -- an unresolved customer-last branch is not a
         precedent, it is an abandoned conversation
      2. longest                  -- more turns is more context
      3. lowest terminal tweet_id -- an arbitrary but DETERMINISTIC tie-break, so
                                     two runs of this pipeline agree
    """
    terminal = branch[-1]
    ends_with_brand = terminal["author_id"] in brand_author_ids
    return (int(ends_with_brand), len(branch), -int(terminal["tweet_id"]))


def build_threads(lazy_frame, brand: str):
    """Reconstruct conversations and filter to one brand.

    Recipe (docs/01 §2):
      1. Build parent -> children from in_response_to_tweet_id (one clean int per
         row) rather than response_tweet_id (comma-separated list).
      2. A tweet is a root if its parent is null OR its parent id is absent from
         the file (dangling reference -- 654 orphan roots for AmazonHelp alone).
      3. BFS from each root; conversation_id := root tweet_id (deterministic).
      4. Sort turns by created_at, tie-break on tweet_id, then reorder any
         split-reply run by its N/M marker.

    BRANCH POLICY (stated, because flattening a tree to a list without one is a
    silent bug -- 222,426 rows have multiple children): prefer the branch ending
    with a brand reply, then the longest branch, then the lowest terminal tweet_id.
    """
    import polars as pl

    frame = _parse_created_at(lazy_frame).collect()
    rows = frame.to_dicts()
    by_id = {r["tweet_id"]: r for r in rows}

    # --- Bug 4: the dual-identity trap -------------------------------------
    # The brand's own tweets have author_id == "SpotifyCares". Inbound tweets have
    # a numeric anonymised author_id and address "@SpotifyCares" only sometimes --
    # frequently they address "@115821". So: identify the brand by its OUTBOUND
    # author_id, then recover the brand of an inbound tweet from the reply edge
    # (who replied to it), never from matching text.
    brand_author_ids = {
        r["author_id"] for r in rows
        if str(r.get("inbound", "")).lower() == "false"
        and r["author_id"].lower() == brand.lower()
    }
    if not brand_author_ids:
        raise ValueError(
            f"no outbound tweets found for brand {brand!r}. Check "
            f"config.brand.handle against the author_id values in twcs.csv -- the "
            f"handle is case-sensitive and appears only on OUTBOUND rows."
        )

    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []
    for r in rows:
        parent = (r.get("in_response_to_tweet_id") or "").strip()
        if not parent or parent not in by_id:
            # Root, or a dangling reference to a tweet absent from the file.
            roots.append(r["tweet_id"])
        else:
            children[parent].append(r["tweet_id"])

    threads = []
    for root_id in roots:
        # Enumerate root-to-leaf branches, then apply the branch policy.
        branches: list[list[dict]] = []
        stack = [[root_id]]
        while stack:
            path = stack.pop()
            kids = children.get(path[-1], [])
            if not kids:
                branches.append([by_id[t] for t in path])
                continue
            for kid in kids:
                stack.append(path + [kid])

        best = max(branches, key=lambda b: _branch_score(b, brand_author_ids))
        if not any(t["author_id"] in brand_author_ids for t in best):
            continue  # not this brand's conversation

        turns = _order_turns(best)
        first = turns[0]
        if str(first.get("inbound", "")).lower() != "true":
            continue  # brand-initiated; not an inbound support request

        threads.append({
            "conversation_id": root_id,
            "brand": brand,
            "n_turns": len(turns),
            "customer_text": first["text"],
            "customer_text_normalised": normalise_text(first["text"]),
            "created_at": first["created_at_dt"],
            "turns": [
                {
                    "tweet_id": t["tweet_id"],
                    "author_id": t["author_id"],
                    "is_brand": t["author_id"] in brand_author_ids,
                    "text": t["text"],
                    "text_normalised": normalise_text(t["text"]),
                }
                for t in turns
            ],
        })

    return pl.DataFrame(threads)


def normalise_text(text: str) -> str:
    """Strip agent sigils, replace URLs with [URL], collapse whitespace.

    Applied before embedding and before BM25 indexing. NOT applied to the text
    stored for display -- the decision record keeps the redacted original.

    The sigil strip is not cosmetic. Leave it in and the nearest-neighbour
    structure of the embedding space is partly organised by which human agent
    signed the tweet, so retrieval returns "replies by AL" rather than "replies
    about playback failures", and the generator learns to sign its own drafts.
    """
    if not text:
        return ""
    out = _URL_RE.sub(URL_TOKEN, text)
    out = _MENTION_RE.sub("@user", out)
    out = _SPLIT_RE.sub("", out.strip())
    # Sigils sit at the end; strip repeatedly in case a split marker exposed another.
    previous = None
    while previous != out:
        previous = out
        out = _SIGIL_RE.sub("", out).strip()
    return _WS_RE.sub(" ", out).strip()


def deterministic_sample(conversation_ids: Iterable[str], percent: int, algo: str = "md5") -> set[str]:
    """Hash-based subsample, stable across chunk boundaries, library versions, machines.

    Hash the ROOT id, never the tweet id: tweet-level sampling leaks turns of the
    same thread across train/eval (docs/05 decision 9). Deliberately not
    df.sample(seed=) -- that is not stable across the above.

    Concretely: a seeded shuffle depends on the row ORDER it is handed, so reading
    the CSV in a different chunk size changes the sample. Hashing the id depends
    only on the id.
    """
    if not 0 < percent <= 100:
        raise ValueError("percent must be in (0, 100]")
    hasher = {"md5": hashlib.md5, "sha256": hashlib.sha256}[algo]
    threshold = percent / 100.0
    selected = set()
    for cid in conversation_ids:
        digest = hasher(str(cid).encode("utf-8")).hexdigest()
        # First 8 hex digits as a uniform draw on [0, 1).
        if int(digest[:8], 16) / 0xFFFFFFFF < threshold:
            selected.add(str(cid))
    return selected


def rejoin(sample_ids_path: Path, golden_path: Path, raw_csv: Path):
    """Rehydrate tweet text by joining committed tweet_ids against the user's twcs.csv.

    We commit tweet_id + labels, never tweet text -- following the TweetSumm
    precedent (Findings of EMNLP 2021), which is the defensible answer on
    redistribution (docs/01 §7). Raises with a clear message if raw_csv is absent.
    """
    import polars as pl

    raw_csv = Path(raw_csv)
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"{raw_csv} not found.\n"
            f"The golden set commits tweet_ids and labels, not tweet text -- the "
            f"corpus is redistributed under Kaggle's terms, so rehydration is the "
            f"reviewer's step (TweetSumm precedent, docs/01 §7).\n"
            f"  1. Download thoughtvector/customer-support-on-twitter from Kaggle\n"
            f"  2. Place twcs.csv at {raw_csv}\n"
            f"  3. Re-run `make rejoin`\n"
            f"`make eval` (Tier 1) reproduces every headline number WITHOUT this file."
        )

    raw = load_raw(raw_csv).select(["tweet_id", "text", "author_id", "created_at"]).collect()
    wanted = [line.strip() for line in Path(sample_ids_path).read_text().splitlines() if line.strip()]
    golden = pl.read_csv(golden_path) if Path(golden_path).exists() else None

    hydrated = raw.filter(pl.col("tweet_id").is_in(wanted))
    missing = set(wanted) - set(hydrated["tweet_id"].to_list())
    if missing:
        print(
            f"WARNING: {len(missing)}/{len(wanted)} committed tweet_ids are absent "
            f"from this copy of twcs.csv. Kaggle has reissued the file before; "
            f"report the coverage rather than silently evaluating on a subset."
        )

    if golden is not None:
        hydrated = golden.join(
            hydrated, left_on="message_id", right_on="tweet_id", how="left"
        )
    return hydrated


def main() -> int:
    import argparse

    from src import REPO_ROOT, load_config

    parser = argparse.ArgumentParser(prog="src.build_sample")
    parser.add_argument("--rejoin", action="store_true")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)

    if args.rejoin:
        frame = rejoin(
            REPO_ROOT / config.corpus["sample_ids"],
            REPO_ROOT / config.evaluation["golden_set"],
            REPO_ROOT / config.corpus["raw_csv"],
        )
        print(f"rehydrated {len(frame)} rows")
        return 0

    raw = load_raw(REPO_ROOT / config.corpus["raw_csv"])
    threads = build_threads(raw, config.brand["handle"])
    keep = deterministic_sample(
        threads["conversation_id"].to_list(),
        config.corpus["sample_percent"],
        config.corpus["sample_hash"],
    )
    import polars as pl
    threads = threads.filter(pl.col("conversation_id").is_in(list(keep)))
    if config.corpus["starters_only"]:
        threads = threads.filter(pl.col("n_turns") >= config.corpus["min_thread_turns"])

    out = REPO_ROOT / config.corpus["threads_parquet"]
    out.parent.mkdir(parents=True, exist_ok=True)
    threads.write_parquet(out)
    print(f"wrote {len(threads)} threads -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
