"""Draft labels for the golden-set candidates, for a human to confirm or correct.

Usage:
    python scripts/prelabel.py data/golden_200_candidates.json

Writes data/golden_200_prelabels.json: the candidates with prelabel_intent,
prelabel_action and prelabel_rationale added. build_labeling_tool.py embeds these
as the starting state of each row.

Why a DIFFERENT model from the system under test: pre-labels anchor the annotator.
If they came from the same model as the classifier being evaluated, every accepted
pre-label would be a point of agreement between the model and "gold" that the model
itself put there. Using a larger model from a different tier limits that; the
report still discloses pre-labelling and the human override rate.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import load_taxonomy  # noqa: E402
from src.llm_client import complete  # noqa: E402

MODEL = {"provider": "openai", "id": "gpt-5.4", "reasoning_effort": "low", "max_tokens": 2000}

SYSTEM = """You label customer tweets sent to @SpotifyCares for an evaluation golden set.
Label the OPENING customer tweet only. Pick exactly one intent using the definitions,
the not_this boundaries and the precedence rule. Then decide the gold action:

auto_handle: a public reply that follows Spotify's usual help (troubleshooting steps,
  how-to, catalogue/licensing explanation, acknowledging feedback) would be an
  acceptable first response, and no account lookup, money decision or human judgement
  is needed first.
escalate: account security or access, any money/refund/charge issue, anything needing
  the customer's account to investigate, threats/legal/safety, pure complaints
  demanding a human, DM pointers, unclear or non-English requests.
For non_support use escalate (it is filtered, never replied to)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "action": {"type": "string", "enum": ["auto_handle", "escalate"]},
        "rationale": {"type": "string"},
    },
}


def menu() -> str:
    tax = load_taxonomy()
    lines = [f"Precedence rule: {tax['precedence_rule'].strip()}", "", "Intents:"]
    for i in tax["intents"]:
        lines.append(f"- {i['name']}: {i['definition'].strip()}")
        for n in i.get("not_this") or []:
            lines.append(f"    not_this: {n}")
    return "\n".join(lines)


def label_one(candidate: dict, menu_text: str, names: set[str]) -> dict:
    prompt = f"{menu_text}\n\nTweet: {candidate['customer_text']}\n\nReturn JSON with intent, action, rationale (one short sentence)."
    for _ in range(3):
        try:
            raw, _ = complete(MODEL, prompt, system=SYSTEM, json_schema=SCHEMA)
            out = json.loads(raw)
            if out["intent"] in names:
                return {**candidate, "prelabel_intent": out["intent"],
                        "prelabel_action": out["action"], "prelabel_rationale": out["rationale"]}
        except Exception as exc:  # noqa: BLE001 - retried, then left blank for the human
            print(f"[{candidate['id']}] {exc}")
    return {**candidate, "prelabel_intent": "", "prelabel_action": "", "prelabel_rationale": ""}


def main() -> int:
    src = Path(sys.argv[1])
    out_path = src.with_name(src.stem.replace("_candidates", "") + "_prelabels.json")
    candidates = json.loads(src.read_text(encoding="utf-8"))
    menu_text = menu()
    names = {i["name"] for i in load_taxonomy()["intents"]}
    with ThreadPoolExecutor(max_workers=8) as pool:
        labelled = list(pool.map(lambda c: label_one(c, menu_text, names), candidates))
    out_path.write_text(json.dumps(labelled, ensure_ascii=False, indent=1), encoding="utf-8")
    blank = sum(1 for r in labelled if not r["prelabel_intent"])
    print(f"wrote {out_path} ({len(labelled)} rows, {blank} left blank)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
