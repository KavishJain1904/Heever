"""Generate a self-contained local HTML tool for blind human scoring of the 30
judge-validation drafts (evidence of judge-human agreement, docs/14 §7).

Usage:
    python scripts/build_judge_tool.py

Reads artifacts/results/judge_verdicts.jsonl (gitignored, full judge reasoning)
and selects the SAME 30 "heever" rows that cmd_judge_sheet in
scripts/run_experiment.py puts in data/judge_human_30.xlsx (identical
random.Random(42).sample over the same pool, sorted by message_id), so the two
routes cover exactly the same drafts.

Customer message text is not carried on the verdict row, so it is joined in from
data/golden_200_prelabels.json on message_id == id.

Output: judge_human_30.html at the repo root. Single file, no server, opens via
file://. Embeds ONLY message_id, customer_text and draft -- never the judge's
per-criterion pass/reasoning/quote for these rows, because the whole point of the
exercise is measuring whether a human independently agrees with the judge. Scoring
autosaves to localStorage and exports a CSV matching what
scripts/run_experiment.py's judge-import command expects.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import load_config  # noqa: E402
from src.judge import CRITERIA, CRITERION_TEXT  # noqa: E402

DATA = REPO_ROOT / "data"
RESULTS = REPO_ROOT / "artifacts" / "results"
CANDIDATES = DATA / "golden_200_prelabels.json"

TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Heever judge-human agreement -- 30 drafts</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 24px;
         background: #0f1115; color: #e8e8ea; }}
  .wrap {{ max-width: 880px; margin: 0 auto; }}
  .progress {{ font-size: 13px; color: #9aa0a6; margin-bottom: 12px; display: flex;
              justify-content: space-between; align-items: center; }}
  .bar {{ height: 6px; background: #23262e; border-radius: 3px; overflow: hidden; margin-bottom: 20px; }}
  .bar-fill {{ height: 100%; background: #4c8bf5; transition: width .15s; }}
  .card {{ background: #171a21; border: 1px solid #262a33; border-radius: 10px; padding: 20px; }}
  .tweet {{ font-size: 18px; line-height: 1.45; margin-bottom: 6px; }}
  .draft {{ font-size: 16px; line-height: 1.5; margin-top: 4px; }}
  .meta {{ font-size: 12px; color: #767c86; margin-bottom: 16px; }}
  .label {{ font-size: 12px; color: #9aa0a6; text-transform: uppercase; letter-spacing: .04em;
           margin: 14px 0 4px; }}
  .msgbox {{ background: #12141a; border: 1px solid #262a33; border-radius: 8px; padding: 12px; }}
  fieldset {{ border: 1px solid #262a33; border-radius: 8px; margin: 14px 0; padding: 12px; }}
  legend {{ font-size: 13px; color: #d6d8dc; padding: 0 6px; }}
  .crit-def {{ font-size: 12px; color: #9aa0a6; margin-bottom: 8px; }}
  .btnrow {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  button.opt {{ background: #1c2028; border: 1px solid #2c313c; color: #e8e8ea; border-radius: 6px;
               padding: 8px 12px; font-size: 13px; cursor: pointer; }}
  button.opt kbd {{ opacity: .55; margin-right: 6px; font-size: 11px; }}
  button.opt.selected.pass {{ background: #235c3a; border-color: #4cd07d; }}
  button.opt.selected.fail {{ background: #7a2b2b; border-color: #e05c5c; }}
  .navrow {{ display: flex; justify-content: space-between; align-items: center; margin-top: 18px; }}
  .navrow button {{ background: #4c8bf5; border: none; color: white; padding: 10px 18px;
                    border-radius: 6px; font-size: 14px; cursor: pointer; }}
  .navrow button.secondary {{ background: #23262e; color: #e8e8ea; }}
  .navrow button:disabled {{ opacity: .4; cursor: not-allowed; }}
  .hints {{ font-size: 11px; color: #5c626c; margin-top: 20px; line-height: 1.7; }}
  .exportbar {{ margin-top: 16px; display: flex; gap: 10px; }}
  .exportbar button {{ background: #23262e; color: #e8e8ea; border: 1px solid #2c313c;
                       border-radius: 6px; padding: 8px 14px; font-size: 13px; cursor: pointer; }}
  .savedflag {{ font-size: 12px; color: #4cd07d; margin-left: 10px; }}
  .blindnote {{ font-size: 11px; color: #e0b84c; margin-bottom: 10px; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="progress">
    <span id="posLabel"></span>
    <span id="scoredCount"></span>
    <span class="savedflag" id="savedFlag"></span>
  </div>
  <div class="bar"><div class="bar-fill" id="barFill"></div></div>

  <div class="blindnote">
    Blind scoring: which system produced this draft is intentionally hidden, and the
    judge model's own verdicts are never shown on this screen.
  </div>

  <div class="card">
    <div class="meta"><span id="idTag"></span></div>

    <div class="label">Customer message</div>
    <div class="msgbox tweet" id="tweetText"></div>

    <div class="label">Drafted reply under review</div>
    <div class="msgbox draft" id="draftText"></div>

    <div id="criteriaBox"></div>

    <div class="navrow">
      <button class="secondary" onclick="prevRow()" id="prevBtn">&larr; prev</button>
      <button onclick="nextRow()" id="nextBtn">next &rarr; (space)</button>
    </div>
  </div>

  <div class="exportbar">
    <button onclick="exportCsv()">Export CSV (scored rows only)</button>
    <button onclick="if(confirm('Clear all scores?')) resetAll()">Clear all</button>
  </div>

  <div class="hints">
    Autosaves to this browser's localStorage after every change (key: {storage_key_js}).
    Closing the tab is safe.<br>
    Keyboard: <b>1</b>-<b>6</b> selects a criterion, then <b>y</b>/<b>n</b> scores pass/fail
    (also <b>j</b>/<b>k</b> to move the selected criterion up/down, <b>space</b> for next row,
    backspace for previous). Click works the same as the buttons.<br>
    A row counts as scored once all six criteria are answered. Export produces a CSV with
    columns message_id and one column per criterion (yes/no), matching
    <code>scripts/run_experiment.py judge-import</code>.
  </div>
</div>

<script>
const ROWS = {rows_json};
const CRITERIA = {criteria_json};
const CRITERION_TEXT = {criterion_text_json};
const STORAGE_KEY = {storage_key_js};

let storageOk = true;
let scores = {{}};
try {{ scores = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}"); }}
catch (e) {{ storageOk = false; }}
let pos = 0;
let selectedCriterion = 0;

function currentId() {{ return ROWS[pos].message_id; }}
function currentScore() {{
  const id = currentId();
  if (!scores[id]) scores[id] = {{}};
  return scores[id];
}}
function isDone(s) {{ return s && CRITERIA.every(c => s[c] === "yes" || s[c] === "no"); }}

function save() {{
  const f = document.getElementById("savedFlag");
  try {{
    localStorage.setItem(STORAGE_KEY, JSON.stringify(scores));
    f.textContent = "saved " + new Date().toLocaleTimeString();
  }} catch (e) {{
    storageOk = false;
    f.style.color = "#e05c5c";
    f.textContent = "NOT autosaving in this browser -- export before closing";
  }}
}}

function setCriterion(name, value) {{
  currentScore()[name] = value;
  save();
  render();
}}

function render() {{
  const r = ROWS[pos];
  const s = currentScore();
  document.getElementById("posLabel").textContent = "row " + (pos + 1) + " / " + ROWS.length;
  const scoredCount = Object.values(scores).filter(isDone).length;
  document.getElementById("scoredCount").textContent = scoredCount + " / " + ROWS.length + " scored";
  document.getElementById("barFill").style.width = (100 * (pos + 1) / ROWS.length) + "%";
  document.getElementById("idTag").textContent = r.message_id;
  document.getElementById("tweetText").textContent = r.customer_text;
  document.getElementById("draftText").textContent = r.draft;

  const box = document.getElementById("criteriaBox");
  box.innerHTML = "";
  CRITERIA.forEach((name, i) => {{
    const fs = document.createElement("fieldset");
    if (i === selectedCriterion) fs.style.borderColor = "#4c8bf5";
    const legend = document.createElement("legend");
    legend.textContent = (i + 1) + ". " + name;
    fs.appendChild(legend);
    const def = document.createElement("div");
    def.className = "crit-def";
    def.textContent = CRITERION_TEXT[name];
    fs.appendChild(def);
    const row = document.createElement("div");
    row.className = "btnrow";
    const yes = document.createElement("button");
    yes.className = "opt" + (s[name] === "yes" ? " selected pass" : "");
    yes.innerHTML = "<kbd>y</kbd>pass";
    yes.onclick = () => {{ selectedCriterion = i; setCriterion(name, "yes"); }};
    const no = document.createElement("button");
    no.className = "opt" + (s[name] === "no" ? " selected fail" : "");
    no.innerHTML = "<kbd>n</kbd>fail";
    no.onclick = () => {{ selectedCriterion = i; setCriterion(name, "no"); }};
    row.appendChild(yes); row.appendChild(no);
    fs.appendChild(row);
    fs.onclick = () => {{ selectedCriterion = i; render(); }};
    box.appendChild(fs);
  }});
  document.getElementById("prevBtn").disabled = pos === 0;
}}

function nextRow() {{ if (pos < ROWS.length - 1) {{ pos++; selectedCriterion = 0; }} render(); }}
function prevRow() {{ if (pos > 0) {{ pos--; selectedCriterion = 0; }} render(); }}

function csvEscape(v) {{
  if (v === null || v === undefined) return "";
  const s = String(v);
  if (/[",\\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}}

function buildCsv() {{
  const header = ["message_id", ...CRITERIA];
  const rows = [header.join(",")];
  ROWS.forEach(r => {{
    const s = scores[r.message_id] || {{}};
    if (!isDone(s)) return;
    rows.push([r.message_id, ...CRITERIA.map(c => s[c])].map(csvEscape).join(","));
  }});
  return rows.join("\\n");
}}

function download(filename, text) {{
  const blob = new Blob([text], {{type: "text/csv"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}}

function exportCsv() {{ download("judge_human_30.csv", buildCsv()); }}
function resetAll() {{ scores = {{}}; save(); render(); }}

document.addEventListener("keydown", (e) => {{
  const k = e.key;
  if (k >= "1" && k <= "6") {{ selectedCriterion = parseInt(k, 10) - 1; render(); return; }}
  if (k === "y") setCriterion(CRITERIA[selectedCriterion], "yes");
  else if (k === "n") setCriterion(CRITERIA[selectedCriterion], "no");
  else if (k === "j") {{ selectedCriterion = Math.min(CRITERIA.length - 1, selectedCriterion + 1); render(); }}
  else if (k === "k") {{ selectedCriterion = Math.max(0, selectedCriterion - 1); render(); }}
  else if (k === " ") {{ e.preventDefault(); nextRow(); }}
  else if (k === "Backspace") prevRow();
}});

render();
</script>
</body>
</html>
"""


def _select_sample() -> list[dict]:
    """Identical selection to cmd_judge_sheet in scripts/run_experiment.py: the
    same random.Random(42).sample over the same "heever"-system pool, sorted by
    message_id, so the HTML and the xlsx route cover the same 30 drafts."""
    config = load_config()
    path = REPO_ROOT / config.evaluation["judge"]["verdicts_path_full"]
    verdicts = [json.loads(l) for l in open(path, encoding="utf-8")]
    pool = [v for v in verdicts if v["system"] == "heever"]
    sample = sorted(random.Random(42).sample(pool, min(30, len(pool))), key=lambda v: v["message_id"])
    return sample


def main() -> int:
    sample = _select_sample()
    cands = {c["id"]: c for c in json.loads(CANDIDATES.read_text(encoding="utf-8"))}

    rows = []
    for v in sample:
        c = cands.get(v["message_id"])
        if c is None:
            raise SystemExit(f"no golden_200_prelabels.json row for message_id {v['message_id']!r}")
        rows.append({
            "message_id": v["message_id"],
            "customer_text": c["customer_text"],
            "draft": v["draft"],
        })

    # Belt-and-suspenders: never let judge fields leak into the embedded payload,
    # even if a future edit to `rows` construction above starts copying from `v`.
    forbidden = {"pass", "reasoning", "quote", "criteria", "all_six_pass",
                 "position_bias_disagreements", "system"}
    for r in rows:
        assert not (forbidden & set(r.keys())), f"blind-scoring leak in row keys: {r.keys()}"

    out_path = REPO_ROOT / "judge_human_30.html"
    html = TEMPLATE.format(
        rows_json=json.dumps(rows, ensure_ascii=False),
        criteria_json=json.dumps(list(CRITERIA)),
        criterion_text_json=json.dumps(CRITERION_TEXT, ensure_ascii=False),
        storage_key_js=json.dumps("heever_judge_human_30"),
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} ({len(rows)} drafts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
