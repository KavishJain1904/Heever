"""Generate a self-contained local HTML labelling tool from a candidates JSON file.

Usage:
    python scripts/build_labeling_tool.py data/golden_pilot_30.json pilot label_pilot_30.html
    python scripts/build_labeling_tool.py data/golden_200_candidates.json v2 label_golden_200.html

Output is a single .html file with the candidate tweets embedded inline (no
server, no fetch/CORS issue -- open it directly in a browser via file://).
Labels autosave to localStorage as you go and export as a CSV matching the
docs/11 §3 schema for data/golden_200.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import intent_names  # noqa: E402

# Read from the frozen taxonomy so the tool can never offer a stale label set.
INTENTS = intent_names()
KEYS = list("1234567890-=[]")

TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Heever golden-set labelling -- {guideline_version}</title>
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
  .tweet {{ font-size: 20px; line-height: 1.45; margin-bottom: 6px; }}
  .meta {{ font-size: 12px; color: #767c86; margin-bottom: 16px; }}
  .stratum {{ display: inline-block; padding: 2px 8px; border-radius: 4px; background: #23262e;
             font-size: 11px; text-transform: uppercase; letter-spacing: .04em; margin-right: 8px; }}
  details {{ margin: 10px 0 18px; }}
  summary {{ cursor: pointer; color: #9aa0a6; font-size: 13px; }}
  .thread {{ margin-top: 10px; font-size: 14px; }}
  .turn {{ padding: 8px 10px; border-radius: 6px; margin-bottom: 6px; }}
  .turn.brand {{ background: #16242f; border-left: 3px solid #4c8bf5; }}
  .turn.customer {{ background: #201f26; border-left: 3px solid #767c86; }}
  .turn .who {{ font-size: 11px; color: #767c86; margin-bottom: 2px; }}
  fieldset {{ border: 1px solid #262a33; border-radius: 8px; margin: 14px 0; padding: 12px; }}
  legend {{ font-size: 12px; color: #9aa0a6; padding: 0 6px; }}
  .btnrow {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  button.opt {{ background: #1c2028; border: 1px solid #2c313c; color: #e8e8ea; border-radius: 6px;
               padding: 8px 12px; font-size: 13px; cursor: pointer; }}
  button.opt kbd {{ opacity: .55; margin-right: 6px; font-size: 11px; }}
  button.opt.selected {{ background: #2b4a86; border-color: #4c8bf5; }}
  button.opt.selected.escalate {{ background: #7a2b2b; border-color: #e05c5c; }}
  button.opt.selected.auto {{ background: #235c3a; border-color: #4cd07d; }}
  input[type=text] {{ width: 100%; background: #12141a; border: 1px solid #262a33; color: #e8e8ea;
                       border-radius: 6px; padding: 8px; font-size: 13px; }}
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
</style>
</head>
<body>
<div class="wrap">
  <div class="progress">
    <span id="posLabel"></span>
    <span id="labeledCount"></span>
    <span class="savedflag" id="savedFlag"></span>
  </div>
  <div class="bar"><div class="bar-fill" id="barFill"></div></div>

  <div class="card">
    <div class="meta"><span class="stratum" id="stratumTag"></span><span id="idTag"></span></div>
    <div class="tweet" id="tweetText"></div>
    <div class="meta" id="prelabelBox"></div>
    <details>
      <summary>Show full thread (context only -- label the opening tweet)</summary>
      <div class="thread" id="threadBox"></div>
    </details>

    <fieldset>
      <legend>intent -- keys 1-9, 0, -, =, [, ]</legend>
      <div class="btnrow" id="intentButtons"></div>
    </fieldset>

    <fieldset>
      <legend>auto_handle_vs_escalate -- a / e</legend>
      <div class="btnrow">
        <button class="opt" data-field="action" data-value="auto_handle" onclick="setField('action','auto_handle')">
          <kbd>a</kbd>auto_handle</button>
        <button class="opt" data-field="action" data-value="escalate" onclick="setField('action','escalate')">
          <kbd>e</kbd>escalate</button>
      </div>
    </fieldset>

    <fieldset>
      <legend>confidence -- h / m / l</legend>
      <div class="btnrow">
        <button class="opt" data-field="confidence" data-value="high" onclick="setField('confidence','high')"><kbd>h</kbd>high</button>
        <button class="opt" data-field="confidence" data-value="med" onclick="setField('confidence','med')"><kbd>m</kbd>med</button>
        <button class="opt" data-field="confidence" data-value="low" onclick="setField('confidence','low')"><kbd>l</kbd>low</button>
      </div>
    </fieldset>

    <fieldset>
      <legend>ambiguous (>1 intent genuinely applied) -- g</legend>
      <div class="btnrow">
        <button class="opt" data-field="ambiguous" data-value="true" onclick="toggleAmbiguous()"><kbd>g</kbd>toggle</button>
        <span id="ambiguousState" style="align-self:center;font-size:13px;color:#9aa0a6;"></span>
      </div>
    </fieldset>

    <fieldset>
      <legend>quality_of_historical_reply (1-5, leave blank if brand never replied) -- q then 1-5</legend>
      <div class="btnrow" id="qualityButtons">
        <button class="opt" onclick="setQuality(1)">1</button>
        <button class="opt" onclick="setQuality(2)">2</button>
        <button class="opt" onclick="setQuality(3)">3</button>
        <button class="opt" onclick="setQuality(4)">4</button>
        <button class="opt" onclick="setQuality(5)">5</button>
        <button class="opt" onclick="setQuality(null)">clear</button>
      </div>
    </fieldset>

    <fieldset>
      <legend>note (optional -- adjudication reasoning for hard cases)</legend>
      <input type="text" id="noteInput" oninput="setNote(this.value)" placeholder="free text, no verbatim tweet text needed">
    </fieldset>

    <div class="navrow">
      <button class="secondary" onclick="prevRow()" id="prevBtn">&larr; prev</button>
      <button onclick="nextRow()" id="nextBtn">next &rarr; (space)</button>
    </div>
  </div>

  <div class="exportbar">
    <button onclick="exportCsv()">Export CSV</button>
    <button onclick="exportPartial()">Export CSV (labelled rows only)</button>
    <button onclick="if(confirm('Clear all labels for this batch?')) resetAll()">Clear all</button>
  </div>

  <div class="hints">
    Autosaves to this browser's localStorage after every change (key: heever_labels_{storage_key}).
    Closing the tab is safe. "Export CSV" any time -- re-exporting overwrites nothing on disk, you
    choose where to save the download.<br>
    Guideline version recorded on every row: <b>{guideline_version}</b>. Label date: today's date, UTC.
  </div>
</div>

<script>
const CANDIDATES = {candidates_json};
const INTENTS = {intents_json};
const KEYS = {keys_json};
const GUIDELINE_VERSION = {guideline_version_json};
const STORAGE_KEY = "heever_labels_{storage_key}";

// Storage can be unavailable (data: URLs, private windows). The tool must still work;
// the page then warns that labels live only in this tab until exported.
let storageOk = true;
let labels = {{}};
try {{ labels = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}"); }}
catch (e) {{ storageOk = false; }}
let pos = 0;

function currentId() {{ return CANDIDATES[pos].id; }}
function currentLabel() {{
  const id = currentId();
  if (!labels[id]) labels[id] = {{}};
  const l = labels[id], c = CANDIDATES[pos];
  // Pre-labels fill intent/action ONCE. Confidence is never pre-filled, so every
  // row needs at least one deliberate keypress before it counts as labelled.
  if (!l._init) {{
    if (c.prelabel_intent) l.intent = c.prelabel_intent;
    if (c.prelabel_action) l.action = c.prelabel_action;
    l._init = true;
  }}
  return l;
}}
function isDone(l) {{ return l && l.reviewed && l.intent && l.action && l.confidence; }}

function save() {{
  const f = document.getElementById("savedFlag");
  try {{
    localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
    f.textContent = "saved " + new Date().toLocaleTimeString();
  }} catch (e) {{
    storageOk = false;
    f.style.color = "#e05c5c";
    f.textContent = "NOT autosaving in this browser -- export before closing";
  }}
}}

function setField(field, value) {{
  currentLabel()[field] = value;
  save();
  render();
}}

function toggleAmbiguous() {{
  const l = currentLabel();
  l.ambiguous = !l.ambiguous;
  save();
  render();
}}

function setQuality(v) {{
  currentLabel().quality = v;
  save();
  render();
}}

function setNote(v) {{
  currentLabel().note = v;
  save();
}}

function setIntent(name) {{
  currentLabel().intent = name;
  save();
  render();
}}

function render() {{
  const c = CANDIDATES[pos];
  const l = currentLabel();
  document.getElementById("posLabel").textContent = "example " + (pos + 1) + " / " + CANDIDATES.length;
  const labelledCount = Object.values(labels).filter(isDone).length;
  document.getElementById("prelabelBox").textContent = c.prelabel_intent
    ? "pre-label: " + c.prelabel_intent + " / " + c.prelabel_action + " -- " + (c.prelabel_rationale || "")
    : "no pre-label";
  document.getElementById("labeledCount").textContent = labelledCount + " fully labelled";
  document.getElementById("barFill").style.width = (100 * (pos + 1) / CANDIDATES.length) + "%";
  document.getElementById("stratumTag").textContent = c.stratum;
  document.getElementById("idTag").textContent = c.id + " (tweet " + c.tweet_id + ")";
  document.getElementById("tweetText").textContent = c.customer_text;

  const threadBox = document.getElementById("threadBox");
  threadBox.innerHTML = "";
  c.thread.forEach(t => {{
    const div = document.createElement("div");
    div.className = "turn " + (t.is_brand ? "brand" : "customer");
    const who = document.createElement("div");
    who.className = "who";
    who.textContent = t.is_brand ? "SpotifyCares" : "customer";
    const body = document.createElement("div");
    body.textContent = t.text;
    div.appendChild(who); div.appendChild(body);
    threadBox.appendChild(div);
  }});
  if (!c.has_brand_reply) {{
    const div = document.createElement("div");
    div.style.color = "#767c86";
    div.style.fontSize = "12px";
    div.textContent = "(no brand reply in this thread -- quality_of_historical_reply stays blank)";
    threadBox.appendChild(div);
  }}

  const ib = document.getElementById("intentButtons");
  ib.innerHTML = "";
  INTENTS.forEach((name, i) => {{
    const key = KEYS[i] || "";
    const btn = document.createElement("button");
    btn.className = "opt" + (l.intent === name ? " selected" : "");
    btn.innerHTML = "<kbd>" + key + "</kbd>" + name;
    btn.onclick = () => setIntent(name);
    ib.appendChild(btn);
  }});

  document.querySelectorAll('[data-field="action"]').forEach(b => {{
    b.classList.toggle("selected", l.action === b.dataset.value);
    b.classList.toggle("auto", l.action === "auto_handle" && b.dataset.value === "auto_handle");
    b.classList.toggle("escalate", l.action === "escalate" && b.dataset.value === "escalate");
  }});
  document.querySelectorAll('[data-field="confidence"]').forEach(b => {{
    b.classList.toggle("selected", l.confidence === b.dataset.value);
  }});
  document.getElementById("ambiguousState").textContent = l.ambiguous ? "ambiguous: true" : "ambiguous: false";
  document.querySelectorAll("#qualityButtons button").forEach((b, i) => {{
    const v = i < 5 ? i + 1 : null;
    b.classList.toggle("selected", l.quality === v && v !== null);
  }});
  document.getElementById("noteInput").value = l.note || "";
  document.getElementById("prevBtn").disabled = pos === 0;
}}

function nextRow() {{
  // Moving past a row is the "I have looked at this" signal.
  const l = currentLabel();
  if (l.confidence) {{ l.reviewed = true; save(); }}
  if (pos < CANDIDATES.length - 1) {{ pos++; }}
  render();
}}
function prevRow() {{ if (pos > 0) {{ pos--; render(); }} }}

function csvEscape(v) {{
  if (v === null || v === undefined) return "";
  const s = String(v);
  if (/[",\\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}}

function buildCsv(onlyLabelled) {{
  const header = ["id","tweet_id","conversation_id","stratum","intent",
    "auto_handle_vs_escalate","quality_of_historical_reply","confidence",
    "ambiguous","note","guideline_version","label_date",
    "prelabel_intent","prelabel_action","reviewed"];
  const rows = [header.join(",")];
  const today = new Date().toISOString().slice(0,10);
  CANDIDATES.forEach(c => {{
    const l = labels[c.id] || {{}};
    if (onlyLabelled && !isDone(l)) return;
    rows.push([
      c.id, c.tweet_id, c.conversation_id, c.stratum,
      l.intent || "", l.action || "", l.quality ?? "", l.confidence || "",
      l.ambiguous ? "true" : "false", l.note || "", GUIDELINE_VERSION, today,
      c.prelabel_intent || "", c.prelabel_action || "", l.reviewed ? "true" : "false"
    ].map(csvEscape).join(","));
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

function exportCsv() {{ download("{export_name}", buildCsv(false)); }}
function exportPartial() {{ download("{export_name}".replace(".csv","_partial.csv"), buildCsv(true)); }}
function resetAll() {{ labels = {{}}; save(); render(); }}

document.addEventListener("keydown", (e) => {{
  if (e.target.tagName === "INPUT") {{
    if (e.key === "Enter") {{ e.target.blur(); }}
    return;
  }}
  const k = e.key;
  const idx = KEYS.indexOf(k);
  if (idx >= 0 && idx < INTENTS.length) {{ setIntent(INTENTS[idx]); return; }}
  if (k === "a") setField("action", "auto_handle");
  else if (k === "e") setField("action", "escalate");
  else if (k === "h") setField("confidence", "high");
  else if (k === "m") setField("confidence", "med");
  else if (k === "l") setField("confidence", "low");
  else if (k === "g") toggleAmbiguous();
  else if (k === " ") {{ e.preventDefault(); nextRow(); }}
  else if (k === "Backspace") prevRow();
}});

render();
</script>
</body>
</html>
"""


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: python scripts/build_labeling_tool.py <candidates.json> <guideline_version> <out.html>")
        return 1
    candidates_path, guideline_version, out_name = sys.argv[1], sys.argv[2], sys.argv[3]
    candidates = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    storage_key = Path(candidates_path).stem
    out_path = Path(candidates_path).resolve().parents[1] / out_name

    html = TEMPLATE.format(
        candidates_json=json.dumps(candidates, ensure_ascii=False),
        intents_json=json.dumps(INTENTS),
        keys_json=json.dumps(KEYS),
        guideline_version_json=json.dumps(guideline_version),
        guideline_version=guideline_version,
        storage_key=storage_key,
        export_name=out_name.replace(".html", ".csv"),
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} ({len(candidates)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
