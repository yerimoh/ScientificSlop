"""Instance grounding, figure surface v0: per figure, two cached VLM calls.
 gate  : is this a schematic/method diagram or flow of the paper's approach?
 quote : does the figure carry a concrete worked example (actual input/output text,
         quoted sentence, specific processed values, not axis numbers)? Quote it.
Confirmed example = VLM says yes AND the quoted text carries a deterministic fingerprint
(sentence with a verb and >=6 words, quoted string, or a specific non-axis value).
Paper score (diagram-bearing papers only): slop 1 = no confirmed example in any diagram.
Resumable: one JSON per figure in cache/."""
import json, os, re, sys, glob
ROOT = os.environ.get("SCISLOP_ROOT", ".")
B = f"{ROOT}/paper/draft_v6/scislopbench/bench165"
FS = f"{B}/results/slop/fig_specimen"
sys.path.insert(0, f"{ROOT}/paper/draft_v6/slop/Artifacts/fig_graph/roi_extract")
from PIL import Image
import vision_calls as V

GATE = ("Is this figure a schematic diagram of a method, system, or workflow (boxes, arrows, "
        "pipeline stages), rather than a results plot, photo, table, or screenshot-only image? "
        'Answer JSON: {"diagram": true/false}')
QUOTE = ("Does this diagram contain a concrete worked example, meaning actual input or output "
         "text, a quoted sentence, a named real object, or specific values being processed "
         "(not axis labels or tick numbers)? If yes, transcribe the example text exactly. "
         'Answer JSON: {"example": true/false, "quote": "..."}')

def fingerprint(q):
    if not q or not isinstance(q, str): return False
    if re.search(r'["“‘\'].{6,}["”’\']', q): return True
    words = re.findall(r"[A-Za-z][A-Za-z']+", q)
    if len(words) >= 6 and re.search(r"\b(is|are|was|put|take|search|answer|think|click|go|has|do|does|make|find|buy|open)\b", q.lower()):
        return True
    if re.search(r"\b\d+\.\d+\b", q) and len(words) >= 3: return True
    if len(words) >= 4 and any(w[0].isupper() for w in words[1:]): return True
    return False

man = [json.loads(l) for l in open(f"{FS}/manifest.jsonl") if json.loads(l).get("png")]
print("figures:", len(man), flush=True)
for i, m in enumerate(man):
    cp = f"{FS}/cache/{m['png']}.json"
    if os.path.exists(cp): continue
    try:
        img = Image.open(f"{FS}/figs/{m['png']}").convert("RGB")
        if max(img.size) > 1600:
            r = 1600 / max(img.size); img = img.resize((int(img.width*r), int(img.height*r)))
        g = V.ask(img, GATE, max_new=30)
        out = {"gate": bool(g.get("diagram"))}
        if out["gate"]:
            q = V.ask(img, QUOTE, max_new=200)
            out["example_vlm"] = bool(q.get("example")); out["quote"] = q.get("quote", "")
            out["example_confirmed"] = out["example_vlm"] and fingerprint(out["quote"])
        json.dump(out, open(cp, "w"))
    except Exception as e:
        json.dump({"err": repr(e)[:80]}, open(cp, "w"))
    if (i+1) % 25 == 0: print(i+1, "done", flush=True)

# score whatever is cached
rows = {}
for m in man:
    cp = f"{FS}/cache/{m['png']}.json"
    if not os.path.exists(cp): continue
    c = json.load(open(cp))
    r = rows.setdefault(m["item_id"], {"n_figs": 0, "n_diagrams": 0, "n_example": 0})
    r["n_figs"] += 1
    if c.get("gate"): r["n_diagrams"] += 1
    if c.get("example_confirmed"): r["n_example"] += 1
with open(f"{FS}/papers.jsonl", "w") as w:
    for iid, r in rows.items():
        slop = None if r["n_diagrams"] == 0 else (1.0 if r["n_example"] == 0 else 0.0)
        w.write(json.dumps({"corpus": "AI" if iid.startswith("AI_") else "HU",
                            "id": iid.split("_", 1)[1], "item_id": iid, **r, "slop_score": slop}) + "\n")
print("scored papers:", len(rows))
