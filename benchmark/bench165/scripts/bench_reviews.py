"""Run one reviewer over bench165 items (default: human side only; the FARS side reuses the
archived 165-run R1 reviews). Same backbones and output schema as bench5."""
import os, sys, json, argparse, time, traceback

ROOT = os.environ.get("SCISLOP_ROOT", ".")
BB = f"{ROOT}/artifact-ai2science/Evaluation/02_baselines_B"
B = f"{ROOT}/paper/draft_v6/scislopbench/bench165"
sys.path.insert(0, f"{ROOT}/artifact-ai2science/_llm")

ap = argparse.ArgumentParser()
ap.add_argument("--system", required=True, choices=["b3a", "b3i", "b2h"])
ap.add_argument("--label", type=int, default=0, help="0 = human side only (default), -1 = all")
a = ap.parse_args()

if a.system == "b3a":
    sys.path.insert(0, f"{BB}/B3a_ai_scientist"); import sakana_reviewer as M
elif a.system == "b3i":
    sys.path.insert(0, f"{BB}/B3i_cmu"); import cmu_reviewer as M
else:
    sys.path.insert(0, f"{BB}/B2h_cyclereviewer"); import cyclerev_reviewer as M
    ep = os.environ.get("CYCLEREV_ENDPOINT")
    if ep:
        M.endpoint = lambda: ep.rstrip("/")

items = json.load(open(f"{B}/items165.json"))["items"]
if a.label != -1:
    items = [i for i in items if i["label"] == a.label]
outdir = f"{B}/results/reviews/{a.system}"
os.makedirs(outdir, exist_ok=True)

for n, it in enumerate(items):
    op = f"{outdir}/{it['item_id']}.json"
    if os.path.exists(op) and os.path.getsize(op) > 50:
        continue
    text = open(f"{B}/views/{it['item_id']}/body.tex", encoding="utf-8", errors="ignore").read()
    t0 = time.time()
    try:
        out = M.perform_review(text)
        rev, info = (out if isinstance(out, tuple) else (out, {}))
    except Exception as e:
        rev, info = None, {"error": str(e)[:300], "trace": traceback.format_exc()[-600:]}
    json.dump({"item_id": it["item_id"], "system": a.system, "final": rev, "info": info,
               "n_chars_in": len(text), "seconds": round(time.time() - t0, 1)},
              open(op, "w"), indent=1, ensure_ascii=False)
    print(f"[{n+1}/{len(items)}]", it["item_id"], round(time.time() - t0, 1), "s",
          "OK" if rev else f"FAIL {str(info)[:120]}", flush=True)
print("ALL DONE", a.system)
