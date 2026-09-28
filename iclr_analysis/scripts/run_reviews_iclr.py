"""Run one reviewer (b3a AI-Scientist or b2h CycleReviewer) over the ICLR reviewer subset
(records/reviewer_subset.json), same modules and output schema as bench165/scripts/bench_reviews.py."""
import os, sys, json, argparse, time, traceback
ROOT = os.environ.get("SCISLOP_ROOT", ".")
BB = f"{ROOT}/artifact-ai2science/Evaluation/02_baselines_B"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/artifact-ai2science/_llm")
ap = argparse.ArgumentParser(); ap.add_argument("--system", required=True, choices=["b3a", "b2h"]); ap.add_argument("--reverse", action="store_true", help="walk the subset backwards (second worker meets the first in the middle)")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--subset", default="reviewer_subset.json", help="file under records/ with an ids list")
ap.add_argument("--views", default="views", help="view directory under records/ (views2026 for the single-year corpus)")
ap.add_argument("--outdir", default="", help="override the output directory name under results/reviews/")
a = ap.parse_args()
if a.system == "b3a":
    sys.path.insert(0, f"{BB}/B3a_ai_scientist"); import sakana_reviewer as M
else:
    sys.path.insert(0, f"{BB}/B2h_cyclereviewer"); import cyclerev_reviewer as M
    ep = os.environ.get("CYCLEREV_ENDPOINT")
    if ep: M.endpoint = lambda: ep.rstrip("/")
ids = json.load(open(f"{HERE}/records/{a.subset}"))["ids"]
if a.reverse: ids = ids[::-1]
if a.nshard > 1: ids = ids[a.shard::a.nshard]
outdir = f"{HERE}/results/reviews/{a.outdir or a.system}"; os.makedirs(outdir, exist_ok=True)
for n, pid in enumerate(ids):
    op = f"{outdir}/{pid}.json"
    if os.path.exists(op):
        try:
            prev = json.load(open(op)); fin = prev.get("final") or {}
            done = isinstance(fin, dict) and isinstance(fin.get("Overall", fin.get("Rating")), (int, float))
        except Exception:
            done = False
        if done:
            continue          # a failed record is retried, otherwise one broken server poisons the whole subset
    text = open(f"{HERE}/records/{a.views}/{pid}/body.tex", encoding="utf-8", errors="ignore").read()
    t0 = time.time()
    try:
        out = M.perform_review(text); rev, info = (out if isinstance(out, tuple) else (out, {}))
    except Exception as e:
        rev, info = None, {"error": str(e)[:300], "trace": traceback.format_exc()[-600:]}
    json.dump({"item_id": pid, "system": a.system, "final": rev, "info": info, "n_chars_in": len(text),
               "seconds": round(time.time() - t0, 1)}, open(op, "w"), indent=1, ensure_ascii=False)
    print(f"[{n+1}/{len(ids)}]", pid, round(time.time() - t0, 1), "s", "OK" if rev else f"FAIL {str(info)[:120]}", flush=True)
print("ALL DONE", a.system)
