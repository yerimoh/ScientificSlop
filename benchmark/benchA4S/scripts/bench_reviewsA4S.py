"""Run one reviewer over benchA4S items. Unlike bench165 there is no archive to reuse, because no
reviewer has ever seen an Agents4Science submission, so both sides are reviewed here and --label
defaults to every item."""
import os, sys, json, argparse, time, traceback
from concurrent.futures import ThreadPoolExecutor

ROOT = os.environ.get("SCISLOP_ROOT", ".")
BB = f"{ROOT}/artifact-ai2science/Evaluation/02_baselines_B"
B = f"{ROOT}/paper/draft_v6/scislopbench/benchA4S"
sys.path.insert(0, f"{ROOT}/artifact-ai2science/_llm")

ap = argparse.ArgumentParser()
ap.add_argument("--system", required=True, choices=["b3a", "b3i", "b2h"])
ap.add_argument("--label", type=int, default=-1, help="-1 = both sides (default), 0 or 1 = one side")
# 494 reviews against one server is a day of wall clock. The reviews are independent, so the corpus
# is cut into disjoint shards and each shard gets its own allocation and its own server; disjoint
# means two shards never write the same file.
# Both reviewers sample, CycleReviewer at temperature 0.4 and AI-Scientist at 0.75 over an
# ensemble, so one run is one draw. --run keeps the draws apart instead of overwriting.
ap.add_argument("--run", type=int, default=0)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
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

# A reviewer can be taken out of a run that is already queued, which a submitted job's own script
# cannot be, by leaving results/reviews/<system>.skip in place.
if os.path.exists(f"{B}/results/reviews/{a.system}.skip"):
    print(f"SKIPPED {a.system} (skip file present)")
    raise SystemExit(0)

items = json.load(open(f"{B}/itemsA4S.json"))["items"]
if a.label != -1:
    items = [i for i in items if i["label"] == a.label]
if a.nshard > 1:
    items = [i for n, i in enumerate(items) if n % a.nshard == a.shard]
# The reviewer modules swallow every transport error and return an empty string, so a run can end
# with 494 empty reviews and no trace of why. With REVIEW_RAW set, every model reply is kept next to
# the review, which is how the b3a and b3i failures were read.
if os.environ.get("REVIEW_RAW") and hasattr(M, "chat"):
    _raw_dir = f"{B}/results/reviews/{a.system}_raw"
    os.makedirs(_raw_dir, exist_ok=True)
    _orig_chat = M.chat
    _cur = {"id": "none"}

    def _chat(messages, temperature, *args, **kw):
        out = _orig_chat(messages, temperature, *args, **kw)
        with open(f"{_raw_dir}/{_cur['id']}.log", "a") as fh:
            fh.write(json.dumps({"prompt_chars": sum(len(m["content"]) for m in messages),
                                 "reply_chars": len(out), "reply": out[:4000]}) + "\n")
        return out

    M.chat = _chat

outdir = f"{B}/results/reviews/{a.system}" + (f"_run{a.run}" if a.run else "")
os.makedirs(outdir, exist_ok=True)

def review_one(n_it):
    n, it = n_it
    op = f"{outdir}/{it['item_id']}.json"
    if os.path.exists(op) and os.path.getsize(op) > 50:
        return
    text = open(f"{B}/views/{it['item_id']}/body.tex", encoding="utf-8", errors="ignore").read()
    if os.environ.get("REVIEW_RAW") and hasattr(M, "chat"):
        _cur["id"] = it["item_id"]
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


# 494 papers one at a time is a month of wall clock for a reviewer that writes a full review; the
# reviews are independent and the vLLM server batches them, so the width is a throughput knob only.
with ThreadPoolExecutor(max_workers=int(os.environ.get("REVIEW_WORKERS", 8))) as ex:
    list(ex.map(review_one, enumerate(items)))
print("ALL DONE", a.system)
