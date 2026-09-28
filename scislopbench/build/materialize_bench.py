"""Rebuild the benchmark working tree from the released Parquet tables.

Reads scislopbench/data/{pairs,papers}.parquet and writes, under $SCISLOP_ROOT,
  paper/draft_v6/scislopbench/bench165/items165.json   + views/<paper_id>/body.{tex,txt}   (FARS half)
  paper/draft_v6/scislopbench/benchA4S/itemsA4S.json  + views/<paper_id>/body.{tex,txt}   (Agents4Science half)
which is the layout every runner in benchmark/ expects. `tex_dir`/`tex_root` point at the released body view,
so the measures re-read exactly the text that was scored for the paper.

Usage: SCISLOP_ROOT=/path/to/root python3 materialize_bench.py [--data scislopbench/data]
"""
import argparse, json, os
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
a = ap.parse_args()
ROOT = os.environ.get("SCISLOP_ROOT", ".")
pairs = pq.read_table(f"{a.data}/pairs.parquet").to_pylist()
papers = {p["paper_id"]: p for p in pq.read_table(f"{a.data}/papers.parquet").to_pylist()}

HALF = {"fars": ("bench165", "items165.json"), "agents4science": ("benchA4S", "itemsA4S.json")}
items = {h: [] for h in HALF}
for pr in pairs:
    bench, _ = HALF[pr["source"]]
    vdir = f"{ROOT}/paper/draft_v6/scislopbench/{bench}/views"
    for pid, label in ((pr["ai_paper_id"], 1), (pr["human_paper_id"], 0)):
        p = papers[pid]
        d = f"{vdir}/{pid}"
        os.makedirs(d, exist_ok=True)
        for k in ("body_tex", "body_txt"):
            with open(f"{d}/body.{k[-3:]}", "w", encoding="utf-8") as f:
                f.write(p[k] or "")
        it = {"item_id": pid, "pair": pr["pair_id"], "label": label, "sim_tier": pr["sim_tier"], "pool": pr["pool"],
              "body_words": p["body_words"], "tex_dir": d, "tex_root": "body.tex"}
        if label == 0:
            it.update(arxiv=p["arxiv"], venue=p["venue"], iclr_rating=pr["human_iclr_rating"])
        else:
            it.update(a4s_status=pr["a4s_decision"], contribution_type=pr["contribution_type"], ai_source=pr["ai_tex_source"])
        items[pr["source"]].append(it)
for src, (bench, fname) in HALF.items():
    out = f"{ROOT}/paper/draft_v6/scislopbench/{bench}/{fname}"
    json.dump({"generated": "materialized from the released Parquet tables", "body_rule": "released body view",
               "n_pairs": len(items[src]) // 2, "build_failures": [], "items": items[src]}, open(out, "w"), indent=1)
    print("wrote", out, len(items[src]), "items")
