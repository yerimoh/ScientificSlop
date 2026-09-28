"""arXiv metadata for every pool id (H1 needs the v1 year, H2 the comment / journal_ref).
Split out of a3 so the slow title-search pass can be skipped when its yield is spent."""
import json, os, re, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("a3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "a3_resolve.py"))
a3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module.__self__ if False else None
src = open(spec.origin).read().split("def main(")[0]
exec(compile(src, spec.origin, "exec"), a3.__dict__)

OUTD = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench/data/Agents4Science"
meta_p = f"{OUTD}/cache/arxiv_meta_a4s.json"
meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
ids = json.load(open(f"{OUTD}/cache/all_ids.json"))
todo = [i for i in ids if i not in meta]
print(f"metadata needed {len(todo)} / pool {len(ids)}", flush=True)
for i in range(0, len(todo), 20):
    batch = todo[i:i + 20]
    xml = a3.get(f"https://export.arxiv.org/api/query?id_list={','.join(batch)}&max_results=20", base=3.0)
    got = dict(a3.parse_entries(xml)) if xml else {}
    for b in batch:
        meta[b] = got.get(b) or {"arxiv": b, "missing": True}
    json.dump(meta, open(meta_p, "w"), ensure_ascii=False)
    if (i // 20) % 5 == 0:
        print(f"  {min(i+20,len(todo))}/{len(todo)}", flush=True)
    time.sleep(3.0)
print("Done. arxiv meta", len(meta))
