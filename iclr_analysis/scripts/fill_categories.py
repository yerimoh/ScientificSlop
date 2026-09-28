"""arXiv primary category for every ICLR record (0916). The 0915 score-extension records were written with
arxiv_primary_category=None (extend_scores_0915.py), so the low-score bands had no domain label. This fills them
from the arXiv API exactly as Evaluation/ICLR/enrich_meta.py pass 1 does (id_list batches), but writes to
records/iclr_primary_category.json instead of touching the corpus meta.json files.
Domain groups used for balancing: LG (cs.LG, stat.ML, cs.NE, math.*, cs.AI), CV (cs.CV, eess.IV), CL (cs.CL), other."""
import json, os, re, time, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS = {"a": "http://www.w3.org/2005/Atom", "ar": "http://arxiv.org/schemas/atom"}
recs = json.load(open(f"{HERE}/records/iclr_records.json"))["records"]
OUT = f"{HERE}/records/iclr_primary_category.json"
cat = json.load(open(OUT)) if os.path.exists(OUT) else {}
cat = {k: v for k, v in cat.items() if isinstance(v, dict)}   # drop the plain-string first pass
for r in recs:
    m = json.load(open(os.path.dirname(r["root"]) + "/meta.json"))
    if m.get("arxiv_primary_category") and r["id"] not in cat:
        cat[r["id"]] = {"primary": m["arxiv_primary_category"], "cats": m.get("arxiv_categories") or [], "source": "meta"}
need = [r for r in recs if r["id"] not in cat and r.get("arxiv_id")]
print("to fetch", len(need))
for i in range(0, len(need), 80):
    batch = need[i:i + 80]
    ids = ",".join(r["arxiv_id"] for r in batch)
    url = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(ids)}&max_results=100"
    for attempt in range(4):
        try:
            xml = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read().decode("utf-8", "ignore")
            root = ET.fromstring(xml); break
        except Exception as e:
            print("  err", str(e)[:80]); time.sleep(8); root = None
    if root is None: continue
    by = {}
    for e in root.findall("a:entry", NS):
        m = re.search(r"abs/([^v]+)", e.findtext("a:id", "", NS))
        if not m: continue
        pc = e.find("ar:primary_category", NS); cats = [c.get("term") for c in e.findall("a:category", NS)]
        by[m.group(1)] = {"primary": (pc.get("term") if pc is not None else (cats[0] if cats else None)), "cats": cats, "source": "api_0916"}
    for r in batch:
        info = by.get(r["arxiv_id"]) or by.get(r["arxiv_id"].split("v")[0])
        if info: cat[r["id"]] = info
    print(f"  batch {i//80+1}/{(len(need)+79)//80} got {len(by)}", flush=True)
    time.sleep(3.2)
json.dump(cat, open(OUT, "w"), indent=0)
missing = [r["id"] for r in recs if r["id"] not in cat]
print("total", len(cat), "missing", len(missing), missing[:10])
