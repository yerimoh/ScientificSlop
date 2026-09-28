"""Prose view for the decided ICLR papers, same rules as bench165/scripts/build_views165.py
(body = expanded tex up to the first \appendix / Acknowledgement / bibliography; prose_view strips floats, math, cites).
Output: records/texts_iclr.jsonl {id, note_id, year, label:0, source:"iclr_tex", text} and records/views/<id>/body.{tex,txt}
"""
import json, os, re, sys
ROOT = os.environ.get("SCISLOP_ROOT", ".")
SB = f"{ROOT}/paper/draft_v6/scislopbench"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{SB}/data/scripts"); sys.path.insert(0, f"{SB}/bench165/scripts")
from slopbench_lib import flatten, split_body, prose_words  # noqa: E402
src = open(f"{SB}/bench165/scripts/build_views165.py").read()
ns = {}
exec(src.split("def hu_dir")[0].split("from slopbench_lib")[0] + "import re\n" + "def prose_view" + src.split("def prose_view")[1].split("def hu_dir")[0], ns)
prose_view = ns["prose_view"]

recs = json.load(open(f"{HERE}/records/iclr_records.json"))["records"]
ONLY_NEW = "--only-new" in sys.argv
if ONLY_NEW:
    old = {json.loads(l)["id"] for l in open(f"{HERE}/records/texts_iclr.jsonl")}
    recs = [r for r in recs if r["id"] not in old]
n = fails = 0
with open(f"{HERE}/records/texts_iclr_new.jsonl" if ONLY_NEW else f"{HERE}/records/texts_iclr.jsonl", "w") as w:
    for r in recs:
        if r["group4"] == "undecided":
            continue
        try:
            body, _, _ = split_body(flatten(r["main_tex"]))
            txt = prose_view(body)
            d = f"{HERE}/records/views/{r['id']}"; os.makedirs(d, exist_ok=True)
            open(f"{d}/body.tex", "w").write(body); open(f"{d}/body.txt", "w").write(txt)
            w.write(json.dumps(dict(id=r["id"], note_id=r["note_id"], year=r["year"], label=0, source="iclr_tex",
                                    body_words=len(txt.split()), text=txt)) + "\n"); n += 1
        except Exception as e:
            fails += 1; print("FAIL", r["id"], str(e)[:100])
print("texts", n, "fails", fails)
