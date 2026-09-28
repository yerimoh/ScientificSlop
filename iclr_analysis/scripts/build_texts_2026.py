"""Prose views for the ICLR 2026 corpus, same rules as bench165 (body up to \appendix/acks/bibliography, prose_view).
Output: records/texts_iclr2026.jsonl and records/views2026/<id>/body.{tex,txt}"""
import json, os, sys
ROOT = os.environ.get("SCISLOP_ROOT", "."); SB = f"{ROOT}/paper/draft_v6/scislopbench"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{SB}/data/scripts")
from slopbench_lib import flatten, split_body  # noqa: E402
src = open(f"{SB}/bench165/scripts/build_views165.py").read()
ns = {}; exec("import re\n" + "def prose_view" + src.split("def prose_view")[1].split("def hu_dir")[0], ns)
prose_view = ns["prose_view"]
recs = json.load(open(f"{HERE}/records/iclr2026_records.json"))["records"]
n = fails = 0
with open(f"{HERE}/records/texts_iclr2026.jsonl", "w") as w:
    for r in recs:
        try:
            body, _, _ = split_body(flatten(r["main_tex"])); txt = prose_view(body)
            d = f"{HERE}/records/views2026/{r['id']}"; os.makedirs(d, exist_ok=True)
            open(f"{d}/body.tex", "w").write(body); open(f"{d}/body.txt", "w").write(txt)
            if len(txt.split()) >= 200:
                w.write(json.dumps({"id": r["id"], "year": 2026, "label": 0, "source": "iclr2026", "body_words": len(txt.split()), "text": txt}) + "\n"); n += 1
        except Exception as e:
            fails += 1; print("FAIL", r["id"], str(e)[:80])
print("texts", n, "fails", fails)
