"""Run one slop checker over the bench165 corpus (143 AI + 143 HU) by monkeypatching
_common/corpus before the checker module is imported. Output: bench165/results/slop/<checker>/.
"""
import os, sys, json, argparse

ROOT = os.environ.get("SCISLOP_ROOT", ".")
SLOP = f"{ROOT}/paper/draft_v6/slop"
B = f"{ROOT}/paper/draft_v6/scislopbench/bench165"

CHECKER_DIR = {
    "macro_redund": f"{SLOP}/Structure/macro_redund/code",
    "xsec_ref": f"{SLOP}/Structure/xsec_ref/code",
    "claim_table": f"{SLOP}/Artifacts/claim_table/code",
    "citation": f"{SLOP}/Argument/citation/code",
    "table_context": f"{SLOP}/Artifacts/table_context/code",
    "evidence_gap": f"{SLOP}/Artifacts/evidence_gap/code",
    "argument_graph": f"{SLOP}/Argument/Argument_Graph/code",
}
ap = argparse.ArgumentParser()
ap.add_argument("--checker", required=True, choices=list(CHECKER_DIR))
ap.add_argument("--extra", default="")
ap.add_argument("--outsuffix", default="")
a = ap.parse_args()

sys.path.insert(0, f"{SLOP}/_common")
sys.path.insert(0, f"{ROOT}/artifact-ai2science/_llm")
import corpus  # noqa: E402

items = json.load(open(f"{B}/items165.json"))["items"]
ai_codes = [i["pair"] for i in items if i["label"] == 1]
hu_records = []
for i in items:
    if i["label"] != 0:
        continue
    main = os.path.join(i["tex_dir"], i["tex_root"])
    assert os.path.isfile(main), main
    hu_records.append({"corpus": "HU", "id": i["arxiv"], "root": i["tex_dir"], "main_tex": main,
                       "paper_dir": i["tex_dir"], "exp_dir": None, "diagram": None,
                       "anchor_of": i["pair"], "anchor_year": None, "anchor_venue": i.get("venue")})

_orig_ai = corpus.ai_papers
corpus.ai_papers = lambda: [r for r in _orig_ai() if r["id"] in set(ai_codes)]
corpus.hu_papers = lambda: hu_records

outdir = f"{B}/results/slop/{a.checker}{a.outsuffix}"
os.makedirs(outdir, exist_ok=True)
cdir = CHECKER_DIR[a.checker]
sys.path.insert(0, cdir)
os.chdir(cdir)
import measure  # noqa: E402

src = open(os.path.join(cdir, "measure.py")).read()
argv = ["measure.py"] + (a.extra.split() if a.extra else [])
if "--out" in src:
    argv += ["--out", outdir]
elif hasattr(measure, "RESULTS"):
    measure.RESULTS = outdir
else:
    raise SystemExit("refusing to clobber shared results")
sys.argv = argv
measure.main()
print("WROTE", outdir)
