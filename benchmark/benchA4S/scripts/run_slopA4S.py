"""Run one slop checker over benchA4S by monkeypatching _common/corpus before the checker imports.

Differs from run_slop165.py in one way that matters. bench165 filters corpus.ai_papers(), which
only knows the FARS layout <root>/code/writing/paper/main.tex. A4S papers have no such layout, so
both sides are synthesised here from itemsA4S.json exactly the way bench165 synthesises its human
records. Fields the FARS record carries and A4S cannot (exp_dir, diagram) are None, so a checker
that needs them fails loudly instead of scoring a paper on missing material.

Output: benchA4S/results/slop/<checker>/
"""
import os, sys, json, argparse

ROOT = os.environ.get("SCISLOP_ROOT", ".")
SLOP = f"{ROOT}/paper/draft_v6/slop"
B = f"{ROOT}/paper/draft_v6/scislopbench/benchA4S"

CHECKER_DIR = {
    "macro_redund": f"{SLOP}/Structure/macro_redund/code",
    "xsec_ref": f"{SLOP}/Structure/xsec_ref/code",
    "citation": f"{SLOP}/Argument/citation/code",
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

items = json.load(open(f"{B}/itemsA4S.json"))["items"]


def record(i):
    main = os.path.join(i["tex_dir"], i["tex_root"])
    assert os.path.isfile(main), main
    r = {"corpus": "AI" if i["label"] == 1 else "HU",
         "id": i["pair"] if i["label"] == 1 else i["arxiv"],
         "root": i["tex_dir"], "main_tex": main, "paper_dir": i["tex_dir"],
         "exp_dir": None, "diagram": None}
    if i["label"] == 0:
        r.update(anchor_of=i["pair"], anchor_year=None, anchor_venue=i.get("venue"))
    return r


ai_records = [record(i) for i in items if i["label"] == 1]
hu_records = [record(i) for i in items if i["label"] == 0]
corpus.ai_papers = lambda: ai_records
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
