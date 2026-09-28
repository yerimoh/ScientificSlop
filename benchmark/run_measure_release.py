"""Run one SciSlop measure over one half of the released benchmark.

Both sides of every pair are synthesised from the items file that materialize_bench.py wrote, so this
works without the original FARS dump (run_slop165.py needs it for the AI side). Output goes to
$SCISLOP_ROOT/paper/draft_v6/scislopbench/<bench>/results/slop/<checker><suffix>/papers.jsonl, the file
table_pooled.py and build_hf_release.py read.

  SCISLOP_ROOT=... python3 run_measure_release.py --half fars --checker xsec_ref
  SCISLOP_ROOT=... python3 run_measure_release.py --half a4s  --checker argument_graph --extra "--stage labels --runs 3"
"""
import os, sys, json, argparse

ROOT = os.environ.get("SCISLOP_ROOT", ".")
SLOP = f"{ROOT}/paper/draft_v6/slop"
BENCH = {"fars": ("bench165", "items165.json"), "a4s": ("benchA4S", "itemsA4S.json")}
CHECKER_DIR = {
    "macro_redund": f"{SLOP}/Structure/macro_redund/code",
    "xsec_ref": f"{SLOP}/Structure/xsec_ref/code",
    "citation": f"{SLOP}/Argument/citation/code",
    "evidence_gap": f"{SLOP}/Artifacts/evidence_gap/code",
    "argument_graph": f"{SLOP}/Argument/Argument_Graph/code",
}
ap = argparse.ArgumentParser()
ap.add_argument("--half", required=True, choices=list(BENCH))
ap.add_argument("--checker", required=True, choices=list(CHECKER_DIR))
ap.add_argument("--extra", default="")
ap.add_argument("--outsuffix", default="")
a = ap.parse_args()
bench, itf = BENCH[a.half]
B = f"{ROOT}/paper/draft_v6/scislopbench/{bench}"

sys.path.insert(0, f"{SLOP}/_common")
sys.path.insert(0, f"{ROOT}/artifact-ai2science/_llm")
import corpus  # noqa: E402

items = json.load(open(f"{B}/{itf}"))["items"]


def record(i):
    main = os.path.join(i["tex_dir"], i["tex_root"])
    assert os.path.isfile(main), main
    r = {"corpus": "AI" if i["label"] == 1 else "HU", "id": i["pair"] if i["label"] == 1 else i["arxiv"],
         "root": i["tex_dir"], "main_tex": main, "paper_dir": i["tex_dir"], "exp_dir": None, "diagram": None}
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
