"""Run one deterministic slop checker over the ICLR tex corpus (records/iclr_records.json) by
monkeypatching _common/corpus before the checker module is imported (same trick as
bench165/scripts/run_slop165.py). AI side is empty; every ICLR paper is loaded as corpus "HU".
Output: results/slop/<checker>/{papers.jsonl, ...}
"""
import os, sys, json, argparse

ROOT = os.environ.get("SCISLOP_ROOT", ".")
SLOP = f"{ROOT}/paper/draft_v6/slop"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKER_DIR = {
    "macro_redund": f"{SLOP}/Structure/macro_redund/code",
    "xsec_ref": f"{SLOP}/Structure/xsec_ref/code",
    "citation": f"{SLOP}/Argument/citation/code",
    "evidence_gap": f"{SLOP}/Artifacts/evidence_gap/code",
    "argument_graph": f"{SLOP}/Argument/Argument_Graph/code",   # 0916: run with --extra "--stage pmi" once labels and PMI are cached
}
ap = argparse.ArgumentParser()
ap.add_argument("--checker", required=True, choices=list(CHECKER_DIR))
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--outsuffix", default="")
ap.add_argument("--extra", default="", help="extra argv for the checker, e.g. \"--stage pmi\"")
ap.add_argument("--records", default="iclr_records.json", help="record file under records/ (iclr2026_records.json for the single-year Pangram corpus)")
a = ap.parse_args()

sys.path.insert(0, f"{SLOP}/_common")
sys.path.insert(0, f"{ROOT}/artifact-ai2science/_llm")
import corpus  # noqa: E402

recs = json.load(open(f"{HERE}/records/{a.records}"))["records"]
if a.limit:
    recs = recs[:a.limit]
corpus.ai_papers = lambda: []
corpus.hu_papers = lambda: recs

outdir = f"{HERE}/results/slop/{a.checker}{a.outsuffix}"
os.makedirs(outdir, exist_ok=True)
cdir = CHECKER_DIR[a.checker]
sys.path.insert(0, cdir)
os.chdir(cdir)
import measure  # noqa: E402

src = open(os.path.join(cdir, "measure.py")).read()
argv = ["measure.py"]
if "--out" in src:
    argv += ["--out", outdir]
elif hasattr(measure, "RESULTS"):
    measure.RESULTS = outdir
else:
    raise SystemExit("refusing to clobber shared results")
argv += a.extra.split()
sys.argv = argv
measure.main()
n = sum(1 for _ in open(os.path.join(outdir, "papers.jsonl")))
print("WROTE", outdir, "papers:", n)
