r"""Retry only the papers whose AI Scientist review could not be parsed, with a raised generation limit.

Why. The wrapper asks the model for one JSON object and keeps it only if a balanced brace scan parses. Sakana's
default generation limit is 3000 tokens, and a successful review here is 630 to 1050 tokens, so the limit normally
has room. On a few long papers the model writes a longer review, the JSON is cut off at the limit, the brace scan
finds no closing brace, and all three truncation levels fail in turn. The input is not the problem, the wrapper
already truncates the paper to 14000 characters at the first level.

What this does. It finds the papers with no parsed score, reruns those alone with B3A_MAX_TOKENS raised, and records
exactly which papers were rerun at which limit in records/b3a_raised_limit.json, so the results table can carry a
footnote naming them. Papers that already have a score are never touched, and the default stays 3000 for everyone
else, so the baseline is unchanged except for the named papers.

Usage: python3 retry_b3a_failed.py --outdir b3a_2026 --views views2026 --limit 6000
"""
import argparse, glob, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--outdir", default="b3a_2026", help="directory under results/reviews/")
ap.add_argument("--views", default="views2026", help="view directory under records/")
ap.add_argument("--limit", type=int, default=6000, help="raised B3A_MAX_TOKENS")
ap.add_argument("--endpoint", default=os.environ.get("LLM_ENDPOINT", ""), help="running vLLM endpoint")
a = ap.parse_args()

failed = []
for f in sorted(glob.glob(f"{HERE}/results/reviews/{a.outdir}/*.json")):
    try:
        r = json.load(open(f))
        fin = r.get("final") or {}
        ok = isinstance(fin, dict) and isinstance(fin.get("Overall", fin.get("Rating")), (int, float))
    except Exception:
        ok = False
    if not ok:
        failed.append(os.path.splitext(os.path.basename(f))[0])
print(f"{a.outdir}: {len(failed)} papers with no parsed review")
if not failed:
    sys.exit(0)

subset = f"{HERE}/records/_retry_{a.outdir}.json"
json.dump({"rule": f"papers of {a.outdir} whose review did not parse at the default limit", "ids": failed},
          open(subset, "w"), indent=1)
for pid in failed:
    os.remove(f"{HERE}/results/reviews/{a.outdir}/{pid}.json")     # the runner keeps a scored file, so clear these

env = dict(os.environ, B3A_MAX_TOKENS=str(a.limit))
if a.endpoint:
    env["LLM_ENDPOINT"] = a.endpoint
t0 = time.time()
subprocess.run([sys.executable, f"{HERE}/scripts/run_reviews_iclr.py", "--system", "b3a",
                "--subset", os.path.basename(subset), "--views", a.views, "--outdir", a.outdir], env=env, check=False)

recovered = []
for pid in failed:
    f = f"{HERE}/results/reviews/{a.outdir}/{pid}.json"
    if not os.path.exists(f):
        continue
    r = json.load(open(f)); fin = r.get("final") or {}
    if isinstance(fin, dict) and isinstance(fin.get("Overall", fin.get("Rating")), (int, float)):
        recovered.append(pid)
rec = f"{HERE}/records/b3a_raised_limit.json"
prev = json.load(open(rec)) if os.path.exists(rec) else {"runs": []}
prev["runs"].append({"outdir": a.outdir, "limit": a.limit, "default_limit": 3000,
                     "attempted": failed, "recovered": recovered,
                     "still_failing": [p for p in failed if p not in recovered],
                     "minutes": round((time.time() - t0) / 60, 1), "when": time.strftime("%Y-%m-%d %H:%M")})
json.dump(prev, open(rec, "w"), indent=1)
print(f"retried {len(failed)} at limit {a.limit}: recovered {len(recovered)}, still failing {len(failed)-len(recovered)}")
print(f"recorded in {rec}")
