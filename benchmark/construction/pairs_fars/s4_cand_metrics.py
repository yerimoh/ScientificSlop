"""Step 4: structural metrics (H4) for candidate e-prints.

Runs slopbench_lib.flatten + metrics over every pool id found in
  ana/reference/data/human_refs/eprints/<id>/          (existing corpus)
  paper/draft_v6/scislopbench/data/eprints_new/<id>/   (5-pair downloads)
  paper/draft_v6/scislopbench/data/pairs165_0911/eprints_new/<id>/  (this build's downloads)
Also records method-figure captions (overview/pipeline/architecture/framework words).
Output: cache/cand_metrics_165.json  {id: {source, tex_root, metrics, method_fig_captions}}
Idempotent: pass --redo to recompute, otherwise only new ids are added.
"""
import os, re, sys, json, glob

ROOT = os.environ.get("SCISLOP_ROOT", ".")
DATA = f"{ROOT}/paper/draft_v6/scislopbench/data"
OUTD = f"{DATA}/pairs165_0911"
sys.path.insert(0, f"{DATA}/scripts")
from slopbench_lib import find_root_tex, flatten, metrics

SRCS = [("human_refs/eprints", f"{ROOT}/ana/reference/data/human_refs/eprints"),
        ("eprints_new", f"{DATA}/eprints_new"),
        ("pairs165/eprints_new", f"{OUTD}/eprints_new")]

METHODFIG = re.compile(r"overview|pipeline|architecture|framework|illustrat|workflow|schematic", re.I)


def method_fig_captions(full):
    caps = re.findall(r"\\caption\s*(?:\[[^\]]*\])?\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", full)
    return [re.sub(r"\s+", " ", c)[:160] for c in caps if METHODFIG.search(c)][:3]


def main():
    out_p = f"{OUTD}/cache/cand_metrics_165.json"
    out = {} if "--redo" in sys.argv else (json.load(open(out_p)) if os.path.exists(out_p) else {})
    ids = json.load(open(f"{OUTD}/cache/all_ids.json"))
    extra = [a for a in sys.argv[1:] if re.match(r"^\d{4}\.\d{4,5}$", a)]
    n_found = 0
    for aid in ids + extra:
        if aid in out and not out[aid].get("error"):
            continue
        for label, base in SRCS:
            d = os.path.join(base, aid)
            if not os.path.isdir(d):
                continue
            try:
                r = find_root_tex(d)
                if r is None:
                    out[aid] = dict(source=label, error="no root tex")
                    break
                full = flatten(r)
                out[aid] = dict(source=label, tex_root=os.path.relpath(r, d), metrics=metrics(full),
                                method_fig_captions=method_fig_captions(full))
            except Exception as e:
                out[aid] = dict(source=label, error=f"{type(e).__name__}: {e}")
            break
    n_found = sum(1 for v in out.values() if v.get("metrics"))
    json.dump(out, open(out_p, "w"), ensure_ascii=False)
    ok = sum(1 for v in out.values() if v.get("metrics") and v["metrics"]["body_words"] >= 2500 and v["metrics"]["n_sections"] >= 4)
    print(f"ids: {len(ids)} | with e-print+metrics: {n_found} | H4 pass: {ok}")


if __name__ == "__main__":
    main()
