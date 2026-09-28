"""Step 4 (Agents4Science): H4 structural metrics for candidate e-prints.

Same reader as the FARS build (slopbench_lib.flatten + metrics), over every directory the
project already holds plus this build's downloads.
Output: cache/cand_metrics_a4s.json  {id: {source, tex_root, metrics, method_fig_captions}}
"""
import os, re, sys, json

ROOT = os.environ.get("SCISLOP_ROOT", ".")
DATA = f"{ROOT}/paper/draft_v6/scislopbench/data"
OUTD = f"{DATA}/Agents4Science"
sys.path.insert(0, f"{DATA}/scripts")
from slopbench_lib import find_root_tex, flatten, metrics

SRCS = [("human_refs/eprints", f"{ROOT}/ana/reference/data/human_refs/eprints"),
        ("eprints_new", f"{DATA}/eprints_new"),
        ("pairs165/eprints_new", f"{DATA}/pairs165_0911/eprints_new"),
        ("a4s/eprints_new", f"{OUTD}/eprints_new")]

METHODFIG = re.compile(r"overview|pipeline|architecture|framework|illustrat|workflow|schematic", re.I)


def method_fig_captions(full):
    caps = re.findall(r"\\caption\s*(?:\[[^\]]*\])?\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", full)
    return [re.sub(r"\s+", " ", c)[:160] for c in caps if METHODFIG.search(c)][:3]


def main():
    out_p = f"{OUTD}/cache/cand_metrics_a4s.json"
    out = {} if "--redo" in sys.argv else (json.load(open(out_p)) if os.path.exists(out_p) else {})
    ids = json.load(open(f"{OUTD}/cache/all_ids.json"))
    for aid in ids:
        if aid in out and not out[aid].get("error"):
            continue
        for label, base in SRCS:
            d = os.path.join(base, aid)
            if not os.path.isdir(d) or not os.listdir(d):
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
    json.dump(out, open(out_p, "w"), ensure_ascii=False)
    nm = sum(1 for v in out.values() if v.get("metrics"))
    ok = sum(1 for v in out.values() if v.get("metrics") and v["metrics"]["body_words"] >= 2500
             and v["metrics"]["n_sections"] >= 4)
    print(f"ids {len(ids)} | with e-print metrics {nm} | H4 pass {ok}")


if __name__ == "__main__":
    main()
