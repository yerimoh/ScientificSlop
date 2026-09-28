r"""Put every system on a [0, 1] axis so one graph can carry all of them, and be explicit about which kind of
normalisation each system allows.

Three columns are written per system, all oriented so that 1 means "most AI-like".

  <sys>_def   definitional. Only for a score whose range is fixed by its own definition.
              ours and each item   slop_score is already failed units / checked units in [0, 1]
              Pangram              fraction_ai is already a fraction in [0, 1]
              reviewers            Overall lives on the 1-10 rating scale, so (10 - x) / 9
              detectors            blank. A perplexity ratio in (0, inf) and a z score in (-inf, inf) have no
                                   definitional end, so nothing can be divided by
  <sys>_mm    robust min-max inside the corpus, (x - p1) / (p99 - p1) clipped to [0, 1]. Available for everything,
              but it is corpus-relative, so a value of 0.5 means "middle of this corpus", not "half AI"
  <sys>_pct   percentile rank inside the corpus, 0 to 1. The distribution-free version, what the figures use

Reads results/scores_2026.csv and results/scores_years.csv, writes *_norm.csv next to them, and prints a table of
what each system got and where its AI and human papers land on the shared axis.
"""
import csv, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# name -> (AI direction, definitional mapping or None)
DEF = {
    "pangram_fraction_ai": (+1, lambda x: x),                       # already a fraction
    "binoculars": (-1, None),                                       # ratio in (0, inf)
    "detectgpt": (+1, None),                                        # z score
    "nts": (+1, None),                                              # normalised sensitivity
    "rev_b2h": (-1, lambda x: (10.0 - x) / 9.0),                    # 1-10 rating
    "rev_b3a": (-1, lambda x: (10.0 - x) / 9.0),
    "agg3": (+1, lambda x: x), "agg4": (+1, lambda x: x), "scislop5": (+1, lambda x: x),
    "macro_redund": (+1, lambda x: x), "xsec_ref": (+1, lambda x: x), "citation": (+1, lambda x: x),
    "evidence_gap": (+1, lambda x: x), "argument_graph": (+1, lambda x: x), "fig_exposition": (+1, lambda x: x),
}


def num(r, c):
    v = r.get(c)
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def process(path, group_col, ai_groups):
    rows = list(csv.DictReader(open(path)))
    out_fields = list(rows[0].keys()) if rows else []
    report = {}
    for sysname, (direction, fdef) in DEF.items():
        vals = [num(r, sysname) for r in rows]
        have = [v for v in vals if v is not None]
        if not have:
            continue
        arr = np.array(have, float)
        p1, p99 = np.percentile(arr, [1, 99])
        order = np.argsort(np.argsort(arr * direction))
        pct = {id(None): None}
        rank = dict(zip([i for i, v in enumerate(vals) if v is not None], order / max(1, len(order) - 1)))
        for col in (f"{sysname}_def", f"{sysname}_mm", f"{sysname}_pct"):
            if col not in out_fields:
                out_fields.append(col)
        for i, r in enumerate(rows):
            v = vals[i]
            if v is None:
                r[f"{sysname}_def"] = r[f"{sysname}_mm"] = r[f"{sysname}_pct"] = ""
                continue
            r[f"{sysname}_def"] = round(float(np.clip(fdef(v), 0, 1)), 4) if fdef else ""
            mm = (v * direction - min(p1 * direction, p99 * direction)) / max(1e-12, abs(p99 - p1))
            r[f"{sysname}_mm"] = round(float(np.clip(mm, 0, 1)), 4)
            r[f"{sysname}_pct"] = round(float(rank[i]), 4)
        ai = [num(r, sysname) for r in rows if r.get(group_col) in ai_groups]
        hu = [num(r, sysname) for r in rows if r.get(group_col) not in ai_groups]
        ai = [x for x in ai if x is not None]; hu = [x for x in hu if x is not None]
        report[sysname] = {
            "n": len(have), "kind": "definitional" if fdef else "corpus-relative only",
            "raw_range": (round(float(arr.min()), 4), round(float(arr.max()), 4)),
            "def_mean_ai": round(float(np.mean([np.clip(fdef(x), 0, 1) for x in ai])), 3) if fdef and ai else None,
            "def_mean_hu": round(float(np.mean([np.clip(fdef(x), 0, 1) for x in hu])), 3) if fdef and hu else None,
            "mm_mean_ai": round(float(np.mean([np.clip((x * direction - min(p1 * direction, p99 * direction)) / max(1e-12, abs(p99 - p1)), 0, 1) for x in ai])), 3) if ai else None,
            "mm_mean_hu": round(float(np.mean([np.clip((x * direction - min(p1 * direction, p99 * direction)) / max(1e-12, abs(p99 - p1)), 0, 1) for x in hu])), 3) if hu else None,
        }
    tmp = path.replace(".csv", "_norm.csv") + ".tmp"
    with open(tmp, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=out_fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    os.replace(tmp, path.replace(".csv", "_norm.csv"))
    return len(rows), report


for path, gcol, ai in [(f"{HERE}/results/scores_years.csv", "group", {"FARS"}),
                       (f"{HERE}/results/scores_2026.csv", "corpus", set())]:
    if not os.path.exists(path):
        continue
    n, rep = process(path, gcol, ai)
    print(f"\n=== {os.path.basename(path)} -> {os.path.basename(path).replace('.csv','_norm.csv')}  ({n} rows)")
    print(f"{'system':22s} {'normalization':22s} {'raw score range':>22s} {'defined [0,1] AI/human':>22s} {'minmax AI/human':>18s}")
    for k, v in rep.items():
        d = f"{v['def_mean_ai']}/{v['def_mean_hu']}" if v["def_mean_ai"] is not None else "-"
        m = f"{v['mm_mean_ai']}/{v['mm_mean_hu']}" if v["mm_mean_ai"] is not None else "-"
        print(f"{k:22s} {v['kind']:22s} {str(v['raw_range']):>22s} {d:>22s} {m:>18s}")
