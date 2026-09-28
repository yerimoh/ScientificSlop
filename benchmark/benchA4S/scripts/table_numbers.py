"""The numbers Tables 2 and 3 carry, for both benches, in one place.

PairAcc, AUROC and TPR at 5 percent FPR, per measure and per baseline, plus the SciSlop aggregate,
which is the paper mean over planes of the item scores with an NA item skipped, exactly as
SLOP_SCORE.md registers it. Printed as LaTeX-ready rows so a table is never typed by hand.
"""
import glob, json, os, statistics as st

SB = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench"
PLANES = {"macro_redund": "Structure", "xsec_ref": "Structure",
          "argument_graph": "Argument", "citation": "Argument",
          "fig_exposition": "Artifacts", "evidence_gap": "Artifacts"}
DET = [("Binoculars", "binoculars.jsonl", "binoculars", -1),
       ("DetectGPT", "detectgpt.jsonl", "detectgpt", +1),
       ("NTS", "nts.jsonl", "nts", +1),
       ("Fast-DetectGPT", "fast_detectgpt.jsonl", "fast_detectgpt", +1)]


def bench(name):
    if name == "bench165":
        items = json.load(open(f"{SB}/bench165/items165.json"))["items"]
        root, sub = f"{SB}/bench165", ""
    else:
        items = json.load(open(f"{SB}/benchA4S/itemsA4S.json"))["items"]
        root, sub = f"{SB}/benchA4S", "baselines/"
    key = {}
    for i in items:
        k = i["pair"] if i["label"] == 1 else i.get("arxiv", i["item_id"].replace("HU_", ""))
        key[("AI" if i["label"] == 1 else "HU", k)] = (i["pair"], i["label"])
    return root, sub, key, {i["item_id"]: (i["pair"], i["label"]) for i in items}


def metrics(by, d):
    both = [(v[1], v[0]) for v in by.values() if 1 in v and 0 in v]
    if not both:
        return None
    w = sum(1 for a, h in both if (a - h) * d > 0)
    t = sum(1 for a, h in both if a == h)
    ai = [v[1] for v in by.values() if 1 in v]
    hu = [v[0] for v in by.values() if 0 in v]
    au = sum(1.0 if (a - b) * d > 0 else 0.5 if a == b else 0.0
             for a in ai for b in hu) / (len(ai) * len(hu))
    # TPR at 5% FPR, threshold from the human side
    hs = sorted(hu, reverse=(d > 0))
    k = max(0, int(0.05 * len(hs)) - 1)
    thr = hs[k] if hs else None
    tpr = sum(1 for x in ai if (x - thr) * d > 0) / len(ai) if thr is not None else None
    return round((w + 0.5 * t) / len(both), 3), round(au, 3), round(tpr, 3), len(both)


def item_scores(root, key, name):
    p = f"{root}/results/slop/{name}/papers.jsonl"
    if not os.path.isfile(p):
        return {}
    out = {}
    for line in open(p):
        r = json.loads(line)
        k = key.get((r["corpus"], r["id"]))
        if k and r.get("slop_score") is not None:
            out.setdefault(k[0], {})[k[1]] = float(r["slop_score"])
    return out


def aggregate(root, key):
    """paper -> mean over planes of that paper's item scores, NA skipped."""
    per = {}
    for name, plane in PLANES.items():
        for pr, v in item_scores(root, key, name).items():
            for lb, sc in v.items():
                per.setdefault((pr, lb), {}).setdefault(plane, []).append(sc)
    by = {}
    for (pr, lb), planes in per.items():
        vals = [st.mean(v) for v in planes.values() if v]
        if vals:
            by.setdefault(pr, {})[lb] = st.mean(vals)
    return by


for bname in ("bench165", "benchA4S"):
    root, sub, key, bykey = bench(bname)
    print(f"\n===== {bname}")
    for name in PLANES:
        m = metrics(item_scores(root, key, name), +1)
        if m:
            print(f"  {name:<16} {m[0]:.3f} & {m[1]:.3f} & {m[2]:.3f}   n={m[3]}")
    m = metrics(aggregate(root, key), +1)
    print(f"  {'SciSlop(agg)':<16} {m[0]:.3f} & {m[1]:.3f} & {m[2]:.3f}   n={m[3]}")
    for nm, f, fld, d in DET:
        p = f"{root}/results/{sub}{f}"
        if not os.path.isfile(p):
            print(f"  {nm:<16} missing")
            continue
        by = {}
        for line in open(p):
            r = json.loads(line)
            k = bykey.get(r["id"])
            if k and r.get(fld) is not None:
                by.setdefault(k[0], {})[k[1]] = float(r[fld])
        m = metrics(by, d)
        print(f"  {nm:<16} {m[0]:.3f} & {m[1]:.3f} & {m[2]:.3f}   n={m[3]}")
    for sysname, nm in (("b2h", "CycleReviewer"), ("b3a", "AI Scientist")):
        by = {}
        for fp in glob.glob(f"{root}/results/reviews/{sysname}/*.json"):
            d0 = json.load(open(fp))
            fin = d0.get("final")
            v = fin.get("Overall", fin.get("Rating")) if isinstance(fin, dict) else None
            k = bykey.get(d0["item_id"])
            if k and isinstance(v, (int, float)):
                by.setdefault(k[0], {})[k[1]] = float(v)
        m = metrics(by, -1)
        print(f"  {nm:<16} " + (f"{m[0]:.3f} & {m[1]:.3f} & {m[2]:.3f}   n={m[3]}" if m else "missing"))
