"""The merged benchmark's numbers. One pool of 390 pairs, not two columns.

SciSlopBench is the 143 FARS pairs and the 247 Agents4Science pairs together, so a pair from either
half is one row of the same evaluation. Scores are pooled per paper and the pair comparison, the
ranking and the threshold are computed once over the pool. The per-half figures stay available for
the appendix, but the table the paper prints is this one.

Reviewer scores on the FARS half come from two places, the human side from bench165/results/reviews
and the AI side from the archived R1 reviews of the survival runs, which is how aggregate165.py
reads them.
"""
import glob, json, os, statistics as st

SB = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench"
ARCH = os.environ.get("SCISLOP_ROOT", ".") + "/artifact-ai2science/Evaluation/02_baselines_B"
SLOP = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/slop"
PLANES = {"macro_redund": "Structure", "xsec_ref": "Structure",
          "argument_graph": "Argument", "citation": "Argument",
          "fig_exposition": "Artifacts", "evidence_gap": "Artifacts"}


def halves():
    out = []
    it = json.load(open(f"{SB}/bench165/items165.json"))["items"]
    key = {}
    idk = {}
    for i in it:
        k = i["pair"] if i["label"] == 1 else i.get("arxiv", i["item_id"].replace("HU_", ""))
        key[(("AI" if i["label"] == 1 else "HU"), k)] = (f"165:{i['pair']}", i["label"])
        idk[i["item_id"]] = (f"165:{i['pair']}", i["label"])
    out.append(("165", f"{SB}/bench165", "", key, idk))
    it = json.load(open(f"{SB}/benchA4S/itemsA4S.json"))["items"]
    key, idk = {}, {}
    for i in it:
        k = i["pair"] if i["label"] == 1 else i.get("arxiv", i["item_id"].replace("HU_", ""))
        key[(("AI" if i["label"] == 1 else "HU"), k)] = (f"a4s:{i['pair']}", i["label"])
        idk[i["item_id"]] = (f"a4s:{i['pair']}", i["label"])
    out.append(("a4s", f"{SB}/benchA4S", "baselines/", key, idk))
    return out


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
    hs = sorted(hu, reverse=(d > 0))
    thr = hs[max(0, int(0.05 * len(hs)) - 1)] if hs else None
    tpr = sum(1 for x in ai if (x - thr) * d > 0) / len(ai) if thr is not None else None
    return round((w + 0.5 * t) / len(both), 3), round(au, 3), round(tpr, 3), len(both)


def item_pool(name):
    by = {}
    for tag, root, _sub, key, _idk in halves():
        p = f"{root}/results/slop/{name}/papers.jsonl"
        if not os.path.isfile(p) and name == "fig_exposition" and tag == "165":
            p = f"{SLOP}/Artifacts/fig_exposition/results/papers.jsonl"
        if not os.path.isfile(p):
            continue
        for line in open(p):
            r = json.loads(line)
            k = key.get((r["corpus"], r["id"]))
            if k and r.get("slop_score") is not None:
                by.setdefault(k[0], {})[k[1]] = float(r["slop_score"])
    return by


def aggregate_pool():
    per = {}
    for name, plane in PLANES.items():
        for pr, v in item_pool(name).items():
            for lb, sc in v.items():
                per.setdefault((pr, lb), {}).setdefault(plane, []).append(sc)
    by = {}
    for (pr, lb), planes in per.items():
        vals = [st.mean(v) for v in planes.values() if v]
        if vals:
            by.setdefault(pr, {})[lb] = st.mean(vals)
    return by


def det_pool(fname, field):
    by = {}
    for tag, root, sub, _key, idk in halves():
        p = f"{root}/results/{sub}{fname}" if tag == "a4s" else f"{root}/results/{fname}"
        if not os.path.isfile(p):
            continue
        for line in open(p):
            r = json.loads(line)
            k = idk.get(r["id"])
            if k and r.get(field) is not None:
                by.setdefault(k[0], {})[k[1]] = float(r[field])
    return by


def rev_pool(sysname, arch_dir):
    by = {}
    for tag, root, _sub, _key, idk in halves():
        for f in glob.glob(f"{root}/results/reviews/{sysname}/*.json"):
            r = json.load(open(f))
            fin = r.get("final")
            v = fin.get("Overall", fin.get("Rating")) if isinstance(fin, dict) else None
            k = idk.get(r["item_id"])
            if k and isinstance(v, (int, float)):
                by.setdefault(k[0], {})[k[1]] = float(v)
        if tag == "165" and arch_dir:            # the FARS side lives in the survival-run archive
            for i in json.load(open(f"{SB}/bench165/items165.json"))["items"]:
                if i["label"] != 1:
                    continue
                fp = f"{ARCH}/{arch_dir}/runs/{i['pair']}/logs/{i['pair']}_{sysname}_R1_review.json"
                if not os.path.isfile(fp):
                    continue
                fin = json.load(open(fp)).get("final")
                v = fin.get("Overall") if isinstance(fin, dict) else None
                if isinstance(v, (int, float)):
                    by.setdefault(f"165:{i['pair']}", {})[1] = float(v)
    return by


ROWS = [("Binoculars", lambda: det_pool("binoculars.jsonl", "binoculars"), -1),
        ("DetectGPT", lambda: det_pool("detectgpt.jsonl", "detectgpt"), +1),
        ("NTS", lambda: det_pool("nts.jsonl", "nts"), +1),
        ("Fast-DetectGPT", lambda: det_pool("fast_detectgpt.jsonl", "fast_detectgpt"), +1),
        ("CycleReviewer", lambda: rev_pool("b2h", "B2h_cyclereviewer"), -1),
        ("AI Scientist", lambda: rev_pool("b3a", "B3a_ai_scientist"), -1)]

print(f"{'row':<22}{'PairAcc':>9}{'AUROC':>8}{'TPR':>8}{'pairs':>7}")
for nm, fn, d in ROWS:
    m = metrics(fn(), d)
    print(f"{nm:<22}" + (f"{m[0]:>9.3f}{m[1]:>8.3f}{m[2]:>8.3f}{m[3]:>7}" if m else "  missing"))
for name in PLANES:
    m = metrics(item_pool(name), +1)
    print(f"{name:<22}" + (f"{m[0]:>9.3f}{m[1]:>8.3f}{m[2]:>8.3f}{m[3]:>7}" if m else "  missing"))
m = metrics(aggregate_pool(), +1)
print(f"{'SciSlop (aggregate)':<22}{m[0]:>9.3f}{m[1]:>8.3f}{m[2]:>8.3f}{m[3]:>7}")
