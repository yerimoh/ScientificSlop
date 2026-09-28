"""Mean and spread over repeated runs of the baselines that sample.

Binoculars, NTS and Fast-DetectGPT are forward passes and return the same score every time, and
every SciSlop measure calls its model at temperature zero, so repeating them is not informative.
DetectGPT samples its T5 perturbations, and both automated reviewers generate at a non-zero
temperature, so each of those is one draw. Three draws are run and the table reports the mean with
the spread across draws, which is what the 0918 meeting asked for.
"""
import glob, json, os, statistics as st

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
items = json.load(open(f"{B}/itemsA4S.json"))["items"]
key = {i["item_id"]: (i["pair"], i["label"]) for i in items}


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
    return (w + 0.5 * t) / len(both), au, tpr, len(both)


def from_jsonl(path, field, d):
    if not os.path.isfile(path):
        return None
    by = {}
    for line in open(path):
        r = json.loads(line)
        k = key.get(r["id"])
        if k and r.get(field) is not None:
            by.setdefault(k[0], {})[k[1]] = float(r[field])
    return metrics(by, d)


def from_reviews(d0, d):
    if not os.path.isdir(d0):
        return None
    by = {}
    for f in glob.glob(f"{d0}/*.json"):
        r = json.load(open(f))
        fin = r.get("final")
        v = fin.get("Overall", fin.get("Rating")) if isinstance(fin, dict) else None
        k = key.get(r["item_id"])
        if k and isinstance(v, (int, float)):
            by.setdefault(k[0], {})[k[1]] = float(v)
    return metrics(by, d)


RUNS = {
    "DetectGPT": [lambda i=i: from_jsonl(f"{B}/results/seeds/detectgpt.seed{i}.jsonl",
                                         "detectgpt", +1) for i in (0, 1, 2)],
    "CycleReviewer": [lambda i=i: from_reviews(f"{B}/results/reviews/b2h" + (f"_run{i}" if i else ""), -1)
                      for i in (0, 1, 2)],
    "AI Scientist": [lambda i=i: from_reviews(f"{B}/results/reviews/b3a" + (f"_run{i}" if i else ""), -1)
                     for i in (0, 1, 2)],
}

print(f"{'baseline':<16}{'draws':>6}  {'PairAcc mean±sd':>18}{'AUROC mean±sd':>18}{'TPR mean±sd':>18}")
tex = []
for name, fns in RUNS.items():
    got = [f() for f in fns]
    got = [g for g in got if g]
    if not got:
        print(f"{name:<16}{0:>6}  not yet available")
        continue
    cols = []
    for j in range(3):
        v = [g[j] for g in got if g[j] is not None]
        m = st.mean(v)
        sd = st.stdev(v) if len(v) > 1 else 0.0
        cols.append((m, sd))
    print(f"{name:<16}{len(got):>6}  " + "".join(f"{m:>12.3f}±{sd:<5.3f}" for m, sd in cols))
    tex.append(f"\\hspace{{0.6em}}{name} & " +
               " & ".join(f"{m:.3f}\\,$\\pm$\\,{sd:.3f}" for m, sd in cols) +
               f" \\\\  % {len(got)} draws")
print("\n% LaTeX rows")
for t in tex:
    print(t)

# Write the rows straight into the appendix table, so the paper never holds a hand-typed number.
TAB = (os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v7/"
       "_ICLR_2027__Scientific_Mold (5)/tables/supp_tables/tab_seed_variance.tex")
if tex and os.path.isfile(TAB):
    body = open(TAB).read()
    start = body.index("\\midrule") + len("\\midrule")
    end = body.index("\\bottomrule")
    open(TAB, "w").write(body[:start] + "\n" + "\n".join(tex) + "\n" + body[end:])
    print("\nWROTE", TAB)
