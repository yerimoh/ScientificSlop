"""Precision / recall / F1 for the pooled 390-pair SciSlopBench, same loaders and same
threshold rule as benchA4S/scripts/table_pooled.py (which reproduces Tables 2 and 3)."""
import os
import json, statistics as st
SRC = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench/benchA4S/scripts/table_pooled.py"
src = open(SRC).read().split("print(f\"{'row'")[0]
ns = {}
exec(src, ns)
metrics, item_pool, aggregate_pool, ROWS, PLANES = ns["metrics"], ns["item_pool"], ns["aggregate_pool"], ns["ROWS"], ns["PLANES"]

def prf(ai, hu, d, thr):
    tp = sum((x - thr) * d > 0 for x in ai); fp = sum((x - thr) * d > 0 for x in hu)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / len(ai)
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F, fp / len(hu), tp, fp

def analyse(by, d):
    both = [(v[1], v[0]) for v in by.values() if 1 in v and 0 in v]
    ai = [v[1] for v in by.values() if 1 in v]; hu = [v[0] for v in by.values() if 0 in v]
    hs = sorted(hu, reverse=(d > 0))
    thr5 = hs[max(0, int(0.05 * len(hs)) - 1)]
    P5, R5, F5, fpr5, tp5, fp5 = prf(ai, hu, d, thr5)
    vals = sorted(set(ai + hu))
    cands = vals + ([vals[0] - 1] if d > 0 else [vals[-1] + 1])
    best = max((prf(ai, hu, d, t) + (t,) for t in cands), key=lambda r: r[2])
    pa, au, tpr, n = metrics(by, d)
    return dict(n_pairs=len(both), n_ai=len(ai), n_hu=len(hu), pairacc=pa, auroc=au, tpr=tpr,
                thr5=thr5, P5=P5, R5=R5, F5=F5, fpr5=fpr5, tp5=tp5, fp5=fp5,
                bestF1=best[2], bestP=best[0], bestR=best[1], best_fpr=best[3], best_thr=best[6])

out = {}
for nm, fn, d in ROWS:
    out[nm] = analyse(fn(), d)
for name in PLANES:
    out[name] = analyse(item_pool(name), +1)
out["SciSlop (aggregate)"] = analyse(aggregate_pool(), +1)

print(f"{'row':<22}{'pairs':>6}{'nAI':>5}{'nHU':>5} | {'PairAcc':>7}{'AUROC':>7}{'TPR':>6} | {'P@5':>6}{'R@5':>6}{'F1@5':>6}{'FPR':>6} | {'bestF1':>7}{'P':>6}{'R':>6}{'FPR':>6}")
for k, r in out.items():
    print(f"{k:<22}{r['n_pairs']:>6}{r['n_ai']:>5}{r['n_hu']:>5} | {r['pairacc']:>7.3f}{r['auroc']:>7.3f}{r['tpr']:>6.3f} | "
          f"{r['P5']:>6.3f}{r['R5']:>6.3f}{r['F5']:>6.3f}{r['fpr5']:>6.3f} | {r['bestF1']:>7.3f}{r['bestP']:>6.3f}{r['bestR']:>6.3f}{r['best_fpr']:>6.3f}")
json.dump(out, open(os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench/benchA4S/results/threshold_metrics.json", "w"), indent=1)
