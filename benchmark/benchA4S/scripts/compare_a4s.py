"""Paired comparison of every benchA4S item against its FARS bench165 counterpart.

Each A4S submission is read beside its own human anchor, so the unit is the pair and the statistic
is the paired one: wins, losses and ties, Cliff's delta on the paired differences, and the share of
pairs the item orders correctly. Every line is also cut by how well the anchor matches, because 62
of the 247 anchors are the nearest admissible paper rather than a topic match, and by whether the
submission shipped LaTeX or the document was rebuilt from its PDF.

Usage: python3 compare_a4s.py [--item ...] -> results/COMPARE_a4s.json and a table on stdout
"""
import argparse, json, os, statistics as st, sys

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS = json.load(open(f"{B}/itemsA4S.json"))["items"]
META = {i["pair"]: i for i in ITEMS if i["label"] == 1}
ANCHOR = {i["pair"]: i for i in ITEMS if i["label"] == 0}
# the FARS side of the comparison is bench165, the 143 pairs the paper reports, not the item's own
# development run
F165 = os.path.join(os.path.dirname(B), "bench165", "results", "slop")
FARS_RESULTS = {"xsec_ref": f"{F165}/xsec_ref", "macro_redund": f"{F165}/macro_redund",
                "citation": f"{F165}/citation", "evidence_gap": f"{F165}/evidence_gap",
                "Argument_Graph": f"{F165}/argument_graph", "fig_exposition": f"{F165}/fig_specimen"}
KEY = "slop_score"


def rows(path):
    if not os.path.isfile(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def paired(rs):
    """(pair, ai, hu) for pairs where both sides carry a score."""
    ai = {r["id"]: r.get(KEY) for r in rs if r["corpus"] == "AI"}
    hu = {r["id"]: r.get(KEY) for r in rs if r["corpus"] == "HU"}
    out = []
    for code, m in META.items():
        a, h = ai.get(code), hu.get(ANCHOR[code].get("arxiv"))
        if a is None or h is None:
            continue
        out.append((code, float(a), float(h)))
    return out


def stats(ps):
    if not ps:
        return None
    w = sum(1 for _, a, h in ps if a > h)
    l = sum(1 for _, a, h in ps if a < h)
    t = len(ps) - w - l
    return {"n": len(ps), "win": w, "loss": l, "tie": t,
            "pair_acc": round((w + 0.5 * t) / len(ps), 3),
            "cliff": round((w - l) / len(ps), 3),
            "AI_mean": round(st.mean(a for _, a, _ in ps), 4),
            "HU_mean": round(st.mean(h for _, _, h in ps), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", default="")
    a = ap.parse_args()
    items = a.item.split(",") if a.item else list(FARS_RESULTS)
    out = {}
    for it in items:
        ps = paired(rows(f"{B}/results/slop/{it}/papers.jsonl"))
        if not ps:
            out[it] = {"note": "no paired scores yet"}
            continue
        cut = {"all": stats(ps)}
        for name, keep in (("matched_anchor", lambda c: (META[c]["sim_tier"] or 0) >= 2),
                           ("loose_anchor", lambda c: (META[c]["sim_tier"] or 0) < 2),
                           ("shipped_tex", lambda c: META[c]["ai_source"] == "shipped_tex"),
                           ("rebuilt_from_pdf", lambda c: META[c]["ai_source"] == "rebuilt_from_pdf")):
            cut[name] = stats([p for p in ps if keep(p[0])])
        f = rows(f"{FARS_RESULTS[it]}/papers.jsonl")
        if f:
            fa = [float(r[KEY]) for r in f if r["corpus"] == "AI" and r.get(KEY) is not None]
            fh = [float(r[KEY]) for r in f if r["corpus"] == "HU" and r.get(KEY) is not None]
            if fa and fh:
                cut["fars_bench165"] = {"n_AI": len(fa), "n_HU": len(fh),
                                        "AI_mean": round(st.mean(fa), 4), "HU_mean": round(st.mean(fh), 4)}
        out[it] = cut
    json.dump(out, open(f"{B}/results/COMPARE_a4s.json", "w"), indent=1)
    hdr = f"{'item':16s} {'n':>4s} {'W/L/T':>12s} {'pairAcc':>8s} {'cliff':>7s} {'AI':>8s} {'HU':>8s}   FARS AI/HU"
    print(hdr)
    print("-" * len(hdr))
    for it, c in out.items():
        s = c.get("all")
        if not s:
            print(f"{it:16s} {'-':>4s}   {c.get('note','')}")
            continue
        f = c.get("fars_bench165")
        fs = f"{f['AI_mean']}/{f['HU_mean']}" if f else "-"
        print(f"{it:16s} {s['n']:>4d} {s['win']:>4d}/{s['loss']:<3d}/{s['tie']:<3d} "
              f"{s['pair_acc']:>8.3f} {s['cliff']:>7.3f} {s['AI_mean']:>8.4f} {s['HU_mean']:>8.4f}   {fs}")
        for k in ("matched_anchor", "loose_anchor", "shipped_tex", "rebuilt_from_pdf"):
            v = c.get(k)
            if v:
                print(f"  {k:14s} {v['n']:>4d} {v['win']:>4d}/{v['loss']:<3d}/{v['tie']:<3d} "
                      f"{v['pair_acc']:>8.3f} {v['cliff']:>7.3f} {v['AI_mean']:>8.4f} {v['HU_mean']:>8.4f}")


if __name__ == "__main__":
    main()
