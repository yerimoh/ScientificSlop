"""Recompute Tables 2/3 after replacing fig_graph with fig_exposition.

Run with Python 3; reads existing measurements only. Writes an audit JSON beside
the benchmark results, leaving all original item measurements untouched.
Metrics follow analyze_faithful.py: AI-higher scores, half-credit pair ties,
AUROC on all available papers, and a strict threshold at at most 5% human FPR.
"""
import hashlib
import json
from pathlib import Path


BENCH = Path(__file__).resolve().parents[1]
DRAFT = BENCH.parents[1]
PLANES = {
    "Structure": ["macro_redund", "xsec_ref"],
    "Argument": ["argument_graph", "citation"],
    "Artifacts": ["fig_exposition", "evidence_gap"],
}


def mean_available(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def evaluate(scores, metadata):
    paired = {}
    for iid, value in scores.items():
        if value is not None:
            m = metadata[iid]
            paired.setdefault(m["pair"], {})[m["label"]] = value
    complete = [v for v in paired.values() if 0 in v and 1 in v]
    wins = sum(v[1] > v[0] for v in complete)
    ties = sum(v[1] == v[0] for v in complete)
    ai = [v for i, v in scores.items() if v is not None and metadata[i]["label"] == 1]
    hu = [v for i, v in scores.items() if v is not None and metadata[i]["label"] == 0]
    threshold = sorted(hu, reverse=True)[int(len(hu) * 0.05)]
    return {
        "n_ai": len(ai), "n_hu": len(hu), "n_pairs": len(complete),
        "ai_higher": wins, "ties": ties,
        "pairacc": (wins + 0.5 * ties) / len(complete),
        "auroc": sum((a > h) + 0.5 * (a == h) for a in ai for h in hu) / (len(ai) * len(hu)),
        "tpr_at_fpr5": sum(a > threshold for a in ai) / len(ai),
        "actual_fpr": sum(h > threshold for h in hu) / len(hu),
        "strict_threshold": threshold,
    }


def main():
    manifest = BENCH / "items165.json"
    metadata = {i["item_id"]: i for i in json.loads(manifest.read_text())["items"]}
    sources = {name: BENCH / "results/slop" / name / "papers.jsonl"
               for names in PLANES.values() for name in names}
    sources["fig_exposition"] = DRAFT / "slop/Artifacts/fig_exposition/results/papers.jsonl"
    sources["fig_graph"] = DRAFT / "slop/Artifacts/fig_graph/results/papers.jsonl"
    raw, aggregate = {}, {}
    for name, path in sources.items():
        raw[name], aggregate[name] = {}, {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            iid = ("AI_" if row["corpus"] == "AI" else "HU_") + row["id"]
            if iid not in metadata:
                continue
            assert iid not in raw[name], (name, iid)
            raw[name][iid] = row.get("slop_score")
            aggregate[name][iid] = row.get("slop_score_agg" if name == "macro_redund" else "slop_score")
            if name == "fig_exposition":
                assert row["slop_denominator"] == 6 and row["coverage"] == 1
                assert abs(row["slop_score"] - row["n_kinds"] / 6) <= 0.00005

    def combined(figure_item):
        result = {}
        for iid in metadata:
            plane_scores = []
            for names in PLANES.values():
                names = [figure_item if n == "fig_exposition" else n for n in names]
                plane_scores.append(mean_available([aggregate[n].get(iid) for n in names]))
            result[iid] = mean_available(plane_scores)
        return result

    old_scores, new_scores = combined("fig_graph"), combined("fig_exposition")
    old_metrics = evaluate(old_scores, metadata)
    assert [round(old_metrics[k], 3) for k in ("pairacc", "auroc", "tpr_at_fpr5")] == [0.902, 0.900, 0.608], old_metrics
    report = {
        "aggregation": "Unweighted item means within planes, then unweighted plane mean; skip NA. Macro redundancy uses slop_score_agg (ceiling 0.10).",
        "metrics": "PairAcc uses complete pairs and half-credit ties. AUROC/TPR use all available papers; AI higher; no ROC interpolation.",
        "sources": {name: {"path": str(p.relative_to(DRAFT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for name, p in {"manifest": manifest, **sources}.items()},
        "items": {name: evaluate(raw[name], metadata) for names in PLANES.values() for name in names},
        "previous_scislop": old_metrics,
        "scislop": evaluate(new_scores, metadata),
        "papers": [{"item_id": iid, "pair": m["pair"], "label": m["label"],
                    "figure_exposition": raw["fig_exposition"].get(iid),
                    "scislop": new_scores[iid]} for iid, m in metadata.items()],
    }
    out = BENCH / "results/figure_exposition_tables.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k in ("items", "previous_scislop", "scislop")}, indent=2))


if __name__ == "__main__":
    main()
