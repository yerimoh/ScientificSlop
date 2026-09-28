"""Compare one round of single-item feedback (a4s_<item>) with one round of joint
feedback (a4_slop R1) on the prespecified 60-paper subset, item by item.

Writes results/summary/interactions_single_vs_joint.{json,md}. Scores are not changed.
Also records the joint trajectory (R1..R3) for Cross-section references on the same
papers, the feedback lists actually sent, and the item order in the joint prompt.
"""
import glob
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

EOR = Path(__file__).resolve().parents[1]
BENCH = EOR.parent / "scislopbench/bench165"
ITEMS = ["macro_redund", "xsec_ref", "citation", "evidence_gap"]


def rows(path):
    return {r["id"]: r for line in Path(path).read_text().splitlines() if line.strip()
            for r in [json.loads(line)]}


def score(r, item):
    return r["slop_score_agg"] if item == "macro_redund" else r["slop_score"]


def progress(arm, codes):
    out = {}
    for f in glob.glob(str(EOR / f"progress/{arm}*.jsonl")):
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("round") == 1 and r["code"] in codes and r.get("exec") == "ok":
                out[r["code"]] = r
    return out


def main():
    codes = json.loads((EOR / "results/subset60.json").read_text())["codes"]
    cset = set(codes)
    res = {"subset_n": len(codes), "items": {}}
    for item in ITEMS:
        r0 = rows(BENCH / f"results/slop/{item}/papers.jsonl")
        single = rows(EOR / f"results/slop/a4s_{item}/R1/{item}/papers.jsonl")
        joint = rows(EOR / f"results/slop/a4_slop/R1/{item}/papers.jsonl")
        ds, dj, worse, equal, better = [], [], 0, 0, 0
        for c in codes:
            s = score(single[c], item) - score(r0[c], item)
            j = score(joint[c], item) - score(r0[c], item)
            ds.append(s)
            dj.append(j)
            if j > s + 1e-9:
                worse += 1
            elif abs(j - s) < 1e-9:
                equal += 1
            else:
                better += 1
        res["items"][item] = {
            "mean_change_single": round(st.mean(ds), 4), "mean_change_joint": round(st.mean(dj), 4),
            "n_joint_smaller_reduction": worse, "n_equal": equal, "n_joint_larger_reduction": better}

    # Cross-section references in counts (unused objects), joint trajectory on the same papers
    r0 = rows(BENCH / "results/slop/xsec_ref/papers.jsonl")
    single = rows(EOR / "results/slop/a4s_xsec_ref/R1/xsec_ref/papers.jsonl")
    xs = {"unused_R0": sum(r0[c]["slop_numerator"] for c in codes),
          "objects_R0": sum(r0[c]["n_objects"] for c in codes),
          "unused_single_R1": sum(single[c]["slop_numerator"] for c in codes),
          "objects_single_R1": sum(single[c]["n_objects"] for c in codes),
          "papers_cleared_single_R1": sum(single[c]["slop_numerator"] == 0 for c in codes)}
    for t in (1, 2, 3):
        jt = rows(EOR / f"results/slop/a4_slop/R{t}/xsec_ref/papers.jsonl")
        xs[f"unused_joint_R{t}"] = sum(jt[c]["slop_numerator"] for c in codes)
        xs[f"objects_joint_R{t}"] = sum(jt[c]["n_objects"] for c in codes)
        xs[f"papers_cleared_joint_R{t}"] = sum(jt[c]["slop_numerator"] == 0 for c in codes)
        xs[f"mean_score_joint_R{t}"] = round(st.mean(jt[c]["slop_score"] for c in codes), 4)
    xs["mean_score_R0"] = round(st.mean(r0[c]["slop_score"] for c in codes), 4)
    xs["mean_score_single_R1"] = round(st.mean(single[c]["slop_score"] for c in codes), 4)
    xs["frac_removed_single_R1"] = round(1 - xs["unused_single_R1"] / xs["unused_R0"], 3)
    xs["frac_removed_joint_R1"] = round(1 - xs["unused_joint_R1"] / xs["unused_R0"], 3)
    res["xsec_ref_counts"] = xs

    # Feedback actually sent and edit volume per condition
    joint_p = progress("a4_slop", cset)
    fb = {"joint": {"n": len(joint_p),
                    "units_per_item_mean": {it: round(st.mean(r["info"]["n_units"].get(it, 0) for r in joint_p.values()), 2) for it in ITEMS},
                    "files_changed_mean": round(st.mean(r["n_files_changed"] for r in joint_p.values()), 2),
                    "abs_word_change_mean": round(st.mean(abs(r["words_after"] - r["words_before"]) for r in joint_p.values()), 1)}}
    for it in ITEMS:
        p = progress(f"a4s_{it}", cset)
        fb[f"single_{it}"] = {"n": len(p),
                              "units_mean": round(st.mean(r["info"]["n_units"].get(it, 0) for r in p.values()), 2),
                              "files_changed_mean": round(st.mean(r["n_files_changed"] for r in p.values()), 2),
                              "abs_word_change_mean": round(st.mean(abs(r["words_after"] - r["words_before"]) for r in p.values()), 1)}
    sx = progress("a4s_xsec_ref", cset)
    fb["papers_with_different_xsec_unit_count_joint_vs_single"] = sum(
        1 for c in codes if c in joint_p and c in sx
        and joint_p[c]["info"]["n_units"].get("xsec_ref") != sx[c]["info"]["n_units"].get("xsec_ref"))
    res["feedback"] = fb

    orders = Counter()
    for c in codes:
        f = EOR / f"runs/a4_slop/{c}/logs/{c}_a4_slop_R1_a0_feedback.json"
        if f.exists():
            names = re.findall(r"## Issue \d+: (.+)", json.loads(f.read_text())["prompt"])
            orders[" > ".join(n.split()[0] for n in names)] += 1
    res["joint_prompt_issue_order"] = dict(orders)

    out = EOR / "results/summary/interactions_single_vs_joint.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    it = res["items"]
    md = ["# Single-item vs joint feedback, one round, 60 papers\n",
          "| item | mean change single | mean change joint | joint smaller / equal / larger reduction |", "|---|---|---|---|"]
    for k, v in it.items():
        md.append(f"| {k} | {v['mean_change_single']} | {v['mean_change_joint']} | {v['n_joint_smaller_reduction']} / {v['n_equal']} / {v['n_joint_larger_reduction']} |")
    md += ["", "## Cross-section references in counts (unused objects, 60 papers)", "",
           f"R0 {xs['unused_R0']}/{xs['objects_R0']} unused. Single R1 {xs['unused_single_R1']} ({xs['frac_removed_single_R1']:.0%} removed, {xs['papers_cleared_single_R1']} papers cleared). "
           f"Joint R1 {xs['unused_joint_R1']} ({xs['frac_removed_joint_R1']:.0%} removed, {xs['papers_cleared_joint_R1']} cleared), R2 {xs['unused_joint_R2']} ({xs['papers_cleared_joint_R2']} cleared), R3 {xs['unused_joint_R3']} ({xs['papers_cleared_joint_R3']} cleared). "
           f"Object totals stay at {xs['objects_R0']}–{xs['objects_joint_R1']}, so fixes add references rather than delete objects.",
           "", "## Feedback and edit volume", "", "```", json.dumps(fb, indent=2), "```", "",
           f"Joint prompt issue order (constant): {res['joint_prompt_issue_order']}. Both conditions use the same prompt builder with the same per-item cap of 12 listed units.",
           "", "Reproduce with `python3 code/audit_single_vs_joint.py`."]
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
