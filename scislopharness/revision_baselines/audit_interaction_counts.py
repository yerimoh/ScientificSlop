"""Check whether off-target redundancy score increases add counted defects.

Run on the completed single-item citation revisions, using the prespecified
60-paper subset. Write a per-paper audit without changing experiment scores.
"""
import json
from pathlib import Path

EOR = Path(__file__).resolve().parents[1]
BENCH = EOR.parent / "scislopbench/bench165"


def rows(path):
    return {r["id"]: r for line in path.read_text().splitlines() if line.strip()
            for r in [json.loads(line)]}


def main():
    codes = json.loads((EOR / "results/subset60.json").read_text())["codes"]
    original = {item: rows(BENCH / f"results/slop/{item}/papers.jsonl")
                for item in ("citation", "macro_redund")}
    revised = {item: rows(EOR / f"results/slop/a4s_citation/R1/{item}/papers.jsonl")
               for item in original}
    affected = []
    n_target_improved = 0
    for code in codes:
        if revised["citation"][code]["slop_score"] >= original["citation"][code]["slop_score"]:
            continue
        n_target_improved += 1
        before, after = original["macro_redund"][code], revised["macro_redund"][code]
        if after["slop_score_agg"] <= before["slop_score_agg"]:
            continue
        affected.append({"id": code, **{
            f"{field}_{when}": record[field]
            for field in ("slop_score_agg", "slop_numerator", "slop_denominator")
            for when, record in (("before", before), ("after", after))}})
    same_count = [r for r in affected
                  if r["slop_numerator_after"] == r["slop_numerator_before"]
                  and r["slop_denominator_after"] < r["slop_denominator_before"]]
    increased_count = [r for r in affected
                       if r["slop_numerator_after"] > r["slop_numerator_before"]]
    result = {"subset_n": len(codes), "n_citation_improved": n_target_improved,
              "n_redundancy_score_increased": len(affected),
              "n_unchanged_recycled_count_smaller_denominator": len(same_count),
              "n_increased_recycled_count": len(increased_count), "papers": affected}
    out = EOR / "results/summary/interactions_count_audit.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    note = (
        "# Audit of apparent citation–redundancy trade-offs\n\n"
        f"Of {n_target_improved} papers with lower Citation isolation, "
        f"{len(affected)} have a higher Macro redundancy score. "
        f"In {len(same_count)} of those papers, the counted recycled sentences "
        "are unchanged and the eligible-sentence denominator decreases. "
        f"The recycled-sentence count increases in {len(increased_count)} paper(s).\n\n"
        "Thus, the score increases do not establish that 26 papers acquired new "
        "repetition defects. Counts also do not establish that the identities of "
        "the counted sentences are unchanged; that requires unit-level tracing. "
        "The numerical interaction matrix remains unchanged.\n\n"
        "Reproduce with `python3 code/audit_interaction_counts.py`. Per-paper "
        "counts are in `interactions_count_audit.json`.\n"
    )
    out.with_suffix(".md").write_text(note)
    print(note)


if __name__ == "__main__":
    main()
