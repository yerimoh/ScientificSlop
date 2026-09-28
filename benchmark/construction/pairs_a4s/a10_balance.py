"""Step 10 (Agents4Science): the balance table and the trivial-feature AUROC that
PAIR_RULE_0911 step 4 asks for, over the assigned pairs.

A single-feature AUROC is the probability that a random AI paper outranks a random human
anchor on that feature, so 0.5 is no signal and 1.0 means the feature alone separates the
two classes. The caveat that matters here: AI-side numbers come from the PDF and human-side
numbers from TeX, so a gap can be an extraction artefact as much as a real difference. The
line to trust is the one where both sides are read the same way (n_sections is closest).

Output: BALANCE_a4s.md
"""
import os
import json, statistics as st

OUTD = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench/data/Agents4Science"
FEATS = ["body_words", "appendix_words", "n_sections", "n_figures", "n_tables",
         "n_unique_cites", "n_internal_refs"]
# whether the two sides are read the same way. The AI side comes from PDF text and the human
# side from TeX, so only counts that mean the same thing in both readings are comparable.
COMPARABLE = {"body_words": "roughly", "appendix_words": "no", "n_sections": "yes",
              "n_figures": "yes", "n_tables": "yes", "n_unique_cites": "no",
              "n_internal_refs": "no"}


def auroc(pos, neg):
    pos = [x for x in pos if x is not None]
    neg = [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    n = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return round(n / (len(pos) * len(neg)), 3)


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 1) if xs else None


def main():
    d = json.load(open(f"{OUTD}/pairs_a4s.json"))
    P = [p for p in d["pairs"] if p.get("human_primary")]
    lines = ["# Balance and trivial-feature AUROC (Agents4Science pairs)", "",
             f"Pairs with a primary anchor: {len(P)} of {len(d['pairs'])}.", "",
             "| feature | AI median | human median | ratio | AUROC (AI vs human) | separability | same reading on both sides |",
             "|---|---|---|---|---|---|---|"]
    for f in FEATS:
        ai = [p["ai"].get(f) for p in P]
        hu = [p["human_primary"].get(f) for p in P]
        a, h = med(ai), med(hu)
        ratio = round(a / h, 2) if a and h else None
        au = auroc(ai, hu)
        sep = round(max(au, 1 - au), 3) if au is not None else None
        lines.append(f"| {f} | {a} | {h} | {ratio} | {au} | {sep} | {COMPARABLE[f]} |")
    lines += ["",
              "AUROC below 0.5 means the human side is the larger one; the separability column is the "
              "same number folded, so 0.95 means one feature alone almost sorts the two classes. Read "
              "this table as a warning, not as a result. Two things are mixed in it. The A4S papers are "
              "genuinely shorter and cite less, and the two sides are not read the same way: the AI side "
              "comes from PDF text and the human side from TeX, so citation counts and internal "
              "references are not the same quantity on the two sides. The last column says which rows "
              "survive that objection. Section and float counts are comparable and are the rows to trust; "
              "the citation and internal-reference rows are not evidence of anything until both sides are "
              "read the same way.",
              "",
              "The practical consequence for the benchmark is that any downstream slop score must be a "
              "rate, never a count, and that a model trained on these pairs would reach a high AUROC "
              "without looking at a single slop feature.", ""]
    tiers = {}
    for p in P:
        t = p["human_primary"].get("sim_tier")
        tiers[t] = tiers.get(t, 0) + 1
    lines.append("Similarity tiers of the assigned anchors: "
                 + ", ".join(f"tier {k}: {v}" for k, v in sorted(tiers.items(), key=lambda x: -(x[0] or 0))) + ".")
    ratios = [p["human_primary"].get("body_ratio") for p in P if p["human_primary"].get("body_ratio")]
    if ratios:
        lines.append(f"Body-word ratio (human/AI) median {st.median(ratios):.2f}, "
                     f"min {min(ratios):.2f}, max {max(ratios):.2f}; "
                     f"{sum(1 for r in ratios if r > 2.5)} pairs exceed the 2.5 the rule prefers.")
    open(f"{OUTD}/BALANCE_a4s.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
