"""Agents4Science pairs under the benchmark spec of Appendix B.

The spec (Appendix B, construction details) keeps the four
selection criteria of PAIR_RULE_0911 and adds one that A4S does not automatically satisfy: the
final benchmark contains only pairs "with LaTeX source on both sides", read with the same parser.
A4S ships PDFs, so the AI side has to be recovered from supplementary.zip, and that recovery sets
the ceiling. Everything downstream is the 0911 assignment, re-verified and re-measured with the
TeX reader on both sides.

Outputs: pairs_a4s_v7.json, BALANCE_a4s_v7.md
"""
import json, os, re, sys, statistics as st

ROOT = os.environ.get("SCISLOP_ROOT", ".")
OUTD = f"{ROOT}/paper/draft_v6/scislopbench/data/Agents4Science"
sys.path.insert(0, f"{ROOT}/paper/draft_v6/scislopbench/data/scripts")
from slopbench_lib import flatten, metrics  # noqa: E402

SRC = {"human_refs/eprints/": f"{ROOT}/ana/reference/data/human_refs/eprints",
       "a4s/eprints_new/": f"{OUTD}/eprints_new",
       "eprints_new/": f"{ROOT}/paper/draft_v6/scislopbench/data/eprints_new",
       "pairs165/eprints_new/": f"{ROOT}/paper/draft_v6/scislopbench/data/pairs165_0911/eprints_new"}


def human_dir(tex_source):
    for k, base in SRC.items():
        if tex_source.startswith(k):
            return os.path.join(base, tex_source[len(k):])
    return None


def main():
    d = json.load(open(f"{OUTD}/pairs_a4s.json"))
    # v7_pick_root.py already chose each paper's root by how much of its own PDF it covers and
    # searched both the supplementary zip and the arXiv e-print.
    MIN_COV = 0.85
    tex_all = json.load(open(f"{OUTD}/cache/a4s_tex_index.json"))
    tex = {k: dict(v, path=v["root"],
                   origin="arXiv" if v["root"].startswith("a4s_arxiv/") else "supplementary.zip")
           for k, v in tex_all.items() if v["pdf_sentence_coverage"] >= MIN_COV}
    version_drops = [(k, v["pdf_sentence_coverage"]) for k, v in sorted(tex_all.items())
                     if v["pdf_sentence_coverage"] < MIN_COV]
    idx = json.load(open(f"{OUTD}/cache/a4s_index.json"))
    meta = json.load(open(f"{OUTD}/cache/arxiv_meta_a4s.json"))
    widen = json.load(open(f"{OUTD}/cache/widen_shortlist.json"))
    wmeta = {c["arxiv"]: c for lst in widen.values() for c in lst}
    byc = {p["code"]: p for p in d["pairs"]}

    pairs, dropped = [], []
    for code in sorted(tex_all):
        p = byc[code]
        if code not in tex:
            dropped.append((code, f"tex covers only {tex_all[code]['pdf_sentence_coverage']:.2f} "
                                  f"of its own PDF, so it is a different version"))
            continue
        if not p.get("human_primary"):
            dropped.append((code, "no human anchor under the four criteria"))
            continue
        hp = p["human_primary"]
        hd = human_dir(hp["tex_source"])
        hm = metrics(flatten(__import__("slopbench_lib").find_root_tex(hd)))
        am = metrics(flatten(os.path.join(OUTD, tex[code]["path"])))
        abstract = (meta.get(hp["arxiv"], {}) or {}).get("abstract") or \
                   (wmeta.get(hp["arxiv"], {}) or {}).get("abstract") or ""
        pairs.append(dict(
            code=code, submission_id=p["submission_id"], a4s_status=p["a4s_status"],
            contribution_type=p["a4s_type"], pool=p["pool"], caveats=p.get("caveats", []),
            ai=dict(title=idx[code]["title"], abstract=idx[code]["abstract"][:2500],
                    tex_root=tex[code]["path"], tex_origin=tex[code]["origin"],
                    title_sim=tex[code]["title_sim"],
                    pdf_sentence_coverage=tex[code]["pdf_sentence_coverage"],
                    pdf=p["ai"]["pdf"], scores=p.get("a4s_scores"), autonomy=p.get("a4s_autonomy"),
                    **{k: am[k] for k in ("body_words", "appendix_words", "has_appendix", "n_sections",
                                          "n_subsections", "n_figures", "n_tables", "n_equations",
                                          "n_unique_cites", "n_internal_refs", "has_related_work")},
                    github=am["github"][:2], template=am["template"][:1]),
            human=dict(arxiv=hp["arxiv"], title=hp["title"], abstract=abstract,
                       venue=hp["venue"], v1_year=hp["v1_year"], sim_tier=hp["sim_tier"],
                       tier_reason=hp["tier_reason"], relation=hp["relation"],
                       iclr_rating=hp.get("iclr_rating"), pangram_ai_frac=hp.get("pangram_ai_frac"),
                       tex_source=hp["tex_source"], tex_root=hp["tex_root"],
                       **{k: hm[k] for k in ("body_words", "appendix_words", "has_appendix", "n_sections",
                                             "n_subsections", "n_figures", "n_tables", "n_equations",
                                             "n_unique_cites", "n_internal_refs", "has_related_work")},
                       github=hm["github"][:2], template=hm["template"][:1]),
            body_ratio=round(hm["body_words"] / am["body_words"], 2) if am["body_words"] else None))
    # caveats are recomputed here because the 0911 file derived the ratio from PDF word counts
    f165 = json.load(open(f"{OUTD}/../pairs165_0911/pairs165_draft.json"))
    fars_anchors = {x["human_primary"]["arxiv"]: x["code"] for x in f165["pairs"]
                    if x.get("human_primary")}
    for q in pairs:
        c = []
        if (q["body_ratio"] or 0) > 2.5:
            c.append("body_ratio_over_2.5")
        if q["human"]["sim_tier"] == 1:
            c.append("similarity_tier_1")
        if q["ai"]["body_words"] < 1000:
            c.append("ai_body_under_1000_words")
        if q["human"]["arxiv"] in fars_anchors:
            c.append("anchor_shared_with_fars")
            q["human"]["also_anchors_fars"] = fars_anchors[q["human"]["arxiv"]]
        q["caveats"] = c
        q["benchmark_ready"] = not c

    out = dict(generated="2026-09-18",
               spec="Appendix B, app:data_detail_matching",
               corpus="Agents4Science 2025, all domains (247 papers)",
               ceiling_note="both sides need LaTeX and the A4S source must be the submitted "
                            "document; 17 of 154 submissions have a title-matching source in "
                            "supplementary.zip or on arXiv, and 14 of those cover at least 0.85 "
                            "of their own PDF by sentence",
               n_a4s=len(idx), n_with_tex=len(tex_all), n_with_tex_same_version=len(tex),
               version_drops=version_drops, n_pairs=len(pairs),
               n_benchmark_ready=sum(1 for q in pairs if q["benchmark_ready"]),
               dropped=dropped, pairs=pairs)
    json.dump(out, open(f"{OUTD}/pairs_a4s_v7.json", "w"), indent=1, ensure_ascii=False)
    print(f"A4S 154 | tex obtained {len(tex)} | v7 pairs {len(pairs)} | dropped {len(dropped)}")
    for c, why in dropped:
        print(f"   - {c}: {why}")
    return out


if __name__ == "__main__":
    main()
