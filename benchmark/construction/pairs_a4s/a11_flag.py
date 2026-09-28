"""Step 11 (Agents4Science): per-pair caveats, written into pairs_a4s.json.

Nothing here changes an assignment. It records the three things a reader of a single pair would
otherwise have to re-derive, and a single recommended flag that combines them.

  body_ratio_over_2.5  the human anchor is more than 2.5x the A4S paper, which the rule prefers
                       to avoid; structure measured as a count will be dominated by this
  similarity_tier_1    same area, different problem. The rule marks these as reserve, not main
  eprint_may_be_extended  the arXiv comment says this is a longer or journal version, so the
                       measured artifact is not the accepted artifact
  anchor_shared_with_fars  the same human paper anchors a FARS pair in pairs165_0911, so the two
                       corpora must not be pooled without dropping one side of the duplicate
  findings_track_unstated  the arXiv comment names an *ACL-family venue without saying which
                       track, and those venues have a Findings tier that H2 is meant to exclude.
                       Other venues are not flagged: naming ICML or CVPR and a year is already
                       positive evidence of the main track
"""
import os
import json, re

OUTD = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench/data/Agents4Science"
EXT = re.compile(r"short(er)? version|extended version|journal version|preliminary version|"
                 r"an? earlier version", re.I)
TRACK = re.compile(r"main|oral|poster|spotlight|long paper|short paper|findings|camera.?ready", re.I)
ACL_FAMILY = re.compile(r"\b(ACL|EMNLP|NAACL|EACL|AACL|COLING)\b", re.I)


def main():
    d = json.load(open(f"{OUTD}/pairs_a4s.json"))
    meta = json.load(open(f"{OUTD}/cache/arxiv_meta_a4s.json"))
    f165 = json.load(open(f"{OUTD}/../pairs165_0911/pairs165_draft.json"))
    fars_anchors = {p["human_primary"]["arxiv"]: p["code"] for p in f165["pairs"]
                    if p.get("human_primary")}
    n = 0
    for p in d["pairs"]:
        hp = p.get("human_primary")
        if not hp:
            p["caveats"], p["benchmark_ready"] = [], False
            continue
        com = (meta.get(hp["arxiv"], {}) or {}).get("comment") or ""
        c = []
        if (hp.get("body_ratio") or 0) > 2.5:
            c.append("body_ratio_over_2.5")
        if hp.get("sim_tier") == 1:
            c.append("similarity_tier_1")
        if EXT.search(com):
            c.append("eprint_may_be_extended")
        if p["pool"] == "cited" and ACL_FAMILY.search(com) and not TRACK.search(com):
            c.append("findings_track_unstated")
        if hp["arxiv"] in fars_anchors:
            c.append("anchor_shared_with_fars")
            hp["also_anchors_fars"] = fars_anchors[hp["arxiv"]]
        p["caveats"] = c
        p["benchmark_ready"] = not c
        n += bool(c)
    d["n_benchmark_ready"] = sum(1 for p in d["pairs"] if p.get("benchmark_ready"))
    json.dump(d, open(f"{OUTD}/pairs_a4s.json", "w"), indent=1, ensure_ascii=False)
    import collections
    cc = collections.Counter(x for p in d["pairs"] for x in p.get("caveats", []))
    print(f"pairs {len(d['pairs'])} | primary {d['n_primary']} | flagged {n} | "
          f"benchmark_ready {d['n_benchmark_ready']}")
    for k, v in cc.most_common():
        print(f"  {k:24} {v}")


if __name__ == "__main__":
    main()
