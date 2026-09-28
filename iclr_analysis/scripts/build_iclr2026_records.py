r"""Records for the single-year ICLR 2026 corpus (Pangram release + arXiv tex, 691 papers).

Why a second corpus. The 2017-2026 corpus mixes review scales (2020 uses {1,3,6,8}, 2023 {3,5,6,8,10}, 2017 a
1-10 scale) and its score bands mix years, so a raw mean rating is not comparable across papers. ICLR 2026 is one
year, one scale, and every paper carries the review dimensions and an independent commercial AI judgment:
  rating_mean, soundness_mean, presentation_mean, contribution_mean, confidence_mean   (OpenReview)
  fraction_ai                                                                          (Pangram, ICLR's own partner)
  accept, tier                                                                         (decision)
Source: artifact-ai2science/Evaluation/ICLR2026_Pangram/data/{texts/<id>/tex, papers_2026.jsonl}.
Output: records/iclr2026_records.json (loader shape of slop/_common/corpus).
"""
import json, glob, os, sys, collections
ROOT = os.environ.get("SCISLOP_ROOT", ".")
PG = f"{ROOT}/artifact-ai2science/Evaluation/ICLR2026_Pangram/data"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/paper/draft_v6/slop/_common")
from corpus import hu_main_tex  # noqa: E402

pp = {}
for l in open(f"{PG}/papers_2026.jsonl"):
    r = json.loads(l)
    pp[r.get("submission_id")] = r
recs, skipped = [], collections.Counter()
for d in sorted(glob.glob(f"{PG}/texts/*")):
    sid = os.path.basename(d)
    if not os.path.isdir(f"{d}/tex"):
        skipped["no_tex"] += 1; continue
    main = hu_main_tex(f"{d}/tex")
    if not main:
        skipped["no_documentclass"] += 1; continue
    m = pp.get(sid) or {}
    if m.get("rating_mean") is None:
        skipped["no_rating"] += 1; continue
    tier = (m.get("tier") or "").lower()
    group = "oral" if tier in ("oral", "talk") else ("accept" if m.get("accept") else "reject")
    recs.append({"corpus": "HU", "id": f"iclr2026_{sid}", "note_id": sid, "year": 2026,
                 "root": f"{d}/tex", "main_tex": main, "paper_dir": f"{d}/tex", "exp_dir": None, "diagram": None,
                 "group4": group, "group5": tier or group, "accept": bool(m.get("accept")), "tier": m.get("tier"),
                 "overall_mean": m.get("rating_mean"), "rating_min": m.get("rating_min"), "rating_max": m.get("rating_max"),
                 "n_reviews": m.get("n_reviews"), "soundness": m.get("soundness_mean"), "presentation": m.get("presentation_mean"),
                 "contribution": m.get("contribution_mean"), "confidence": m.get("confidence_mean"),
                 "fraction_ai": m.get("fraction_ai"), "review_ai_mean": m.get("review_ai_mean"),
                 "primary_area": m.get("primary_area"), "arxiv_id": m.get("arxiv_id"), "title": m.get("title")})
json.dump({"n": len(recs), "skipped": dict(skipped), "records": recs}, open(f"{HERE}/records/iclr2026_records.json", "w"), indent=1)
print("records", len(recs), "skipped", dict(skipped))
print("groups", dict(collections.Counter(r["group4"] for r in recs)))
print("rating dist", dict(sorted(collections.Counter(round(r["overall_mean"]) for r in recs).items())))
print("pangram fraction_ai == 0:", sum(1 for r in recs if r["fraction_ai"] == 0))
