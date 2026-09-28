r"""Build the ICLR record list for the review-score staircase (ANALYSIS_SECTION_PLAN_0914 §3, paragraph 3, evidence (b)).

Input : artifact-ai2science/Evaluation/ICLR/data/{year}/{note_id}/{meta.json, tex/}
Output: records/iclr_records.json  (one record per paper whose tex directory has a \documentclass main file)

Decision groups (from meta.json["decision"]):
  reject    accept=false, venue in {Reject, Invite to Workshop Track}
  poster    accept=true and no oral/spotlight tier
  spotlight tier spotlight, or venue notable-top-25%
  oral      tier oral, or venue Oral / Talk / notable-top-5%
  undecided accept=false with an empty venue (2024 only; no decision recorded) -> kept in the file, excluded from every statistic
The paper figure uses four groups: FARS / reject / accept (= poster + spotlight) / oral.
"""
import glob, json, os, re, sys, collections

ROOT = os.environ.get("SCISLOP_ROOT", ".")
D = f"{ROOT}/artifact-ai2science/Evaluation/ICLR/data"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/paper/draft_v6/slop/_common")
from corpus import hu_main_tex  # noqa: E402


def group_of(dec: dict) -> str:
    acc = bool(dec.get("accept"))
    tier = dec.get("tier") or ""
    venue = (dec.get("venue") or "").lower()
    if not acc:
        return "undecided" if venue == "" else "reject"
    if tier == "oral" or "oral" in venue or "talk" in venue or "top-5%" in venue:
        return "oral"
    if tier == "spotlight" or "spotlight" in venue or "top-25%" in venue:
        return "spotlight"
    return "poster"


def main():
    recs, skipped = [], collections.Counter()
    for y in range(2017, 2027):
        for d in sorted(glob.glob(f"{D}/{y}/*")):
            mp = f"{d}/meta.json"
            if not os.path.isfile(mp) or not os.path.isdir(f"{d}/tex"):
                skipped["no_tex_dir"] += 1
                continue
            m = json.load(open(mp))
            main = hu_main_tex(f"{d}/tex")
            if not main:
                skipped["no_documentclass"] += 1
                continue
            dec = m.get("decision") or {}
            g = group_of(dec)
            revs = m.get("reviews") or []
            recs.append({
                "corpus": "HU", "id": f"iclr{y}_{m['note_id']}", "note_id": m["note_id"], "year": y,
                "root": f"{d}/tex", "main_tex": main, "paper_dir": f"{d}/tex", "exp_dir": None, "diagram": None,
                "group5": g, "group4": {"poster": "accept", "spotlight": "accept"}.get(g, g),
                "accept": bool(dec.get("accept")), "tier": dec.get("tier"), "venue": dec.get("venue"),
                "overall_mean": m.get("overall_mean"), "n_reviews": len(revs),
                "arxiv_id": m.get("arxiv_id"), "title": m.get("title"),
            })
    json.dump({"n": len(recs), "skipped": dict(skipped), "records": recs},
              open(f"{HERE}/records/iclr_records.json", "w"), indent=1)
    c = collections.Counter((r["year"], r["group4"]) for r in recs)
    years = sorted({r["year"] for r in recs})
    print("year   reject accept oral undecided total")
    for y in years:
        print(f"{y}   {c[(y,'reject')]:6d} {c[(y,'accept')]:6d} {c[(y,'oral')]:4d} {c[(y,'undecided')]:9d} {sum(v for k,v in c.items() if k[0]==y):5d}")
    tot = collections.Counter(r["group4"] for r in recs)
    print("total ", dict(tot), "skipped", dict(skipped))
    print("group5", dict(collections.Counter(r["group5"] for r in recs)))
    print("rating missing:", sum(1 for r in recs if r["overall_mean"] is None))


if __name__ == "__main__":
    main()
