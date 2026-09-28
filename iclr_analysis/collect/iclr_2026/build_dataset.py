"""
Build the ICLR-2026 x Pangram scaled dataset.

Sources (all free, no API credits -- Pangram was the official AI-detection partner for
ICLR 2026 and published per-submission / per-review verdicts):
  data/pangram_submissions.json  https://iclr.pangram.com/data/submissions.json  (19,490 submissions)
  data/pangram_reviews.json      https://iclr.pangram.com/data/reviews.json      (75,800 reviews)
  data/iclr2026_meta.parquet     HF ai-conferences/ICLR2026                      (5,352 accepted + arxiv_id)

Out:
  data/papers_2026.jsonl   one row per submission: pangram fraction_ai + rating + decision + arxiv_id
  data/reviews_2026.jsonl  one row per review: pangram 5-class + rating/soundness/presentation/contribution
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# Pangram's 5 authorship classes, ordered
PRED_ORDER = ["Fully human-written", "Lightly AI-edited", "Moderately AI-edited",
              "Heavily AI-edited", "Fully AI-generated"]
PRED_IDX = {p: i for i, p in enumerate(PRED_ORDER)}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build():
    subs = json.load(open(f"{DATA}/pangram_submissions.json"))
    revs = json.load(open(f"{DATA}/pangram_reviews.json"))

    import pandas as pd
    meta = pd.read_parquet(f"{DATA}/iclr2026_meta.parquet")
    accepted = {r.paper_id: r for r in meta.itertuples()}

    # per-paper review aggregates
    by_paper = {}
    for r in revs:
        by_paper.setdefault(r["submission_id"], []).append(r)

    rows = []
    for s in subs:
        sid = s["submission_id"]
        acc = accepted.get(sid)
        rv = by_paper.get(sid, [])
        preds = [PRED_IDX[r["prediction_display"]] for r in rv if r.get("prediction_display") in PRED_IDX]
        ratings = [_f(r.get("rating")) for r in rv]
        ratings = [x for x in ratings if x is not None]
        rows.append({
            "submission_id": sid,
            "submission_number": s.get("submission_number"),
            "title": s.get("title"),
            "abstract": s.get("abstract"),
            # Pangram, computed on the full submission text
            "fraction_ai": s.get("fraction_ai"),
            "avg_rating": s.get("avg_rating"),
            "n_reviews": len(rv),
            "rating_mean": round(st.mean(ratings), 3) if ratings else None,
            "rating_min": min(ratings) if ratings else None,
            "rating_max": max(ratings) if ratings else None,
            "soundness_mean": _mean_field(rv, "soundness"),
            "presentation_mean": _mean_field(rv, "presentation"),
            "contribution_mean": _mean_field(rv, "contribution"),
            "confidence_mean": _mean_field(rv, "confidence"),
            # review-side AI-ness (0=all human .. 4=all fully AI)
            "review_ai_mean": round(st.mean(preds), 3) if preds else None,
            "review_ai_frac": round(sum(1 for p in preds if p >= 1) / len(preds), 3) if preds else None,
            "review_fully_ai_frac": round(sum(1 for p in preds if p == 4) / len(preds), 3) if preds else None,
            # decision
            "accept": acc is not None,
            "tier": (acc.type.lower() if acc is not None else "reject"),
            "arxiv_id": (acc.arxiv_id if acc is not None and isinstance(acc.arxiv_id, str) else None),
            "primary_area": (acc.primary_area if acc is not None else None),
            "dashboard_link": s.get("dashboard_link"),
        })

    with open(f"{DATA}/papers_2026.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(f"{DATA}/reviews_2026.jsonl", "w") as f:
        for r in revs:
            f.write(json.dumps({
                "review_id": r.get("review_id"),
                "submission_id": r.get("submission_id"),
                "submission_number": r.get("submission_number"),
                "prediction": r.get("prediction_display"),
                "pred_idx": PRED_IDX.get(r.get("prediction_display")),
                "rating": _f(r.get("rating")),
                "confidence": _f(r.get("confidence")),
                "soundness": _f(r.get("soundness")),
                "presentation": _f(r.get("presentation")),
                "contribution": _f(r.get("contribution")),
                "n_chars": len(r.get("text") or ""),
            }, ensure_ascii=False) + "\n")

    n_acc = sum(1 for r in rows if r["accept"])
    n_pgm = sum(1 for r in rows if r["fraction_ai"] is not None)
    print(f"papers={len(rows)} accepted={n_acc} with_pangram={n_pgm} with_arxiv={sum(1 for r in rows if r['arxiv_id'])}")
    print(f"reviews={len(revs)}")


def _mean_field(rv, key):
    vs = [_f(r.get(key)) for r in rv]
    vs = [v for v in vs if v is not None]
    return round(st.mean(vs), 3) if vs else None


if __name__ == "__main__":
    build()
