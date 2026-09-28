"""
Build 2024/2025 ICLR label tables from HF mirrors (OpenReview api2 is Cloudflare-blocked here).
2024: smallari/openreview-iclr2024-peer-reviews-RAW  (id, decision, reviews[rating,confidence,text])
2025: QAQqaq/ICLR2025Openreview (id, ratings) + ai-conferences/ICLR2025 (id->accept+type tier, arxiv_id)
Out: data/_labels_2024.jsonl, data/_labels_2025.jsonl
     rows: {id, year, title, accept, tier, overall_mean, reviews:[{overall,confidence,text}], arxiv_id}
"""
import json
import os
import re
import math, statistics as st
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def first_int(s):
    m = re.search(r"-?\d+", str(s))
    return int(m.group(0)) if m else None


def tier_from(s):
    s = (s or "").lower()
    if "oral" in s or "talk" in s or "top-5" in s:
        return "oral"
    if "spotlight" in s or "notable" in s or "top-25" in s:
        return "spotlight"
    return "none"


def build_2024():
    ds = load_dataset("smallari/openreview-iclr2024-peer-reviews-RAW", split="train")
    out = {}
    for r in ds:
        pid = r["paper_id"]
        if pid in out:
            continue
        revs = []
        for rv in (r.get("reviews") or []):
            if not isinstance(rv, dict):
                continue
            ov = first_int(rv.get("rating"))
            revs.append({"overall": ov, "confidence": first_int(rv.get("confidence")),
                         "soundness": first_int(rv.get("soundness")) if rv.get("soundness") else None,
                         "text": " ".join(str(rv.get(k, "")) for k in
                                          ("summary", "strengths", "weaknesses", "questions", "review",
                                           "strengths_and_weaknesses") if rv.get(k))[:8000]})
        ovs = [x["overall"] for x in revs if isinstance(x["overall"], (int, float))]
        dec = str(r.get("decision") or "")
        out[pid] = {"id": pid, "year": 2024, "title": r.get("title"),
                    "accept": bool(r.get("label")) or dec.lower().startswith("accept"),
                    "tier": tier_from(dec), "decision_raw": dec,
                    "overall_mean": round(st.mean(ovs), 3) if ovs else None,
                    "soundness_mean": None, "reviews": revs, "arxiv_id": None}
    return list(out.values())


def build_2025():
    # accepted list (tier + arxiv_id) keyed by forum id
    acc = {}
    for r in load_dataset("ai-conferences/ICLR2025", split="train"):
        acc[r["paper_id"]] = {"tier": tier_from(r.get("type")), "arxiv_id": r.get("arxiv_id"),
                              "type": r.get("type")}
    out = []
    for r in load_dataset("QAQqaq/ICLR2025Openreview", split="train"):
        pid = r["id"]
        ratings = [x for x in (r.get("ratings") or []) if isinstance(x, (int, float))]
        a = acc.get(pid)
        revs = [{"overall": int(x), "confidence": None, "soundness": None, "text": ""} for x in ratings]
        out.append({"id": pid, "year": 2025, "title": r.get("title"),
                    "accept": a is not None, "tier": a["tier"] if a else "none",
                    "decision_raw": (a["type"] if a else "Reject"),
                    "overall_mean": round(r["avg_rating"], 3) if isinstance(r.get("avg_rating"),(int,float)) and not math.isnan(r["avg_rating"]) else
                    (round(st.mean(ratings), 3) if ratings else None),
                    "soundness_mean": None, "reviews": revs,
                    "arxiv_id": (a["arxiv_id"] if a and a.get("arxiv_id") else None)})
    return out


def save(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    dec = collections.Counter("accept" if r["accept"] else "reject" for r in rows)
    sc = collections.Counter(max(1, min(10, round(r["overall_mean"]))) for r in rows if isinstance(r["overall_mean"],(int,float)) and not math.isnan(r["overall_mean"]))
    ax = sum(1 for r in rows if r["arxiv_id"])
    print(f"{path}: {len(rows)} | {dict(dec)} | arxiv_id_known={ax} | scores={dict(sorted(sc.items()))}")


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    save(build_2024(), os.path.join(DATA, "_labels_2024.jsonl"))
    save(build_2025(), os.path.join(DATA, "_labels_2025.jsonl"))
