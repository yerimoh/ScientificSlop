"""Evidence records, four-state verdicts, and the shared aggregation rules.

Every checker writes the same record schema (slop_definitions_fars_only_0908.md section 13.1):

  {"instance_id": "FA0017:claim_table:0003", "type": "claim_table",
   "status": "violation" | "consistent" | "inconclusive" | "not_applicable",
   "locus_claim":    {"kind": "text", "span": [a, b], "quote": "..."},
   "locus_evidence": {"kind": "table", ...},
   "expected": ..., "reported": ..., "tolerance": ...,
   "checker": "claim_table_v2", "human_check": "pending"}

Aggregation rules shared by all eight checkers (slop_definition_revised_0909.md, last section):
  * a ratio whose denominator is zero is reported as None (N/A), never as 0;
  * "no evidence found", "extraction or interpretation failed" and "contradicted" are kept apart;
  * per-paper metrics keep their denominators so that group comparisons can be bootstrapped at
    the paper level.
"""
from __future__ import annotations
import json
import math
import os
import random
import statistics
from collections import Counter

VIOLATION, CONSISTENT, INCONCLUSIVE, NA = "violation", "consistent", "inconclusive", "not_applicable"


def instance(paper_id: str, typ: str, k: int, status: str, locus_claim: dict, locus_evidence: dict,
             expected=None, reported=None, tolerance=None, checker: str = "", **extra) -> dict:
    rec = {"instance_id": f"{paper_id}:{typ}:{k:04d}", "type": typ, "status": status,
           "locus_claim": locus_claim, "locus_evidence": locus_evidence,
           "expected": expected, "reported": reported, "tolerance": tolerance,
           "checker": checker, "human_check": "pending"}
    rec.update(extra)
    return rec


def rate(num: int, den: int):
    return None if not den else round(num / den, 4)


def write_jsonl(path: str, rows) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def paper_label(statuses: list[str]) -> str:
    """Paper-level four-state label from its instance statuses.
    positive if any violation; negative only if every checked relation is consistent
    (completeness condition); inconclusive if the checker could not decide anything; NA if there
    was nothing to check."""
    c = Counter(statuses)
    if not statuses or c[NA] == len(statuses):
        return "NA"
    if c[VIOLATION]:
        return "positive"
    if c[CONSISTENT] and not c[INCONCLUSIVE]:
        return "negative"
    if c[CONSISTENT] and c[INCONCLUSIVE]:
        return "negative_incomplete"
    return "inconclusive"


# --------------------------------------------------------------------------- statistics

def auroc(pos: list[float], neg: list[float]):
    if not pos or not neg:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1 if a > b else 0.5 if a == b else 0
    return round(wins / (len(pos) * len(neg)), 4)


def bootstrap_auroc(pos, neg, n_boot=1000, seed=0):
    if not pos or not neg:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        p = [rng.choice(pos) for _ in pos]
        q = [rng.choice(neg) for _ in neg]
        vals.append(auroc(p, q))
    vals.sort()
    return {"auroc": auroc(pos, neg), "ci95": [vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]]}


def cliff_delta(a: list[float], b: list[float]):
    if not a or not b:
        return None
    more = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    return round((more - less) / (len(a) * len(b)), 4)


def mannwhitney_p(a, b):
    try:
        from scipy.stats import mannwhitneyu
        if not a or not b:
            return None
        return round(float(mannwhitneyu(a, b, alternative="two-sided").pvalue), 5)
    except Exception:
        return None


def describe(vals: list[float]) -> dict:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return {"n": 0}
    return {"n": len(v), "mean": round(statistics.mean(v), 4), "median": round(statistics.median(v), 4),
            "min": round(min(v), 4), "max": round(max(v), 4)}


def compare_groups(name: str, ai: list[float], hu: list[float]) -> dict:
    """AI vs HU summary for one metric. Reported as an observation in the definition section,
    never as a benchmark label."""
    return {"metric": name, "AI": describe(ai), "HU": describe(hu),
            "cliff_delta_AI_minus_HU": cliff_delta(ai, hu), "mwu_p": mannwhitney_p(ai, hu),
            "auroc_HU_over_AI": auroc(hu, ai)}


def confound_audit(label_pos: list[float], label_neg: list[float], feature_name: str) -> dict:
    """AUROC of a trivial feature (word count, ref count ...) against the label, inside one
    corpus. Above 0.70 the type must be conditioned or flagged (section 13.4)."""
    a = auroc(label_pos, label_neg)
    return {"feature": feature_name, "auroc": a, "flag_over_0.70": (a is not None and a > 0.70),
            "pos_median": statistics.median(label_pos) if label_pos else None,
            "neg_median": statistics.median(label_neg) if label_neg else None}


# --------------------------------------------------------------------------- slop level (SLOP_SCORE.md)

def slop_score(numerator: int, denominator: int, min_units: int = 3) -> dict:
    """One-directional normalised score: failed units / checked units, higher = more slop.
    None when nothing was checked; `weak` flags a denominator below min_units (reported, not interpreted)."""
    if not denominator:
        return {"slop_score": None, "slop_numerator": numerator, "slop_denominator": denominator, "weak": True}
    return {"slop_score": round(numerator / denominator, 4), "slop_numerator": numerator, "slop_denominator": denominator,
            "weak": denominator < min_units}


def aggregate_scores(scores: list, skip_na: bool = False):
    """Unweighted mean of item (or plane) slop scores. N/A if any constituent is N/A unless skip_na."""
    vals = [s for s in scores if s is not None]
    if not vals or (not skip_na and len(vals) != len(scores)):
        return None
    return round(sum(vals) / len(vals), 4)
