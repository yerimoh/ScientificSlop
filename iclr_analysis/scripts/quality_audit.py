"""Outcome-independent extraction-quality audit of the ICLR papers (rules fixed 0915).
Writes results/quality_flags.json and records/exclude_quality.json."""
import json, re, collections, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
recs = {r["id"]: r for r in json.load(open(f"{HERE}/records/iclr_records.json"))["records"]}
def rows(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p)}
M = rows(f"{HERE}/results/slop/macro_redund/papers.jsonl"); X = rows(f"{HERE}/results/slop/xsec_ref/papers.jsonl"); C = rows(f"{HERE}/results/slop/citation/papers.jsonl")
T = {json.loads(l)["id"]: json.loads(l)["body_words"] for l in open(f"{HERE}/records/texts_iclr.jsonl")}
flags = collections.defaultdict(set)
for i, r in recs.items():
    if r["group4"] == "undecided": continue
    m, x, c = M.get(i), X.get(i), C.get(i)
    if m and m.get("source_incomplete"): flags["tex_incomplete(missing \\input)"].add(i)
    if c and c.get("status") == "no_sections": flags["citation:no_sections"].add(i)
    if m and (m.get("n_sections") or 0) < 3: flags["macro:<3 sections"].add(i)
    if x and (x.get("n_sections_body") or 0) < 3: flags["xsec:<3 body sections"].add(i)
    if x and (x.get("n_objects") or 0) < 5: flags["xsec:<5 objects"].add(i)
    if T.get(i, 0) < 1500: flags["prose view <1500 words"].add(i)
    if m and m.get("weak"): flags["macro:weak"].add(i)
    if x and x.get("weak"): flags["xsec:weak"].add(i)
    if c and c.get("weak"): flags["citation:weak(<8 units)"].add(i)
    try:
        tex = open(r["main_tex"], errors="ignore").read()
        mt = re.search(r"\\title\s*(?:\[[^\]]*\])?\s*\{(.{3,300}?)\}", tex, re.S)
        if mt and re.search(r"supplement|appendix|rebuttal|response to review", mt.group(1), re.I): flags["main tex is supplement"].add(i)
    except Exception: pass
ex = set().union(*flags.values()) if flags else set()
json.dump({k: sorted(v) for k, v in flags.items()}, open(f"{HERE}/results/quality_flags.json", "w"), indent=1)
json.dump({"rules": sorted(flags), "n_excluded": len(ex), "ids": sorted(ex)}, open(f"{HERE}/records/exclude_quality.json", "w"), indent=1)
n_dec = sum(1 for r in recs.values() if r["group4"] != "undecided")
print("decided", n_dec, "excluded", len(ex), "kept", n_dec - len(ex), {k: len(v) for k, v in flags.items()})
