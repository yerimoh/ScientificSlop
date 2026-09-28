"""Which figure of a paper is its method figure, decided from the caption list.

The paired census decided this by hand from the full caption list (LOCALISATION.md), because the
regex caption gate mis-chose or missed in many papers. For 1.4k ICLR papers the same decision is
delegated to the measurement LLM (Qwen2.5-32B, greedy, 3 runs, majority), reading only the caption
list. This is an extraction call, not a judgment of quality. `validate` scores the picker against the
144 hand decisions of the pairs (figure number or "none"); `iclr` writes results/figexp/picks.json.

  python3 figexp_pick.py validate
  python3 figexp_pick.py iclr
"""
import json, os, sys, time, glob
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOP = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/slop"
CENSUS = f"{SLOP}/Artifacts/candidte/_census_0914"
sys.path.insert(0, f"{SLOP}/_common"); sys.path.insert(0, os.environ.get("SCISLOP_ROOT", ".") + "/artifact-ai2science/_llm")
sys.path.insert(0, CENSUS)
import llm as LLM
import llm_backend as B

TAG = "figexp_pick_v1"
PROMPT = """Below is the list of figure captions of a machine-learning research paper, in order.

{captions}

Which ONE figure is the paper's METHOD FIGURE: the figure that depicts the proposed method, model,
framework, system, pipeline or architecture itself (an overview or schematic of how the method works)?

Do NOT choose: result plots or tables of numbers, qualitative examples or samples, a teaser that only
motivates the problem, a figure showing a baseline or prior method only, datasets or statistics,
attention maps, or figures in the appendix.

If the paper has no such figure, answer null.
Return strict JSON only: {{"figure": <integer figure number or null>}}"""


def fmt(caps, max_words=60):
    lines = []
    seen = set()
    for c in caps:
        if c["num"] in seen:
            continue
        seen.add(c["num"])
        words = c["text"].split()[:max_words]
        lines.append(f"Figure {c['num']}: " + " ".join(words))
    return "\n".join(lines)


def ask(prompt, run, tries=5):
    cp = os.path.join(LLM.CACHE_DIR, LLM._key(prompt, None, f"{TAG}#{run}") + ".json")
    if os.path.exists(cp):
        return json.load(open(cp))
    for t in range(tries):
        try:
            out = B.llm_json(prompt, default=None, max_tokens=30)
        except Exception:
            out = None
        if isinstance(out, dict) and "figure" in out:
            json.dump(out, open(cp, "w")); return out
        time.sleep(5 * (t + 1))
    return None


def pick(caps, runs=3):
    """-> (figure number as str, or None for 'no method figure', or 'unanswered'), per-run answers."""
    if not caps:
        return None, []
    p = PROMPT.format(captions=fmt(caps))
    got = []
    for r in range(runs):
        o = ask(p, r)
        if o is None:
            got.append("unanswered"); continue
        v = o.get("figure")
        got.append(str(int(v)) if isinstance(v, (int, float)) and not isinstance(v, bool) else (str(v).strip() if isinstance(v, str) and v.strip().isdigit() else None))
    valid = [g for g in got if g != "unanswered"]
    if not valid:
        return "unanswered", got
    top = Counter(valid).most_common(1)[0][0]
    nums = {c["num"] for c in caps}
    if top is not None and top not in nums:
        top = None
    return top, got


def validate():
    import pairs165_locate_hu_figs as L
    man = json.load(open(f"{CENSUS}/pairs165_full/data/pairs165_manifest.json"))
    ov = json.load(open(f"{CENSUS}/pairs165_hu_overrides.json"))
    rows = []
    def one(m):
        hand = ov.get(m["arxiv"])
        if hand is None:
            return None
        handnum = None if hand == "none" else str(hand["fig"] if isinstance(hand, dict) else hand)
        caps = L.find_captions(L.words_by_page(m["pdf"]))
        caps = [{"num": c["num"], "text": c["text"]} for c in caps]
        auto, got = pick(caps)
        gated = [c["num"] for c in caps if (L.STRONG.search(c["text"]) and not L.RESULTY.search(c["text"][:60])) or L.METHODY.search(c["text"])]
        return {"arxiv": m["arxiv"], "hand": handnum, "llm": auto, "llm_runs": got, "regex": gated[0] if gated else None, "n_captions": len(caps)}
    with ThreadPoolExecutor(8) as ex:
        rows = [r for r in ex.map(one, man) if r]
    def tab(key):
        c = Counter()
        for r in rows:
            h, a = r["hand"], r[key]
            if a == "unanswered": c["unanswered"] += 1
            elif h is None and a is None: c["none_none"] += 1
            elif h is None: c["hand_none_auto_fig"] += 1
            elif a is None: c["hand_fig_auto_none"] += 1
            elif a == h: c["agree"] += 1
            else: c["disagree"] += 1
        return dict(c)
    out = {"n": len(rows), "llm": tab("llm"), "regex": tab("regex"), "rows": rows,
           "note": "hand = pairs165_hu_overrides.json (figure number or none); llm = Qwen2.5-32B picker (3 runs majority); regex = census caption gate, first gated caption"}
    os.makedirs(f"{R}/results/figexp", exist_ok=True)
    json.dump(out, open(f"{R}/results/figexp/pick_validation.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


def iclr():
    files = sorted(glob.glob(f"{R}/results/figexp/captions/*.json"))
    prev = {}
    pth = f"{R}/results/figexp/picks.json"
    if os.path.exists(pth):
        prev = json.load(open(pth))
    def one(f):
        j = json.load(open(f))
        if "captions" not in j:
            return j["id"], {"pick": "error", "runs": [], "n_captions": 0}
        caps = [{"num": c["num"], "text": c["text"]} for c in j["captions"]]
        a, got = pick(caps)
        return j["id"], {"pick": a, "runs": got, "n_captions": len({c["num"] for c in caps}),
                         "caption": next((c["text"][:200] for c in caps if c["num"] == a), None)}
    todo = [f for f in files if os.path.basename(f)[:-5] not in prev or prev[os.path.basename(f)[:-5]]["pick"] == "unanswered"]
    with ThreadPoolExecutor(16) as ex:
        for k, (pid, row) in enumerate(ex.map(one, todo)):
            prev[pid] = row
            if k % 100 == 0:
                json.dump(prev, open(pth, "w"), indent=0)
    json.dump(prev, open(pth, "w"), indent=0)
    print("picks:", len(prev), Counter("none" if v["pick"] is None else ("fig" if v["pick"] not in ("unanswered", "error") else v["pick"]) for v in prev.values()))


if __name__ == "__main__":
    {"validate": validate, "iclr": iclr}[sys.argv[1]]()
