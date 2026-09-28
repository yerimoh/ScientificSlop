"""Pangram 3 on the pooled SciSlopBench (bench165 + benchA4S), for the Pangram row of Table 2.

Reads the exact detector input (scripts/texts.jsonl of each half, the prose view that Binoculars /
DetectGPT / NTS scored), submits each paper to Pangram's async task endpoint with model "default",
which this key resolves to Pangram 3 (response version 3.3.2, probed 2026-09-24; the key has no
"pangram-3" selector, only "default" and "pangram-4"). Billing: $0.05 per started 1,000-word block.

Raw responses are cached one JSON per paper under <half>/results/pangram_raw/<id>.json, so a re-run
resumes and never pays twice. Detector rows go to bench165/results/pangram.jsonl and
benchA4S/results/baselines/pangram.jsonl in the same shape as binoculars.jsonl, field "pangram"
= fraction_ai (+ fraction_ai_assisted, fraction_human, n_words, version). Higher = more AI, d=+1.

  python3 pangram_bench.py            # dry run: counts words, cost, what is cached
  python3 pangram_bench.py --go       # submit (stops on 402 so nothing more is billed)
  python3 pangram_bench.py --table    # Table 2 metrics for the Pangram row (pooled 390 pairs)
"""
import argparse, json, math, os, sys, time, urllib.error, urllib.request

SB = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench"
KEYFILE = os.environ.get("SCISLOP_ROOT", ".") + "/paper/api_pangram.txt"
ENDPOINT = "https://text.external-api.pangram.com/task"
MODEL = "default"          # == Pangram 3 for this key (version 3.3.2)
RATE_PER_BLOCK, BLOCK = 0.05, 1000
HALVES = [("165", f"{SB}/bench165", f"{SB}/bench165/results/pangram.jsonl"),
          ("a4s", f"{SB}/benchA4S", f"{SB}/benchA4S/results/baselines/pangram.jsonl")]


def key():
    return open(KEYFILE).read().strip()


def submit(text, k):
    req = urllib.request.Request(ENDPOINT, method="POST",
                                 data=json.dumps({"text": text, "model": MODEL}).encode(),
                                 headers={"x-api-key": k, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def poll(task_id, k, timeout=900):
    req = urllib.request.Request(f"{ENDPOINT}/{task_id}", headers={"x-api-key": k})
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = json.load(r)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(10); continue
        stage = body.get("stage") or ""
        if stage == "STAGE_SUCCESS":
            return body
        if "FAIL" in stage or "ERROR" in stage:
            raise RuntimeError(f"{task_id}: {stage}")
        time.sleep(4)
    raise TimeoutError(task_id)


def rows_of(root):
    return [json.loads(l) for l in open(f"{root}/scripts/texts.jsonl")]


def write_jsonl(root, out):
    raw = f"{root}/results/pangram_raw"
    n = 0
    with open(out, "w") as w:
        for r in rows_of(root):
            p = f"{raw}/{r['id']}.json"
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            w.write(json.dumps({"id": r["id"], "year": 0, "source": r["source"],
                                "pangram": d.get("fraction_ai"),
                                "fraction_ai_assisted": d.get("fraction_ai_assisted"),
                                "fraction_human": d.get("fraction_human"),
                                "n_windows": len(d.get("windows") or []),
                                "n_words": d.get("_n_words"), "version": d.get("version")}) + "\n")
            n += 1
    return n


def run(go):
    k = key() if go else None
    todo, words, blocks, cached = [], 0, 0, 0
    for tag, root, out in HALVES:
        raw = f"{root}/results/pangram_raw"; os.makedirs(raw, exist_ok=True)
        for r in rows_of(root):
            p = f"{raw}/{r['id']}.json"
            if os.path.exists(p):
                cached += 1; continue
            nw = len(r["text"].split())
            todo.append((tag, root, r, p)); words += nw; blocks += math.ceil(nw / BLOCK)
    print(f"cached {cached}, to score {len(todo)}, {words:,} words, {blocks} billable blocks "
          f"-> ${blocks*RATE_PER_BLOCK:,.2f} at Pangram 3 rates", flush=True)
    if not go:
        print("dry run — pass --go to submit"); return
    n = len(todo)
    for i, (tag, root, r, p) in enumerate(todo, 1):
        text = r["text"]
        for attempt in range(4):
            try:
                task = submit(text, k)
                res = poll(task["task_id"], k)
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="ignore")[:200]
                print(f"[{i}/{n}] {tag} {r['id']} HTTP {e.code} {body}", flush=True)
                if e.code == 402:
                    sys.exit("out of credits — stopping so the remaining papers stay unbilled")
                if e.code == 429 or e.code >= 500:
                    time.sleep(30 * (attempt + 1)); continue
                res = None; break
            except Exception as e:
                print(f"[{i}/{n}] {tag} {r['id']} {type(e).__name__}: {e}", flush=True)
                time.sleep(15); res = None
        else:
            res = None
        if res is None:
            continue
        res["_paper_id"] = r["id"]; res["_n_words"] = len(text.split())
        json.dump(res, open(p, "w"))
        print(f"[{i}/{n}] {tag} {r['id']} ok v{res.get('version')} fraction_ai={res.get('fraction_ai')}",
              flush=True)
        if i % 20 == 0:
            for _t, _root, out in HALVES:
                write_jsonl(_root, out)
    for _t, _root, out in HALVES:
        print(f"wrote {write_jsonl(_root, out)} rows -> {out}", flush=True)
    print("PANGRAM_DONE", flush=True)


def table():
    """Same pooling/metric logic as table_pooled.py (det_pool + metrics), d=+1."""
    import statistics as st
    by = {}
    for tag, root, out in HALVES:
        items = json.load(open(f"{root}/{'items165' if tag == '165' else 'itemsA4S'}.json"))["items"]
        idk = {i["item_id"]: (f"{tag}:{i['pair']}", i["label"]) for i in items}
        if not os.path.isfile(out):
            continue
        for line in open(out):
            r = json.loads(line); kk = idk.get(r["id"])
            if kk and r.get("pangram") is not None:
                by.setdefault(kk[0], {})[kk[1]] = float(r["pangram"])
    d = +1
    both = [(v[1], v[0]) for v in by.values() if 1 in v and 0 in v]
    w = sum(1 for a, h in both if (a - h) * d > 0); t = sum(1 for a, h in both if a == h)
    ai = [v[1] for v in by.values() if 1 in v]; hu = [v[0] for v in by.values() if 0 in v]
    au = sum(1.0 if (a - b) * d > 0 else 0.5 if a == b else 0.0 for a in ai for b in hu) / (len(ai) * len(hu))
    hs = sorted(hu, reverse=True); thr = hs[max(0, int(0.05 * len(hs)) - 1)]
    tpr = sum(1 for x in ai if x > thr) / len(ai)
    print(f"Pangram (pooled): pairs={len(both)} ai={len(ai)} hu={len(hu)}  "
          f"PairAcc={(w+0.5*t)/len(both):.3f}  AUROC={au:.3f}  TPR@5%FPR={tpr:.3f}  (thr={thr:.3f})")
    print(f"mean fraction_ai  AI={st.mean(ai):.3f}  HU={st.mean(hu):.3f}; "
          f"AI papers with fraction_ai==0: {sum(1 for x in ai if x == 0)}/{len(ai)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true"); ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    table() if a.table else run(a.go)
