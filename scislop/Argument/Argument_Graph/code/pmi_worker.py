#!/usr/bin/env python3
"""Fill the PMI cache of Argument_Graph for a list of paper records, on one GPU.

The PMI matrix of an introduction depends only on its sentence texts (lm_score.pmi_matrix), not on the
labels, so this stage runs in parallel with the labelling. Same scorer, same batch size, same cache key
as measure.py; a later `measure.py --stage pmi` finds every matrix on disk and needs no GPU.

  python3 pmi_worker.py --papers A.json [--papers B.json] --shard i --nshards n [--log FILE]
Own shard first, then a sweep over whatever is still missing, so extra workers only shorten the tail.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "_common"))
import importlib.util
_spec = importlib.util.spec_from_file_location("ag_measure", os.path.join(HERE, "measure.py"))
AG = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(AG)
from views import load_doc
import lm_score as LMS


def cache_path(texts):
    key = hashlib.sha256((LMS.MODEL + "\n" + "\n\x1e".join(texts)).encode()).hexdigest()
    return os.path.join(LMS.CACHE_DIR, key + ".json")


def texts_of(rec):
    try:
        doc = load_doc(rec); intro = AG.intro_text(doc)
    except Exception:
        return None
    if not intro or len(intro.split()) < 80:
        return None
    sents = AG.intro_sentences(intro)
    if len(sents) < 4:
        return None
    return [s["text"] for s in sents]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", action="append", required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--log", default="")
    a = ap.parse_args()
    logf = open(a.log, "a") if a.log else sys.stdout
    def log(s):
        logf.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {s}\n"); logf.flush()
    papers = []
    for p in a.papers:
        papers += json.load(open(p))
    mine = [r for k, r in enumerate(papers) if k % a.nshards == a.shard]
    log(f"shard {a.shard}/{a.nshards}: {len(mine)} of {len(papers)} papers; device {LMS.device()} model {LMS.MODEL}")
    done = skipped = 0; t0 = time.time()
    for phase, lst in (("own", mine), ("sweep", papers)):
        for r in lst:
            tx = texts_of(r)
            if tx is None:
                skipped += 1; continue
            if os.path.exists(cache_path(tx)):
                continue
            t1 = time.time()
            try:
                LMS.pmi_matrix(tx)
            except Exception as e:
                log(f"FAIL {r.get('id')} {e!r}"[:200]); continue
            done += 1
            if done % 10 == 0 or time.time() - t1 > 120:
                log(f"{phase} {done} matrices, last {r.get('id')} n={len(tx)} {time.time() - t1:.0f}s, {(time.time() - t0) / 60:.0f} min")
    log(f"done: {done} matrices computed, {skipped} skipped, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
