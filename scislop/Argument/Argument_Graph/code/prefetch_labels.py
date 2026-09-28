#!/usr/bin/env python3
"""Warm the label cache of Argument_Graph for an arbitrary list of paper records, many papers at once.

measure.py labels one paper at a time (8 threads inside a paper). This driver builds the very same
prompts (same template, same numbering, same cache key through _common/llm) for every sentence of
every paper in a list and sends them to the vLLM server with one large thread pool, so a corpus of
thousands of introductions is labelled in hours instead of days. It writes nothing but cache files,
so a later `measure.py --stage pmi` (or the experiment wrappers) reads exactly what it would have
asked itself.

Guard that measure.py's helper lacks: an empty or malformed server answer is retried and never
written to the cache. (llm.llm_json would cache a None, which label_one then reads as "none".)

  python3 prefetch_labels.py --papers LIST.json [--workers 64] [--runs 3] [--log FILE]
LIST.json = [{"corpus":..,"id":..,"root":..,"main_tex":..,"paper_dir":..}, ...] (load_doc records).
"""
from __future__ import annotations
import argparse, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "_common"))
sys.path.insert(0, os.environ.get("SCISLOP_ROOT", ".") + "/artifact-ai2science/_llm")
import importlib.util
_spec = importlib.util.spec_from_file_location("ag_measure", os.path.join(HERE, "measure.py"))
AG = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(AG)
from views import load_doc
import llm as LLM
import llm_backend as B

TPL = open(os.path.join(HERE, "PROMPT_label.txt")).read()
LOCK = threading.Lock()
STAT = {"cached": 0, "fetched": 0, "failed": 0, "papers": 0, "skipped": 0}


def prompt_for(sents, i):
    return TPL.replace("{numbered}", AG.numbered(sents)).replace("{i}", str(i + 1)).replace("{target}", sents[i]["text"])


def cache_path(prompt, run):
    return os.path.join(LLM.CACHE_DIR, LLM._key(prompt, None, f"{AG.LABEL_TAG}#{run}") + ".json")


def fetch(prompt, run, tries=6):
    cp = cache_path(prompt, run)
    if os.path.exists(cp):
        with LOCK: STAT["cached"] += 1
        return True
    for t in range(tries):
        try:
            out = B.llm_json(prompt, default=None, max_tokens=40)
        except Exception:
            out = None
        lab = out.get("label") if isinstance(out, dict) else None
        if isinstance(lab, str) and lab.strip().lower() in AG.LABELS:
            tmp = cp + f".tmp{os.getpid()}{threading.get_ident()}"
            json.dump(out, open(tmp, "w"), ensure_ascii=False); os.replace(tmp, cp)
            with LOCK: STAT["fetched"] += 1
            return True
        time.sleep(min(60, 5 * (t + 1)))
    with LOCK: STAT["failed"] += 1
    return False


def tasks_for(rec, runs):
    try:
        doc = load_doc(rec); intro = AG.intro_text(doc)
    except Exception as e:
        return None, f"load_error {e!r}"[:120]
    if not intro or len(intro.split()) < 80:
        return None, "no_intro"
    sents = AG.intro_sentences(intro)
    if len(sents) < 4:
        return None, "too_short"
    return [(prompt_for(sents, i), r) for i in range(len(sents)) for r in range(runs)], f"{len(sents)} sentences"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", required=True); ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3); ap.add_argument("--log", default="")
    ap.add_argument("--reverse", action="store_true", help="walk the list from the end (a second prefetcher on another server)")
    a = ap.parse_args()
    papers = json.load(open(a.papers))
    if a.reverse:
        papers = papers[::-1]
    logf = open(a.log, "a") if a.log else sys.stdout
    def log(s):
        logf.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {s}\n"); logf.flush()
    log(f"start {len(papers)} papers, workers {a.workers}, runs {a.runs}, endpoint {B.endpoint()}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        pending = []
        for k, rec in enumerate(papers):
            tk, note = tasks_for(rec, a.runs)
            if tk is None:
                STAT["skipped"] += 1; log(f"skip {rec.get('id')} {note}"); continue
            futs = [ex.submit(fetch, p, r) for p, r in tk]
            pending.append((rec.get("id"), futs))
            # keep the queue bounded: wait for older papers to finish
            while len(pending) > 6:
                pid, fs = pending.pop(0)
                ok = all(f.result() for f in fs)
                STAT["papers"] += 1
                if STAT["papers"] % 20 == 0 or not ok:
                    dt = time.time() - t0
                    log(f"{STAT['papers']}/{len(papers)} papers  fetched {STAT['fetched']} cached {STAT['cached']} failed {STAT['failed']}  "
                        f"{STAT['fetched'] / max(dt, 1):.1f} calls/s  last {pid} {'ok' if ok else 'INCOMPLETE'}")
        for pid, fs in pending:
            ok = all(f.result() for f in fs); STAT["papers"] += 1
            if not ok: log(f"INCOMPLETE {pid}")
    dt = time.time() - t0
    log(f"done {STAT} in {dt / 60:.1f} min")
    if not a.reverse:
        open(a.papers + ".done", "w").write(json.dumps({**STAT, "minutes": round(dt / 60, 1)}))


if __name__ == "__main__":
    main()
