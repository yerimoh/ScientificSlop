"""Extra transcription workers for the ICLR method-figure crops (same instrument as figexp_transcribe.py).
Walk the manifest in DESCENDING order, worker i of n taking positions with index % n == i, and re-read every
transcripts*.jsonl before each figure so nothing already read by another worker is read twice.
-> results/figexp/transcripts.w<i>.jsonl.   python3 figexp_transcribe2.py --worker i --nworkers n"""
import argparse, glob, json, os, sys, time
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figexp_transcribe as T1
from pairs165_transcribe import load
import pairs165_retranscribe as RT


def done_keys():
    ks = set()
    for f in glob.glob(f"{R}/results/figexp/transcripts*.jsonl"):
        for l in open(f):
            try:
                ks.add(json.loads(l)["key"])
            except Exception:
                pass
    return ks


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--worker", type=int, required=True); ap.add_argument("--nworkers", type=int, required=True)
    a = ap.parse_args()
    figs = T1.manifest()[::-1]
    mine = [f for k, f in enumerate(figs) if k % a.nworkers == a.worker]
    outp = f"{R}/results/figexp/transcripts.w{a.worker}.jsonl"
    print(f"worker {a.worker}/{a.nworkers}: {len(mine)} candidates of {len(figs)}", flush=True)
    t0 = time.time(); n = 0
    with open(outp, "a", encoding="utf-8") as fo:
        for f in mine:
            if f["key"] in done_keys():
                continue
            try:
                answer, ntok = RT.ask_full(load(f["path"]), T1.PROMPT)
                lines, how = RT.parse(answer)
                rec = {"key": f["key"], "id": f["id"], "path": f["path"], "result": {"lines": lines}, "parse": how,
                       "n_tokens": ntok, "hit_cap": ntok >= RT.MAX_NEW, "full_raw": answer}
            except Exception as e:
                rec = {"key": f["key"], "id": f["id"], "path": f["path"], "result": {"error": repr(e)[:200]}}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush(); n += 1
            if n % 10 == 0:
                print(f"{n} done, last {f['key']} {rec.get('parse')} {(time.time() - t0) / n:.0f}s/fig", flush=True)
    print("WORKER DONE", n, flush=True)


if __name__ == "__main__":
    main()
