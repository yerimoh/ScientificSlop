#!/usr/bin/env python3
"""PMI worker held open on one GPU.

Argument_Graph's second stage scores every ordered sentence pair of an introduction with a local
7B LM. Loading that model costs minutes and the harness pays it once per round, which is most of a
round's clock. The daemon loads it once and then answers from a warm process.

The channel is the shared filesystem, not a socket, because `lm_score.pmi_matrix` already caches to
disk under a key derived from the model name and the texts. A client drops its texts in
`_pmi_cache/_requests/<key>.json`; the daemon computes the matrix, which writes `<key>.json` next to
it, and the client's own cache lookup then succeeds. Nothing about how a number is produced changes.
"""
import json, os, sys, time, traceback
from pathlib import Path

COMMON = Path(os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/slop/_common")
sys.path.insert(0, str(COMMON))
import lm_score as LMS

REQ = COMMON / '_pmi_cache' / '_requests'
REQ.mkdir(parents=True, exist_ok=True)
DONE = REQ / 'done'; DONE.mkdir(exist_ok=True)


def main():
    t0 = time.time()
    print(f'loading {LMS.MODEL}', flush=True)
    LMS.device()
    print(f'ready on {LMS.device()} in {time.time()-t0:.0f}s', flush=True)
    (REQ / 'READY').write_text(str(os.getpid()))
    while True:
        reqs = sorted(p for p in REQ.glob('*.json'))
        if not reqs:
            time.sleep(1.0); continue
        for p in reqs:
            try:
                texts = json.loads(p.read_text())['texts']
            except Exception:
                p.unlink(missing_ok=True); continue
            s = time.time()
            try:
                LMS.pmi_matrix(texts)
                print(f'{p.name} n={len(texts)} {time.time()-s:.1f}s', flush=True)
            except Exception:
                traceback.print_exc(); print(f'{p.name} failed', flush=True)
            p.replace(DONE / p.name)


if __name__ == '__main__':
    main()
