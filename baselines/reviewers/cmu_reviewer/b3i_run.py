"""B3i baseline — Reviewer-loop, backbone = CMU Paper Reviewer System, ported onto
Qwen (see cmu_reviewer.py header). Everything except review generation is identical
to B3/B3a: stage / run_claude / judge / paper_dir reused from the ES survival code,
rewrite prompt shared, corpus = same 165 tex-eligible pairs, R5. The swapped block
(3a) produces a markdown review of <=5 critical issues (Claim/Evidence/Action).
HARD RULE: no mold concept anywhere in the loop.

Usage:  python3 b3i_run.py [N_rounds=5] [limit=0(all)] [shard_i] [K]
Env:    B3I_MAX_ITEMS (5), B3I_PRESET (neurips|nature).
"""
from __future__ import annotations
import os
import json, csv, sys, time
from pathlib import Path

ES_CODE = (os.environ.get("SCISLOP_ROOT", ".") + "/artifact-ai2science/"
           'Content_Mold/Evaluation_Surface_Area/results/code')
sys.path.insert(0, ES_CODE)
import eslib                      # noqa: E402
import iter_run as ir             # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cmu_reviewer as srev       # noqa: E402  the swapped-in review backbone

TAG = 'b3i'
RUNS = HERE / 'runs'
PAIRS_JSON = (os.environ.get("SCISLOP_ROOT", ".") + "/artifact-ai2science/"
              'Evaluation/00_DATA_AUDIT/pairs_final.json')
CORPUS = [q['code'] for q in json.load(open(PAIRS_JSON)) if q.get('tex_pair_eligible')]


def prompt_rewrite(review):
    body = review.get("review_markdown") or json.dumps(review, indent=2)
    return ("A reviewer left the following review of this paper. Revise the paper to "
            "address the review.\n\n" + body +
            "\n\nDo not fabricate experiments, numbers, or citations; if evidence is "
            "missing, narrow the claim instead.")


def prog_path(shard=None):
    return HERE / (f'progress_{TAG}.shard{shard}.csv' if shard is not None
                   else f'progress_{TAG}.csv')


def fields(N):
    f = ['code']
    for n in range(1, N + 1):
        f += [f'R{n}_rating', f'R{n}_decision', f'R{n}_cells', f'R{n}_reward']
    return f + ['status']


def done_codes():
    seen = set()
    for p in [prog_path()] + sorted(HERE.glob(f'progress_{TAG}.shard*.csv')):
        if p.exists():
            seen |= set(r['code'] for r in csv.DictReader(open(p)))
    return seen


def append_prog(row, N, shard=None):
    p = prog_path(shard)
    new = not p.exists()
    with open(p, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields(N), extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)


def run_one(code, N):
    r0 = ir.paper_dir(code)
    base = RUNS / code
    base.mkdir(parents=True, exist_ok=True)
    logdir = base / 'logs'
    logdir.mkdir(exist_ok=True)
    out = {'code': code}
    for n in range(1, N + 1):
        tag = f'{code}_{TAG}_R{n}'
        src = r0 if n == 1 else base / f'R{n-1}'
        dst = ir.stage(src, base / f'R{n}')
        review, info = srev.perform_review(eslib.paper_tex(dst),
                                           log=logdir / f'{tag}_review.json')
        if review is None:
            out['status'] = f'ERR:review_R{n}'
            return out
        ok, dt = ir.run_claude(dst, prompt_rewrite(review), tag, logdir)
        try:
            m = eslib.measure(dst, tag)
        except Exception:
            m = {}
        js = ir.judge(dst, tag, logdir)
        out[f'R{n}_rating'] = None                 # CMU review has no numeric score
        out[f'R{n}_decision'] = review.get('n_items')
        out[f'R{n}_cells'] = m.get('evaluation_surface_cells')
        out[f'R{n}_reward'] = js.get('overall')
        print(f"  [{code} R{n}] ok={ok} {dt:.0f}s | n_items={review.get('n_items')} "
              f"lvl={info.get('level')} cells={m.get('evaluation_surface_cells')} "
              f"reward={js.get('overall')}", flush=True)
    out['status'] = 'ok'
    return out


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    shard = int(sys.argv[3]) if len(sys.argv) > 3 else None
    nshards = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    RUNS.mkdir(parents=True, exist_ok=True)
    seen = done_codes()
    todo = [c for c in CORPUS if c not in seen and ir.paper_dir(c)]
    if limit:
        todo = todo[:limit]
    if shard is not None and nshards > 1:
        todo = todo[shard::nshards]
    print(f'B3i CMU reviewer-loop N={N} max_items={srev.MAX_ITEMS} preset={srev.PRESET} '
          f'shard={shard}/{nshards} papers to run: {len(todo)} '
          f'(already done: {len(seen)})', flush=True)
    for i, code in enumerate(todo):
        t0 = time.time()
        try:
            row = run_one(code, N)
        except Exception as e:
            row = {'code': code, 'status': f'ERR:{str(e)[:60]}'}
        append_prog(row, N, shard)
        print(f"[{i+1}/{len(todo)}] {code} {row.get('status')} "
              f"R{N}_reward={row.get(f'R{N}_reward')} ({time.time()-t0:.0f}s)", flush=True)
    print('B3i BATCH DONE')


if __name__ == '__main__':
    main()
