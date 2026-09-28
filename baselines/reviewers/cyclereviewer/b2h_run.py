"""B2h baseline — Reviewer-loop, backbone = CycleReviewer (WestlakeNLP, actual
fine-tuned checkpoint).

0720 meeting D1: B2/B3 reviewer-loop must use REAL reviewer systems — AI
Scientist (Sakana, done as B3a) + CycleReviewer. This is the CycleReviewer
variant (labelled B2h in the master baseline table). Everything except review
generation is byte-identical to B3/B3a: stage/run_claude/judge/paper_dir reused
by import from the ES survival code, rewrite prompt shared, corpus = same 165
tex-eligible pairs, R5.

Per round n (per paper):
  1   stage       ir.stage(): copy R(n-1) tex tree -> runs/<code>/R<n>
  3a  review      cyclerev_reviewer.perform_review(): ONE generation of the
                  served CycleReviewer-ML-Llama-3.1-8B -> 4 simulated ICLR
                  reviewers + meta review + Accept/Reject (temp .4, canonical).
                  HARD RULE: no mold concept anywhere in the pipeline (the
                  CycleReviewer prompt is generic peer review — satisfied by
                  construction; the checkpoint is frozen upstream).
  3b  rewrite     ir.run_claude(): Haiku agent revises the paper to address the
                  review (same prompt wrapper as B3/B3a, no-fabrication guard).
  2/4 measure     eslib.measure(): LOGGED ONLY — never injected into the loop.
  5   judge       ir.judge(): Qwen median-of-3 {soundness, presentation,
                  contribution, overall}

Usage:
  python3 b2h_run.py [N_rounds=5] [limit=0(all)] [shard_i] [K]
Requires BOTH servers up: Qwen (_llm/llm_endpoint.txt, judge) and CycleReviewer
(cyclerev_endpoint.txt, serve_cyclerev.sh).
"""
from __future__ import annotations
import os
import json, csv, sys, time
from pathlib import Path

ES_CODE = (os.environ.get("SCISLOP_ROOT", ".") + "/artifact-ai2science/"
           'Content_Mold/Evaluation_Surface_Area/results/code')
sys.path.insert(0, ES_CODE)
import eslib                      # noqa: E402  paper_tex + measure (log-only here)
import iter_run as ir             # noqa: E402  stage/run_claude/judge/paper_dir — unchanged

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cyclerev_reviewer as crev  # noqa: E402  the swapped-in review backbone

RUNS = HERE / 'runs'
PAIRS_JSON = (os.environ.get("SCISLOP_ROOT", ".") + "/artifact-ai2science/"
              'Evaluation/00_DATA_AUDIT/pairs_final.json')
CORPUS = [q['code'] for q in json.load(open(PAIRS_JSON)) if q.get('tex_pair_eligible')]


# ---- (3b) rewrite prompt — identical wrapper to b3_run.py / b3a_run.py --------
def prompt_rewrite(review):
    return ("A reviewer left the following review of this paper. Revise the paper to "
            "address the review.\n\n" + json.dumps(review, indent=2) +
            "\n\nDo not fabricate experiments, numbers, or citations; if evidence is "
            "missing, narrow the claim instead.")


# ---- progress CSV (resumable, sharded — mirrors b3a_run.py) --------------------
def prog_path(shard=None):
    if shard is None:
        return HERE / 'progress_b2h.csv'
    return HERE / f'progress_b2h.shard{shard}.csv'


def fields(N):
    f = ['code']
    for n in range(1, N + 1):
        f += [f'R{n}_rating', f'R{n}_decision', f'R{n}_cells', f'R{n}_reward']
    return f + ['status']


def done_codes():
    seen = set()
    for p in [prog_path()] + sorted(HERE.glob('progress_b2h.shard*.csv')):
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


# ---- per-paper loop ------------------------------------------------------------
def run_one(code, N):
    r0 = ir.paper_dir(code)
    base = RUNS / code
    base.mkdir(parents=True, exist_ok=True)
    logdir = base / 'logs'
    logdir.mkdir(exist_ok=True)
    out = {'code': code}
    for n in range(1, N + 1):
        tag = f'{code}_b2h_R{n}'
        src = r0 if n == 1 else base / f'R{n-1}'
        dst = ir.stage(src, base / f'R{n}')
        # (3a) CycleReviewer review of the CURRENT draft (served fine-tune)
        review, info = crev.perform_review(eslib.paper_tex(dst),
                                           log=logdir / f'{tag}_review.json')
        if review is None:
            out['status'] = f'ERR:review_R{n}'
            return out
        # (3b) Haiku rewrite addressing the review (edits tex in place under dst)
        ok, dt = ir.run_claude(dst, prompt_rewrite(review), tag, logdir)
        # (2/4) measurement — LOGGED ONLY, never fed back into the loop.
        try:
            m = eslib.measure(dst, tag)
        except Exception:
            m = {}
        # (5) judge — Qwen median-of-3
        js = ir.judge(dst, tag, logdir)
        out[f'R{n}_rating'] = review.get('Overall')
        out[f'R{n}_decision'] = review.get('Decision')
        out[f'R{n}_cells'] = m.get('evaluation_surface_cells')
        out[f'R{n}_reward'] = js.get('overall')
        print(f"  [{code} R{n}] ok={ok} {dt:.0f}s | overall={review.get('Overall')} "
              f"dec={review.get('Decision')} lvl={info['level']} n_valid={info['n_valid']} "
              f"cells={m.get('evaluation_surface_cells')} reward={js.get('overall')}", flush=True)
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
    print(f'B2h CycleReviewer reviewer-loop N={N} shard={shard}/{nshards} '
          f'papers to run: {len(todo)} (already done: {len(seen)})', flush=True)
    for i, code in enumerate(todo):
        t0 = time.time()
        try:
            row = run_one(code, N)
        except Exception as e:
            row = {'code': code, 'status': f'ERR:{str(e)[:60]}'}
        append_prog(row, N, shard)
        print(f"[{i+1}/{len(todo)}] {code} {row.get('status')} "
              f"R{N}_rating={row.get(f'R{N}_rating')} R{N}_reward={row.get(f'R{N}_reward')} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print('B2H BATCH DONE')


if __name__ == '__main__':
    main()
