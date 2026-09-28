#!/usr/bin/env python3
"""Runs ten papers through round 3 and leaves the per-round outputs of each paper, whole, in its own folder.

There is one configuration. Editor Haiku called once, gate Sonnet in parallel over chunks of three changes each, all six items.
Splitting the editor per file makes rounds faster, but xsec_ref does not come down. Cross-section references are the unit of that item,
and an editor confined to one file cannot touch the other sections, so the editor is called only once.

What is left per paper
    final_10/<CODE>/result.json      per-round scores for the six items, time, cost, gate tallies, guards
    final_10/<CODE>/figures/         R0~R3 method diagram, erase request, edit record
    final_10/<CODE>/run_tree/        R1~R3 manuscript full text, gate/ (CHANGES, VERDICTS, RETIRE, per-chunk records),
                                     measure/ (per-item measurer output), logs/, trajectory.json
    final_10/results.jsonl           one line appended each time a paper finishes. If interrupted midway, the run resumes

  python3 run_final10.py [--codes FA0002,...] [--workers 3] [--config haiku_sonnetgate]
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent
ROOT = API.parent
sys.path.insert(0, str(ROOT / 'temp' / 'code'))
sys.path.insert(0, str(HERE))
OUT = API / 'final_10'
CONFIGS = {'haiku_sonnetgate': ('claude-haiku-4-5-20251001', 'claude-sonnet-5'),
           'haiku_haikugate': ('claude-haiku-4-5-20251001', 'claude-haiku-4-5-20251001')}
LOCK = threading.Lock()


def base_env(editor: str, gate: str) -> dict:
    ep = Path(os.environ.get("SCISLOP_ROOT", ".") + "/artifact-ai2science/_llm/llm_endpoint_qwen_pc.txt")
    e = {'SH_GATE': '1', 'SH_EXTRA_ITEMS': '1', 'SH_ROUNDS': '3',
         'SH_MODEL': editor, 'SH_GATE_MODEL': gate,
         'SH_AG_SUBMIT': '0', 'SH_AG_RUNS': '1', 'SH_AG_LIGHT': '1',
         'SH_GPU_SUBMIT': '0', 'SH_FIG_METHOD': 'lama',
         'AG_PMI_DAEMON': '1', 'AG_LABEL_WORKERS': '16',
         'SH_EDITOR_PARALLEL': '0',                 # editor runs once. Splitting per file kills xsec_ref
         'SH_GATE_CHUNK': '3', 'SH_GATE_INLINE': '1', 'SH_GATE_RETIRE_SPLIT': '1',
         'SH_LIMIT_RETRIES': '2', 'SH_LIMIT_SLEEP': '180'}
    if ep.exists() and ep.read_text().strip():
        e['LLM_ENDPOINT'] = ep.read_text().strip()
    return e


def one_paper(code: str, cname: str, log) -> dict:
    """One paper. Runs in a separate process. The harness uses environment variables and module globals, so papers must not be mixed as threads."""
    dest = OUT / code
    dest.mkdir(parents=True, exist_ok=True)
    editor, gate = CONFIGS[cname]
    env = {**os.environ, **base_env(editor, gate), 'SH_RUNS_DIR': f'runs10_{code}'}
    t0 = time.time()
    import subprocess
    r = subprocess.run([sys.executable, str(HERE / 'run_one_paper.py'), code, cname, str(dest)],
                       env=env, capture_output=True, text=True, timeout=14400)
    (dest / 'stdout.txt').write_text((r.stdout or '') + '\n=== STDERR ===\n' + (r.stderr or ''))
    wall = round(time.time() - t0, 1)
    rf = dest / 'result.json'
    if rf.exists():
        rec = json.loads(rf.read_text()); rec['wall_s'] = wall
        rf.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    else:
        rec = {'code': code, 'config': cname, 'wall_s': wall, 'error': (r.stderr or '')[-1500:], 'rc': r.returncode}
        rf.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    # If the editor cost is 0 or fewer than three rounds completed, the run was blocked by the usage limit. That is not a result, so it is
    # not recorded; the folder is deleted and the paper is queued again. Otherwise an empty result would remain as finished and resume would skip it.
    if (rec.get('editor_usd') or 0) == 0 or (rec.get('rounds') or 0) < 3:
        log(f'[{code}] not finished because of the limit (rounds {rec.get("rounds")}, editing ${rec.get("editor_usd")}). Reverting and queuing again')
        shutil.rmtree(dest, ignore_errors=True)
        return {'code': code, 'requeue': True}
    with LOCK:
        with open(OUT / 'results.jsonl', 'a') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    sc = rec.get('per_round', {}).get('R3') or {}
    log(f'[{code}] {wall:.0f}s ${(rec.get("editor_usd") or 0) + (rec.get("gate_usd") or 0):.2f} '
        f'stop={rec.get("stop")} R3=' + ', '.join(f'{k} {v}' for k, v in sc.items()))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--codes', default='')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--config', default='haiku_sonnetgate')
    ap.add_argument('--list', default='papers10.json', help='paper list file')
    ap.add_argument('--out', default='final_10', help='output folder')
    a = ap.parse_args()
    global OUT
    OUT = API / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    codes = a.codes.split(',') if a.codes else json.load(open(API / a.list))['codes']
    logf = open(OUT / 'run.log', 'a')

    def log(s):
        line = time.strftime('%H:%M:%S ') + s
        with LOCK:
            print(line, flush=True); logf.write(line + '\n'); logf.flush()

    todo = [c for c in codes if not (OUT / c / 'result.json').exists()]
    log(f'=== Start. Total {len(codes)} papers, remaining {len(todo)}, concurrency {a.workers}, config {a.config}')
    t0 = time.time()
    q = list(todo)
    qlock = threading.Lock()

    # The limit applies to the whole account, so when one paper is blocked the rest are too. Workers back off together instead of waiting separately.
    pause_until = [0.0]
    fails = [0]

    def worker():
        while True:
            w = pause_until[0] - time.time()
            if w > 0:
                time.sleep(min(w, 60)); continue
            with qlock:
                if not q:
                    return
                c = q.pop(0)
            try:
                rec = one_paper(c, a.config, log)
            except Exception as e:
                import traceback
                log(f'[{c}] ERROR {e!r}')
                (OUT / f'error_{c}.txt').write_text(traceback.format_exc())
                continue
            if rec.get('requeue'):
                with qlock:
                    q.append(c)
                    fails[0] += 1
                    nap = min(300 * fails[0], 1800)
                    pause_until[0] = time.time() + nap
                log(f'Backing off {nap / 60:.0f} min for everyone because of the limit (cumulative {fails[0]} times)')
            else:
                fails[0] = max(0, fails[0] - 1)

    ts = [threading.Thread(target=worker) for _ in range(min(a.workers, len(q) or 1))]
    for t in ts:
        t.start(); time.sleep(3)
    for t in ts:
        t.join()
    wall = time.time() - t0
    done = [json.loads(l) for l in open(OUT / 'results.jsonl')] if (OUT / 'results.jsonl').exists() else []
    log(f'=== End. {len(todo)} papers in {wall / 60:.1f} min, average {wall / max(len(todo), 1) / 60:.1f} min per paper, '
        f'cumulative records {len(done)} lines')


if __name__ == '__main__':
    main()
