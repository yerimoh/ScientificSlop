"""Runs several papers in parallel with the v14 method (per-file edit-block editor + text gate + skill v0.5).

run_final1.run_one rebuilds the environment and reimports modules on every call, so it is not thread-safe. Here the environment is set
once, the harness is imported once, and H.run_paper runs in one thread per paper. The output format is the same as run_final1.

  python3 run_batch_v14.py --papers FA0005,FA0198,... [--workers 10] [--suffix _v14]
"""
from __future__ import annotations
import argparse, json, os, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent
ROOT = API.parent
sys.path.insert(0, str(ROOT / 'temp' / 'code')); sys.path.insert(0, str(HERE))
import run_final1 as RF   # noqa: E402
import run_v2 as V2       # noqa: E402  (FAST defaults)

lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--papers', required=True)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--suffix', default='_v14')
    ap.add_argument('--stagger', type=float, default=15.0)
    ap.add_argument('--final-dir', default='', help='final directory holding the summary and per-paper folders (default final<suffix>)')
    ap.add_argument('--dest-config-dir', default='', help='parent name of the per-paper folders (default haiku_sonnetgate<suffix>)')
    ap.add_argument('--results-file', default='results.jsonl')
    ap.add_argument('--log-file', default='batch.log')
    a = ap.parse_args()
    cname = 'haiku_sonnetgate' + a.suffix
    editor, gate = RF.CONFIGS['haiku_sonnetgate']
    RF.setup_env(editor, gate, f'runs_{cname}')
    for k, v in V2.FAST.items():
        os.environ.setdefault(k, v)
    import harness as H
    import extra_items as X
    out = API / (a.final_dir or f'final{a.suffix}')
    out.mkdir(parents=True, exist_ok=True)
    runs = API / f'runs_{cname}'
    dest_cfg = a.dest_config_dir or cname
    logf = open(out / a.log_file, 'a')

    def log(s):
        line = time.strftime('%H:%M:%S ') + s
        with lock:
            print(line, flush=True); logf.write(line + '\n'); logf.flush()

    skill = ROOT / 'temp' / 'skill' / os.environ.get('SH_SKILL', 'SciSlop_v0.5.md')
    codes = [c for c in a.papers.split(',') if c]
    sem = threading.Semaphore(a.workers)
    recs = {}
    t_batch = time.time()

    def one(code):
        with sem:
            t0 = time.time()
            try:
                dest = out / dest_cfg / code
                dest.mkdir(parents=True, exist_ok=True)
                traj_p = runs / 'A-loc' / code / 'trajectory.json'
                prev = json.load(open(traj_p)) if traj_p.exists() else None
                if prev and prev.get('stop') not in (None, 'in_progress', 'session_limit'):
                    t = prev; log(f'{code} already finished, collecting results only')
                elif prev and prev.get('stop') in ('in_progress', 'session_limit') and prev.get('rounds') and len(prev['rounds']) < 3:
                    # A paper stopped by the usage limit resumes after its last round. Records, trees and measurements stay as they are.
                    last = prev['rounds'][-1]['round']
                    r0m, m0 = H.r0(code, runs)
                    prev_measure = json.load(open(runs / 'A-loc' / code / 'measure' / f'R{last}' / 'summary.json'))
                    log(f'{code} has up to R{last}, resuming from R{last + 1}')
                    t = H.run_paper('A-loc', code, skill, runs, r0m, m0, model=editor, round_limit=3, log=log,
                                    resume={'traj': prev, 'prev_dir': str(runs / 'A-loc' / code / f'R{last}'),
                                            'prev_measure': prev_measure, 'start_round': last + 1})
                else:
                    r0m, m0 = H.r0(code, runs)
                    log(f'{code} R0 ' + ', '.join(f'{k} {r0m["scores"].get(k)}' for k in H.ITEMS + X.EXTRA))
                    t = H.run_paper('A-loc', code, skill, runs, r0m, m0, model=editor, round_limit=3, log=log)
                wall = round(time.time() - t0, 1)
                figs = RF.save_round_figures(code, runs, dest / 'figures')
                rec = {'config': cname, 'editor': editor, 'gate': gate, 'code': code, 'skill': skill.name,
                       'stop': t['stop'], 'rounds': len(t['rounds']), 'wall_s': wall,
                       'editor_usd': round(t['cost_total_usd'], 3),
                       'gate_usd': round(sum((r['gate']['call'] or {}).get('cost_usd') or 0 for r in t['rounds'] if r.get('gate')), 3),
                       'per_round': {'R0': t['r0']['scores'], **{f"R{r['round']}": r['scores_after'] for r in t['rounds']}},
                       'figure_edits': [(r['round'], (r.get('figure_edit') or {}).get('status'), (r.get('figure_edit') or {}).get('file_changed')) for r in t['rounds']],
                       'gate_counts': [(r['round'], (r.get('gate') or {}).get('kept'), (r.get('gate') or {}).get('reverted')) for r in t['rounds']],
                       'guards': [(r['round'], r['guards']['violations']) for r in t['rounds']],
                       'round_seconds': [(r['round'], r['call']['wall_s'], ((r.get('gate') or {}).get('call') or {}).get('wall_s'), r.get('measure_s')) for r in t['rounds']],
                       'figures_saved': figs}
                (dest / 'result.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False))
                with lock:
                    recs[code] = rec
                    with open(out / a.results_file, 'a') as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                log(f'{code} done {wall:.0f}s rounds {rec["rounds"]} editing ${rec["editor_usd"]} gate ${rec["gate_usd"]} '
                    f'R3 {t["final_scores"]}')
            except Exception as e:
                import traceback
                log(f'{code} ERROR {e!r}')
                (out / f'error_{code}.txt').write_text(traceback.format_exc())

    log(f'Batch start {len(codes)} papers, concurrency {a.workers}, skill {skill.name}, mode {os.environ.get("SH_EDITOR_MODE")}')
    ts = []
    for c in codes:
        th = threading.Thread(target=one, args=(c,)); th.start(); ts.append(th); time.sleep(a.stagger)
    for th in ts:
        th.join()
    total = round(time.time() - t_batch, 1)
    log(f'Batch end {len(recs)}/{len(codes)} papers, wall clock {total/60:.1f} min')
    (out / ('batch_summary.json' if a.results_file == 'results.jsonl' else a.results_file.replace('.jsonl', '_summary.json'))).write_text(json.dumps({'codes': codes, 'done': sorted(recs), 'wall_s': total,
                                                        'per_paper_wall_s': {c: r['wall_s'] for c, r in recs.items()}}, indent=1))


if __name__ == '__main__':
    main()
