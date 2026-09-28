"""Component ablation of the information given to the editor (§5, 0922).

Same harness, same editor (Haiku 4.5, per-file edit blocks, thinking 1024), same reviewer (Sonnet 5 text gate with the
FULL SciSlop_v0.5.md and the FULL located-instance list), same three rounds and the same 60 papers as the shipped method
(final_v14). Only the skill file the editor reads changes, and whether SLOP_FINDINGS.md is attached to the editor:

  arm      editor skill                          locations to editor   harness arm
  name     SciSlop_v0.5_name.md   (names)        no                    A-def
  def      SciSlop_v0.5_def.md    (+definition)  no                    A-def
  defloc   SciSlop_v0.5_defloc.md (+units fmt)   yes                   A-loc
  full     SciSlop_v0.5.md        (+fix)         yes                   A-loc   = final_v14, not re-run

Differences from run_batch_v14.py, all deliberate: SH_TARGETS=0 (v14 predates the per-file targets of v15; its
trajectories carry targets=None), SH_GATE_SKILL / SH_GATE_FULL_FINDINGS / SH_EDITS_PARALLEL_DEF (the 0922 env-gated
additions in temp/code/harness.py), runs and results under ablation_components/, R0 measures copied from v14's _R0.

  python3 run_ablation.py --arm name [--papers FA0002 | --papers-file ...] [--workers 6] [--rounds 3]
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABL = HERE.parent
VER1 = ABL.parent / 'ver1'
API = VER1 / 'api'
sys.path.insert(0, str(VER1 / 'temp' / 'code')); sys.path.insert(0, str(API / 'code'))
import run_final1 as RF   # noqa: E402
import run_v2 as V2       # noqa: E402  (FAST defaults of v14)

FULL_SKILL = VER1 / 'temp' / 'skill' / 'SciSlop_v0.5.md'
# arm -> (harness arm, editor skill path, extra env). Component arms of EXPERIMENTS_0919.md §5.3 paragraph 3 (0922 evening):
#   nogate   E6  definitions + locations, no gate          (ours minus the gate)
#   gateonly E7  generic instruction only + gate           (ours minus definitions and locations)
#   noloc    E8  full skill (definitions + fix) + gate, no locations (ours minus the located instances)
#   full     = final_v14, not re-run
# The earlier skill-wording arms (name / def / defloc) are kept for the appendix and archived under _abandoned_skillwording/.
ARMS = {
    'nogate': ('A-loc', FULL_SKILL, {'SH_GATE': '0'}),
    'gateonly': ('A-def', FULL_SKILL, {'SH_EDITOR_GENERIC': '1'}),
    'noloc': ('A-def', FULL_SKILL, {}),
    'name': ('A-def', ABL / 'skills' / 'SciSlop_v0.5_name.md', {}),
    'def': ('A-def', ABL / 'skills' / 'SciSlop_v0.5_def.md', {}),
    'defloc': ('A-loc', ABL / 'skills' / 'SciSlop_v0.5_defloc.md', {}),
}
V14_R0 = API / 'runs_haiku_sonnetgate_v14' / '_R0'
V14_RESULTS = API / 'final_v14' / 'results.jsonl'
lock = threading.Lock()


def v14_codes() -> list[str]:
    last = {}
    for l in open(V14_RESULTS):
        if l.strip():
            r = json.loads(l); last[r['code']] = r
    return sorted(last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=sorted(ARMS))
    ap.add_argument('--papers', default='', help='comma list; default = the 60 papers of final_v14')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--stagger', type=float, default=10.0)
    ap.add_argument('--tag', default='', help='suffix for smoke runs, e.g. _smoke')
    a = ap.parse_args()
    harness_arm, skill, extra_env = ARMS[a.arm]
    cname = f'abl_{a.arm}{a.tag}'
    editor, gate = RF.CONFIGS['haiku_sonnetgate']
    RF.setup_env(editor, gate, f'runs_{cname}')
    for k, v in V2.FAST.items():
        os.environ.setdefault(k, v)
    os.environ['SH_TARGETS'] = '0'                       # v14 had no per-file targets (trajectory: targets None)
    os.environ['SH_GATE_SKILL'] = str(FULL_SKILL)        # reviewer always reads the full skill
    os.environ['SH_GATE_FULL_FINDINGS'] = '1'            # reviewer always gets the full located-instance list
    os.environ['SH_EDITS_PARALLEL_DEF'] = '1'            # A-def arms use the same per-file edit-block editor
    os.environ.update(extra_env)
    import harness as H
    import extra_items as X
    if a.arm in ('name', 'def', 'defloc'):
        H.ITEM_HEADING['citation'] = 'Citation isolation'    # the skill-wording variants use the paper's item name as heading
    skill = Path(skill)
    runs = ABL / f'runs_{cname}'
    runs.mkdir(parents=True, exist_ok=True)
    if not (runs / '_R0').exists():
        shutil.copytree(V14_R0, runs / '_R0')            # original-manuscript measures, identical to the shipped run
    out = ABL / f'final_{cname}'
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'batch.log', 'a')

    def log(s):
        line = time.strftime('%H:%M:%S ') + s
        with lock:
            print(line, flush=True); logf.write(line + '\n'); logf.flush()

    codes = [c for c in a.papers.split(',') if c] or v14_codes()
    env_snapshot = {k: v for k, v in os.environ.items() if k.startswith(('SH_', 'AG_', 'LLM_'))}
    (out / 'config.json').write_text(json.dumps({
        'arm': a.arm, 'harness_arm': harness_arm, 'editor': editor, 'gate': gate, 'rounds': a.rounds, 'workers': a.workers,
        'editor_skill': str(skill), 'editor_skill_md5': hashlib.md5(skill.read_bytes()).hexdigest(),
        'gate_skill': str(FULL_SKILL), 'gate_skill_md5': hashlib.md5(FULL_SKILL.read_bytes()).hexdigest(),
        'papers': codes, 'env': env_snapshot, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}, indent=1))
    sem = threading.Semaphore(a.workers)
    recs = {}
    t_batch = time.time()

    def one(code):
        with sem:
            t0 = time.time()
            try:
                base = runs / harness_arm / code
                traj_p = base / 'trajectory.json'
                prev = json.load(open(traj_p)) if traj_p.exists() else None
                if prev and prev.get('stop') not in (None, 'in_progress', 'session_limit'):
                    t = prev; log(f'{code} already finished; collecting')
                elif prev and prev.get('stop') in ('in_progress', 'session_limit') and prev.get('rounds') and len(prev['rounds']) < a.rounds:
                    last = prev['rounds'][-1]['round']
                    r0m, m0 = H.r0(code, runs)
                    # a gated_noop round keeps the previous measure and writes no measure/R<n>/, so walk back to the
                    # latest round that was measured (R0 = the copied original measure)
                    prev_measure = r0m
                    for k in range(last, 0, -1):
                        sp = base / 'measure' / f'R{k}' / 'summary.json'
                        if sp.exists():
                            prev_measure = json.load(open(sp)); break
                    log(f'{code} has R{last}; resuming from R{last + 1}')
                    t = H.run_paper(harness_arm, code, skill, runs, r0m, m0, model=editor, round_limit=a.rounds, log=log,
                                    resume={'traj': prev, 'prev_dir': str(base / f'R{last}'), 'prev_measure': prev_measure, 'start_round': last + 1})
                else:
                    r0m, m0 = H.r0(code, runs)
                    log(f'{code} R0 ' + ', '.join(f'{k} {r0m["scores"].get(k)}' for k in H.ITEMS + X.EXTRA))
                    t = H.run_paper(harness_arm, code, skill, runs, r0m, m0, model=editor, round_limit=a.rounds, log=log)
                wall = round(time.time() - t0, 1)
                rec = {'config': cname, 'arm': a.arm, 'editor': editor, 'gate': gate, 'code': code, 'skill': skill.name,
                       'stop': t['stop'], 'rounds': len(t['rounds']), 'wall_s': wall,
                       'editor_usd': round(t['cost_total_usd'], 3),
                       'gate_usd': round(sum(((r.get('gate') or {}).get('call') or {}).get('cost_usd') or 0 for r in t['rounds']), 3),
                       'per_round': {'R0': t['r0']['scores'], **{f"R{r['round']}": r['scores_after'] for r in t['rounds']}},
                       'exec': [(r['round'], r.get('exec')) for r in t['rounds']],
                       'figure_edits': [(r['round'], (r.get('figure_edit') or {}).get('status'), (r.get('figure_edit') or {}).get('file_changed')) for r in t['rounds']],
                       'gate_counts': [(r['round'], (r.get('gate') or {}).get('kept'), (r.get('gate') or {}).get('reverted')) for r in t['rounds']],
                       'guards': [(r['round'], r['guards']['violations']) for r in t['rounds']],
                       'soft_flags': [(r['round'], r['guards']['soft_flags']) for r in t['rounds']],
                       'round_seconds': [(r['round'], r['call']['wall_s'], ((r.get('gate') or {}).get('call') or {}).get('wall_s'), r.get('measure_s')) for r in t['rounds']]}
                dest = out / code
                dest.mkdir(parents=True, exist_ok=True)
                (dest / 'result.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False))
                with lock:
                    recs[code] = rec
                    with open(out / 'results.jsonl', 'a') as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                log(f'{code} done {wall:.0f}s rounds {rec["rounds"]} editor ${rec["editor_usd"]} gate ${rec["gate_usd"]} final {t["final_scores"]}')
            except Exception as e:
                import traceback
                log(f'{code} ERROR {e!r}')
                (out / f'error_{code}.txt').write_text(traceback.format_exc())

    log(f'batch start arm={a.arm} ({harness_arm}) {len(codes)} papers, workers {a.workers}, rounds {a.rounds}, editor skill {skill.name}, '
        f'gate skill {FULL_SKILL.name}, mode {os.environ.get("SH_EDITOR_MODE")}, targets {os.environ.get("SH_TARGETS")}')
    ts = []
    for c in codes:
        th = threading.Thread(target=one, args=(c,)); th.start(); ts.append(th); time.sleep(a.stagger)
    for th in ts:
        th.join()
    total = round(time.time() - t_batch, 1)
    log(f'batch end {len(recs)}/{len(codes)} papers, wall {total/60:.1f} min')
    (out / 'batch_summary.json').write_text(json.dumps({'codes': codes, 'done': sorted(recs), 'wall_s': total,
                                                        'per_paper_wall_s': {c: r['wall_s'] for c, r in recs.items()}}, indent=1))


if __name__ == '__main__':
    main()
