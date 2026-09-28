"""Runs one paper for 3 rounds and saves the method diagram after every round. Runs three times, switching the editor model.

The goal is to finish quickly per configuration. Everything heavy is already cached.
  argument_graph  introduction hash cache, labelled once, local. About 1 s even when re-measured
  fig_exposition  transcription and coordinates cached by figure hash. Editing takes 30 s on CPU with LaMa
  remaining 4 items  4 s as before

What remains is the editor call and the gate review, both model calls that cannot be cached. Changing the gate model
changes the time, so it is recorded per configuration.

  python3 run_final1.py FA0002 --config haiku_sonnetgate --config haiku_haikugate --config sonnet_sonnetgate
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent
ROOT = API.parent
sys.path.insert(0, str(ROOT / 'temp' / 'code'))
sys.path.insert(0, str(HERE))
OUT = API / 'final_1'

CONFIGS = {
    'haiku_sonnetgate': ('claude-haiku-4-5-20251001', 'claude-sonnet-5'),
    'haiku_haikugate': ('claude-haiku-4-5-20251001', 'claude-haiku-4-5-20251001'),
}


def setup_env(editor: str, gate: str, runs_dir: str):
    os.environ.update({
        'SH_GATE': '1', 'SH_EXTRA_ITEMS': '1', 'SH_ROUNDS': '3',
        'SH_MODEL': editor, 'SH_GATE_MODEL': gate,
        'SH_AG_SUBMIT': '0', 'SH_AG_RUNS': '1', 'SH_AG_LIGHT': '1',
        'SH_GPU_SUBMIT': '0', 'SH_FIG_METHOD': 'lama',
        'SH_RUNS_DIR': runs_dir,
        'AG_PMI_DAEMON': '1', 'AG_LABEL_WORKERS': '16',
        # The wall clock of one round is one editor call and one gate call. Both run serially inside, so they are split up.
        # The editor gets one copy per manuscript file, the gate one chunk per four changes. Verdict criteria and instructions are unchanged.
        'SH_EDITOR_PARALLEL': os.environ.get('SH_EDITOR_PARALLEL', '1'), 'SH_GATE_CHUNK': '3', 'SH_GATE_INLINE': '1', 'SH_GATE_RETIRE_SPLIT': '1',
    })
    ep = Path(os.environ.get("SCISLOP_ROOT", ".") + "/artifact-ai2science/_llm/llm_endpoint_qwen_pc.txt")
    if ep.exists() and ep.read_text().strip():          # the label server held open for this experiment
        os.environ['LLM_ENDPOINT'] = ep.read_text().strip()


def save_round_figures(code: str, runs: Path, dest: Path):
    """Saves the method diagram of every round as is. The edit request and edit record are kept alongside."""
    import harness as H
    import extra_items as X
    fig = X.method_figure(code)
    if not fig:
        return []
    name = Path(fig).name
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    src0 = H.fars_paper_dir(code) / 'figures' / name
    if src0.exists():
        shutil.copy2(src0, dest / f'R0{Path(name).suffix}')
        saved.append('R0')
    for r in (1, 2, 3):
        d = runs / 'A-loc' / code / f'R{r}' / 'figures'
        if (d / name).exists():
            shutil.copy2(d / name, dest / f'R{r}{Path(name).suffix}')
            saved.append(f'R{r}')
        for extra, tag in (('ERASE.txt', 'erase_request'), ('EDIT_REPORT.json', 'edit_report')):
            if (d / extra).exists():
                shutil.copy2(d / extra, dest / f'R{r}_{tag}{Path(extra).suffix}')
    return saved


def run_one(code: str, cname: str, log, tag: str = ''):
    editor, gate = CONFIGS[cname]
    runs_dir = f'runs_{cname}{tag}'
    setup_env(editor, gate, runs_dir)
    for m in list(sys.modules):
        if m in ('harness', 'quality_gate', 'extra_items', 'figure_edit', 'figure_bands', 'figure_inpaint'):
            del sys.modules[m]
    import harness as H
    import extra_items as X
    skill = ROOT / 'temp' / 'skill' / os.environ.get('SH_SKILL', 'SciSlop_v0.4.md')
    if not skill.exists():
        skill = ROOT / 'temp' / 'skill' / 'SciSlop_v0.4.md'
    runs = API / runs_dir
    dest = OUT / cname
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    traj_p = runs / 'A-loc' / code / 'trajectory.json'

    def round_log(s):
        log(s)
        if ' ok wall=' in s or ' stopped' in s:
            try:
                save_round_figures(code, runs, dest / 'figures')
            except Exception as e:
                log(f'[{cname}] figure save skipped {e!r}')

    if traj_p.exists():
        t = json.load(open(traj_p))
    else:
        r0m, m0 = H.r0(code, runs)
        log(f'[{cname}] R0 ' + ', '.join(f'{k} {r0m["scores"].get(k)}' for k in H.ITEMS + X.EXTRA))
        t = H.run_paper('A-loc', code, skill, runs, r0m, m0, model=editor, round_limit=3, log=round_log)
    wall = round(time.time() - t0, 1)
    figs = save_round_figures(code, runs, dest / 'figures')
    rec = {'config': cname, 'editor': editor, 'gate': gate, 'code': code, 'skill': skill.name,
           'stop': t['stop'], 'rounds': len(t['rounds']), 'wall_s': wall,
           'editor_usd': round(t['cost_total_usd'], 3),
           'gate_usd': round(sum((r['gate']['call'] or {}).get('cost_usd') or 0
                                 for r in t['rounds'] if r.get('gate')), 3),
           'per_round': {'R0': t['r0']['scores'], **{f"R{r['round']}": r['scores_after'] for r in t['rounds']}},
           'figure_edits': [(r['round'], (r.get('figure_edit') or {}).get('status'),
                             (r.get('figure_edit') or {}).get('file_changed')) for r in t['rounds']],
           'gate_counts': [(r['round'], (r.get('gate') or {}).get('kept'), (r.get('gate') or {}).get('reverted'))
                           for r in t['rounds']],
           'guards': [(r['round'], r['guards']['violations']) for r in t['rounds']],
           'figures_saved': figs,
           'round_seconds': [(r['round'], r['call']['wall_s'],
                              ((r.get('gate') or {}).get('call') or {}).get('wall_s'), r.get('measure_s'))
                             for r in t['rounds']]}
    (dest / 'result.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    shutil.copytree(runs / 'A-loc' / code, dest / 'run_tree', dirs_exist_ok=True)
    log(f'[{cname}] done {wall:.0f}s editing ${rec["editor_usd"]} gate ${rec["gate_usd"]} '
        f'diagram {rec["figure_edits"]} saved {figs}')
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('code')
    ap.add_argument('--config', action='append', default=[])
    ap.add_argument('--tag', default='', help='run tag. Appended to the run directory and the output folder')
    a = ap.parse_args()
    global OUT
    if a.tag:
        OUT = OUT / f'_{a.tag.lstrip("_")}'
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / (('run_%s.log' % a.config[0]) if len(a.config) == 1 else 'run.log'), 'a')

    def log(s):
        line = time.strftime('%H:%M:%S ') + s
        print(line, flush=True); logf.write(line + '\n'); logf.flush()

    names = a.config or list(CONFIGS)
    recs = []
    for c in names:
        try:
            recs.append(run_one(a.code, c, log, a.tag))
        except Exception as e:
            import traceback
            log(f'[{c}] ERROR {e!r}')
            (OUT / f'error_{c}.txt').write_text(traceback.format_exc())
    (OUT / 'all_results.json').write_text(json.dumps(recs, indent=1, ensure_ascii=False))
    log(f'All done. {len(recs)} configurations')


if __name__ == '__main__':
    main()
