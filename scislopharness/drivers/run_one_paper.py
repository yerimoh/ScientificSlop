#!/usr/bin/env python3
"""Runs one paper through round 3 and leaves the outputs in its own folder. run_final10.py spawns this as a process per paper.

Processes are separated because the harness reads environment variables and module globals (run directory, editor model,
gate model). Running several papers as threads in one process would overwrite each other's settings.

  python3 run_one_paper.py <CODE> <config> <dest>
"""
from __future__ import annotations
import json, os, shutil, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent
ROOT = API.parent
sys.path.insert(0, str(ROOT / 'temp' / 'code'))
sys.path.insert(0, str(HERE))
CONFIGS = {'haiku_sonnetgate': ('claude-haiku-4-5-20251001', 'claude-sonnet-5'),
           'haiku_haikugate': ('claude-haiku-4-5-20251001', 'claude-haiku-4-5-20251001')}


def save_figures(code: str, runs: Path, dest: Path, H, X) -> list:
    fig = X.method_figure(code)
    if not fig:
        return []
    name = Path(fig).name
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    src0 = H.fars_paper_dir(code) / 'figures' / name
    if src0.exists():
        shutil.copy2(src0, dest / f'R0{Path(name).suffix}'); saved.append('R0')
    for r in (1, 2, 3):
        d = runs / 'A-loc' / code / f'R{r}' / 'figures'
        if (d / name).exists():
            shutil.copy2(d / name, dest / f'R{r}{Path(name).suffix}'); saved.append(f'R{r}')
        for extra, tag in (('ERASE.txt', 'erase_request'), ('EDIT_REPORT.json', 'edit_report')):
            if (d / extra).exists():
                shutil.copy2(d / extra, dest / f'R{r}_{tag}{Path(extra).suffix}')
    return saved


def main():
    code, cname, dest = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    editor, gate = CONFIGS[cname]
    import harness as H
    import extra_items as X
    skill = ROOT / 'temp' / 'skill' / 'SciSlop_v0.4.md'
    if not skill.exists():
        skill = ROOT / 'temp' / 'skill' / 'SciSlop_v0.3.md'
    runs = API / os.environ['SH_RUNS_DIR']
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(s):
        line = time.strftime('%H:%M:%S ') + s
        print(line, flush=True)
        with open(dest / 'rounds.log', 'a') as f:
            f.write(line + '\n')
        try:                                    # save the diagram after every round. It survives even if a later round dies
            if ' ok wall=' in s:
                save_figures(code, runs, dest / 'figures', H, X)
        except Exception:
            pass

    r0m, m0 = H.r0(code, runs)
    log(f'[{code}] R0 ' + ', '.join(f'{k} {r0m["scores"].get(k)}' for k in H.ITEMS + X.EXTRA))
    t = H.run_paper('A-loc', code, skill, runs, r0m, m0, model=editor, round_limit=3, log=log)
    wall = round(time.time() - t0, 1)
    figs = save_figures(code, runs, dest / 'figures', H, X)
    rec = {'code': code, 'config': cname, 'editor': editor, 'gate': gate, 'skill': skill.name,
           'stop': t['stop'], 'rounds': len(t['rounds']), 'wall_s': wall,
           'editor_usd': round(t['cost_total_usd'], 3),
           'gate_usd': round(sum((r['gate']['call'] or {}).get('cost_usd') or 0
                                 for r in t['rounds'] if r.get('gate')), 3),
           'per_round': {'R0': t['r0']['scores'], **{f"R{r['round']}": r['scores_after'] for r in t['rounds']}},
           'figure_edits': [(r['round'], (r.get('figure_edit') or {}).get('status'),
                             (r.get('figure_edit') or {}).get('file_changed')) for r in t['rounds']],
           'gate_counts': [(r['round'], (r.get('gate') or {}).get('kept'), (r.get('gate') or {}).get('reverted'),
                            len((r.get('gate') or {}).get('retire') or [])) for r in t['rounds']],
           'guards': [(r['round'], r['guards']['violations'], r['guards'].get('soft_flags')) for r in t['rounds']],
           'figures_saved': figs,
           'round_seconds': [(r['round'], r['call']['wall_s'],
                              ((r.get('gate') or {}).get('call') or {}).get('wall_s'), r.get('measure_s'))
                             for r in t['rounds']]}
    (dest / 'result.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    shutil.copytree(runs / 'A-loc' / code, dest / 'run_tree', dirs_exist_ok=True)
    log(f'[{code}] done {wall:.0f}s editing ${rec["editor_usd"]} gate ${rec["gate_usd"]} diagram {figs}')


if __name__ == '__main__':
    main()
