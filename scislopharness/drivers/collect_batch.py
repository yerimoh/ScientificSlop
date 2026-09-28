"""After a batch finishes, gather each paper's per-round outputs under final_<suffix>/<config>/<code>/ (one place for analysis).

What is copied: rounds/R0..R3 (diagram, ERASE, EDIT_REPORT, scores, SLOP_FINDINGS, REVIEW_NOTES, VERDICTS, RETIRE, CHANGES,
measure summary, patches), ROUNDS.md, trajectory.json, logs/, gate/R<n>/{CHANGES.md,VERDICTS.json,RETIRE.json,chunk_*/,retire_*},
measure/, and the per-round manuscript trees R1..R3 (tex, figures, materials). The originals stay in runs_<config>/.

  python3 collect_batch.py --suffix _v14
"""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
API = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='_v14')
    ap.add_argument('--final-dir', default='')
    ap.add_argument('--dest-config-dir', default='')
    a = ap.parse_args()
    cname = 'haiku_sonnetgate' + a.suffix
    runs = API / f'runs_{cname}' / 'A-loc'
    out = API / (a.final_dir or f'final{a.suffix}') / (a.dest_config_dir or cname)
    n = 0
    for src in sorted(runs.iterdir()):
        if not src.is_dir() or not (src / 'trajectory.json').exists():
            continue
        dst = out / src.name
        dst.mkdir(parents=True, exist_ok=True)
        for name in ('rounds', 'logs', 'measure', 'gate'):
            if (src / name).exists():
                shutil.copytree(src / name, dst / name, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('_bands', 'ORIGINAL', 'PREVIOUS', 'materials'))
        for name in ('ROUNDS.md', 'trajectory.json'):
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        for r in ('R1', 'R2', 'R3'):
            if (src / r).exists():
                shutil.copytree(src / r, dst / 'manuscript' / r, dirs_exist_ok=True, ignore=shutil.ignore_patterns('_bands'))
        (dst / 'README.md').write_text(
            f'# {src.name} per-round outputs\n\n'
            f'- `rounds/R<n>/` round summary folder (diagram image, ERASE, EDIT_REPORT, scores.json, location list, review notes, verdicts, retire, changes, measure summary, patches) + a README in each folder\n'
            f'- `manuscript/R<n>/` final manuscript of that round (after the gate). tex, figures/, materials/\n'
            f'- `gate/R<n>/` reviewer directory (REVISED = manuscript before the gate, CHANGES, VERDICTS, RETIRE, chunk_k inputs/outputs)\n'
            f'- `logs/` full text of editor instructions and outputs (R<n>_editor/ per file), patches\n'
            f'- `measure/R<n>/` 6-item measurement details\n'
            f'- `ROUNDS.md`, `trajectory.json`, `result.json`, `figures/` (R0..R3 diagrams)\n'
            f'- original = `runs_{cname}/A-loc/{src.name}/`\n')
        n += 1
    print(f'{n} papers collected into {out}')


if __name__ == '__main__':
    main()
