"""Record separately how the method figure changed each round, and break down time and cost per round.

  python3 fig_rounds.py <code> [--runs runs6]

What is recorded
  runs6/A-loc/<code>/figure/R<n>.tex      figure source of that round (the includegraphics line if it is an image)
  runs6/A-loc/<code>/figure/R<n>.txt      text the figure renders on screen, what the measurer reads
  runs6/A-loc/<code>/figure/SUMMARY.md    per round: source kind, flagged kinds, score, what disappeared
"""
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
API = HERE.parent
sys.path.insert(0, str(API.parent / 'temp' / 'code'))
sys.path.insert(0, str(HERE))
import harness as H          # noqa: E402
import extra_items as X      # noqa: E402


def figure_block(code: str, tree: Path) -> str:
    kind, payload = X.figure_source(code, tree)
    if kind == 'tikz':
        return payload or ''
    fig = X.method_figure(code)
    stem = Path(fig).stem if fig else 'framework'
    for f in H.tex_files(tree):
        t = f.read_text(errors='ignore')
        import re
        for mm in re.finditer(r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}', t, re.S):
            if stem in mm.group(1):
                return mm.group(0)
    return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('code')
    ap.add_argument('--runs', default='runs6')
    a = ap.parse_args()
    base = API / a.runs / 'A-loc' / a.code
    t = json.load(open(base / 'trajectory.json'))
    out = base / 'figure'
    out.mkdir(parents=True, exist_ok=True)
    md = [f'# {a.code} method figure, changes per round\n',
          '`R<n>.jpeg` is the figure image of that round, `R<n>.tex` the source, `R<n>.txt` the text rendered on screen.',
          '`R<n>_erase_request.txt` holds the phrases the editor asked to erase, `R<n>_edit_report.json` the rectangles',
          'and coordinates actually erased.',
          'The measurer reads the latter: the transcription cache for image figures, the source text for figures redrawn from source.\n',
          '| Round | Figure | Flagged kinds | Score | Chars |', '|---|---|---|---|---|']
    prev_kinds = None
    rows = [('R0', H.fars_paper_dir(a.code))] + [(f"R{r['round']}", base / f"R{r['round']}") for r in t['rounds']]
    for name, tree in rows:
        if not Path(tree).exists():
            continue
        s = X.fig_exposition_score(a.code, Path(tree))
        blk = figure_block(a.code, Path(tree))
        (out / f'{name}.tex').write_text(blk)
        lines = (X.edited_transcript(Path(tree))
                 or (X._transcript(a.code) if s['source_kind'] == 'image' else X.tikz_lines(blk)))
        (out / f'{name}.txt').write_text('\n'.join(lines))
        # Keep the figure image and edit record of that round as they are. What was erased can only be verified from the image.
        fig = X.method_figure(a.code)
        if fig:
            src = Path(tree) / 'figures' / Path(fig).name
            if not src.exists():
                src = H.fars_paper_dir(a.code) / 'figures' / Path(fig).name
            if src.exists():
                import shutil
                shutil.copy2(src, out / f'{name}{src.suffix}')
        rep = Path(tree) / 'figures' / 'EDIT_REPORT.json'
        if rep.exists():
            import shutil
            shutil.copy2(rep, out / f'{name}_edit_report.json')
        req = Path(tree) / 'figures' / 'ERASE.txt'
        if req.exists():
            import shutil
            shutil.copy2(req, out / f'{name}_erase_request.txt')
        kinds = s.get('kinds') or []
        md.append(f"| {name} | {s['source_kind']} | {', '.join(kinds) or 'none'} | "
                  f"{'.' if s['slop_score'] is None else format(s['slop_score'], '.4f')} | {len(lines)} |")
        if prev_kinds is not None and set(prev_kinds) != set(kinds):
            md.append(f'|  |  | kinds gone {sorted(set(prev_kinds) - set(kinds)) or "none"}, '
                      f'kinds new {sorted(set(kinds) - set(prev_kinds)) or "none"} |  |  |')
        prev_kinds = kinds
    (out / 'SUMMARY.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
