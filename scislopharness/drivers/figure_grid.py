"""Collect the method figures per model and per round into a single grid.

Rows are rounds (from R0), columns are editor models. Under each cell we write the fig_exposition score and flagged kinds at that point,
and the number of phrases erased in that round. The aim is to see at a glance how the figure evolved.

  python3 figure_grid.py <code> --run runs6=Haiku --run runs6_sonnet=Sonnet [--out grid.png]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent
sys.path.insert(0, str(API.parent / 'temp' / 'code'))
sys.path.insert(0, str(HERE))
import harness as H       # noqa: E402
import extra_items as X   # noqa: E402

ARM = 'A-loc'


def round_figure(code: str, runs: str, rnd: str) -> Path | None:
    fig = X.method_figure(code)
    if not fig:
        return None
    name = Path(fig).name
    if rnd == 'R0':
        p = H.fars_paper_dir(code) / 'figures' / name
        return p if p.exists() else None
    p = API / runs / ARM / code / rnd / 'figures' / name
    return p if p.exists() else None


def round_state(code: str, runs: str, rnd: str) -> dict:
    tree = H.fars_paper_dir(code) if rnd == 'R0' else API / runs / ARM / code / rnd
    if not Path(tree).exists():
        return {}
    s = X.fig_exposition_score(code, Path(tree))
    rep = Path(tree) / 'figures' / 'EDIT_REPORT.json'
    if rep.exists():
        try:
            s['edit'] = json.loads(rep.read_text())
        except Exception:
            pass
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('code')
    ap.add_argument('--run', action='append', default=[], help='runsdir=Label')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--out', default='')
    ap.add_argument('--width', type=int, default=560)
    a = ap.parse_args()
    from PIL import Image, ImageDraw
    cols = [tuple(r.split('=', 1)) for r in a.run]
    rounds = ['R0'] + [f'R{i}' for i in range(1, a.rounds + 1)]
    W, CAP, PAD, HEAD = a.width, 44, 12, 34
    cells, heights = {}, {}
    for rnd in rounds:
        for runs, _ in cols:
            p = round_figure(a.code, runs, rnd)
            if p:
                im = Image.open(p).convert('RGB')
                sc = W / im.width
                im = im.resize((W, max(1, int(im.height * sc))), Image.LANCZOS)
                cells[(rnd, runs)] = im
                heights[rnd] = max(heights.get(rnd, 0), im.height)
    if not cells:
        print('no figures'); return
    total_h = HEAD + sum(heights.get(r, 0) + CAP + PAD for r in rounds)
    total_w = 70 + len(cols) * (W + PAD)
    sheet = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for j, (runs, label) in enumerate(cols):
        d.text((70 + j * (W + PAD) + 4, 10), f'{label}  ({runs})', fill=(0, 0, 0))
    y = HEAD
    summary = []
    for rnd in rounds:
        if rnd not in heights:
            continue
        d.text((8, y + heights[rnd] // 2), rnd, fill=(150, 0, 0))
        for j, (runs, label) in enumerate(cols):
            im = cells.get((rnd, runs))
            x = 70 + j * (W + PAD)
            if im is None:
                d.text((x + 4, y + 10), '(none)', fill=(120, 120, 120)); continue
            sheet.paste(im, (x, y))
            st = round_state(a.code, runs, rnd)
            sc = st.get('slop_score')
            kinds = ', '.join(st.get('kinds') or []) or 'none'
            ed = (st.get('edit') or {})
            nerase = ed.get('n_erased')
            pres = (ed.get('preservation') or {}).get('outside_ratio')
            line1 = f"score {('.' if sc is None else format(sc, '.4f'))}   flagged kinds {kinds}"
            line2 = (f"phrases erased this round {nerase}, changed outside target {pres}"
                     if nerase is not None else 'no figure edit this round')
            d.text((x + 2, y + im.height + 4), line1, fill=(0, 0, 0))
            d.text((x + 2, y + im.height + 20), line2, fill=(70, 70, 70))
            summary.append({'round': rnd, 'run': runs, 'label': label, 'score': sc,
                            'kinds': st.get('kinds'), 'n_erased': nerase, 'outside_ratio': pres})
        y += heights[rnd] + CAP + PAD
    out = Path(a.out) if a.out else API / 'figure_edit_study' / f'{a.code}_rounds_by_model.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    (out.with_suffix('.json')).write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f'{out}  ({sheet.width}x{sheet.height})')
    for s in summary:
        print(f"  {s['round']:3} {s['label']:8} score {s['score']}  kinds {s['kinds']}  erased {s['n_erased']}")


if __name__ == '__main__':
    main()
