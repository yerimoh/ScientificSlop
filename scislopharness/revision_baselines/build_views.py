"""Prose views of revised AI trees for the text detectors, bench165 rules (build_views165.py).
  python3 build_views.py --arm a4_slop --round 1 [--round 3]
-> views/<arm>/R<n>/AI_<code>/body.txt and views/<arm>/R<n>/texts.jsonl (AI side only; the
human side reuses bench165 scores)."""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EOR, ROOT, round_dir, key  # noqa: E402
from measure_rounds import measured_codes  # noqa: E402
SB = ROOT / 'paper/draft_v6/scislopbench'
sys.path.insert(0, str(SB / 'data/scripts'))
sys.path.insert(0, str(SB / 'bench165/scripts'))
from slopbench_lib import flatten, split_body  # noqa: E402
import importlib.util
spec = importlib.util.spec_from_file_location('bv165', str(SB / 'bench165/scripts/build_views165.py'))
# build_views165 runs at import; take prose_view by exec of its source up to the function only
src = open(SB / 'bench165/scripts/build_views165.py').read()
ns = {}
exec(src.split('def hu_dir')[0].replace('from slopbench_lib import', '# from slopbench_lib import'), ns)
prose_view = ns['prose_view']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True); ap.add_argument('--round', type=int, action='append', required=True)
    a = ap.parse_args()
    for n in a.round:
        out = EOR / 'views' / key(a.arm) / f'R{n}'
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for c in measured_codes(a.arm, n):
            d = round_dir(a.arm, c, n)
            body, _, _ = split_body(flatten(str(d / 'main.tex')))
            (out / f'AI_{c}').mkdir(exist_ok=True)
            txt = prose_view(body)
            (out / f'AI_{c}' / 'body.txt').write_text(txt)
            rows.append(dict(id=f'AI_{c}', label=1, year=0, source=f'{a.arm}_R{n}', text=txt))
        with open(out / 'texts.jsonl', 'w') as w:
            for r in rows:
                w.write(json.dumps(r) + '\n')
        print(f'{a.arm} R{n}: {len(rows)} views -> {out}')


if __name__ == '__main__':
    main()
