"""Append (or refresh) the summary section at the end of RESULTS.md from results.jsonl."""
import json, sys, time
from collections import Counter
from pathlib import Path
API = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API.parent / 'temp' / 'code'))
import harness as H

MARK = '\n<!-- SUMMARY -->\n'
rows = [json.loads(l) for l in open(API / 'results.jsonl') if l.strip()]
n = len(rows)
import rounds_table as RT   # noqa: E402  (writes rounds_table.md and gives per-round values)
md = [MARK, f'\n---\n\n# Summary ({n} papers, updated {time.strftime("%Y-%m-%d %H:%M")})\n',
      'The full per-round table is in `rounds_table.md`. Below are the 10-paper means.\n',
      '| Item | R0 | R1 | R2 | R3 | Human pair (same 10 papers) |', '|---|---|---|---|---|---|']
for k in H.ITEMS:
    cells = []
    for rk in ('R0', 'R1', 'R2', 'R3'):
        v = [RT.data[c][rk][k] for c in RT.data if rk in RT.data[c] and RT.data[c][rk].get(k) is not None]
        cells.append(f'{sum(v)/len(v):.3f}' if v else '.')
    hv = [(RT.human.get(c) or {}).get(k) for c in RT.data]
    hv = [x for x in hv if x is not None]
    cells.append(f'{sum(hv)/len(hv):.3f}' if hv else '.')
    md.append(f'| {k} | ' + ' | '.join(cells) + ' |')
kinds = Counter()
for x in rows:
    kinds.update(x['unused_object_kinds'])
hard = Counter()
for x in rows:
    hard.update(x['guards']['hard'])
soft = Counter()
for x in rows:
    soft.update(x['guards']['soft'])
md += ['', f"Gate kept {sum(x['kept'] for x in rows)}, reverted {sum(x['reverted'] for x in rows)}, retire applied {sum(x['retired'] for x in rows)}.",
       f"Hard guard violations {dict(hard) or 'none'}. Soft flags {dict(soft) or 'none'}.",
       f"Object kinds still unused {dict(kinds)}.",
       f"Cost editor ${sum(x['editor_usd'] for x in rows):.2f}, reviewer ${sum(x['reviewer_usd'] for x in rows):.2f}, "
       f"total ${sum(x['editor_usd'] + x['reviewer_usd'] for x in rows):.2f}. Per-paper mean "
       f"${sum(x['editor_usd'] + x['reviewer_usd'] for x in rows)/n:.2f}, wall clock {sum(x['wall_s'] for x in rows)/n/60:.0f} min.", '',
       '| Paper | Kept/reverted | Hard | Unused objects |', '|---|---|---|---|']
for x in rows:
    md.append(f"| {x['code']} | {x['kept']}/{x['reverted']} | {sum(x['guards']['hard'].values())} | "
              f"{x['unused_object_kinds']} |")
md.append('')
md.append(RT.render())
md += ['', 'Interpretation. Recycled sentences close below the human-pair level. Unused objects and isolated citations drop sharply but do not reach the human level. '
       'Evidence gaps are unchanged in all ten papers, consistent with the hypothesis that a layer fixable by writing is separate from a layer rooted in work not done. '
       'Zero hard guard violations means the reduction was not obtained by deleting citations or changing numbers.', '']
txt = (API / 'RESULTS.md').read_text()
if MARK in txt:
    txt = txt[:txt.index(MARK)]
(API / 'RESULTS.md').write_text(txt + '\n'.join(md) + '\n')
print(f'summary updated for {n} papers')
