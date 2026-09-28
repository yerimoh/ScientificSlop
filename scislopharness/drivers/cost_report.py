"""Break one paper's time and cost down by round and by stage.

There are three stages: editor call, gate review, measurement. Measurement splits into the 4 deterministic items and argument graph;
the latter makes a label call and a PMI computation per introduction sentence and takes most of the time.

  python3 cost_report.py <code> [--runs runs6]
"""
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
API = HERE.parent
sys.path.insert(0, str(API.parent / 'temp' / 'code'))
import harness as H   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('code')
    ap.add_argument('--runs', default='runs6')
    a = ap.parse_args()
    base = API / a.runs / 'A-loc' / a.code
    t = json.load(open(base / 'trajectory.json'))
    md = [f'# {a.code} time and cost\n',
          f"Editor {t['model']}, rounds {len(t['rounds'])}, stop {t['stop']}.\n",
          '| Round | Editor s | Editor $ | Review s | Review $ | Measure s | Round total s | Round total $ |',
          '|---|---|---|---|---|---|---|---|']
    te = tr = tm = 0.0
    ce = cr = 0.0
    for r in t['rounds']:
        e_s = r['call']['wall_s'] or 0
        e_c = r['call'].get('cost_usd') or 0
        g = r.get('gate') or {}
        gc = (g.get('call') or {})
        r_s = gc.get('wall_s') or 0
        r_c = gc.get('cost_usd') or 0
        m_s = r.get('measure_s') or 0
        te += e_s; tr += r_s; tm += m_s; ce += e_c; cr += r_c
        md.append(f"| R{r['round']} | {e_s:.0f} | {e_c:.3f} | {r_s:.0f} | {r_c:.3f} | {m_s:.0f} | "
                  f"{e_s + r_s + m_s:.0f} | {e_c + r_c:.3f} |")
    md.append(f"| **Total** | **{te:.0f}** | **{ce:.3f}** | **{tr:.0f}** | **{cr:.3f}** | **{tm:.0f}** | "
              f"**{te + tr + tm:.0f}** | **{ce + cr:.3f}** |")
    n = max(1, len(t['rounds']))
    md += ['', f"One paper: wall clock {(te + tr + tm)/60:.0f} min, API ${ce + cr:.2f}.",
           f"Per-round mean: editor {te/n:.0f}s, review {tr/n:.0f}s, measure {tm/n:.0f}s.", '',
           'Most of the time is measurement, dominated by argument graph. If even one introduction sentence changes, the label cache',
           'no longer matches, so every sentence is relabeled and PMI is recomputed. Sending PMI to a GPU job would cut this.', '']
    ag = []
    for r in t['rounds']:
        x = ((r.get('extra') or {}).get('argument_graph') or {})
        ag.append(f"R{r['round']} {x.get('status')} score {x.get('slop_score')} key claims {x.get('n_key_claims')}")
    md.append('argument graph status per round. ' + '; '.join(ag))
    out = base / 'COST.md'
    out.write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
