"""Recompute every round's guards from the stored trees with the corrected guard code, write them back into
trajectory.json, and note the correction. Needed once because a variable name collision introduced on 0919 made
the citation guard read a number counter instead of the citation-key set."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
API = HERE.parent
sys.path.insert(0, str(API.parent / 'temp' / 'code'))
import harness as H

ARM = 'A-loc'
changed = 0
for code in json.load(open(API / 'papers10.json'))['codes']:
    f = API / 'runs' / ARM / code / 'trajectory.json'
    if not f.exists():
        continue
    t = json.load(open(f))
    m0 = H.fars_paper_dir(code)
    for r in t['rounds']:
        d = API / 'runs' / ARM / code / f"R{r['round']}"
        ms = API / 'runs' / ARM / code / 'measure' / f"R{r['round']}" / 'summary.json'
        meas = json.load(open(ms)) if ms.exists() else {}
        g2 = H.guards(m0, d, t['r0']['scores'], meas.get('scores', r['scores_after']),
                      {k: None for k in H.ITEMS}, meas.get('denominators', {}))
        if g2['violations'] != r['guards']['violations']:
            changed += 1
            print(f"  {code} R{r['round']}: {r['guards']['violations']} -> {g2['violations']}")
        r['guards_recorded_at_runtime'] = r['guards']
        r['guards'] = g2
    t['guard_note'] = ('guards recomputed 0919 with the corrected code. The run-time values are kept under '
                       'guards_recorded_at_runtime and must not be cited; a variable name collision made the '
                       'citation guard read a number counter.')
    json.dump(t, open(f, 'w'), indent=1, ensure_ascii=False)
print(f'verdicts changed in {changed} rounds.')
