"""Runs one paper with the 0919 evening code (diagram carry-over, Claude reading, derived transcription, specimen acquisition, gate image verdict, per-round saving).

Uses the configuration and output format of run_final1 as is, appending _v2 to the configuration name and writing to runs_<config>_v2/ and final_v2/.
This keeps it from mixing with the trees run_final1 is currently running.

  SH_SUFFIX=_v3 python3 run_v2.py FA0002 --config haiku_sonnetgate
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_final1 as RF   # noqa: E402

SUFFIX = os.environ.get('SH_SUFFIX', '_v2')
RF.OUT = RF.API / f'final{SUFFIX}'
for k, v in list(RF.CONFIGS.items()):
    RF.CONFIGS[k + SUFFIX] = v

# Defaults for the 5-minute-per-round cap (0919 evening). 8 locations per item, gate in parallel over 3 changes each, call turn caps are a safety net.
FAST = {'SH_LOCATION_CAP': '8', 'SH_GATE_CHUNK': '3', 'SH_EDITOR_MAX_TURNS': '40', 'SH_GATE_MAX_TURNS': '6',
        'SH_FIG_READER': 'claude', 'SH_SPECIMEN': '1',
        # v4. Editor: per-file parallel text attachment (one turn); gate: REVISED full text inlined in each chunk; retire decision as a separate parallel call
        # Confirmed configuration (v8). Editor keeps the agent attachment (thinking 1024), gate is text-only. edits_parallel/agent_parallel are faster but
        # items move less (v5) or edits that cut citation by merging sentences passed the gate (v9).
        # Confirmed 0919 23:50. The only configuration that keeps the 5-minute round cap is edits_parallel (per-file edit blocks + context of other files) (v13 rounds 1~2 min).
        # The agent editor (v8/v10) moves items more but varies too much at 4~8 min per round to keep the cap. It can be selected with SH_EDITOR_MODE=agent.
        'SH_EDITOR_MODE': 'edits_parallel', 'SH_GATE_INLINE': '1', 'SH_GATE_RETIRE_SPLIT': '1',
        # Thinking-token caps. Editor (Haiku) 0, reviewer (Sonnet) 2048. Most of the time was thinking tokens
        'SH_EDITOR_THINK': '1024', 'SH_GATE_THINK': '2048', 'SH_RETIRE_MAX_TURNS': '4', 'SH_EDITOR_CONTEXT': '1',
        # v8. The gate is a text call without tools. Each chunk puts the changes + REVISED/ORIGINAL full text + recorded numeric evidence in one message and gets JSON back
        'SH_GATE_TEXT': '1', 'SH_SKILL': 'SciSlop_v0.5.md', 'SH_EG_ACK': '1',
        # Only the retire-decision call uses Haiku. Its scope is section pointers and evidence-gap observations, so the verdict is simple, and the 100~200 s tail Sonnet took becomes 20 s
        'SH_RETIRE_MODEL': 'claude-haiku-4-5-20251001', 'SH_RETIRE_THINK': '1024'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('code')
    ap.add_argument('--config', action='append', default=[])
    a = ap.parse_args()
    RF.OUT.mkdir(parents=True, exist_ok=True)
    for k, v in FAST.items():
        os.environ.setdefault(k, v)
    logf = open(RF.OUT / 'run.log', 'a')

    def log(s):
        line = time.strftime('%H:%M:%S ') + s
        print(line, flush=True); logf.write(line + '\n'); logf.flush()

    names = [c if c.endswith(SUFFIX) else c + SUFFIX for c in (a.config or ['haiku_sonnetgate'])]
    recs = []
    for c in names:
        try:
            recs.append(RF.run_one(a.code, c, log))
        except Exception as e:
            import traceback
            log(f'[{c}] ERROR {e!r}')
            (RF.OUT / f'error_{c}.txt').write_text(traceback.format_exc())
    (RF.OUT / 'all_results.json').write_text(json.dumps(recs, indent=1, ensure_ascii=False))
    log(f'All done. {len(recs)} configurations')


if __name__ == '__main__':
    main()
