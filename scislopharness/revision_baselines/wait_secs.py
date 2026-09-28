"""Seconds to wait after a session-limit stop. Reads the reset time the CLI printed
("You've hit your session limit - resets 2:10am (Asia/Seoul)") from the newest exec logs and
sleeps until then, so a blocked supervisor does not spend the next window's quota on polling.
Falls back to 1800s.  python3 wait_secs.py ARM"""
import glob, os, re, sys
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import runs_dir

RESET_RE = re.compile(r'resets\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)', re.I)

arm = sys.argv[1]
logs = sorted(glob.glob(str(runs_dir(arm) / '*' / 'logs' / '*_exec.txt')), key=os.path.getmtime)[-40:]
hour = minute = None
for p in reversed(logs):
    try:
        m = RESET_RE.search(open(p, errors='ignore').read())
    except OSError:
        continue
    if m:
        hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == 'pm' else 0)
        minute = int(m.group(2) or 0)
        break

if hour is None:
    print(1800)
    raise SystemExit

now = datetime.now()
tgt = now.replace(hour=hour, minute=minute, second=0, microsecond=0) + timedelta(minutes=2)
if tgt <= now:
    tgt += timedelta(days=1)
print(max(300, min(int((tgt - now).total_seconds()), 5 * 3600)))
