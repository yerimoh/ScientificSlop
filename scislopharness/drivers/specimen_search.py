"""Checks the premise of the evidence-gap item: is there really no specimen anywhere in the project that the manuscript could display?

So far only one folder, `code/exp/EXPERIMENT_RESULTS`, was shown to the editor, and if nothing was there we wrote that nothing exists. That folder
is only part of the project's records. This step builds an index of every record the project produced itself and hands it to a read-only
searcher that looks for a displayable specimen. If one is found we get the file path and a verbatim quotation; if not, we get where and
how the search looked. Either way the result is recorded and becomes auditable.

Repositories cloned from outside are not the project's records and are excluded. Any directory containing at least one of .git, setup.py,
pyproject.toml is treated wholesale as someone else's code.

  python3 specimen_search.py [--papers FA0002,FA0005] [--workers 5]
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent
sys.path.insert(0, str(API.parent / 'temp' / 'code'))
import harness as H          # noqa: E402
import quality_gate as Q     # noqa: E402

OUT = API / 'specimen_search'
RESULT = API / 'SPECIMEN_SEARCH.md'
RESULT_JSONL = API / 'specimen_search.jsonl'
BIN = {'.pdf', '.png', '.jpg', '.jpeg', '.pkl', '.pt', '.bin', '.safetensors', '.npy', '.npz',
       '.zip', '.tar', '.gz', '.so', '.whl', '.ico', '.svg', '.woff', '.woff2', '.ttf'}
VENDOR_MARK = {'.git', 'setup.py', 'pyproject.toml', 'package.json'}
lock = threading.Lock()


OWN_DIRS = {'', 'code', 'code/exp', 'code/idea', 'traces', 'code/exp/EXPERIMENT_RESULTS'}


def vendored_roots(root: Path) -> list[Path]:
    """Top-level directories of external repositories cloned inside the project.

    The FARS project's own directories (code, code/exp, code/idea, traces) may themselves carry Python package markers,
    so they are excluded from the candidates. Without this, code alone gets caught as an external repository and the whole experiment record disappears.
    That is exactly what happened in the first run on 0919. EXPERIMENT_RESULTS under the records is never excluded in any case."""
    out = []
    for d in sorted(root.rglob('*')):
        if not d.is_dir():
            continue
        rel = str(d.relative_to(root))
        if rel in OWN_DIRS or 'EXPERIMENT_RESULTS' in d.parts:
            continue
        if any((d / m).exists() for m in VENDOR_MARK):
            if not any(str(d).startswith(str(o) + '/') for o in out):
                out.append(d)
    return out


def own_records(root: Path) -> tuple[list[Path], list[Path]]:
    """(the project's own record files, top-level directories of the excluded external repositories)"""
    vend = vendored_roots(root)
    keep = []
    for f in root.rglob('*'):
        if not f.is_file():
            continue
        parts = set(f.relative_to(root).parts)
        if 'writing' in parts or '__pycache__' in parts or '.git' in parts:
            continue
        if 'EXPERIMENT_RESULTS' not in f.parts and any(str(f).startswith(str(v) + '/') for v in vend):
            continue
        if f.suffix.lower() in BIN:
            continue
        if f.suffix.lower() in ('.py', '.sh', '.cu', '.c', '.h', '.js', '.css'):
            continue
        keep.append(f)
    return keep, vend


def build_dir(code: str) -> tuple[Path, dict]:
    root = list((H.ROOT / 'fars/papers').glob(f'{code}_*'))[0]
    d = OUT / code
    if d.exists():
        shutil.rmtree(d)
    (d / 'records').mkdir(parents=True)
    files, vend = own_records(root)
    files.sort(key=lambda f: f.stat().st_size)
    copied, skipped_big = [], []
    total = 0
    for f in files:
        sz = f.stat().st_size
        if sz > 2_000_000 or total > 40_000_000:
            skipped_big.append((str(f.relative_to(root)), sz)); continue
        rel = f.relative_to(root)
        (d / 'records' / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, d / 'records' / rel)
        copied.append((str(rel), sz)); total += sz
    H.stage(H.fars_paper_dir(code), d / 'MANUSCRIPT')
    idx = [f'# {code} 기록 목록. 파일 {len(copied)}개, 합계 {total/1024:.0f} KB\n',
           '`records/` 아래가 이 프로젝트가 직접 만든 기록 전부다. 외부에서 클론한 저장소는 프로젝트의 기록이',
           '아니므로 제외했다. `MANUSCRIPT/` 는 현재 원고이며 근거가 아니라 대조용이다.\n']
    if vend:
        idx.append('제외한 외부 저장소: ' + ', '.join(str(v.relative_to(root)) for v in vend) + '\n')
    if skipped_big:
        idx.append('크기 때문에 복사하지 않은 파일: ' + ', '.join(f'{p} ({s//1024}KB)' for p, s in skipped_big) + '\n')
    idx.append('| 파일 | 바이트 |'); idx.append('|---|---|')
    for p, s in sorted(copied):
        idx.append(f'| `{p}` | {s} |')
    (d / 'RECORDS_INDEX.md').write_text('\n'.join(idx) + '\n')
    return d, {'n_records': len(copied), 'bytes': total,
               'vendored_excluded': [str(v.relative_to(root)) for v in vend],
               'skipped_big': skipped_big}


INSTRUCTION = """이 원고는 집계 결과만 보고하고 구체적인 실물을 하나도 보여 주지 않는다는 판정을 받았다. 실물이란 예시 입력과 그 출력, 하나의 사례, 실패 사례, 풀어 쓴 예제, 인용된 표본을 말한다. 당신이 할 일은 그 실물이 이 프로젝트의 기록 어딘가에 실제로 존재하는지 찾아 결론을 내리는 것이다.

이 디렉터리에 `records/`, `RECORDS_INDEX.md`, `MANUSCRIPT/` 가 있다. `records/` 는 이 프로젝트가 직접 만든 기록 전부이고 `RECORDS_INDEX.md` 는 그 전체 목록이다. `MANUSCRIPT/` 는 현재 원고이며 근거가 아니라 무엇이 이미 실려 있는지 확인하는 용도다.

먼저 `RECORDS_INDEX.md` 를 읽어 무엇이 있는지 파악하고, Grep 과 Read 로 `records/` 안을 실제로 뒤져라. 집계 지표만 있는 파일을 넘기고 개별 사례가 담긴 곳을 찾아라. 예를 들어 모델에 들어간 입력 문자열, 모델이 낸 출력 문자열, 하나의 질문과 답, 하나의 스키마, 하나의 전사본, 하나의 실패 기록이다. 파일 이름만 보고 판단하지 말고 내용을 열어 확인하라. 기록이 다른 파일을 가리키기만 하고 그 파일이 없을 수도 있으니, 언급된 경로가 실제로 존재하는지 확인하라.

찾았다면 원고에 그대로 실을 수 있는 가장 좋은 것 하나를 고르고, 원문을 글자 그대로 옮겨라. 요약하거나 다듬지 마라. 못 찾았다면 어디를 어떻게 확인했는지 적어라.

이 디렉터리에 `SPECIMEN.json` 을 다음 형태로 써라. 다른 파일은 만들지 말고 아무것도 수정하지 마라.
{"found": true 또는 false,
 "searched": ["실제로 열어 본 파일 경로", ...],
 "reason": "찾았다면 이것이 왜 원고가 세는 실물인지, 못 찾았다면 기록에 무엇만 있고 무엇이 없는지 한두 문장",
 "specimen": {"file": "records 아래 경로", "verbatim": "원문 그대로", "what_it_is": "이것이 무엇인지 한 문장"} 또는 null}
질문하지 말고 요약하지 말고, `SPECIMEN.json` 이 생기면 턴을 끝내라."""


def search_one(code: str, model: str):
    d, meta = build_dir(code)
    res = H.run_cli_json(['-p', INSTRUCTION, '--model', model, '--permission-mode', 'acceptEdits',
                          '--add-dir', str(d), '--disallowed-tools', 'WebSearch,WebFetch,Bash,Edit',
                          '--settings', str(Q.GATE_SETTINGS), '--restricted'], d, timeout=2400)
    (d / 'searcher_stdout.txt').write_text((res.get('result') or '') + '\n\n=== STDERR ===\n' + (res.get('stderr') or ''))
    rec = {'code': code, 'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'model': model, **meta,
           'wall_s': res.get('wall_s'), 'cost_usd': res.get('cost_usd'), 'found': None}
    f = d / 'SPECIMEN.json'
    if f.exists():
        try:
            import re
            j = json.loads(re.sub(r'^```(?:json)?|```$', '', f.read_text().strip(), flags=re.M))
            rec.update({'found': bool(j.get('found')), 'reason': j.get('reason'),
                        'searched': j.get('searched'), 'specimen': j.get('specimen')})
        except Exception as e:
            rec['parse_error'] = repr(e)
    with lock:
        with open(RESULT_JSONL, 'a') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"{code} found={rec['found']} records {rec['n_records']} "
              f"excluded repos {len(rec['vendored_excluded'])} ${rec.get('cost_usd') or 0:.2f} "
              f"{(rec.get('reason') or '')[:110]}", flush=True)
    return rec


def write_md():
    rows = [json.loads(l) for l in open(RESULT_JSONL) if l.strip()]
    seen, uniq = set(), []
    for r in reversed(rows):
        if r['code'] not in seen:
            seen.add(r['code']); uniq.append(r)
    uniq.reverse()
    md = ['# Does a displayable specimen exist in the project?\n',
          'This check separates two reasons why the evidence-gap item does not close: the records lack a specimen, or the harness failed to show one.',
          'Every record the project produced itself is indexed, and a read-only searcher actually opens the files to look for one.',
          'Repositories cloned from outside are not records of the project and are excluded.\n',
          '| Paper | Specimen found | Record files | Excluded external repos | Basis |', '|---|---|---|---|---|']
    for r in uniq:
        md.append(f"| {r['code']} | {'yes' if r.get('found') else 'no'} | {r['n_records']} | "
                  f"{len(r.get('vendored_excluded') or [])} | {(r.get('reason') or '').replace('|', ' ')[:200]} |")
    found = [r for r in uniq if r.get('found')]
    md += ['', f"Of the {len(uniq)} papers checked, {len(found)} have a specimen.", '']
    for r in found:
        sp = r.get('specimen') or {}
        md += [f"## Specimen found in {r['code']}", f"File `{sp.get('file')}`. {sp.get('what_it_is')}", '',
               '```', str(sp.get('verbatim'))[:1500], '```', '']
    RESULT.write_text('\n'.join(md) + '\n')
    print(f'{RESULT} updated, {len(uniq)} papers')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--papers', default='')
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--model', default='claude-sonnet-5')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    codes = a.papers.split(',') if a.papers else json.load(open(API / 'papers10.json'))['codes']
    sem = threading.Semaphore(a.workers)

    def worker(c):
        with sem:
            try:
                search_one(c, a.model)
            except Exception as e:
                import traceback
                with lock:
                    print(f'{c} ERROR {e!r}', flush=True)
                (OUT / f'error_{c}.txt').write_text(traceback.format_exc())

    ts = [threading.Thread(target=worker, args=(c,)) for c in codes]
    for t in ts:
        t.start(); time.sleep(2)
    for t in ts:
        t.join()
    write_md()


if __name__ == '__main__':
    main()
