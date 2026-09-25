#!/usr/bin/env python3
"""Run/export the pinned public 69 Basic + 19 Hard tool-use suite."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PIN = 'cf54b4bfe705f12f71e8866f10730572497c8105'
ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(result):
    if result.get('status') != 'completed':
        raise ValueError('Tool benchmark did not complete')
    scores = result['scores']
    rows = scores['scenario_results']
    expected = {f'TC-{i:02d}' for i in range(1,89)}
    if len(rows) != 88 or {r['scenario_id'] for r in rows} != expected:
        raise ValueError('Expected exactly the pinned 88 public scenarios')
    excluded = set(scores.get('excluded_scenarios', []))
    groups = {}
    for name, selected in [('Basic',[r for r in rows if int(r['scenario_id'][3:])<=69]),
                           ('Hard',[r for r in rows if int(r['scenario_id'][3:])>=70]),('Total',rows)]:
        graded = [r for r in selected if r['scenario_id'] not in excluded]
        if any(r['points'] not in (0,1,2) or r['status'] not in ('pass','partial','fail') for r in selected):
            raise ValueError('Unknown score/status')
        earned = sum(r['points'] for r in graded)
        maximum = 2*len(graded)
        groups[name] = {'scenarios':len(selected),'graded':len(graded),'points':earned,'max_points':maximum,
                        'score_percent':100*earned/maximum if maximum else None,
                        **{state:sum(r['status']==state for r in graded) for state in ('pass','partial','fail')},
                        'excluded':[r['scenario_id'] for r in selected if r['scenario_id'] in excluded]}
    if groups['Total']['points'] != scores['total_points'] or groups['Total']['max_points'] != scores['max_points']:
        raise ValueError('Split totals disagree with upstream scoring')
    return {'schema':'dots3-tool-quality-v1','benchmark_commit':PIN,'complete':not excluded,
            'groups':groups,'upstream_final_score':scores['final_score'],
            'safety_warnings':scores.get('safety_warnings',[]),'excluded_scenarios':sorted(excluded),
            'misses':[{'scenario_id':r['scenario_id'],'status':r['status'],'points':r['points'],
                       'failure_kind':r.get('failure_kind'),'summary':r.get('summary')}
                      for r in rows if r['status']!='pass']}


def export(output):
    path=output/'tools.json'; result=json.loads(path.read_text()); summary=validate(result)
    report=Path(result['report_path'])
    if not report.is_absolute():report=output/report
    if not report.is_file():raise ValueError('Missing complete upstream Markdown trace report')
    if report.resolve()!=(output/'tools.md').resolve():shutil.copy2(report,output/'tools.md')
    db=output/'data/benchmarks.sqlite'
    if not db.is_file():raise ValueError('Missing mandatory raw SQLite persistence')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['| Suite | Points | Score | Pass / Partial / Fail |','|---|---:|---:|---:|']
    for name,g in summary['groups'].items():
        score='N/A' if g['score_percent'] is None else f"{g['score_percent']:.2f}%"
        lines.append(f"| {name} | {g['points']}/{g['max_points']} | {score} | {g['pass']} / {g['partial']} / {g['fail']} |")
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    files=[p for p in output.rglob('*') if p.is_file() and p.name!='receipt.json']
    receipt={'complete':summary['complete'],'benchmark_commit':PIN,'artifact':'tools.json','sha256':sha(path),
             'raw_artifacts':{str(p.relative_to(output)):sha(p) for p in sorted(files)}}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not summary['complete']:raise ValueError('Infrastructure exclusions prevent qualification; raw results retained')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout',type=Path,default=ROOT/'.cache/tool-eval-bench-cf54b4b')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base-url',default='http://127.0.0.1:8001/v1')
    p.add_argument('--model',default='dots3-note-exl3-k4')
    p.add_argument('--parallel',type=int,default=8)
    p.add_argument('--reference-date',default='2026-09-07')
    p.add_argument('--plan',action='store_true')
    p.add_argument('--export-only',action='store_true')
    a=p.parse_args();a.output=a.output.resolve();a.checkout=a.checkout.resolve()
    if a.export_only:print(json.dumps(export(a.output)));return
    if a.parallel<1:raise ValueError('parallel must be positive')
    revision=subprocess.check_output(['git','-C',str(a.checkout),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(a.checkout),'status','--porcelain','--untracked-files=no'],text=True)
    if revision!=PIN or dirty:raise ValueError('Expected clean pinned tool-eval-bench checkout '+PIN)
    python=a.checkout/'.venv/bin/python'
    if not python.is_file():raise ValueError('Run uv sync --frozen --no-dev in pinned checkout first')
    base=a.base_url.rstrip('/');base=base[:-3] if base.endswith('/v1') else base
    command=[str(python),'-m','tool_eval_bench','run','--model',a.model,'--backend','vllm','--base-url',base+'/v1',
             '--hardmode','--parallel',str(a.parallel),'--timeout','900','--max-turns','8','--trials','1','--temperature','0',
             '--reference-date',a.reference_date,'--backend-kwargs','{"chat_template_kwargs":{"enable_thinking":true}}',
             '--json-file',str(a.output/'tools.json'),'--output-dir',str(a.output/'runs'),'--no-live']
    provenance={'schema':'dots3-tool-quality-run-v1','benchmark_commit':revision,'command':command,
                'uv_lock_sha256':sha(a.checkout/'uv.lock'),'wrapper_sha256':sha(__file__),
                'platform_scope':'Caller must bind final image/profile runtime snapshots to this run',
                'basic':69,'hard':19,'total':88,'reference':'Qwen recipe pinned public suite; no held-out scenario pack'}
    if a.plan:print(json.dumps(provenance,indent=2));return
    if a.output.exists():raise FileExistsError('Use a fresh output directory; raw runs are never overwritten')
    a.output.mkdir(parents=True)
    (a.output/'manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    with (a.output/'run.log').open('w') as log:
        rc=subprocess.run(command,cwd=a.output,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    (a.output/'exit.json').write_text(json.dumps({'returncode':rc,'finished_unix':time.time()})+'\n')
    if rc:raise SystemExit(rc)
    print(json.dumps(export(a.output)))


if __name__=='__main__':main()
