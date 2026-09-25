#!/usr/bin/env python3
"""CPU-only final-stage planning, trace integrity and score export checks."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import tool_quality
import qualify_rtx
import qualify_spark
import report

args=SimpleNamespace(base_url='http://localhost:8001',model='test',max_model_len=524288)
for runner in (qualify_rtx,qualify_spark):
    steps=runner.plan(args)
    tools=[s for s in steps if s['name']=='tool-quality']
    assert len(tools)==1 and tools[0]['requests']==88
    step=tools[0]
    with TemporaryDirectory() as directory:
        root=Path(directory)
        artifact,command=qualify_rtx.stage_command(step,root)
        assert artifact==root/'tool-quality/tools.json'
        assert command[-1]==str(root/'tool-quality')
        artifact.parent.mkdir()
        rows=[{'scenario_id':f'TC-{i:02d}','points':2 if i<=69 else 1,
               'status':'pass' if i<=69 else 'partial'} for i in range(1,89)]
        data={'status':'completed','scores':{'scenario_results':rows,'total_points':157,'max_points':176,'final_score':89}}
        artifact.write_text(json.dumps(data))
        files={'tools.md':'trace','data/benchmarks.sqlite':'fixture',
               'manifest.json':json.dumps({'benchmark_commit':tool_quality.PIN,'command':['run','--hardmode']}),
               'exit.json':'{"returncode":0}'}
        for name,text in files.items():
            p=artifact.parent/name;p.parent.mkdir(exist_ok=True);p.write_text(text)
        hashes={str(p.relative_to(artifact.parent)):qualify_rtx.digest(p) for p in artifact.parent.rglob('*') if p.is_file()}
        (artifact.parent/'receipt.json').write_text(json.dumps({'complete':True,'benchmark_commit':tool_quality.PIN,'sha256':qualify_rtx.digest(artifact),'raw_artifacts':hashes}))
        result=runner.validate(step,artifact,524288)
        assert result['summary']['groups']['Basic']['score_percent']==100
        assert result['summary']['groups']['Hard']['score_percent']==50
        assert report.stage_metrics('tool-quality',data)['groups']['Total']['points']==157
        (artifact.parent/'tools.md').write_text('tampered')
        try:runner.validate(step,artifact,524288)
        except AssertionError:pass
        else:raise AssertionError('tampered trace accepted')
print('PASS CPU: both mandatory plans, directory output, partial-credit splits, report metrics, and raw-trace tamper rejection')
