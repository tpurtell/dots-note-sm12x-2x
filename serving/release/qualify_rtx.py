#!/usr/bin/env python3
"""Sequential, resumable qualification of an already running final RTX image.

Default is plan-only. Never starts, stops or changes the serving container.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
DEPTHS=[2048,8192,32768,65536,131072,261888]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,obj):
    with path.open('x') as f:
        json.dump(obj,f,indent=2);f.write('\n')


def plan(args):
    common=['--base-url',args.base_url,'--model',args.model]
    def step(name,script,options,count,kind='jsonl'):
        return {'name':name,'script':script,'options':common+options,
                'requests':count,'format':kind}
    steps=[step('prefix','serving/qualify_prefix_xgrammar.py',[],4,'json'),
           step('reasoning-api','serving/benchmarks/reasoning_api.py',
                ['--repeats','2','--max-tokens','1024','--require-mtp','--require-boundary-chunk'],80),
           step('multimodal','serving/benchmarks/multimodal.py',[],2,'json'),
           step('seven','serving/benchmarks/workloads.py',['--suite','seven','--runs','3','--warmups','1'],28),
           step('code-agent','serving/benchmarks/code_agent.py',
                ['--depths','0','8192','24000','--runs','3','--warmups','1','--output-tokens','256'],12),
           step('clients','serving/benchmarks/clients.py',
                ['--concurrency','1','2','4','8','16','--runs','3','--warmup-runs','2',
                 '--output-tokens','256','--nonce-prefix','dots3-final-release-v1'],155,'json'),
           step('coding','serving/benchmarks/coding_clients.py',
                ['--concurrency','1','2','4','--runs','3','--warmup-runs','1','--output-tokens','8192'],48)]
    for depth in DEPTHS:
        steps.append(step(f'context-{depth}','serving/benchmarks/context.py',
                          ['--depths',str(depth),'--runs','3','--warmups','1','--output-tokens','256'],4))
    for depth in (8192,260000):
        steps.append(step(f'retrieval-{depth}','serving/benchmarks/retrieval.py',
                          ['--filler-tokens',str(depth),'--positions','0.05','0.5','0.95'],3))
    return steps


def option(step,name,default=None):
    options=step['options']
    if name not in options:return default
    start=options.index(name)+1
    end=next((i for i in range(start,len(options)) if options[i].startswith('--')),len(options))
    return options[start:end]


def validate(step,path):
    """Validate completion independently of exit status; retain measured quality misses."""
    name=step['name']; text=path.read_text()
    runs=int(option(step,'--runs',['3'])[0])
    output_tokens=int(option(step,'--output-tokens',['128'])[0])
    if step['format']=='json':
        data=json.loads(text)
        if name=='prefix':
            assert data['warm_prefix_hits']>0 and data['warm_prefix_queries']>0
            assert data['xgrammar_json']=={'answer':42}
        elif name=='multimodal':
            assert len(data['source_examples'])==2 and all(r['contract_passed'] for r in data['source_examples'])
        elif name=='clients':
            assert [p['concurrency'] for p in data['points']]==[int(x) for x in option(step,'--concurrency')]
            for point in data['points']:
                assert len(point['runs'])==runs
                for run in point['runs']:
                    assert len(run['request_results'])==point['concurrency']
                    for row in run['request_results']:
                        assert row['completion_tokens']==output_tokens and len(row['token_times_seconds'][0])==output_tokens
        return {'complete':True}
    rows=[json.loads(line) for line in text.splitlines() if line.strip()]
    assert rows[0]['record']=='meta'
    if name=='coding':
        sys.path.insert(0,str(ROOT/'serving/benchmarks'))
        from compare_coding import load_candidate
        report=load_candidate('release',path)
        assert report['complete']
        requests=[r for w in rows if w['record']=='wave' for r in w['request_results']]
        assert len(requests)==48
        assert all(r['error'] is None and r['saw_done'] and r['token_accounting_valid'] for r in requests)
        return {'complete':True,'summary':rows[-1]}
    if name=='reasoning-api':
        assert rows[-1]['record']=='summary' and rows[-1]['passed'] and rows[-1]['cases_total']==80
        return {'complete':True,'summary':rows[-1]}
    measurements=[r for r in rows if r['record']=='measurement']
    assert len(measurements)==step['requests']
    if name=='seven':
        assert rows[-1]['record']=='summary' and rows[-1]['contracts_total']==21
        assert {(r['run'],r['case']) for r in measurements}=={(run,case) for run in (-1,0,1,2) for case in rows[-1]['median_tps_by_case']}
        return {'complete':True,'summary':rows[-1]}
    if name.startswith('retrieval-'):
        assert all(r['passed'] for r in measurements)
        assert {r['position_fraction'] for r in measurements}=={.05,.5,.95}
        assert all(r['usage']['prompt_tokens']+r['usage']['completion_tokens']<=262144 for r in measurements)
    else:
        expected_depths=[int(x) for x in option(step,'--depths')]
        warmups=int(option(step,'--warmups',['1'])[0])
        assert {(r['depth'],r['run']) for r in measurements}=={(d,r) for d in expected_depths for r in range(-warmups,runs)}
        for row in measurements:
            assert row['usage']['completion_tokens']==output_tokens
            assert sum(len(c['token_ids']) for c in row['chunks'])==output_tokens
            target=row.get('actual_prompt_tokens',row['depth'])
            assert row['usage']['prompt_tokens']==target
            if name=='context-261888':
                assert row['usage']['prompt_tokens']+row['usage']['completion_tokens']==262144
    return {'complete':True}


def identity(args):
    data=json.loads(subprocess.check_output(['docker','inspect',args.container],text=True))[0]
    assert data['State']['Running'], 'container is not running'
    assert data['Image']==args.expected_image_id, 'unexpected immutable image ID'
    command=data['Args']
    def flag(name):
        for i,value in enumerate(command):
            if value==name:return command[i+1]
            if value.startswith(name+'='):return value.split('=',1)[1]
        raise ValueError(f'missing explicit server flag {name}')
    assert int(flag('--max-model-len'))==262144
    assert int(flag('--tensor-parallel-size'))==2
    assert flag('--reasoning-parser')=='dots3', 'require Dots-aware reasoning parser'
    assert flag('--tool-call-parser')=='dots', 'require Dots tool parser'
    for enabled in ('--enable-prefix-caching','--enable-auto-tool-choice'):
        assert enabled in command and '--no-'+enabled[2:] not in command, f'require explicit {enabled}'
    structured=json.loads(flag('--structured-outputs-config'))
    assert structured['backend']=='xgrammar', 'require explicit xgrammar backend'
    spec=json.loads(flag('--speculative-config'))
    assert spec['num_speculative_tokens']==args.expected_mtp and spec['method']=='mtp'
    assert spec.get('num_speculative_tokens_per_batch_size') is None, 'qualification expects fixed MTP K'
    return {'id':data['Id'],'image_id':data['Image'],'started_at':data['State']['StartedAt'],
            'restart_count':data['RestartCount'],'args':command,
            'environment_sha256':hashlib.sha256(json.dumps(data['Config']['Env'],sort_keys=True).encode()).hexdigest()}


def snapshot(args,directory,label):
    subprocess.run([sys.executable,str(ROOT/'serving/capture_runtime.py'),args.container,
                    '--output',str(directory/f'runtime-{label}.json')],check=True,cwd=ROOT)
    with (directory/f'container-{label}.log').open('x') as log:
        subprocess.run(['docker','logs','--timestamps',args.container],stdout=log,stderr=subprocess.STDOUT,check=True)
    with urllib.request.urlopen(args.base_url+'/metrics',timeout=30) as response:
        (directory/f'metrics-{label}.txt').write_bytes(response.read())


def memory_monitor(directory,stop):
    with (directory/'memory.jsonl').open('x') as f:
        while not stop.is_set():
            try:
                result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,power.draw',
                                       '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
                row={'time':time.time(),'returncode':result.returncode,'gpu_csv':result.stdout,'error':result.stderr,
                     'meminfo':Path('/proc/meminfo').read_text()}
            except Exception as exc:
                row={'time':time.time(),'error':repr(exc)}
            f.write(json.dumps(row)+'\n');f.flush();stop.wait(5)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container',required=True)
    parser.add_argument('--expected-image-id',required=True,help='Immutable local sha256 image ID')
    parser.add_argument('--expected-mtp',required=True,type=int,choices=[1,2,3,4])
    parser.add_argument('--base-url',default='http://127.0.0.1:8001')
    parser.add_argument('--model',default='dots3-note-exl3-k4')
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--execute',action='store_true',help='Run tests; omitted prints plan without contacting Docker/API')
    args=parser.parse_args();args.base_url=args.base_url.rstrip('/').removesuffix('/v1')
    steps=plan(args)
    if not args.execute:
        print(json.dumps({'requests':sum(s['requests'] for s in steps),'steps':steps},indent=2));return
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    source_paths=sorted((ROOT/'serving/benchmarks').glob('*.py'))+[Path(__file__),ROOT/'serving/qualify_prefix_xgrammar.py',ROOT/'serving/capture_runtime.py',ROOT/'serving/benchmarks/code-agent-prompt.txt']
    binding={'schema':'dots3-rtx-release-qualification-v1','identity':identity(args),
             'base_url':args.base_url,'model':args.model,'plan':steps,
             'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in source_paths}}
    manifest=out/'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())==binding, 'resume identity, source, or plan drift; use a new output directory'
    else:save(manifest,binding)
    attempt=out/f'attempt-{time.time_ns()}';attempt.mkdir()
    stop=threading.Event();monitor=threading.Thread(target=memory_monitor,args=(attempt,stop),daemon=True);monitor.start()
    try:
        snapshot(args,attempt,'before')
        for step in steps:
            assert identity(args)==binding['identity'], 'container changed during qualification'
            stage=out/step['name'];stage.mkdir(exist_ok=True);receipt=stage/'receipt.json'
            if receipt.exists():
                saved=json.loads(receipt.read_text());artifact=stage/saved['artifact']
                assert digest(artifact)==saved['sha256'], 'completed artifact modified'
                validate(step,artifact)
                print(json.dumps({'stage':step['name'],'status':'validated-resume-skip'}),flush=True);continue
            run_dir=stage/f'attempt-{time.time_ns()}';run_dir.mkdir()
            artifact=run_dir/('result.'+step['format'])
            command=[sys.executable,str(ROOT/step['script'])]+step['options']+['--output',str(artifact)]
            save(run_dir/'command.json',command)
            print(json.dumps({'stage':step['name'],'status':'running','command':command}),flush=True)
            with (run_dir/'stdout.log').open('x') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            evidence=validate(step,artifact)
            assert identity(args)==binding['identity'], 'container changed during stage'
            save(receipt,{'artifact':str(artifact.relative_to(stage)),'sha256':digest(artifact),
                          'validated':evidence,'finished_unix':time.time()})
        curves=[]
        for depth in DEPTHS:
            stage=out/f'context-{depth}';receipt=json.loads((stage/'receipt.json').read_text())
            rows=[json.loads(line) for line in (stage/receipt['artifact']).read_text().splitlines()]
            rows=[r for r in rows if r.get('record')=='measurement' and r['timed']]
            curves.append({'prompt_tokens':depth,'ttft_seconds_median':statistics.median(r['ttft_seconds'] for r in rows),
                           'effective_prefill_tps_median':statistics.median(depth/r['ttft_seconds'] for r in rows),
                           'decode_tps_median':statistics.median(r['decode_tps'] for r in rows),
                           'raw_receipt':str((stage/'receipt.json').relative_to(out))})
        completion={'completed':True,'requests_in_full_plan':sum(s['requests'] for s in steps),
                                     'curves':curves,'prefill_method':'Exact unique context prompt tokens / client TTFT from same context requests; includes first-token handoff, not a kernel-only rate',
                                     'quality_scope':'Seven/coding static misses retained in receipts; completion is not a claim of perfect model quality.'}
    finally:
        try:snapshot(args,attempt,'after')
        finally:stop.set();monitor.join(timeout=20)
    save(attempt/'complete.json',completion)


if __name__=='__main__':main()
