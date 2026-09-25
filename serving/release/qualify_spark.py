#!/usr/bin/env python3
"""Qualify two already running Spark containers; plan-only unless --execute.

Benchmark definitions and common validators come from qualify_rtx.py. No serving
container is started, stopped, or reconfigured by this runner.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import urllib.request

import qualify_rtx as shared
import evidence_lineage

ROOT=shared.ROOT

# Executed through ssh python3 -; uses only CPU/inspection commands. Full snapshots
# invoke the existing remote capture_runtime.py and retain its temporary receipt.
REMOTE = r'''
import datetime,hashlib,json,os,pathlib,subprocess,sys,tempfile,time
mode,container,project=sys.argv[1:]
def command(*args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=25)
d=json.loads(command('docker','inspect',container))[0]
runtime=next(x['Source'] for x in d['Mounts'] if x['Destination']=='/root/.cache/vllm-runtime')
root=pathlib.Path(runtime);pid=int((root/'memory-watch.pid').read_text())
proc=pathlib.Path('/proc')/str(pid)
cmdline=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
boottime=next(int(line.split()[1]) for line in pathlib.Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
started=boottime+int(stat[19])/os.sysconf('SC_CLK_TCK')
marker=root/f'memory-watch-ready-{pid}'
logpath=root/'memory-watch.log'
with logpath.open('rb') as f:
 f.seek(max(0,logpath.stat().st_size-65536));tail=f.read().decode(errors='replace')
samples=[]
for line in tail.splitlines():
 try:row=json.loads(line)
 except ValueError:continue
 if 'available_bytes'in row:samples.append(row)
mem={line.split(':')[0]:int(line.split()[1])*1024 for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0]in ('MemTotal','MemAvailable','SwapTotal','SwapFree')}
guard={'pid':pid,'proc_state':stat[0],'start_ticks':stat[19],'started_unix':started,'cmdline':cmdline,'log_path':str(logpath),'last_sample':samples[-1] if samples else None,'ready_marker_exists':marker.exists(),'ready_marker_value':marker.read_text().strip() if marker.exists() else None,'ready_note':'Launcher consumes readiness marker; absence alone does not imply unready.'}
identity={'id':d['Id'],'image_id':d['Image'],'started_at':d['State']['StartedAt'],'restart_count':d['RestartCount'],'args':d['Args'],'environment_sha256':hashlib.sha256(json.dumps(d['Config']['Env'],sort_keys=True).encode()).hexdigest(),'runtime_mount':next(x for x in d['Mounts'] if x['Destination']=='/root/.cache/vllm-runtime'),'guard_pid':pid,'guard_start_ticks':stat[19]}
env=[e for e in d['Config']['Env'] if e.startswith(('NCCL_','GLOO_','VLLM_HOST_IP=','DOTS3_','B12X_','CUTE_','OMP_'))]
result={'identity':identity,'running':d['State']['Running'],'physical_memory_bytes':mem,'guard':guard,'network_environment':env,'time':time.time()}
if mode!='sample':
 image=json.loads(command('docker','image','inspect',d['Image']))[0]
 result['image_repo_digests']=image.get('RepoDigests',[])
 result['architecture']=image.get('Architecture')
if mode=='snapshot':
 temp=pathlib.Path(project)/'.cache'/'qualification-runtime';temp.mkdir(parents=True,exist_ok=True)
 output=temp/f'{container}-{time.time_ns()}.json'
 command('python3',str(pathlib.Path(project)/'serving/capture_runtime.py'),container,'--output',str(output))
 result['runtime']=json.loads(output.read_text());result['runtime_remote_path']=str(output)
 result['container_log']=command('docker','logs','--timestamps',container)
 result['guard_log']=logpath.read_text(errors='replace')
 result['nic_addresses']=command('ip','-j','address','show')
 result['nic_links']=command('ip','-j','link','show')
 result['source_sha256']={name:hashlib.sha256((pathlib.Path(project)/name).read_bytes()).hexdigest() for name in ('serving/capture_runtime.py','serving/watch_spark_memory.py')}
print(json.dumps(result))
'''


def plan(args):
    reference=shared.plan(args)
    result=[copy.deepcopy(s) for s in reference if not s['name'].startswith(('context-','retrieval-'))]
    context=next(s for s in reference if s['name'].startswith('context-'))
    retrieval=next(s for s in reference if s['name'].startswith('retrieval-'))
    depths=list(shared.DEPTHS[:-1])
    if args.max_model_len==524288:depths.append(262144)
    depths.append(args.max_model_len-256)
    for depth in depths:
        item=copy.deepcopy(context);item['name']=f'context-{depth}'
        item['options'][item['options'].index('--depths')+1]=str(depth);result.append(item)
    for depth in (8192,args.max_model_len-2144):
        item=copy.deepcopy(retrieval);item['name']=f'retrieval-{depth}'
        item['options'][item['options'].index('--filler-tokens')+1]=str(depth);result.append(item)
    return evidence_lineage.apply_plan(result,getattr(args,'inherit_evidence',None))


def validate(step,path,limit):
    if step.get('evidence_mode')=='inherited':
        return evidence_lineage.validate_inherited(step,path)
    if not step['name'].startswith('retrieval-'):
        result=shared.validate(step,path,limit)
    else:
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert rows[0]['record']=='meta'
        records=[r for r in rows if r['record']=='measurement']
        assert len(records)==step['requests']==3 and all(r['passed'] for r in records)
        assert {r['position_fraction'] for r in records}=={.05,.5,.95}
        assert all(r['usage']['prompt_tokens']+r['usage']['completion_tokens']<=limit for r in records)
        result={'complete':True,'actual_prompt_tokens':[r['usage']['prompt_tokens'] for r in records]}
    if step['name']==f'context-{limit-256}':
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        assert all(r['usage']['prompt_tokens']+r['usage']['completion_tokens']==limit for r in rows if r.get('record')=='measurement')
        result['exact_total_token_boundary']=limit
    return result


def hosts(args):
    return [(args.head_host,args.head_container),(args.worker_host,args.worker_container)]


def remote(args,host,container,mode):
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,
             shlex.join(['python3','-',mode,container,args.remote_project])]
    result=subprocess.run(command,input=REMOTE,text=True,capture_output=True,timeout=100,check=True)
    return json.loads(result.stdout)


def validate_host(args,data,container):
    assert data['running'], f'{container} stopped'
    ident=data['identity'];assert ident['image_id']==args.expected_image_id
    command=ident['args']
    def flag(name):
        matches=[v.split('=',1)[1] if '='in v else command[i+1] for i,v in enumerate(command) if v==name or v.startswith(name+'=')]
        assert matches and len(set(matches))==1, f'missing/conflicting {name}'
        return matches[0]
    for key,value in {'--max-model-len':str(args.max_model_len),'--tensor-parallel-size':'2',
                      '--reasoning-parser':'dots3','--tool-call-parser':'dots'}.items():assert flag(key)==value
    assert float(flag('--gpu-memory-utilization'))==args.gpu_memory_utilization
    spec=json.loads(flag('--speculative-config'));assert spec['method']=='mtp' and spec['num_speculative_tokens']==args.expected_mtp
    assert spec.get('num_speculative_tokens_per_batch_size') is None
    assert json.loads(flag('--structured-outputs-config'))['backend']=='xgrammar'
    for enabled in ('--enable-prefix-caching','--enable-auto-tool-choice'):
        assert enabled in command and '--no-'+enabled[2:] not in command
    assert ident['runtime_mount']['RW']
    guard=data['guard'];parts=shlex.split(guard['cmdline'])
    assert any(x.endswith('watch_spark_memory.py') for x in parts)
    assert parts[parts.index('--container')+1]==container
    threshold=float(parts[parts.index('--min-available-gib')+1]) if '--min-available-gib'in parts else 1.0
    assert threshold>=args.min_host_available_gib>=1
    assert guard['proc_state'] not in ('Z','X')
    start=datetime.datetime.fromisoformat(ident['started_at'].replace('Z','+00:00')).timestamp()
    assert guard['started_unix']>=start-2, 'guard predates this container instance'
    assert guard['last_sample'] and data['time']-guard['last_sample']['time']<30, 'guard log is stale'
    assert guard['last_sample']['low_samples']==0
    if guard['ready_marker_exists']:assert guard['ready_marker_value']==ident['id']
    assert data['physical_memory_bytes']['MemAvailable']>=args.min_host_available_gib*1024**3
    return ident


def inspect_hosts(args,mode='identity'):
    # Complete both calls and surface every host failure, not just the first.
    records={};errors={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending={h:pool.submit(remote,args,h,c,mode) for h,c in hosts(args)}
        for host,container in hosts(args):
            try:
                records[host]=pending[host].result();validate_host(args,records[host],container)
            except Exception as exc:errors[host]=repr(exc)
    if errors:raise RuntimeError(json.dumps({'remote_errors':errors,'observed':records}))
    return records


def snapshot(args,directory,label):
    records=inspect_hosts(args,'snapshot')
    for host,data in records.items():
        shared.save(directory/f'{host}-{label}.json',data)
    with urllib.request.urlopen(args.base_url+'/metrics',timeout=30) as response:
        (directory/f'metrics-{label}.txt').write_bytes(response.read())
    return records


def monitor(args,directory,stop,failed):
    with (directory/'physical-memory.jsonl').open('x') as f:
        while not stop.is_set():
            try:row={'time':time.time(),'hosts':inspect_hosts(args,'sample')}
            except Exception as exc:
                row={'time':time.time(),'error':repr(exc)};failed.set()
            f.write(json.dumps(row)+'\n');f.flush()
            if failed.is_set():return
            stop.wait(15)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head-host',default='rhea');p.add_argument('--worker-host',default='moa')
    p.add_argument('--head-container',default='dots3-vllm-head');p.add_argument('--worker-container',default='dots3-vllm-worker')
    p.add_argument('--remote-project',default='/home/tj/dots-note-work/recipe')
    p.add_argument('--expected-image-id',required=True)
    p.add_argument('--expected-mtp',required=True,type=int,choices=[1,2,3,4])
    p.add_argument('--max-model-len',required=True,type=int,choices=[262144,524288])
    p.add_argument('--gpu-memory-utilization',type=float,default=.80)
    p.add_argument('--min-host-available-gib',type=float,default=1)
    p.add_argument('--base-url',default='http://10.55.1.5:8000/v1')
    p.add_argument('--model',default='dots3-note-exl3-k4')
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--inherit-evidence',type=Path,
                   help='Explicit hash-bound prior evidence; inherited stages send no new requests')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args();args.base_url=args.base_url.rstrip('/').removesuffix('/v1')
    if args.min_host_available_gib<1:p.error('require physical headroom guard >=1 GiB')
    steps=plan(args)
    if not args.execute:
        print(json.dumps({'requests':sum(s['requests'] for s in steps),'hosts':hosts(args),'steps':steps},indent=2));return
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    shared.preflight_tool_quality(steps,out)
    first=inspect_hosts(args)
    sources=sorted((ROOT/'serving/benchmarks').glob('*.py'))+[Path(__file__),Path(shared.__file__),ROOT/'serving/capture_runtime.py',ROOT/'serving/watch_spark_memory.py',ROOT/'serving/qualify_prefix_xgrammar.py',ROOT/'serving/benchmarks/code-agent-prompt.txt']
    sources.append(Path(evidence_lineage.__file__))
    if args.inherit_evidence:
        assert args.inherit_evidence.resolve().is_relative_to(ROOT)
        sources.append(args.inherit_evidence.resolve())
    binding={'schema':'dots3-spark-release-qualification-v2','identity':{h:d['identity'] for h,d in first.items()},
             'repo_digests':{h:d['image_repo_digests'] for h,d in first.items()},'plan':steps,
             'limits':{'max_model_len':args.max_model_len,'gpu_memory_utilization':args.gpu_memory_utilization,'min_host_available_gib':args.min_host_available_gib},
             'source_sha256':{str(path.relative_to(ROOT)):shared.digest(path) for path in sources}}
    manifest=out/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==binding, 'resume source/config/container/guard drift; use new output directory'
    else:shared.save(manifest,binding)
    attempt=out/f'attempt-{time.time_ns()}';attempt.mkdir();stop=threading.Event();failed=threading.Event()
    watcher=threading.Thread(target=monitor,args=(args,attempt,stop,failed),daemon=True);watcher.start()
    try:
        snapshot(args,attempt,'before')
        target_runtimes={h:json.loads((attempt/f'{h}-before.json').read_text())['runtime'] for h,_ in hosts(args)}
        for step in steps:
            current=inspect_hosts(args)
            assert {h:d['identity']for h,d in current.items()}==binding['identity'], 'container/guard changed'
            assert not failed.is_set(), 'physical memory/guard monitoring failed'
            stage=out/step['name'];stage.mkdir(exist_ok=True);receipt=stage/'receipt.json'
            if receipt.exists():
                saved=json.loads(receipt.read_text());artifact=stage/saved['artifact']
                assert shared.digest(artifact)==saved['sha256'];validate(step,artifact,args.max_model_len)
                if step.get('evidence_mode')=='inherited':
                    evidence_lineage.validate_inherited(step,artifact,binding['identity'])
                print(json.dumps({'stage':step['name'],'status':'validated-resume-skip'}),flush=True);continue
            run=stage/f'attempt-{time.time_ns()}';run.mkdir()
            if step.get('evidence_mode')=='inherited':
                artifact=evidence_lineage.materialize(step,run,binding['identity'],target_runtimes)
                evidence=evidence_lineage.validate_inherited(step,artifact,binding['identity'])
                shared.save(receipt,{'artifact':str(artifact.relative_to(stage)),'sha256':shared.digest(artifact),
                                    'validated':evidence,'identity':None,'target_identity':binding['identity'],
                                    'evidence_mode':'inherited','finished_unix':time.time()})
                print(json.dumps({'stage':step['name'],'status':'inherited-no-requests',
                                  'source_kind':step['inherited_evidence']['source_kind'],
                                  'sample_count':step['inherited_evidence']['sample_count']}),flush=True)
                continue
            artifact,command=shared.stage_command(step,run)
            shared.save(run/'command.json',command);print(json.dumps({'stage':step['name'],'command':command}),flush=True)
            with (run/'stdout.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:
                        if failed.wait(2):raise RuntimeError('physical memory/guard monitor failed; cancelling benchmark client')
                    if child.returncode:raise subprocess.CalledProcessError(child.returncode,command)
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
            evidence=validate(step,artifact,args.max_model_len)
            current=inspect_hosts(args)
            assert {h:d['identity']for h,d in current.items()}==binding['identity'], 'container/guard changed'
            shared.save(receipt,{'artifact':str(artifact.relative_to(stage)),'sha256':shared.digest(artifact),'validated':evidence,'identity':binding['identity'],'finished_unix':time.time()})
    finally:
        stop.set();watcher.join(timeout=110)
        snapshot(args,attempt,'after')
    assert not failed.is_set(), 'monitor failed'
    shared.save(attempt/'complete.json',{'completed':True,'requests_in_full_plan':sum(s['requests']for s in steps),
                'inherited_stages':[s['name']for s in steps if s.get('evidence_mode')=='inherited'],
                'executed_stages':[s['name']for s in steps if s.get('evidence_mode')!='inherited'],
                'context_receipts':[str((out/s['name']/'receipt.json').relative_to(out))for s in steps if s['name'].startswith('context-')],
                'prefill_method':'Exact unique prompt length / TTFT from context rows; includes first-token handoff.',
                'quality_scope':'Seven/coding static misses remain measurements, not executed-code correctness.'})


if __name__=='__main__':main()
