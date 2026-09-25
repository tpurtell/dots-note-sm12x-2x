#!/usr/bin/env python3
"""Audited pending-only Spark continuation with a larger client read timeout.

Frozen benchmark files remain unchanged. The child loads the original retrieval
module, then replaces its workloads module with the same source whose single
urlopen timeout literal is raised. No prompt, timing or validation code changes.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import types

import qualify_spark as q

ROOT=q.ROOT
LEDGER='timeout-continuation.json'
OLD='urllib.request.urlopen(req, timeout=900)'
NEW='urllib.request.urlopen(req, timeout=3600)'
WORKLOAD=ROOT/'serving/benchmarks/workloads.py'


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def transformed():
    raw=WORKLOAD.read_text();assert raw.count(OLD)==1
    return raw.replace(OLD,NEW)


def verify(out,manifest):
    ledger=read(out/LEDGER)
    assert ledger['schema']=='dots3-spark-timeout-continuation-v1'
    assert ledger['identity']==manifest['identity']
    assert ledger['timeout_seconds']=={'original':900,'continued':3600}
    assert ledger['effective_workloads_sha256']==hashlib.sha256(transformed().encode()).hexdigest()
    for name,expected in ledger['sources'].items():assert digest(ROOT/name)==expected, f'source drift: {name}'
    for name,expected in manifest['source_sha256'].items():assert digest(ROOT/name)==expected, f'frozen source drift: {name}'
    for name,expected in ledger['preserved_files'].items():
        p=(out/name).resolve();assert p.is_relative_to(out.resolve());assert digest(p)==expected, f'prior evidence changed: {name}'
    assert ledger['remaining_stages']==['retrieval-522144']
    assert ledger['preserved_stages']==[s['name']for s in manifest['plan']if s['name']!='retrieval-522144']
    return ledger


def prepare(out,manifest,evidence):
    preserved={};names=[]
    for step in manifest['plan']:
        receipt=out/step['name']/'receipt.json'
        if step['name']=='retrieval-522144':assert not receipt.exists();continue
        r=read(receipt);assert r['validated']['complete']
        assert (r.get('target_identity') if step.get('evidence_mode')=='inherited' else r['identity'])==manifest['identity']
        a=receipt.parent/r['artifact'];assert digest(a)==r['sha256'];q.validate(step,a,524288)
        names.append(step['name'])
    # Preserve every pre-continuation artifact, including the failed attempt and
    # monitor/snapshot logs. There is no active qualifier when this is prepared.
    for p in out.rglob('*'):
        if p.is_file() and p.name!=LEDGER:preserved[str(p.relative_to(out))]=digest(p)
    for artifact in (out/'retrieval-522144').glob('attempt-*/result.jsonl'):
        rows=[json.loads(line) for line in artifact.read_text().splitlines() if line.strip()]
        assert all(row.get('record')=='meta' for row in rows), 'refuse to repeat any completed retrieval measurement'
    failure=read(evidence);assert failure['completed_retrieval_measurements']==0
    for filename,d in failure['files'].items():assert digest(ROOT/filename)==d['sha256']
    return {'schema':'dots3-spark-timeout-continuation-v1','identity':manifest['identity'],
            'timeout_seconds':{'original':900,'continued':3600},'source_delta':{'original':OLD,'effective':NEW},
            'effective_workloads_sha256':hashlib.sha256(transformed().encode()).hexdigest(),
            'sources':{str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))},
            'preserved_files':preserved,'preserved_stages':names,'remaining_stages':['retrieval-522144'],
            'failure_evidence':failure,'failure_evidence_sha256':digest(evidence),
            'scope':'Same containers, image, profile, prompts, timing and validators; only socket read timeout changes. Prior stages are not repeated.'}


def child(argv):
    spec=importlib.util.spec_from_file_location('retrieval_timeout_continuation',ROOT/'serving/benchmarks/retrieval.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    changed=types.ModuleType('workloads_timeout_continuation');changed.__file__=str(WORKLOAD)
    exec(compile(transformed(),str(WORKLOAD),'exec'),changed.__dict__)
    mod.workloads=changed;sys.argv=[str(ROOT/'serving/benchmarks/retrieval.py'),*argv];mod.main()


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--child':child(sys.argv[2:]);return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--failure-evidence',type=Path,required=True)
    p.add_argument('--expected-image-id',required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args();out=a.output_dir.resolve();m=read(out/'manifest.json')
    assert m['schema']=='dots3-spark-release-qualification-v2'
    assert all(i['image_id']==a.expected_image_id for i in m['identity'].values())
    assert m['limits']['max_model_len']==524288
    if not (out/LEDGER).exists():ledger=prepare(out,m,a.failure_evidence)
    else:ledger=verify(out,m)
    if not a.execute:
        print(json.dumps({'preserve':ledger['preserved_stages'],'run':ledger['remaining_stages'],'timeout_seconds':ledger['timeout_seconds']},indent=2));return
    retrieval=next(s for s in m['plan'] if s['name']=='retrieval-522144')
    a.base_url=retrieval['options'][retrieval['options'].index('--base-url')+1].rstrip('/').removesuffix('/v1')
    a.head_host='rhea';a.worker_host='moa';a.head_container='dots3-vllm-head';a.worker_container='dots3-vllm-worker'
    a.remote_project='/home/tj/dots-note-work/recipe';a.expected_mtp=3;a.max_model_len=524288
    a.gpu_memory_utilization=m['limits']['gpu_memory_utilization'];a.min_host_available_gib=m['limits']['min_host_available_gib']
    def identity():return {h:d['identity']for h,d in q.inspect_hosts(a).items()}
    assert identity()==m['identity'],'container/guard changed'
    if not (out/LEDGER).exists():q.shared.save(out/LEDGER,ledger)
    verify(out,m)
    attempt=out/f'attempt-{time.time_ns()}';attempt.mkdir();stop=threading.Event();failed=threading.Event()
    watcher=threading.Thread(target=q.monitor,args=(a,attempt,stop,failed),daemon=True);watcher.start()
    try:
        q.snapshot(a,attempt,'before')
        step=next(s for s in m['plan']if s['name']=='retrieval-522144');stage=out/step['name']
        assert not (stage/'receipt.json').exists(),'completed retrieval must not be repeated'
        run=stage/f'attempt-{time.time_ns()}';run.mkdir();artifact,original=q.shared.stage_command(step,run)
        command=[original[0],str(Path(__file__).resolve()),'--child',*original[2:]]
        q.shared.save(run/'command.json',command);q.shared.save(run/'original-command.json',original)
        with (run/'stdout.log').open('x') as log:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    if failed.wait(2):raise RuntimeError('physical memory/guard monitor failed')
                if proc.returncode:raise subprocess.CalledProcessError(proc.returncode,command)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
        validated=q.validate(step,artifact,524288);assert identity()==m['identity'];verify(out,m)
        q.shared.save(stage/'receipt.json',{'artifact':str(artifact.relative_to(stage)),'sha256':digest(artifact),'validated':validated,'identity':m['identity'],'finished_unix':time.time(),'timeout_continuation':{'artifact':LEDGER,'sha256':digest(out/LEDGER)}})
    finally:
        stop.set();watcher.join(timeout=110);q.snapshot(a,attempt,'after')
    assert not failed.is_set();assert identity()==m['identity'];verify(out,m)
    q.shared.save(attempt/'complete.json',{'completed':True,'requests_in_full_plan':sum(s['requests']for s in m['plan']),
        'inherited_stages':[s['name']for s in m['plan']if s.get('evidence_mode')=='inherited'],
        'executed_stages':[s['name']for s in m['plan']if s.get('evidence_mode')!='inherited'],
        'timeout_continuation':{'artifact':LEDGER,'sha256':digest(out/LEDGER)},'uninterrupted_run':False,
        'preserved_stages':ledger['preserved_stages'],'continued_stages':ledger['remaining_stages']})


if __name__=='__main__':main()
