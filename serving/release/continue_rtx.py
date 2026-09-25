#!/usr/bin/env python3
"""Explicit audited continuation after restart; never reruns finished RTX stages.

Keeps the original manifest and receipts byte-identical. Requires unchanged
qualified image/model/profile/workload sources; records the lifecycle change.
Plan-only by default. Does not start, stop, reset, or reconfigure any GPU.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time

import qualify_rtx as q
import restart_continuation as audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--container',required=True)
    p.add_argument('--expected-image-id',required=True)
    p.add_argument('--expected-mtp',required=True,type=int,choices=[1,2,3,4])
    p.add_argument('--max-model-len',type=int,choices=[262144,524288],default=524288)
    p.add_argument('--base-url',default='http://127.0.0.1:8001')
    p.add_argument('--model',default='dots3-note-exl3-k4')
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--interruption-reason',required=True)
    p.add_argument('--failure-evidence',required=True,type=Path)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args();args.base_url=args.base_url.rstrip('/').removesuffix('/v1')
    out=args.output_dir.resolve();manifest=audit.read(out/'manifest.json')
    assert manifest['schema']=='dots3-rtx-release-qualification-v2'
    assert manifest['base_url']==args.base_url and manifest['model']==args.model
    assert manifest['limits']['max_model_len']==args.max_model_len
    assert manifest['plan']==q.plan(args), 'stage definitions changed'
    audit.verify_sources(manifest)
    pending=[s for s in manifest['plan'] if not (out/s['name']/'receipt.json').exists()]
    if not args.execute:
        print(json.dumps({'interrupted':True,'will_run':[s['name'] for s in pending],
                          'will_preserve':[s['name'] for s in manifest['plan'] if s not in pending]},indent=2))
        return
    current=q.identity(args)
    ledger_path=out/audit.LEDGER
    if not ledger_path.exists():
        ledger=audit.prepare(out,manifest,current,q,args.interruption_reason,args.failure_evidence)
        q.save(ledger_path,ledger)
    ledger=audit.verify(out,manifest,current)
    attempt=out/f'attempt-{time.time_ns()}';attempt.mkdir()
    stop=threading.Event();monitor=threading.Thread(target=q.memory_monitor,args=(attempt,stop),daemon=True);monitor.start()
    try:
        q.snapshot(args,attempt,'before')
        for step in manifest['plan']:
            assert q.identity(args)==current, 'runtime changed during continuation'
            audit.verify(out,manifest,current)
            stage=out/step['name'];stage.mkdir(exist_ok=True);receipt=stage/'receipt.json'
            if receipt.exists():
                saved=audit.read(receipt);artifact=audit.contained(stage,saved['artifact'])
                assert saved['identity']==audit.stage_identity(ledger,step['name'])
                assert q.digest(artifact)==saved['sha256']
                q.validate(step,artifact,args.max_model_len)
                print(json.dumps({'stage':step['name'],'status':'validated-preserved-skip'}),flush=True)
                continue
            assert step['name'] in ledger['remaining_stages'], 'missing preserved stage'
            run_dir=stage/f'attempt-{time.time_ns()}';run_dir.mkdir()
            artifact,command=q.stage_command(step,run_dir);q.save(run_dir/'command.json',command)
            print(json.dumps({'stage':step['name'],'status':'running-after-restart','command':command}),flush=True)
            with (run_dir/'stdout.log').open('x') as log:
                subprocess.run(command,cwd=q.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            evidence=q.validate(step,artifact,args.max_model_len)
            assert q.identity(args)==current, 'runtime changed during stage'
            q.save(receipt,{'artifact':str(artifact.relative_to(stage)),'sha256':q.digest(artifact),
                           'validated':evidence,'identity':current,'finished_unix':time.time()})
        audit.verify(out,manifest,current)
        curves=[]
        for depth in q.context_depths(args.max_model_len):
            stage=out/f'context-{depth}';receipt=audit.read(stage/'receipt.json')
            rows=[json.loads(line) for line in (stage/receipt['artifact']).read_text().splitlines()]
            rows=[r for r in rows if r.get('record')=='measurement' and r['timed']]
            curves.append({'prompt_tokens':depth,'ttft_seconds_median':statistics.median(r['ttft_seconds'] for r in rows),
                           'effective_prefill_tps_median':statistics.median(depth/r['ttft_seconds'] for r in rows),
                           'decode_tps_median':statistics.median(r['decode_tps'] for r in rows),
                           'raw_receipt':str((stage/'receipt.json').relative_to(out))})
        completion={'completed':True,'requests_in_full_plan':sum(s['requests'] for s in manifest['plan']),
                    'curves':curves,'restart_continuation':{'artifact':audit.LEDGER,'sha256':q.digest(ledger_path)},
                    'uninterrupted_run':False,'preserved_stages':ledger['preserved_stages'],
                    'continued_stages':ledger['remaining_stages'],
                    'prefill_method':'Exact unique context prompt tokens / client TTFT; includes first-token handoff.',
                    'quality_scope':'Pre-restart and post-restart stages share image/profile/sources; lifecycle interruption explicitly preserved. Quality misses retained.'}
    finally:
        try:q.snapshot(args,attempt,'after')
        finally:stop.set();monitor.join(timeout=20)
    q.save(attempt/'complete.json',completion)


if __name__=='__main__':main()
