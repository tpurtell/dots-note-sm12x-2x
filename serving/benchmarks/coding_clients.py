#!/usr/bin/env python3
"""Matched C1/C2/C4 coding/reasoning streams with natural termination.

Content checks are static response checks, not execution-based code correctness.
Each concurrency processes the same four tasks per run in synchronized waves.
"""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import re
import statistics
import threading
import time
from urllib.parse import urlparse

TASKS = (
    ('merge_intervals', '''You are fixing a production Python utility. The input is an unsorted list of integer [start,end] pairs with start<=end. Closed intervals that touch must merge. The caller's list and inner lists must remain unchanged. Current code:
```python
def merge_intervals(intervals):
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start < merged[-1][1]:
            merged[-1][1] = end
        else:
            merged.append([start,end])
    return merged
```
Diagnose every bug and provide a corrected merge_intervals in one Python code block, plus focused assertion tests covering empty input, nesting, touching, negative endpoints, unsorted input and no mutation. Explain the complexity and why your merge invariant is sufficient. Keep the final explanation concise.''', 'merge_intervals'),
    ('async_pool', '''Review this Python3.12 async service helper. It is intended to bound simultaneously running fetch calls while preserving input order, promptly propagate failure, cancel remaining work on failure or caller cancellation, and await cancellation so no orphan tasks remain:
```python
async def fetch_all(items, fetch, limit):
    sem = asyncio.Semaphore(limit)
    async def run(item):
        async with sem:
            return await fetch(item)
    tasks = [asyncio.create_task(run(item)) for item in items]
    return await asyncio.gather(*tasks)
```
Provide corrected fetch_all in one Python code block, using the standard library only. Reject nonpositive limits, keep empty input valid, and do not swallow asyncio.CancelledError. Explain the original failure behavior and include focused tests or test sketches for concurrency limit, ordering, failure and cancellation. Mention whether task creation itself is bounded. Keep the final explanation concise.''', 'fetch_all'),
    ('retry_boundary', '''Debug this Python retry helper used around an idempotent RPC. max_attempts counts the initial attempt; only TransientError is retryable; sleep uses exponential delays base_delay*2**retry_index after failures that will be retried. Never sleep after the last failure. Preserve the last TransientError traceback; unrelated errors propagate immediately. Reject max_attempts<1 and base_delay<0.
```python
def retry(op, max_attempts, base_delay, sleep):
    for i in range(max_attempts):
        try:
            return op()
        except Exception:
            sleep(base_delay * 2**i)
    return None
```
Provide corrected retry in one Python code block, assuming TransientError is already defined. Include tests/sketches for eventual success, exhaustion, nontransient failure, one attempt and invalid arguments. In a final JSON code block, report the sleep delays when base_delay=0.25,max_attempts=4 and every attempt fails transiently, with key delays. Explain the boundary reasoning concisely.''', 'retry'),
    ('reservation_race', '''You are reviewing an inventory reservation API backed by PostgreSQL. READ COMMITTED transactions currently SELECT stock then UPDATE stock=old_stock-quantity; the idempotency key is checked separately before the transaction. Concurrent retries can oversell or reserve twice. Design one transaction that atomically claims a unique request_id, checks positive quantity, conditionally decrements sufficient inventory, and records the durable outcome so a retry after a lost response returns the same result. Explain same-key/different-payload conflict handling, rollback versus recording a rejected request, and why ON CONFLICT does not alone solve the inventory race. Give SQL or pseudocode plus focused concurrent test scenarios. For two DISTINCT request IDs each reserving4 from initial stock5, with successful requests committed and unsuccessful inventory updates making no change, give a final JSON code block with keys successful_reservations and final_stock. Keep the final explanation concise.''', None),
)


def content_check(task, content, finish):
    name,prompt,function = task
    # A configured reasoning parser may split reasoning into a separate field;
    # without it the template's think tags remain in content.
    final = content.rsplit('</think>',1)[-1]
    issues = []
    if finish != 'stop':
        issues.append('response did not finish naturally')
    blocks = re.findall(r'```(?:python|py)\s*\n(.*?)```',final,re.S|re.I)
    parsed = []
    for block in blocks:
        try:
            parsed.append(ast.parse(block))
        except SyntaxError:
            pass
    if function and not any(any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==function for n in ast.walk(t)) for t in parsed):
        issues.append(f'no syntactically valid Python block defining {function}')
    expected = {'retry_boundary': {'delays':[.25,.5,1.0]},
                'reservation_race': {'successful_reservations':1,'final_stock':1}}.get(name)
    json_answers = []
    for block in re.findall(r'```json\s*\n(.*?)```',final,re.S|re.I):
        try:
            json_answers.append(json.loads(block))
        except json.JSONDecodeError:
            pass
    if expected and not any(isinstance(a,dict) and all(a.get(k)==v for k,v in expected.items()) for a in json_answers):
        issues.append('final JSON reference answer missing or incorrect')
    if function and not re.search(r'\b(test|assert|pytest|unittest)\b',final,re.I):
        issues.append('no test evidence in final answer')
    return {'static_checks_passed':not issues,'issues':issues,
            'code_executed':False,'scope':'response syntax/required artifacts and bounded reference answers; not behavioral code correctness'}



def reasoning_status(row):
    """Infer a visible reasoning boundary; unknown stays unknown.

    Times are SSE observations, so a burst may contain the end marker and the
    beginning of the answer. No tokenizer or hidden server timing is inferred.
    This helper also accepts saved v1 rows for postprocessing older runs.
    """
    content=''; saw_reasoning=False; boundary=None; token_count=0; boundary_tokens=None
    for item in row.get('sse_events',[]):
        for choice in item['event'].get('choices',[]):
            delta=choice.get('delta',{})
            token_count+=len(choice.get('token_ids') or [])
            piece=delta.get('content') or ''
            separate=delta.get('reasoning_content') or delta.get('reasoning') or ''
            content+=piece
            saw_reasoning = saw_reasoning or bool(separate) or '<think>' in content or '</think>' in content
            closed='</think>' in content
            split_answer=bool(piece and saw_reasoning and '<think>' not in content)
            if boundary is None and (closed or split_answer):
                boundary=item['seconds']; boundary_tokens=token_count
    finished=True if boundary is not None else (False if saw_reasoning else None)
    ttft=row.get('ttft_seconds')
    return {'finished':finished,'observed':saw_reasoning,
            'boundary_seconds_approx':boundary,
            'duration_seconds_approx':max(0,boundary-ttft) if boundary is not None and ttft is not None else None,
            'tokens_through_boundary_burst':boundary_tokens,
            'measurement':'SSE boundary observation; may include final-answer tokens in same burst'}


def request(base,model,task,run,index,concurrency,args,barrier):
    tag=hashlib.sha256(f'{args.seed}:{concurrency}:{run}:{task[0]}'.encode()).hexdigest()[:16]
    payload={'model':model,'messages':[{'role':'user','content':f'Request identifier {tag}; ignore it.\n'+task[1]}],
        'chat_template_kwargs':{'enable_thinking':True},'temperature':args.temperature,
        'seed':args.seed+run*17+index,'max_tokens':args.output_tokens,'n':1,
        'stream':True,'stream_options':{'include_usage':True},'return_token_ids':True}
    parsed=urlparse(base)
    cls=http.client.HTTPSConnection if parsed.scheme=='https' else http.client.HTTPConnection
    connection=cls(parsed.hostname,parsed.port,timeout=args.timeout)
    prefix=parsed.path.rstrip('/')
    endpoint=prefix+'/chat/completions' if prefix.endswith('/v1') else prefix+'/v1/chat/completions'
    barrier.wait()
    started=time.perf_counter()
    row={'task':task[0],'request_payload':payload,'started_perf_seconds':started,
         'raw_sse_lines':[],'sse_events':[],'chunks':[],'content':'','reasoning':'','usage':None,
         'finish_reason':None,'error':None,'saw_done':False}
    try:
        connection.request('POST',endpoint,body=json.dumps(payload),headers={'Content-Type':'application/json'})
        response=connection.getresponse()
        row['http_status']=response.status
        if response.status!=200:
            row['response_body']=response.read().decode(errors='replace')
            raise RuntimeError(f'HTTP {response.status}')
        while True:
            line=response.readline()
            if not line:
                break
            elapsed=time.perf_counter()-started
            decoded=line.decode(errors='replace')
            row['raw_sse_lines'].append({'seconds':elapsed,'line':decoded})
            if not decoded.startswith('data:'):
                continue
            raw=decoded[5:].strip()
            if raw=='[DONE]':
                row['saw_done']=True
                break
            event=json.loads(raw)
            row['sse_events'].append({'seconds':elapsed,'event':event})
            if event.get('usage'):
                row['usage']=event['usage']
            for choice in event.get('choices',[]):
                delta=choice.get('delta',{})
                row['content']+=delta.get('content') or ''
                row['reasoning']+=delta.get('reasoning_content') or delta.get('reasoning') or ''
                if choice.get('token_ids'):
                    row['chunks'].append({'seconds':elapsed,'token_ids':choice['token_ids']})
                row['finish_reason']=choice.get('finish_reason') or row['finish_reason']
    except Exception as exc:
        row['error']=f'{type(exc).__name__}: {exc}'
    finally:
        connection.close()
    row['completion_latency_seconds']=time.perf_counter()-started
    chunks=row['chunks']; count=sum(len(c['token_ids']) for c in chunks)
    row['stream_token_count']=count
    row['truncated']=row['finish_reason']=='length'
    row['completed']=row['finish_reason']=='stop' and row['saw_done'] and row['error'] is None
    row['token_accounting_valid']=bool(chunks and row['usage'] and count==row['usage'].get('completion_tokens'))
    row['ttft_seconds']=chunks[0]['seconds'] if chunks else None
    row['decode_seconds']=chunks[-1]['seconds']-chunks[0]['seconds'] if chunks else None
    row['decode_tokens']=count-len(chunks[0]['token_ids']) if chunks else 0
    row['decode_tps']=row['decode_tokens']/row['decode_seconds'] if row['decode_seconds'] else None
    row['reasoning_visible']=bool(row['reasoning'] or '<think>' in row['content'] or '</think>' in row['content'])
    row['reasoning_status']=reasoning_status(row)
    row['content_result']=content_check(task,row['content'],row['finish_reason'])
    return row


def overlap(rows):
    events=[]
    for r in rows:
        if r['chunks']:
            start=r['started_perf_seconds']
            events.extend([(start+r['chunks'][0]['seconds'],1),(start+r['chunks'][-1]['seconds'],-1)])
    active=peak=0
    for _,delta in sorted(events,key=lambda v:(v[0],v[1])):
        active+=delta;peak=max(active,peak)
    return peak


def summarize_wave(rows):
    valid=[r for r in rows if r['token_accounting_valid'] and r['error'] is None]
    timed=[r for r in rows if r['chunks']]
    start=min(r['started_perf_seconds'] for r in rows)
    end=max(r['started_perf_seconds']+r['completion_latency_seconds'] for r in rows)
    decode_window=(max(r['started_perf_seconds']+r['chunks'][-1]['seconds'] for r in timed)-
        min(r['started_perf_seconds']+r['chunks'][0]['seconds'] for r in timed)) if timed else 0
    all_valid=len(valid)==len(rows)
    return {'requests':len(rows),'completed':sum(r['completed'] for r in rows),
        'truncated':sum(r['truncated'] for r in rows),'errors':sum(r['error'] is not None for r in rows),
        'static_checks_passed':sum(r['content_result']['static_checks_passed'] for r in rows),
        'all_token_accounting_valid':all_valid,'peak_overlapping_stream_intervals':overlap(rows),
        'wall_seconds':end-start,'decode_window_seconds':decode_window,
        'aggregate_decode_tps':sum(r['decode_tokens'] for r in valid)/decode_window if all_valid and decode_window>0 else None,
        'aggregate_output_tps':sum(r['stream_token_count'] for r in valid)/(end-start) if all_valid else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:8001/v1')
    p.add_argument('--model',default='dots3-note-exl3-k4')
    p.add_argument('--concurrency',type=int,nargs='+',default=[1,2,4])
    p.add_argument('--runs',type=int,default=3)
    p.add_argument('--warmup-runs',type=int,default=1)
    p.add_argument('--output-tokens',type=int,default=2048)
    p.add_argument('--temperature',type=float,default=.2)
    p.add_argument('--seed',type=int,default=20260925)
    p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if any(c not in (1,2,4) for c in args.concurrency) or args.runs<1 or args.warmup_runs<0 or args.output_tokens<1:
        p.error('C must be1,2,4; positive runs/tokens and nonnegative warmups required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    measured=[]
    with args.output.open('x') as f:
        def write(row):
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
        write({'record':'meta','schema':'dots3-coding-clients-v1','args':vars(args)|{'output':str(args.output)},
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'task_sha256':hashlib.sha256(json.dumps(TASKS).encode()).hexdigest(),
            'reasoning':'enable_thinking=true; cached model template defaults true and only emits no_think for false',
            'timing':'SSE token IDs include reasoning; exclude whole initial burst per request for decode; natural EOS, no min_tokens/ignore_eos',
            'scope':'standalone coding/reasoning requests, not autonomous tool-use agent evaluation'})
        for c in args.concurrency:
            for run in range(-args.warmup_runs,args.runs):
                for offset in range(0,len(TASKS),c):
                    barrier=threading.Barrier(c)
                    with ThreadPoolExecutor(max_workers=c) as pool:
                        futures=[pool.submit(request,args.base_url,args.model,TASKS[i],run,i,c,args,barrier) for i in range(offset,offset+c)]
                        rows=[future.result() for future in futures]
                    summary=summarize_wave(rows)
                    result={'record':'wave','concurrency':c,'run':run,'timed':run>=0,'task_offset':offset,
                            'summary':summary,'request_results':rows}
                    write(result)
                    print(json.dumps({k:v for k,v in result.items() if k!='request_results'}),flush=True)
                    if run>=0:
                        measured.append(result)
        summaries={}
        for c in args.concurrency:
            waves=[w for w in measured if w['concurrency']==c]
            rows=[r for w in waves for r in w['request_results']]
            def median(key):
                v=[r[key] for r in rows if r[key] is not None]
                return statistics.median(v) if v else None
            completed_latencies=[r['completion_latency_seconds'] for r in rows if r['completed']]
            reasoning_states=[r.get('reasoning_status') or reasoning_status(r) for r in rows]
            summaries[str(c)]={'requests':len(rows),'completed':sum(r['completed'] for r in rows),
                'truncated':sum(r['truncated'] for r in rows),'errors':sum(r['error'] is not None for r in rows),
                'static_checks_passed':sum(r['content_result']['static_checks_passed'] for r in rows),
                'median_ttft_seconds':median('ttft_seconds'),'median_completion_latency_seconds':median('completion_latency_seconds'),
                'median_completed_latency_seconds':statistics.median(completed_latencies) if completed_latencies else None,
                'reasoning_finished':sum(r['finished'] is True for r in reasoning_states),
                'reasoning_incomplete':sum(r['finished'] is False for r in reasoning_states),
                'reasoning_unobserved':sum(r['finished'] is None for r in reasoning_states),
                'median_per_request_decode_tps':median('decode_tps'),
                'wave_peak_overlap':[w['summary']['peak_overlapping_stream_intervals'] for w in waves],
                'aggregate_output_tps':sum(r['stream_token_count'] for r in rows)/sum(w['summary']['wall_seconds'] for w in waves)
                    if all(w['summary']['all_token_accounting_valid'] for w in waves) else None}
        write({'record':'summary','by_concurrency':summaries})
        print(json.dumps({'record':'summary','by_concurrency':summaries}),flush=True)


if __name__=='__main__':
    main()
