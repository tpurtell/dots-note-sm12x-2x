#!/usr/bin/env python3
"""Compare named coding JSONL candidates with explicit match/completeness gates.

Example: --candidate mtp2=path2.jsonl --candidate mtp3=path3.jsonl --output comparison.json
Only measured requests are summarized. Incomplete runs remain visible but are
not ranked or compared as completed runs. No code-quality claim is inferred.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

from coding_clients import reasoning_status


def median(values):
    values=[v for v in values if isinstance(v,(int,float))]
    return statistics.median(values) if values else None


def key_for(wave,row):
    return (int(wave['concurrency']),int(wave['run']),row['task'])


def load_candidate(name,path):
    records=[];parse_issues=[]
    data=path.read_bytes();lines=data.splitlines()
    for i,line in enumerate(lines,1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except (ValueError,UnicodeDecodeError) as exc:
            parse_issues.append({'line':i,'error':str(exc),'possibly_partial_last_line':i==len(lines)})
    metas=[r for r in records if r.get('record')=='meta']
    if len(metas)!=1:
        raise ValueError(f'{name}: expected exactly one metadata record')
    meta=metas[0];args=meta.get('args',{});rows={};duplicates=[];waves=[]
    for record in records:
        if record.get('record')!='wave' or record.get('run',-1)<0 or not record.get('timed',True):
            continue
        waves.append(record)
        for row in record.get('request_results',[]):
            key=key_for(record,row)
            if key in rows:
                duplicates.append(list(key))
            rows[key]=row
    # Four tasks are part of v1's declared suite. Recover their identifiers from
    # any wave (including warmup) rather than importing today's prompt texts.
    tasks=sorted({r['task'] for w in records if w.get('record')=='wave' for r in w.get('request_results',[])})
    expected_count=len(args.get('concurrency',[]))*args.get('runs',0)*4
    expected_keys={(int(c),run,task) for c in args.get('concurrency',[]) for run in range(args.get('runs',0)) for task in tasks}
    missing=sorted(expected_keys-set(rows));unexpected=sorted(set(rows)-expected_keys)
    terminal=any(r.get('record')=='summary' for r in records)
    complete=bool(terminal and expected_count>0 and len(tasks)==4 and len(rows)==expected_count
                  and not missing and not unexpected and not duplicates and not parse_issues)
    return {'name':name,'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),
        'meta':meta,'rows':rows,'waves':waves,'complete':complete,
        'completeness':{'terminal_summary':terminal,'expected_requests':expected_count,
            'observed_requests':len(rows),'observed_task_ids':tasks,'missing_keys':[list(k) for k in missing],
            'unexpected_keys':[list(k) for k in unexpected],'duplicate_keys':duplicates,'parse_issues':parse_issues}}


def row_summary(rows):
    total=len(rows)
    finished=[r for r in rows if r.get('completed',r.get('finish_reason')=='stop') and r.get('error') is None]
    valid=[r for r in rows if r.get('token_accounting_valid',False) and r.get('error') is None]
    statuses=[r.get('reasoning_status') or reasoning_status(r) for r in rows]
    passing=sum(bool(r.get('content_result',{}).get('static_checks_passed',False)) for r in rows)
    count=lambda predicate: sum(bool(predicate(r)) for r in rows)
    natural=count(lambda r:r.get('finish_reason')=='stop')
    truncated=count(lambda r:r.get('truncated',r.get('finish_reason')=='length'))
    return {'requests':total,'completed':len(finished),'natural_stop':natural,'truncated':truncated,
        'errors':count(lambda r:r.get('error') is not None),'static_checks_passed':passing,
        'natural_stop_rate':natural/total if total else None,'truncation_rate':truncated/total if total else None,
        'static_check_rate':passing/total if total else None,'valid_token_accounting':len(valid),
        'median_decode_tps':median([r.get('decode_tps') for r in valid]),
        'median_completed_latency_seconds':median([r.get('completion_latency_seconds') for r in finished]),
        'median_all_terminal_latency_seconds':median([r.get('completion_latency_seconds') for r in rows]),
        'median_output_tokens':median([r.get('stream_token_count') for r in rows]),
        'median_completed_output_tokens':median([r.get('stream_token_count') for r in finished]),
        'median_ttft_seconds':median([r.get('ttft_seconds') for r in valid]),
        'reasoning_finished':sum(s['finished'] is True for s in statuses),
        'reasoning_incomplete':sum(s['finished'] is False for s in statuses),
        'reasoning_unobserved':sum(s['finished'] is None for s in statuses),
        'median_reasoning_duration_seconds_approx':median([s.get('duration_seconds_approx') for s in statuses]),
        'quality_scope':'Static artifacts/reference answers only; generated code was not executed.'}


def summarize(candidate):
    grouped=defaultdict(list);task_groups=defaultdict(list)
    for (c,run,task),row in candidate['rows'].items():
        grouped[c].append(row);task_groups[c,task].append(row)
    by_c={}
    for c,rows in sorted(grouped.items()):
        summary=row_summary(rows)
        waves=[w for w in candidate['waves'] if w['concurrency']==c]
        summary['wave_peak_overlap']=[w.get('summary',{}).get('peak_overlapping_stream_intervals') for w in waves]
        if waves and all(w.get('summary',{}).get('all_token_accounting_valid',False) for w in waves):
            wall=sum(w['summary']['wall_seconds'] for w in waves)
            decode=sum(w['summary']['decode_window_seconds'] for w in waves)
            summary['aggregate_output_tps']=sum(r['stream_token_count'] for r in rows)/wall if wall else None
            summary['aggregate_decode_tps']=sum(r['decode_tokens'] for r in rows)/decode if decode else None
        else:
            summary['aggregate_output_tps']=summary['aggregate_decode_tps']=None
        summary['by_task']={task:row_summary(values) for (cc,task),values in sorted(task_groups.items()) if cc==c}
        by_c[str(c)]=summary
    return {k:candidate[k] for k in ('name','path','sha256','complete','completeness')}|{'by_concurrency':by_c}


def compare(base,other):
    left,right=base['rows'],other['rows'];shared=sorted(set(left)&set(right))
    mismatches=[]
    for key in shared:
        # Compare the full API payload: prompts, reasoning flags, sampling,
        # seeds and generation budget, not merely task names or file labels.
        if (not isinstance(left[key].get('request_payload'),dict)
                or not isinstance(right[key].get('request_payload'),dict)
                or left[key]['request_payload'] != right[key]['request_payload']):
            mismatches.append(list(key))
    keys_equal=set(left)==set(right)
    task_hash_equal=base['meta'].get('task_sha256')==other['meta'].get('task_sha256')
    eligible=base['complete'] and other['complete'] and keys_equal and not mismatches and task_hash_equal
    result={'baseline':base['name'],'candidate':other['name'],'eligible_for_complete_run_comparison':eligible,
        'common_requests':len(shared),'request_key_sets_equal':keys_equal,'task_hash_equal':task_hash_equal,
        'payload_mismatch_keys':mismatches,'only_baseline_keys':[list(k) for k in sorted(set(left)-set(right))],
        'only_candidate_keys':[list(k) for k in sorted(set(right)-set(left))],
        'interpretation':'Decode rate and completed latency are separate. Shorter output can lower latency without improving decode. No automatic quality ranking.'}
    if not eligible:
        result['comparison_withheld']='One or both runs incomplete, request sets differ, or task/payload contracts differ.'
        return result
    ratios=defaultdict(lambda:defaultdict(list))
    for key in shared:
        a,b=left[key],right[key];c,_,task=key
        metrics={}
        if a.get('token_accounting_valid') and b.get('token_accounting_valid') and not a.get('error') and not b.get('error'):
            metrics['candidate_over_baseline_decode_tps']=(b.get('decode_tps'),a.get('decode_tps'))
        if a.get('completed') and b.get('completed'):
            metrics['candidate_over_baseline_completed_latency']=(b.get('completion_latency_seconds'),a.get('completion_latency_seconds'))
            metrics['candidate_over_baseline_completed_output_tokens']=(b.get('stream_token_count'),a.get('stream_token_count'))
        for name,(num,den) in metrics.items():
            if num is not None and den and den>0:
                ratios[str(c)][name].append(num/den)
                ratios[f'{c}:{task}'][name].append(num/den)
    result['paired_median_ratios']={group:{name:{'median':median(values),'pairs':len(values)} for name,values in metrics.items()} for group,metrics in ratios.items()}
    result['ratio_direction']='Candidate / baseline. Higher decode ratio is faster; lower completed-latency ratio is shorter. Read output-token ratio alongside latency.'
    return result


def human(result):
    lines=['Coding/reasoning comparison (static checks are not executed-code correctness)',
           'candidate C complete n stop/trunc static decode_tps completed_s tokens ttft_s reasoning_done/open']
    fmt=lambda v:'—' if v is None else f'{v:.2f}'
    for candidate in result['candidates']:
        if not candidate['by_concurrency']:
            lines.append(f"{candidate['name']}: complete={candidate['complete']}, no measured requests yet")
        for c,s in candidate['by_concurrency'].items():
            lines.append(f"{candidate['name']} C{c} {candidate['complete']} {s['requests']} "
                f"{s['natural_stop']}/{s['truncated']} {s['static_checks_passed']}/{s['requests']} "
                f"{fmt(s['median_decode_tps'])} {fmt(s['median_completed_latency_seconds'])} "
                f"{fmt(s['median_output_tokens'])} {fmt(s['median_ttft_seconds'])} "
                f"{s['reasoning_finished']}/{s['reasoning_incomplete']}")
    for comparison in result['comparisons']:
        if not comparison['eligible_for_complete_run_comparison']:
            lines.append(f"{comparison['candidate']} vs {comparison['baseline']}: comparison withheld (incomplete or unmatched); see JSON gates.")
        else:
            lines.append(f"{comparison['candidate']} vs {comparison['baseline']}: complete matched runs; paired ratios in JSON, no automatic ranking.")
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',action='append',required=True,metavar='NAME=JSONL')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();candidates=[];seen=set()
    for item in args.candidate:
        name,sep,path=item.partition('=')
        if not sep or not name or not path or name in seen:
            p.error('Each candidate must be a unique NAME=JSONL')
        seen.add(name);candidates.append(load_candidate(name,Path(path)))
    result={'schema':'dots3-coding-comparison-v1','candidates':[summarize(c) for c in candidates],
            'comparisons':[compare(candidates[0],c) for c in candidates[1:]],
            'baseline':candidates[0]['name'],'ranking':None}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(human(result))
    print(f'JSON: {args.output}')


if __name__=='__main__':
    main()
