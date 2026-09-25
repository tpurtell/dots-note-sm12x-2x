"""Explicit prior-image evidence reuse; CPU-only and never claims final-image execution."""
from __future__ import annotations
import copy
import gzip
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
KINDS={'prefix','reasoning-api','multimodal','prefill','context'}


def sha(data):return hashlib.sha256(data).hexdigest()


def descriptor(path, root=ROOT):
    path=Path(path).resolve();assert path.is_relative_to(root.resolve())
    data=path.read_bytes();raw=gzip.decompress(data) if path.suffix=='.gz' else data
    return {'path':str(path.relative_to(root)),'sha256':sha(data),'raw_sha256':sha(raw)}


def blob(entry, root=ROOT):
    path=(root/entry['path']).resolve();assert path.is_relative_to(root.resolve())
    data=path.read_bytes();assert sha(data)==entry['sha256'], f"source archive changed: {entry['path']}"
    raw=gzip.decompress(data) if path.suffix=='.gz' else data
    assert sha(raw)==entry['raw_sha256'], 'raw source hash mismatch'
    return raw


def decode(raw,kind):
    text=raw.decode()
    return [json.loads(s) for s in text.splitlines() if s.strip()] if kind in {'reasoning-api','context'} else json.loads(text)


def profile(runtime):
    runtime=runtime.get('runtime',runtime)
    env=runtime['selected_environment']
    if isinstance(env,list):env=dict(s.split('=',1) for s in env)
    result={k:runtime[k] for k in ('image_id','args','started_at')}
    result['selected_environment']=env
    result['allocator_policy']=runtime.get('allocator_policy')
    return result


def revision(profile):
    values=[s.split('/snapshots/',1)[1].split('/',1)[0] for s in profile['args'] if '/snapshots/' in s]
    assert len(values)==1, 'require exact cached checkpoint revision'
    return values[0]


def context_limit(p):
    args=p['args'];i=args.index('--max-model-len')
    return int(args[i+1])


def source_profiles(entry,root=ROOT):
    assert set(entry['source_runtimes'])=={'rhea','moa'}, 'bind both source hosts'
    return {h:profile(json.loads(blob(d,root))) for h,d in entry['source_runtimes'].items()}


def validate_source(entry,root=ROOT):
    kind=entry['source_kind'];assert kind in KINDS
    assert entry['coverage'].strip(), 'explicit coverage limitations required'
    data=decode(blob(entry['artifact'],root),kind)
    profiles=source_profiles(entry,root)
    assert len({revision(p) for p in profiles.values()})==1
    for d in entry.get('supporting_evidence',[]):blob(d,root)
    if kind=='prefix':
        assert data['warm_prefix_hits']>0 and data['warm_prefix_queries']>0
        assert data['xgrammar_json']=={'answer':42}
        count=4;budget=None
    elif kind=='multimodal':
        rows=data['source_examples'];assert len(rows)==2 and all(r['contract_passed'] for r in rows)
        count=2;budget=None
    elif kind=='reasoning-api':
        assert data[0]['record']=='meta' and data[-1]['record']=='summary'
        rows=[r for r in data if r.get('record')=='case']
        # Earlier API versions use result records; identify their complete case tuple.
        if not rows:rows=[r for r in data if all(k in r for k in ('case','thinking','streaming','repeat','passed'))]
        s=data[-1];count=len(rows);budget=None
        assert count>0 and s['passed'] and s['cases_total']==s['cases_passed']==count
        assert all(r['passed'] and not r['issues'] for r in rows)
        keys={(r['case'],r['thinking'],r['streaming'],r['repeat']) for r in rows}
        assert len(keys)==count
        assert all({(r['thinking'],r['streaming']) for r in rows if r['case']==case and r['repeat']==repeat}=={(False,False),(False,True),(True,False),(True,True)} for case,_,_,repeat in keys)
        assert not s['evidence_issues'] and s['speculative_draft_token_delta']>0
    elif kind=='prefill':
        assert data['schema']=='dots3-prefill-depth.v1'
        points=[p for p in data['points'] if p['prompt_tokens']==entry['depth']];assert len(points)==1
        rows=points[0]['runs'];count=len(rows);budget=1
        assert count==data['runs_per_point'] and count>0
        assert len({r['prompt_sha256'] for r in rows})==count
        assert all(r['usage']['prompt_tokens']==entry['depth'] and r['usage']['completion_tokens']==1 and r['cached_tokens'] in (None,0) and r['ttft_seconds']>0 for r in rows)
    else:
        rows=[r for r in data if r.get('record')=='measurement' and r.get('timed',True)]
        count=len(rows);budget=entry['output_tokens_per_request'];assert count>0
        assert all(r['depth']==entry['depth'] and r['usage']['prompt_tokens']==entry['depth'] and r['usage']['completion_tokens']==budget and sum(len(c['token_ids']) for c in r['chunks'])==budget and r['cached_tokens'] in (None,0) and r['ttft_seconds']>0 and r['decode_tps']>0 for r in rows)
    if kind in {'prefill','context'}:
        assert all(entry['depth']+budget<=context_limit(p) for p in profiles.values()), 'source exceeds declared context'
    assert count==entry['sample_count'], 'source sample count mismatch'
    assert budget==entry['output_tokens_per_request'], 'source output budget mismatch'
    return data,profiles


def load_spec(path,root=ROOT):
    spec=json.loads(Path(path).read_text());assert spec['schema']=='dots3-prior-evidence-v1'
    assert spec['authorization']=='preserve-completed-tests-with-explicit-prior-profile-provenance'
    assert spec['stages']
    for name,entry in spec['stages'].items():
        kind=entry['source_kind']
        assert (name==kind and kind in {'prefix','reasoning-api','multimodal'}) or (kind in {'prefill','context'} and name==f"context-{entry['depth']}")
        validate_source(entry,root)
    return spec


def apply_plan(steps,path,root=ROOT):
    if path is None:return steps
    spec=load_spec(path,root);names={s['name'] for s in steps};assert set(spec['stages'])<=names
    result=copy.deepcopy(steps)
    for step in result:
        if entry:=spec['stages'].get(step['name']):
            step['requests']=0;step['format']='json';step['evidence_mode']='inherited'
            step['inherited_evidence']=entry
    return result


def differences(old,new):
    result={}
    for host in old:
        a,b=old[host],new[host];changes={}
        for key in ('image_id','args','started_at','allocator_policy'):
            if a[key]!=b[key]:changes[key]={'source':a[key],'target':b[key]}
        changes['environment']={k:{'source':a['selected_environment'].get(k),'target':b['selected_environment'].get(k)} for k in sorted(set(a['selected_environment'])|set(b['selected_environment'])) if a['selected_environment'].get(k)!=b['selected_environment'].get(k)}
        result[host]=changes
    return result


def materialize(step,directory,target_identity,target_runtimes,root=ROOT):
    entry=step['inherited_evidence'];_,old=validate_source(entry,root)
    targets={h:profile(r) for h,r in target_runtimes.items()}
    assert set(targets)==set(old)
    assert {revision(p) for p in old.values()}=={revision(p) for p in targets.values()}, 'checkpoint changed'
    if entry['source_kind'] in {'prefill','context'}:
        assert all(entry['depth']+entry['output_tokens_per_request']<=context_limit(p) for p in targets.values())
    assert all(targets[h]['image_id']==target_identity[h]['image_id'] and targets[h]['args']==target_identity[h]['args'] and targets[h]['started_at']==target_identity[h]['started_at'] for h in targets)
    entries=[entry['artifact'],*entry['source_runtimes'].values(),*entry.get('supporting_evidence',[])]
    archived={}
    for i,d in enumerate(entries):
        raw=blob(d,root);name=f'source-{i:03d}.raw';(directory/name).write_bytes(raw)
        packed=(root/d['path']).read_bytes();packedname=f'source-{i:03d}.original';(directory/packedname).write_bytes(packed)
        archived[name]={'sha256':sha(raw),'source':d,'original':packedname}
    provenance={'mode':'inherited','source_kind':entry['source_kind'],'sample_count':entry['sample_count'],
                'output_tokens_per_request':entry['output_tokens_per_request'],'coverage':entry['coverage'],
                'source_artifact':entry['artifact'],'source_profiles':old,'target_profiles':targets,
                'profile_differences':differences(old,targets)}
    envelope={'schema':'dots3-inherited-evidence-v1','stage':step['name'],'entry':entry,
              'target_identity':target_identity,'source_artifact':'source-000.raw','archived_sources':archived,
              'evidence_provenance':provenance}
    path=directory/'inherited-evidence.json';path.write_text(json.dumps(envelope,indent=2)+'\n')
    return path


def validate_inherited(step,path,target_identity=None,root=ROOT,target_runtimes=None):
    e=json.loads(path.read_text());assert e['schema']=='dots3-inherited-evidence-v1'
    assert step['requests']==0 and step['evidence_mode']=='inherited'
    assert e['stage']==step['name'] and e['entry']==step['inherited_evidence']
    if target_identity is not None:assert e['target_identity']==target_identity, 'target runtime changed'
    data,old=validate_source(e['entry'],root)
    descriptors=[e['entry']['artifact'],*e['entry']['source_runtimes'].values(),*e['entry'].get('supporting_evidence',[])]
    assert set(e['archived_sources'])=={f'source-{i:03d}.raw' for i in range(len(descriptors))}
    assert e['source_artifact']=='source-000.raw'
    for i,d in enumerate(descriptors):
        assert e['archived_sources'][f'source-{i:03d}.raw']['source']==d
    for name,d in e['archived_sources'].items():
        for relative in (name,d['original']):assert (path.parent/relative).resolve().is_relative_to(path.parent.resolve())
        assert sha((path.parent/name).read_bytes())==d['sha256']==d['source']['raw_sha256']
        assert sha((path.parent/d['original']).read_bytes())==d['source']['sha256']
    assert (path.parent/e['source_artifact']).read_bytes()==blob(e['entry']['artifact'],root)
    p=e['evidence_provenance'];assert p['mode']=='inherited' and p['source_profiles']==old
    assert p['source_artifact']==e['entry']['artifact']
    if target_runtimes is not None:
        assert p['target_profiles']=={h:profile(r) for h,r in target_runtimes.items()}, 'final runtime profile differs'
    assert p['source_kind']==e['entry']['source_kind'] and p['sample_count']==e['entry']['sample_count']
    assert p['output_tokens_per_request']==e['entry']['output_tokens_per_request'] and p['coverage']==e['entry']['coverage']
    assert {revision(v) for v in old.values()}=={revision(v) for v in p['target_profiles'].values()}
    assert p['profile_differences']==differences(old,p['target_profiles'])
    assert all(p['target_profiles'][h]['image_id']==e['target_identity'][h]['image_id'] and p['target_profiles'][h]['args']==e['target_identity'][h]['args'] and p['target_profiles'][h]['started_at']==e['target_identity'][h]['started_at'] for h in old)
    return {'complete':True,'evidence_mode':'inherited','sample_count':p['sample_count']}


def metrics(step,path,stage_metrics):
    e=json.loads(path.read_text());kind=e['entry']['source_kind']
    data=decode((path.parent/e['source_artifact']).read_bytes(),kind)
    if kind=='prefill':
        depth=e['entry']['depth'];point=next(p for p in data['points'] if p['prompt_tokens']==depth);rows=point['runs']
        result={'points':[{'depth':depth,'samples':len(rows),'actual_prompt_tokens':[depth],
                          'completion_tokens':[1],'ttft_seconds_median':statistics.median(r['ttft_seconds'] for r in rows),
                          'decode_tokens_per_second_median':None,
                          'effective_prefill_tokens_per_second_median':statistics.median(depth/r['ttft_seconds'] for r in rows)}],
                'generation_scope':'Prior-profile one-token prefill measurements; no decode-rate evidence.'}
    else:result=stage_metrics(step['name'],data)
    result['evidence_provenance']=e['evidence_provenance']
    return result
