#!/usr/bin/env python3
"""Archive a completed native qualification and extract measured report data.

CPU/filesystem only. Never calls Docker, SSH, or the model API. Does not set
release profiles qualified or establish anonymous registry access.
"""
import argparse
import csv
import math
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def contained(root, relative):
    p = (root / relative).resolve()
    require(p.is_relative_to(root.resolve()) and p.is_file(), f'Missing or escaping artifact: {relative}')
    return p


def flag(args, key):
    values = [args[i+1] for i, value in enumerate(args[:-1]) if value == key]
    values += [value.split('=', 1)[1] for value in args if value.startswith(key+'=')]
    require(len(values) == 1, f'Require one explicit {key}')
    return values[0]


def load_artifact(path, fmt):
    return read(path) if fmt == 'json' else [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def distribution(values):
    values=[v for v in values if isinstance(v,(int,float)) and math.isfinite(v)]
    return {'samples':len(values),'min':min(values) if values else None,
            'median':statistics.median(values) if values else None,
            'max':max(values) if values else None}


def hybrid_attestation(runtime):
    env = dict(item.split('=',1) for item in runtime.get('selected_environment',[]) if '=' in item)
    if not env.get('VLLM_HYBRID_LAYER_PARTITION'):
        return None
    proof = runtime.get('hybrid_attestation', {})
    require(proof.get('status') == 'captured', 'Hybrid final runtime lacks captured ownership evidence')
    require(proof.get('path') == env.get('VLLM_HYBRID_ATTESTATION_PATH'), 'Hybrid receipt path mismatch')
    raw = proof.get('raw_json','')
    require(hashlib.sha256(raw.encode()).hexdigest() == proof.get('sha256'), 'Hybrid receipt hash mismatch')
    receipt = json.loads(raw)
    require(receipt.get('passed') is True, 'Hybrid ownership attestation failed')
    workers = receipt.get('workers', [])
    require(len(workers) == 2 and {w.get('rank') for w in workers} == {0,1}, 'Hybrid ownership must attest both ranks')
    require(all(w.get('passed') is True and not w.get('errors') and w.get('world_size') == 2 for w in workers), 'Hybrid worker ownership failure')
    partition=tuple(int(x) for x in env['VLLM_HYBRID_LAYER_PARTITION'].split(','))
    require(len(partition)==2 and min(partition)>0 and sum(partition)==46, 'Invalid hybrid runtime partition')
    expected_owners=[0]*partition[0]+[1]*partition[1]
    mm_text=env.get('VLLM_HYBRID_MM_OWNERS','')
    mm_plan=None
    if mm_text:
        owners=tuple(int(x) for x in mm_text.split(','))
        require(len(owners)==2 and all(x in (0,1) for x in owners), 'Invalid runtime MM owner plan')
        mm_plan=dict(zip(('visual','audio_tower'),owners))
    runtime_args=runtime.get('args',[])
    for worker in workers:
        layers=worker.get('layers',[])
        require(len(layers)==46 and {row.get('layer') for row in layers}==set(range(46)), 'Ownership receipt must include all46 unique layers')
        for row in layers:
            owner=expected_owners[row['layer']]
            require(row.get('owner_rank')==owner and row.get('local_owner') is (owner==worker['rank']), 'Receipt layer ownership differs from runtime partition')
        if mm_plan is not None:
            require(worker.get('multimodal_owner_plan')==mm_plan, 'Receipt MM owner plan differs from runtime')
        else:
            require(not worker.get('multimodal_owner_plan'), 'Receipt unexpectedly enables MM ownership')
        admission=worker.get('cache_admission_inputs',{})
        require(admission.get('max_model_len')==int(flag(runtime_args,'--max-model-len')), 'Receipt context admission differs from runtime')
        if '--block-size' in runtime_args or any(x.startswith('--block-size=') for x in runtime_args):
            require(admission.get('block_size')==int(flag(runtime_args,'--block-size')), 'Receipt block size differs from runtime')
        # Older receipts record async in-flight tokens, not the batch-token cap.
        # Compare the latter only when explicitly recorded; do not infer a
        # universal multiplier from the scheduler implementation.
        if 'max_num_batched_tokens' in admission:
            require(admission['max_num_batched_tokens']==int(flag(runtime_args,'--max-num-batched-tokens')), 'Receipt batch admission differs from runtime')
    return {'path':proof['path'], 'sha256':proof['sha256'], 'receipt':receipt}


def spark_hybrid_attestations(runtimes):
    """EngineCore head owns the aggregate receipt; headless worker references it."""
    environments={host:dict(item.split('=',1) for item in runtime.get('selected_environment',[]) if '=' in item)
                  for host,runtime in runtimes.items()}
    enabled={host for host,env in environments.items() if env.get('VLLM_HYBRID_LAYER_PARTITION')}
    if not enabled:
        return {host:None for host in runtimes}
    require(enabled==set(runtimes) and len(runtimes)==2, 'Both Spark runtimes must enable hybrid ownership')
    require(len({runtime.get('image_id') for runtime in runtimes.values()})==1
            and all(runtime.get('image_id') for runtime in runtimes.values()), 'Spark ownership images differ')
    keys=['VLLM_HYBRID_LAYER_PARTITION','VLLM_HYBRID_PACKED_ROUTING','VLLM_HYBRID_FUSED_PACK',
          'VLLM_HYBRID_OVERLAP_SHARED','VLLM_HYBRID_MM_OWNERS','VLLM_HYBRID_BOUNDARY_OWNERS',
          'VLLM_HYBRID_BALANCE_KV_GROUPS','DOTS3_COMPACT_DSA_CACHE','DOTS3_INDEXER_PREFILL_CONTEXTS']
    for key in keys:
        require(len({env.get(key,'') for env in environments.values()})==1, f'Spark ownership profile mismatch: {key}')
    for key in ['--max-model-len','--max-num-seqs','--max-num-batched-tokens','--gpu-memory-utilization','--kv-cache-dtype','--tensor-parallel-size']:
        require(len({flag(runtime.get('args',[]),key) for runtime in runtimes.values()})==1, f'Spark runtime profile mismatch: {key}')
    heads=[host for host,runtime in runtimes.items() if runtime.get('hybrid_attestation',{}).get('status')=='captured']
    require(len(heads)==1, 'Require exactly one captured EngineCore aggregate ownership receipt')
    head=heads[0]
    require('--headless' not in runtimes[head].get('args',[]), 'Aggregate receipt must belong to EngineCore head')
    proof=hybrid_attestation(runtimes[head])
    result={head:proof}
    for host in runtimes:
        if host==head:continue
        require('--headless' in runtimes[host].get('args',[]), 'Non-head receipt reference requires headless worker')
        result[host]={'status':'aggregate-head-reference','head_host':head,'sha256':proof['sha256'],
                      'scope':'Both rank proofs are in the head EngineCore receipt; worker has no local aggregate file'}
    return result


def memory_summary(source, platform):
    """Observed sample extrema, not exact instantaneous allocation peaks."""
    hosts={};errors=[];files=[]
    def add(host,memory,timestamp):
        row=hosts.setdefault(host,{'samples':0,'timestamps':[],'available':[],'swap_used':[],'gpus':{}})
        row['samples']+=1;row['timestamps'].append(timestamp)
        if 'MemAvailable' in memory:row['available'].append(memory['MemAvailable'])
        if 'SwapTotal' in memory and 'SwapFree' in memory:row['swap_used'].append(memory['SwapTotal']-memory['SwapFree'])
        return row
    pattern='memory.jsonl' if platform=='rtx' else 'physical-memory.jsonl'
    for path in sorted(source.glob('attempt-*/'+pattern)):
        files.append(str(path.relative_to(source)))
        for line_number,line in enumerate(path.read_text().splitlines(),1):
            try:
                data=json.loads(line)
                if data.get('error') and ('hosts' not in data and 'meminfo' not in data):
                    errors.append({'file':files[-1],'line':line_number,'error':data['error']});continue
                if platform=='spark':
                    for host,observed in data.get('hosts',{}).items():
                        add(host,observed['physical_memory_bytes'],data['time'])
                else:
                    memory={parts[0].rstrip(':'):int(parts[1])*1024 for text in data.get('meminfo','').splitlines()
                            if len(parts:=text.split())>=2 and parts[0].rstrip(':') in ('MemAvailable','SwapTotal','SwapFree')}
                    row=add('rtx',memory,data['time'])
                    if data.get('returncode')!=0:
                        errors.append({'file':files[-1],'line':line_number,'error':data.get('error'),'returncode':data.get('returncode')});continue
                    for fields in csv.reader(data.get('gpu_csv','').splitlines(),skipinitialspace=True):
                        if len(fields) not in (6,9,11):raise ValueError('unexpected nvidia-smi column count')
                        index,uuid,used,total,util,power=fields[:6]
                        gpu=row['gpus'].setdefault(uuid,{'index':index,'memory_used_mib':[],'memory_total_mib':[],'utilization_percent':[],'power_watts':[]})
                        for key,value in zip(('memory_used_mib','memory_total_mib','utilization_percent','power_watts'),(used,total,util,power)):
                            try:number=float(value)
                            except ValueError:continue
                            if math.isfinite(number):gpu[key].append(number)
                        for key,value in zip(('temperature_celsius','sm_clock_mhz','memory_clock_mhz'),fields[6:9]):
                            try:number=float(value)
                            except ValueError:continue
                            if math.isfinite(number):gpu.setdefault(key,[]).append(number)
                        for key,value in zip(('sw_thermal_slowdown','hw_thermal_slowdown'),fields[9:11]):
                            if value.strip().lower() in ('active','not active'):
                                gpu.setdefault(key,[]).append(value.strip().lower()=='active')
            except (ValueError,KeyError,TypeError) as exc:
                errors.append({'file':files[-1],'line':line_number,'error':str(exc)})
    result={}
    for host,data in hosts.items():
        result[host]={'samples':data['samples'],'first_sample_unix':min(data['timestamps']),
                      'last_sample_unix':max(data['timestamps']),
                      'minimum_mem_available_bytes':min(data['available']) if data['available'] else None,
                      'maximum_swap_used_bytes':max(data['swap_used']) if data['swap_used'] else None,
                      'gpus':{uuid:{'index':gpu['index'],
                          'peak_observed_memory_used_mib':max(gpu['memory_used_mib']) if gpu['memory_used_mib'] else None,
                          'memory_used_mib':distribution(gpu['memory_used_mib']),
                          'memory_total_mib':distribution(gpu['memory_total_mib']),
                          'utilization_percent':distribution(gpu['utilization_percent']),
                          'power_watts':distribution(gpu['power_watts']),
                          'optional_telemetry':{key:distribution(gpu.get(key,[])) for key in ('temperature_celsius','sm_clock_mhz','memory_clock_mhz')},
                          'thermal_slowdown':{key:{'observations':len(gpu.get(key,[])), 'active_samples':sum(gpu.get(key,[]))} for key in ('sw_thermal_slowdown','hw_thermal_slowdown')}}for uuid,gpu in data['gpus'].items()}}
    return {'scope':'All recorded runner attempts, including earlier resumed/failed attempts; sampled extrema can miss brief peaks. Host swap use is system-wide, not attributable solely to vLLM.',
            'gpu_scope':'RTX records nvidia-smi MiB. Spark monitor records physical unified host memory; no separate GPU allocation series is inferred.',
            'files':files,'hosts':result,'monitor_errors':errors}


def startup_memory(runtime):
    patterns={'model_loading_gib':r'Model loading took ([0-9.]+) GiB',
              'available_kv_gib':r'Available KV cache memory: ([0-9.]+) GiB',
              'gpu_kv_cache_tokens':r'GPU KV cache size: ([0-9,]+) tokens',
              'graph_capture_gib':r'Graph capturing finished.*took ([0-9.]+) GiB'}
    result={name:[] for name in patterns}
    for line in runtime.get('selected_startup_lines',[]):
        for name,pattern in patterns.items():
            if match:=re.search(pattern,line):
                value=match.group(1)
                result[name].append({'value':int(value.replace(',','')) if name.endswith('_tokens') else float(value),'raw_line':line})
    result['scope']='Directly parsed startup log entries; rank-specific/repeated graph phases are preserved and not summed or treated as simultaneous allocations.'
    return result


def stage_metrics(name, data):
    if name == 'tool-quality':
        sys.path.insert(0,str(ROOT/'serving/benchmarks'))
        from tool_quality import validate as validate_tools
        return validate_tools(data)
    if name == 'prefix':
        return {key: data[key] for key in ['warm_prefix_hits', 'warm_prefix_queries', 'xgrammar_json', 'cold', 'warm', 'forced_tool']}
    if name == 'multimodal':
        return {'examples': data['source_examples']}
    if name == 'clients':
        return {'method': data['method'], 'output_tokens_per_sequence': data['output_tokens_per_sequence'],
            'points': [{'concurrency': p['concurrency'],
                'aggregate_decode_tokens_per_second': p['aggregate_decode_tokens_per_second'],
                'minimum_overlap': min(r['peak_overlapping_stream_intervals'] for r in p['runs'])} for p in data['points']]}
    if name in ('seven', 'coding'):
        result = dict(data[-1])
        if name == 'coding':
            measured=[(wave,r) for wave in data if wave.get('record')=='wave' and wave['timed'] for r in wave['request_results']]
            result['distributions_by_concurrency']={}
            for c in sorted({wave['concurrency'] for wave,r in measured}):
                rows=[r for wave,r in measured if wave['concurrency']==c]
                completed=[r for r in rows if r['completed'] and r.get('error') is None]
                truncated=[r for r in rows if r['truncated'] and r.get('error') is None]
                groups={'all_terminal':rows,'naturally_completed':completed,'truncated':truncated}
                result['distributions_by_concurrency'][str(c)]={key:{
                    'requests':len(group),
                    'output_tokens':distribution(r['stream_token_count'] for r in group if r.get('token_accounting_valid')),
                    'latency_seconds':distribution(r['completion_latency_seconds'] for r in group)} for key,group in groups.items()}
            result['distribution_scope']='Output token distributions include reasoning and require valid stream/usage accounting. Natural completion and length truncation latency distributions are separate; all_terminal includes any errors.'
            result['quality_misses'] = [{'concurrency': wave['concurrency'], 'run': wave['run'],
                'task': r['task'], 'truncated': r['truncated'], 'finish_reason': r['finish_reason'],
                'content_result': r['content_result']} for wave in data if wave.get('record') == 'wave' and wave['timed']
                for r in wave['request_results'] if r['truncated'] or not r['content_result']['static_checks_passed']]
        return result
    if name == 'reasoning-api':
        s = data[-1]
        return {key: s[key] for key in ['passed', 'cases_passed', 'cases_total', 'failed_cases', 'evidence_issues',
            'speculative_draft_token_delta', 'structured_boundary_chunks', 'boundary_evidence_scope']}
    rows = [r for r in data if r.get('record') == 'measurement' and r.get('timed', True)]
    if name.startswith('retrieval-'):
        return {'passed': all(r['passed'] for r in rows), 'requests': len(rows),
            'positions': [{'position_fraction': r['position_fraction'], 'passed': r['passed'],
                          'usage': r['usage']} for r in rows]}
    result = []
    for depth in sorted({r['depth'] for r in rows}):
        group = [r for r in rows if r['depth'] == depth]
        result.append({'depth': depth, 'samples': len(group),
            'actual_prompt_tokens': sorted({r['usage']['prompt_tokens'] for r in group}),
            'completion_tokens': sorted({r['usage']['completion_tokens'] for r in group}),
            'ttft_seconds_median': statistics.median(r['ttft_seconds'] for r in group),
            'decode_tokens_per_second_median': statistics.median(r['decode_tps'] for r in group),
            'effective_prefill_tokens_per_second_median': statistics.median(r['usage']['prompt_tokens']/r['ttft_seconds'] for r in group)})
    if name=='code-agent':
        for point in result:
            group=[r for r in rows if r['depth']==point['depth']]
            point['reference_n_minus_one_tokens_per_second']=distribution(r.get('reference_n_minus_one_tps') for r in group)
        return {'points':result,
            'decode_timing':'decode_tokens_per_second_median excludes every token in the first SSE burst; elapsed time is last burst minus first burst.',
            'reference_timing':'reference_n_minus_one_tokens_per_second uses (output_tokens-1) / the same elapsed time, matching the older reference convention; speculative multi-token first bursts can make this larger.',
            'headline_mapping':'Sampled async coding baseline is depth0, temperature0.2, thinking disabled, fixed256-token completion. State which timing convention is used.',
            'generation_scope':'Fixed-length throughput probe; natural completion and code correctness are not established.'}
    return {'points': result, 'generation_scope': 'Fixed-length throughput probe; natural completion and code correctness are not established.'}


def export(args):
    source = args.input.resolve()
    target = args.output.resolve()
    require(not target.is_relative_to('/mnt/scratch'), 'Use native project storage, not /mnt/scratch')
    require(not target.exists(), 'Output must be a new directory')
    manifest = read(source/'manifest.json')
    require(manifest['schema'] in {f'dots3-{args.platform}-release-qualification-v1',f'dots3-{args.platform}-release-qualification-v2'}, 'Wrong runner schema/platform')
    current_schema=manifest['schema'].endswith('-v2')
    if current_schema:
        require(sum(s['name']=='tool-quality' for s in manifest['plan'])==1, 'New release requires full hard-mode tool-quality stage')
    completions = sorted(source.glob('attempt-*/complete.json'))
    require(bool(completions), 'No ending complete.json: qualification is incomplete; no report written')
    complete_path = completions[-1]
    completion = read(complete_path)
    require(completion.get('completed') is True, 'Ending receipt is not complete')
    require(completion['requests_in_full_plan'] == sum(s['requests'] for s in manifest['plan']), 'Plan count mismatch')
    for relative, expected in manifest['source_sha256'].items():
        require(sha(contained(ROOT, relative)) == expected, f'Runner/validator source drift: {relative}; use exact recorded checkout')
    validator_name = f'qualify_{args.platform}'
    require(f'serving/release/{validator_name}.py' in manifest['source_sha256'], 'Validator source is not bound')
    validator = importlib.import_module(validator_name)
    identities = {'rtx': manifest['identity']} if args.platform == 'rtx' else manifest['identity']
    image_ids = {i['image_id'] for i in identities.values()}
    require(len(image_ids) == 1, 'Ranks use different image IDs')
    require(re.fullmatch(r'ghcr\.io/[a-z0-9_./-]+@sha256:[a-f0-9]{64}', args.image), 'Require immutable GHCR digest')
    registry = read(args.registry_receipt) if args.registry_receipt else None
    if registry:
        require(registry['wrapper_image_id'] in image_ids, 'Registry receipt image ID mismatch')
        require(registry['registry_digest'] == args.image.split('@')[1], 'Registry digest mismatch')
        require(registry['image_tag'].rsplit(':', 1)[0] == args.image.split('@')[0], 'Registry repository mismatch')
    else:
        require(args.platform == 'spark' and all(args.image in digests for digests in manifest['repo_digests'].values()),
            'RTX needs registry receipt binding local image ID to published digest')
    cache = read(args.cache_manifest)
    if registry:
        require(cache['source_image_id'] == registry['parent_image_id'], 'Cache provenance parent mismatch')
        require(cache['model_revision'] == registry['model_revision'], 'Cache checkpoint mismatch')
    profiles = {}
    for host, identity in identities.items():
        command = identity['args']
        require(any(cache['model_revision'] in item for item in command), 'Checkpoint revision absent from launch args')
        profiles[host] = {key[2:].replace('-', '_'): flag(command, key) for key in
            ['--max-model-len', '--max-num-seqs', '--max-num-batched-tokens', '--gpu-memory-utilization',
             '--kv-cache-dtype', '--reasoning-parser', '--tool-call-parser', '--speculative-config', '--tensor-parallel-size']}
        # Ordinary TP releases predate the hybrid block-size/worker extension
        # flags. Preserve explicit values when present without requiring them
        # in historical or newly qualified ordinary TP profiles.
        for key in ('--block-size', '--worker-extension-cls'):
            if any(item == key or item.startswith(key+'=') for item in command):
                profiles[host][key[2:].replace('-', '_')] = flag(command, key)
    artifacts = {source/'manifest.json', complete_path}
    results = {}
    require(len({s['name'] for s in manifest['plan']}) == len(manifest['plan']), 'Duplicate stage names')
    for step in manifest['plan']:
        stage = source/step['name']
        require(stage.resolve().is_relative_to(source), 'Stage path escapes source')
        receipt_path = stage/'receipt.json'
        receipt = read(receipt_path)
        artifact = contained(stage, receipt['artifact'])
        if current_schema:
            require(receipt.get('identity')==manifest['identity'], 'Stage runtime image/profile identity mismatch')
        require(sha(artifact) == receipt['sha256'], f'Artifact hash mismatch: {step["name"]}')
        require(receipt['validated'].get('complete') is True, 'Unvalidated stage receipt')
        if args.platform == 'spark':
            validator.validate(step, artifact, manifest['limits']['max_model_len'])
        else:
            validator.validate(step, artifact, manifest.get('limits',{}).get('max_model_len',262144))
        results[step['name']] = stage_metrics(step['name'], load_artifact(artifact, step['format']))
        artifacts.add(receipt_path)
        artifacts.update(p for p in artifact.parent.rglob('*') if p.is_file())
        if step['name']=='tool-quality':
            artifacts.update(p for p in artifact.parent.parent.iterdir() if p.is_file())
    # Keep monitoring and before/after snapshots from all attempts, including
    # earlier resume attempts; stage artifacts include only accepted attempts.
    for attempt in source.glob('attempt-*'):
        artifacts.update(p for p in attempt.iterdir() if p.is_file())
    hardware = {}
    if args.platform == 'rtx':
        runtime = read(complete_path.parent/'runtime-after.json')
        require(runtime['image_id'] in image_ids, 'Final runtime image mismatch')
        hardware['rtx'] = {k: runtime[k] for k in ['host', 'architecture', 'gpu_inventory_csv', 'meminfo', 'docker_stats']}
        profiles['rtx']['selected_environment'] = runtime['selected_environment']
        hardware['rtx']['startup_memory_evidence']=startup_memory(runtime)
        hardware['rtx']['hybrid_attestation']=hybrid_attestation(runtime)
    else:
        spark_runtimes={}
        for host in identities:
            snapshot = read(complete_path.parent/f'{host}-after.json')
            require(snapshot['identity'] == identities[host], 'Final host identity changed')
            hardware[host] = {k: v for k, v in snapshot.items() if k not in ('container_log', 'guard_log', 'identity')}
            hardware[host]['startup_memory_evidence']=startup_memory(snapshot.get('runtime',{}))
            spark_runtimes[host]=snapshot.get('runtime',{})
            require(spark_runtimes[host].get('image_id') in image_ids, 'Spark final runtime image mismatch')
        proofs=spark_hybrid_attestations(spark_runtimes)
        for host,proof in proofs.items():
            hardware[host]['hybrid_attestation']=proof
    report = {'schema': 'dots3-release-report-v1', 'status': 'runner-completed; release-profile approval separate',
        'qualification_schema':manifest['schema'], 'tool_quality_required':current_schema,
        'platform': args.platform, 'image': args.image, 'image_id': next(iter(image_ids)),
        'model_revision': cache['model_revision'], 'recipe_revision': cache['recipe_revision'],
        'parent_image_id': cache['source_image_id'], 'source_sha256': manifest['source_sha256'],
        'hardware': hardware, 'profile': profiles, 'stage_results': results,
        'memory_observations':memory_summary(source,args.platform),
        'completion': completion, 'registry_receipt': {k: registry[k] for k in ('registry_digest', 'registry', 'image_tag') if k in registry} if registry else None,
        'quality_scope': 'Static code/content checks are not executed-code correctness. Misses/truncations remain in stage results. Fixed-output timing probes are not natural-completion tests.',
        'prefill_scope': 'Actual prompt tokens / client TTFT, including first-token handoff; not kernel-only prefill speed.'}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.report-', dir=target.parent) as temp:
        temp = Path(temp); archive = []
        sources = [(p, 'runner/'+str(p.relative_to(source))) for p in sorted(artifacts)]
        sources.extend((contained(ROOT, path), 'provenance/source/'+path) for path in manifest['source_sha256'])
        sources.append((Path(__file__).resolve(), 'provenance/source/serving/release/report.py'))
        sources.append((args.cache_manifest.resolve(), 'provenance/cache-seed-manifest.json'))
        if args.registry_receipt:
            sources.append((args.registry_receipt.resolve(), 'provenance/registry-receipt.json'))
        for path, relative in sources:
            raw = path.read_bytes()
            require(not re.search(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|Authorization: (?:Bearer|Basic) [A-Za-z0-9])', raw), f'Potential credential in {path}')
            packed = gzip.compress(raw, compresslevel=9, mtime=0)
            dest = temp/(relative+'.gz');dest.parent.mkdir(parents=True, exist_ok=True);dest.write_bytes(packed)
            archive.append({'path': str(dest.relative_to(temp)), 'source': str(path), 'raw_bytes': len(raw),
                'raw_sha256': hashlib.sha256(raw).hexdigest(), 'gzip_sha256': hashlib.sha256(packed).hexdigest()})
        (temp/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        (temp/'archive-manifest.json').write_text(json.dumps({'schema': 1, 'report_sha256': sha(temp/'report.json'), 'files': archive}, indent=2)+'\n')
        os.rename(temp, target)
    print(json.dumps({'report': str(target/'report.json'), 'sha256': sha(target/'report.json'), 'archived_files': len(archive)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--platform', choices=['rtx', 'spark'], required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--cache-manifest', type=Path, required=True)
    p.add_argument('--registry-receipt', type=Path)
    export(p.parse_args())


if __name__ == '__main__':
    main()
