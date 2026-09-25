#!/usr/bin/env python3
"""Render a review-only dual-platform README draft from completed report JSON."""
import argparse
import hashlib
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2]


def number(value):
    return '—' if value is None else f'{value:,.2f}'


def load(path,platform):
    if path is None:return None
    result=json.loads(path.read_text())
    if result.get('platform')!=platform or result.get('completion',{}).get('completed') is not True:
        raise ValueError('Require completed report for matching platform')
    if not re.fullmatch(r'ghcr\.io/[^@]+@sha256:[0-9a-f]{64}',result.get('image','')):
        raise ValueError('Report lacks immutable published image')
    return result


def render(reports,paths):
    lines=['# Dots3 Note Preview: one EXL3 K4 checkpoint, two GPU platforms','',
      'The FP8 model core is preserved. Routed experts come from the BF16 checkpoint and use uniform EXL3 K4 across projections. Both recipes use the same installed checkpoint. [Quantization provenance](quantization/README.md).','',
      '**Review draft.** Supplied completed reports drive every measurement below; a pending platform has no inferred results. Recipe activation and anonymous GHCR access must still be confirmed separately.','',
      '## Containers and measured profiles','',
      'RTX and Spark have separate native containers. Mount the entire existing Hugging Face cache with `HF_HOME`; model files are already local. Runtime caches must be writable. See [release fast path](serving/release/README.md).']
    def table(headers,rows):
        lines.extend(['','| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|'])
        lines.extend('| '+' | '.join(str(x) for x in row)+' |' for row in rows)
    for platform in ('spark','rtx'):
        r=reports[platform];lines+=['',f'### {"2× Spark" if platform=="spark" else "2× RTX"}']
        if r is None:lines+=['','Pending final published-image qualification.'];continue
        profile=next(iter(r['profile'].values()));historical=not str(r.get('qualification_schema','')).endswith('-v2')
        lines+=['',('Historical completed profile; new native-context results pending.' if historical else 'Completed report. Activate reviewed settings bound to this digest before running the launch command below.'),'','```bash',f'docker pull {r["image"]}',f'export HF_HOME="$HOME/.cache/huggingface"',(f'bash serving/release/run.sh {platform} start' if platform=='rtx' else '# Set per-host network/rank variables and follow worker-first commands below'),'```','',f"Measured context **{profile['max_model_len']}**, memory utilization **{profile['gpu_memory_utilization']}**, batch cap **{profile['max_num_batched_tokens']}**, MTP config `{profile['speculative_config']}`."]
        if platform=='spark':lines+=['','On both hosts set `MASTER_ADDR=HEAD_NETWORK_IP` and `SOCKET_IFNAME=RDMA_NETWORK_INTERFACE`. Start worker first, then head with the same digest; preserve the physical-memory guard.','','```bash','# Worker:','NODE_RANK=1 HOST_IP=WORKER_NETWORK_IP bash serving/release/run.sh spark start','# Head, after worker:','NODE_RANK=0 HOST_IP=HEAD_NETWORK_IP bash serving/release/run.sh spark start','```']
        env=profile.get('selected_environment') or r.get('hardware',{}).get(next(iter(r['profile'])),{}).get('runtime',{}).get('selected_environment',[])
        env=dict(x.split('=',1) for x in env if '=' in x)
        table(['Measured runtime choice','Value'],[(k,env.get(k,'not recorded')) for k in ['VLLM_HYBRID_LAYER_PARTITION','VLLM_HYBRID_MM_OWNERS','VLLM_HYBRID_PACKED_ROUTING','VLLM_HYBRID_FUSED_PACK','VLLM_HYBRID_OVERLAP_SHARED','VLLM_HYBRID_BALANCE_KV_GROUPS','DOTS3_B12X_EXACT_FP8','DOTS3_B12X_EXACT_FP8_ROWS']])
        restart=r.get('restart_continuation')
        if restart:
            lines+=['',f"Qualification spans an explicitly recorded restart: {len(restart['preserved_stages'])} completed stages were preserved, and {len(restart['remaining_stages'])} unfinished stages were continued with the same image, model, launch profile and workload sources. This was not one uninterrupted run. The report retains the interruption evidence, old/new runtime identities and prior-artifact hashes."]
    lines+=['','Hybrid profiles assign attention/norm/shared-expert parameters and their KV caches to layer owners while retaining routed-expert TP2 across both GPUs. Compact DSA records, bounded SWA pools and owner-aware cache grouping reduce wasted capacity. These settings are platform-qualified; sparse-MQA and boundary ownership experiments are not implied by this explanation.','',
      'The [Brandon-derived RTX recipe](https://github.com/tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx) and [Qwen Spark recipe](https://github.com/tpurtell/sm12x-exl3-qwen3.8-flash-next) are optimization/reporting references. Selected Dots3 features and rejected candidates are recorded in the [optimization matrix](docs/optimization-matrix.md).','',
      '## Headline measurements']
    def stage(platform,name):return reports[platform]['stage_results'].get(name,{}) if reports[platform] else {}
    table(['Metric','2× Spark','2× RTX'],
      [(f'C{c} reasoning/coding aggregate output tokens/s',*[number(stage(p,'coding').get('by_concurrency',{}).get(c,{}).get('aggregate_output_tps')) for p in ('spark','rtx')]) for c in ('1','2','4')]+
      [(label,*[number(stage(p,'seven').get(key)) for p in ('spark','rtx')]) for label,key in [('C1 seven-workload weighted decode tokens/s','weighted_decode_tps')]]+
      [('Seven content contracts',*[f"{stage(p,'seven')['contracts_passed']}/{stage(p,'seven')['contracts_total']}" if stage(p,'seven') else 'Pending' for p in ('spark','rtx')])])
    for label,name,selector in [
        ('C1 greedy merge_intervals median tokens/s','seven',lambda s: s.get('median_tps_by_case',{}).get('code')),
        ('Sampled async code, depth 0, burst-excluded tokens/s','code-agent',lambda s: next((x['decode_tokens_per_second_median'] for x in s.get('points',[]) if x['depth']==0),None)),
        ('C16 sampled clients aggregate tokens/s','clients',lambda s: next((x['aggregate_decode_tokens_per_second']['median'] for x in s.get('points',[]) if x['concurrency']==16),None)),
    ]:
        lines.append('| '+label+' | '+' | '.join(number(selector(stage(p,name))) for p in ('spark','rtx'))+' |')
    lines+=['','## Reasoning-enabled coding: C1–C4']
    rows=[]
    for p in ('spark','rtx'):
        coding=stage(p,'coding')
        for c in ('1','2','4'):
            row=coding.get('by_concurrency',{}).get(c,{})
            lengths=coding.get('distributions_by_concurrency',{}).get(c,{}).get('all_terminal',{}).get('output_tokens',{})
            rows.append([p,c,number(row.get('aggregate_output_tps')),number(row.get('median_completed_latency_seconds')),f"{row.get('completed','—')}/{row.get('requests','—')}",f"{row.get('static_checks_passed','—')}/{row.get('requests','—')}",row.get('truncated','—'),f"{number(lengths.get('median'))} ({lengths.get('min','—')}–{lengths.get('max','—')})"])
    table(['Platform','C','Aggregate output tokens/s','Completed latency s','Natural','Static checks','Truncated','Output tokens median (range)'],rows)
    lines+=['','Coding aggregate throughput is total streamed output tokens divided by summed concurrent-wave wall time, including reasoning and prefill. It is measured directly, not concurrency times a per-stream median. Completed latency excludes truncated answers. The benchmark output budget 8192 includes reasoning and is not the server output limit; clients may request more within context. Static checks are not execution-based code correctness. Variable output lengths affect latency.']
    lines+=['','## Seven content workloads']
    table(['Case','Spark median tokens/s','RTX median tokens/s'],[(case,*[number(stage(p,'seven').get('median_tps_by_case',{}).get(case)) for p in ('spark','rtx')]) for case in ['code','math','fable','hello','topic','structured-json','multilingual']])
    for p in ('spark','rtx'):
        failed=stage(p,'seven').get('failed_contracts',[])
        if failed:lines+=['',f'{p} content contract misses: '+ '; '.join(f"{x['case']} run{x['run']}: "+', '.join(x['contract']['quality_contract_issues']) for x in failed)+'.']
    lines+=['','## Sampled prose clients']
    table(['Platform','Clients','Aggregate tokens/s','Minimum overlap'],[[p,row['concurrency'],number(row['aggregate_decode_tokens_per_second']['median']),row['minimum_overlap']] for p in ('spark','rtx') for row in stage(p,'clients').get('points',[])])
    lines+=['','## Sampled async code and context scaling']
    rows=[]
    for p,r in reports.items():
        if not r:continue
        for name,s in r['stage_results'].items():
            if name=='code-agent' or name.startswith('context-'):
                for row in s['points']:rows.append([p,name,row['depth'],number(row['ttft_seconds_median']),number(row['effective_prefill_tokens_per_second_median']),number(row['decode_tokens_per_second_median'])])
    table(['Platform','Probe','Input depth','TTFT s','Prompt tokens / TTFT','Decode tokens/s'],rows)
    lines+=['','Decode excludes the entire first SSE token burst. Prompt tokens / TTFT includes tokenization and first-token handoff. For unique cold context probes this estimates effective prefill throughput. Sampled code-agent history probes can reuse prefixes; their ratios are not prefill-speed measurements. Fixed 256-output probes do not demonstrate natural completion.']
    lines+=['','## Tool-use quality: hard mode, Basic / Hard / Total']
    rows=[]
    for p in ('spark','rtx'):
        tool=stage(p,'tool-quality')
        for name in ('Basic','Hard','Total'):
            g=tool.get('groups',{}).get(name,{})
            rows.append([p,name,f"{g.get('points','—')}/{g.get('max_points','—')}",number(g.get('score_percent')),f"{g.get('pass','—')}/{g.get('partial','—')}/{g.get('fail','—')}"])
    table(['Platform','Suite','Points','Score %','Pass/partial/fail'],rows)
    lines+=['','Pinned public suite: 69 Basic + 19 Hard = 88, partial credit 0/1/2. Missing entries are unmeasured, not zero. Total weights scenario counts. Parser/API compatibility checks below are a separate measure. [Reproduce the hard-mode suite](serving/benchmarks/tool_quality.md).','', '## Functional checks and sampled memory']
    rows=[]
    for p,r in reports.items():
        if not r:continue
        api=stage(p,'reasoning-api');prefix=stage(p,'prefix');retrieval=[s for name,s in r['stage_results'].items() if name.startswith('retrieval-')]
        rows.append([p,f"{api.get('cases_passed')}/{api.get('cases_total')}",f"{prefix.get('warm_prefix_hits')}/{prefix.get('warm_prefix_queries')}",number(prefix.get('cold',{}).get('ttft_seconds')),number(prefix.get('warm',{}).get('ttft_seconds')),sum(s['requests'] for s in retrieval if s['passed']),len(stage(p,'multimodal').get('examples',[]))])
    table(['Platform','Reasoning/tools/JSON','Prefix hits/queries','Cold TTFT s','Warm TTFT s','Passed retrieval probes','MM examples'],rows)
    lines+=['','Prefix counters establish reuse. Cold/warm latency includes first-use overhead and is not an isolated cache-speedup measurement.']
    for p,r in reports.items():
        if not r:continue
        for host,m in r.get('memory_observations',{}).get('hosts',{}).items():
            lines+=['',f"{p}/{host}: minimum sampled MemAvailable {number(m['minimum_mem_available_bytes']/2**30) if m.get('minimum_mem_available_bytes') is not None else '—'}GiB; "+', '.join(f"GPU{g['index']} peak {number(g['peak_observed_memory_used_mib']/1024)}GiB" for g in m.get('gpus',{}).values())+'.']
    lines+=['','Sampled peaks can miss brief excursions; Spark memory is shared with the host. Quality misses/truncations and raw traces remain in the reports.','', '## Evidence']
    for p,path in paths.items():
        if path:lines+=['',f'- {p}: `{path}`; SHA256 `{hashlib.sha256(path.read_bytes()).hexdigest()}`.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rtx',type=Path);p.add_argument('--spark',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.resolve()==ROOT/'README.md':raise ValueError('Draft output only; review before replacing root README')
    paths={'spark':a.spark,'rtx':a.rtx};reports={key:load(path,key) for key,path in paths.items()}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(render(reports,paths))
if __name__=='__main__':main()
