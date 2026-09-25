"""Refine shared global SWA groups against each worker's history page budget.

Existing global group IDs stay in place; overflow groups append once before
worker projection. Layer pages, attention types, retention and block-table
semantics are unchanged. Only the set of compatible layers sharing a physical
block changes, preventing bounded state groups from widening history blocks.
"""
from vllm.v1.kv_cache_interface import KVCacheGroupSpec,UniformTypeKVCacheSpecs,SlidingWindowSpec


def refine_owner_state_groups(groups,worker_specs):
    def specs(group):
        if isinstance(group.kv_cache_spec,UniformTypeKVCacheSpecs):
            return {n:group.kv_cache_spec.kv_cache_specs[n] for n in group.layer_names}
        return {n:group.kv_cache_spec for n in group.layer_names}
    def is_state(group):
        values=specs(group).values()
        return bool(group.layer_names) and all(isinstance(s,SlidingWindowSpec) for s in values)
    anchors=[max((sum(s.page_size_bytes for n,s in specs(g).items() if n in worker)
                  for g in groups if not is_state(g)),default=0) for worker in worker_specs]
    limits=[max(anchor,max((s.page_size_bytes for g in groups if is_state(g)
                            for n,s in specs(g).items() if n in worker),default=0))
            for anchor,worker in zip(anchors,worker_specs)]
    def fits(candidate):
        return all(sum(s.page_size_bytes for n,s in candidate.items() if n in worker)<=limit
                   for worker,limit in zip(worker_specs,limits))
    def group_from(source,items):
        uniform=UniformTypeKVCacheSpecs.from_specs(items)
        if uniform is None:raise ValueError('owner state refinement lost uniform attention type')
        return KVCacheGroupSpec(list(items),uniform,is_eagle_group=source.is_eagle_group)
    output=[];overflow=[]
    for group in groups:
        if not is_state(group):output.append(group);continue
        retained={}
        for name,spec in specs(group).items():
            candidate={**retained,name:spec}
            if fits(candidate):retained=candidate
            else:overflow.append((group,name,spec))
        if not retained:raise ValueError('owner state refinement unexpectedly emptied a group')
        output.append(group_from(group,retained))
    extra=[]
    for source,name,spec in overflow:
        for i,(prototype,items) in enumerate(extra):
            candidate={**items,name:spec}
            if (prototype.is_eagle_group==source.is_eagle_group and
                UniformTypeKVCacheSpecs.is_uniform_type(candidate) and fits(candidate)):
                extra[i]=(prototype,candidate);break
        else:extra.append((source,{name:spec}))
    output.extend(group_from(source,items) for source,items in extra)
    before=[n for g in groups for n in g.layer_names]
    after=[n for g in output for n in g.layer_names]
    if len(after)!=len(set(after)) or set(after)!=set(before):
        raise ValueError('owner state refinement duplicated or lost cache layers')
    return output
