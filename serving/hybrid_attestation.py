"""Read-only metadata attestation of a loaded hybrid model and bound KV cache.

No tensor values are copied and no kernels or synchronization are launched.
Call after weights and KV allocation. The receipt verifies storage/geometry and
ownership; it does not replace numeric correctness tests.
"""
import re


def _tensor(tensor):
    storage = tensor.untyped_storage()
    return {'shape': list(tensor.shape), 'stride': list(tensor.stride()),
            'dtype': str(tensor.dtype), 'device': str(tensor.device),
            'storage_offset': tensor.storage_offset(), 'storage_bytes': storage.nbytes(),
            'storage_pointer': storage.data_ptr()}


def _parameter_bytes(module):
    return sum(p.numel() * p.element_size() for p in module.parameters())


def _module_storage(module):
    if module is None:
        return {'present': False, 'parameter_bytes': 0, 'unique_storage_bytes': 0}
    parameters = list(module.parameters())
    buffers = list(module.buffers())
    storages = {}
    by_dtype = {}
    for tensor in parameters + buffers:
        detail = _tensor(tensor)
        storages[(detail['device'], detail['storage_pointer'])] = detail['storage_bytes']
        by_dtype[detail['dtype']] = by_dtype.get(detail['dtype'], 0) + tensor.numel() * tensor.element_size()
    return {'present': True, 'parameter_count': sum(p.numel() for p in parameters),
            'parameter_bytes': sum(p.numel()*p.element_size() for p in parameters),
            'buffer_bytes': sum(p.numel()*p.element_size() for p in buffers),
            'unique_storage_bytes': sum(storages.values()), 'logical_bytes_by_dtype': by_dtype}


def attest_model(model, vllm_config, *, require_bound_cache=True):
    from vllm.distributed import get_tp_group
    language = getattr(model, 'language_model', model)
    if language is model and hasattr(model, 'get_language_model'):
        language = model.get_language_model()
    decoder = language.model
    group = get_tp_group()
    config = decoder.config
    expected_routed = list(range(config.first_k_dense_replace, config.num_hidden_layers))
    rows, routed = [], []
    errors = []
    for layer in decoder.layers:
        owner = layer.context.owns_parameters
        row = {'layer': layer.layer_idx, 'owner_rank': layer.owner,
               'local_owner': owner, 'attention_parameter_bytes': _parameter_bytes(layer.self_attn),
               'input_norm_parameter_bytes': _parameter_bytes(layer.input_layernorm),
               'post_attention_norm_parameter_bytes': _parameter_bytes(layer.post_attention_layernorm)}
        if owner:
            expected_heads = config.swa_num_attention_heads if config.layer_types[layer.layer_idx] == 'sliding_attention' else config.num_attention_heads
            attn = layer.self_attn
            backend = attn.mla_attn
            backend = getattr(backend, 'mla_attn', backend)
            actual_heads = backend.num_heads
            row['full_attention_heads'] = actual_heads
            row['expected_full_attention_heads'] = expected_heads
            if actual_heads != expected_heads: errors.append(f'layer{layer.layer_idx}: attention heads')
            linears = {}
            for name, module in attn.named_modules():
                if hasattr(module, 'tp_size'):
                    linears[name] = {'tp_size': module.tp_size, 'tp_rank': getattr(module, 'tp_rank', None)}
                    if module.tp_size != 1: errors.append(f'layer{layer.layer_idx}.{name}: dense TP must be1')
            row['attention_parallel_modules'] = linears
        elif any(row[k] for k in ('attention_parameter_bytes','input_norm_parameter_bytes','post_attention_norm_parameter_bytes')):
            errors.append(f'layer{layer.layer_idx}: nonowner dense parameters')
        if layer.is_moe:
            expert = layer.mlp.experts.routed_experts
            prepared = expert.dots3_b12x_experts
            detail = {'layer':layer.layer_idx, 'tp_size':expert.exl3_tp_size,
                'tp_rank':expert.exl3_tp_rank, 'num_experts':prepared.num_experts,
                'hidden_size':prepared.hidden_size, 'intermediate_size':prepared.intermediate_size,
                'w1':_tensor(prepared._impl.w1_fp4), 'w2':_tensor(prepared._impl.w2_fp4)}
            if (detail['tp_size'] != group.world_size or detail['tp_rank'] != group.rank_in_group
                    or detail['num_experts'] != config.n_routed_experts
                    or detail['intermediate_size'] != config.moe_intermediate_size // group.world_size):
                errors.append(f'layer{layer.layer_idx}: expert tensor partition')
            if not detail['w1']['storage_bytes'] or not detail['w2']['storage_bytes']:
                errors.append(f'layer{layer.layer_idx}: missing prepared expert storage')
            routed.append(detail)
            row['router_parameter_bytes'] = _parameter_bytes(layer.mlp.gate)
            row['shared_expert_parameter_bytes'] = _parameter_bytes(layer.mlp.shared_experts)
            if not owner and (row['router_parameter_bytes'] or row['shared_expert_parameter_bytes']):
                errors.append(f'layer{layer.layer_idx}: nonowner router/shared parameters')
        rows.append(row)
    if [x['layer'] for x in routed] != expected_routed:
        errors.append('not all routed layers present on this rank')
    caches, storages = [], {}
    for name, module in vllm_config.compilation_config.static_forward_context.items():
        match = re.search(r'(?:^|\.)layers\.(\d+)\.', name)
        if not match: continue
        index = int(match.group(1))
        owner_rank = decoder.layers[0].plan.owners[index] if index < config.num_hidden_layers else decoder.layers[0].plan.output_owner
        cache = getattr(module, 'kv_cache', None)
        if cache is None: continue
        if owner_rank != group.rank_in_group: errors.append(f'{name}: nonowner registered cache')
        if not hasattr(cache, 'untyped_storage'):
            errors.append(f'{name}: unsupported cache representation')
            continue
        detail = _tensor(cache)
        detail.update(name=name, layer=index, owner_rank=owner_rank)
        if require_bound_cache and cache.numel() == 0: errors.append(f'{name}: KV not bound')
        caches.append(detail)
        if detail['storage_bytes']:
            storages[(detail['device'], detail['storage_pointer'])] = detail['storage_bytes']
    if require_bound_cache and not caches: errors.append('no bound owner caches')
    receipt = {'schema':'hybrid-owner-attestation-v1','rank':group.rank_in_group,
               'world_size':group.world_size,
               'multimodal_storage': {name: _module_storage(getattr(model, name, None))
                                      for name in ('visual', 'audio_tower')},
               'layers':rows,'routed_experts':routed,
               'caches':caches,'unique_kv_storage_bytes':sum(storages.values()),
               'passed':not errors,'errors':errors}
    return receipt


class HybridAttestationWorkerExtension:
    def hybrid_ownership_receipt(self):
        return attest_model(self.model_runner.model, self.vllm_config)


def write_startup_receipt(executor, path):
    """Engine-local, opt-in RPC after KV binding; no HTTP RPC surface needed."""
    import json
    import os
    from pathlib import Path
    receipts = executor.collective_rpc('hybrid_ownership_receipt')
    passed = bool(receipts) and all(receipt.get('passed') for receipt in receipts)
    result = {'schema': 'hybrid-owner-startup-attestation-v1', 'passed': passed,
              'workers': receipts}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(target)
    if not passed:
        raise RuntimeError(f'hybrid loaded ownership attestation failed: {target}')
