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


def _modules_storage(modules):
    modules = [m for m in modules if m is not None]
    if not modules:
        return {'present': False, 'parameter_bytes': 0, 'unique_storage_bytes': 0,
                'cuda_storage_bytes': 0, 'unique_storage_bytes_by_device': {}}
    parameters = list({id(p): p for m in modules for p in m.parameters()}.values())
    buffers = list({id(p): p for m in modules for p in m.buffers()}.values())
    storages = {}
    by_dtype = {}
    for tensor in parameters + buffers:
        detail = _tensor(tensor)
        storages[(detail['device'], detail['storage_pointer'])] = detail['storage_bytes']
        by_dtype[detail['dtype']] = by_dtype.get(detail['dtype'], 0) + tensor.numel() * tensor.element_size()
    by_device = {}
    for (device, _), size in storages.items():
        by_device[device] = by_device.get(device, 0) + size
    return {'present': True, 'cuda_storage_bytes': sum(size for device,size in by_device.items() if device.startswith('cuda')),
            'unique_storage_bytes_by_device': by_device, 'parameter_count': sum(p.numel() for p in parameters),
            'parameter_bytes': sum(p.numel()*p.element_size() for p in parameters),
            'buffer_bytes': sum(p.numel()*p.element_size() for p in buffers),
            'unique_storage_bytes': sum(storages.values()), 'logical_bytes_by_dtype': by_dtype}


def _module_storage(module):
    return _modules_storage([module])


def _owner_dense_storage(layer):
    modules = [layer.self_attn, layer.input_layernorm, layer.post_attention_layernorm]
    modules += ([layer.mlp.gate, layer.mlp.shared_experts] if layer.is_moe else [layer.mlp])
    return _modules_storage(modules)


def _parallel_module_evidence(module):
    from vllm.model_executor.layers.linear import ReplicatedLinear
    detail = {'tp_size': module.tp_size, 'tp_rank': getattr(module, 'tp_rank', None),
              'class': type(module).__name__}
    if isinstance(module, ReplicatedLinear):
        # ReplicatedLinear retains global tp_size metadata by default, but its
        # loader and forward use the entire weight and no TP collectives.
        detail.update(replicated=True, input_size=module.input_size,
                      output_size=module.output_size,
                      output_partition_sizes=list(module.output_partition_sizes),
                      loaded_weight=_tensor(module.weight))
        valid = (sum(module.output_partition_sizes) == module.output_size and
                 module.weight.numel() == module.input_size * module.output_size)
    else:
        valid = module.tp_size == 1
    return detail, valid


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
                    linears[name], valid = _parallel_module_evidence(module)
                    if not valid: errors.append(f'layer{layer.layer_idx}.{name}: dense projection must be full owner-local weight')
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
        row['owner_dense_storage'] = _owner_dense_storage(layer)
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
    mm_storage = {name: _module_storage(getattr(model, name, None))
                  for name in ('visual', 'audio_tower')}
    mm_plan = getattr(model, 'hybrid_mm_owners', None)
    if mm_plan is not None:
        limits = model.multimodal_config
        video_enabled = limits.get_limit_per_prompt('video') > 0
        for name, modality in (('visual', 'image'), ('audio_tower', 'audio')):
            enabled = video_enabled or limits.get_limit_per_prompt(modality) > 0
            expected = enabled and mm_plan[name] == group.rank_in_group
            state = mm_storage[name]
            state['expected_local_owner'] = expected
            if state['present'] != expected or (expected and state['cuda_storage_bytes'] <= 0):
                errors.append(f'{name}: tower presence/storage disagrees with owner plan')
    receipt = {'schema':'hybrid-owner-attestation-v1','rank':group.rank_in_group,
               'world_size':group.world_size,
               'multimodal_storage': mm_storage, 'multimodal_owner_plan': mm_plan,
               'layers':rows,'routed_experts':routed,
               'caches':caches,'unique_kv_storage_bytes':sum(storages.values()),
               'passed':not errors,'errors':errors}
    return receipt


def _boundary_storage(module, rank, errors, name):
    detail = _module_storage(module)
    detail['owner_rank'] = getattr(module, 'hybrid_owner', None)
    if getattr(module, '_hybrid_owned_boundary', False):
        owner = module.hybrid_owner == rank
        if owner:
            weight = getattr(module, 'weight', None)
            if weight is None:
                errors.append(f'{name}: missing owner boundary weight')
            else:
                detail['weight'] = _tensor(weight)
                if (module.tp_size != 1 or weight.ndim != 2 or
                        weight.shape[0] < module.num_embeddings or
                        weight.shape[1] != module.embedding_dim):
                    errors.append(f'{name}: boundary weight is not full owner table')
        elif detail['parameter_bytes'] or detail['cuda_storage_bytes']:
            errors.append(f'{name}: nonowner boundary storage remains')
    return detail


def attest_draft_and_boundaries(model, draft, rank):
    language = getattr(model, 'language_model', model)
    errors = []
    boundary = {'target_embedding': _boundary_storage(language.model.embed_tokens,rank,errors,'target_embedding'),
                'target_head': _boundary_storage(language.lm_head,rank,errors,'target_head')}
    layers = []
    if draft is not None:
        boundary['draft_embedding'] = _boundary_storage(draft.model.embed_tokens,rank,errors,'draft_embedding')
        for name, layer in draft.model.layers.items():
            block = layer.mtp_block
            dense = _owner_dense_storage(block)
            auxiliary = _modules_storage([layer.enorm,layer.hnorm,layer.eh_proj,layer.shared_head.norm])
            local_owner = block.owner == rank
            row = {'layer':int(name),'owner_rank':block.owner,'local_owner':local_owner,
                   'dense_storage':dense,'normalization_projection_storage':auxiliary,
                   'head_is_shared_with_target':layer.shared_head.head is language.lm_head}
            if local_owner:
                backend = block.self_attn.mla_attn
                backend = getattr(backend,'mla_attn',backend)
                row['attention_heads'] = backend.num_heads
                if backend.num_heads != 64:
                    errors.append(f'draft layer{name}: Dots MTP requires full64 SWA heads')
                if dense['cuda_storage_bytes'] <= 0 or auxiliary['cuda_storage_bytes'] <= 0:
                    errors.append(f'draft layer{name}: missing owner dense weights')
            elif dense['parameter_bytes'] or auxiliary['parameter_bytes'] or dense['cuda_storage_bytes'] or auxiliary['cuda_storage_bytes']:
                errors.append(f'draft layer{name}: nonowner dense weights remain')
            if not row['head_is_shared_with_target']:
                errors.append(f'draft layer{name}: head is not shared with target')
            layers.append(row)
    return {'boundary_storage':boundary,'draft_layers':layers},errors


class HybridAttestationWorkerExtension:
    def hybrid_ownership_receipt(self):
        receipt = attest_model(self.model_runner.model, self.vllm_config)
        speculator = getattr(self.model_runner, 'speculator', None)
        if speculator is None:
            speculator = getattr(self.model_runner, 'drafter', None)
        draft = getattr(speculator, 'model', None)
        extra, errors = attest_draft_and_boundaries(self.model_runner.model, draft, receipt['rank'])
        receipt.update(extra)
        if self.vllm_config.speculative_config is not None and not extra['draft_layers']:
            errors.append('configured MTP model missing from runtime attestation')
        receipt['errors'].extend(errors)
        receipt['passed'] = not receipt['errors']
        budget = getattr(self, 'available_kv_cache_memory_bytes', None)
        receipt['available_kv_cache_memory_bytes'] = None if budget is None else int(budget)
        receipt['memory_profile_bytes'] = {
            name: (None if getattr(self, name, None) is None else int(getattr(self, name)))
            for name in ('total_consumed', 'peak_activation_memory',
                         'cudagraph_memory_estimate', 'requested_memory')
        }
        model_memory = getattr(self.model_runner, 'model_memory_usage', None)
        receipt['memory_profile_bytes']['model_memory_usage'] = None if model_memory is None else int(model_memory)
        receipt['memory_profile_note'] = 'peak_activation_memory includes applied CUDA graph estimate; do not add cudagraph_memory_estimate again'
        receipt['cache_admission_inputs'] = {
            'max_model_len': self.vllm_config.model_config.max_model_len,
            'max_in_flight_tokens': self.vllm_config.max_in_flight_tokens,
            'block_size': self.vllm_config.cache_config.block_size,
        }
        return receipt


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
