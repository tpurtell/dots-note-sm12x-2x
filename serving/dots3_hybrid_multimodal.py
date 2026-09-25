"""Owner-local native Dots vision/audio towers with replicated output embeddings.

Native towers already use local (unsharded) linears. The nonowner does not
construct or load that tower. Encoder execution stays eager; decoder graphs
are unaffected. Both TP workers receive the same encoder-cache entries.
"""
import os
import torch
from vllm.distributed import get_tp_group
from vllm.distributed.hybrid_parallel import PyNcclOwnerTransport


def owners():
    value = os.environ.get('VLLM_HYBRID_MM_OWNERS', '')
    if not value:
        return None
    if not os.environ.get('VLLM_HYBRID_LAYER_PARTITION'):
        raise ValueError('MM ownership requires hybrid layer ownership')
    result = tuple(int(x) for x in value.split(','))
    group = get_tp_group()
    if len(result) != 2 or any(not 0 <= rank < group.world_size for rank in result):
        raise ValueError('MM owners must be vision,audio worker ranks')
    return dict(zip(('visual', 'audio_tower'), result))


def create_owned_tower(factory, name):
    plan = owners()
    if plan is None or get_tp_group().rank_in_group == plan[name]:
        return factory()
    # Native constructor's None mapper skips all checkpoint tensors for this
    # tower before materialization. Patched processing supplies remote outputs.
    return None


def broadcast_embeddings(*, transport, owner, item_count, device, encode):
    """Transfer variable per-item lengths and contiguous native embedding bytes.

    The small metadata transfer intentionally synchronizes once per encoder
    batch. It is outside decoder CUDA graphs and handles native audio lengths
    without duplicating/respecifying the encoder's padding/downsampling math.
    """
    if item_count == 0:
        return ()
    if device.type == 'cuda' and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('owner MM encoders require eager execution')
    dtypes = (torch.bfloat16, torch.float16, torch.float32)
    result = None
    metadata = torch.empty(item_count + 2, device=device, dtype=torch.int64)
    owner_error = None
    if transport.rank == owner:
        try:
            outputs = tuple(encode())
            if len(outputs) != item_count or any(x.ndim != 2 for x in outputs):
                raise ValueError('native encoder output count/shape mismatch')
            dtype, hidden = outputs[0].dtype, outputs[0].shape[1]
            if dtype not in dtypes or any(x.dtype != dtype or x.shape[1] != hidden for x in outputs):
                raise ValueError('inconsistent native encoder output dtype/width')
            metadata.copy_(torch.tensor([hidden, dtypes.index(dtype)] + [x.shape[0] for x in outputs],
                                        dtype=torch.int64, device=device))
            result = torch.cat(outputs, dim=0)
        except Exception as error:
            owner_error = error
            metadata.fill_(-1)
    transport.broadcast(metadata, owner)
    if owner_error is not None:
        raise RuntimeError('owner MM encoder failed') from owner_error
    hidden, dtype_code, *lengths = metadata.tolist()
    if hidden <= 0 or dtype_code not in range(len(dtypes)) or any(x < 0 for x in lengths):
        raise ValueError('invalid encoder output metadata')
    if result is None:
        result = torch.empty((sum(lengths), hidden), device=device, dtype=dtypes[dtype_code])
    if result.numel():
        transport.broadcast(result, owner)
    return result.split(lengths)


def install(namespace):
    if not os.environ.get('VLLM_HYBRID_MM_OWNERS'):
        return
    cls = namespace['Dots3NoteForCausalLM']
    original_init = cls.__init__
    image, audio = cls._process_image_input, cls._process_audio_input
    def init(self, *, vllm_config, prefix=''):
        mm = vllm_config.model_config.multimodal_config
        if mm.mm_encoder_tp_mode != 'weights':
            raise ValueError('owner MM encoding requires identical TP batches (weights mode)')
        if vllm_config.compilation_config.cudagraph_mm_encoder:
            raise ValueError('owner MM encoding currently requires eager encoder execution')
        original_init(self, vllm_config=vllm_config, prefix=prefix)
        self.hybrid_mm_owners = owners()
        self.hybrid_mm_transport = PyNcclOwnerTransport(get_tp_group())
    def process_image(self, pixel_values, image_grid_thw):
        return broadcast_embeddings(transport=self.hybrid_mm_transport,
            owner=self.hybrid_mm_owners['visual'], item_count=image_grid_thw.shape[0],
            device=pixel_values.device, encode=lambda: image(self,pixel_values,image_grid_thw))
    def process_audio(self, audio_values, audio_lengths):
        return broadcast_embeddings(transport=self.hybrid_mm_transport,
            owner=self.hybrid_mm_owners['audio_tower'], item_count=audio_lengths.numel(),
            device=audio_values.device, encode=lambda: audio(self,audio_values,audio_lengths))
    cls.__init__ = init
    cls._process_image_input = process_image
    cls._process_audio_input = process_audio
