"""Explicit fork/join lifetime for owner auxiliary work during expert TP.

Only owner-local shared-expert work uses this side stream. Routed B12x work
and its shared scratch arena stay serialized on the original stream. Every
fork joins before the layer returns, retaining input/output references through
that join; no tensors from a completed layer remain live on the side stream.
"""
import torch

_STREAMS={}


def prepared_stream(device):
    key=(device.type,device.index)
    if key not in _STREAMS:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('shared-expert side stream must be created before capture')
        _STREAMS[key]=torch.cuda.Stream(device=device)
    return _STREAMS[key]


class PendingShared:
    def __init__(self,stream,inputs,output):
        self.stream=stream
        self.inputs=inputs
        self.output=output
    def join(self):
        torch.cuda.current_stream(self.output.device).wait_stream(self.stream)
        return self.output


def launch_shared(function, inputs):
    stream=prepared_stream(inputs.device)
    main=torch.cuda.current_stream(inputs.device)
    stream.wait_stream(main)
    with torch.cuda.stream(stream):
        output=function(inputs)
    return PendingShared(stream,inputs,output)
