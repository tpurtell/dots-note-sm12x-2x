#!/usr/bin/env python3
"""GPU component gate for fused routing copies; no model performance claim."""
import argparse
import json
import statistics

import torch
from hybrid_parallel import allocate_packed_routing
from hybrid_pack_kernel import pack_routing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    device = torch.device('cuda', args.device)
    torch.manual_seed(712)
    records = []
    geometries = [(5120, r, s) for r in (1, 2, 4, 8, 16, 512)
                  for s in (False, True)] + [(3, 7, True)]
    cases = [(h, r, s, torch.int32, torch.float32) for h, r, s in geometries]
    cases += [(5120, r, True, torch.int64, dtype)
              for r in (1, 4, 8, 16, 512) for dtype in (torch.float32, torch.bfloat16, torch.float16)]
    for hidden, rows, strided, id_dtype, weight_dtype in cases:
        factor = 2 if strided else 1
        a = torch.randn(rows, hidden * factor, device=device, dtype=torch.bfloat16)[:, ::factor]
        ids = torch.randint(0, 256, (rows, 8 * factor), device=device, dtype=id_dtype)[:, ::factor]
        weights = torch.randn(rows, 8 * factor, device=device, dtype=weight_dtype)[:, ::factor]
        dst = allocate_packed_routing(capacity=512, hidden=hidden, topk=8,
                                     dtype=a.dtype, device=device).active(rows)
        def reference():
            dst.activation.copy_(a)
            dst.route_ids.copy_(ids)
            dst.route_weights.copy_(weights)
        def fused():
            pack_routing(dst, a, ids, weights)
        def verify():
            for got, wanted in zip((dst.activation, dst.route_ids, dst.route_weights), (a, ids, weights)):
                assert torch.equal(got.contiguous().view(torch.uint8), wanted.to(got.dtype).contiguous().view(torch.uint8))
        fused()
        verify()
        graphs = {}
        for name, function in [('reference', reference), ('fused', fused)]:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): function()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): function()
            graphs[name] = graph
        for replay in range(3):
            a.normal_()
            ids.random_(0, 256)
            weights.normal_()
            # Bit copies preserve signed zero and IEEE exceptional values too.
            weights[0, :4] = torch.tensor([0., -0., float('inf'), float('nan')], device=device)
            graphs['fused'].replay()
            verify()
        timing = {}
        for name, graph in graphs.items():
            samples = []
            for _ in range(5):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100): graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 10)
            timing[name] = {'samples_us': samples, 'median_us': statistics.median(samples)}
        row = dict(rows=rows, hidden=hidden, strided=strided, id_dtype=str(id_dtype),
                   weight_dtype=str(weight_dtype), passed=True, timings=timing)
        records.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({'schema': 'hybrid-fused-pack-component-v1', 'device': args.device,
                      'passed': True, 'cases': len(records)}), flush=True)


if __name__ == '__main__':
    main()
