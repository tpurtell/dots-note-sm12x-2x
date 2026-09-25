# Optional Spark vLLM RoCE adapter

`b12x_roce_all_reduce.py` and `port_roce.py` are prepared for a controlled
whole-model A/B. They have passed CPU contract/source-patch checks, **not GPU
adapter or serving qualification**. The underlying transport probe passed on
Rhea/Moa with B12x commit`c5e23d830c3d1e76be56a5df290d13e30bc66702`; that is a
prerequisite, not proof of this integration.

## Image and launcher integration points

The native Spark runtime Dockerfile now applies these additions after the existing vLLM port:

```dockerfile
COPY serving/b12x_roce_all_reduce.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/b12x_roce_all_reduce.py
COPY serving/port_roce.py /tmp/port_roce.py
RUN python3 /tmp/port_roce.py /usr/local/lib/python3.12/dist-packages/vllm
```

This source patch expects the native vLLM0.30 custom-allreduce slot. Do not
combine it with the RTX `port_pcie.py` on the same image. All source anchors
are verified before writing any file; already-applied or incompatible sources
are rejected. The Spark launcher now forwards the variables below with RoCE disabled by default.

Pass these environment variables through the Spark launcher when testing:

| Variable | Default | Meaning |
|---|---|---|
| `DOTS3_B12X_ROCE` | `0` | Optional TP2 adapter; set identically on both ranks |
| `DOTS3_B12X_ROCE_ROWS` | `1-64` | Bounded row set; alternatively comma-separated integers within1..64 |
| `DOTS3_B12X_ROCE_EAGER` | `0` | Graph-only by default; `1` also allows eager calls |
| `B12X_ROCE_HCA` | library discovery | Local HCA names in matched rank order |
| `B12X_ROCE_GID_INDEX` | `3` or NCCL override | HCA GID index |
| `B12X_ROCE_SPIN_LIMIT` | library20M default | Bounded peer wait |
| `B12X_ROCE_CACHE_DIR` | base image`/opt/dots3/native-roce`; release wrapper runtime mount | Native proxy cache inherited from the image |

First match the qualified single-link probe with
`B12X_ROCE_HCA=roceP2p1s0f0 B12X_ROCE_GID_INDEX=3`. Two-link striping is a
separate experiment; the successful probe receipt used the single fast link.
The adapter requires custom-allreduce to remain enabled and only occupies the
exact `tp` communicator's slot. ETP/EP/DP/PP communicators do not create extra
RoCE runtimes. TP2 on two distinct SM121 integrated-GPU hosts is enforced via
CPU-group configuration exchange, GPU identity and kernel boot IDs.

## Ownership and failure behavior

One public prepared declaration covers BF16 width5120 over the admitted row
set. Native kernels accept payload length/grid at launch, so preparing64
separate copies would duplicate ownership. A CPU rendezvous after compilation
prevents an early rank's initial GPU collective from exhausting its peer wait.
Compilation stays in the worker process to avoid a large compiler-process pool.

The default640KiB payload capacity allocates about3.75MiB of transport slots,
1.25MiB of alignment buffers and1.25MiB of retained primer input/output, plus
small flags and executable metadata. Gather capacity is explicitly16bytes.
Each model collective returns its **own** `torch.empty_like` output; sharing one
output among layers would overwrite live residuals. CUDA graph capture retains
stable addresses for these outputs in its graph pool. There are no retained
per-layer transport arenas.

Unsupported shapes/dtypes retain native dispatch while the runtime is healthy.
Poison is checked before eligibility, including before any alternate backend can
bypass the custom slot. Both model-runner generations check health after their
existing output-copy event synchronization and before publishing results; this
adds no GPU synchronization. Non-output TP ranks may not unwrap an async output,
and vLLM can swallow their RPC exceptions. Therefore a50ms host-only watchdog
also observes the mapped error words/proxy liveness and **terminates a poisoned
worker**, allowing the executor's worker-death handling to end the engine.
No post-poison fallback can return untrusted data.

Normal destruction stops the watchdog, waits for GPU work, releases the
preparation session, closes the transport, then allows vLLM to destroy native
NCCL and its process groups. A fatal poisoned-worker exit cannot perform normal
Python cleanup; OS teardown releases that process's registrations.

## Qualification still required

1. Extend the standalone matrix with`--rows 1 2 4 8 16 32 48 64` on both hosts,
   using fresh receipts and the same corrected library. The existing probe's1MiB
   cap supports these shapes. Verify changed-input graphs at48/64 before
   enabling the full range in serving. Every other intermediate row is covered
   by the length-parametric kernel but has not independently been measured.
2. Validate the adapter itself with native/unsupported fallback, changed inputs,
   graphs, consecutive model-layer outputs (no alias), and intentional proxy
   failure in disposable workers. The adapter CPU checks do not establish
   transport failure behavior on GPUs.
3. Compare native and RoCE on the same image/configuration at C1/C16, identical
   MTP/context/FP8 settings. Re-run tool, grammar, multimodal, prefix, retrieval
   and content contracts. Record host memory and full startup/runtime logs.
4. The probe's NCCL graph timing was310–447µs, versus27–136µs eager. Single-call
   graph overhead may not represent a model graph containing many collectives.
   This anomaly is not evidence of a corresponding whole-model speedup. A
   follow-up graph with repeated dependent collectives per replay can separate
   per-replay overhead from per-collective cost; model A/B remains decisive.

## Native artifact packaging

The successful probe produced this **72,312-byte ARM64 shared library** on both
hosts:

```text
/home/tj/dots-note-work/recipe/.cache/roce-probe/proxy/roce_proxy-a35f54cf75d6abf2.so
```

Inside those probe containers the same file was under
`/probe/.cache/roce-probe/proxy/`. This is the C/libibverbs proxy; the CuTe GPU
executables are separate B12x compile-cache `.o` artifacts already handled by
the release exporter. No additional C++ extension cache was found or required.

For a native Spark base build, a small dedicated image directory can hold the
proxy without mounting over it:

```dockerfile
ENV B12X_ROCE_CACHE_DIR=/opt/dots3/native-roce
RUN python3 -c 'from b12x.comm.roce._proxy import load; load()'
```

Run this after copying the final B12x source. It needs a C compiler and verbs
headers but no GPU; the private helper is the library's existing native build
path. The shared-library filename incorporates the C source hash. Validate
native loading during the build and actual transport use during qualification.

`serving/release/cache_bundle.py` now optionally exports **only** that configured
proxy directory, verifies a`roce_proxy-*.so` exists, records hashes and marks the
manifest. It refuses to export an enabled RoCE profile without a compiled proxy.
The release wrapper seeds it into the mounted
`/root/.cache/vllm-runtime/b12x/roce` directory and sets the matching environment
variable. It fingerprints relevant C/C++/CUDA headers/sources as well as Python
sources. Export fresh bundles after this fingerprint change. This avoids
shipping unrelated user caches and keeps the native image architectures apart.

Release profiles now explicitly pin `b12x_roce`, `b12x_roce_eager` and the admitted `b12x_roce_rows`; inherited environment values cannot enable an unqualified transport. Older RTX profiles may omit these Spark-only fields and retain RoCE disabled.
