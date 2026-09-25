# Compact DSA cache component qualification

B12x commit `c963d8f7c98792a026eaf81a96a72d01b4aa0047` adds an explicit
`physical_record_width` to strided sparse MLA. Its default remains 1088;
compact 576-byte FP8 records retain logical QK576/V512 geometry. Width is
bound into the native plan, compilation query and cache validation.

Both physical RTX GPUs passed all 16 selected TP2 kernel cases, covering both
record formats, oracle output/LSE, invalid indices, ordinary interleaving, the
initial 1102-record mixed-block fixture with a 448-record layer offset, changed-input
CUDA graphs and addresses beyond 2 GiB. Both also passed actual vLLM adapter
and SWA cache-gather gates, including ragged boundaries and graph replay.

Five CPU metadata tests passed before GPU qualification. The final test file
tightens one CompileJob assertion after that CPU run; this assertion was not
rerun. Kernel source bytes are unchanged and match the candidate build hashes.

The [manifest](manifest.json) records exact image/source/environment/commands,
per-device results, and lossless archive hashes. GPU pytest's only warning was
its optional cache failing to write into the read-only source mount. Historical
build metadata is retained verbatim; its pre-test status fields are superseded
by these completed receipts.

The allocator CPU receipt measures the proposed 524,288-context cache layout
at about 4.902 GiB per rank versus 9.086 GiB for the old layout. Full-model
startup, prefix reuse, retrieval and exact native-context boundary remain
separate qualification gates; these component results do not establish them.

Subsequent full-model inspection identified MTP as SWA: the real compact pool
uses 589,248 bytes per block (1,023 records), with 13 DSA, 13 indexer and
34 SWA layers. The initial 1,102-record fixture remains a stride stress case;
the actual geometry has now passed its targeted component gate on both RTX
GPUs: three kernel tests plus actual adapter and gather checks per GPU. The
fixture-only follow-up is `cb70291a52d84ae3dd03558d568941bf31d369a5`; kernel
implementation bytes remain those of `c963d8f7`. Corrected allocator accounting
uses 4.5642 GiB per rank at 524,288 context. The initial 4.902 GiB figure above
is retained as historical CPU evidence, not the actual-runtime estimate.

The current CPU indexer workspace check also passed: default behavior,
4-context allocation/planner agreement, full-context and more-than-four-request
chunk coverage. Its saved stdout and source hash are in the archive.
