# RTX native v2 parent warm

Accepted for compiled-cache export only; **release qualification remains false**. Full native image from frozen recipe `76fb772` and B12x `bf5677c6`.

40/40 API cases, prefix/XGrammar, both MM examples, 7/7 content contracts, C8/C16 clients and exact 524,032+256 context boundary completed. Single-run client rates were 469.82/670.41 tokens/s. Boundary TTFT 217.005 s, decode 206.56 tokens/s. These are warm receipts, not final release measurements.

The API omitted boundary cached-token accounting (`null`); the unique prompt nonce/hash is retained. Recovered allocator retries during prefill are preserved, with successful boundary completion. Full published-wrapper qualification, including hard-mode tools, remains required.

`manifest.json` binds each original and lossless gzip artifact; `summary.json` records actual profile, image, source and ownership proof.
