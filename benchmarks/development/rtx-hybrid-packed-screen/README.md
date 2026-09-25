# Packed hybrid matched screen

Development evidence: 40/40 API cases and12/12 natural coding completions/static checks. Prefix and passed loaded ownership receipts preserved. Same script/task hashes and request settings; packed uses127.0.0.1 while controls usedlocalhost on the same local port.

| C | Native TP2 | Unpacked hybrid | Packed hybrid | Packed vs native |
|---:|---:|---:|---:|---:|
| 1 | 184.683 | 154.955 | 158.793 | -14.02% |
| 2 | 135.084 | 130.538 | 127.573 | -5.56% |
| 4 | 101.626 | 97.973 | 95.762 | -5.77% |

Rates are median per-request decode tokens/s, excluding initial SSE burst and including reasoning. Packed improves C1 versus unpacked but does not recover native TP2 speed; C2/C4 are slower in this small one-run screen. Variable stochastic output lengths affect completion latency. No general quality or final performance claim is made.

Raw files are losslessly compressed and hashed. Historical container states do not describe current liveness. Native and unpacked evidence lives in sibling development archives.
