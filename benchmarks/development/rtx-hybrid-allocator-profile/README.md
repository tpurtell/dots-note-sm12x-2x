# Matched RTX allocator profile18/profile17

Both runs use image `e54f202f`. Raw before-create, before-profile and after-profile snapshots plus allocator counters are preserved.

| Cut | Rank1 before reserved−allocated GiB | After inactive split GiB | Active allocation growth GiB | Non-Torch GiB |
|---:|---:|---:|---:|---:|
| 18 | 0.840 | 3.025 | 0.495 | 1.124 |
| 17 | 0.830 | 0.805 | 0.496 | 1.124 |

Before-profile reserved−allocated is a free-reserved proxy; the exact inactive-split counter is captured after profiling. The much larger cut18 post-profile inactive split explains the reserved-memory difference while active allocation growth and non-Torch usage are similar. This does not identify allocation origin or prove a lifetime fix.

Profile17 completed12/12 coding requests naturally with static checks:
- C1: 162.567 median per-request decode tokens/s.
- C2: 131.845 median per-request decode tokens/s.
- C4: 96.695 median per-request decode tokens/s.

This is development evidence only. Lifetime-fix candidate results are separate and not claimed here.
