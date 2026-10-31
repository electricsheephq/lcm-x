# #1586 existing duty probe — imported receipt

Claim class: advisory; 2-vCPU floor surrogate only. No benchmark was run by this PR lane.

| Receipt | Multiplier | Shape | CPU mean / p95, max across pair (cores) | Probe verdict |
| --- | --- | --- | --- | --- |
| [39888055698](https://github.com/electricsheephq/lcm-x/actions/runs/39888055698) | 1.0 | mixed | 0.50 / 0.57 | FAIL coverage |
| [39888063665](https://github.com/electricsheephq/lcm-x/actions/runs/39888063665) | 1.0 | short | 0.50 / 0.52 | within bars |
| [39888076007](https://github.com/electricsheephq/lcm-x/actions/runs/39888076007) | 0.82 | mixed | 0.55 / 0.63 | PASS probe |
| [39888085612](https://github.com/electricsheephq/lcm-x/actions/runs/39888085612) | 0.82 | short | 0.55 / 0.58 | within bars |

Headline: maximum measured mean CPU over the four 0.82 cells = 0.5505486880915895 cores (n=4 cells; one run per shape, no repeated-run variance estimate).
Re-score: select `idle_multiplier == 0.82` from the evidence JSON, then take the maximum of the four `cpu.mean_cores` values.
Inputs: `1586-idle-duty-probe-2026-10-23.json` (public-safe per-cell extraction with source hashes).
T1: `bench/scoreboard/results.d/local-idle-duty-floor-probe-2026-10-23.json` (same PR).
T2 incomplete: original full outputs were not supplied to this lane; no publication or cleanup is authorized.
Not proven: completed 72-hour catch-up, full-floor recall latency, customer-box behavior, or Mac 24-hour behavior.
