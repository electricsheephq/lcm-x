# D2 instrument operating-point fixture proof

Claim class: advisory (instrument code with local fixtures).

| SD | loss | metric | target | observed | simulation SE | proof |
| --- | --- | --- | --- | --- | --- | --- |
| 0.067728807 | 0.00 | A/A acceptance | 92.04% | 91.977% | 0.156 points | within 3 SE; observations.json |
| 0.093770528 | 0.00 | A/A acceptance | 74.56% | 74.530% | 0.251 points | within 3 SE; observations.json |
| 0.067728807 | 0.05 | non-acceptance | 90.00% | 90.097% | 0.173 points | within 3 SE; observations.json |
| 0.093770528 | 0.05 | non-acceptance | 90.00% | 90.087% | 0.173 points | within 3 SE; observations.json |

30,000 independent Gaussian comparisons per row; RNG seed 1538; 3 seeds, 9 runs per arm/seed; pooled-t df 48, one-sided 90% lower bound. Complete-valid cp-304 NI rule only, matching the calibration Gaussian model. No placebo, A/A point cap, provider failure, replacement or elapsed components. These are simulation results, not measured A/A runs.

## Exact reproduction

```sh
python -m pytest bench/instruments/compaction_probe/track_s/decision/tests/test_d2_calibrated.py::test_operating_points -q -s
```

Measured code: [0481889e](https://github.com/electricsheephq/lcm-x/tree/0481889e356b24699ba25a1717879e2689928225). Raw test stdout: raw-output.txt. Numeric snapshot: observations.json. Each row passes its 3-simulation-SE assertion.

Not proven: empirical coverage, real D2 facts kept, prospective A/A, throughput, release or customer readiness.

T1: not registered; the lane permits changes only inside decision/. T2: not published; no result cleanup is authorized or performed. The public fixture card and inputs accompany this instrument PR.
