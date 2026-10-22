# Previous-GA differential — PASS

[GitHub Actions run](https://github.com/electricsheephq/lcm-x/actions/runs/39854613143), exact candidate158191079b4ec61e676f9d03ea3f34069543e8fb.

`LCM_DIFFERENTIAL=1 LCM_DIFFERENTIAL_OUT=$RUNNER_TEMP/differential python -m pytest tests/differential -q -rx -p no:randomly`

Observed:`97 passed, 2 xfailed in 183.61s (0:03:03)`. Existing xfails #1225 objective exceeds room and #1247 history coverage remain.

18 reports: generated60, projection1000, 16 named single shapes (15 vs v0.28.0 bare,1 managed v0.26.2). Total1076 reported cases. All layer/family/oracle violation counts `{}`; unexplained events0; accepted deltas0; all four existing delta absorption counts0. No new accepted delta or blanket exception.

## generated families and windows

```json
{
  "families": {
    "parallel/1004/user": 1,
    "sequential/2400/tool": 1,
    "sequential/1044/user": 1,
    "mixed/12000/tool": 2,
    "parallel/2400/user": 2,
    "mixed/48001/user": 4,
    "parallel/1044/tool": 1,
    "sequential/48001/tool": 3,
    "parallel/48001/user": 2,
    "parallel/1004/tool": 1,
    "mixed/12000/user": 2,
    "parallel/1024/tool": 1,
    "mixed/1044/user": 1,
    "parallel/1200/assistant": 1,
    "parallel/6000/tool": 1,
    "sequential/12000/user": 3,
    "mixed/12000/assistant": 1,
    "sequential/12000/assistant": 2,
    "sequential/1199/assistant": 2,
    "sequential/1200/user": 1,
    "mixed/2400/tool": 1,
    "sequential/1024/tool": 1,
    "mixed/1044/assistant": 1,
    "parallel/3000/user": 1,
    "parallel/48001/tool": 1,
    "sequential/12000/tool": 4,
    "mixed/2400/assistant": 1,
    "parallel/1024/user": 1,
    "mixed/1200/tool": 1,
    "mixed/3000/tool": 1,
    "parallel/48001/assistant": 1,
    "mixed/1201/assistant": 1,
    "parallel/6000/assistant": 1,
    "sequential/2401/user": 1,
    "sequential/1004/tool": 1,
    "parallel/3000/assistant": 1,
    "mixed/1199/assistant": 1,
    "parallel/1004/assistant": 1,
    "mixed/48001/assistant": 1,
    "parallel/12000/assistant": 1,
    "mixed/1004/user": 1,
    "sequential/3000/assistant": 1,
    "mixed/1199/tool": 1,
    "mixed/2401/assistant": 1
  },
  "windows": {
    "128000": 11,
    "96000": 11,
    "1000000": 8,
    "200000": 10,
    "64000": 6,
    "16000": 4,
    "32000": 10
  }
}
```

## projection families and windows

```json
{
  "families": {
    "p34": 1000
  },
  "windows": {
    "32768": 1000
  }
}
```

## Per-shape summaries

### generated

# Differential report
Baseline: `709099b3a397a7893a7abae8802ebe7bf605f79e`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.28.0`; canonical tag: `v0.28.0`.
Baseline config: bare.
Cases: 60; seed: 1227; unexplained events: 0.
Inherited user marks: 0 cases; 0 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'public/ga': {'wall_seconds': 24.599723369000003, 'mean_case_seconds': 0.40461465714999795}, 'public/candidate': {'wall_seconds': 24.637873153999976, 'mean_case_seconds': 0.4050481502833335}}
Missing seams: {'public/ga': [], 'public/candidate': []}

## Unexplained


### i1107_first_sight_stub

# Differential report
Baseline: `709099b3a397a7893a7abae8802ebe7bf605f79e`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.28.0`; canonical tag: `v0.28.0`.
Baseline config: bare.
Cases: 1; seed: 1227; unexplained events: 0.
Inherited user marks: 0 cases; 0 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'public/ga': {'wall_seconds': 0.31404893600000605, 'mean_case_seconds': 0.12949157200000627}, 'public/candidate': {'wall_seconds': 0.3050908820000018, 'mean_case_seconds': 0.13597035500001198}}
Missing seams: {'public/ga': [], 'public/candidate': []}

## Unexplained


### i1139_historical_sweep_32k

# Differential report
Baseline: `709099b3a397a7893a7abae8802ebe7bf605f79e`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.28.0`; canonical tag: `v0.28.0`.
Baseline config: bare.
Cases: 1; seed: 1227; unexplained events: 0.
Inherited user marks: 0 cases; 0 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'public/ga': {'wall_seconds': 2.2703108540000017, 'mean_case_seconds': 2.086165529999988}, 'public/candidate': {'wall_seconds': 2.4246690150000063, 'mean_case_seconds': 2.2414048459999947}}
Missing seams: {'public/ga': [], 'public/candidate': []}

## Unexplained


### i1139_small_window_128k

# Differential report
Baseline: `709099b3a397a7893a7abae8802ebe7bf605f79e`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.28.0`; canonical tag: `v0.28.0`.
Baseline config: bare.
Cases: 1; seed: 1227; unexplained events: 0.
Inherited user marks: 0 cases; 0 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'public/ga': {'wall_seconds': 11.697599106000013, 'mean_case_seconds': 11.478183825000002}, 'public/candidate': {'wall_seconds': 11.708963165, 'mean_case_seconds': 11.462109470000001}}
Missing seams: {'public/ga': [], 'public/candidate': []}

## Unexplained


### i1139_small_window_32k

# Differential report
Baseline: `709099b3a397a7893a7abae8802ebe7bf605f79e`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.28.0`; canonical tag: `v0.28.0`.
Baseline config: bare.
Cases: 1; seed: 1227; unexplained events: 0.
Inherited user marks: 0 cases; 0 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'public/ga': {'wall_seconds': 5.324926782000006, 'mean_case_seconds': 5.112641729000018}, 'public/candidate': {'wall_seconds': 5.135327059000019, 'mean_case_seconds': 4.947036937000007}}
Missing seams: {'public/ga': [], 'public/candidate': []}

## Unexplained


### i1139_small_window_40k

# Differential report
Baseline: `709099b3a397a7893a7abae8802ebe7bf605f79e`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.28.0`; canonical tag: `v0.28.0`.
Baseline config: bare.
Cases: 1; seed: 1227; unexplained events: 0.
Inherited user marks: 0 cases; 0 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'public/ga': {'wall_seconds': 8.554517492000002, 'mean_case_seconds': 8.368116399000002}, 'public/candidate': {'wall_seconds': 8.934265414999999, 'mean_case_seconds': 8.746764890999998}}
Missing seams: {'public/ga': [], 'public/candidate': []}

## Unexplained


### i1139_small_window_48k

# Differential report
Baseline: `709099b3a397a7893a7abae8802ebe7bf605f79e`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.28.0`; canonical tag: `v0.28.0`.
Baseline config: bare.
Cases: 1; seed: 1227; unexplained events: 0.
Inherited user marks: 0 cases; 0 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'public/ga': {'wall_seconds': 5.926044088000026, 'mean_case_seconds': 5.739050625000004}, 'public/candidate': {'wall_seconds': 5.908955645999981, 'mean_case_seconds': 5.722223764999995}}
Missing seams: {'public/ga': [], 'public/candidate': []}

## Unexplained


### i1139_small_window_64k

# Differential report
Baseline: `709099b3a397a7893a7abae8802ebe7bf605f79e`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.28.0`; canonical tag: `v0.28.0`.
Baseline config: bare.
Cases: 1; seed: 1227; unexplained events: 0.
Inherited user marks: 0 cases; 0 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'public/ga': {'wall_seconds': 3.875846418000009, 'mean_case_seconds': 3.686884851000002}, 'public/candidate': {'wall_seconds': 3.955346775999999, 'mean_case_seconds': 3.7624741719999975}}
Missing seams: {'public/ga': [], 'public/candidate': []}

## Unexplained


### i1220_parallel_fleet_n128

# Differential report
Baseline: `709099b3a397a7893a7abae8802ebe7bf605f79e`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.28.0`; canonical tag: `v0.28.0`.
Baseline config: bare.
Cases: 1; seed: 1227; unexplained events: 0.
Inherited user marks: 0 cases; 0 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'public/ga': {'wall_seconds': 0.33450848200000394, 'mean_case_seconds': 0.16845426500000826}, 'public/candidate': {'wall_seconds': 0.3294238370000073, 'mean_case_seconds': 0.15913854499999047}}
Missing seams: {'public/ga': [], 'public/candidate': []}

## Unexplained


### i1221_historical_assembly_32k

# Differential report
Baseline: `709099b3a397a7893a7abae8802ebe7bf605f79e`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.28.0`; canonical tag: `v0.28.0`.
Baseline config: bare.
Cases: 1; seed: 1227; unexplained events: 0.
Inherited user marks: 0 cases; 0 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'public/ga': {'wall_seconds': 0.22516120299999898, 'mean_case_seconds': 0.059744504999997616}, 'public/candidate': {'wall_seconds': 0.2258758890000081, 'mean_case_seconds': 0.05968657299999336}}
Missing seams: {'public/ga': [], 'public/candidate': []}

## Unexplained


### i1221_objective_default_32k

# Differential report
Baseline: `709099b3a397a7893a7abae8802ebe7bf605f79e`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.28.0`; canonical tag: `v0.28.0`.
Baseline config: bare.
Cases: 1; seed: 1227; unexplained events: 0.
Inherited user marks: 0 cases; 0 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'public/ga': {'wall_seconds': 0.29805411199998844, 'mean_case_seconds': 0.10356859200001622}, 'public/candidate': {'wall_seconds': 0.2831733799999938, 'mean_case_seconds': 0.09925756700002353}}
Missing seams: {'public/ga': [], 'public/candidate': []}

## Unexplained


### i1225_historical_assembly_16k

# Differential report
Baseline: `709099b3a397a7893a7abae8802ebe7bf605f79e`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.28.0`; canonical tag: `v0.28.0`.
Baseline config: bare.
Cases: 1; seed: 1227; unexplained events: 0.
Inherited user marks: 0 cases; 0 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'public/ga': {'wall_seconds': 0.22763057200000958, 'mean_case_seconds': 0.06197042000002284}, 'public/candidate': {'wall_seconds': 0.22543074700001853, 'mean_case_seconds': 0.06100400000002537}}
Missing seams: {'public/ga': [], 'public/candidate': []}

## Unexplained


### i1225_objective_exceeds_room_32k

# Differential report
Baseline: `709099b3a397a7893a7abae8802ebe7bf605f79e`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.28.0`; canonical tag: `v0.28.0`.
Baseline config: bare.
Cases: 1; seed: 1227; unexplained events: 0.
Inherited user marks: 0 cases; 0 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'public/ga': {'wall_seconds': 0.2559186979999595, 'mean_case_seconds': 0.08947368000002598}, 'public/candidate': {'wall_seconds': 0.2603055379999546, 'mean_case_seconds': 0.09289809599999899}}
Missing seams: {'public/ga': [], 'public/candidate': []}

## Unexplained


### i1227_seq_outage_n113

# Differential report
Baseline: `709099b3a397a7893a7abae8802ebe7bf605f79e`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.28.0`; canonical tag: `v0.28.0`.
Baseline config: bare.
Cases: 1; seed: 1227; unexplained events: 0.
Inherited user marks: 0 cases; 0 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'public/ga': {'wall_seconds': 0.42300465200000303, 'mean_case_seconds': 0.24841227900000717}, 'public/candidate': {'wall_seconds': 0.4584802579999945, 'mean_case_seconds': 0.27655617099999574}}
Missing seams: {'public/ga': [], 'public/candidate': []}

## Unexplained


### i1227_seq_outage_n128

# Differential report
Baseline: `709099b3a397a7893a7abae8802ebe7bf605f79e`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.28.0`; canonical tag: `v0.28.0`.
Baseline config: bare.
Cases: 1; seed: 1227; unexplained events: 0.
Inherited user marks: 0 cases; 0 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'public/ga': {'wall_seconds': 0.49757927500000676, 'mean_case_seconds': 0.3162804739999956}, 'public/candidate': {'wall_seconds': 0.45113400299999284, 'mean_case_seconds': 0.279746779000007}}
Missing seams: {'public/ga': [], 'public/candidate': []}

## Unexplained


### i1227_seq_outage_n90

# Differential report
Baseline: `709099b3a397a7893a7abae8802ebe7bf605f79e`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.28.0`; canonical tag: `v0.28.0`.
Baseline config: bare.
Cases: 1; seed: 1227; unexplained events: 0.
Inherited user marks: 0 cases; 0 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'public/ga': {'wall_seconds': 0.6914193499999897, 'mean_case_seconds': 0.1886379259999984}, 'public/candidate': {'wall_seconds': 1.0439406469999994, 'mean_case_seconds': 0.8667926239999986}}
Missing seams: {'public/ga': [], 'public/candidate': []}

## Unexplained


### i1281_g0

# Differential report
Baseline: `6800cf45ad05280e6cf6fa9e1ad96c62b7132646`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.26.2`; canonical tag: `v0.26.2`.
Baseline config: managed.
Cases: 1; seed: 1227; unexplained events: 0.
Inherited user marks: 0 cases; 0 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'public/ga': {'wall_seconds': 0.8967652790000216, 'mean_case_seconds': 0.46011533300003293}, 'public/candidate': {'wall_seconds': 0.703909633999956, 'mean_case_seconds': 0.5339554039999825}}
Missing seams: {'public/ga': [], 'public/candidate': []}

## Unexplained


### projection

# Differential report
Baseline: `709099b3a397a7893a7abae8802ebe7bf605f79e`; candidate: `158191079b4ec61e676f9d03ea3f34069543e8fb`.
Baseline ref: `v0.28.0`; canonical tag: `v0.28.0`.
Baseline config: bare.
Cases: 1000; seed: 1227; unexplained events: 0.
Inherited user marks: 238 cases; 282 marks.

## Counts (layer / family / oracle)

```json
{}
```

Delta absorption (cases): {'D916_GA_MARKED_USER': 0, 'D1233_LAST_RESORT_USER_MARK': 0, 'D1123_64K_LEAF_TRADE': 0, 'D1123_40K_COMPRESS_TRADE': 0}
Zero-absorption deltas (stale on a full run): ['D916_GA_MARKED_USER', 'D1233_LAST_RESORT_USER_MARK', 'D1123_64K_LEAF_TRADE', 'D1123_40K_COMPRESS_TRADE']
Timings: {'projection/ga': {'wall_seconds': 19.37457832700005, 'mean_case_seconds': 0.018952909347000342}, 'projection/candidate': {'wall_seconds': 19.553217464, 'mean_case_seconds': 0.019162320344000023}}
Missing seams: {'projection/ga': [], 'projection/candidate': []}

## Unexplained


