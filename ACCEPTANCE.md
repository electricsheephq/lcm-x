# Pressure-relief acceptance

Branch: `fix/preflight-relief-host-pressure` at base `62a34e512557fde216bde3d32594f9c73f3fbf15`.

## Change

`should_compress_preflight` now treats pressure as relieved only when both its
rough estimate and the host's latest `last_prompt_tokens` observation are under
the threshold. The regression coverage exercises sustained host pressure with
an under-threshold LCM estimate and the preserved under-threshold relief path.

## Acceptance bundle

- `PYTHONPATH=/Users/m1/hermes-work/wt-hermes-gauntlet uv run --with pytest python -m pytest tests/test_fresh_tail_pressure_yield.py tests/test_issue_6_fresh_tail_pressure.py tests/test_lcm_engine.py tests/test_compaction_telemetry.py -q -p no:cacheprovider`
  - Verbatim tail: `.... [100%]` / `867 passed, 1 skipped, 1567 warnings in 18.95s`
- Requested `ruff check compaction.py engine.py` had no standalone `ruff` binary (exit 127); the one-time bootstrap `uv run --with ruff ruff check compaction.py engine.py` passed: `All checks passed!`
- Requested `python -m compileall -q compaction.py engine.py` had no `python` binary (exit 127); `python3 -m compileall -q compaction.py engine.py` passed.
- `git diff --check` passed.

## Offline reproduction

Ran a temporary permitted local harness copy (`repro_local.py`) with the host
observation line added. Verbatim tail:

```text
k= 99 pre=0 status=error      reason=summary publication could not prove co streak=49 tok 6598->6598 len 99->99 YIELD/PUBFAIL
k=101 pre=0 status=error      reason=summary publication could not prove co streak=50 tok 6650->6650 len 101->101 YIELD/PUBFAIL
summary_nodes: 0
```

This proves the offline pressure-yield streak reaches sustained observations and
emits `YIELD`; it does **not** prove live-host behavior, which belongs to the
release gauntlet.
