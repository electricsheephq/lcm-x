from __future__ import annotations

import hashlib
import json

from bench.scoreboard import generate


def _row(row_id: str, *, benchmark: str = "Bench", date: str = "2026-07-30") -> dict:
    return {
        "id": row_id,
        "benchmark": benchmark,
        "metric": "accuracy",
        "value": 0.91,
        "display": "91.0%",
        "tier": "P",
        "date": date,
        "system_commit": "system-sha",
        "harness_commit": "harness-sha",
        "judge": "judge (prompt.md)",
        "reader": "reader / medium",
        "retrieval_config": "delivery=cards; cap=25",
        "dataset_exposure": "none documented",
        "breakdown": "all=91.0%",
        "variance": "n=2; noise floor=1pp",
        "failclose": "0",
        "evidence": ["bench/evidence.md"],
        "caveats": [],
    }


def _write_jsonl(path, rows, malformed: str | None = None):
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if malformed is not None:
        body += malformed + "\n"
    path.write_text(body, encoding="utf-8")


def test_golden_render_two_rows(tmp_path):
    results = tmp_path / "results.jsonl"
    first = _row("bench-old-2026-07-29", date="2026-07-29")
    second = _row("bench-new-2026-07-30", date="2026-07-30")
    _write_jsonl(results, [first, second])

    rendered = generate.render(generate.load_rows(results), generate.sha256_file(results))

    expected_hash = hashlib.sha256(results.read_bytes()).hexdigest()
    assert rendered == f"""# Scoreboard

Every number ships with its full run config, variance, fail-close accounting, and known dataset defects — rows that cannot meet the standard do not render.

Generated from `results.jsonl` (sha256: `{expected_hash}`, rows: 2)

## Summary

| Benchmark | Metric | Result | Tier | Date | Details |
|---|---|---|---|---|---|
| Bench | accuracy | 91.0% | P | 2026-07-30 | [details](#bench-new-2026-07-30) |
| Bench | accuracy | 91.0% | P | 2026-07-29 | [details](#bench-old-2026-07-29) |

## Row disclosures

### <a id="bench-new-2026-07-30"></a>bench-new-2026-07-30

**id:**
bench-new-2026-07-30

**benchmark:**
Bench

**metric:**
accuracy

**value:**
0.91

**display:**
91.0%

**tier:**
P

**date:**
2026-07-30

**system_commit:**
system-sha

**harness_commit:**
harness-sha

**judge:**
judge (prompt.md)

**reader:**
reader / medium

**retrieval_config:**
delivery=cards; cap=25

**dataset_exposure:**
none documented

**breakdown:**
all=91.0%

**variance:**
n=2; noise floor=1pp

**failclose:**
0

**evidence:**
- bench/evidence.md

**caveats:**
- none

### <a id="bench-old-2026-07-29"></a>bench-old-2026-07-29

**id:**
bench-old-2026-07-29

**benchmark:**
Bench

**metric:**
accuracy

**value:**
0.91

**display:**
91.0%

**tier:**
P

**date:**
2026-07-29

**system_commit:**
system-sha

**harness_commit:**
harness-sha

**judge:**
judge (prompt.md)

**reader:**
reader / medium

**retrieval_config:**
delivery=cards; cap=25

**dataset_exposure:**
none documented

**breakdown:**
all=91.0%

**variance:**
n=2; noise floor=1pp

**failclose:**
0

**evidence:**
- bench/evidence.md

**caveats:**
- none
"""


def test_missing_field_refusal(tmp_path, capsys):
    results = tmp_path / "results.jsonl"
    row = _row("missing-metric")
    del row["metric"]
    _write_jsonl(results, [row])
    output = tmp_path / "SCOREBOARD.md"
    output.write_text("keep this", encoding="utf-8")

    assert generate.main(["--results", str(results), "--output", str(output)]) == 2
    error = capsys.readouterr().err
    assert "line 1" in error
    assert "metric" in error
    assert output.read_text(encoding="utf-8") == "keep this"


def test_malformed_line_refusal(tmp_path, capsys):
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [_row("good")], malformed="{not-json")

    assert generate.main(["--check", "--results", str(results)]) == 2
    error = capsys.readouterr().err
    assert "line 2" in error
    assert "malformed JSON" in error


def test_supersession_strikes_old_row_and_links_successor(tmp_path):
    results = tmp_path / "results.jsonl"
    old = _row("old-2026-07-29", date="2026-07-29")
    old["superseded_by"] = "new-2026-07-30"
    new = _row("new-2026-07-30", date="2026-07-30")
    _write_jsonl(results, [old, new])

    rendered = generate.render(generate.load_rows(results), generate.sha256_file(results))

    assert "~~Bench~~" in rendered
    assert "→ [successor](#new-2026-07-30)" in rendered


def test_byte_stable_two_runs(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [_row("stable")])
    output_one = tmp_path / "one.md"
    output_two = tmp_path / "two.md"

    assert generate.main(["--results", str(results), "--output", str(output_one)]) == 0
    assert generate.main(["--results", str(results), "--output", str(output_two)]) == 0
    assert output_one.read_bytes() == output_two.read_bytes()


def test_check_does_not_write(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [_row("check-only")])
    output = tmp_path / "SCOREBOARD.md"

    assert generate.main(["--check", "--results", str(results), "--output", str(output)]) == 0
    assert not output.exists()
