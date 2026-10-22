from __future__ import annotations

import hashlib
import json

import pytest

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
        "evidence": ["https://github.com/example/project/releases/download/v1.0/RESULT.md"],
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
- https://github.com/example/project/releases/download/v1.0/RESULT.md

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
- https://github.com/example/project/releases/download/v1.0/RESULT.md

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


def test_new_row_rejects_local_evidence(tmp_path, capsys):
    row = _row("new-local", date="2026-10-16")
    row["evidence"] = ["/machine/raw.jsonl"]
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [row])
    assert generate.main(["--check", "--results", str(results)]) == 2
    assert "durable" in capsys.readouterr().err


def _assert_durable_row_with_raw(tmp_path, evidence):
    row = _row("new-durable", date="2026-10-16")
    row["evidence"] = [evidence]
    row["local_raw"] = ["raw/run.jsonl"]
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [row])
    assert generate.main(["--check", "--results", str(results)]) == 0
    rendered = generate.render(generate.load_rows(results), "source-hash")
    assert f"**evidence:**\n- {evidence}" in rendered
    assert "**local_raw:**\n- raw/run.jsonl" in rendered


def test_new_row_accepts_release_asset(tmp_path):
    _assert_durable_row_with_raw(
        tmp_path,
        "https://github.com/example/project/releases/download/v1.0/RESULT.md",
    )


def test_new_row_accepts_sha_blob(tmp_path):
    _assert_durable_row_with_raw(
        tmp_path,
        "https://github.com/example/project/blob/" + "a" * 40 + "/RESULT.md",
    )


def test_new_row_rejects_branch_blob(tmp_path, capsys):
    row = _row("new-branch", date="2026-10-16")
    row["evidence"] = ["https://github.com/example/project/blob/main/RESULT.md"]
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [row])
    assert generate.main(["--check", "--results", str(results)]) == 2
    assert "durable" in capsys.readouterr().err


def test_legacy_row_preserves_local_raw(tmp_path):
    row = _row("longmemeval-v1-s500-accuracy-2026-07-29", date="2026-10-15")
    row["evidence"] = []
    row["local_raw"] = ["raw/legacy.jsonl"]
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [row])
    assert generate.main(["--check", "--results", str(results)]) == 0
    rendered = generate.render(generate.load_rows(results), "source-hash")
    assert "**local_raw:**\n- raw/legacy.jsonl" in rendered


@pytest.mark.parametrize("evidence", [["raw/backfill.jsonl"], []], ids=["backdated", "empty"])
def test_nonlegacy_row_requires_durable_evidence(tmp_path, capsys, evidence):
    row = _row("nonlegacy-backfill", date="2026-07-29")
    row["evidence"] = evidence
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [row])
    assert generate.main(["--check", "--results", str(results)]) == 2
    assert "durable" in capsys.readouterr().err


def test_only_frozen_legacy_ids_are_grandfathered(tmp_path):
    row = _row("longmemeval-v1-s500-accuracy-2026-07-29", date="2026-10-16")
    row["evidence"] = ["legacy-reference"]
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [row])
    assert generate.main(["--check", "--results", str(results)]) == 0
    row["id"] = "new-row-same-date-and-evidence"
    _write_jsonl(results, [row])
    assert generate.main(["--check", "--results", str(results)]) == 2


@pytest.mark.parametrize("suffix", ["#L12", "#L12-L20"])
def test_sha_blob_accepts_line_anchor(tmp_path, suffix):
    _assert_durable_row_with_raw(
        tmp_path, "https://github.com/example/project/blob/" + "a" * 40 + "/RESULT.md" + suffix,
    )


@pytest.mark.parametrize("url", [
    "https://github.com/example/project/blob/" + "a" * 40 + "/../../blob/main/A.md",
    "https://github.com/example/project/releases/download/v1/..",
    "https://github.com/./project/releases/download/v1/RESULT.md",
    "https://github.com/example/project/blob/" + "a" * 40 + "/%2e%2e/A.md",
    "https://github.com/example/project/blob/" + "a" * 40 + "/%252e%252e/A.md",
    "https://github.com/example/project/blob/" + "a" * 40 + "/%25252e%25252e/A.md",
])
def test_durable_evidence_rejects_dot_segments(tmp_path, capsys, url):
    row = _row("dot-segments", date="2026-10-16")
    row["evidence"] = [url]
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [row])
    assert generate.main(["--check", "--results", str(results)]) == 2
    assert "durable" in capsys.readouterr().err


@pytest.mark.parametrize("path", ["/machine/raw.jsonl", "~/raw.jsonl", "C:\\raw.jsonl"])
def test_local_raw_rejects_machine_absolute_paths(tmp_path, capsys, path):
    row = _row("absolute-raw", date="2026-10-16")
    row["local_raw"] = [path]
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [row])
    assert generate.main(["--check", "--results", str(results)]) == 2
    assert "local_raw" in capsys.readouterr().err


@pytest.mark.parametrize("suffix", ["note", "notes"])
@pytest.mark.parametrize("field", [
    *(field for field in generate.SCALAR_FIELDS if field != "value"),
    "superseded_by", "evidence", "caveats", "local_raw", "extra",
])
def test_rejects_session_folder_references(tmp_path, capsys, field, suffix):
    row = _row("longmemeval-v1-s500-accuracy-2026-07-29")
    reference = "session-" + suffix + " 2026-07-30/example"
    row[field] = [reference] if field in {"evidence", "caveats", "local_raw"} else reference
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [row])
    output = tmp_path / "SCOREBOARD.md"
    output.write_text("keep this", encoding="utf-8")

    assert generate.main(["--results", str(results), "--output", str(output)]) == 2
    error = capsys.readouterr().err
    assert f"field '{field}" in error
    assert "local session folders" in error
    assert output.read_text(encoding="utf-8") == "keep this"


@pytest.mark.parametrize("reference", [
    "SESSION-" + "NOTES/2026-07-30/example",
    "prefix/session-" + "note/example",
])
def test_rejects_nested_session_folder_references(tmp_path, capsys, reference):
    row = _row("nested")
    row["extra"] = {"receipts": [{"source": reference}]}
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [row])

    assert generate.main(["--check", "--results", str(results)]) == 2
    assert "field 'extra.receipts[0].source'" in capsys.readouterr().err
