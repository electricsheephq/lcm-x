from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

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


@pytest.mark.parametrize("row_directory", [False, True], ids=["absent", "empty"])
def test_golden_render_two_rows(tmp_path, row_directory):
    results = tmp_path / "results.jsonl"
    first = _row("bench-old-2026-07-29", date="2026-07-29")
    second = _row("bench-new-2026-07-30", date="2026-07-30")
    _write_jsonl(results, [first, second])
    if row_directory:
        (tmp_path / "results.d").mkdir()

    rendered = generate.render(generate.load_rows(results), generate.sha256_sources(results))

    manifest = [["results.jsonl", hashlib.sha256(results.read_bytes()).hexdigest()]]
    expected_hash = hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()
    assert generate.sha256_sources(results) == expected_hash
    assert rendered == f"""# Scoreboard

Every number ships with its full run config, variance, fail-close accounting, and known dataset defects — rows that cannot meet the standard do not render.

Generated from `results.jsonl` + `results.d/*.json` (combined sha256: `{expected_hash}`, rows: 2)

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
    generate.generate(results, output)
    frozen = output.read_bytes()

    assert generate.main(["--check", "--results", str(results), "--output", str(output)]) == 0
    assert output.read_bytes() == frozen


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
    output = tmp_path / "page.md"
    generate.generate(results, output)
    assert generate.main(["--check", "--results", str(results), "--output", str(output)]) == 0
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
    output = tmp_path / "page.md"
    generate.generate(results, output)
    assert generate.main(["--check", "--results", str(results), "--output", str(output)]) == 0
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
    output = tmp_path / "page.md"
    generate.generate(results, output)
    assert generate.main(["--check", "--results", str(results), "--output", str(output)]) == 0
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


def _write_row_file(results, row, *, name=None):
    directory = results.parent / "results.d"
    directory.mkdir(exist_ok=True)
    path = directory / f"{name or row['id']}.json"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def test_independent_row_files_load_in_filename_order(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [_row("legacy")])
    _write_row_file(results, _row("z-row"))
    _write_row_file(results, _row("a-row"))
    assert [row["id"] for row in generate.load_rows(results)] == ["legacy", "a-row", "z-row"]
    output = tmp_path / "page.md"
    rendered = generate.generate(results, output)
    assert "[registry row](results.d/a-row.json)" in rendered
    assert "[registry row](results.d/z-row.json)" in rendered
    assert "results.jsonl#L" not in rendered  # legacy disclosures gain no new bytes


def test_duplicate_id_across_sources_is_rejected(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [_row("duplicate")])
    _write_row_file(results, _row("duplicate"))
    with pytest.raises(generate.ScoreboardError, match="duplicate id"):
        generate.load_rows(results)


def test_row_filename_must_match_id(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [])
    _write_row_file(results, _row("actual"), name="wrong")
    with pytest.raises(generate.ScoreboardError, match="filename.*id"):
        generate.load_rows(results)


def test_existing_page_changes_only_source_header(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [_row("legacy-old", date="2026-07-29"), _row("legacy-new")])
    rendered = generate.generate(results, tmp_path / "page.md").encode().splitlines(keepends=True)
    # Body digest from BASE's regenerated fixture page, not its stale committed page.
    # Fixture rows keep this check independent of future compatible legacy appends.
    assert hashlib.sha256(b"".join(rendered[:4] + rendered[5:])).hexdigest() == "e3f2bffe27fb2fa2d30ac88221b714b82d3a45c238c0568ee8ee03e5290ff1fa"
    assert rendered[4].startswith(b"Generated from `results.jsonl` + `results.d/*.json`")


def test_combined_digest_covers_names_and_exact_bytes(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [_row("legacy")])
    row_path = _write_row_file(results, _row("new"))
    manifest = [["results.jsonl", hashlib.sha256(results.read_bytes()).hexdigest()],
                ["results.d/new.json", hashlib.sha256(row_path.read_bytes()).hexdigest()]]
    expected = hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()
    assert generate.sha256_sources(results) == expected
    row_path.write_text(row_path.read_text() + " ", encoding="utf-8")
    assert generate.sha256_sources(results) != expected


@pytest.mark.parametrize("body", ["{broken", "[]", "{}"])
def test_row_file_validation_leaves_output_untouched(tmp_path, body):
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [])
    path = _write_row_file(results, _row("bad"))
    path.write_text(body, encoding="utf-8")
    output = tmp_path / "page.md"
    output.write_text("keep this", encoding="utf-8")
    assert generate.main(["--results", str(results), "--output", str(output)]) == 2
    assert output.read_text() == "keep this"


def test_supersession_can_cross_sources(tmp_path):
    results = tmp_path / "results.jsonl"
    old = _row("old")
    old["superseded_by"] = "new"
    _write_jsonl(results, [old])
    _write_row_file(results, _row("new"))
    assert len(generate.load_rows(results)) == 2


def test_successor_supersedes_frozen_legacy_rows(tmp_path):
    results = tmp_path / "results.jsonl"
    old = _row("old")
    _write_jsonl(results, [old, _row("other")])
    frozen = results.read_bytes()
    new = _row("new") | {"supersedes": ["old", "other"]}
    _write_row_file(results, new)
    rows = generate.load_rows(results)
    assert results.read_bytes() == frozen
    assert [row.get("superseded_by") for row in rows] == ["new", "new", None]
    expected_old = old | {"superseded_by": "new"}
    assert generate._render_disclosure(rows[0]) == generate._render_disclosure(expected_old)
    assert generate._render_summary(rows) == generate._render_summary([
        expected_old, _row("other") | {"superseded_by": "new"}, new,
    ])


@pytest.mark.parametrize("case,message", [
    ("unknown", "missing"), ("self", "itself"),
    ("two-successors", "two successors"), ("disagrees", "disagrees"),
])
def test_supersedes_errors(tmp_path, case, message):
    results = tmp_path / "results.jsonl"
    old, new = _row("old"), _row("new") | {"supersedes": ["old"]}
    if case == "unknown":
        new["supersedes"] = ["missing"]
    elif case == "self":
        new["supersedes"] = ["new"]
    elif case == "two-successors":
        _write_row_file(results, _row("other") | {"supersedes": ["old"]})
    else:
        old["superseded_by"] = "other"
        _write_row_file(results, _row("other"))
    _write_jsonl(results, [old])
    _write_row_file(results, new)
    with pytest.raises(generate.ScoreboardError, match=message):
        generate.load_rows(results)


def test_supersedes_agrees_with_existing_forward_link(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [_row("old") | {"superseded_by": "new"}])
    _write_row_file(results, _row("new") | {"supersedes": ["old"]})
    assert generate.load_rows(results)[0]["superseded_by"] == "new"


@pytest.mark.parametrize("value", [None, "old", [1], [""]])
def test_supersedes_requires_list_of_nonempty_ids(tmp_path, value):
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [_row("old")])
    _write_row_file(results, _row("new") | {"supersedes": value})
    with pytest.raises(generate.ScoreboardError, match="supersedes"):
        generate.load_rows(results)


def test_pending_lists_both_sources_without_writing(tmp_path, capsys):
    results, output = tmp_path / "results.jsonl", tmp_path / "page.md"
    _write_jsonl(results, [_row("rendered")])
    generate.generate(results, output)
    frozen = output.read_bytes()
    _write_jsonl(results, [_row("rendered"), _row("legacy-pending")])
    _write_row_file(results, _row("file-pending"))
    assert generate.main(["--pending", "--results", str(results), "--output", str(output)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "pending: legacy-pending", "pending: file-pending",
    ]
    assert output.read_bytes() == frozen


@pytest.mark.parametrize("change", ["removed", "summary", "disclosure", "supersession"])
def test_pending_rejects_stale_rendered_rows(tmp_path, capsys, change):
    results, output = tmp_path / "results.jsonl", tmp_path / "page.md"
    row = _row("old")
    _write_jsonl(results, [row])
    if change == "supersession":
        _write_row_file(results, _row("new"))
    generate.generate(results, output)
    if change == "removed":
        _write_jsonl(results, [])
    elif change == "summary":
        output.write_text(output.read_text().replace("| 91.0% |", "| 90.0% |"))
    elif change == "disclosure":
        output.write_text(output.read_text().replace("**value:**\n0.91", "**value:**\n0.90"))
    else:
        _write_row_file(results, _row("new") | {"supersedes": ["old"]})
    frozen = output.read_bytes()
    assert generate.main(["--pending", "--results", str(results), "--output", str(output)]) == 2
    assert "stale: old" in capsys.readouterr().out
    assert output.read_bytes() == frozen


@pytest.mark.parametrize("change", [None, "summary", "disclosure", "partial-strike"])
def test_pending_successor_defers_only_supersession_effects(tmp_path, capsys, change):
    results, output = tmp_path / "results.jsonl", tmp_path / "page.md"
    _write_jsonl(results, [_row("old")])
    generate.generate(results, output)
    legacy = results.read_bytes()
    _write_row_file(results, _row("new") | {"supersedes": ["old"]})
    if change == "summary":
        output.write_text(output.read_text().replace("| 91.0% |", "| 90.0% |"))
    elif change == "disclosure":
        output.write_text(output.read_text().replace("**value:**\n0.91", "**value:**\n0.90"))
    elif change == "partial-strike":
        output.write_text(output.read_text().replace("| 91.0% |", "| ~~91.0%~~ |"))
    frozen = output.read_bytes()
    args = ["--results", str(results), "--output", str(output)]
    assert generate.main(["--pending", *args]) == (0 if change is None else 2)
    lines = capsys.readouterr().out.splitlines()
    assert "pending: new" in lines
    if change is None:
        assert "pending: old (superseded by new)" in lines
        assert len(lines) == 2
    else:
        assert "stale: old" in lines
    assert generate.main(["--check", *args]) == 2
    assert output.read_bytes() == frozen
    assert results.read_bytes() == legacy


def test_check_requires_exact_render_and_never_writes(tmp_path, capsys):
    results, output = tmp_path / "results.jsonl", tmp_path / "page.md"
    _write_jsonl(results, [_row("old")])
    assert generate.main(["--check", "--results", str(results), "--output", str(output)]) == 2
    generate.generate(results, output)
    assert generate.main(["--check", "--results", str(results), "--output", str(output)]) == 0
    frozen = output.read_bytes()
    _write_row_file(results, _row("pending"))
    assert generate.main(["--check", "--results", str(results), "--output", str(output)]) == 2
    assert "out of date" in capsys.readouterr().err
    assert output.read_bytes() == frozen


def test_two_row_prs_merge_without_page_conflict(tmp_path):
    transcript = []

    def git(*args):
        result = subprocess.run(
            ["git", "-c", "user.name=Scoreboard test", "-c", "user.email=test@example.invalid", *args],
            cwd=tmp_path, capture_output=True, text=True, check=True,
        )
        transcript.append(f"git {' '.join(args)}\n{result.stdout}{result.stderr}")
        return result.stdout.strip()

    git("init", "-b", "main")
    results, output = tmp_path / "results.jsonl", tmp_path / "SCOREBOARD.md"
    _write_jsonl(results, [_row("legacy")])
    generate.generate(results, output)
    frozen = output.read_bytes()
    git("add", ".")
    git("commit", "-m", "base")
    for branch in ("row-a", "row-b"):
        git("checkout", "-b", branch, "main")
        _write_row_file(results, _row(branch))
        git("add", f"results.d/{branch}.json")
        git("commit", "-m", branch)
        assert git("diff", "--name-only", "main") == f"results.d/{branch}.json"
    git("checkout", "main")
    git("merge", "--no-edit", "row-a")
    git("merge", "--no-edit", "row-b")
    pending = subprocess.run(
        [sys.executable, str(Path(generate.__file__)), "--pending", "--results", str(results),
         "--output", str(output)], capture_output=True, text=True, check=True,
    )
    transcript.append(f"generate --pending\n{pending.stdout}{pending.stderr}")
    print("\n".join(transcript))
    assert pending.stdout.splitlines() == ["pending: row-a", "pending: row-b"]
    assert output.read_bytes() == frozen
    assert not git("status", "--porcelain")
