"""Exercise the compare CLI with verdicts from recorded synthetic matrices."""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

FIXTURES = Path(__file__).with_name("fixtures") / "compare"
NIGHTS = ("44230043164", "44451000342", "44667323840")
BUILD = "ref-0.21.5@f97608f178d1ffeca59860195ab7da295f7c8e5f"
CELL = "default-window/objective-1221"


@pytest.fixture
def recorded(tmp_path):
    def unpack(run):
        root = tmp_path / run
        files = defaultdict(list)
        for line in (FIXTURES / f"{run}.jsonl").read_text().splitlines():
            row = json.loads(line)
            path = root / f"reliability-{row['host']}" / row.pop("round") / "results.jsonl"
            files[path].append(json.dumps(row))
        for path, lines in files.items():
            path.parent.mkdir(parents=True)
            path.write_text("\n".join(lines) + "\n")
        return root

    return unpack


def invoke(candidate, baseline, *extra, mode="nightly"):
    return subprocess.run(
        [sys.executable, "-m", "bench.instruments.reliability.ci", "compare",
         "--candidate", str(candidate), "--baseline", str(baseline), "--mode", mode, *map(str, extra)],
        capture_output=True, text=True,
    )


def table(result, status):
    assert result.returncode == status, result.stderr + result.stdout
    lines = result.stdout.strip().splitlines()
    assert lines[0] == "| Cell | Baseline verdict | Candidate verdict | Class |"
    return [tuple(field.strip().strip("`") for field in line.split("|")[1:-1]) for line in lines[2:]]


@pytest.mark.parametrize(("previous", "current", "status"), [(0, 1, 0), (1, 2, 1)])
def test_recorded_nightly(recorded, previous, current, status):
    rows = table(invoke(recorded(NIGHTS[current]), recorded(NIGHTS[previous])), status)
    new = [row for row in rows if row[-1] == "NEW"]
    assert len(rows) == 126
    assert new == ([(f"{BUILD}/default-window-r1/{CELL}", "PASS", "FAIL", "NEW")] if status else [])


def test_recorded_respin(recorded):
    rows = table(invoke(recorded("respin-final"), recorded("respin-base"), mode="respin"), 1)
    assert len(rows) == 784
    new = [row for row in rows if row[-1] == "NEW"]
    assert {cell.split("@")[0] for cell, *_ in new} == {
        "floor-0.21.3", "r34.10-0.21.5", "r34.15-0.21.5", "r34.4-0.21.5",
        "ref-0.21.5", "upstream-main", "upstream-uid",
    }
    assert len(new) == 7
    assert all(cell.endswith(f"/default-window-r1/{CELL}") and (base, cand) == ("PASS", "FAIL")
               for cell, base, cand, _ in new)
    failed = {"FAIL", "INCONCLUSIVE", "ERROR"}
    assert sum(base in failed and cand in failed and kind == "MATCHED" for _, base, cand, kind in rows) == 67
    assert sum(kind == "FIXED" for _, _, _, kind in rows) == 1


def test_known_new_keeps_second_night_red(recorded, tmp_path):
    night = recorded(NIGHTS[2])
    assert not any(row[-1] == "NEW" for row in table(invoke(night, night), 0))
    known = tmp_path / "known.json"
    known.write_text(json.dumps([f"{BUILD}/default-window-r1/{CELL}"]))
    rows = table(invoke(night, night, "--known-new", known), 1)
    assert [row for row in rows if row[-1] == "NEW"] == [
        (f"{BUILD}/default-window-r1/{CELL}", "FAIL", "FAIL", "NEW"),
    ]


def planted(root, verdict=None, *, flake=False):
    folder = root / "reliability-ref-0.21.5" / "default-window-r1"
    folder.mkdir(parents=True)
    host, sha = BUILD.split("@")
    rows = [dict(cell=cid, host=host, host_sha=sha, verdict=v)
            for cid, v in (("default-window/stable", "PASS"), (CELL, verdict)) if v]
    (folder / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    if flake:
        (folder / "run.json").write_text(json.dumps({"comparison_flakes": [CELL]}))
    return root


def test_flake_once_is_advisory_and_twice_in_seven_nights_is_new(tmp_path):
    baseline = planted(tmp_path / "baseline", "PASS")
    candidate = planted(tmp_path / "candidate", "FAIL", flake=True)
    key = f"{BUILD}/default-window-r1/{CELL}"
    assert (key, "PASS", "FAIL", "FLAKE") in table(invoke(candidate, baseline), 0)
    recent = planted(tmp_path / "recent", "FAIL", flake=True)
    passing = [planted(tmp_path / str(i), "PASS") for i in range(5)]
    history = [arg for p in (*passing[:4], recent) for arg in ("--history", p)]
    assert (key, "PASS", "FAIL", "NEW") in table(invoke(candidate, baseline, *history), 1)
    expired = [arg for p in (*passing, recent) for arg in ("--history", p)]
    assert (key, "PASS", "FAIL", "FLAKE") in table(invoke(candidate, baseline, *expired), 0)
    assert (key, "FAIL", "FAIL", "NEW") in table(invoke(candidate, recent), 1)


def test_unreadable_baseline_exits_two(tmp_path):
    result = invoke(planted(tmp_path / "candidate", "PASS"), tmp_path / "absent")
    assert result.returncode == 2
    assert "baseline" in result.stderr and "results.jsonl" in result.stderr


def test_baseline_cell_absent_from_candidate_is_missing(tmp_path):
    baseline = planted(tmp_path / "baseline", "PASS")
    candidate = planted(tmp_path / "candidate")
    assert (f"{BUILD}/default-window-r1/{CELL}", "PASS", "—", "MISSING") in table(invoke(candidate, baseline), 1)
