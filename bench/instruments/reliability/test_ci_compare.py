"""Exercise the compare CLI with verdicts from recorded synthetic matrices."""
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

from bench.instruments.reliability import ci

FIXTURES = Path(__file__).with_name("fixtures") / "compare"
NIGHTS = ("44230043164", "44451000342", "44667323840")
BUILD = "ref-0.21.5@f97608f178d1ffeca59860195ab7da295f7c8e5f"
HOST = BUILD.split("@")[0]
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
    assert new == ([(f"{HOST}/default-window-r1/{CELL}", "PASS", "FAIL", "NEW")] if status else [])


def test_recorded_respin(recorded):
    rows = table(invoke(recorded("respin-final"), recorded("respin-base"), mode="respin"), 1)
    assert len(rows) == 784
    new = [row for row in rows if row[-1] == "NEW"]
    assert {cell.split("/")[0] for cell, *_ in new} == {
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
        (f"{HOST}/default-window-r1/{CELL}", "FAIL", "FAIL", "NEW"),
    ]


def planted(root, verdict=None, *, flake=False, host=HOST):
    folder = root / f"reliability-{host}" / "default-window-r1"
    folder.mkdir(parents=True)
    sha = BUILD.split("@")[1]
    rows = [dict(cell=cid, host=host, host_sha=sha, verdict=v)
            for cid, v in (("default-window/stable", "PASS"), (CELL, verdict)) if v]
    (folder / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    if flake:
        (folder / "run.json").write_text(json.dumps({"comparison_flakes": [CELL]}))
    return root


def test_flake_once_is_advisory_and_twice_in_seven_nights_is_new(tmp_path):
    baseline = planted(tmp_path / "baseline", "PASS")
    candidate = planted(tmp_path / "candidate", "FAIL", flake=True)
    key = f"{HOST}/default-window-r1/{CELL}"
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
    assert (f"{HOST}/default-window-r1/{CELL}", "PASS", "—", "MISSING") in table(invoke(candidate, baseline), 1)


@pytest.mark.parametrize(("before", "status", "kind"), [("FAIL", 0, "MATCHED"), ("PASS", 1, "NEW")])
def test_nightly_reuses_newest_verdict_for_returning_host(tmp_path, before, status, kind):
    host = "returning-host"
    older = planted(tmp_path / "100", "PASS" if before == "FAIL" else "FAIL", host=host)
    planted(older, "FAIL", host="history-only-host")
    first = planted(tmp_path / "101", "PASS")
    planted(first, before, host=host)
    second = planted(tmp_path / "102", "PASS")
    third = planted(tmp_path / "103", "PASS")
    planted(third, "FAIL", host=host)
    key = f"{host}/default-window-r1/{CELL}"
    assert (key, before, "—", "MISSING") in table(invoke(second, first), 1)

    history = ("--history", tmp_path / "unreadable", "--history", first, "--history", older)
    rows = table(invoke(third, second, *history), status)
    assert (key, f"{before} (run 101)", "FAIL", kind) in rows
    assert [row[0] for row in rows if row[-1] == "NEW"] == ([key] if status else [])
    assert all(not row[0].startswith("history-only-host/") for row in rows)
    assert (key, "—", "FAIL", "NEW") in table(invoke(third, second), 1)
    assert (key, "—", "FAIL", "NEW") in table(invoke(third, second, *history, mode="respin"), 1)


def test_respin_matches_the_same_cells_after_host_pin_bump(recorded):
    candidate, baseline = recorded("respin-final"), recorded("respin-base")
    original = table(invoke(candidate, baseline, mode="respin"), 1)
    for file in (candidate / "reliability-upstream-main").glob("*/results.jsonl"):
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        original_sha = rows[0]["host_sha"]
        file.write_text("".join(json.dumps(dict(row, host_sha="f" * 40)) + "\n" for row in rows))
    changed = invoke(candidate, baseline, mode="respin")
    rows = table(changed, 1)
    assert [(row[0], row[-1]) for row in rows] == [(row[0], row[-1]) for row in original]
    assert original_sha in changed.stdout and "f" * 40 in changed.stdout


def test_sha_free_known_key_keeps_a_standing_failure_red(tmp_path):
    baseline = planted(tmp_path / "baseline", "FAIL")
    candidate = planted(tmp_path / "candidate", "FAIL")
    known = tmp_path / "known.json"
    key = f"ref-0.21.5/default-window-r1/{CELL}"
    known.write_text(json.dumps([key]))
    assert (key, "FAIL", "FAIL", "NEW") in table(invoke(candidate, baseline, "--known-new", known), 1)


def test_older_unreadable_history_warns_and_other_history_still_counts(tmp_path):
    baseline = planted(tmp_path / "baseline", "PASS")
    candidate = planted(tmp_path / "candidate", "FAIL", flake=True)
    older = planted(tmp_path / "older", "FAIL", flake=True)
    result = invoke(candidate, baseline, "--history", tmp_path / "unreadable", "--history", older)
    assert any(row[-1] == "NEW" for row in table(result, 1))
    assert "warning" in result.stderr and "history" in result.stderr
    result = invoke(tmp_path / "unreadable", baseline, "--history", older)
    assert result.returncode == 2 and "candidate:" in result.stderr


def test_pass_to_unsupported_is_missing(tmp_path):
    baseline = planted(tmp_path / "baseline", "PASS")
    candidate = planted(tmp_path / "candidate", "UNSUPPORTED")
    rows = table(invoke(candidate, baseline), 1)
    assert any(row[1:] == ("PASS", "UNSUPPORTED", "MISSING") for row in rows)


def test_unsupported_keeps_an_existing_alarm_cell_tracked(tmp_path):
    baseline = planted(tmp_path / "baseline", "FAIL")
    candidate = planted(tmp_path / "candidate", "UNSUPPORTED")
    known, alarm = tmp_path / "known.json", tmp_path / "alarm.json"
    known.write_text(json.dumps([[f"{BUILD}/default-window-r1", [CELL]]]))
    rows = table(invoke(candidate, baseline, "--known-new", known, "--alarm-result", alarm), 0)
    assert json.loads(alarm.read_text())["known"] == [["ref-0.21.5/default-window-r1", [CELL]]]
    assert any(row[1:] == ("FAIL", "UNSUPPORTED", "MATCHED") for row in rows)


def github_context(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/project")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_RUN_ID", "105")


def test_baselines_require_publication(monkeypatch):
    github_context(monkeypatch)

    def gh(*args):
        if any(arg.endswith("/actions/artifacts") for arg in args):
            # This includes a dispatch, the current run, an unjudged old run, and two published runs.
            return "106\n105\n104\n102\n101\n"
        if any(arg.endswith("/runs") for arg in args):
            assert {"branch=main", "event=schedule", "status=completed", "per_page=100"} <= set(args)
            return "105\n104\n103\n102\n101\n"
        assert "select(.name == \"publish\" and .conclusion == \"success\")" in args[-1]
        return "" if "/104/" in args[1] else "123\n"

    monkeypatch.setattr(ci, "gh", gh)
    assert ci.baseline_ids() == ["102", "101"]


def test_baselines_query_only_the_latest_100_runs(monkeypatch):
    github_context(monkeypatch)

    def gh(*args):
        if any(arg.endswith("/actions/artifacts") for arg in args):
            return "102\n"
        assert "--paginate" not in args and "per_page=100" in args
        return ""

    monkeypatch.setattr(ci, "gh", gh)
    assert ci.baseline_ids() == []


@pytest.mark.parametrize("run_id", ["105", "000105", " 101", "101\n", "-101", "١٠١", ""])
def test_named_baseline_refuses_self_and_non_ascii_digits(tmp_path, monkeypatch, run_id):
    github_context(monkeypatch)
    monkeypatch.setattr(ci, "gh", lambda *args: pytest.fail("invalid run id reached GitHub"))
    with pytest.raises(ValueError, match="baseline_run"):
        ci.download_named_baseline(run_id, tmp_path)


@pytest.mark.parametrize("compact", [True, False])
def test_named_baseline_download_keeps_only_tables_in_host_layout(tmp_path, monkeypatch, compact):
    github_context(monkeypatch)
    hosts = ("upstream-main", "ref-0.21.5")
    names = [f"reliability-{host}" for host in hosts]
    rounds = ("r1", "r2", "default-window-r1")
    files = ("MATRIX.md", "results.jsonl", "run.json")
    downloads = []

    def gh(*args):
        if args[0] == "api":
            if args[1].endswith("/101"):
                return ".github/workflows/reliability-nightly.yml@refs/heads/main\n"
            assert any(arg.endswith("/101/artifacts?per_page=100") for arg in args)
            return "\n".join(["reliability-tables"] if compact else [*names, "reliability-table-shard-ref-0.21.5"])
        assert args[:3] == ("run", "download", "101")
        name, folder = args[4], Path(args[6])
        downloads.append(name)
        assert folder.is_relative_to(tmp_path)
        if compact:
            assert folder == tmp_path / "101"
            return ""
        assert folder != tmp_path / "101" / name
        for round_name in rounds:
            out = folder / round_name
            (out / "cells").mkdir(parents=True)
            for file in files:
                (out / file).write_text(f"{name}/{round_name}/{file}")
                (out / "cells" / file).write_text("large data")
            (out / "extra.md").write_text("other report")
        return ""

    monkeypatch.setattr(ci, "gh", gh)
    ci.download_named_baseline("101", tmp_path)
    assert downloads == (["reliability-tables"] if compact else names)
    if not compact:
        contents = {str(file.relative_to(tmp_path)) for file in tmp_path.rglob("*") if file.is_file()}
        assert contents == {f"101/{name}/{r}/{f}" for name in names for r in rounds for f in files}


def test_named_baseline_rejects_a_run_of_another_workflow(tmp_path, monkeypatch):
    github_context(monkeypatch)
    monkeypatch.setattr(ci, "gh", lambda *args: ".github/workflows/another.yml\n")
    with pytest.raises(ValueError, match="reliability-nightly.yml"):
        ci.download_named_baseline("101", tmp_path)


def test_null_alarm_body_fails_closed_when_comments_have_no_tracked_block(monkeypatch):
    github_context(monkeypatch)
    monkeypatch.setattr(ci, "open_alarm", lambda: {"number": 42, "body": None})
    monkeypatch.setattr(ci, "gh", lambda *args: "")
    with pytest.raises(ValueError, match="no tracked cells block"):
        ci.alarm_state()
    key = f"ref-0.21.5/default-window-r1/{CELL}"
    monkeypatch.setattr(ci, "gh", lambda *args: json.dumps(ci.ALARM_MARKER + json.dumps([key]) + "\n```"))
    assert ci.alarm_state() == {key}


@pytest.mark.parametrize(("timed_out", "reason"), [(False, "20-cell cap"), (True, "60-minute limit")])
def test_retry_budget_leaves_unretried_cells_new(tmp_path, monkeypatch, capsys, timed_out, reason):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    host, sha = BUILD.split("@")
    cells = [f"cell/{i:02}" for i in range(23)]
    for root, verdict in ((baseline, "PASS"), (candidate, "FAIL")):
        folder = root / f"reliability-{host}" / "r1"
        folder.mkdir(parents=True)
        (folder / "results.jsonl").write_text("".join(
            json.dumps(dict(cell=cell, host=host, host_sha=sha, verdict=verdict)) + "\n" for cell in cells
        ))
        (folder / "run.json").write_text("{}")

    attempted, deadlines = [], set()

    def prepare(_host, _root, out):
        spec = {"hosts": {host: {"sha": sha}}}
        out.write_text(json.dumps(spec))
        return spec

    def runner(command, deadline):
        if deadline is not None:
            deadlines.add(deadline)
        if "pip" in command:
            return
        if "prep" in command:
            prepare(host, None, Path(command[command.index("--out") + 1]))
            return
        selected = command[command.index("--cells") + 1].split(",")
        attempted.extend(selected)
        if timed_out:
            raise subprocess.TimeoutExpired(command, 3600)
        out = Path(command[command.index("--out") + 1])
        out.mkdir(parents=True)
        (out / "results.jsonl").write_text("".join(
            json.dumps(dict(cell=cell, host=host, host_sha=sha, verdict="PASS")) + "\n" for cell in selected
        ))

    if timed_out:
        class RetryDirectory(tempfile.TemporaryDirectory):
            def __init__(self, *, ignore_cleanup_errors=False, **kwargs):
                super().__init__(ignore_cleanup_errors=ignore_cleanup_errors, **kwargs)
                self.ignore_cleanup_errors = ignore_cleanup_errors

            def __exit__(self, *exc):
                super().__exit__(*exc)
                if not self.ignore_cleanup_errors:
                    raise OSError("retry directory is still in use")

        monkeypatch.setattr(ci.tempfile, "TemporaryDirectory", RetryDirectory)
    monkeypatch.setattr(ci, "_run_retry", runner, raising=False)
    monkeypatch.setattr(ci, "prep", prepare)
    monkeypatch.setattr(ci.subprocess, "run", lambda command, **kwargs: runner(command, None))
    assert ci.main(["compare", "--mode", "nightly", "--candidate", str(candidate), "--baseline", str(baseline),
                    "--retry-scratch", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert attempted == cells[:20]
    assert len(deadlines) == 1
    assert output.count("| NEW |") == (23 if timed_out else 3)
    assert f"{20 if timed_out else 3} ({reason})" in output


def test_unexpected_retry_runner_error_keeps_the_comparison_and_new_cells(tmp_path, monkeypatch, capsys):
    baseline = planted(tmp_path / "baseline", "PASS")
    candidate = planted(tmp_path / "candidate", "FAIL")
    alarm = tmp_path / "alarm.json"

    def runner(command, deadline):
        raise ValueError("unexpected runner failure")

    monkeypatch.setattr(ci, "_run_retry", runner)
    assert ci.main(["compare", "--mode", "nightly", "--candidate", str(candidate), "--baseline", str(baseline),
                    "--retry-scratch", str(tmp_path), "--alarm-result", str(alarm)]) == 1
    captured = capsys.readouterr()
    key = f"{HOST}/default-window-r1/{CELL}"
    assert f"| `{key}` | PASS | FAIL | NEW |" in captured.out
    assert "1 (retry unavailable)" in captured.out
    assert "retries unavailable" in captured.err and "ValueError" in captured.err
    assert json.loads(alarm.read_text())["new"] == [[key, "PASS", "FAIL", "NEW"]]
