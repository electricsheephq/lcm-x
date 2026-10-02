"""Reliability harness cell scratch: a cell's Hermes home and DB copies live in a private scratch dir (default
$TMPDIR) and are deleted once the cell is scored, so no state.db / lcm.db (or -wal / -shm) is left in --out unless
--keep-dbs / --keep-homes asks for it. No Hermes: the host phase is faked and writes live WAL-mode databases."""
from __future__ import annotations

import errno
import json
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.instruments.reliability import cells  # noqa: E402
from bench.instruments.reliability import process_cell as PC, run_matrix as RM  # noqa: E402

DB_SUFFIXES = (".db", ".db-wal", ".db-shm")
PASS = {"verdict": "PASS", "failed_bars": {}, "numbers": {}, "inconclusive_bars": [], "applicable_bars": []}


def db_files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.name.endswith(DB_SUFFIXES))


@pytest.fixture
def held():
    """Open writers on the fake host's databases: their -wal / -shm stay on disk, as after a host crash."""
    cons = []
    yield cons
    for con in cons:
        con.close()


def live_dbs(home: Path, held: list) -> None:
    for name in ("state.db", "lcm.db"):
        con = sqlite3.connect(home / name, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE IF NOT EXISTS t (x)")
        con.execute("INSERT INTO t VALUES (1)")
        con.commit()
        held.append(con)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    tree = tmp_path / "tree"
    tree.mkdir()
    ctx = types.SimpleNamespace(
        out=tmp_path / "out", scratch=tmp_path / "scratch", scored=[], verdict="PASS",
        plugin={"ref": "r", "sha": "a" * 40, "tree": str(tree), "dir": "p", "enabled": "p", "engine": "e"},
        host={"python": sys.executable, "src": str(tmp_path), "sha": "s"},
        cell=cells.select("baseline/in-place/acp")[0])
    ctx.scratch.mkdir()

    def score(cell, d, db_dir=None):  # positive control: the copies exist while the cell is scored
        ctx.scored.append({"db_dir": Path(db_dir), "in_scratch": db_files(ctx.scratch), "in_out": db_files(ctx.out)})
        return {**PASS, "verdict": ctx.verdict, "failed_bars": {"B1": {}} if ctx.verdict == "FAIL" else {}}
    monkeypatch.setattr(RM.bars, "score", score)
    return ctx


def r1(ctx, monkeypatch, held, **kw):
    def fake_probe(argv, **k):  # one finished phase that leaves the host's databases open
        live_dbs(Path(k["env"]["HERMES_HOME"]), held)
        return types.SimpleNamespace(stdout='{"exit": "done"}', stderr="", returncode=0)
    monkeypatch.setattr(RM.subprocess, "run", fake_probe)
    return RM.run_cell(ctx.cell, "h", ctx.host, ctx.plugin, ctx.out, 5, kw.pop("keep", False),
                       identity={"method": "test"}, **{"scratch_root": ctx.scratch, **kw})


def r2(ctx, monkeypatch, held, network=False, **kw):
    def fake_phase(self, first):
        live_dbs(self.home, held)
        if network:
            PC.append(self.d / "proxy-attempts.jsonl", {"kind": "proxy", "host": "example.com"})
        return {"exit": "done", "next_turn": None}
    monkeypatch.setattr(PC.ProcessCell, "run_phase", fake_phase)
    return PC.run_cell_process(ctx.cell, "h", ctx.host, ctx.plugin, ctx.out, 5, kw.pop("keep", False),
                               identity={"method": "test"}, **{"scratch_root": ctx.scratch, **kw})


@pytest.mark.parametrize("verdict", ["PASS", "FAIL"])  # a FAIL used to keep its copies by default (--keep-dbs fail)
@pytest.mark.parametrize("run", [r1, r2], ids=["r1-in-process", "r2-acp-process"])
def test_a_finished_cell_leaves_no_database_behind_by_default(setup, monkeypatch, held, run, verdict):
    setup.verdict = verdict
    rec = run(setup, monkeypatch, held)
    assert rec["verdict"] == verdict
    [seen] = setup.scored
    assert seen["db_dir"].parent.parent == setup.scratch.resolve()  # scored from <scratch>/lcmx-cell-*/db, not from --out
    assert {"db/lcm.db", "db/state.db", "hermes-home/state.db-wal", "hermes-home/lcm.db-shm"} <= \
        {p.split("/", 1)[1] for p in seen["in_scratch"]}
    assert seen["in_out"] == []
    assert db_files(setup.out) == [] and list(setup.scratch.iterdir()) == []
    d = Path(rec["dir"])
    assert json.loads((d / "verdict.json").read_text())["verdict"] == verdict and (d / "cell.json").exists()
    assert not (d / "db").exists() and not (d / "hermes-home").exists()


@pytest.mark.parametrize("run", [r1, r2], ids=["r1-in-process", "r2-acp-process"])
def test_keep_flags_move_the_copies_into_the_cell_dir(setup, monkeypatch, held, run):
    rec = run(setup, monkeypatch, held, keep_dbs="all", keep=True)
    d = Path(rec["dir"])
    assert {"db/lcm.db", "db/state.db", "hermes-home/state.db", "hermes-home/lcm.db-wal"} <= set(db_files(d))
    assert list(setup.scratch.iterdir()) == []


@pytest.mark.parametrize("run", [r1, r2], ids=["r1-in-process", "r2-acp-process"])
def test_keep_dbs_fail_drops_a_clean_pass(setup, monkeypatch, held, run):
    run(setup, monkeypatch, held, keep_dbs="fail")
    assert db_files(setup.out) == [] and list(setup.scratch.iterdir()) == []


def test_r2_error_exit_releases_the_scratch(setup, monkeypatch, held):
    """The R2 early ERROR returns (network, provenance, unexpected requests) used to skip every cleanup."""
    rec = r2(setup, monkeypatch, held, network=True)
    assert rec["verdict"] == "ERROR" and "non-localhost" in rec["reason"]
    assert db_files(setup.out) == [] and list(setup.scratch.iterdir()) == []
    rec = r2(setup, monkeypatch, held, network=True, keep_dbs="fail")  # an ERROR is kept on request
    assert set(db_files(Path(rec["dir"]))) == {"db/lcm.db", "db/state.db"}


def test_a_harness_failure_releases_the_scratch(setup, monkeypatch):
    def broken(argv, **k):
        raise RuntimeError("harness bug")
    monkeypatch.setattr(RM.subprocess, "run", broken)
    with pytest.raises(RuntimeError):
        RM.run_cell(setup.cell, "h", setup.host, setup.plugin, setup.out, 5, False, identity={"method": "test"},
                    scratch_root=setup.scratch)
    assert list(setup.scratch.iterdir()) == []


def test_a_failed_keep_move_deletes_nothing(setup, monkeypatch, held):
    """Review (#786): the cleanup deleted the scratch even when moving a kept copy into the cell dir failed
    (a cross-filesystem copy that runs out of space), so the copies the flags asked to keep were lost."""
    def disk_full(src, dst):
        raise OSError(errno.ENOSPC, "injected disk full")
    monkeypatch.setattr(RM.shutil, "move", disk_full)
    with pytest.raises(OSError, match="nothing deleted"):
        r1(setup, monkeypatch, held, keep_dbs="all")
    [s] = list(setup.scratch.iterdir())
    assert {"db/lcm.db", "db/state.db", "hermes-home/state.db"} <= set(db_files(s))


def test_a_scratch_dir_that_cannot_be_deleted_is_an_error(setup, monkeypatch, held):
    """Review (#786): rmtree(ignore_errors=True) hid a failed cleanup, leaving the cell's DBs in $TMPDIR silently."""
    real = RM.shutil.rmtree
    monkeypatch.setattr(RM.shutil, "rmtree", lambda path, **kw: None if Path(path).parent == setup.scratch.resolve()
                        else real(path, **kw))  # the scratch dir's own delete fails: nothing is removed
    with pytest.raises(OSError, match="could not be fully deleted"):
        r1(setup, monkeypatch, held)
    [s] = list(setup.scratch.iterdir())
    assert {"db/lcm.db", "db/state.db"} <= set(db_files(s))


def test_default_scratch_root_is_tmpdir(setup, monkeypatch, held, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmpdir"))
    (tmp_path / "tmpdir").mkdir()
    rec = r1(setup, monkeypatch, held, scratch_root=None)
    assert rec["verdict"] == "PASS" and setup.scored[0]["db_dir"].parent.parent == (tmp_path / "tmpdir").resolve()
    assert list((tmp_path / "tmpdir").iterdir()) == [] and db_files(setup.out) == []


def test_cli_refuses_a_scratch_root_under_the_live_hermes_and_unknown_keep_values(capsys):
    out = Path(__file__).resolve().parent / "no-such-reliability-out"  # never created: both refusals come first
    args = ["--hosts", "h", "--plugin-ref", "HEAD", "--cells", "all", "--out", str(out)]
    live = RM.H.real_hermes_dir() / "lcmx-cell-scratch-test"
    with pytest.raises(SystemExit) as exc:
        RM.main(args + ["--scratch-root", str(live)])
    assert exc.value.code == 2 and "--scratch-root must not be under the live ~/.hermes" in capsys.readouterr().err
    assert not live.exists() and not out.exists()
    with pytest.raises(SystemExit):
        RM.main(args + ["--keep-dbs", "some"])


@pytest.mark.parametrize("cell_id", [f"drain/{size}/{mode}" for size in ("hidden-backlog", "hidden-backlog-large")
                                    for mode in ("in-place", "rotation")])
def test_r1_declines_the_acp_process_only_drain_fixture_before_any_probe(tmp_path, monkeypatch, cell_id):
    """R1 has no per-compaction counters: every drain cell is declined before any scratch or probe."""
    from bench.instruments.reliability import cells, run_matrix
    cell = cells.select(cell_id)[0]
    monkeypatch.setattr(run_matrix.subprocess, "run", lambda *a, **k: pytest.fail("R1 ran a drain probe"))
    out, scratch = tmp_path / "out", tmp_path / "scratch"
    out.mkdir()
    scratch.mkdir()
    rec = run_matrix.run_cell(cell, "h", {"src": "/nonexistent", "python": "/nonexistent", "sha": "b" * 40},
                              {"sha": "a" * 40, "ref": "HEAD", "dir": "lcm-x", "tree": "/nonexistent",
                               "engine": "lcm-x", "enabled": "lcm-x"}, out, 1, False,
                              identity={"sha": "b" * 40}, scratch_root=scratch)
    assert rec["verdict"] == "UNSUPPORTED", rec
    assert rec["reason"] == "drain cells need the R2 observer's per-compaction counters (acp-process only)"
    assert list(scratch.iterdir()) == []


def test_global_lcm_override_wins_in_both_process_phases_and_config(setup, monkeypatch, held):
    setup.cell = cells.select("drain/hidden-backlog-large/in-place")[0]
    seen = []

    def phase(self, first):
        live_dbs(self.home, held)
        seen.append((self.phase, self.env()["LCM_CONTEXT_THRESHOLD"], (self.home / "config.yaml").read_text()))
        self.fired.add("clean_exit_before_turn")
        return {"exit": "clean_exit", "next_turn": 151} if self.phase == "A" else {"exit": "done"}
    monkeypatch.setattr(PC.ProcessCell, "run_phase", phase)
    monkeypatch.setattr(PC, "forget_host_rows", lambda *a: 1)
    rec = PC.run_cell_process(setup.cell, "h", setup.host, setup.plugin, setup.out, 5, False,
                               lcm_env={"LCM_CONTEXT_THRESHOLD": "0.75"}, identity={"method": "test"},
                               scratch_root=setup.scratch)
    assert rec["verdict"] == "PASS" and [p for p, _, _ in seen] == ["A", "B"]
    assert all(value == "0.75" and "context_threshold: 0.75" in config for _, value, config in seen)


@pytest.mark.parametrize("returncode,killed", [(0, False), (1, False), (-15, False), (-15, True), (-9, True), (0, True)])
def test_clean_exit_checks_close_before_mutation_or_restore(setup, monkeypatch, held, returncode, killed):
    setup.cell = {**cells.select("drain/hidden-backlog-large/in-place")[0], "turns": 2,
                  "faults": [{"kind": "clean_exit_before_turn", "turn": 2}], "final_compaction_check": False}
    steps = []

    class Peer:
        killed = False

        def __init__(self, *a):
            pass

        def initialize(self, *a):
            pass

        def new_session(self, *a):
            return "session"

        def load_session(self, *a):
            steps.append("restore")

        def close(self):
            steps.append("close")
            self.killed = killed
            return returncode
    monkeypatch.setattr(PC.AD, "AcpProcess", Peer)
    monkeypatch.setattr(PC.ProcessCell, "argv", lambda self: [])
    monkeypatch.setattr(PC.ProcessCell, "turn", lambda self, t: live_dbs(self.home, held))
    monkeypatch.setattr(PC, "forget_host_rows", lambda *a: steps.append("forget") or 1)
    rec = PC.run_cell_process(setup.cell, "h", setup.host, setup.plugin, setup.out, 5, False,
                               identity={"method": "test"}, scratch_root=setup.scratch)
    d = Path(rec["dir"])
    first = json.loads((d / "phase-A.json").read_text())
    if returncode in (0, -15) and not killed:  # EOF, or the harness's own SIGTERM within the grace (#801 CI)
        assert first["exit"] == "clean_exit" and steps == ["close", "forget", "restore", "close"]
    else:
        assert first["exit"] == "error" and "clean exit failed" in first["reason"]
        assert str(returncode) in first["reason"] and f"killed={killed}" in first["reason"]
        assert rec["verdict"] == "ERROR" and steps == ["close"]
        assert not (d / "phase-B.json").exists() and not (d / "fixture.jsonl").exists()
        assert not (d / "faults-fired.jsonl").exists()
