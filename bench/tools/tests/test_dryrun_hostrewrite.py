"""Read-only dry-run proof against synthetic host and plugin stores."""
import hashlib
import sqlite3

from bench.instruments.gate_kit.tests.test_audit import Home
from bench.tools.dryrun_hostrewrite import run
from tests.test_host_rewrite_replay import A, B, COMPOSITE, FOLD, SNAP, TODO, row


def test_flags_only_unexpected_fold_and_snapshot(tmp_path, capsys):
    displayed = row("user", TODO, 110)
    original = [("S", A), ("S", COMPOSITE), ("S", B)]
    stored = original + [("S", FOLD), ("S", SNAP), ("S", displayed)]
    host = [(s, dict(m, _ts=m["timestamp"])) for s, m in original + [("S", displayed)]]
    stored = [(s, dict(m, _obs=m.get("timestamp"))) for s, m in stored]
    home = Home(tmp_path, {"S": ("acp", None, 2000, "exit")}, host, stored)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in home.dir.iterdir()}
    run(home.state, home.lcm)
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["store_id=4 role=assistant arm=M stamp=100.0 length=10 parts=[1, 3] EXPECTED=no",
                     f"store_id=5 role=user arm=S stamp=None length={len(TODO)} parts=[] EXPECTED=no",
                     "summary M=1 S=1 EXPECTED=yes:0 merged_after_S=0"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in home.dir.iterdir()} == before
    run(home.state, home.lcm, "other")
    assert capsys.readouterr().out == "summary M=0 S=0 EXPECTED=yes:0 merged_after_S=0\n"


def test_expected_fold_is_a_design_failure(tmp_path, capsys):
    home = Home(tmp_path, {"S": ("acp", None, 2000, "exit")}, [("S", dict(FOLD, _ts=100))],
                [("S", dict(m, _obs=m["timestamp"])) for m in [A, B, FOLD]])
    run(home.state, home.lcm)
    assert "EXPECTED=yes\nsummary M=1 S=0 EXPECTED=yes:1" in capsys.readouterr().out
    # A read-only URI is used even on files writable by this process.
    from bench.instruments.gate_kit.audit import ro
    with ro(home.lcm) as con:
        try:
            con.execute("DELETE FROM messages")
        except sqlite3.OperationalError as exc:
            assert "readonly" in str(exc)
        else:
            raise AssertionError("dry-run connection allowed a write")


def test_merged_prompt_after_snapshot_ids_only(tmp_path, capsys):
    merged = row("user", TODO + "\n\nprompt two", 120)
    original = [("S", A), ("S", COMPOSITE), ("S", B)]
    stored = original + [("S", SNAP), ("other", merged), ("S", merged)]
    host = [(s, dict(m, _ts=m["timestamp"])) for s, m in original + [("S", merged)]]
    home = Home(tmp_path, {"S": ("acp", None, 2000, "exit"), "other": ("acp", None, 2000, "exit")},
                host, [(s, dict(m, _obs=m.get("timestamp"))) for s, m in stored])
    run(home.state, home.lcm)
    lines = capsys.readouterr().out.splitlines()
    assert "merged_after_S store_id=6 of=4" in lines
    assert not any("store_id=5" in line for line in lines)
    assert lines[-1] == "summary M=0 S=1 EXPECTED=yes:0 merged_after_S=1"
    assert not any("prompt two" in line or TODO in line for line in lines)
