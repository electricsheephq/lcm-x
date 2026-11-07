"""Synthetic selection and deterministic workload contract."""

import hashlib
import json
import shlex
import sqlite3
import subprocess

import pytest

from bench.instruments.d5_clone.plan import (
    LARGE_COMMAND,
    SMALL_COMMAND,
    PlanError,
    build_plan,
    choose_sessions,
    declaration,
    plan_sha256,
)


@pytest.fixture
def databases(tmp_path):
    host_path, lcm_path = tmp_path / "host ?#.db", tmp_path / "store ?#.db"
    with sqlite3.connect(host_path) as host, sqlite3.connect(lcm_path) as lcm:
        host.executescript("CREATE TABLE sessions(id TEXT, end_reason TEXT);"
                           "CREATE TABLE messages(session_id TEXT);")
        lcm.execute("CREATE TABLE messages(session_id TEXT)")
        for sid, count, end in [("large", 1200, None), ("tie-a", 40, None),
                                ("tie-b", 40, None), ("tiny", 19, None),
                                ("ended", 2000, "done"), ("uncovered", 3000, None)]:
            host.execute("INSERT INTO sessions VALUES (?, ?)", (sid, end))
            host.executemany("INSERT INTO messages VALUES (?)", [(sid,)] * (count // 10))
            lcm.executemany("INSERT INTO messages VALUES (?)", [(sid,)] * count)
    return host_path, lcm_path


def test_choose_sessions(databases):
    seen = []

    def covered(sid):
        seen.append(sid)
        return sid != "uncovered"

    selected = choose_sessions(*databases, covered)
    ties = sorted(["tie-a", "tie-b"], key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    assert selected == [dict(name=name, session_id=sid, lcm_rows=count, host_rows=count // 10,
                             chat_id=f"d5-{name}")
                        for name, sid, count in zip(["L", "S1", "S2"], ["large", *ties],
                                                    [1200, 40, 40])]
    assert "ended" not in seen
    assert "uncovered" in seen


@pytest.mark.parametrize("kwargs", [{"min_large": 1201}, {"n_small": 3}])
def test_missing_slot_has_counts_only(databases, kwargs):
    with pytest.raises(PlanError) as exc:
        choose_sessions(*databases, lambda sid: sid != "uncovered", **kwargs)
    message = str(exc.value)
    assert "eligible=4" in message
    assert all(sid not in message for sid in ("tie-a", "tie-b", "uncovered", "ended"))


@pytest.fixture
def plan(databases):
    conversations = choose_sessions(*databases, lambda sid: sid != "uncovered")
    return build_plan(list(reversed(conversations)), nonce="synthetic-nonce")


def test_workload_and_provider(plan):
    assert plan["schema"] == "d5-clone-plan/1"
    steps = [s for epoch in plan["epochs"] for s in epoch["steps"]]
    assert len(steps) == 36
    assert sum(s["kind"] == "cold" for s in steps) == 12
    assert len({s["user_text"] for s in steps}) == 36
    assert set(plan["provider"]["steps"]) == {s["id"] for s in steps}
    for e, epoch in enumerate(plan["epochs"]):
        assert epoch["epoch"] == e
        assert epoch["restart_before"] == (e > 0)
        assert epoch["min_seconds"] == (180 if e == 2 else 0)
        assert [s["conversation"] for s in epoch["steps"]] == ["L"] * 3 + ["S1"] * 3 + ["S2"] * 3
        for s in epoch["steps"]:
            k = int(s["id"].rsplit("-", 1)[1])
            digest = hashlib.sha256(("synthetic-nonce" + s["id"]).encode()).hexdigest()
            rewrite = e in (0, 2) and k == 1
            assert s["kind"] == ("cold" if k == 0 else "warm")
            assert s["rewrite"] == ("whitespace" if rewrite else None)
            assert s["user_text"] == f"d5 plan v1 {s['id']} {digest[:16]}" + ("  \r\n" if rewrite else "")
            if rewrite:
                assert plan["epochs"][e + 1]["steps"][k - 1]["kind"] == "cold"
            tool = ("large" if k == 1 else "small") if (
                e in (1, 2) and s["conversation"] == "L" and k in (1, 2)) else None
            assert s["tools"] == tool
            entries = plan["provider"]["steps"][s["id"]]
            assert entries[-1] == {"text": f"d5 reply {s['id']} {digest[:12]}"}
            assert len(entries) == (2 if tool else 1)
            if tool:
                assert entries[0] == {"tool_calls": [{"id": f"call_{s['id']}_1", "name": "terminal",
                                        "arguments": {"command": LARGE_COMMAND if tool == "large"
                                                      else SMALL_COMMAND}}]}
    digest = hashlib.sha256(b"synthetic-nonce").hexdigest()[:12]
    assert plan["cron"] == dict(schedule="every 2m", repeat=1, created_before_epoch=2,
                                prompt=f"d5 cron {digest}")
    assert plan["provider"]["cron"] == [{"text": f"d5 cron reply {digest}"}]


def test_determinism_and_declaration(plan):
    assert plan_sha256(plan) == plan_sha256(build_plan(plan["conversations"], nonce="synthetic-nonce"))
    assert plan_sha256(plan) != plan_sha256(build_plan(plan["conversations"], nonce="different-nonce"))
    expected = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False).encode()).hexdigest()
    assert plan_sha256(plan) == expected
    declared = declaration(plan)
    assert declared["steps_by_kind"] == {"cold": 12, "warm": 24}
    assert declared["cold_binds"] == {"L": 4, "S1": 4, "S2": 4}
    assert declared["rewrite_steps"] == 6
    assert declared["tool_steps"] == 4
    assert declared["plan_sha256"] == expected
    assert declared["cron"] == dict(schedule="every 2m", repeat=1, created_before_epoch=2)
    serialized = json.dumps(declared)
    assert "user_text" not in serialized
    assert "synthetic-nonce" not in serialized
    assert all(s["user_text"] not in serialized for e in plan["epochs"] for s in e["steps"])


@pytest.mark.parametrize("command,size", [(LARGE_COMMAND, 204800), (SMALL_COMMAND, 64)])
def test_tool_output(command, size):
    result = subprocess.run(shlex.split(command), check=True, capture_output=True)
    assert result.stdout == b"d" * size
    assert result.stderr == b""
