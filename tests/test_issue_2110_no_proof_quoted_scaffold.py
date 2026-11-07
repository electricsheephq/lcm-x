"""#2110 characterization: under a legacy (v2/v3) proof, cursor reconciliation gives identity None to a row that
passes the verified-scaffold test: a user row that exactly quotes one complete summary part of its own session, or
one that starts with the preserved-objective or preserved-task-list prefix. Does a real user turn of that shape get skipped after a
restart? Each case places the turn after the matched stored tail and before its last row. Controls: a plain user turn
in the same places, and the current v4 proof. On main the legacy cells before the last row lose the turn (strict
xfails); every other cell stores it once."""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from hermes_lcm.engine import LCMEngine
from hermes_lcm.reconcile import _PRESERVED_OBJECTIVE_CONTEXT_PREFIX, _PRESERVED_TODO_CONTEXT_PREFIX
from tests.test_compression_boundary import _config, _turn
from tests.test_issue_488_emission_proof import _phase1_compacted_engine, _summary_block
from tests.test_issue_1116_stub_first_exit_commit_proof import (  # noqa: F401 (fixtures)
    host_for, leave_proof_stale, stub_first_session,
)


def _legacy_restart(engine, tmp_path, host, version):
    """Store the host list, downgrade the durable proof to a legacy version (no emissions), restart."""
    engine.ingest(host)
    key = "compaction_commit_proof:S0"
    payload = engine._store.read_metadata_json(key)
    if version != 4:  # 4: the current proof, kept as written (the control)
        payload["version"] = version
        payload.pop("emissions", None)
        engine._store.write_metadata_json([key], json.dumps(payload, sort_keys=True))
    engine.shutdown()
    resumed = LCMEngine(config=_config(tmp_path, fresh_tail_count=0), hermes_home=str(tmp_path / "home"))
    resumed.on_session_start("S0", platform="acp", context_length=200_000)
    if resumed._active_emission_proof()["version"] != version:
        pytest.fail("setup: the restarted engine does not read the intended proof version")
    return resumed


def _contents(engine, role):
    return [c for (c,) in engine._store._conn.execute(
        "SELECT content FROM messages WHERE role = ? ORDER BY store_id", (role,))]


_LOST = pytest.mark.xfail(strict=True, raises=AssertionError, reason="#2110: under a legacy (v2/v3) proof a scaffold-shaped user turn placed "
                                                                     "before a stored row gets identity None and is skipped")
_HEADER = pytest.mark.xfail(strict=True, raises=AssertionError, reason="#2418: a user turn that begins with the host's "
                                                                       "task-list header is skipped when reconciled")
_HEADER_AND_LOST = pytest.mark.xfail(strict=True, raises=AssertionError, reason="#2418 (the task-list header) and "
                                                                                "#2110 (no proof): both skip the turn")


def _legacy_mark(place, shape, version):
    if shape == "todo-prefix":
        if version == 4:
            return _HEADER
        return _HEADER_AND_LOST if place == "before-last-row" else ()
    return _LOST if place == "before-last-row" and shape != "plain" and version != 4 else ()


_CELLS = [pytest.param(place, shape, version, id=f"{place}-{shape}-v{version}", marks=_legacy_mark(place, shape, version))
          for place in ("after-tail", "before-last-row") for shape in ("summary-part", "objective-prefix", "todo-prefix", "plain")
          for version in (3, 2, 4)]


@pytest.mark.parametrize("place,shape,version", _CELLS)
def test_a_user_turn_shaped_like_scaffold_is_stored_after_a_legacy_restart(tmp_path, monkeypatch, place, shape,
                                                                            version):
    engine, _pre, compressed = _phase1_compacted_engine(tmp_path, monkeypatch, tail=0)
    quoted = {"summary-part": _summary_block(engine, compressed),
              "objective-prefix": f"{_PRESERVED_OBJECTIVE_CONTEXT_PREFIX}\nKeep this objective exactly as I wrote it.",
              "todo-prefix": f"{_PRESERVED_TODO_CONTEXT_PREFIX}\n- [ ] the task list exactly as I wrote it",
              "plain": "A plain user turn with no scaffold shape."}[shape]
    host = [dict(m) for m in compressed] + [_turn(13)[1]]
    engine = _legacy_restart(engine, tmp_path, host, version)
    try:
        turn = {"role": "user", "content": quoted}
        if engine._is_verified_replay_scaffold_message(turn) != (shape != "plain"):  # what the no-proof path skips
            pytest.fail("setup: the turn is not the shape under test")
        reply = {"role": "assistant", "content": "reply to the quoting turn"}
        view = [*host, turn, reply] if place == "after-tail" else [*host[:-1], turn, host[-1], reply]
        users_before = _contents(engine, "user").count(quoted)
        engine.ingest([dict(m) for m in view])
        if _contents(engine, "assistant").count(reply["content"]) != 1:
            pytest.fail("setup: the batch was not ingested")
        assert _contents(engine, "user").count(quoted) == users_before + 1, "the quoting user turn was skipped"
        engine.ingest([dict(m) for m in view])
        assert _contents(engine, "user").count(quoted) == users_before + 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("stale", [True, False], ids=["stale-v4-proof", "fresh-stub-first-proof"])
@pytest.mark.parametrize("shape", ["summary-part", "objective-prefix", pytest.param("todo-prefix", marks=_HEADER), "plain"])
def test_a_scaffold_shaped_user_turn_is_stored_past_a_stale_v4_proof(stub_first_session, caplog, shape, stale):  # noqa: F811
    """The retired-v4 state (#671): a stale proof (left by an earlier version) is retired only when the re-reconcile
    replays stored rows, so a scaffold-shaped turn before the last row keeps today's decision and is stored. A fresh
    v0.30.0 stub-first pass writes its own proof and retires nothing."""
    host = stub_first_session("summary", stub_first=not stale)
    if stale:
        leave_proof_stale(host, caplog)
    quoted = {"summary-part": host.live[0]["content"].split("\n\n---\n\n")[0],
              "objective-prefix": f"{_PRESERVED_OBJECTIVE_CONTEXT_PREFIX}\nKeep this objective exactly as I wrote it.",
              "todo-prefix": f"{_PRESERVED_TODO_CONTEXT_PREFIX}\n- [ ] the task list exactly as I wrote it",
              "plain": "A plain user turn with no scaffold shape."}[shape]
    if host.engine._is_verified_replay_scaffold_message({"role": "user", "content": quoted}) != (shape != "plain"):
        pytest.fail("setup: the quoted turn is not the shape under test")
    host.clock += 0.5
    host.live = [*host.live[:-1], {"role": "user", "content": quoted, "timestamp": host.clock}, host.live[-1]]
    host.cold_rebind()
    rows = host.engine._store.get_session_messages(host.engine._session_id)
    assert sum(row["content"] == quoted for row in rows) == 1, "the quoting user turn was skipped"


_RETIRED_LOST = pytest.mark.xfail(strict=True, raises=AssertionError,
                                  reason="#2110: once a stale v4 proof is retired, the identity anchor treats a "
                                         "scaffold-shaped user turn placed before a stored row as LCM scaffold "
                                         "(identity None) and skips it")


@pytest.mark.parametrize("shape", [pytest.param("summary-part", marks=_RETIRED_LOST),
                                   pytest.param("objective-prefix", marks=_RETIRED_LOST),
                                   pytest.param("todo-prefix", marks=_HEADER_AND_LOST), "plain"])
def test_a_scaffold_shaped_user_turn_is_stored_after_a_v4_proof_is_retired(stub_first_session, caplog, shape):  # noqa: F811
    """A stale v4 proof (left by an earlier version) is retired by a cold rebind that replays stored rows (here with a
    plain new turn). A later scaffold-shaped turn before the last row must still be stored; a plain one is. The loss
    runs through the identity anchor's scaffold check (``_identity_is_lcm_scaffold``), not the no-proof cursor."""
    host = stub_first_session("summary", stub_first=False)
    leave_proof_stale(host, caplog)
    host.clock += 0.5
    host.live = [*host.live[:-1], {"role": "user", "content": "A plain user turn that retires the stale proof.",
                                   "timestamp": host.clock}, host.live[-1]]
    host.cold_rebind()
    if getattr(host.engine, "_retired_durable_proof", None) is None:
        pytest.fail("setup: the stale proof was not retired")
    quoted = {"summary-part": host.live[0]["content"].split("\n\n---\n\n")[0],
              "objective-prefix": f"{_PRESERVED_OBJECTIVE_CONTEXT_PREFIX}\nKeep this objective exactly as I wrote it.",
              "todo-prefix": f"{_PRESERVED_TODO_CONTEXT_PREFIX}\n- [ ] the task list exactly as I wrote it",
              "plain": "A second plain user turn."}[shape]
    if host.engine._is_verified_replay_scaffold_message({"role": "user", "content": quoted}) != (shape != "plain"):
        pytest.fail("setup: the quoted turn is not the shape under test")
    host.clock += 0.5
    host.live = [*host.live[:-1], {"role": "user", "content": quoted, "timestamp": host.clock}, host.live[-1]]
    host.engine.ingest(deepcopy(host.live))
    rows = host.engine._store.get_session_messages(host.engine._session_id)
    assert sum(row["content"] == quoted for row in rows) == 1, "the quoting user turn was skipped"
