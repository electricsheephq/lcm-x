"""#2110 characterization: under a legacy (v2/v3) proof, cursor reconciliation gives identity None to a row that
passes the verified-scaffold test: a user row that exactly quotes one complete summary part of its own session, or one
that starts with the preserved-objective or preserved-task-list prefix. Does a real user turn of that shape get skipped
after a restart? Each case places the turn after the matched stored tail and before its last row. Controls: a plain
user turn in the same places, and the current v4 proof. Strict xfails on main: the legacy cells before the last row
(#2110), the cells after a stale v4 proof is retired (#2110, through the identity anchor), and the task-list-header
cells (#2418, apart from the legacy cells after the tail); every other cell stores the turn once."""
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


class ScaffoldTurnSkipped(AssertionError):
    """Only the complete #2110/#2418 loss shape, never a setup assertion."""


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


def _stored(engine):
    return engine._store._conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? AND role != 'system' ORDER BY store_id",
        (engine._session_id,)).fetchall()


_LOST = pytest.mark.xfail(strict=True, raises=ScaffoldTurnSkipped, reason="#2110: under a legacy (v2/v3) proof a scaffold-shaped user turn placed "
                                                                     "before a stored row gets identity None and is skipped")
_HEADER = pytest.mark.xfail(strict=True, raises=ScaffoldTurnSkipped, reason="#2418: a user turn that begins with the host's "
                                                                       "task-list header is skipped when reconciled")
_HEADER_AND_LOST = pytest.mark.xfail(strict=True, raises=ScaffoldTurnSkipped, reason="#2418 (the task-list header) and "
                                                                                "#2110 (no proof): both skip the turn")


def _expect_stored_once(stored, fixed, known=None):
    """Compare whole stored lists; only the exact known loss may satisfy the xfail."""
    if stored == fixed:
        return
    if known is not None and stored == known:
        raise ScaffoldTurnSkipped("#2110/#2418 known shape: the scaffold-shaped user turn is skipped")
    first_difference = next((i for i, (a, b) in enumerate(zip(stored, fixed)) if a != b),
                            min(len(stored), len(fixed)))
    pytest.fail(f"not the #2110/#2418 shape: stored rows={len(stored)}, fixed rows={len(fixed)}, "
                f"known rows={len(known) if known is not None else 0}, first difference={first_difference}")


@pytest.mark.parametrize("order,outcome", [
    pytest.param(["head", "last", "reply"], AssertionError, id="known"),
    pytest.param(["head", "turn", "last", "reply"], None, id="fixed"),
    pytest.param(["head", "turn", "last", "last", "reply"], pytest.fail.Exception, id="last-twice"),
    pytest.param(["head", "last", "last", "reply"], pytest.fail.Exception, id="lost-and-last-twice"),
    pytest.param(["head", "turn", "reply", "last"], pytest.fail.Exception, id="reply-before-last"),
    pytest.param(["head", "reply", "last"], pytest.fail.Exception, id="lost-and-reply-before-last"),
])
def test_stored_once_accepts_only_the_known_or_fixed_list(order, outcome):
    rows = {"head": ("user", "head"), "turn": ("user", "turn"),
            "last": ("assistant", "last"), "reply": ("assistant", "reply")}
    stored = [rows[name] for name in order]
    fixed = [rows[name] for name in ("head", "turn", "last", "reply")]
    known = [rows[name] for name in ("head", "last", "reply")]
    if outcome is None:
        _expect_stored_once(stored, fixed, known)
    else:
        with pytest.raises(outcome):
            _expect_stored_once(stored, fixed, known)


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
        # Compaction keeps the original raw history; the emitted summary head is not stored.
        prefix = [(row["role"], row["content"]) for row in [*_pre, host[-1]] if row["role"] != "system"]
        _expect_stored_once(_stored(engine), prefix)
        # Before-last reconciliation stores the last row again, except the v4 task-header path, which appends only
        # reply: unstamped rewrites cost duplicates, never loss (#2203, docs/exactly-once-ingest.md section 3.5).
        replay_last = place == "before-last-row" and not (shape == "todo-prefix" and version == 4)
        additions = [row for row in view if row is turn or row is reply or (replay_last and row is host[-1])]
        fixed = prefix + [(row["role"], row["content"]) for row in additions]
        known = (prefix + [(row["role"], row["content"]) for row in additions if row is not turn]
                 if _legacy_mark(place, shape, version) else None)
        engine.ingest([dict(m) for m in view])
        _expect_stored_once(_stored(engine), fixed, known)
        engine.ingest([dict(m) for m in view])
        _expect_stored_once(_stored(engine), fixed)
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
    prefix = _stored(host.engine)
    host.clock += 0.5
    host.live = [*host.live[:-1], {"role": "user", "content": quoted, "timestamp": host.clock}, host.live[-1]]
    # Raw history stays in store order; stale-proof recovery also appends the summary head for these two shapes.
    # That is main's shape today, pinned so any change shows; it is a surplus row to check against section 5.
    additions = ([host.live[0]] if stale and shape in ("summary-part", "objective-prefix") else []) + [host.live[-2]]
    fixed = prefix + [(row["role"], row["content"]) for row in additions]
    host.cold_rebind()
    _expect_stored_once(_stored(host.engine), fixed, prefix if shape == "todo-prefix" else None)


_RETIRED_LOST = pytest.mark.xfail(strict=True, raises=ScaffoldTurnSkipped,
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
    prefix = _stored(host.engine)
    host.clock += 0.5
    host.live = [*host.live[:-1], {"role": "user", "content": quoted, "timestamp": host.clock}, host.live[-1]]
    # Retired-proof reconciliation appends only the new turn after the unchanged raw history.
    fixed = prefix + [(host.live[-2]["role"], host.live[-2]["content"])]
    host.engine.ingest(deepcopy(host.live))
    _expect_stored_once(_stored(host.engine), fixed, prefix if shape != "plain" else None)
