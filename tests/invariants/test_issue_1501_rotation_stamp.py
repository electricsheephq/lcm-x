"""I1: a proven rotation replay survives a carried host stamp, without losing fresh repeats."""
from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tests.harness import engine_context


def message(text: Any, stamp: float | None, role: str = "user") -> dict[str, Any]:
    return {"role": role, "content": text, "timestamp": stamp}


def lineage(root: Path, verified: bool = True) -> None:
    home = root / "home"
    home.mkdir(exist_ok=True)
    with sqlite3.connect(home / "state.db") as db:
        db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.executemany("INSERT INTO sessions VALUES (?, ?, ?)",
                       [("parent", None, "compression" if verified else "normal"), ("child", "parent", None)])


def test_rotation_replay_then_fresh_repeat(tmp_path: Path) -> None:
    """Real ingest and cold restart: the stored mis-stamp stays, the replay does not duplicate it."""
    lineage(tmp_path)
    previous, current = message("interrupted turn", 10.), message("next request", 20.)
    predecessor = message("earlier reply", 9., "assistant")
    reply = message("reply", 21., "assistant")
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest([predecessor, previous])
        engine.on_session_start("child", platform="cli", context_length=128000, conversation_id="conv",
                                boundary_reason="compression")
        engine.ingest([predecessor, {**current, "timestamp": 10.}, reply])
        before = engine._store.get_session_messages("child")
        assert [r["observed_at"] for r in before] == [10., 21.]
    with engine_context(tmp_path, window=128000, session_id="child", conversation_id="conv") as engine:
        replay = [previous, current, reply]
        engine.ingest(deepcopy(replay))
        assert engine._store.get_session_messages("child") == before
        fresh = message("next request", 30.)
        engine.ingest([*deepcopy(replay), fresh])
        rows = engine._store.get_session_messages("child")
        assert [r["content"] for r in rows] == ["next request", "reply", "next request"]
        assert [r["observed_at"] for r in rows] == [10., 21., 30.]
        engine.ingest([*deepcopy(replay), fresh])
        assert engine._store.get_session_messages("child") == rows


@pytest.mark.parametrize("guard", ["accepted", "ancestry", "gap", "previous-unproven", "null",
                                   "equal-stamp", "consumed", "content", "uid", "consecutive",
                                   "assistant-predecessor", "backward-stamp"])
def test_carry_tolerance_requires_every_proof(tmp_path: Path, guard: str) -> None:
    """Exercise the audit plan on real storage; each missing proof stores the uncertain occurrence."""
    lineage(tmp_path, guard != "ancestry")
    previous = message("previous", 10., "assistant" if guard == "assistant-predecessor" else "user")
    current = message("current", 20.)
    predecessor = message("earlier reply", 9., "assistant")
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest([predecessor, previous])
        engine.on_session_start("child", platform="cli", context_length=128000, conversation_id="conv")
        stored = [message("gap", 11., "assistant")] if guard == "gap" else []
        stored += [message("current", None if guard == "null" else 10.)]
        if guard == "consecutive":
            stored += [message("third", 20.)]
        engine.ingest([predecessor, *stored])
        if guard == "previous-unproven":
            previous = message("previous", 9.)
        if guard == "equal-stamp":
            current = message("current", 10.)
        if guard == "backward-stamp":
            current = message("current", 5.)
        if guard == "content":
            current = message("different", 20.)
        if guard == "uid":
            current["message_uid"] = "current-occurrence"
        view = [previous, current]
        if guard == "consumed":
            view = [previous, message("current", 10.), current]
        if guard == "consecutive":
            view += [message("third", 30.)]
        plan = engine._identity_anchor_prematch(view, view, len(view), 0)
        if guard in {"accepted", "equal-stamp"}:
            assert plan["cursor"] == len(view)
        elif guard == "consecutive":
            assert plan["cursor"] == 2  # the tolerated second row cannot prove the third
        else:
            assert plan["cursor"] <= len(view) - 1
            assert len(view) - 1 not in plan.get("matched", {})
        matched = [int(row["store_id"]) for rows in plan.get("matched", {}).values() for row in rows]
        assert len(matched) == len(set(matched))


PARTS = [{"type": "text", "text": "same"}]


@pytest.mark.parametrize("previous_content,current_content", [
    ("same", "same"),
    (" same ", "same"),  # equal after the edge-whitespace normalization matching applies
    (PARTS, json.dumps(PARTS, ensure_ascii=False, sort_keys=True)),  # structured vs its canonical text
], ids=["identical", "edge-whitespace", "structured"])
def test_identical_predecessor_text_gives_no_carry_proof(tmp_path: Path, previous_content: Any,
                                                         current_content: Any) -> None:
    """Stored A@10, B@10, B@10 and view A@10, B@10, B@20: the fresh B@20 is stored, not absorbed."""
    lineage(tmp_path)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest([message("first", 10.)])
        engine.on_session_start("child", platform="cli", context_length=128000, conversation_id="conv")
        engine.ingest([message(current_content, 10.), message(current_content, 10.)])
        view = [message("first", 10.), message(previous_content, 10.), message(current_content, 20.)]
        plan = engine._identity_anchor_prematch(view, view, len(view), 0)
        assert plan["cursor"] <= len(view) - 1
        assert len(view) - 1 not in plan.get("matched", {})
    with engine_context(tmp_path, window=128000, session_id="child", conversation_id="conv") as engine:
        engine.ingest(deepcopy(view))
        assert 20. in [r["observed_at"] for r in engine._store.get_session_messages("child")]


def test_fresh_repeat_immediately_after_same_text(tmp_path: Path) -> None:
    lineage(tmp_path)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest([message("same", 10.)])
        engine.on_session_start("child", platform="cli", context_length=128000, conversation_id="conv")
        engine.ingest([message("same", 10.), message("same", 20.)])
        assert [r["observed_at"] for r in engine._store.get_session_messages("child")] == [20.]


def test_null_predecessor_requires_canonical_binding(tmp_path: Path) -> None:
    lineage(tmp_path)
    previous = {**message("previous", None), "message_uid": "previous-occurrence"}
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest([previous, message("current", 10.)])
        engine.on_session_start("child", platform="cli", context_length=128000, conversation_id="conv")
        view = [{**previous, "timestamp": 10.}, message("current", 20.)]
        plan = engine._identity_anchor_prematch(view, view, 2, 0)
        assert plan["cursor"] == 1  # a canonical NULL binding cannot prove a store-time carried stamp
        assert len(plan["backfill"]) == 1  # only the independently bound NULL predecessor
        assert len({r["store_id"] for group in plan["matched"].values() for r in group}) == 1


def test_alternate_stamp_alias_gives_no_carry_proof(tmp_path: Path) -> None:
    """Stored A@10, B@20 with a recorded alternate stamp 10 on B, view A@10, B@30: B@30 is stored."""
    lineage(tmp_path)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest([message("first", 10.)])
        engine.on_session_start("child", platform="cli", context_length=128000, conversation_id="conv")
        engine.ingest([message("second", 20.)])
        (row,) = engine._store.get_session_messages("child")
        engine._store.add_message_relations([[(int(row["store_id"]), "alt_stamp", None, None, 10.)]])
    view = [message("first", 10.), message("second", 30.)]
    with engine_context(tmp_path, window=128000, session_id="child", conversation_id="conv") as engine:
        engine.ingest(deepcopy(view))
        assert [r["observed_at"] for r in engine._store.get_session_messages("child")] == [20., 30.]


def test_model_switch_wrapper_gives_no_carry_proof(tmp_path: Path) -> None:
    """Stored first@10, wrapped-X@10, X@10 and view first@10, wrapped-X@10, X@20: X@20 is stored."""
    lineage(tmp_path)
    wrapped = "[Note: model was just switched from model-a to model-b via router.]\n\nrequest"
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest([message("first", 10.)])
        engine.on_session_start("child", platform="cli", context_length=128000, conversation_id="conv")
        engine.ingest([message(wrapped, 10.), message("request", 10.)])
    view = [message("first", 10.), message(wrapped, 10.), message("request", 20.)]
    with engine_context(tmp_path, window=128000, session_id="child", conversation_id="conv") as engine:
        engine.ingest(deepcopy(view))
        assert 20. in [r["observed_at"] for r in engine._store.get_session_messages("child")]


@pytest.mark.parametrize("cursor", [0, 2], ids=["cold-restart", "audit"])
def test_alternate_stamp_predecessor_gives_no_carry_proof(tmp_path: Path, cursor: int) -> None:
    """Stored A@5 with a recorded alternate stamp 10, then B@10 next to it; view A@10, B@20: B@20 is stored."""
    lineage(tmp_path)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest([message("first", 5.)])
        (prior,) = engine._store.get_session_messages("parent")
        engine.on_session_start("child", platform="cli", context_length=128000, conversation_id="conv")
        engine.ingest([message("second", 10.)])
        engine._store.add_message_relations([[(int(prior["store_id"]), "alt_stamp", None, None, 10.)]])
    view = [message("first", 10.), message("second", 20.)]
    with engine_context(tmp_path, window=128000, session_id="child", conversation_id="conv") as engine:
        plan = engine._identity_anchor_prematch(deepcopy(view), deepcopy(view), cursor, 0)
        assert [r["store_id"] for r in plan["matched"][0]] == [prior["store_id"]]  # A matches through its alias
        assert 1 not in plan["matched"]
        engine.ingest(deepcopy(view))
        assert [r["observed_at"] for r in engine._store.get_session_messages("child")] == [10., 20.]


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_collision_then_resend_keeps_fresh_occurrence(tmp_path: Path, cold: bool) -> None:
    """I1: retaining A while retrying B must not mistake a same-second collision for a host merge."""
    lineage(tmp_path)
    a = message([{"type": "text", "text": "distinct A"}], 1800000000.)
    b = message("repeat B", 1800000000.)
    later = [deepcopy(a), message("repeat B", 1800000010.)]
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest([deepcopy(a)])
        engine.on_session_start("child", platform="cli", context_length=128000, conversation_id="conv",
                                boundary_reason="compression")
        engine.ingest([deepcopy(a), deepcopy(b)])
        if not cold:
            engine.ingest(deepcopy(later))
            assert 1800000010. in [r["observed_at"] for r in engine._store.get_session_messages("child")]
    if cold:
        with engine_context(tmp_path, window=128000, session_id="child", conversation_id="conv") as engine:
            engine.ingest(deepcopy(later))
            assert 1800000010. in [r["observed_at"] for r in engine._store.get_session_messages("child")]


@pytest.mark.parametrize("shape", ["rotation", "same-session", "trim", "new-tail", "summary", "steer",
                                    "present-elsewhere", "wrong-q", "wrong-q-stamp", "no-q", "r3-null", "uid",
                                    "unverified-chain"])
def test_marker_requires_store_time_replacement_signature(tmp_path: Path, shape: str) -> None:
    lineage(tmp_path, shape != "unverified-chain")
    q = message("earlier reply", 9., "assistant")
    p = message("interrupted turn", 10.)
    r = message("next request", 10.)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest([p] if shape == "no-q" else [q, p])
        prior = engine._store.get_session_messages("parent")[-1]
        session = "parent" if shape == "same-session" else "child"
        if session == "child":
            engine.on_session_start(session, platform="cli", context_length=128000, conversation_id="conv",
                                    boundary_reason="compression")
        view = [deepcopy(q), r]
        if shape in {"trim", "new-tail", "no-q"}:
            view = [r]
        elif shape == "summary":
            view[0] = message("[Recent Summary (d0, node 1)]\nsummary", 9.)
        elif shape == "steer":
            view[0] = message("steer request", 9.)
        elif shape == "present-elsewhere":
            view = [deepcopy(p), *view]
        elif shape == "wrong-q":
            view[0] = message("different reply", 9., "assistant")
        elif shape == "wrong-q-stamp":
            view[0]["timestamp"] = 8.
        elif shape == "r3-null":
            r["content"] = "interrupted turn\n\nnext request"
        elif shape == "uid":
            r["message_uid"] = "new-occurrence"
        engine.ingest(deepcopy(view))
        rows = engine._store.get_session_messages(session)
        stored = rows[-1]
        relations = engine._store.get_message_relations([row["store_id"] for row in rows], "carried_stamp")
        if shape in {"rotation", "same-session"}:
            assert [(rel["store_id"], rel["related_store_id"], rel["ordinal"], rel["observed_at"])
                    for rel in relations] == [(stored["store_id"], prior["store_id"], None, 10.)]
        else:
            assert not relations
        if shape == "r3-null":
            assert stored["content"] == "next request"
            assert stored["observed_at"] is None


@pytest.mark.parametrize("owner", ["parent", "unrelated"])
def test_interleaved_unrelated_predecessor_writes_no_marker(tmp_path: Path, owner: str) -> None:
    """The row at store_id - 1 must belong to R's session or its verified chain, not merely sit there."""
    lineage(tmp_path)
    q = message("earlier reply", 9., "assistant")
    p = message("interrupted turn", 10.)
    r = message("next request", 10.)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest([deepcopy(q)])
    with engine_context(tmp_path, window=128000, session_id=owner, conversation_id="conv") as engine:
        engine.ingest([deepcopy(q), deepcopy(p)] if owner == "parent" else [deepcopy(p)])
        (prior,) = [row for row in engine._store.get_session_messages(owner) if row["observed_at"] == 10.]
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.on_session_start("child", platform="cli", context_length=128000, conversation_id="conv",
                                boundary_reason="compression")
        engine.ingest([deepcopy(q), deepcopy(r)])
        (stored,) = [row for row in engine._store.get_session_messages("child") if row["observed_at"] == 10.]
        assert stored["store_id"] - 1 == prior["store_id"]
        relations = engine._store.get_message_relations([stored["store_id"]], "carried_stamp")
        assert [(rel["related_store_id"], rel["observed_at"]) for rel in relations] == (
            [(prior["store_id"], 10.)] if owner == "parent" else [])


def test_marker_write_failure_keeps_the_post_store_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The marker is provenance only: a failed write leaves the duplicate fallback, never a skipped commit."""
    lineage(tmp_path)
    remembered: list[Any] = []
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        def fail(*_args: Any, **_kwargs: Any) -> None:
            raise sqlite3.OperationalError("database is locked")

        original = engine._identity_anchor_remember
        monkeypatch.setattr(engine, "_identity_anchor_mark_carried", fail)
        monkeypatch.setattr(engine, "_identity_anchor_remember", lambda rows: (remembered.extend(rows), original(rows)))
        engine.ingest([message("earlier reply", 9., "assistant"), message("first", 10.)])
        assert [row["observed_at"] for row in engine._store.get_session_messages("parent")] == [9., 10.]
    assert len(remembered) == 2


@pytest.mark.parametrize("relation", ["missing", "wrong-kind", "wrong-prior", "wrong-stamp"])
def test_unmarked_or_mismatched_marker_duplicates_safely(tmp_path: Path, relation: str) -> None:
    """An old store and unrelated relation metadata never authorize dropping a later occurrence."""
    lineage(tmp_path)
    p, r = message("previous", 10.), message("current", 10.)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest([p])
        engine.on_session_start("child", platform="cli", context_length=128000, conversation_id="conv")
        engine.ingest([r])
        (stored,) = engine._store.get_session_messages("child")
        if relation != "missing":
            engine._store.add_message_relations([[(stored["store_id"],
                                                  "alt_stamp" if relation == "wrong-kind" else "carried_stamp",
                                                  stored["store_id"] - (2 if relation == "wrong-prior" else 1),
                                                  None, 9. if relation == "wrong-stamp" else 10.)]])
    with engine_context(tmp_path, window=128000, session_id="child", conversation_id="conv") as engine:
        engine.ingest([p, {**r, "timestamp": 20.}])
        assert [row["observed_at"] for row in engine._store.get_session_messages("child")] == [10., 20.]
