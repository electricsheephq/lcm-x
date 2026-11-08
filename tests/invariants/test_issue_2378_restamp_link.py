"""A turn re-stamped while a manual compress is in flight is stored twice and linked.

Host-view sequence (content-free evidence, ACP rotation 185a62de and TUI in-place c85ae0):
1. the host writes the user row T at the compress request (stamp t_req) and LCM ingests it;
2. compress() runs over the list ending in T; the assistant reply lands while it runs and is ingested;
3. the compaction commits (in place, or rotating into a child) and the host's compacted list carries
   T again with the commit-time stamp (Hermes' ``_insert_message_rows`` stamps ``now_ts`` when no
   active donor row matches), optionally with the same ``message_uid``;
4. the next turn is ingested.
LCM holds T twice (t_req and t_commit) with a ``possible_replay_of`` link. The control keeps
T's stamp and stores T once. No fault is injected.
"""

import logging
import sqlite3
from copy import deepcopy
from typing import Any

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.reconcile as reconcile_module
from tests.harness import make_engine
from tests.test_compression_boundary import _stub_summarizer

pytestmark = pytest.mark.issue(2378)

T = "please list the three deploy steps again for the staging box"
REPLY = "steps: build, push, restart"
T_REQ, T_REPLY, T_COMMIT = 2000.146, 2003.167, 2155.192
Row = dict[str, Any]


def history(uids: bool) -> list[Row]:
    rows: list[Row] = []
    for i in range(14):
        rows.append({"role": "user", "content": f"[T{i:02d}] user turn {i}: " + "alpha beta gamma delta " * 40,
                     "timestamp": 1000.0 + 10 * i})
        rows.append({"role": "assistant", "content": f"reply to T{i:02d}: noted item {i}.",
                     "timestamp": 1000.0 + 10 * i + 1})
    if uids:
        for i, row in enumerate(rows):
            row["message_uid"] = f"{i + 1:032x}"
    return rows


def all_rows(engine: Any) -> list[Row]:
    cols = ("store_id", "session_id", "role", "content", "observed_at")
    return [dict(zip(cols, r)) for r in engine._store._conn.execute(
        "SELECT store_id, session_id, role, content, observed_at FROM messages ORDER BY store_id")]


def replay_links(engine: Any) -> set[tuple[int, int]]:
    conn = engine._store._conn
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='message_relations'").fetchone():
        return set()
    return {(a, b) for a, b in conn.execute(
        "SELECT store_id, related_store_id FROM message_relations WHERE kind='possible_replay_of'")}


def lineage(root) -> None:
    """Hermes' state.db for an ACP rotation: the child ``rotated`` of ``S``, ended by compression."""
    home = root / "home"
    home.mkdir(exist_ok=True)
    with sqlite3.connect(home / "state.db") as db:
        db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.executemany("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?)",
                       [("S", None, "compression"), ("rotated", "S", None)])


@pytest.fixture
def build(tmp_path, monkeypatch):
    engines: list[Any] = []
    monkeypatch.setattr(engine_module, "summarize_with_escalation", _stub_summarizer())

    def create(rotating: bool = False, session_id: str = "S", **overrides: Any) -> Any:
        if rotating:
            lineage(tmp_path)
        config = dict(fresh_tail_count=6, leaf_chunk_tokens=400,
                      large_output_externalization_enabled=False,
                      large_output_active_replay_stubbing_enabled=False,
                      temporal_rollups_enabled=False, embeddings_enabled=False,
                      deferred_maintenance_enabled=False)
        engine = make_engine(tmp_path, window=200_000, session_id=session_id, conversation_id="conversation",
                             **{**config, **overrides})
        engines.append(engine)
        return engine

    yield create
    for engine in engines:
        engine.shutdown()


def run(build, caplog, *, mode: str, uids: bool, restamp: bool,
        prior_copies: int = 1, before_ingest=None) -> tuple[Any, list[Row]]:
    engine = build(rotating=mode == "rotating")
    hist = history(uids)
    engine.ingest(deepcopy(hist))
    user_t = {"role": "user", "content": T, "timestamp": T_REQ}
    reply_t = {"role": "assistant", "content": REPLY, "timestamp": T_REPLY}
    if uids:
        user_t["message_uid"], reply_t["message_uid"] = "a" * 32, "b" * 32
    # 1. the host writes T at the compress request; LCM ingests it
    before = hist + ([{**user_t, "timestamp": T_REQ - 2, "message_uid": "c" * 32},
                      {"role": "assistant", "content": "earlier steps", "timestamp": T_REQ - 1}]
                     if prior_copies == 2 else []) + [user_t]
    engine.ingest(deepcopy(before))
    # 2. manual compress over that list; the reply lands while it runs
    output = engine.compress(deepcopy(before), force=True)
    assert engine._last_compression_status == "compacted"
    engine.ingest(deepcopy(before + [reply_t]))
    # 3. the compaction commits (in place: same id; ACP: rotation into a child)
    sid = engine._session_id
    engine.on_session_end(sid, deepcopy(before))
    engine.on_session_start("rotated" if mode == "rotating" else sid, boundary_reason="compression",
                            old_session_id=sid, conversation_id=engine._conversation_id, platform="cli")
    view = deepcopy(output)
    carried = [row for row in view if row.get("role") == "user" and row.get("content") == T]
    assert len(carried) == prior_copies, "compress output must carry the in-flight user rows"
    if restamp:
        for idx, row in enumerate(carried):
            row["timestamp"] = T_COMMIT + idx  # the host re-persists T at commit time
    view.append(dict(reply_t))
    # 4. the next turn
    view += [{"role": "assistant", "content": "steps again: build, push, restart", "timestamp": 2161.0},
             {"role": "user", "content": "thanks, next question", "timestamp": 2170.0}]
    caplog.clear()
    if before_ingest is not None:
        before_ingest(engine, view)
    with caplog.at_level(logging.INFO):
        engine.ingest(deepcopy(view))
    for record in caplog.records:
        print("LOG", record.levelname, record.getMessage()[:200])
    rows = all_rows(engine)
    for row in rows[-12:]:
        print(row["store_id"], row["session_id"], row["role"], repr(str(row["content"])[:40]), row["observed_at"])
    return engine, rows


@pytest.mark.parametrize("uids", [False, True], ids=["no-uid", "same-uid"])
@pytest.mark.parametrize("mode", ["inplace", "rotating"])
def test_inflight_turn_restamped_at_commit_is_not_a_silent_duplicate(build, caplog, mode, uids):
    engine, rows = run(build, caplog, mode=mode, uids=uids, restamp=True)
    copies = [row for row in rows if row["role"] == "user" and row["content"] == T]
    links = replay_links(engine)
    assert len(copies) == 2
    assert [row["observed_at"] for row in copies] == [T_REQ, T_COMMIT]
    assert links == {(copies[1]["store_id"], copies[0]["store_id"])}


@pytest.mark.parametrize("mode", ["inplace", "rotating"])
def test_control_kept_stamp_stores_once(build, caplog, mode):
    _engine, rows = run(build, caplog, mode=mode, uids=False, restamp=False)
    assert sum(row["role"] == "user" and row["content"] == T for row in rows) == 1
    assert replay_links(_engine) == set()


@pytest.mark.parametrize("uids", [False, True], ids=["no-uid", "new-uid"])
def test_rotation_kept_stamp_then_equal_new_turn_is_not_linked(build, caplog, uids):
    later_stamp = 2200.0

    def append_equal_turn(_engine, view):
        later = {"role": "user", "content": T, "timestamp": later_stamp}
        if uids:
            later["message_uid"] = "d" * 32
        view.append(later)

    engine, rows = run(build, caplog, mode="rotating", uids=uids, restamp=False,
                       before_ingest=append_equal_turn)
    copies = [row for row in rows if row["role"] == "user" and row["content"] == T]
    assert [(row["session_id"], row["observed_at"]) for row in copies] == [
        ("S", T_REQ), ("rotated", later_stamp),
    ]
    assert replay_links(engine) == set()
    assert engine._ingest_failure_count == 0


@pytest.mark.parametrize("mode", ["inplace", "rotating"])
def test_different_user_turn_is_never_linked(build, caplog, mode):
    different = "please cancel staging and deploy a different service"

    def replace(_engine, view):
        next(row for row in view if row.get("content") == T)["content"] = different

    engine, rows = run(build, caplog, mode=mode, uids=False, restamp=True, before_ingest=replace)
    assert sum(row["content"] == T for row in rows) == 1
    assert sum(row["content"] == different for row in rows) == 1
    assert replay_links(engine) == set()


@pytest.mark.parametrize("mode", ["inplace", "rotating"])
def test_model_switch_prefix_removed_restamp_is_stored_and_linked(build, caplog, monkeypatch, mode):
    plain = T
    prefixed = "[Note: model was just switched from X to Y]\n\n" + plain
    monkeypatch.setattr(__name__ + ".T", prefixed)

    def strip_prefix(_engine, view):
        next(row for row in view if row.get("content") == prefixed)["content"] = plain

    engine, rows = run(build, caplog, mode=mode, uids=False, restamp=True, before_ingest=strip_prefix)
    copies = [row for row in rows if row["content"] in (prefixed, plain)]
    assert [(row["content"], row["observed_at"]) for row in copies] == [
        (prefixed, T_REQ), (plain, T_COMMIT),
    ]
    stored = engine._store.get_batch([row["store_id"] for row in copies])
    digests = [engine_module._commit_proof_identity_digest(engine_module._proof_user_identity(
        engine._message_replay_identity(stored[row["store_id"]], stored_row=True))) for row in copies]
    assert digests[0] == digests[1]
    # The host strips its model-switch note on the next turn: the same occurrence, so it is linked.
    assert replay_links(engine) == {(copies[1]["store_id"], copies[0]["store_id"])}


@pytest.mark.parametrize("mode", ["inplace", "rotating"])
def test_todo_annotation_folded_into_restamped_row_is_linked(build, caplog, mode):
    """The host folds its todo snapshot into the trailing user row; the re-stamped copy is still linked."""
    annotation = "\n\n" + reconcile_module._PRESERVED_TODO_CONTEXT_PREFIX + "\n- [>] 1. deploy staging (in_progress)"

    def fold(_engine, view):
        next(row for row in view if row.get("role") == "user" and row.get("content") == T)["content"] = T + annotation

    engine, rows = run(build, caplog, mode=mode, uids=False, restamp=True, before_ingest=fold)
    copies = [row for row in rows if row["role"] == "user" and str(row["content"]).startswith(T)]
    assert [(row["content"], row["observed_at"]) for row in copies] == [(T, T_REQ), (T + annotation, T_COMMIT)]
    assert replay_links(engine) == {(copies[1]["store_id"], copies[0]["store_id"])}


@pytest.mark.parametrize("mode", ["inplace", "rotating"])
@pytest.mark.parametrize("prior_copies", [1, 2])
def test_restamp_links_are_one_to_one(build, caplog, mode, prior_copies):
    def duplicate(_engine, view):
        if prior_copies == 1:
            idx = next(idx for idx, row in enumerate(view) if row.get("content") == T)
            # Retain the published prefix so its commit proof remains usable.
            view.append({**view[idx], "timestamp": T_COMMIT + 1})

    engine, rows = run(build, caplog, mode=mode, uids=False, restamp=True,
                       prior_copies=prior_copies, before_ingest=duplicate)
    copies = [row for row in rows if row["content"] == T]
    assert len(copies) == prior_copies + 2
    links = replay_links(engine)
    assert links == {(copies[prior_copies + idx]["store_id"], copies[idx]["store_id"])
                     for idx in range(prior_copies)}
    assert len({target for _source, target in links}) == len(links)


@pytest.mark.parametrize("mode", ["inplace", "rotating"])
def test_relation_write_failure_still_stores(build, caplog, monkeypatch, mode):
    attempted = []

    def inject(engine, _view):
        write = engine._store.add_message_relations

        def fail(groups):
            result = write(groups)
            if any(relation[1] == "possible_replay_of" for group in groups for relation in group):
                assert engine._store._conn.in_transaction
                assert engine._store._transaction_owner.owns_connection(engine._store._conn)
                attempted.append(True)
                raise RuntimeError("injected relation write failure after SQL")
            return result

        monkeypatch.setattr(engine._store, "add_message_relations", fail)

    engine, rows = run(build, caplog, mode=mode, uids=False, restamp=True, before_ingest=inject)
    assert attempted == [True]
    assert engine._ingest_failure_count == 0
    assert sum(row["content"] == T for row in rows) == 2
    assert replay_links(engine) == set()
    assert "re-stamped occurrence link write failed" in caplog.text


def test_rotation_without_snapshot_reuses_durable_recovery(build, caplog, monkeypatch):
    called = []

    def remove_snapshot(engine, _view):
        key = engine._replay_snapshot_metadata_key("host_occurrence_snapshot", "S")
        engine._store._conn.execute("DELETE FROM metadata WHERE key=?", (key,))
        engine._store._conn.commit()
        link = engine._identity_anchor_link_durable_recovery

        def spy(attempt, lineage):
            called.append(True)
            return link(attempt, lineage)

        monkeypatch.setattr(engine, "_identity_anchor_link_durable_recovery", spy)

    engine, rows = run(build, caplog, mode="rotating", uids=False, restamp=True,
                       before_ingest=remove_snapshot)
    assert called
    assert sum(row["content"] == T for row in rows) == 2
    assert engine._ingest_failure_count == 0


def test_cold_inplace_restamp_is_stored_and_linked(build, caplog):
    def rebind(engine, _view):
        engine.on_session_start("S", conversation_id=engine._conversation_id, platform="cli")

    engine, rows = run(build, caplog, mode="inplace", uids=False, restamp=True, before_ingest=rebind)
    copies = [row for row in rows if row["content"] == T]
    assert len(copies) == 2
    assert replay_links(engine) == {(copies[1]["store_id"], copies[0]["store_id"])}


@pytest.mark.parametrize("mode", ["inplace", "rotating"])
@pytest.mark.parametrize("target_count", [0, 2])
def test_missing_or_ambiguous_prior_target_stores_without_link(build, caplog, monkeypatch, mode, target_count):
    def change_targets(engine, _view):
        lookup = engine._store.find_rows_by_observed_at
        if target_count == 2:
            engine._store.append("S", {"role": "user", "content": T, "timestamp": T_REQ},
                                 conversation_id=engine._conversation_id)

        def find(conversation, sessions, stamps):
            rows = lookup(conversation, sessions, stamps)
            return [row for row in rows if target_count != 0 or row["observed_at"] != T_REQ]

        monkeypatch.setattr(engine._store, "find_rows_by_observed_at", find)

    engine, rows = run(build, caplog, mode=mode, uids=False, restamp=True, before_ingest=change_targets)
    assert sum(row["content"] == T for row in rows) == (3 if target_count == 2 else 2)
    assert replay_links(engine) == set()
    assert engine._ingest_failure_count == 0


@pytest.mark.parametrize("uids", [False, True], ids=["no-uid", "same-uid"])
def test_rotation_unmatched_row_before_the_restamp_keeps_the_link(build, caplog, uids):
    """A child row the snapshot never held must not end the ordered alignment early."""
    def insert_note(_engine, view):
        index = next(i for i, row in enumerate(view) if row.get("role") == "user" and row.get("content") == T)
        view.insert(index, {"role": "user", "content": "a host note the predecessor snapshot never held",
                            "timestamp": T_COMMIT - 1})

    engine, rows = run(build, caplog, mode="rotating", uids=uids, restamp=True, before_ingest=insert_note)
    copies = [row for row in rows if row["role"] == "user" and row["content"] == T]
    assert [(row["session_id"], row["observed_at"]) for row in copies] == [("S", T_REQ), ("rotated", T_COMMIT)]
    assert replay_links(engine) == {(copies[1]["store_id"], copies[0]["store_id"])}


def test_rotation_reordered_carried_rows_are_each_linked(build, caplog):
    """The host may reorder carried rows; each re-stamped copy still pairs with its own occurrence."""
    moved = {}

    def reorder(_engine, view):
        users = [i for i, row in enumerate(view) if row.get("role") == "user"]
        t_index = next(i for i in users if view[i].get("content") == T)
        other = max(i for i in users if i < t_index)
        view[other]["timestamp"] += 0.5  # re-stamped too
        moved["content"] = view[other]["content"]
        view[other], view[t_index] = view[t_index], view[other]

    engine, rows = run(build, caplog, mode="rotating", uids=False, restamp=True, before_ingest=reorder)
    links = replay_links(engine)
    for content in (T, moved["content"]):
        copies = [row for row in rows if row["role"] == "user" and row["content"] == content]
        assert [row["session_id"] for row in copies] == ["S", "rotated"], content[:30]
        assert (copies[1]["store_id"], copies[0]["store_id"]) in links, content[:30]
    assert len(links) == len({target for _source, target in links})


PREFIXED = "[Note: model was just switched from X to Y]\n\n" + T
TODO_FOLD = "\n\n" + reconcile_module._PRESERVED_TODO_CONTEXT_PREFIX + "\n- [>] 1. deploy staging (in_progress)"


def _with_prefixed_turn(monkeypatch) -> float:
    """The predecessor's last carried user turn is T behind a model-switch note: one replay identity with T."""
    plain_history = history

    def history_with_prefixed_turn(uids):
        rows = plain_history(uids)
        last_user = max(i for i, row in enumerate(rows) if row["role"] == "user")
        rows[last_user]["content"] = PREFIXED
        return rows

    monkeypatch.setattr(__name__ + ".history", history_with_prefixed_turn)
    rows = plain_history(False)
    return rows[max(i for i, row in enumerate(rows) if row["role"] == "user")]["timestamp"]


def _carried(view, content):
    return next(i for i, row in enumerate(view) if row.get("role") == "user" and row.get("content") == content)


def _original(rows, content):
    return next(row["store_id"] for row in rows if row["session_id"] == "S" and row["content"] == content)


@pytest.mark.issue(2390)
def test_rotation_reordered_identity_equal_rows_with_different_content_link_to_their_own(build, caplog, monkeypatch):
    """#2390: P (a model-switch note before T's text) and T share one replay identity. Both are carried,
    re-stamped and reordered; each copy must link to its own original, never crossed."""
    _with_prefixed_turn(monkeypatch)

    def reorder(_engine, view):
        p_index, t_index = _carried(view, PREFIXED), _carried(view, T)
        view[p_index]["timestamp"] += 0.5  # re-stamped too
        view[p_index], view[t_index] = view[t_index], view[p_index]

    engine, rows = run(build, caplog, mode="rotating", uids=False, restamp=True, before_ingest=reorder)
    links = replay_links(engine)
    for content in (PREFIXED, T):
        copies = [row for row in rows if row["role"] == "user" and row["content"] == content]
        assert [row["session_id"] for row in copies] == ["S", "rotated"], content[:30]
        assert (copies[1]["store_id"], copies[0]["store_id"]) in links, content[:30]
    assert len(links) == len({target for _source, target in links})


@pytest.mark.issue(2390)
def test_rotation_rewritten_identity_equal_rows_in_order_keep_their_own_originals(build, caplog, monkeypatch):
    """Review probe: the host strips P's note (P' == T) and folds the todo into T (T' == T + todo), in order.
    P' equals T's stored text only by coincidence of the rewrite; content must not cross the pair."""
    _with_prefixed_turn(monkeypatch)

    def rewrite(_engine, view):
        p_index, t_index = _carried(view, PREFIXED), _carried(view, T)
        view[p_index]["timestamp"] += 0.5
        view[p_index]["content"] = T
        view[t_index]["content"] = T + TODO_FOLD

    engine, rows = run(build, caplog, mode="rotating", uids=False, restamp=True, before_ingest=rewrite)
    links = replay_links(engine)
    stripped = next(row["store_id"] for row in rows if row["session_id"] == "rotated" and row["content"] == T)
    folded = next(row["store_id"] for row in rows if row["session_id"] == "rotated" and row["content"] == T + TODO_FOLD)
    assert (stripped, _original(rows, PREFIXED)) in links
    assert (folded, _original(rows, T)) in links


@pytest.mark.issue(2390)
def test_rotation_content_of_a_non_snapshot_row_at_a_stamp_never_redirects_a_link(build, caplog, monkeypatch):
    """Review probe: a stored T that the snapshot never held sits at P's stamp. It must not let T' claim P's
    occurrence: T' stays linked to its own original."""
    p_stamp = _with_prefixed_turn(monkeypatch)

    def add_stray_and_fold(engine, view):
        engine._store.append("S", {"role": "user", "content": T, "timestamp": p_stamp},
                             conversation_id=engine._conversation_id)
        p_index = _carried(view, PREFIXED)
        view[p_index]["timestamp"] += 0.5
        view[p_index]["content"] = PREFIXED + TODO_FOLD

    engine, rows = run(build, caplog, mode="rotating", uids=False, restamp=True, before_ingest=add_stray_and_fold)
    links = replay_links(engine)
    t_copy = next(row["store_id"] for row in rows if row["session_id"] == "rotated" and row["content"] == T)
    t_original = next(row["store_id"] for row in rows if row["session_id"] == "S" and row["content"] == T
                      and row["observed_at"] == T_REQ)
    assert (t_copy, t_original) in links


@pytest.mark.issue(2390)
def test_rotation_identical_copies_keep_their_order_inside_a_mixed_content_group(build, caplog, monkeypatch):
    """Review probe: carried [T, T, P] (P = T behind a model-switch note, one replay identity), re-stamped and
    carried as [T, P, T]. The two identical T copies keep their order; P links to P."""
    _with_prefixed_turn(monkeypatch)

    def mix(_engine, view):
        prefixed = view.pop(_carried(view, PREFIXED))
        prefixed["timestamp"] += 0.5
        view.insert(_carried(view, T) + 1, prefixed)  # carried order: T, P, T

    engine, rows = run(build, caplog, mode="rotating", uids=False, restamp=True, prior_copies=2, before_ingest=mix)
    links = replay_links(engine)
    originals = [row["store_id"] for row in rows if row["session_id"] == "S" and row["content"] == T]
    copies = [row["store_id"] for row in rows if row["session_id"] == "rotated" and row["content"] == T]
    assert len(originals) == len(copies) == 2
    assert {(copies[0], originals[0]), (copies[1], originals[1])} <= links
    p_copy = next(row["store_id"] for row in rows if row["session_id"] == "rotated" and row["content"] == PREFIXED)
    assert (p_copy, _original(rows, PREFIXED)) in links


@pytest.mark.issue(2390)
def test_rotation_list_content_row_keeps_its_links(build, caplog, monkeypatch):
    """Review probe: multimodal (list) content never decides the pairing and never drops the bind's links."""
    parts = [{"type": "text", "text": PREFIXED}]
    plain_history = history

    def history_with_list_turn(uids):
        rows = plain_history(uids)
        rows[max(i for i, row in enumerate(rows) if row["role"] == "user")]["content"] = parts
        return rows

    monkeypatch.setattr(__name__ + ".history", history_with_list_turn)

    def reorder(_engine, view):
        users = [i for i, row in enumerate(view) if row.get("role") == "user" and row.get("content") in (T, parts)]
        p_index = next(i for i in users if view[i]["content"] == parts)
        t_index = next(i for i in users if view[i]["content"] == T)
        view[p_index]["timestamp"] += 0.5
        view[p_index], view[t_index] = view[t_index], view[p_index]

    engine, rows = run(build, caplog, mode="rotating", uids=False, restamp=True, before_ingest=reorder)
    assert len(replay_links(engine)) == 2
    assert not [record for record in caplog.records if "TypeError" in record.getMessage()]


@pytest.mark.issue(2390)
def test_rotation_retained_uids_pair_reordered_equal_copies_with_their_own(build, caplog):
    """#2387 review thread: two equal T copies keep distinct message_uids, are re-stamped and reversed.
    The host's uid names each occurrence, so each copy links to its own original."""
    def reverse(_engine, view):
        carried = [i for i, row in enumerate(view) if row.get("role") == "user" and row.get("content") == T]
        view[carried[0]], view[carried[1]] = view[carried[1]], view[carried[0]]

    engine, rows = run(build, caplog, mode="rotating", uids=True, restamp=True, prior_copies=2, before_ingest=reverse)
    links = replay_links(engine)
    by_stamp = {(row["session_id"], row["observed_at"]): row["store_id"] for row in rows if row["content"] == T}
    # restamp gives carried copy k the stamp T_COMMIT + k in its original order (k=0: uid c, k=1: uid a)
    assert (by_stamp[("rotated", T_COMMIT)], by_stamp[("S", T_REQ - 2)]) in links
    assert (by_stamp[("rotated", T_COMMIT + 1)], by_stamp[("S", T_REQ)]) in links


@pytest.mark.issue(2390)
def test_rotation_acp_trimmed_restamp_is_linked(build, caplog, monkeypatch):
    """#2387 review thread: ACP persists prompt.strip(); a trimmed re-stamped carried row is still linked."""
    padded = "  " + T + "  \n"
    monkeypatch.setattr(__name__ + ".T", padded)

    def trim(_engine, view):
        next(row for row in view if row.get("role") == "user" and row.get("content") == padded)["content"] = padded.strip()

    engine, rows = run(build, caplog, mode="rotating", uids=True, restamp=True, before_ingest=trim)
    original = next(row["store_id"] for row in rows if row["session_id"] == "S" and row["content"] == padded)
    copy = next(row["store_id"] for row in rows if row["session_id"] == "rotated" and row["content"] == padded.strip())
    assert (copy, original) in replay_links(engine)


def _run_with_a_failing_restamp_helper(build, caplog, monkeypatch):
    """The re-stamp helper raises during the carried ingest; returns (engine, rows, whether the recovery ran)."""
    called = []

    def break_helper(engine, _view):
        def boom(_messages):
            raise RuntimeError("synthetic snapshot read failure")

        link = engine._identity_anchor_link_durable_recovery

        def spy(attempt, lineage):
            called.append(True)
            return link(attempt, lineage)

        monkeypatch.setattr(engine, "_rotation_restamp_metadata", boom)
        monkeypatch.setattr(engine, "_identity_anchor_link_durable_recovery", spy)

    engine, rows = run(build, caplog, mode="rotating", uids=False, restamp=True, before_ingest=break_helper)
    return engine, rows, bool(called)


@pytest.mark.issue(2490)
def test_rotation_metadata_failure_degrades_to_durable_recovery(build, caplog, monkeypatch):
    """Grok review thread on #2387: if the re-stamp helper raises, the #1523 durable recovery still runs.

    The recovery links only rows of the current session, so a carried row whose original is in the predecessor
    session stays unlinked, as on main before #2387 (#2392 review thread). That unlinked copy is #2490, a named
    surplus limit (v0.30.0), pinned by the strict xfail below."""
    engine, rows, recovered = _run_with_a_failing_restamp_helper(build, caplog, monkeypatch)
    assert recovered
    assert sum(row["content"] == T for row in rows) == 2
    assert engine._ingest_failure_count == 0


@pytest.mark.issue(2490)
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#2490 named surplus limit (v0.30.0): when the re-stamp helper raises, the carried "
                          "turn's second copy in the child session is stored with no link to its original")
def test_rotation_metadata_failure_still_links_the_carried_copy(build, caplog, monkeypatch):
    try:
        engine, rows, recovered = _run_with_a_failing_restamp_helper(build, caplog, monkeypatch)
    except AssertionError as error:
        pytest.fail(f"setup: {error}")
    copies = {row["session_id"]: row["store_id"] for row in rows if row["content"] == T}
    if not recovered or set(copies) != {"S", "rotated"} or engine._ingest_failure_count:
        pytest.fail(f"setup: recovery ran={recovered}, T copies by session={sorted(copies)}")
    assert (copies["rotated"], copies["S"]) in replay_links(engine), "#2490 known shape: the carried copy is unlinked"


@pytest.mark.issue(2390)
def test_rotation_a_shared_uid_never_decides_the_pairing(build, caplog, monkeypatch):
    """Review probe: the stored P and T share one uid (bad host input) and keep it when re-stamped and
    reordered. The uid names no single occurrence, so content decides: each copy links to its own original."""
    plain_history = history

    def history_with_shared_uid(uids):
        rows = plain_history(uids)
        last_user = max(i for i, row in enumerate(rows) if row["role"] == "user")
        rows[last_user]["content"], rows[last_user]["message_uid"] = PREFIXED, "a" * 32  # T's uid too
        return rows

    monkeypatch.setattr(__name__ + ".history", history_with_shared_uid)

    def reorder(_engine, view):
        p_index, t_index = _carried(view, PREFIXED), _carried(view, T)
        view[p_index]["timestamp"] += 0.5
        view[p_index], view[t_index] = view[t_index], view[p_index]

    engine, rows = run(build, caplog, mode="rotating", uids=True, restamp=True, before_ingest=reorder)
    links = replay_links(engine)
    for content in (PREFIXED, T):
        copy = next(r["store_id"] for r in rows if r["session_id"] == "rotated" and r["content"] == content)
        assert (copy, _original(rows, content)) in links, content[:30]
