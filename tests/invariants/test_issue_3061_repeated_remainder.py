"""I1, I2, I11: a stored summary carrier must preserve a new user occurrence."""

from copy import deepcopy
from dataclasses import replace

import pytest

from hermes_lcm import host_uid_emit
from hermes_lcm.head_copy import is_known_head_copy
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_1116_stub_first_exit_commit_proof import Host, committed_session, host_for  # noqa: F401


RELATION = "generated_unproven"
REMAINDERS = ("continue", "yes", "Approved", "go ahead")


def _append_user(host, content, *, stamped=True):
    top = host.engine._store.max_store_id()
    host.clock += 30
    row = {"role": "user", "content": content}
    if stamped:
        row["timestamp"] = host.clock
    host.live.append(row)
    host.engine.ingest(host.live)
    stored, = host.engine._store.get_session_messages_after(host.sid, top)
    assert stored["role"] == "user" and stored["content"] == content
    return stored["store_id"]


def _head_sources(engine, head):
    end, node_ids = engine._verified_lcm_summary_prefix(head)
    assert end == len(head)
    messages = set()
    while node_ids:
        node = engine._dag.get_node(node_ids.pop())
        if node.source_type == "nodes":
            node_ids.extend(node.source_ids)
        else:
            assert node.source_type == "messages"
            messages.update(node.source_ids)
    return messages


def _leaf_sources(host):
    return {
        store_id
        for node in host.engine._dag.get_session_nodes(host.sid)
        if node.source_type == "messages"
        for store_id in node.source_ids
    }


def _view(host):
    engine = host.engine
    return [
        {**engine._store.to_openai_msg(row), "store_id": row["store_id"], "session_id": row["session_id"]}
        for row in engine._store.get_session_messages_after(host.sid, engine._last_compacted_store_id)
    ]


def _assembled_ids(host):
    return {row.get("store_id") for row in host.engine._assemble_context(None, _view(host), persist=False)}


def _compact_store_view(host):
    engine = host.engine
    host.turn(tool_words=1_900)
    view = _view(host)
    engine._ingest_cursor = len(view)
    engine._ingest_cursor_needs_reconcile = False
    engine._compress_commit_proof = None
    return engine.compress(view, current_tokens=engine.threshold_tokens + 5_000, force=True)


@pytest.fixture(params=["unset", "on"])
def session(request, host_for, monkeypatch):  # noqa: F811
    if request.param == "on":
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    else:
        monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    host_config = Host._config
    monkeypatch.setattr(Host, "_config", lambda host: replace(
        host_config(host), embeddings_enabled=False, temporal_rollups_enabled=False
    ))
    host = host_for("carrier")
    home = host.tmp_path / "home"
    home.mkdir(exist_ok=True)
    _state_db(home, [(host.sid, None, None)])
    assert host.engine._uid_recognition_enabled() == (request.param == "on")
    return host


def _commit_head(host, remainder):
    host.turn()
    earlier_id = _append_user(host, remainder)
    host.clock += 0.5
    host.live.append({"role": "assistant", "content": "Earlier reply.", "timestamp": host.clock})
    host.engine.ingest(host.live)
    committed_session(host, "carrier")
    emitted = host.live[0]["content"]
    end = host.engine._verified_lcm_summary_prefix_end(emitted)
    assert end
    head = emitted[:end]
    assert earlier_id in _head_sources(host.engine, head)
    return head, earlier_id


def _assert_carrier(host, carrier_id, head, remainder):
    store = host.engine._store
    row = store.get(carrier_id)
    assert row["content"] == head + "\n\n" + remainder
    assert store.get_message_relations([carrier_id], RELATION)
    return row


@pytest.mark.parametrize("remainder", REMAINDERS)
@pytest.mark.parametrize("merged", [True, False], ids=["merged", "own-row"])
def test_new_message_survives_assembly_and_compaction(session, remainder, merged):
    host = session
    head, earlier_id = _commit_head(host, remainder)
    target = _append_user(host, head + "\n\n" + remainder if merged else remainder)
    assert target > earlier_id
    if merged:
        _assert_carrier(host, target, head, remainder)

    assert target in _assembled_ids(host)
    frontier = host.frontier()
    output = _compact_store_view(host)
    assert host.frontier() > frontier
    assert target in _leaf_sources(host) or (
        target > host.frontier() and target in _assembled_ids(host)
        and any(row.get("store_id") == target for row in output)
    )


def _replay_carrier(host, head, own_row, *, stamp):
    # Present the list again, replacing the separate row with the host's merge.
    # Ingest can already recognize this replay. In that case seed the stored
    # representation that rc1's read-time predicate is responsible for.
    carrier = {"role": "user", "content": head + "\n\n" + own_row["content"]}
    if stamp is not None:
        carrier["timestamp"] = stamp
    top = host.engine._store.max_store_id()
    host.live[-1] = carrier
    engine = host.engine
    engine.ingest(deepcopy(host.live))
    added = [
        row for row in engine._store.get_session_messages_after(host.sid, top)
        if row["role"] == "user" and row["content"] == carrier["content"]
    ]
    if added:
        stored, = added
        target = stored["store_id"]
    else:
        target, = engine._store.append_batch(host.sid, [carrier], conversation_id=engine._conversation_id)
        engine._store.add_message_relations([[(target, RELATION, None, None, None)]])
    return target


@pytest.mark.parametrize("later_covered", [False, True], ids=["fresh", "later-covered"])
def test_replay_twin_stays_a_known_head_copy(session, later_covered):
    host = session
    remainder = "continue"
    head, earlier_id = _commit_head(host, remainder)
    own_id = _append_user(host, remainder)
    engine = host.engine
    own_row = engine._store.get(own_id)
    old_host_view = deepcopy(host.live)
    assert {earlier_id, own_id} & _head_sources(engine, head) == {earlier_id}
    if later_covered:
        for _ in range(4):
            if own_id in _leaf_sources(host):
                break
            _compact_store_view(host)
        assert own_id in _leaf_sources(host)
        assert own_id not in _head_sources(engine, head)
        host.live = old_host_view

    assert own_row["observed_at"] is not None
    carrier_id = _replay_carrier(host, head, own_row, stamp=own_row["observed_at"])
    carrier = _assert_carrier(host, carrier_id, head, remainder)
    assert carrier["observed_at"] == own_row["observed_at"]
    assert is_known_head_copy(engine._store.connection, carrier)
    assert carrier_id not in _assembled_ids(host)


@pytest.mark.parametrize("stamped", [True, False], ids=["distinct-stamps", "unstamped"])
def test_fresh_repeat_with_an_uncovered_twin_stays_visible(session, stamped):
    host = session
    remainder = "continue"
    head, _ = _commit_head(host, remainder)
    own_id = _append_user(host, remainder, stamped=stamped)
    target = _append_user(host, head + "\n\n" + remainder, stamped=stamped)
    assert own_id not in _head_sources(host.engine, head)
    _assert_carrier(host, target, head, remainder)
    assert target in _assembled_ids(host)


def test_unstamped_replay_without_emission_evidence_stays_visible(session):
    host = session
    remainder = "continue"
    head, _ = _commit_head(host, remainder)
    own_id = _append_user(host, remainder, stamped=False)
    own_row = host.engine._store.get(own_id)
    carrier_id = _replay_carrier(host, head, own_row, stamp=None)
    carrier = _assert_carrier(host, carrier_id, head, remainder)
    assert own_row["observed_at"] is carrier["observed_at"] is None
    assert not is_known_head_copy(host.engine._store.connection, carrier)
    assert carrier_id in _assembled_ids(host)
