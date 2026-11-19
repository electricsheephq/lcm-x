"""#2996 Part A: cold UID reconciliation of a host-folded summary carrier."""

from unittest.mock import Mock

import pytest

from hermes_lcm import host_uid_emit
from hermes_lcm.tokens import count_messages_tokens
from tests.invariants.test_issue_2815_c4 import _closing
from tests.invariants.test_issue_2815_c18 import _seed
from tests.invariants.test_issue_2815_uid_restart import _open, _view
from tests.invariants.test_retry_r4 import _merge_absorbed
from tests.test_issue_581_582_store_complete_survival import _rows, summaries as summaries


@pytest.fixture(autouse=True)
def uid_on(monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)


@pytest.fixture
def carrier(tmp_path, summaries, request):
    path = tmp_path / "store"
    case = getattr(request, "param", None)
    with _closing(_seed(path)) as warm:
        raw = _view(6)
        turn = _view(7)[-4:]
        if case == "backfill":
            for row in (raw[-4], turn[-1]):
                row.pop("timestamp")
        warm._config.fresh_tail_count = 4
        warm.ingest(raw)
        published = warm.compress(raw, force=True, current_tokens=count_messages_tokens(raw))
        assert warm._last_compress_leaves[1] == 1
        summary, user = published[:2]
        assert warm._is_lcm_emitted_head_row(summary, warm._last_compacted_store_id)
        assert user["role"] == "user"
        # The host view omits a stored reply from the new turn, so the old
        # compaction proof cannot verify the whole post-compaction continuation.
        hidden = {"role": "assistant", "content": "Stored reply outside the host view."}
        warm.ingest([*published, *turn, hidden])
        host = _merge_absorbed([*published, *turn, *([hidden] if case == "matching" else [])])
        assert host[0]["_absorbed_message_uids"] == [user["message_uid"]]
        rows = _rows(warm)
        [source] = [row["store_id"] for row in rows if row["content"] == user["content"]]
        assert source > warm._last_compacted_store_id
    with _closing(_open(path, embeddings_enabled=False)) as cold:
        assert cold._cursor_from_durable_commit_proof(host) == (len(host) if case == "matching" else None)
        yield cold, host, source


def test_cold_bind_carrier_absorbed_above_frontier(carrier):
    cold, host, _ = carrier
    before = len(_rows(cold))
    cold.ingest(host)
    rows = _rows(cold)
    assert (cold._last_ingest_reconciliation["reason"], len(rows) - before) == (
        "replayed UID-bound prefix", 0,
    )
    assert cold._ingest_cursor == len(host)
    assert all(row["content"] != host[0]["content"] for row in rows)


@pytest.mark.parametrize("carrier", ["matching"], indirect=True)
def test_cold_bind_carrier_proof_rescues_unchanged(carrier):
    cold, host, _ = carrier
    before = _rows(cold)
    cold.ingest(host)
    print(f"R1b reason: {cold._last_ingest_reconciliation['reason']}")
    assert cold._ingest_cursor == len(host)
    assert _rows(cold) == before
    assert all(row["content"] != host[0]["content"] for row in before)


def test_next_pass_after_carrier_cold_bind(carrier, caplog):
    cold, host, source = carrier
    cold.ingest(host)
    previous = {node.node_id for node in cold._dag.get_uncondensed_at_depth("S", 0)}
    cold.compress(host, force=True, current_tokens=count_messages_tokens(host))
    assert cold._last_compress_leaves[1] == 1
    assert cold._store_complete_hole is None
    assert "LCM contiguity hole" not in caplog.text
    [leaf] = [node for node in cold._dag.get_uncondensed_at_depth("S", 0) if node.node_id not in previous]
    assert source in leaf.source_ids


@pytest.mark.parametrize("shape", ["two", "edited", "foreign", "canonical", "self", "separator", "shadow", "exception"])
def test_carrier_declines(carrier, monkeypatch, shape):
    cold, host, source = carrier
    head = host[0]
    if shape == "two":
        head["_absorbed_message_uids"].append("second-absorbed")
    elif shape == "edited":
        head["content"] = head["content"][:-1] + "!"
    elif shape == "foreign":
        lineage, _ = cold._host_uid_lineage_key()
        foreign = cold._store.append("B", cold._store.get_batch([source])[source], conversation_id="conv")
        cold._store.add_host_uid_bindings(lineage, [(foreign, "foreign", "canonical", "stored_new")])
        head["_absorbed_message_uids"] = ["foreign"]
    elif shape == "canonical":
        head["message_uid"] = host[1]["message_uid"]
    elif shape == "self":
        head["_absorbed_message_uids"] = [head["message_uid"]]
    elif shape == "separator":
        end = cold._verified_lcm_summary_prefix_end(head["content"])
        head["content"] = head["content"][:end + 2] + "---\n\n" + head["content"][end + 2:]
    elif shape == "shadow":
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "shadow")
    elif shape == "exception":
        verify, resolve = cold._verified_lcm_summary_prefix_end, cold._host_uid_resolve
        def prefix_resolve(*args, **kwargs):
            monkeypatch.setattr(cold, "_verified_lcm_summary_prefix_end", verify)
            return resolve(*args, **kwargs)

        monkeypatch.setattr(cold, "_verified_lcm_summary_prefix_end", Mock(side_effect=RuntimeError("synthetic fault")))
        monkeypatch.setattr(cold, "_host_uid_resolve", prefix_resolve)
    before = _rows(cold)
    cold.ingest(host)
    rows = _rows(cold)
    assert cold._last_ingest_reconciliation["reason"] == "persisted ambiguous delta"
    assert cold._last_ingest_reconciliation["cursor"] == 0
    assert cold._ingest_cursor == len(host)
    assert rows[:-1] == before
    assert [row["content"] for row in rows[len(before):]] == [head["content"]]


def test_probe_returns_original_without_one_absorbed_uid(carrier):
    cold, host, _ = carrier
    for absorbed in (None, [], "uid", {"uid": True}, ["first", "second"]):
        view = [dict(host[0], _absorbed_message_uids=absorbed), *host[1:]]
        assert cold._host_uid_carrier_probe(view) is view
    empty = []
    assert cold._host_uid_carrier_probe(empty) is empty


def test_carrier_partial_uid_prefix(carrier, monkeypatch):
    cold, host, _ = carrier
    j = len(host)
    pair = [{"role": "user", "content": f"Fresh question {i}", "message_uid": f"new-{i}"} for i in range(2)]
    host += _merge_absorbed(pair)
    search = Mock(wraps=cold._find_reconciled_cursor_for_store_tail)
    monkeypatch.setattr(cold, "_find_reconciled_cursor_for_store_tail", search)
    before = _rows(cold)
    cold.ingest(host)
    assert search.call_args.kwargs["uid_cursor"] == j
    assert search.call_args.kwargs["cut_budget"] > 0
    rows = _rows(cold)
    assert rows[:-1] == before
    assert [row["content"] for row in rows[len(before):]] == [host[j]["content"]]


@pytest.mark.parametrize("carrier", ["backfill"], indirect=True)
def test_carrier_backfill_and_ledger(carrier, monkeypatch):
    cold, host, source = carrier
    follow = host[-1]
    [bound] = [row["store_id"] for row in _rows(cold) if row["content"] == follow["content"]]
    assert all(row["observed_at"] is None for row in cold._store.get_batch([source, bound]).values())
    host[0]["timestamp"], follow["timestamp"] = 2000.0, 2001.0
    ledger = Mock(wraps=cold._write_ingest_ledger)
    monkeypatch.setattr(cold, "_write_ingest_ledger", ledger)
    cold.ingest(host)
    assert ledger.call_args.args[0]["cursor0_batch"] is False
    rows = cold._store.get_batch([source, bound])
    assert rows[source]["observed_at"] is None
    assert rows[bound]["observed_at"] == follow["timestamp"]
