"""U1: an older salvaged tool remains a replay after an earlier user restamp."""

from copy import deepcopy

import pytest

from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER as CLEARED
from tests.harness import engine_context


def u1_rows():
    """The sealed U1 main cell: 24 filler rows followed by four tool turns."""
    def msg(role, content, stamp):
        return dict(role=role, content=content, timestamp=float(stamp))

    rows = []
    for i in range(12):
        rows += [msg("user", f"Synthetic filler {i}: " + "alpha beta gamma delta " * 40, i * 2 + 1),
                 msg("assistant", f"Synthetic filler reply {i}", i * 2 + 2)]
    for i in range(4):
        cid, start = f"u1-synthetic-call-{i}", 100 + i * 4
        rows += [
            {**msg("user", f"Synthetic request {i}", start), "message_uid": f"{i + 1:032x}"},
            {**msg("assistant", f"Synthetic call {i}", start + 1), "tool_calls": [
                dict(id=cid, type="function", function=dict(name="lookup", arguments="{}"))]},
            {**msg("tool", f"Synthetic unique output {i} " * 30, start + 2),
             "tool_call_id": cid, "tool_name": "lookup"},
            msg("assistant", f"Synthetic final {i}", start + 3),
        ]
    return rows


@pytest.mark.parametrize("anchor", ["0", "1"], ids=["anchor-off", "anchor-on"])
def test_older_salvage_and_user_restamp_match_ga(tmp_path, monkeypatch, anchor):
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", anchor)
    with engine_context(
        tmp_path, window=200_000, session_id="S", conversation_id="C",
        fresh_tail_count=20, leaf_chunk_tokens=400,
        large_output_externalization_enabled=False,
        large_output_active_replay_stubbing_enabled=False,
        temporal_rollups_enabled=False, embeddings_enabled=False,
        deferred_maintenance_enabled=False,
    ) as engine:
        rows = u1_rows()
        engine.ingest(deepcopy(rows))
        before = engine._store.get_session_messages("S")
        assert len(before) == 40
        assert engine._compress_commit_proof is None
        host = deepcopy(rows)
        host[24].update(timestamp=999.0, message_uid=f"{999:032x}")
        host[26]["content"] = CLEARED
        assert [i for i, row in enumerate(host) if row["role"] == "tool"][-2:] == [34, 38]
        assert engine._message_replay_identity(host[26]) == engine._message_replay_identity(rows[26])
        # GA stores only the restamped user, preserving all 40 original rows.
        fields = ("role", "content", "tool_call_id", "observed_at", "store_id")
        expected = [tuple(row.get(key) for key in fields) for row in before]
        expected.append(("user", rows[24]["content"], None, 999.0, 41))
        for stage in ("after", "repeat", "end"):
            if stage == "end":
                engine.on_session_end("S", host)
            else:
                engine.ingest(host)
            stored = engine._store.get_session_messages("S")
            assert stored[:40] == before, stage
            assert [tuple(row.get(key) for key in fields) for row in stored] == expected, stage


@pytest.mark.parametrize("anchor", ["0", "1"], ids=["anchor-off", "anchor-on"])
@pytest.mark.parametrize("tool_index", [34, 38], ids=["second-newest", "newest"])
def test_newest_placeholder_is_changed_result_after_user_restamp(
    tmp_path, monkeypatch, anchor, tool_index,
):
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", anchor)
    with engine_context(
        tmp_path, window=200_000, session_id="S", conversation_id="C",
        fresh_tail_count=20, leaf_chunk_tokens=400,
        large_output_externalization_enabled=False,
        large_output_active_replay_stubbing_enabled=False,
        temporal_rollups_enabled=False, embeddings_enabled=False,
        deferred_maintenance_enabled=False,
    ) as engine:
        rows = u1_rows()
        engine.ingest(deepcopy(rows))
        before = engine._store.get_session_messages("S")
        assert len(before) == 40
        assert engine._compress_commit_proof is None
        host = deepcopy(rows)
        host[24].update(timestamp=999.0, message_uid=f"{999:032x}")
        host[tool_index]["content"] = CLEARED
        assert [i for i, row in enumerate(host) if row["role"] == "tool"][-2:] == [34, 38]
        assert (engine._message_replay_identity(host[tool_index])
                == engine._message_replay_identity(rows[tool_index]))
        filter_rows = engine._ingest_filter_rows
        rewinds = []

        def check_changed_result(**kwargs):
            if kwargs["new_prefix_occurrences"]:
                rewinds.append(kwargs["cursor"])
                assert kwargs["new_prefix_occurrences"] == {24}
                assert tool_index not in kwargs["unchanged_prefix_occurrences"]
            return filter_rows(**kwargs)

        monkeypatch.setattr(engine, "_ingest_filter_rows", check_changed_result)
        fields = ("role", "content", "tool_call_id", "observed_at", "store_id")
        expected = [tuple(row.get(key) for key in fields) for row in before]
        expected.append(("user", rows[24]["content"], None, 999.0, 41))
        # The changed-result path appends with anchors off; with anchors on,
        # the independent occurrence matcher recognises the original result.
        if anchor == "0":
            expected.append(("tool", CLEARED, rows[tool_index]["tool_call_id"],
                             rows[tool_index]["timestamp"], 42))
        for stage in ("after", "repeat", "end"):
            if stage == "end":
                engine.on_session_end("S", host)
            else:
                engine.ingest(host)
            stored = engine._store.get_session_messages("S")
            assert stored[:40] == before, stage
            assert [tuple(row.get(key) for key in fields) for row in stored] == expected, stage
        assert rewinds == [24]
