"""A surplus summary head remains durable without entering model context (I1, I2, I11)."""

import json

import pytest

from hermes_lcm import tools
from tests.invariants.test_issue_3006_post_refusal import _host_flush
from tests.test_issue_671_stub_first_exit import (
    THRESHOLD,
    exit_engine,
    make_engine,  # noqa: F401
    run,
    stop_line,
    summaries,  # noqa: F401
    text_view,
    tool_view,
)

RELATION = "generated_unproven"
NEW_TURN = [
    {"role": "user", "content": "Please compare the two accounts."},
    {"role": "assistant", "content": "Both accounts reconcile."},
]


@pytest.fixture(params=["default", "on"])
def uid_mode(request, monkeypatch):
    if request.param == "on":
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    else:
        monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)


def _append(engine, row, relation=None, *, session=None):
    store_id, = engine._store.append_batch(
        session or engine._session_id, [row], conversation_id=engine._conversation_id,
    )
    if relation:
        engine._store.add_message_relations([[(store_id, relation, None, None, None)]])
    return store_id


@pytest.fixture
def twins(uid_mode, make_engine, summaries, caplog):  # noqa: F811
    engines = []
    for with_copy in (False, True):
        engine = exit_engine(make_engine, threshold_full_sweep_enabled=False, l3_truncate_tokens=1)
        engine.on_session_start("hf2", conversation_id="hf2", context_length=200_000)
        engine.threshold_tokens = THRESHOLD
        seed = text_view()[1:]
        seed[0]["content"] = "Earlier turns. " + seed[0]["content"]
        first = run(engine, seed, caplog)
        head_input = {**first[0], "content": first[0]["content"] + "\n\n---\n\n" + first[1]["content"]}
        supplied = tool_view(prefix=[head_input, *first[2:]])[1:]
        returned = run(engine, supplied, caplog)
        assert stop_line(caplog).group(1) == "stub_first_exit"
        assert len(returned) == len(supplied) and returned[0]["content"] != head_input["content"]
        head = returned[0]
        remainder = engine._generated_context_carrier_remainder(head)
        assert remainder and any(
            r["content"] == remainder for r in engine._store.get_session_messages(engine._session_id)
        )
        old_top = engine._store.max_store_id()
        engine.ingest(_host_flush()(returned + NEW_TURN))
        assert not any(
            r["content"] == head["content"]
            for r in engine._store.get_session_messages_after(engine._session_id, old_top)
        )  # This synthetic warm ingest leaves the copy out. Seed the stored shape through the ingest writer.
        copy_id = (_append(engine, head, RELATION) if with_copy else None)
        if not with_copy:
            # Occupy the same global ID outside this conversation's session, so leaf IDs are directly comparable.
            _append(engine, NEW_TURN[0], session="unrelated")
        engine._config.large_output_active_replay_stubbing_enabled = False
        engines.append((engine, head, copy_id))
    return engines


def _view(engine):
    return [
        {**engine._store.to_openai_msg(row), "store_id": row["store_id"], "session_id": row["session_id"]}
        for row in engine._store.get_session_messages_after(engine._session_id, engine._last_compacted_store_id)
    ]


def _snapshot(engine):
    conn = engine._store.connection
    return (
        conn.execute("SELECT store_id, role, content FROM messages ORDER BY store_id").fetchall(),
        conn.execute("SELECT * FROM message_relations ORDER BY relation_id").fetchall(),
    )


def test_assembly_matches_twin(twins):
    (clean, _, _), (copied, _, _) = twins
    assert copied._assemble_context(None, _view(copied), persist=False) == clean._assemble_context(
        None, _view(clean), persist=False
    )


def test_assembly_preserves_rows_and_relations(twins):
    _, (engine, head, copy_id) = twins
    before = _snapshot(engine)
    assert (copy_id, "user", head["content"]) in before[0]
    engine._assemble_context(None, _view(engine), persist=False)
    assert _snapshot(engine) == before


@pytest.mark.parametrize("position", ["leaf", "tail"])
def test_compaction_matches_twin(twins, summaries, caplog, position):  # noqa: F811
    results = []
    for engine, _, copy_id in twins:
        if position == "leaf":
            _append(engine, {"role": "user", "content": "One more unrelated question."})
            _append(engine, {"role": "assistant", "content": "One more reply."})
        before = _snapshot(engine)
        view = _view(engine)
        # These rows came from the store and are already durable. The host's loaded list is the input to compress.
        engine._ingest_cursor = len(view)
        engine._ingest_cursor_needs_reconcile = False
        engine._compress_commit_proof = None
        start = len(summaries)
        nodes_before = {node.node_id for node in engine._dag.get_session_nodes(engine._session_id)}
        result = run(engine, view, caplog, force=True)
        leaves = [
            node.source_ids for node in engine._dag.get_session_nodes(engine._session_id)
            if node.source_type == "messages" and node.node_id not in nodes_before
        ]
        assert engine.last_compression_status == "compacted"
        assert "lacks raw store lineage" not in engine.last_compression_noop_reason
        assert _snapshot(engine)[0] == before[0]
        assert copy_id is None or all(copy_id not in ids for ids in leaves)
        results.append((leaves, summaries[start:], result))
    assert results[0] == results[1]
    assert results[0][0] and results[0][1]


@pytest.mark.xfail(strict=True, reason="v0.30.3 ships the prompt and compaction exclusion only; recall exclusion follows in v0.30.4")
def test_search_preserves_original_hits_and_summary(twins):
    _, (engine, _, copy_id) = twins
    hits = json.loads(tools.lcm_grep({"query": "Earlier turns", "limit": 20}, engine=engine))["results"]
    assert all(hit.get("store_id") != copy_id for hit in hits)
    assert any(hit.get("store_id") for hit in hits)
    assert any(hit.get("node_id") for hit in hits)
    expanded = json.loads(tools.lcm_expand({"store_id": copy_id}, engine=engine))
    assert "error" in expanded
    loaded = json.loads(tools.lcm_load_session({"session_id": engine._session_id, "limit": 100}, engine=engine))
    assert all(row["store_id"] != copy_id for row in loaded["messages"])


@pytest.mark.parametrize("shape", ["unverified", "unique_remainder", "no_relation"])
def test_uncertain_rows_are_still_assembled_and_searchable(twins, shape):
    (engine, head, _), _ = twins
    if shape == "unverified":
        content = head["content"].replace("Earlier turns", "Quoted turns", 1)
    elif shape == "unique_remainder":
        end = engine._verified_lcm_summary_prefix_end(head["content"])
        content = head["content"][:end] + "\n\nOne uniquely stored instruction."
    else:
        content = head["content"]
    row = {"role": "user", "content": content}
    store_id = _append(engine, row, None if shape == "no_relation" else RELATION)
    assembled = engine._assemble_context(None, _view(engine), persist=False)
    assert any(r["content"] == content for r in assembled)
    hits = json.loads(tools.lcm_grep({"query": "turns", "limit": 20}, engine=engine))["results"]
    assert any(hit.get("store_id") == store_id for hit in hits)
