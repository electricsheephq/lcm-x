"""#1250: a leaf can atomically take ownership of its session's provisional rows (I1, I2, I5, I14)."""

import ast
from collections import Counter
from contextlib import ExitStack, nullcontext
from hashlib import sha256
import json
import logging
from pathlib import Path

import pytest

from hermes_lcm import engine as engine_module, plugin_identity
from hermes_lcm.aux_session import _DELEGATE_SOURCE
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.engine_registry import (
    ActiveEngineUseStatus, use_active_lcm_engine, use_cold_lcm_engine, use_lcm_engine,
)
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError
from tests.compat.conftest import host_root, old_tree  # noqa: F401
from tests.compat.helpers import run_old

SESSION = "S-1250"
GATEWAY = "agent:main:local:dm:S-1250"
PREVIOUS_GA = {"v0.30.4": "0ab1160716f1c5167d17885a92aaa1ed10b0919f"}
SETTINGS = dict(fresh_tail_count=1, leaf_chunk_tokens=1, incremental_max_depth=0,
                temporal_rollups_enabled=False, embeddings_enabled=False)
EARLY = [
    {"role": "user", "content": "Early GUI turn: project cobalt uses SQLite."},
    {"role": "assistant", "content": "Cobalt keeps authoritative history in SQLite."},
    {"role": "user", "content": "Early GUI turn: the deployment checkpoint is approved."},
    {"role": "assistant", "content": "Checkpoint approval recorded for cobalt."},
]
LATER = [
    {"role": "user", "content": "Desktop turn: add a nightly backup job."},
    {"role": "assistant", "content": "Nightly backup job added to the cobalt plan."},
    {"role": "user", "content": "Desktop turn: rotate the logs weekly."},
    {"role": "assistant", "content": "Weekly log rotation added."},
    {"role": "user", "content": "Desktop turn: keep the newest turn in context."},
]

pytestmark = pytest.mark.issue(1250)


@pytest.fixture
def open_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **_: ("Cobalt summary.", 1))
    with ExitStack() as stack:
        def open_one(**overrides):
            engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), **(SETTINGS | overrides)))
            stack.callback(engine.shutdown)
            return engine

        yield open_one


def _hook_for(engine):
    # The real hook body, with the production registry and ingest, from the investigation repro.
    source = Path(engine_module.__file__).with_name("__init__.py")
    tree = ast.parse(source.read_text())
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in {"_engine_bound_session_id", "_on_post_llm_call"}
    ]
    namespace = dict(
        engine=engine, active=True, _identity=plugin_identity, nullcontext=nullcontext,
        logger=logging.getLogger(__name__), ActiveEngineUseStatus=ActiveEngineUseStatus,
        use_active_lcm_engine=use_active_lcm_engine, use_cold_lcm_engine=use_cold_lcm_engine,
        use_lcm_engine=use_lcm_engine,
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_on_post_llm_call"]


def _bind_gateway(engine):
    engine.on_session_start(SESSION, platform="desktop", conversation_id=GATEWAY, context_length=200_000)


def _rows(engine):
    return engine._store.get_session_messages(SESSION)


def _digests(rows):
    return Counter((row["role"], sha256(row["content"].encode()).hexdigest()) for row in rows)


def _frontier(engine):
    return engine._lifecycle.get_by_conversation(GATEWAY).current_frontier_store_id


def _coverage(engine):
    return {store_id for node in engine._dag.get_session_nodes(SESSION) for store_id in node.source_ids}


def _owners(engine):
    return {row["store_id"]: row["conversation_id"] for row in _rows(engine)}


def _seed_mixed(engine):
    engine.on_session_start(SESSION, platform="gui", context_length=200_000)
    engine.ingest(list(EARLY))
    _bind_gateway(engine)
    engine.ingest(EARLY + LATER)
    assert list(_owners(engine).values()) == [SESSION] * len(EARLY) + [GATEWAY] * len(LATER)
    return EARLY + LATER


def _seed_compacted(engine):
    engine.on_session_start(SESSION, platform="gui", context_length=200_000)
    engine.ingest(list(EARLY))
    host = list(engine.compress(list(EARLY)))
    assert engine.last_compression_status == "compacted"
    return host


def _publish(engine, host):
    before = _rows(engine)
    frontier = _frontier(engine)
    nodes = engine._dag.get_session_node_count(SESSION)
    output = list(engine.compress(list(host)))
    assert engine.last_compression_status == "compacted"
    assert _frontier(engine) > frontier
    assert engine._dag.get_session_node_count(SESSION) > nodes
    after = _rows(engine)
    assert _digests(after) == _digests(before)
    assert after == [
        dict(old, conversation_id=new["conversation_id"])
        for old, new in zip(before, after, strict=True)
    ]
    return output


def _add_turn(engine, host, index):
    turn = [
        {"role": "assistant", "content": f"Desktop reply {index}."},
        {"role": "user", "content": f"Desktop follow-up {index}."},
    ]
    expected = _digests(_rows(engine)) + _digests(turn)
    engine.ingest(host + turn)
    assert _digests(_rows(engine)) == expected
    return host + turn


@pytest.mark.parametrize("cold_hook", [False, True], ids=["identityless-bind", "cold-post-turn-hook"])
def test_mixed_session_publishes_first_and_later_passes(open_engine, cold_hook, caplog):
    engine = open_engine()
    if cold_hook:
        hook = _hook_for(engine)
        hook(session_id=SESSION, platform="gui", conversation_history=list(EARLY))
        assert list(_owners(engine).values()) == [SESSION] * len(EARLY)
        _bind_gateway(engine)
        hook(session_id=SESSION, platform="desktop", conversation_id=GATEWAY, conversation_history=EARLY + LATER)
        host = EARLY + LATER
    else:
        host = _seed_mixed(engine)
    before = _rows(engine)
    with caplog.at_level(logging.INFO, logger="hermes_lcm.compaction"):
        for index in range(3):
            if index:
                host = _add_turn(engine, host, index)
            host = _publish(engine, host)
    assert all(owner == GATEWAY for owner in _owners(engine).values())
    first, last = before[0]["store_id"], before[len(EARLY) - 1]["store_id"]
    assert [record.getMessage() for record in caplog.records if "re-owned provisional rows" in record.getMessage()] == [
        f"LCM publication re-owned provisional rows: conversation={GATEWAY} session={SESSION} "
        f"count={len(EARLY)} store_ids={first}-{last}"
    ]


@pytest.mark.parametrize("unmatched", [False, True], ids=["absent", "unmatched"])
def test_finding5_does_not_move_rows_the_leaf_skips(open_engine, unmatched):
    engine = open_engine()
    _seed_mixed(engine)
    before = _rows(engine)
    early_ids = {key for key, owner in _owners(engine).items() if owner == SESSION}
    host = [dict(row, content="[gui] " + row["content"]) for row in EARLY] + LATER if unmatched else LATER
    _publish(engine, host)
    assert _frontier(engine) > max(early_ids)
    engine.compress(EARLY + LATER)
    assert engine.last_compression_status in {"compacted", "noop", "idle"}
    assert not (early_ids & _coverage(engine))
    assert _rows(engine) == before


def test_gateway_only_control(open_engine):
    engine = open_engine()
    _bind_gateway(engine)
    engine.ingest(EARLY + LATER)
    _publish(engine, EARLY + LATER)
    assert set(_owners(engine).values()) == {GATEWAY}


def test_identityless_rebind_keeps_gateway_after_retirement(open_engine):
    engine = open_engine()
    host = _seed_mixed(engine)
    assert engine._lifecycle.get_by_conversation(SESSION).current_session_id == SESSION
    assert engine._lifecycle.get_by_conversation(GATEWAY).current_session_id == SESSION
    engine.on_session_start(SESSION, platform="tui", context_length=200_000)
    assert engine._conversation_id == GATEWAY
    _publish(engine, host)
    fresh = open_engine()
    fresh.on_session_start(SESSION, platform="tui", context_length=200_000)
    assert fresh._conversation_id == GATEWAY
    old = fresh._lifecycle.get_by_conversation(SESSION)
    assert old.current_session_id is None
    assert old.last_finalized_session_id == SESSION


@pytest.mark.parametrize("cold", [False, True], ids=["host-turn-loop", "fresh-engine-per-pass"])
def test_previously_compacted_session_preserves_nodes_and_never_copies_tail(open_engine, cold):
    engine = open_engine()
    host = _seed_compacted(engine)
    history = engine._dag.get_session_nodes(SESSION)
    prior_covered = _coverage(engine)
    prior_frontier = engine._lifecycle.get_by_conversation(SESSION).current_frontier_store_id
    provisional_tail = set(_owners(engine)) - prior_covered
    assert prior_frontier > 0 and provisional_tail

    if cold:
        engine = open_engine()
    _bind_gateway(engine)
    host += LATER
    engine.ingest(list(host))
    assert engine._last_ingest_reconciliation["cursor"] == 2
    assert _digests(_rows(engine)) == _digests(EARLY + LATER)

    for index in range(3):
        if cold:
            engine = open_engine()
            _bind_gateway(engine)
        if index:
            host = _add_turn(engine, host, index)
        host = _publish(engine, host)
    assert engine._dag.get_session_nodes(SESSION)[:len(history)] == history
    owners = _owners(engine)
    assert {key for key, owner in owners.items() if owner == SESSION} == prior_covered
    assert all(owners[store_id] == GATEWAY for store_id in provisional_tail)
    old = engine._lifecycle.get_by_conversation(SESSION)
    assert old.current_session_id is None and old.current_frontier_store_id == 0
    assert old.last_finalized_session_id == SESSION and old.last_finalized_frontier_store_id == prior_frontier


@pytest.mark.parametrize("claim_late", [False, True], ids=["original-host-view", "late-row-in-view"])
def test_still_bound_writer_publishes_late_rows_when_claimed(open_engine, claim_late):
    writer = open_engine()
    writer.on_session_start(SESSION, platform="gui", context_length=200_000)
    writer.ingest(list(EARLY))
    gateway = open_engine()
    _bind_gateway(gateway)
    gateway.ingest(EARLY + LATER)

    late = {"role": "assistant", "content": "GUI reply after the gateway bind."}
    writer.ingest(EARLY + [late])
    late_id = _rows(gateway)[-1]["store_id"]
    assert _owners(gateway)[late_id] == SESSION
    assert _digests(_rows(gateway)) == _digests(EARLY + LATER + [late])
    host = EARLY + LATER
    if claim_late:
        # Resume the gateway with the full transcript, including the late reply now held by the host.
        gateway = open_engine()
        _bind_gateway(gateway)
        host = _add_turn(gateway, host + [late], 1)
    _publish(gateway, host)
    assert (late_id in _coverage(gateway)) is claim_late
    assert _owners(gateway)[late_id] == (GATEWAY if claim_late else SESSION)
    assert writer._conversation_id == SESSION


def _stage_leaf(engine, covered):
    return engine._dag.add_node(
        SummaryNode(session_id=SESSION, summary="Cobalt summary.", source_ids=covered),
        before_commit=lambda conn, node_id: engine._lifecycle.stage_compaction_publication(
            conn, GATEWAY, SESSION, node_id, _frontier(engine), covered,
        ),
    )


def test_other_conversation_in_mixed_snapshot_still_refuses(open_engine):
    engine = open_engine()
    _seed_mixed(engine)
    foreign = engine._store.append(
        SESSION, {"role": "user", "content": "Another conversation's row."}, conversation_id="foreign",
    )
    before = _rows(engine)
    with pytest.raises(LifecyclePublicationConflictError, match="^Compaction publication source ownership changed$"):
        _stage_leaf(engine, sorted(_owners(engine)))
    assert _rows(engine) == before
    assert _owners(engine)[foreign] == "foreign"
    assert _frontier(engine) == 0 and not _coverage(engine)


def test_previously_claimed_provisional_row_still_refuses(open_engine):
    engine = open_engine()
    _seed_compacted(engine)
    covered = sorted(_coverage(engine))
    _bind_gateway(engine)
    before = _rows(engine)
    with pytest.raises(LifecyclePublicationConflictError, match="^Compaction publication source ownership changed$"):
        _stage_leaf(engine, covered)
    assert _rows(engine) == before
    assert _coverage(engine) == set(covered)
    assert _frontier(engine) == 0


def test_provisional_proof_does_not_skip_same_text_with_a_new_stamp(open_engine):
    engine = open_engine()
    early = [dict(row, timestamp=100 + index) for index, row in enumerate(EARLY)]
    engine.on_session_start(SESSION, platform="gui", context_length=200_000)
    engine.ingest(list(early))
    output = list(engine.compress(list(early)))
    assert output[-1]["timestamp"] == 103
    _bind_gateway(engine)
    repeated = dict(output[-1], timestamp=200)
    engine.ingest(output[:-1] + [repeated] + LATER)
    assert _digests(_rows(engine)) == _digests(EARLY + [repeated] + LATER)
    assert [(row["conversation_id"], row["observed_at"]) for row in _rows(engine)
            if row["content"] == repeated["content"]] == [(SESSION, 103), (GATEWAY, 200)]


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_rebind_does_not_copy_tail_when_summary_carries_a_user_turn(open_engine, cold):
    early = EARLY + [{"role": "user", "content": "One more request before the bind."}]
    engine = open_engine(fresh_tail_count=3)
    engine.on_session_start(SESSION, platform="gui", context_length=200_000)
    engine.ingest(list(early))
    output = list(engine.compress(list(early)))
    assert engine.last_compression_status == "compacted"
    assert engine._generated_context_carrier_remainder(output[0]) == early[2]["content"]
    if cold:
        engine = open_engine(fresh_tail_count=3)
    _bind_gateway(engine)
    engine.ingest(output + LATER)
    assert engine._last_ingest_reconciliation["cursor"] == len(output)
    assert _digests(_rows(engine)) == _digests(early + LATER)
    _publish(engine, output + LATER)


def test_delegate_source_excluded_from_transfer_and_bind_replay(open_engine):
    engine = open_engine()
    host = _seed_compacted(engine)
    delegate_turn = {"role": "user", "content": "Delegated child turn."}
    [delegate_id] = engine._append_off_current_session_end_suffix(
        SESSION, [delegate_turn], source=_DELEGATE_SOURCE, conversation_id=SESSION,
    )
    _bind_gateway(engine)
    engine.ingest(list(host) + LATER)
    assert engine._last_ingest_reconciliation["cursor"] == 2
    assert _digests(_rows(engine)) == _digests(EARLY + LATER + [delegate_turn])
    with pytest.raises(LifecyclePublicationConflictError, match="^Compaction publication source ownership changed$"):
        _stage_leaf(engine, [delegate_id])
    _publish(engine, host + LATER)
    assert _owners(engine)[delegate_id] == SESSION
    assert delegate_id not in _coverage(engine)


def test_delegate_row_alone_cannot_prove_a_gateway_occurrence(open_engine):
    engine = open_engine()
    turns = EARLY[:2]
    delegate_ids = engine._append_off_current_session_end_suffix(
        SESSION, turns, source=_DELEGATE_SOURCE, conversation_id=SESSION,
    )
    _bind_gateway(engine)
    engine.ingest(list(turns))
    assert all(_owners(engine)[store_id] == SESSION for store_id in delegate_ids)
    assert [row["conversation_id"] for row in _rows(engine)] == [SESSION, SESSION, GATEWAY, GATEWAY]


def test_rollback_restores_ownership_nodes_and_both_frontiers(open_engine, monkeypatch, caplog):
    engine = open_engine()
    host = _seed_mixed(engine)
    before = _rows(engine)
    old = engine._lifecycle.get_by_conversation(SESSION)
    real = engine._lifecycle.stage_compaction_publication

    def fail_after_staging(conn, *args, **kwargs):
        real(conn, *args, **kwargs)
        assert conn.execute(
            "SELECT conversation_id FROM messages WHERE store_id = ?", (before[0]["store_id"],),
        ).fetchone()[0] == GATEWAY
        assert conn.execute(
            "SELECT current_session_id FROM lcm_lifecycle_state WHERE conversation_id = ?", (SESSION,),
        ).fetchone()[0] is None
        raise LifecyclePublicationConflictError("Injected after ownership and frontier staging")

    with monkeypatch.context() as patch, caplog.at_level(logging.INFO):
        patch.setattr(engine._lifecycle, "stage_compaction_publication", fail_after_staging)
        engine.compress(list(host))
    assert engine.last_compression_status == "error"
    assert "Injected after ownership and frontier staging" in caplog.text
    assert "re-owned provisional rows" not in caplog.text
    assert _rows(engine) == before
    assert engine._lifecycle.get_by_conversation(SESSION) == old
    assert _frontier(engine) == 0 and not _coverage(engine)


OLD_RESUME = """
inputs = json.loads(data)
engine = LCMEngine(config=LCMConfig(database_path=db, **inputs["settings"]))
try:
    engine.on_session_start(inputs["session"], platform="desktop", context_length=200_000)
    assert engine._conversation_id == inputs["conversation"]
    state = engine._lifecycle
    before = state.get_by_conversation(engine._conversation_id).current_frontier_store_id
    nodes_before = engine._dag.get_session_node_count(engine._session_id)
    view = inputs["view"]
    engine.ingest(view)
    engine.compress(view)
    result = dict(status=engine.last_compression_status,
                  frontier_before=before,
                  frontier_after=state.get_by_conversation(engine._conversation_id).current_frontier_store_id,
                  nodes_before=nodes_before, nodes_after=engine._dag.get_session_node_count(engine._session_id),
                  rows=[[r["store_id"], r["conversation_id"], r["role"], r["content"]]
                        for r in engine._store.get_session_messages(engine._session_id)],
                  integrity=[r[0] for r in engine._store.connection.execute("PRAGMA integrity_check")])
finally:
    engine.shutdown()
print(json.dumps(result))
"""


def test_previous_ga_reopens_and_publishes_after_reownership(open_engine, tmp_path, old_tree, host_root):  # noqa: F811
    engine = open_engine()
    host = _seed_compacted(engine)
    _bind_gateway(engine)
    engine.ingest(host + LATER)
    output = _publish(engine, host + LATER)
    before = _rows(engine)
    frontier = _frontier(engine)
    engine.shutdown()

    new_turns = [
        {"role": "assistant", "content": "Rollback reader's first reply."},
        {"role": "user", "content": "Rollback reader's next request."},
    ]
    result = run_old(
        old_tree("v0.30.4", PREVIOUS_GA), tmp_path, host_root, OLD_RESUME,
        data=json.dumps(dict(settings=SETTINGS, session=SESSION, conversation=GATEWAY, view=output + new_turns)),
    )
    assert result["integrity"] == ["ok"]
    assert result["status"] == "compacted" and result["nodes_after"] > result["nodes_before"]
    assert result["frontier_after"] > result["frontier_before"] == frontier
    assert result["rows"][:len(before)] == [
        [row["store_id"], row["conversation_id"], row["role"], row["content"]] for row in before
    ]
    assert [(owner, role, content) for _, owner, role, content in result["rows"][len(before):]] == [
        (GATEWAY, row["role"], row["content"]) for row in new_turns
    ]
