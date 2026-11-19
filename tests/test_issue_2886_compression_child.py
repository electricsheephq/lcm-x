"""Issue #2886 (I1): replaying a compression child with name-only tool results."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER

PARENT, CHILD = "S", "S2"
CONVERSATION = "compression-child"
WINDOW = 200_000
TOOL = "read_file"
SUMMARY = "Earlier synthetic file reads.\nExpand for details about: file reads"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _exchange(index, both_names):
    rows = [{"role": "user", "content": f"Read the two synthetic files for exchange {index}."}]
    for part in ("a", "b"):
        call_id = f"call_{index}_{part}"
        rows.extend([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "type": "function", "function": {
                    "name": TOOL, "arguments": json.dumps({"path": f"sample-{index}-{part}.txt"}),
                }},
            ]},
            {"role": "tool", "tool_call_id": call_id, "name": TOOL,
             "content": f"Contents of file {index}-{part}." + " Synthetic tool output." * 20},
        ])
        if both_names:
            rows[-1]["tool_name"] = TOOL
    rows.append({"role": "assistant", "content": f"Read both files for exchange {index}."})
    for offset, row in enumerate(rows):
        row["timestamp"] = 1_796_000_000 + 10 * index + offset
        row["message_uid"] = f"message-{index}-{offset}"
    return rows


def _start(engine, session, **boundary):
    engine.on_session_start(session, platform="telegram", conversation_id=CONVERSATION,
                            context_length=WINDOW, **boundary)


def _complete(engine, live):
    engine.ingest(live)
    engine.note_turn_complete()
    engine.on_turn_complete(live)


def _stored(engine):
    return engine._store.connection.execute(
        "SELECT store_id, session_id, role, tool_call_id, content, tool_calls "
        "FROM messages ORDER BY store_id"
    ).fetchall()


def _row_key(role, call_id, content, tool_calls):
    calls_digest = hashlib.sha256(tool_calls.encode()).hexdigest() if tool_calls else None
    return role, call_id, content, calls_digest


@dataclass
class Measurement:
    added: list
    copies: list
    returned_digest: str
    expected: list


def _measure(engine, live, new_exchange, cold, both_names):
    before = _stored(engine)
    existing = {_row_key(*row[2:]): row[1] for row in before}
    proof_results = []
    routes = []
    walk = LCMEngine._cursor_from_durable_commit_proof
    log_route = LCMEngine._log_ingest_route

    def observe_walk(self, *args, **kwargs):
        result = walk(self, *args, **kwargs)
        proof_results.append(result)
        return result

    def observe_route(session, route, cursor, reason, ids, pairs=(0, 0)):
        if session == CHILD and ids:
            routes.append((cursor, reason))
        return log_route(session, route, cursor, reason, ids, pairs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LCMEngine, "_cursor_from_durable_commit_proof", observe_walk)
        patch.setattr(LCMEngine, "_log_ingest_route", staticmethod(observe_route))
        returned = engine.compress(live, force=True)

    after = _stored(engine)
    prior_ids = {row[0] for row in before}
    added = [(row[1], _row_key(*row[2:])) for row in after if row[0] not in prior_ids]
    copies = [existing[key] for _, key in added if key in existing]
    expected = [
        (CHILD, _row_key(row["role"], row.get("tool_call_id"), row["content"],
                         json.dumps(row["tool_calls"]) if "tool_calls" in row else None))
        for row in new_exchange
    ]
    result = Measurement(added, copies, _digest(returned), expected)
    print(json.dumps({
        "arm": f"{'cold' if cold else 'warm'}/{'both-names' if both_names else 'name-only'}",
        "incoming": len(live), "before": len(before), "stored": len(added), "copies": len(copies),
        "parent_copies": copies.count(PARENT), "child_copies": copies.count(CHILD),
        "routes": routes, "proof": proof_results, "returned": len(returned), "digest": result.returned_digest,
    }), flush=True)
    return result


def _run_arm(path, cold, both_names):
    path.mkdir()
    with closing(sqlite3.connect(path / "state.db")) as state:
        state.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        state.executemany("INSERT INTO sessions VALUES (?, ?, ?)",
                          [(PARENT, None, None), (CHILD, PARENT, None)])
        state.commit()

    # Use a smaller leaf so the synthetic parent can compact with a short transcript.
    config = LCMConfig(database_path=str(path / "lcm.db"), embeddings_enabled=False, leaf_chunk_tokens=2_000)

    def open_engine():
        return LCMEngine(config=config, hermes_home=str(path))

    engine = open_engine()
    try:
        _start(engine, PARENT)
        parent = [{"role": "system", "content": "Synthetic session system prompt."}]
        for index in range(18):
            parent.extend(_exchange(index, both_names))
            _complete(engine, parent)
        carried = engine.compress(parent, force=True)
        if not (len(carried) < len(parent) and carried[0]["role"] == "system"
                and "Earlier synthetic file reads." in carried[1]["content"]
                and any(row["role"] == "tool" for row in carried[2:])):
            pytest.fail(f"Parent compress() returned {len(carried)}/{len(parent)} rows "
                        f"with status {engine._last_compression_status}")
        engine.on_session_end(PARENT, parent)
        with closing(sqlite3.connect(path / "state.db")) as state:
            state.execute("UPDATE sessions SET end_reason = 'compression' WHERE id = ?", (PARENT,))
            state.commit()
        _start(engine, CHILD, boundary_reason="compression", old_session_id=PARENT)

        live = list(carried)
        for index in range(18, 30):
            # Hermes clears old tool output in its host list while the store keeps the full result.
            if index == 26:
                live = [
                    {**row, "content": _HERMES_PRUNED_TOOL_PLACEHOLDER}
                    if pos >= len(carried) and row["role"] == "tool" else row
                    for pos, row in enumerate(live)
                ]
            live.extend(_exchange(index, both_names))
            _complete(engine, live)

        new_exchange = _exchange(30, both_names)
        if cold:
            engine.shutdown()
            engine = open_engine()
            _start(engine, CHILD)
        return _measure(engine, [*live, *new_exchange], new_exchange, cold, both_names)
    finally:
        engine.shutdown()


@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    path = tmp_path_factory.mktemp("compression-child")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(lcm_engine, "summarize_with_escalation", lambda **kwargs: (SUMMARY, 1))
        return {
            (cold, both_names): _run_arm(path / f"cold-{cold}-both-names-{both_names}", cold, both_names)
            for both_names in (False, True)
            for cold in (False, True)
        }


@pytest.mark.issue(2886)
@pytest.mark.parametrize(("cold", "both_names"), [
    pytest.param(False, False, id="warm-name-only"),
    pytest.param(True, False, id="cold-name-only", marks=pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason="#2886: compression-child cold bind stores 64 rows instead of 6, including 58 copies",
    )),
    pytest.param(False, True, id="warm-both-names"),
    pytest.param(True, True, id="cold-both-names"),
])
def test_compression_child_replay(arms, cold, both_names):
    warm, actual = arms[False, both_names], arms[cold, both_names]
    if actual.returned_digest != warm.returned_digest:
        pytest.fail(f"Returned list differs from warm: {actual.returned_digest} != {warm.returned_digest}")
    if [row for row in actual.added if row in actual.expected] != actual.expected:
        pytest.fail("The six new rows were not stored exactly once in order")
    assert not actual.copies, f"Stored {len(actual.copies)} copies among {len(actual.added)} new rows"
    assert actual.added == actual.expected == warm.added
