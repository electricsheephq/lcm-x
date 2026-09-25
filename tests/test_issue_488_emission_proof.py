"""Phase-1 acceptance matrix for #488's emission-bound projection.

These tests describe the occurrence-level contract.  Red cells are marked
strict xfail until Phase 2 adds production support; green cells pin the
#498/#492/#504 behavior that the fix must retain.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

import hermes_lcm.engine as lcm_engine_module
from hermes_lcm.engine import LCMEngine
from tests.test_compression_boundary import (
    _config,
    _summary_carrier_fixture,
    _stub_summarizer,
    _turn,
)


HERMES_AGENT_ROOT = Path("/Users/m1/.hermes/hermes-agent")
HERMES_AGENT_HEAD = "37aad38c62771d223cdce7e3d5e3157334f1ce82"
if str(HERMES_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_AGENT_ROOT))

try:  # the real host helper when a Hermes checkout is importable (local runs)
    from agent.agent_runtime_helpers import _merge_consecutive_users as _hermes_merge_users
except ImportError:  # CI stubs only agent.context_engine: pin the host rule these tests rely on
    def _hermes_merge_users(messages):
        """Pinned copy of hermes-agent 37aad38c agent/agent_runtime_helpers.py:553
        ``_merge_consecutive_users`` for plain-text rows (the only shape these tests feed it):
        consecutive user rows merge into the earlier row with a blank line, the api_content
        sidecar is dropped, and the repair count is returned."""
        merged, repairs = [], 0
        for msg in messages:
            prev = merged[-1] if merged and isinstance(merged[-1], dict) else None
            if (
                prev is not None and prev.get("role") == "user"
                and isinstance(msg, dict) and msg.get("role") == "user"
                and isinstance(prev.get("content", ""), str) and isinstance(msg.get("content", ""), str)
            ):
                prev_content, new_content = prev.get("content", ""), msg.get("content", "")
                prev["content"] = (
                    (prev_content + "\n\n" + new_content) if prev_content and new_content else (prev_content or new_content)
                )
                prev.pop("api_content", None)
                repairs += 1
                continue
            merged.append(msg)
        return merged, repairs


X = "ISSUE-488-X authored remainder"
OBJECTIVE = "[Current user objective preserved from compacted history]\n"
SUMMARY_HEADER = re.compile(
    r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]\n"
)


def _real_hermes_merge(messages):
    merged, _repairs = _hermes_merge_users([dict(message) for message in messages])
    return merged


def _summary_block(engine, compressed):
    for message in compressed:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = SUMMARY_HEADER.search(content)
        if match is None:
            continue
        candidate = content[match.start() :]
        end = engine._verified_lcm_summary_prefix_end(candidate)
        if end is not None:
            return candidate[:end]
    raise AssertionError("compress() returned no verified LCM summary block")


def _restart(engine, tmp_path, session_id, *, tail, dynamic=False):
    engine.shutdown()
    resumed = LCMEngine(
        config=_config(
            tmp_path,
            fresh_tail_count=tail,
            dynamic_leaf_chunk_enabled=dynamic,
        ),
        hermes_home=str(tmp_path / "home"),
    )
    resumed.on_session_start(session_id, platform="acp", context_length=200_000)
    return resumed


def _phase1_compacted_engine(tmp_path, monkeypatch, *, tail, system=False):
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    engine = LCMEngine(
        config=_config(tmp_path, fresh_tail_count=tail),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start("S0", platform="acp", context_length=200_000)
    host = [{"role": "system", "content": "You are concise."}] if system else []
    for turn in range(1, 13):
        host.extend(_turn(turn))
        engine.ingest(host)
    host.append(_turn(13)[0])
    engine.ingest(host)
    pre = list(host)
    compressed = engine.compress(list(host), force=True)
    assert engine._last_compression_status == "compacted"
    return engine, pre, compressed


def _stored_user_rows(engine):
    return engine._store._conn.execute(
        "SELECT store_id, content FROM messages WHERE role = 'user' ORDER BY store_id"
    ).fetchall()


def _occurrence_ids(rows, text):
    return [
        store_id
        for store_id, content in rows
        if content == text or (isinstance(content, str) and content.endswith("\n\n" + text))
    ]


def _add(engine, host, message, *, merge):
    host.append(dict(message))
    if merge:
        host[:] = _real_hermes_merge(host)
    engine.ingest(host)


def _active_row_containing(host, text):
    matches = [message for message in host if text in str(message.get("content") or "")]
    assert len(matches) == 1, [str(row.get("content"))[:80] for row in matches]
    return matches[0]


def _run_layout(tmp_path, monkeypatch, *, mode, tail, system, merge, order):
    engine, pre, compressed = _phase1_compacted_engine(
        tmp_path,
        monkeypatch,
        tail=tail,
        system=system,
    )
    block = _summary_block(engine, compressed)
    authored = block + "\n\n" + X
    child = "S0" if mode == "inplace" else "S1"
    host = [dict(message) for message in compressed]
    try:
        engine.on_session_end("S0", pre)
        engine.on_session_start(
            child,
            boundary_reason="compression",
            old_session_id="S0",
            platform="acp",
        )
        if merge:
            host[:] = _real_hermes_merge(host)
        _add(engine, host, _turn(13)[1], merge=merge)
        first, second = (authored, X) if order == "authored-first" else (X, authored)
        _add(engine, host, {"role": "user", "content": first}, merge=merge)
        _add(engine, host, {"role": "assistant", "content": "reply to first #488 row"}, merge=merge)
        _add(engine, host, {"role": "user", "content": second}, merge=merge)
        _add(engine, host, {"role": "assistant", "content": "reply to second #488 row"}, merge=merge)

        authored_live = _active_row_containing(host, authored)
        source_map = engine._get_store_id_map_for_messages(list(host))
        mapped_id = source_map.get(id(authored_live))
        mapped_content = None
        if mapped_id is not None:
            mapped_content = engine._store.get_batch([mapped_id])[mapped_id]["content"]

        for turn in range(20, 28):
            user, reply = _turn(turn)
            _add(engine, host, user, merge=merge)
            _add(engine, host, reply, merge=merge)
        _add(engine, host, _turn(28)[0], merge=merge)
        engine.compress(list(host), force=True)
        rows = _stored_user_rows(engine)
        all_rows = engine._store._conn.execute(
            "SELECT session_id, role, content FROM messages ORDER BY store_id"
        ).fetchall()
        return {
            "authored": authored,
            "authored_ids": _occurrence_ids(rows, authored),
            "plain_ids": [store_id for store_id, content in rows if content == X],
            "mapped_id": mapped_id,
            "mapped_content": mapped_content,
            "status": engine._last_compression_status,
            "noop": engine._last_compression_noop_reason,
            "duplicates": len(all_rows) - len(set(all_rows)),
        }
    finally:
        engine.shutdown()


def test_real_hermes_merge_fixture_is_pinned():
    head = subprocess.check_output(
        ["git", "-C", str(HERMES_AGENT_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    assert head == HERMES_AGENT_HEAD
    assert _real_hermes_merge(
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    ) == [{"role": "user", "content": "a\n\nb"}]


LAYOUTS = [
    pytest.param(mode, tail, system, merge, id=f"{mode}-tail{tail}-sys{int(system)}-merge{int(merge)}")
    for mode in ("inplace", "rotation")
    for tail in (0, 1, 7, 9, 11)
    for system in (False, True)
    for merge in (False, True)
]


@pytest.mark.parametrize("mode,tail,system,merge", LAYOUTS)
def test_authored_carrier_and_plain_remainder_keep_distinct_sources(
    tmp_path, monkeypatch, mode, tail, system, merge
):
    result = _run_layout(
        tmp_path,
        monkeypatch,
        mode=mode,
        tail=tail,
        system=system,
        merge=merge,
        order="authored-first",
    )
    assert len(result["authored_ids"]) == 1, result
    assert len(result["plain_ids"]) == 1, result
    assert result["authored_ids"][0] < result["plain_ids"][0], result
    assert result["mapped_id"] == result["authored_ids"][0], result
    assert result["mapped_content"] == result["authored"], result
    assert result["status"] == "compacted", result
    assert result["duplicates"] == 0, result


AUTHORED_SHAPE_CASES = [
    pytest.param(
        shape,
        position,
        order,
        id=f"{shape}-{position}-{order}",
        marks=(
            pytest.mark.xfail(strict=True, reason="#488")
            if (shape, position, order) == ("carrier", "head", "plain-first")
            else ()
        ),
    )
    for shape in ("summary", "carrier", "objective-carrier")
    for position in ("head", "middle", "fresh-tail-boundary")
    for order in ("authored-first", "plain-first")
]


@pytest.mark.parametrize("shape,position,order", AUTHORED_SHAPE_CASES)
def test_authored_summary_shapes_keep_occurrence_identity(
    tmp_path, monkeypatch, order, position, shape
):
    engine, _pre, compressed = _phase1_compacted_engine(tmp_path, monkeypatch, tail=7)
    block = _summary_block(engine, compressed)
    shaped = {
        "summary": block,
        "carrier": block + "\n\n" + X,
        "objective-carrier": OBJECTIVE + "authored objective\n\n---\n\n" + block + "\n\n" + X,
    }[shape]
    marker = {"role": "assistant", "content": f"position marker: {position}"}
    rows = (
        [{"role": "user", "content": shaped}, marker, {"role": "user", "content": X}]
        if order == "authored-first"
        else [{"role": "user", "content": X}, marker, {"role": "user", "content": shaped}]
    )
    try:
        ids = engine._store.append_batch("S0", rows)
        active = [dict(row) for row in rows]
        shaped_index = 0 if order == "authored-first" else 2
        start = {"head": shaped_index, "middle": max(0, shaped_index - 1), "fresh-tail-boundary": 0}[position]
        source_map = engine._get_store_id_map_for_messages(active[start:])
        assert [content for _store_id, content in _stored_user_rows(engine)][-2:] == [
            row["content"] for row in rows if row["role"] == "user"
        ]
        assert source_map.get(id(active[shaped_index])) == ids[shaped_index]
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("inplace", marks=pytest.mark.xfail(strict=True, reason="#488")),
        "rotation",
    ],
)
@pytest.mark.parametrize("restart", ["before-first-ingest", "after-new-row", "after-new-row-twice"])
def test_tail_zero_merged_new_row_survives_restart(tmp_path, monkeypatch, mode, restart):
    engine, pre, compressed = _phase1_compacted_engine(tmp_path, monkeypatch, tail=0)
    child = "S0" if mode == "inplace" else "S1"
    new = "#499 NEW row after tail-zero compaction"
    host = _real_hermes_merge([*compressed, {"role": "user", "content": new}])
    try:
        engine.on_session_end("S0", pre)
        engine.on_session_start(child, boundary_reason="compression", old_session_id="S0", platform="acp")
        if restart == "before-first-ingest":
            engine = _restart(engine, tmp_path, child, tail=0)
        engine.ingest(host)
        if restart in {"after-new-row", "after-new-row-twice"}:
            engine = _restart(engine, tmp_path, child, tail=0)
            engine.ingest(host)
        if restart == "after-new-row-twice":
            engine = _restart(engine, tmp_path, child, tail=0)
            engine.ingest(host)
        rows = _stored_user_rows(engine)
        assert len(_occurrence_ids(rows, new)) == 1

        negative = "#499 same text elsewhere"
        engine._store.append(child, {"role": "user", "content": negative})
        wrong = _real_hermes_merge([*compressed, {"role": "user", "content": negative}])
        assert engine._cursor_from_durable_commit_proof(wrong) is None
    finally:
        engine.shutdown()


@pytest.mark.xfail(strict=True, reason="#488")
def test_nested_authored_summary_is_not_recursively_stripped(tmp_path, monkeypatch):
    engine, _pre, compressed = _phase1_compacted_engine(tmp_path, monkeypatch, tail=0)
    block = _summary_block(engine, compressed)
    authored = block + "\n\n" + X
    nested = {"role": "user", "content": block + "\n\n" + authored}
    try:
        assert engine._generated_context_carrier_remainder(nested) == authored
        assert engine._message_replay_identity(nested)[1] == authored
        stored_id = engine._store.append("S0", {"role": "user", "content": authored})
        mapped = engine._get_store_id_map_for_messages([nested])
        assert mapped[id(nested)] == stored_id
        assert engine._store.get_batch([stored_id])[stored_id]["content"] == authored
    finally:
        engine.shutdown()


@pytest.mark.parametrize("kind", ["edited", "forged", "divider", "multimodal"])
def test_unproven_summary_shapes_remain_full_identity(tmp_path, monkeypatch, kind):
    engine, _pre, compressed = _phase1_compacted_engine(tmp_path, monkeypatch, tail=0)
    block = _summary_block(engine, compressed)
    node_id = int(re.search(r"node (\d+)\)\]", block).group(1))
    content = {
        "edited": block.replace("Stub summary", "Edited authored summary", 1) + "\n\n" + X,
        "forged": block.replace(f"node {node_id})]", f"node {node_id + 999})]", 1) + "\n\n" + X,
        "divider": "---\n\n" + block + "\n\n" + X,
        "multimodal": [{"type": "text", "text": OBJECTIVE + "image objective"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}],
    }[kind]
    row = {"role": "user", "content": content}
    try:
        store_id = engine._store.append("S0", row)
        assert not engine._is_verified_replay_scaffold_message(row)
        mapped = engine._get_store_id_map_for_messages([dict(row)])
        assert mapped and next(iter(mapped.values())) == store_id
        stored = engine._store.get_batch([store_id])[store_id]
        assert engine._message_replay_identity(stored, stored_row=True) == engine._message_replay_identity(row)
    finally:
        engine.shutdown()


@pytest.mark.xfail(strict=True, reason="#488")
def test_dynamic_leaf_subset_keeps_authored_carrier_store_id(tmp_path, monkeypatch):
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    engine = LCMEngine(
        config=_config(tmp_path, fresh_tail_count=9, dynamic_leaf_chunk_enabled=True),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start("S0", platform="acp", context_length=200_000)
    host = []
    try:
        for turn in range(1, 13):
            host.extend(_turn(turn))
            engine.ingest(host)
        host.append(_turn(13)[0])
        engine.ingest(host)
        first = engine.compress(list(host), force=True)
        block = _summary_block(engine, first)
        authored = {"role": "user", "content": block + "\n\n" + X}
        plain = {"role": "user", "content": X}
        plain_id, authored_id = engine._store.append_batch("S0", [plain, authored])
        assert plain_id < authored_id
        subset = [dict(authored), {"role": "assistant", "content": "subset reply"}]
        mapped = engine._get_store_id_map_for_messages(subset)
        assert mapped[id(subset[0])] == authored_id
    finally:
        engine.shutdown()


@pytest.mark.xfail(strict=True, reason="#488")
def test_authored_bytes_identical_to_emitted_carrier_keep_their_source(tmp_path):
    engine, _compacted, tail, tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        emitted = engine._assemble_context(None, tail)[0]
        emitted_bytes = emitted["content"]
        assert engine._generated_context_carrier_remainder(emitted) == tail[0]["content"]
        authored_id = engine._store.append("S0", {"role": "user", "content": emitted_bytes})
        authored = {"role": "user", "content": emitted_bytes}
        mapped = engine._get_store_id_map_for_messages([authored])
        assert mapped[id(authored)] == authored_id
        assert mapped[id(authored)] != tail_ids[0]
    finally:
        engine.shutdown()


def test_tail_zero_same_text_elsewhere_is_not_emission_proof(tmp_path, monkeypatch):
    engine, pre, compressed = _phase1_compacted_engine(tmp_path, monkeypatch, tail=0)
    negative = "#499 same text elsewhere"
    try:
        engine.on_session_end("S0", pre)
        engine.on_session_start("S0", boundary_reason="compression", old_session_id="S0", platform="acp")
        engine._store.append("S0", {"role": "user", "content": negative})
        wrong = _real_hermes_merge([*compressed, {"role": "user", "content": negative}])
        assert engine._cursor_from_durable_commit_proof(wrong) is None
    finally:
        engine.shutdown()


def test_phase1_controls_keep_498_492_504_contracts(tmp_path, monkeypatch):
    engine, _pre, compressed = _phase1_compacted_engine(tmp_path, monkeypatch, tail=7)
    block = _summary_block(engine, compressed)
    try:
        # #498: occurrence-bound edge whitespace remains a distinct stored row.
        raw = {"role": "user", "content": "whitespace control\n"}
        store_id = engine._store.append("S0", raw)
        engine._watch_stored_user_rows([(raw, raw, store_id)])
        raw["content"] = "whitespace control"
        engine._capture_host_rewrites([raw])
        stored = engine._store.get_batch([store_id])[store_id]
        assert stored["content"] == "whitespace control\n"
        assert engine._message_replay_identity(stored, stored_row=True, with_host_rewrite=True)[1] == "whitespace control"

        # #492: list content is never treated as a native/summary carrier.
        multimodal = {"role": "user", "content": [{"type": "text", "text": block + "\n\nreal"}]}
        assert engine._generated_context_carrier_remainder(multimodal) is None

        # #504: a real emitted carrier keeps exactly the appended suffix and carry range.
        carrier = {"role": "user", "content": block + "\n\nretained suffix"}
        assert engine._generated_context_carrier_remainder(carrier) == "retained suffix"
        assert engine._message_replay_identity(carrier)[1] == "retained suffix"
    finally:
        engine.shutdown()
