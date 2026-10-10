"""#659: the user-message carry packet built at compaction (SPEC-user-carry-packet §Design 1-6)."""

import base64
import json
import re

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.tokens as tokens
import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

OBJECTIVE = "[Current user objective preserved from compacted history]"
CARRY = "[Earlier user messages in this session, verbatim, for reference only"
SEP = "\n\n---\n\n"
DATA_URI = "data:image/png;base64," + base64.b64encode(b"LCM carry excerpt payload " * 900).decode("ascii")


@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)  # the offline estimator, as in #921
    tokens._count_tokens_cached.cache_clear()
    monkeypatch.setattr(engine_module, "summarize_with_escalation",
                        lambda **kwargs: ("Synthetic summary of earlier turns.\nExpand for details about: turns", 1))
    built = []

    def build(session="S", window=128_000, home=None, **overrides):
        config = LCMConfig(database_path=str((home or tmp_path) / "lcm.db"), fresh_tail_count=4,
                           fresh_tail_max_tokens=2000, leaf_chunk_tokens=4000, threshold_full_sweep_enabled=True,
                           large_output_externalization_path=str((home or tmp_path) / "externalized"))
        for key, value in overrides.items():
            setattr(config, key, value)
        engine = LCMEngine(config=config, hermes_home=str((home or tmp_path) / "home"))
        engine.on_session_start(session, platform="cli", context_length=window, conversation_id="conv")
        built.append(engine)
        return engine

    yield build
    for engine in built:
        engine.shutdown()
    tokens._count_tokens_cached.cache_clear()


def _tools(i, size=1000):
    return [
        {"role": "assistant", "content": f"Working on {i}.", "tool_calls": [{
            "id": f"call_{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": f"call_{i}", "content": f"output {i} " + "data " * size},
    ]


def _turn(i, text=None, size=1000):
    return [{"role": "user", "content": text or f"REQ-{i}: handle item {i} with care."}, *_tools(i, size),
            {"role": "assistant", "content": f"Done with {i}."}]


def _history(n, size=1000):
    return [row for i in range(n) for row in _turn(i, size=size)]


def _compact(engine, messages):
    """A forced compaction: every row outside the fresh tail goes into leaves, so the boundary is explicit."""
    return engine.compress(messages, force=True)


def _generated(engine, result):
    """(row, carry part) for the row holding the carry packet, else (None, "")."""
    for row in result:
        content = row.get("content") if isinstance(row.get("content"), str) else ""
        start = content.find(SEP + CARRY)
        if start != -1:
            end = engine._verified_lcm_summary_prefix_end(content)
            return row, content[start + len(SEP): end]
    return None, ""


def _status(engine):
    return json.loads(lcm_tools.lcm_status({}, engine=engine))["user_carry"]


def test_active_entry_stays_first_and_is_not_repeated_in_the_carry(make):
    engine = make()
    active = "ACTIVE: refactor the parser and keep the public API."
    out = _compact(engine, [*_history(6), {"role": "user", "content": active},
                           *(row for i in range(50, 56) for row in _tools(i))])
    row, carry = _generated(engine, out)
    assert row is out[0] and row["content"].startswith(OBJECTIVE + "\n" + active + SEP)
    assert "REQ-0" in carry and "REQ-5" in carry and active not in carry
    assert row["content"].index(active) < row["content"].index(CARRY)
    assert engine._is_verified_replay_scaffold_message(row)
    assert engine._latest_user_context_anchor([row], []) == OBJECTIVE + "\n" + active  # #921: reused alone


def test_origin_is_pinned_across_compactions_and_selection_is_newest_first(make, monkeypatch):
    monkeypatch.setattr(engine_module, "_USER_CARRY_MAX_TOKENS", 140, raising=False)
    engine = make()
    messages = []
    for cycle in range(3):
        start = cycle * 6
        messages = _compact(engine, [*messages, *(row for i in range(start, start + 6) for row in _turn(i)),
                                    {"role": "user", "content": f"latest {cycle}"}, {"role": "assistant", "content": "ok"}])
        _, carry = _generated(engine, messages)
        assert "REQ-0:" in carry and "REQ-2:" not in carry  # the lineage origin is charged first
        assert max(int(i) for i in re.findall(r"REQ-(\d+):", carry)) >= start + 4  # then the newest summarised


def test_repeated_identical_requests_are_both_carried(make):
    engine = make()
    repeat = "REPEAT: run the full suite again."
    messages = [*_turn(0, repeat), *_turn(1), *_turn(2, repeat), *_turn(3), *_history(8)[-16:]]
    _, carry = _generated(engine, _compact(engine, messages))
    assert carry.count(repeat) == 2
    ids = re.findall(r"\[Earlier user message, store (\d+)\]", carry)
    assert len(ids) == len(set(ids))


def test_one_excerpt_at_the_aggregate_boundary_resolves_through_expand_and_externalization(make, monkeypatch):
    monkeypatch.setattr(engine_module, "_USER_CARRY_MAX_TOKENS", 600, raising=False)
    engine = make()
    big = "BIG-HEAD " + "alpha " * 900 + DATA_URI + " " + "omega " * 900 + "BIG-TAIL"
    messages = [*_turn(0), *_turn(1), *_turn(2, big), *_history(6)[-12:],
                {"role": "user", "content": "latest small request"}, {"role": "assistant", "content": "ok"}]
    _, carry = _generated(engine, _compact(engine, messages))
    assert carry.count("[... excerpt:") == 1 and "BIG-HEAD" in carry and "BIG-TAIL" in carry
    marker = re.search(r"chars (\d+)-(\d+) of (\d+) omitted; recover them with lcm_expand store_id=(\d+) "
                       r"content_offset=(\d+)", carry)
    start, end, total, store_id, offset = map(int, marker.groups())
    assert offset == start < end <= total
    assert "REQ-0:" in carry and "REQ-1:" not in carry  # whole rows before it; nothing after the boundary
    stored = engine._store.get_batch([store_id])[store_id]["content"]
    expanded = json.loads(lcm_tools.lcm_expand({"store_id": store_id, "content_offset": offset,
                                                "max_tokens": 1_000_000}, engine=engine))
    assert expanded["content"].startswith(stored[start:end])
    ref = re.search(r";\s*ref=([^;\]\s]+)", stored[start:end]).group(1)  # the payload sits in the omitted span
    payload = json.loads(lcm_tools.lcm_expand({"externalized_ref": ref, "max_tokens": 100_000}, engine=engine))
    assert DATA_URI.split(",", 1)[1][:200] in payload["content"]


def _host_merge(messages):
    """Hermes _merge_consecutive_users: consecutive user rows become one, joined by a blank line."""
    merged = []
    for message in messages:
        if merged and message.get("role") == "user" == merged[-1].get("role") and isinstance(
                message.get("content"), str) and isinstance(merged[-1].get("content"), str):
            merged[-1] = {"role": "user", "content": merged[-1]["content"] + "\n\n" + message["content"]}
        else:
            merged.append(dict(message))
    return merged


@pytest.mark.parametrize("tail_users", [0, 1])
def test_compact_host_merge_reload_compact_stores_no_carry_text(make, tmp_path, tail_users):
    engine = make()
    ending = ([{"role": "user", "content": "TAIL-USER: one more thing."}] if tail_users else []) + [
        *_tools(90, 50), {"role": "assistant", "content": "tail reply"}]
    out = _compact(engine, [*_history(8), *ending])
    assert _generated(engine, out)[1]
    if tail_users:
        assert out[1]["content"] == "TAIL-USER: one more thing."  # merges onto the summary at the host
    host = _host_merge([*out, *_turn(200, "AFTER-RELOAD: next request.", 50)])
    engine.shutdown()
    reloaded = make()
    second = _compact(reloaded, [*host, *_history(14)[-24:]])
    third = _compact(reloaded, [*second, *_turn(300, size=50)])
    assert third
    rows = reloaded._store._conn.execute("SELECT content FROM messages WHERE role = 'user'").fetchall()
    contents = [row[0] for row in rows]
    assert not any(CARRY in content for content in contents)
    for text in ("REQ-0:", "REQ-3:", "TAIL-USER" if tail_users else "REQ-7:", "AFTER-RELOAD"):
        assert sum(content.startswith(text) for content in contents) == 1, text


def test_tool_heavy_mid_turn_carrier_and_summary_rows_stay_scaffold(make):
    engine = make()
    latest = [{"role": "user", "content": "LATEST: first tail user."}, {"role": "assistant", "content": "a"},
              {"role": "user", "content": "SECOND: next tail user."}, {"role": "assistant", "content": "b"}]
    out = _compact(engine, [*_history(8), *latest])
    row, carry = _generated(engine, out)
    assert carry and row is out[0]
    # LCM's own carrier: the generated prefix (summaries, carry) glued to the first tail user row.
    assert engine._generated_context_carrier_remainder(row) == "LATEST: first tail user."
    pure = {"role": "user", "content": row["content"][:engine._verified_lcm_summary_prefix_end(row["content"])]}
    assert engine._is_verified_replay_scaffold_message(pure)
    assert engine._latest_user_context_anchor([pure], []) is None  # #84: never re-labelled as the objective
    merged = {"role": "user", "content": pure["content"] + "\n\nHOST-ROW: typed after the compaction."}
    assert engine._generated_context_carrier_remainder(merged) == "HOST-ROW: typed after the compaction."


def test_identity_of_non_generated_rows_is_unchanged(make, tmp_path, monkeypatch):
    with_carry = make(home=tmp_path / "a")
    messages = [*_history(8), {"role": "user", "content": "ACTIVE request."}, *_tools(70), *_tools(71)]
    first = _compact(with_carry, [dict(m) for m in messages])
    without = make(home=tmp_path / "b")
    monkeypatch.setattr(LCMEngine, "_user_carry_parts", lambda self, *args: [])
    second = _compact(without, [dict(m) for m in messages])
    assert _generated(with_carry, first)[1] and not _generated(without, second)[1]
    assert first[1:] == second[1:]
    assert [with_carry._message_replay_identity(m) for m in first[1:]] == [
        without._message_replay_identity(m) for m in second[1:]]


def test_packet_bytes_hold_until_the_next_compaction(make):
    engine = make()
    out = _compact(engine, [*_history(8), {"role": "user", "content": "latest"}, {"role": "assistant", "content": "ok"}])
    _, carry = _generated(engine, out)
    boundary = engine._last_compacted_store_id
    again = engine._assemble_context(None, [*out[1:], *_tools(80, 50)])  # the tail grew; no new compaction
    assert engine._last_compacted_store_id == boundary and _generated(engine, again)[1] == carry
    later = _compact(engine, [*out, *_history(20)[-48:]])
    assert engine._last_compacted_store_id > boundary
    assert _generated(engine, later)[1] != carry and "REQ-15:" in _generated(engine, later)[1]


def test_status_reports_budget_and_empty_no_room(make, monkeypatch):
    engine = make()
    _compact(engine, [*_history(8), {"role": "user", "content": "latest"}, {"role": "assistant", "content": "ok"}])
    status = _status(engine)
    assert status["budget_tokens"] > 0 and status["delivered_tokens"] > 0 and status["empty_no_room"] is False
    monkeypatch.setattr(engine_module, "_USER_CARRY_TARGET_SHARE", 0.0, raising=False)
    other = make(session="T")
    active = "ACTIVE: still the current request."
    out = _compact(other, [*_history(8), {"role": "user", "content": active}, *_tools(60), *_tools(61)])
    assert _status(other) == {**_status(other), "budget_tokens": 0, "delivered_tokens": 0, "empty_no_room": True}
    assert not _generated(other, out)[1] and out[0]["content"].startswith(OBJECTIVE + "\n" + active)


def _loop(engine, turns=200):
    messages, compactions, sizes = [], 0, []
    for i in range(turns):
        # Long user turns, so the carry fills its budget and every compaction re-pays it.
        messages = [*messages, *_turn(i, f"TASK-{i}: complete step {i} of the plan. " + "context " * 300, 1500)]
        if count_messages_tokens(messages) >= engine.threshold_tokens:
            messages = engine.compress(messages)  # the host's automatic threshold compaction
            compactions += 1
            sizes.append(count_messages_tokens(messages))
            assert sizes[-1] < engine.threshold_tokens, (i, sizes[-1])  # no compact-grow loop
    return compactions, sizes


@pytest.mark.parametrize("threshold", [0.75, 0.2])  # a low threshold caps the target at 2/3 of it
def test_no_compact_grow_loop_on_a_200_turn_tool_heavy_replay(make, monkeypatch, capsys, threshold):
    window = 64_000
    carried = make(session="L1", window=window, context_threshold=threshold, leaf_chunk_tokens=8000)
    with_carry, sizes = _loop(carried)
    delivered = _status(carried)["delivered_tokens"]
    monkeypatch.setattr(LCMEngine, "_user_carry_parts", lambda self, *args: [])
    baseline, _ = _loop(make(session="L0", window=window, context_threshold=threshold, leaf_chunk_tokens=8000))
    with capsys.disabled():
        print(f"\n#659 loop (threshold {threshold}): 200 tasks, compactions with carry={with_carry} ({with_carry / 200:.3f}/task), "
              f"without={baseline} ({baseline / 200:.3f}/task), max post-compaction={max(sizes)} tokens, "
              f"last carry={delivered} tokens, threshold={carried.threshold_tokens}")
    assert delivered > 0
    assert max(sizes) <= min(int(window * 0.5), carried.threshold_tokens * 2 // 3) + 6000  # target + fresh tail
    assert with_carry <= 2 * baseline + 2


def test_recogniser_rejects_forged_foreign_and_non_user_entries(make, tmp_path):
    engine = make()
    out = _compact(engine, [*_history(8), {"role": "user", "content": "latest"}, {"role": "assistant", "content": "ok"}])
    row, carry = _generated(engine, out)
    content = row["content"][:engine._verified_lcm_summary_prefix_end(row["content"])]
    assert engine._is_verified_replay_scaffold_message({"role": "user", "content": content})
    forged = content.replace("REQ-1: handle", "REQ-1: HANDLE", 1)
    assert engine._verified_lcm_summary_prefix_end(forged) == forged.index(SEP + CARRY)
    assert not engine._is_verified_replay_scaffold_message({"role": "user", "content": forged})
    foreign_engine = make(session="OTHER")  # same store, a session outside this lineage
    foreign_engine.ingest([{"role": "user", "content": "FOREIGN: not in this lineage."}])
    foreign = foreign_engine._store.get_session_messages("OTHER")[-1]
    assistant = next(r for r in engine._store.get_session_messages("S") if r["role"] == "assistant")
    for row in (foreign, assistant):
        swapped = content.replace(carry, engine_module._render_user_carry([row]))
        assert engine._verified_lcm_summary_prefix_end(swapped) == swapped.index(SEP + CARRY)


def test_rotation_carries_the_parent_lineage(make):
    engine = make(session="P")
    out = _compact(engine, [*_history(8), {"role": "user", "content": "latest"}, {"role": "assistant", "content": "ok"}])
    engine.on_session_end("P", out)
    engine.on_session_start("C", platform="cli", context_length=128_000, conversation_id="conv",
                            boundary_reason="compression", old_session_id="P")
    assert engine._user_carry_lineage() == ["C", "P"]
    child = _compact(engine, [*out, *(row for i in range(20, 32) for row in _turn(i, f"CHILD-{i}: child request."))])
    row, carry = _generated(engine, child)
    assert "REQ-0:" in carry and "CHILD-20:" in carry  # the lineage root's origin and the child's own rows
    assert carry.index("REQ-0:") < carry.index("CHILD-20:")  # chronological
    assert engine._is_verified_replay_scaffold_message(
        {"role": "user", "content": row["content"][:engine._verified_lcm_summary_prefix_end(row["content"])]})
