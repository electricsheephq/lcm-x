"""#2539: saved-output markers compare as storage wrote them (I1, I6)."""

import json
import tempfile
import time
from collections import Counter
from pathlib import Path

import pytest

from tests.test_issue_10_persisted_output_spillover import FULL_OUTPUT, _marker, _write_output
from tests.test_issue_1931_delegate_children import child_rows, host as host
from tests.test_issue_2482_interrupted_child import _setup, call
from tests.test_issue_2521_spillover_upgrade_replay import (
    CONVERSATION, SESSION, _engine, _host_history, _v0292_store,
)


@pytest.fixture(autouse=True)
def isolated_tempdir(tmp_path, monkeypatch):
    directory = tmp_path / "host-temp"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))


def _path(home, location, name="c1"):
    root = Path(home) / "cache" / "spillover" if location == "spillover" else Path(tempfile.gettempdir()) / "hermes-results"
    return root / f"{name}.txt"


def _view(content):
    return [{"role": "user", "content": "Dump the ledger."}, call(1),
            {"role": "tool", "tool_call_id": "c1", "content": content}]


def _assert_once(parent, handed):
    rows = child_rows(parent)
    assert len(rows) == len(handed)
    assert [(r["role"], r.get("tool_call_id")) for r in rows] == [
        (m["role"], m.get("tool_call_id")) for m in handed]
    assert Counter(r["tool_call_id"] for r in rows if r["role"] == "tool") == Counter(
        m["tool_call_id"] for m in handed if m["role"] == "tool")
    # Also verify the stored identities, not just row totals.
    assert [parent._session_end_prefix_compare_identity(r, session_id="child") for r in rows] == [
        parent._session_end_prefix_compare_identity(m, session_id="child") for m in handed]


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
@pytest.mark.parametrize("grows", [False, True], ids=["A-retry", "B-grow"])
def test_three_handovers_and_child_end_store_once(host, location, grows):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    handed = _view(_write_output(_path(engine._hermes_home, location)))
    for n in range(3):
        if grows and n:
            handed = [*handed, {"role": "assistant", "content": f"new row {n}"}]
        child.api_call(ctx, handed, n + 1)
        _assert_once(parent, handed)
    child.turn(ctx, handed, engine)
    child.end(engine, handed)
    _assert_once(parent, handed)


@pytest.mark.parametrize("kind", ["plain", "missing"])
def test_unchanged_controls(host, kind):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    content = "small result" if kind == "plain" else _marker(_path(engine._hermes_home, "spillover"))
    handed = _view(content)
    for n in range(3):
        child.api_call(ctx, handed, n + 1)
        _assert_once(parent, handed)
    child.turn(ctx, handed, engine)
    child.end(engine, handed)
    _assert_once(parent, handed)


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
@pytest.mark.parametrize("resize", [False, True], ids=["same-size", "different-size"])
def test_changed_marker_is_stored_again_identical_marker_keeps_first_content(host, location, resize):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    path = _path(engine._hermes_home, location)
    handed = _view(_write_output(path))
    child.api_call(ctx, handed, 1)
    child.api_call(ctx, handed, 2)  # warm cache; unchanged before rewrite
    _assert_once(parent, handed)
    changed = FULL_OUTPUT[:-10] + ("Z" * (20 if resize else 10))
    marker = _write_output(path, changed)
    updated = _view(marker)
    child.api_call(ctx, updated, 3)
    rows = child_rows(parent)
    assert len(rows) == len(handed) + int(resize)
    tools = [r for r in rows if r["role"] == "tool"]
    assert len(tools) == 1 + int(resize)
    payload = engine._session_end_prefix_compare_value(tools[0]["content"], session_id="child")
    assert payload == FULL_OUTPUT  # The first-ingest content was never overwritten.
    assert (marker != handed[-1]["content"]) is resize
    if resize:
        assert engine._session_end_prefix_compare_value(tools[1]["content"], session_id="child") == changed
    child.api_call(ctx, updated, 4)
    child.end(engine, updated)
    assert len(child_rows(parent)) == len(rows)


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
def test_A1_pruned_file_metadata_prevents_another_copy(host, tmp_path, location):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    path = _path(engine._hermes_home, location)
    handed = _view(_write_output(path))
    child.api_call(ctx, handed, 1)
    path.rename(tmp_path / "pruned-output")  # absent at the host path; no fixture deletes
    for n in (2, 3):
        child.api_call(ctx, handed, n)
        _assert_once(parent, handed)
    child.end(engine, handed)
    _assert_once(parent, handed)


def test_A1_inline_recovered_row_without_metadata_gets_one_extra_then_stabilizes(host, tmp_path):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    path = _path(engine._hermes_home, "spillover")
    handed = _view(_write_output(path))
    # A pre-existing inline recovered row has no source metadata. It cannot prove the missing marker.
    engine._append_off_current_session_end_suffix("child", _view(FULL_OUTPUT),
        source="delegate_child", conversation_id="child")
    path.rename(tmp_path / "pruned-output")
    for n in (1, 2, 3):
        child.api_call(ctx, handed, n)
        assert len(child_rows(parent)) == len(handed) + 1
    child.end(engine, handed)
    assert len(child_rows(parent)) == len(handed) + 1


def test_child_END_store_recovers_marker(host):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    handed = _view(_write_output(_path(engine._hermes_home, "spillover")))
    child.api_call(ctx, handed, 1)
    child.turn(ctx, handed, engine)
    child.end(engine, handed)
    _assert_once(parent, handed)


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
def test_off_current_session_end_recovers_marker(tmp_path, location):
    engine = _engine(tmp_path)
    try:
        handed = _view(_write_output(_path(engine._hermes_home, location)))
        engine.on_session_start("old", platform="cli", context_length=200_000, conversation_id="old-conv")
        engine.ingest(handed)
        engine.on_session_start("foreground", platform="cli", context_length=200_000)
        final = [*handed, {"role": "assistant", "content": "done"}]
        engine.on_session_end("old", final)
        rows = engine._store.get_session_messages("old")
        assert len(rows) == len(final)
        assert rows[-1]["content"] == "done"
        # The saved suffix outgrows the remembered raw prefix. Only the durable
        # compare can now authorize the later final row.
        later = [*final, {"role": "assistant", "content": "late final"}]
        engine.on_session_end("old", later)
        assert len(engine._store.get_session_messages("old")) == len(later)
        assert engine._store.get_session_messages("old")[-1]["content"] == "late final"
        assert engine._store.get_session_messages("foreground") == []
    finally:
        engine.shutdown()


def test_A2_v0292_bare_marker_off_current_end_and_recovered_suffix(tmp_path):
    original, marker, _ = _v0292_store(tmp_path)
    engine = _engine(tmp_path)
    try:
        engine.on_session_start(SESSION, platform="cli", context_length=200_000, conversation_id=CONVERSATION)
        engine.on_session_start("foreground", platform="cli", context_length=200_000)
        handed = _host_history(marker)
        engine.on_session_end(SESSION, handed)
        assert len(engine._store.get_session_messages(SESSION)) == len(original)
        assert sum(r["role"] == "tool" for r in engine._store.get_session_messages(SESSION)) == 1
        # Newly appended recovered markers must also remain stable on a second end.
        extra = {"role": "tool", "tool_call_id": "new-call", "content": _write_output(
            _path(engine._hermes_home, "spillover", "new-call"))}
        engine.on_session_end(SESSION, [*handed, extra])
        engine.on_session_end(SESSION, [*handed, extra])
        assert len(engine._store.get_session_messages(SESSION)) == len(original) + 1
        engine.on_session_end(SESSION, [*handed, extra, {"role": "assistant", "content": "late final"}])
        assert len(engine._store.get_session_messages(SESSION)) == len(original) + 2
    finally:
        engine.shutdown()


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
def test_stored_tail_anchor_recovers_marker(host, location):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    handed = _view(_write_output(_path(engine._hermes_home, location)))
    child.api_call(ctx, handed, 1)
    # Host native compaction rewrites the head; only the marker tool row anchors the saved tail.
    updated = [{"role": "user", "content": "Host's compacted head"}, handed[-1],
               {"role": "assistant", "content": "after host compaction"}]
    rows = child_rows(parent)
    assert engine._delegate_child_anchor_end("child", updated, [("child", r) for r in rows]) == 2
    child.api_call(ctx, updated, 2)
    assert len(child_rows(parent)) == len(rows) + 2  # unknown head and suffix are both preserved
    child.end(engine, updated)
    assert len(child_rows(parent)) == len(rows) + 2


@pytest.mark.parametrize("field", ["persisted_output_source_path", "persisted_output_expected_chars",
                                 "persisted_output_redacted_preview_sha256"])
def test_A1_partial_metadata_never_identifies_missing_marker(tmp_path, field):
    from hermes_lcm.prefix_matching import _marker_payload_metadata

    engine = _engine(tmp_path)
    try:
        path = _path(engine._hermes_home, "spillover")
        handed = _view(_write_output(path))
        engine._append_off_current_session_end_suffix("child", handed, source="delegate_child", conversation_id="child")
        stored = engine._store.get_session_messages("child")[-1]
        payload = _marker_payload_metadata(stored["content"], engine._config, engine._hermes_home)
        path.rename(tmp_path / "pruned-output")
        # Alter the synthetic payload through the loader only, leaving durable rows untouched.
        original_loader = engine._session_end_marker_identity.__globals__["_marker_payload_metadata"]
        namespace = engine._session_end_marker_identity.__globals__
        damaged = dict(payload)
        damaged.pop(field)
        try:
            namespace["_marker_payload_metadata"] = lambda *a, **kw: damaged
            assert engine._session_end_prefix_compare_identity(stored, session_id="child") != engine._session_end_prefix_compare_identity(
                handed[-1], session_id="child")
        finally:
            namespace["_marker_payload_metadata"] = original_loader
    finally:
        engine.shutdown()


@pytest.mark.parametrize("pruned", [False, True], ids=["present", "gone"])
def test_A1_byte_identical_host_marker_text_equals_stored_rows(tmp_path, pruned):
    engine = _engine(tmp_path)
    try:
        path = _path(engine._hermes_home, "spillover")
        for full in (FULL_OUTPUT, FULL_OUTPUT[:-10] + "Z" * 10):
            row = _view(_write_output(path, full))[-1]
            engine._append_off_current_session_end_suffix("child", [row],
                source="delegate_child", conversation_id="child")
        rows = engine._store.get_session_messages("child")
        assert len(rows) == 2 and rows[0]["content"] != rows[1]["content"]
        if pruned:
            path.rename(tmp_path / "pruned-output")
        assert engine._session_end_prefix_compare_identity(rows[0], session_id="child") == engine._session_end_prefix_compare_identity(
            rows[1], session_id="child")
        assert engine._session_end_prefix_compare_identity(rows[0], session_id="child") == engine._session_end_prefix_compare_identity(
            row, session_id="child")
    finally:
        engine.shutdown()


def test_cost_ten_megabyte_marker_handover(host, capsys):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    handed = [{"role": "user", "content": "Read ten synthetic outputs."}]
    full = "synthetic measurement\n" + "x" * (1024 * 1024 - len("synthetic measurement\n"))
    for n in range(10):
        handed.extend([call(n), {"role": "tool", "tool_call_id": f"c{n}", "content": _write_output(
            _path(engine._hermes_home, "spillover", f"c{n}"), full[:-1] + str(n))}])
    times = []
    for n in range(2):
        began = time.perf_counter()
        child.api_call(ctx, handed, n + 1)
        times.append(time.perf_counter() - began)
        _assert_once(parent, handed)
    with capsys.disabled():
        print("2539_COST_SECONDS " + json.dumps(dict(zip(("cold", "warm"), times))))


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
def test_offcurrent_same_history_changed_file_keeps_first_content(tmp_path, location):
    """Named limit: replacing a result in place keeps the first-ingested content."""
    engine = _engine(tmp_path)
    try:
        path = _path(engine._hermes_home, location)
        marker = _write_output(path)
        handed = _view(marker)
        engine.on_session_start("old", platform="cli", context_length=200_000, conversation_id="old-conv")
        engine.ingest(handed)
        engine.on_session_start("foreground", platform="cli", context_length=200_000)
        changed = FULL_OUTPUT[:-10] + "Z" * 10
        assert _write_output(path, changed) == marker
        final = [*handed, {"role": "assistant", "content": "after reused call"}]
        engine.on_session_end("old", final)
        rows = engine._store.get_session_messages("old")
        tools = [r for r in rows if r["role"] == "tool"]
        assert len(tools) == 1
        assert engine._session_end_prefix_compare_value(tools[0]["content"], session_id="old") == FULL_OUTPUT
        assert all(engine._session_end_prefix_compare_value(r["content"], session_id="old") != changed for r in tools)
        assert engine._session_end_prefix_compare_identity(tools[-1], session_id="old") == engine._session_end_prefix_compare_identity(
            handed[-1], session_id="old")
        assert rows[-1]["content"] == "after reused call"
        engine.on_session_end("old", final)
        assert sum(r["content"] == "after reused call" for r in engine._store.get_session_messages("old")) == 1
        engine.on_session_end("old", [*final, {"role": "assistant", "content": "later row"}])
        assert sum(r["content"] == "later row" for r in engine._store.get_session_messages("old")) == 1
        assert sum(r["role"] == "tool" for r in engine._store.get_session_messages("old")) == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
def test_child_end_and_next_compression_marker_boundary_covered(host, monkeypatch, location):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    handed = _view(_write_output(_path(engine._hermes_home, location)))
    child.api_call(ctx, handed, 1)
    child.end(engine, handed)
    _assert_once(parent, handed)
    aux = ctx.module("aux_session")
    covered_calls = []
    remember = engine._remember_delegate_child_outputs
    monkeypatch.setattr(engine, "_remember_delegate_child_outputs",
                        lambda *args: (covered_calls.append(args), remember(*args))[-1])
    # Isolate the store/covered/boundary contract from the host compressor's choice of summary.
    monkeypatch.setattr(engine, "_compress_bypassed", lambda messages, *a: messages)
    child.compress(engine, handed)  # marker is the last row: covered must be True
    assert len(covered_calls) == 1
    assert aux._DELEGATE_RESUME_BOUNDARIES.get(engine._delegate_child_key("child"))
    updated = [*handed, {"role": "assistant", "content": "after child end"}]
    child.compress(engine, updated)
    child.end(engine, updated)
    _assert_once(parent, updated)
    assert len(covered_calls) == 2
