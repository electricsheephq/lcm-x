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


def _assert_once(parent, handed, session_id="child"):
    rows = child_rows(parent, session_id)
    assert len(rows) == len(handed)
    assert [(r["role"], r.get("tool_call_id")) for r in rows] == [
        (m["role"], m.get("tool_call_id")) for m in handed]
    assert Counter(r["tool_call_id"] for r in rows if r["role"] == "tool") == Counter(
        m["tool_call_id"] for m in handed if m["role"] == "tool")
    assert all(parent._session_end_rows_equal(r, m, session_id=session_id) for r, m in zip(rows, handed))
    # Independent of the helper under test: every non-tool row keeps the host's exact text.
    assert [r["content"] for r in rows if r["role"] != "tool"] == [
        m["content"] for m in handed if m["role"] != "tool"]


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


@pytest.mark.parametrize("field", ["persisted_output_source_path", "persisted_output_expected_chars"])
def test_A1_partial_metadata_never_identifies_missing_marker(host, tmp_path, monkeypatch, field):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    path = _path(engine._hermes_home, "spillover")
    handed = _view(_write_output(path))
    child.api_call(ctx, handed, 1)
    path.rename(tmp_path / "pruned-output")
    namespace = engine._session_end_prefix_compare_value.__func__.__globals__
    loader = namespace["_marker_payload_metadata"]

    def damaged(*args):
        payload = dict(loader(*args))
        payload.pop(field, None)
        return payload

    monkeypatch.setitem(namespace, "_marker_payload_metadata", damaged)
    for n in (2, 3):
        child.api_call(ctx, handed, n)
        assert len(child_rows(parent)) == 4
    child.end(engine, handed)
    assert len(child_rows(parent)) == 4


@pytest.mark.parametrize("pruned", [False, True], ids=["present", "gone"])
def test_A1_byte_identical_host_marker_text_equals_stored_rows(host, tmp_path, pruned):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    path = _path(engine._hermes_home, "spillover")
    marker = _write_output(path)
    handed = _view(marker)
    child.api_call(ctx, handed, 1)
    assert _write_output(path, FULL_OUTPUT[:-10] + "Z" * 10) == marker
    if pruned:
        path.rename(tmp_path / "pruned-output")
    for n in (2, 3):
        child.api_call(ctx, handed, n)
        assert len(child_rows(parent)) == 3
    child.end(engine, handed)
    rows = child_rows(parent)
    assert len(rows) == 3
    assert engine._session_end_prefix_compare_value(rows[-1]["content"], session_id="child") == FULL_OUTPUT


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
        assert engine._session_end_rows_equal(tools[-1], handed[-1], session_id="old")
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


_SHARED = "SHARED_NEEDLE:\n" + "payload line abcdef 0123456789\n" * 700


def _shared_view(engine, shape):
    first = _write_output(_path(engine._hermes_home, "spillover"), _SHARED)
    second = (_SHARED if shape == "marker-then-plain" else
              _write_output(_path(engine._hermes_home, "hermes-results"), _SHARED))
    return [*_view(first), {"role": "assistant", "content": "again"}, call(1),
            {"role": "tool", "tool_call_id": "c1", "content": second}]


@pytest.mark.parametrize("shape", ["marker-then-plain", "spill-then-temp"])
def test_shared_payload_handovers_store_every_host_row_once_and_append_new_turn(host, shape):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    handed = _shared_view(engine, shape)
    counts = []
    for n in range(3):
        child.api_call(ctx, handed, n + 1)
        counts.append(len(child_rows(parent)))
        _assert_once(parent, handed)
    child.end(engine, handed)
    counts.append(len(child_rows(parent)))
    _assert_once(parent, handed)
    # Main: marker-then-plain 6,7,7,7; spill-then-temp 6,8,10,12.
    # 13211798: both shapes 6,9,12,15. Corrected bar: the host's six rows once.
    assert counts == [6, 6, 6, 6]
    updated = [*handed, {"role": "user", "content": "new question"},
               {"role": "assistant", "content": "new answer"}]
    child.api_call(ctx, updated, 4)
    _assert_once(parent, updated)
    assert len(child_rows(parent)) == 8


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
@pytest.mark.parametrize("change", ["leading-whitespace", "missing-boilerplate"])
def test_astra_distinct_bare_markers_store_four_rows(host, location, change):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    path = _path(engine._hermes_home, location)
    if change == "leading-whitespace":
        old = _write_output(path, "\n\n" + "A" * 6000)
        new = _marker(path, " \n" + "A" * 6000)
    else:
        old = _marker(path)
        new = old.replace("Use the read_file tool", "SECOND literal message: Use the read_file tool")
    child.api_call(ctx, _view(old), 1)
    assert child_rows(parent)[-1]["content"] == old  # Both shapes are BARE, not recovered.
    if change == "leading-whitespace":
        assert _write_output(path, " \n" + "A" * 6000) == new
    for n in (2, 3):
        child.api_call(ctx, _view(new), n)
        rows = child_rows(parent)
        assert len(rows) == 4
        assert [r["content"] for r in rows if r["role"] == "tool"] == [old, new]
    child.end(engine, _view(new))
    assert len(child_rows(parent)) == 4


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
@pytest.mark.parametrize("change,extra", [
    ("last-preview-character", 1), ("recovered-boilerplate", 0), ("bare-boilerplate", 1),
    ("crlf", 0), ("redaction", 1), ("empty-preview", 1),
    ("other-path-present", 0), ("other-path-missing", 1), ("nested-marker", 0),
])
def test_marker_identity_through_live_handover(host, tmp_path, location, change, extra):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    path = _path(engine._hermes_home, location)
    full = FULL_OUTPUT
    if change == "redaction":
        engine._config.sensitive_patterns_enabled = True
        engine._config.sensitive_patterns = ["api_key"]
        full = "api_key=synthetic2539secret\n" + FULL_OUTPUT
    elif change == "nested-marker":
        full = _marker(_path(engine._hermes_home, location, "inner"))
    old = _marker(path) if change == "bare-boilerplate" else _write_output(path, full)
    if change == "crlf":
        path.write_bytes(full.replace("\n", "\r\n").encode())
    child.api_call(ctx, _view(old), 1)
    stored = child_rows(parent)[-1]
    restored = engine._session_end_prefix_compare_value(stored["content"], session_id="child")
    if change not in ("bare-boilerplate", "nested-marker"):
        assert restored != old  # Prove this exercises a recovered row.
    new = old
    if change == "last-preview-character":
        preview = full[:30]
        new = _marker(path, full, preview=preview[:-1] + "!")
    elif change.endswith("boilerplate"):
        new = old.replace("Use the read_file tool", "SECOND literal message: Use the read_file tool")
    elif change == "empty-preview":
        new = _marker(path, full, preview="")
        path.rename(tmp_path / "pruned-empty-preview")  # Path/count alone must not identify it.
    elif change.startswith("other-path"):
        other = _path(engine._hermes_home, location, "other")
        new = _write_output(other, full) if change.endswith("present") else _marker(other, full)
    if change == "redaction":
        assert not restored.startswith(full[:30])
        # With the file present the counted fallback recovers and redacts both sides.
        # Pruning leaves only the mismatched preview; that uncertainty stores one extra copy.
        path.rename(tmp_path / "pruned-redaction")
    if change == "nested-marker":
        # Main's replay counts are 3,4,5,6; the new tree keeps 3,3,3,3 and the inner text.
        assert full in restored
    assert engine._session_end_rows_equal(stored, _view(new)[-1], session_id="child") is (extra == 0)
    for n in (2, 3):
        child.api_call(ctx, _view(new), n)
        rows = child_rows(parent)
        assert len(rows) == 3 + extra
        assert rows[0]["content"] == "Dump the ledger."
        assert rows[1]["tool_calls"]
        assert rows[2]["content"] == stored["content"]  # First content is never replaced.
        if extra:
            if change == "redaction":
                assert engine._session_end_prefix_compare_identity(rows[-1], session_id="child") == (
                    engine._session_end_prefix_compare_identity(_view(new)[-1], session_id="child"))
            else:
                assert engine._session_end_rows_equal(rows[-1], _view(new)[-1], session_id="child")
    child.end(engine, _view(new))
    assert len(child_rows(parent)) == 3 + extra


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
@pytest.mark.parametrize("ordering", ["inline-then-marker", "marker-then-inline"])
def test_host_inline_marker_orderings_store_every_row_once_and_append_new_turn(tmp_path, location, ordering):
    engine = _engine(tmp_path)
    try:
        marker = _write_output(_path(engine._hermes_home, location))
        first, second = (FULL_OUTPUT, marker) if ordering == "inline-then-marker" else (marker, FULL_OUTPUT)
        # Context-engine tools ingest foreground messages; delegated sessions suppress that ingest.
        # Include a plain row sharing the payload so it cannot borrow the marker identity.
        tail = [{"role": "assistant", "content": "again"}, call(1),
                {"role": "tool", "tool_call_id": "c1", "content": FULL_OUTPUT}]
        engine.on_session_start("old", platform="cli", context_length=200_000, conversation_id="old-conv")
        engine.handle_tool_call("lcm_grep", {"query": "SPILLOVER_RECOVERY_NEEDLE"}, messages=[*_view(first), *tail])
        assert len(engine._store.get_session_messages("old")) == 6
        engine.on_session_start("foreground", platform="cli", context_length=200_000)
        handed = [*_view(second), *tail]
        counts = []
        for _ in range(4):
            engine.on_session_end("old", handed)
            _assert_once(engine, handed, "old")
            counts.append(len(engine._store.get_session_messages("old")))
        assert counts == [6, 6, 6, 6]
        updated = [*handed, {"role": "user", "content": "new question"},
                   {"role": "assistant", "content": "new answer"}]
        engine.on_session_end("old", updated)
        _assert_once(engine, updated, "old")
        assert len(engine._store.get_session_messages("old")) == 8
    finally:
        engine.shutdown()


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
@pytest.mark.parametrize("ordering", ["inline-then-marker", "marker-then-inline"])
def test_child_inline_marker_orderings_store_six_rows_once_and_append_new_turn(host, location, ordering):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    marker = _write_output(_path(engine._hermes_home, location))
    first, second = (FULL_OUTPUT, marker) if ordering == "inline-then-marker" else (marker, FULL_OUTPUT)
    tail = [{"role": "assistant", "content": "again"}, call(1),
            {"role": "tool", "tool_call_id": "c1", "content": FULL_OUTPUT}]
    child.api_call(ctx, [*_view(first), *tail], 1)
    handed = [*_view(second), *tail]
    counts = []
    for n in (2, 3, 4):
        child.api_call(ctx, handed, n)
        _assert_once(parent, handed)
        counts.append(len(child_rows(parent)))
    child.end(engine, handed)
    _assert_once(parent, handed)
    counts.append(len(child_rows(parent)))
    assert counts == [6, 6, 6, 6]
    updated = [*handed, {"role": "user", "content": "new question"},
               {"role": "assistant", "content": "new answer"}]
    child.api_call(ctx, updated, 5)
    _assert_once(parent, updated)
    assert len(child_rows(parent)) == 8


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
@pytest.mark.parametrize("limit", ["missing-file", "rewritten-file", "reused-call-id"])
def test_counted_fallback_keeps_every_row_with_at_most_one_extra_copy(host, tmp_path, caplog, location, limit):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    path = _path(engine._hermes_home, location)
    old = _write_output(path)
    # A trailing row absent from the next view prevents a positional or stored-tail match.
    first = [*_view(old), {"role": "assistant", "content": "stored tail outside the host view"}]
    child.api_call(ctx, first, 1)
    if limit == "missing-file":
        path.rename(tmp_path / "pruned-counted")
        tool = _view(old)[-1]
    else:
        changed = FULL_OUTPUT + "new output"
        tool = _view(_write_output(path, changed))[-1]
    updated = [tool, {"role": "user", "content": "new question"},
               {"role": "assistant", "content": "new answer"}]
    if limit == "reused-call-id":
        # Two occurrences under one call id; the first is inline so a changed file cannot rewrite it.
        updated.insert(0, _view(FULL_OUTPUT)[-1])
    before = child_rows(parent)
    caplog.clear()
    child.api_call(ctx, updated, 2)
    assert "resuming with counted identities" in caplog.text
    rows = child_rows(parent)
    assert rows[:len(before)] == before
    assert len(rows) == len(before) + 3  # One distinct/uncertain result plus the two genuinely new rows.
    assert engine._session_end_rows_equal(rows[-3], tool, session_id="child")
    assert [r["content"] for r in rows[-2:]] == ["new question", "new answer"]
    child.api_call(ctx, updated, 3)
    child.end(engine, updated)
    assert len(child_rows(parent)) == len(rows)


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
def test_recovered_leading_whitespace_then_own_marker_keeps_content_with_one_duplicate(host, location):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    full = " \n" + FULL_OUTPUT
    marker = _write_output(_path(engine._hermes_home, location), full)
    # The host first delivers the recovered body inline (no recorded source metadata).
    # Storage's current preview parser rejects recovery of a leading-whitespace preview.
    child.api_call(ctx, _view(full), 1)
    first = child_rows(parent)[-1]
    assert engine._session_end_prefix_compare_value(first["content"], session_id="child") == full
    assert not engine._session_end_rows_equal(first, _view(marker)[-1], session_id="child")
    for n in (2, 3, 4):
        child.api_call(ctx, _view(marker), n)
        rows = child_rows(parent)
        assert len(rows) == 4
        assert rows[2]["content"] == first["content"]
        assert rows[-1]["content"] == marker
    child.end(engine, _view(marker))
    assert len(child_rows(parent)) == 4


@pytest.mark.parametrize("shape", ["marker-then-plain", "spill-then-temp"])
def test_shared_payload_compression_boundary_preserves_plain_rows_and_new_turn(host, monkeypatch, shape):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    handed = _shared_view(engine, shape)
    child.api_call(ctx, handed, 1)
    monkeypatch.setattr(engine, "_compress_bypassed", lambda messages, *args: messages)
    child.compress(engine, handed)
    _assert_once(parent, handed)
    updated = [*handed, {"role": "user", "content": "new question"},
               {"role": "assistant", "content": "new answer"}]
    child.api_call(ctx, updated, 2)
    child.compress(engine, updated)
    child.end(engine, updated)
    _assert_once(parent, updated)


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
@pytest.mark.parametrize("route", ["pairwise", "counted"])
def test_bare_marker_whose_file_appears_later_stores_the_inline_body(host, tmp_path, location, route):
    """Review of a1208c83: a stored bare marker (file missing at ingest) is never re-read from the host's file. When the
    file appears and the host hands the body inline, the body is stored. Red on a1208c83 (body dropped); main keeps it."""
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    path = _path(engine._hermes_home, location)
    marker = _marker(path)
    first = _view(marker) + ([{"role": "assistant", "content": "old tail"}] if route == "counted" else [])
    child.api_call(ctx, first, 1)
    assert child_rows(parent)[2]["content"] == marker
    _write_output(path)
    view = (_view(FULL_OUTPUT) if route == "pairwise" else [_view(FULL_OUTPUT)[-1]]) + [
        {"role": "assistant", "content": "FINAL"}]
    child.api_call(ctx, view, 2)
    child.end(engine, view)
    path.rename(tmp_path / "pruned-source")  # the body must be durable, not read back from the host file
    restored = [engine._session_end_prefix_compare_value(r["content"], session_id="child") for r in child_rows(parent)]
    assert restored.count(FULL_OUTPUT) == 1 and restored.count("FINAL") == 1


def test_anchor_handover_needs_no_full_store_read(host, monkeypatch):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    handed = [*_view(_write_output(_path(engine._hermes_home, "spillover"))),
              {"role": "assistant", "content": "stored tail"}]
    child.api_call(ctx, handed, 1)
    # Drop the prefix; the last stored plain row is now the entire anchor,
    # with no preceding rows to account for. Main does no full-store read.
    updated = [handed[-1], {"role": "assistant", "content": "after anchor"}]
    reads, anchors = [], []
    read = engine._store.get_session_messages
    anchor = engine._delegate_child_anchor_end
    def counted_read(*args, **kwargs):
        reads.append(args)
        return read(*args, **kwargs)
    def observe_anchor(*args, **kwargs):
        end = anchor(*args, **kwargs)
        anchors.append(end)
        return end
    monkeypatch.setattr(engine._store, "get_session_messages", counted_read)
    monkeypatch.setattr(engine, "_delegate_child_anchor_end", observe_anchor)
    child.api_call(ctx, updated, 2)
    assert anchors == [1], "handover must actually resolve by the stored-tail anchor"
    assert len(reads) == 0, f"main requires zero get_session_messages calls; got {len(reads)}"
    assert len(read("child")) == len(handed) + 1
