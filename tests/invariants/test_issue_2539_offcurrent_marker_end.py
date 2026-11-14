"""Probe (#2539 class audit): engine-offcurrent-end-T1 and engine-coldtail-s3-T1.

A stored tool row recovered from a host saved-output marker; a late off-current session end (and its busy cold-tail
retry) re-sends the history with the marker plus a final reply. Claimed LOSS: the final reply is never stored.
Control: the same history with a plain tool result.
"""
import copy
import tempfile
from pathlib import Path

import pytest

from tests.test_issue_2055_ingest_ledger_writes import engines  # noqa: F401
from tests.test_issue_10_persisted_output_spillover import FULL_OUTPUT, _marker, _write_output
from tests.test_issue_1931_delegate_children import host as host
from tests.test_issue_2482_interrupted_child import _setup as setup_child
from tests.test_issue_2539_child_marker_compare import _path, _shared_view
from tests.invariants.test_p1e_cold_tails import fail_end_read
from hermes_lcm.session_patterns import compile_session_patterns

CALL = "m1"


def _tool_content(kind, root):
    if kind in ("spillover", "pruned"):
        return _write_output(root / "cache" / "spillover" / f"{CALL}.txt")
    if kind == "tempdir":
        return _write_output(Path(tempfile.gettempdir()) / "hermes-results" / f"{CALL}.txt")
    if kind == "nofile":
        return _marker(root / "cache" / "spillover" / f"{CALL}.txt")
    if kind == "nofile-tempdir":
        return _marker(Path(tempfile.gettempdir()) / "hermes-results" / f"{CALL}.txt")
    return "plain tool result: 17 rows"


def _history(tool_content):
    rows = [("user", "A"), ("assistant", "B"), ("user", "C")]
    msgs = [{"role": r, "content": t, "timestamp": float(i + 1)} for i, (r, t) in enumerate(rows)]
    msgs.append({"role": "assistant", "content": "", "timestamp": 4.0, "tool_calls": [
        {"id": CALL, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]})
    msgs.append({"role": "tool", "tool_call_id": CALL, "content": tool_content, "timestamp": 5.0})
    for i, (r, t) in enumerate((("assistant", "D"), ("user", "E"), ("assistant", "F"), ("user", "G"),
                                ("assistant", "H"))):
        msgs.append({"role": r, "content": t, "timestamp": float(6 + i)})
    return msgs


def _setup(make_engines, monkeypatch, tmp_path, kind):
    temp_root = tmp_path / "tmpdir"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))
    engine = make_engines(sessions=[("S1", None, None), ("S2", None, None)])
    if kind.endswith("-noext"):
        # Externalization disabled: a recovered marker is stored inline with no recorded marker metadata,
        # so the compare takes the recovered-content path.
        kind = kind[:-len("-noext")]
        engine._config.large_output_externalization_enabled = False
    root = Path(engine._hermes_home)
    history = _history(_tool_content(kind, root))
    engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
    engine.ingest(copy.deepcopy(history))
    stored = engine._store.get_session_messages("S1")
    assert len(stored) == len(history), [r["role"] for r in stored]
    tool = [r for r in stored if r["role"] == "tool"][0]
    form = "bare" if (tool["content"] or "").startswith("<persisted-output>") else (
        "externalized" if (tool["content"] or "").startswith("[Externalized") else "other")
    if kind == "pruned":
        for f in list((root / "cache" / "spillover").glob("*.txt")):
            f.rename(root / ("pruned-" + f.name))
    engine.on_session_start("S2", platform="cli", context_length=200_000, conversation_id="conv2")
    view = copy.deepcopy(history) + [{"role": "assistant", "content": "FINAL", "timestamp": 11.0}]
    return engine, view, form, len(history)


_NOEXT_REASON = ("#2547: class audit row transforms-T1-inline-marker-metadata (externalization disabled, not a default route): "
                 "storage keeps the marker inline with LCM identity and generation lines, so neither side yields the "
                 "marker key; pre-existing on main 7e67daf1; out of this PR by release-manager ruling")
_NOEXT = [pytest.param(kind, marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=_NOEXT_REASON))
          for kind in ("spillover-noext", "tempdir-noext")]


def _final_count(engine):
    return sum(1 for r in engine._store.get_session_messages("S1") if r["content"] == "FINAL")


@pytest.mark.parametrize("kind", ["spillover", "tempdir", "nofile", "nofile-tempdir", "plain", "pruned", *_NOEXT])
def test_offcurrent_end_stores_final_reply(engines, monkeypatch, tmp_path, kind):  # noqa: F811
    engine, view, form, n = _setup(engines, monkeypatch, tmp_path, kind)
    engine.on_session_end("S1", view)
    rows = engine._store.get_session_messages("S1")
    assert (_final_count(engine), len(rows)) == (1, n + 1), f"form={form} rows={len(rows)}"


@pytest.mark.parametrize("kind", ["spillover", "tempdir", "nofile", "nofile-tempdir", "plain", "pruned", *_NOEXT])
def test_offcurrent_busy_cold_tail_retry_stores_final_reply(engines, monkeypatch, tmp_path, kind):  # noqa: F811
    engine, view, form, n = _setup(engines, monkeypatch, tmp_path, kind)
    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "get_range", fail_end_read(engine))
        engine.on_session_end("S1", view)
    assert len(engine._pending_session_tails) == 1, "no cold tail queued"
    engine._retry_pending_session_tails()
    engine._retry_pending_session_tails()
    rows = engine._store.get_session_messages("S1")
    assert (_final_count(engine), len(rows), len(engine._pending_session_tails)) == (1, n + 1, 0), f"form={form}"


def _call(cid, n):
    return {"role": "assistant", "content": f"call {n}", "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": "dump", "arguments": "{}"}}]}


def _reused_first(marker):
    pad = [{"role": ("user", "assistant")[i % 2], "content": f"pad {i}"} for i in range(7)]
    return pad + [{"role": "user", "content": "go"}, _call("c1", 1),
                  {"role": "tool", "tool_call_id": "c1", "content": marker}]


def _reused_view(first, marker):
    return first + [{"role": "assistant", "content": "again"}, _call("c1", 2),
                    {"role": "tool", "tool_call_id": "c1", "content": marker},
                    {"role": "assistant", "content": "FINAL"}]


def _assert_reused_once(engine, sid, view):
    rows = engine._store.get_session_messages(sid)
    assert len(rows) == len(view)
    assert all(engine._session_end_rows_equal(r, m, session_id=sid) for r, m in zip(rows, view))
    assert rows[-1]["content"] == "FINAL"
    tools = [r for r in rows if r["role"] == "tool"]
    assert len(tools) == 2
    assert [engine._session_end_prefix_compare_value(r["content"], session_id=sid) for r in tools] == [
        FULL_OUTPUT, FULL_OUTPUT[:-10] + "Z" * 10]


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
def test_reused_call_id_two_occurrences_offcurrent(engines, monkeypatch, tmp_path, location):  # noqa: F811
    engine, _, _, _ = _setup(engines, monkeypatch, tmp_path, "plain")
    engine.on_session_start("old", platform="cli", context_length=200_000, conversation_id="old-conv")
    path = _path(engine._hermes_home, location)
    marker = _write_output(path)
    first = _reused_first(marker)
    engine.ingest(first)
    assert len(engine._store.get_session_messages("old")) == 10
    engine.on_session_start("foreground", platform="cli", context_length=200_000)
    assert _write_output(path, FULL_OUTPUT[:-10] + "Z" * 10) == marker
    view = _reused_view(first, marker)
    engine.on_session_end("old", view)
    _assert_reused_once(engine, "old", view)
    engine.on_session_end("old", view)
    _assert_reused_once(engine, "old", view)


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
def test_reused_call_id_two_occurrences_child_end(host, tmp_path, monkeypatch, location):
    temp_root = tmp_path / "host-temp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))
    _, _, ctx, parent, child, engine = setup_child(host, "own_engine")
    path = _path(engine._hermes_home, location)
    marker = _write_output(path)
    first = _reused_first(marker)
    child.api_call(ctx, first, 1)
    assert len(parent._store.get_session_messages("child")) == 10
    assert _write_output(path, FULL_OUTPUT[:-10] + "Z" * 10) == marker
    view = _reused_view(first, marker)
    child.end(engine, view)
    _assert_reused_once(parent, "child", view)
    child.end(engine, view)
    _assert_reused_once(parent, "child", view)


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
@pytest.mark.parametrize("route", ["direct-bypass", "same-id-bypass"])
def test_class_A_bypass_decision_with_marker(engines, monkeypatch, tmp_path, location, route):  # noqa: F811
    kind = "spillover" if location == "spillover" else "tempdir"
    engine, view, _, n = _setup(engines, monkeypatch, tmp_path, kind)
    if route == "direct-bypass":
        # Pin the audit's non-default, suppressed auxiliary id reused as normal.
        engine._auxiliary_foreground_reused_session_ids.add("S1")
        engine._compiled_ignore_session_patterns = compile_session_patterns(["S1"])
        assert engine._ended_session_directly_bypasses_lcm("S1")
        assert engine._thread_context_session_id() != "S1"
    else:
        engine._compiled_ignore_session_patterns = compile_session_patterns(["cron"])
        engine.on_session_start("S1", platform="cron", context_length=200_000)
        engine.ingest(view[:-1])
        engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
        assert engine._has_lcm_bypass_lineage_session("S1")
        assert not engine._bypasses_lcm_context_management()
        # The caller under test (the same-id bypass-lineage prefix count): the stored marker row matches its host
        # row, so the durable prefix covers every stored row. Red on aa1f63cd, where it stops at the marker row.
        assert engine._session_end_store_prefix_count("S1", view, conversation_id=engine._conversation_id) == n
        engine.on_session_end("S1", view)
        rows = engine._store.get_session_messages("S1")
        # No stored row is stored again. This route does not store FINAL for a plain tool row either (the same
        # end with kind "plain" also keeps n rows); that is the route's behaviour, tracked with #2543, not the marker's.
        assert len(rows) == n and sum(r["role"] == "tool" for r in rows) == 1
        return
    engine.on_session_end("S1", view)
    rows = engine._store.get_session_messages("S1")
    assert len(rows) == n + 1
    assert _final_count(engine) == 1
    assert sum(r["role"] == "tool" for r in rows) == 1


_ROUTE_2543 = ("#2543: pre-existing on v0.29.2; the off-current route skips the batch on any prefix mismatch "
               "(a host edit of a plain text row loses the same way)")


def _changed_marker_view(engine, view, location):
    path = (Path(engine._hermes_home) / "cache" / "spillover" / f"{CALL}.txt" if location == "spillover"
            else Path(tempfile.gettempdir()) / "hermes-results" / f"{CALL}.txt")
    changed = FULL_OUTPUT + "NEW RESULT"
    old_marker = view[4]["content"]
    view[4]["content"] = _write_output(path, changed)
    assert view[4]["content"] != old_marker
    return changed


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
def test_rule_b_changed_marker_text_compares_unequal(engines, monkeypatch, tmp_path, location):  # noqa: F811
    """Companion to the #2543 xfails: rule B itself says a different host marker text is a different row."""
    kind = "spillover" if location == "spillover" else "tempdir"
    engine, view, _, _n = _setup(engines, monkeypatch, tmp_path, kind)
    stored = engine._store.get_session_messages("S1")[4]
    assert engine._session_end_rows_equal(stored, view[4], session_id="S1")
    _changed_marker_view(engine, view, location)
    assert not engine._session_end_rows_equal(stored, view[4], session_id="S1")


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
def test_offcurrent_busy_cold_tail_changed_marker_text_keeps_new_result_and_final(
        engines, monkeypatch, tmp_path, location):  # noqa: F811
    kind = "spillover" if location == "spillover" else "tempdir"
    engine, view, _, n = _setup(engines, monkeypatch, tmp_path, kind)
    _changed_marker_view(engine, view, location)
    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "get_range", fail_end_read(engine))
        engine.on_session_end("S1", view)
    engine._retry_pending_session_tails()
    engine._retry_pending_session_tails()
    rows = engine._store.get_session_messages("S1")
    assert sum(r["role"] == "tool" for r in rows) == 2 and _final_count(engine) == 1 and len(rows) == n + 2


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
def test_offcurrent_changed_marker_text_keeps_new_result_and_final(engines, monkeypatch, tmp_path, location):  # noqa: F811
    kind = "spillover" if location == "spillover" else "tempdir"
    engine, view, _, n = _setup(engines, monkeypatch, tmp_path, kind)
    changed = _changed_marker_view(engine, view, location)
    engine.on_session_end("S1", view)
    rows = engine._store.get_session_messages("S1")
    tools = [r for r in rows if r["role"] == "tool"]
    assert len(tools) == 2, "different host marker text was skipped"
    assert engine._session_end_prefix_compare_value(tools[0]["content"], session_id="S1") == FULL_OUTPUT
    assert engine._session_end_prefix_compare_value(tools[1]["content"], session_id="S1") == changed
    assert _final_count(engine) == 1
    assert len(rows) == n + 2


@pytest.mark.parametrize("shape", ["marker-then-plain", "spill-then-temp"])
@pytest.mark.parametrize("busy", [False, True], ids=["offcurrent", "cold-tail"])
def test_shared_payload_offcurrent_end_preserves_every_row_and_final(engines, monkeypatch, tmp_path, shape, busy):  # noqa: F811
    root = tmp_path / "shared-temp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    engine = engines(sessions=[("S1", None, None), ("foreground", None, None)])
    handed = [{"role": ("user", "assistant")[n % 2], "content": f"pad {n}"} for n in range(6)]
    handed += _shared_view(engine, shape)
    engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="old-conv")
    engine.ingest(handed)
    engine.on_session_start("foreground", platform="cli", context_length=200_000)
    updated = [*handed, {"role": "user", "content": "new question"},
               {"role": "assistant", "content": "FINAL"}]
    if busy:
        with monkeypatch.context() as patch:
            patch.setattr(engine._store, "get_range", fail_end_read(engine))
            engine.on_session_end("S1", updated)
        assert len(engine._pending_session_tails) == 1
        engine._retry_pending_session_tails()
        engine._retry_pending_session_tails()
        assert not engine._pending_session_tails
    else:
        engine.on_session_end("S1", updated)
    rows = engine._store.get_session_messages("S1")
    assert len(rows) == len(updated)
    assert all(engine._session_end_rows_equal(r, m, session_id="S1") for r, m in zip(rows, updated))
    assert [r["content"] for r in rows[-2:]] == ["new question", "FINAL"]
    engine.on_session_end("S1", updated)
    assert len(engine._store.get_session_messages("S1")) == len(updated)



_DATA_URI = "data:image/png;base64," + "QUJD" * 100  # 400 base64 characters: ingest swaps it for a placeholder


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
@pytest.mark.parametrize("transform", ["data-uri-default-config", "redaction"])
def test_unchanged_bare_marker_transformed_at_ingest_keeps_final(engines, monkeypatch, tmp_path, location, transform):  # noqa: F811
    """Review of a1208c83: a bare marker (file missing) whose stored text an ingest transform changed must still
    equal the unchanged host marker. Red on a1208c83 (FINAL lost); main 7e67daf1 and 13211798 keep it."""
    temp_root = tmp_path / "tmpdir"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))
    engine = engines(sessions=[("S1", None, None), ("S2", None, None)])
    if transform == "redaction":
        engine._config.sensitive_patterns_enabled = True
        engine._config.sensitive_patterns = ["api_key"]
    head = (_DATA_URI if transform == "data-uri-default-config" else "api_key=synthetic2539secret") + "\n"
    folder = (Path(engine._hermes_home) / "cache" / "spillover" if location == "spillover"
              else Path(tempfile.gettempdir()) / "hermes-results")
    marker = _marker(folder / f"{CALL}.txt", head + FULL_OUTPUT, preview=(head + FULL_OUTPUT)[:600])
    history = _history(marker)
    engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
    engine.ingest(copy.deepcopy(history))
    stored = engine._store.get_session_messages("S1")
    assert len(stored) == len(history)
    assert stored[4]["content"] != marker  # ingest transformed the stored text
    engine.on_session_start("S2", platform="cli", context_length=200_000, conversation_id="conv2")
    engine.on_session_end("S1", copy.deepcopy(history) + [{"role": "assistant", "content": "FINAL", "timestamp": 11.0}])
    rows = engine._store.get_session_messages("S1")
    assert (_final_count(engine), len(rows)) == (1, len(history) + 1)
