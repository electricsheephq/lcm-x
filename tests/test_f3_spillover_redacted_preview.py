"""A spillover-folder saved output whose preview redaction changes is stored as the marker the host sent.

With sensitive-pattern redaction on and externalization on, ingest recovered a Hermes saved-output file and externalized
its redacted text. When redaction changed the part the marker's preview shows, the stored text no longer started with
the host's (unredacted) preview, so the session-end rule never matched the host's marker: an off-current end lost the
rows after it, usually the final reply. v0.29.2 stores the marker for this folder, and keeps them. The fix stores the
marker as the host sent it (redacted like any stored text) for a spillover-folder file in that case. Without redaction,
or when the preview is unchanged by it, nothing changes. The temp-folder kind is not changed (both versions lose there).

Self-contained: helpers come only from test modules present on main and on release/v0.30.x.
"""

import copy
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_issue_10_persisted_output_spillover import _marker, _write_output
from tests.test_issue_1931_delegate_children import child_rows, host  # noqa: F401
from tests.test_issue_2482_interrupted_child import _setup as setup_child, call

CALL = "m1"
# Redactable bodies: the preview (the first 30 characters) shows part of the redacted span, except "later".
BODIES = {
    "digest": "api_key=synthetic1234567890abcdef0123456789\n" + "x" * 3000,
    "lossy": "password=synthetic1234\n" + "x" * 3000,
    "cut": "y" * 20 + "api_key=synthetic1234567890abcdef0123456789\n" + "x" * 3000,
    "later": "z" * 100 + "\napi_key=synthetic1234567890abcdef0123456789\n" + "x" * 3000,
}
REDACTION = pytest.mark.parametrize("body", ["digest", "lossy", "cut"])


@pytest.fixture(autouse=True)
def isolated_tempdir(tmp_path, monkeypatch):
    root = tmp_path / "tmpdir"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))


def _engine(home, *, redact=True):
    config = LCMConfig(database_path=str(home / "lcm.db"), large_output_externalization_path=str(home / "externalized"))
    config.sensitive_patterns_enabled = redact
    config.sensitive_patterns = ["all"] if redact else []
    return LCMEngine(config=config, hermes_home=str(home))


def _path(kind, home, call=CALL):
    if kind == "spillover":
        return Path(home) / "cache" / "spillover" / f"{call}.txt"
    return Path(tempfile.gettempdir()) / "hermes-results" / f"{call}.txt"


def _history(marker):
    rows = [("user", "A"), ("assistant", "B"), ("user", "C")]
    msgs = [{"role": r, "content": t, "timestamp": float(i + 1)} for i, (r, t) in enumerate(rows)]
    msgs.append({"role": "assistant", "content": "", "timestamp": 4.0, "tool_calls": [
        {"id": CALL, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]})
    msgs.append({"role": "tool", "tool_call_id": CALL, "content": marker, "timestamp": 5.0})
    msgs += [{"role": r, "content": t, "timestamp": float(6 + i)} for i, (r, t) in
             enumerate((("assistant", "D"), ("user", "E"), ("assistant", "F"), ("user", "G"), ("assistant", "H")))]
    return msgs


def _form(content):
    content = content or ""
    if content.startswith("[Externalized"):
        return "externalized"
    if content.startswith("<persisted-output>"):
        return "marker" if "[LCM persisted-output" not in content else "marker-with-lines"
    return "other"


def _offcurrent_end(home, kind, body, *, redact=True):
    engine = _engine(home, redact=redact)
    try:
        marker = _write_output(_path(kind, home), BODIES[body])
        history = _history(marker)
        engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
        engine.ingest(copy.deepcopy(history))
        form = _form([r for r in engine._store.get_session_messages("S1") if r["role"] == "tool"][0]["content"])
        engine.on_session_start("S2", platform="cli", context_length=200_000, conversation_id="conv2")
        engine.on_session_end("S1", copy.deepcopy(history) + [{"role": "assistant", "content": "FINAL", "timestamp": 11.0}])
        rows = engine._store.get_session_messages("S1")
        return form, sum(r["content"] == "FINAL" for r in rows), len(rows), len(history)
    finally:
        engine.shutdown()


@REDACTION
def test_a_spillover_marker_whose_preview_redaction_changes_keeps_the_final_reply(tmp_path, body):
    form, final, rows, n = _offcurrent_end(tmp_path / "hermes", "spillover", body)
    assert (form, final, rows) == ("marker", 1, n + 1)  # v0.29.2: the marker, 1, n + 1


def test_redaction_that_leaves_the_preview_unchanged_keeps_the_recovered_form(tmp_path):
    form, final, rows, n = _offcurrent_end(tmp_path / "hermes", "spillover", "later")
    assert (form, final, rows) == ("externalized", 1, n + 1)


@pytest.mark.parametrize("kind", ["spillover", "tempdir"])
def test_without_redaction_nothing_changes(tmp_path, kind):
    form, final, rows, n = _offcurrent_end(tmp_path / "hermes", kind, "digest", redact=False)
    assert (form, final, rows) == ("externalized", 1, n + 1)


_TEMPDIR_REASON = ("known in v0.30.0: a temp-folder saved output whose preview redaction changes loses the rows after "
                   "it at an off-current end; v0.29.2 loses them too (with or without redaction)")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_TEMPDIR_REASON)
@REDACTION
def test_a_temp_folder_marker_whose_preview_redaction_changes_keeps_the_final_reply(tmp_path, body):
    form, final, rows, n = _offcurrent_end(tmp_path / "hermes", "tempdir", body)
    assert form == "externalized"  # not changed by this fix
    assert (final, rows) == (1, n + 1)


@REDACTION
def test_a_delegated_child_with_a_redacted_preview_stores_its_result_once(host, body):  # noqa: F811
    _, _, ctx, parent, child, engine = setup_child(host, "own_engine")
    engine._config.sensitive_patterns_enabled = True
    engine._config.sensitive_patterns = ["all"]
    marker = _write_output(_path("spillover", engine._hermes_home, "c1"), BODIES[body])
    handed = [{"role": "user", "content": "Dump the ledger."}, call(1),
              {"role": "tool", "tool_call_id": "c1", "content": marker}]
    copies = []
    for n in range(3):
        child.api_call(ctx, handed, n + 1)
        rows = child_rows(parent)
        copies.append(Counter(r.get("tool_call_id") for r in rows if r["role"] == "tool")["c1"])
    child.end(engine, [*handed, {"role": "assistant", "content": "FINAL"}])
    rows = child_rows(parent)
    copies.append(Counter(r.get("tool_call_id") for r in rows if r["role"] == "tool")["c1"])
    assert (copies, len(rows), sum(r["content"] == "FINAL" for r in rows)) == ([1, 1, 1, 1], 4, 1)


@REDACTION
def test_a_restart_on_a_store_that_holds_the_marker_ends_off_current_with_the_final_reply(tmp_path, body):
    """The stored row is the redacted marker (what v0.29.2 stores for this folder, and what the fix stores); a new
    engine binds the old session first, starts another, and the old session ends with its rows and a final reply."""
    home = tmp_path / "hermes"
    marker = _write_output(_path("spillover", home), BODIES[body])
    history = _history(marker)
    first = _engine(home)
    try:
        first.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
        first.ingest(copy.deepcopy(history))
        assert _form([r for r in first._store.get_session_messages("S1") if r["role"] == "tool"][0]["content"]) == "marker"
    finally:
        first.shutdown()
    second = _engine(home)
    try:
        second.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
        second.on_session_start("S2", platform="cli", context_length=200_000, conversation_id="conv2")
        second.on_session_end("S1", copy.deepcopy(history) + [{"role": "assistant", "content": "FINAL", "timestamp": 11.0}])
        rows = second._store.get_session_messages("S1")
        assert (sum(r["content"] == "FINAL" for r in rows), len(rows)) == (1, len(history) + 1)
    finally:
        second.shutdown()


# Three cold replays with redaction on (each ingest a new engine on the same store, two more rows each time; "partial"
# re-sends from the call on with the identical marker). Stored rows after the ingest and each replay, pinned as measured
# on this change (with and without #2584's keep they are equal), beside rc5's and v0.29.2's. No handed row is lost in
# any cell. Stampless replays of a digest-redacted spillover marker store the replay again, as v0.29.2 does (rc5 stored
# the externalized form there and did not).
_T0 = 1762700000.0
_RCALL = "call_spill"
_REPLAY_PINS = {  # (kind, body, stamps, shape): (this change, rc5, v0.29.2)
    ("K1", "cut", "all", "full"): ((4, 6, 8, 10), (4, 6, 8, 10), (4, 6, 8, 10)),
    ("K1", "cut", "all", "partial"): ((4, 9, 11, 13), (4, 9, 11, 13), (4, 9, 11, 13)),
    ("K1", "cut", "none", "full"): ((4, 6, 8, 10), (4, 6, 8, 10), (4, 10, 18, 28)),
    ("K1", "cut", "none", "partial"): ((4, 9, 11, 13), (4, 9, 11, 13), (4, 12, 22, 34)),
    ("K1", "digest", "all", "full"): ((4, 6, 8, 10), (4, 6, 8, 10), (4, 6, 8, 10)),
    ("K1", "digest", "all", "partial"): ((4, 9, 11, 13), (4, 9, 11, 13), (4, 9, 11, 13)),
    ("K1", "digest", "none", "full"): ((4, 10, 18, 28), (4, 6, 8, 10), (4, 10, 18, 28)),
    ("K1", "digest", "none", "partial"): ((4, 12, 22, 34), (4, 9, 11, 13), (4, 12, 22, 34)),
    ("K1", "lossy", "all", "full"): ((4, 6, 8, 10), (4, 6, 8, 10), (4, 8, 12, 16)),
    ("K1", "lossy", "all", "partial"): ((4, 11, 13, 19), (4, 11, 13, 19), (4, 11, 17, 23)),
    ("K1", "lossy", "none", "full"): ((4, 10, 18, 28), (4, 10, 18, 28), (4, 10, 18, 28)),
    ("K1", "lossy", "none", "partial"): ((4, 12, 22, 34), (4, 12, 22, 34), (4, 12, 22, 34)),
    ("K2", "cut", "all", "full"): ((4, 6, 8, 10), (4, 6, 8, 10), (4, 6, 8, 10)),
    ("K2", "cut", "all", "partial"): ((4, 9, 11, 13), (4, 9, 11, 13), (4, 9, 11, 13)),
    ("K2", "cut", "none", "full"): ((4, 6, 8, 10), (4, 6, 8, 10), (4, 6, 8, 10)),
    ("K2", "cut", "none", "partial"): ((4, 9, 11, 13), (4, 9, 11, 13), (4, 9, 11, 13)),
    ("K2", "digest", "all", "full"): ((4, 6, 8, 10), (4, 6, 8, 10), (4, 6, 8, 10)),
    ("K2", "digest", "all", "partial"): ((4, 9, 11, 13), (4, 9, 11, 13), (4, 9, 11, 13)),
    ("K2", "digest", "none", "full"): ((4, 6, 8, 10), (4, 6, 8, 10), (4, 6, 8, 10)),
    ("K2", "digest", "none", "partial"): ((4, 9, 11, 13), (4, 9, 11, 13), (4, 9, 11, 13)),
    ("K2", "lossy", "all", "full"): ((4, 6, 8, 10), (4, 6, 8, 10), (4, 8, 12, 16)),
    ("K2", "lossy", "all", "partial"): ((4, 11, 13, 19), (4, 11, 13, 19), (4, 11, 17, 23)),
    ("K2", "lossy", "none", "full"): ((4, 10, 18, 28), (4, 10, 18, 28), (4, 10, 18, 28)),
    ("K2", "lossy", "none", "partial"): ((4, 12, 22, 34), (4, 12, 22, 34), (4, 12, 22, 34)),
}


def _replay_transcript(tool, shape):
    def row(i, role, content, **extra):
        return dict(role=role, content=content, timestamp=_T0 + i, **extra)

    call_row = row(2, "assistant", "", tool_calls=[{"id": _RCALL, "type": "function",
                                                    "function": {"name": "dump", "arguments": "{}"}}])
    t = [row(1, "user", "dump it"), call_row, row(3, "tool", tool, tool_call_id=_RCALL), row(4, "assistant", "saved")]
    if shape == "full":
        t += [row(5 + i, ("user", "assistant")[i % 2], f"next {i}") for i in range(6)]
        return t, [(0, 4)] + [(0, 4 + 2 * k) for k in range(1, 4)]
    t += [row(5 + i, ("user", "assistant")[i % 2], f"next {i}") for i in range(2)]
    t += [dict(call_row), row(10, "tool", tool, tool_call_id=_RCALL), row(11, "assistant", "saved again")]
    t += [row(10 + i, ("user", "assistant")[i % 2], f"next {i}") for i in range(2, 6)]
    return t, [(0, 4)] + [(1, 7 + 2 * k) for k in range(1, 4)]


@pytest.mark.parametrize("kind,body,stamps,shape", sorted(_REPLAY_PINS), ids=["-".join(c) for c in sorted(_REPLAY_PINS)])
def test_three_cold_replays_with_redaction_on(tmp_path, kind, body, stamps, shape):
    home = tmp_path / "hermes"
    home.mkdir()
    path = (home / "cache" / "spillover" if kind == "K1" else Path(tempfile.gettempdir()) / "hermes-results") / f"{_RCALL}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(BODIES[body].encode("utf-8"))
    t, slices = _replay_transcript(_marker(path, BODIES[body]), shape)
    counts = []
    for n, (a, b) in enumerate(slices):
        handed = copy.deepcopy(t[a:b])
        if n:
            for row in handed:
                if stamps == "none":
                    row.pop("timestamp", None)
        engine = LCMEngine(config=LCMConfig(database_path=str(home / "lcm.db"), sensitive_patterns_enabled=True,
                                            sensitive_patterns=["all"]), hermes_home=str(home))
        try:
            engine.on_session_start("S1", platform="cli", context_length=128000, conversation_id="conv1")
            engine.ingest(handed)
            counts.append(len(engine._store.get_session_messages("S1")))
        finally:
            engine.shutdown()
    this_change, rc5, v0292 = _REPLAY_PINS[(kind, body, stamps, shape)]
    assert tuple(counts) == this_change, f"got {counts}; pinned {this_change}; rc5 {rc5}; v0.29.2 {v0292}"
