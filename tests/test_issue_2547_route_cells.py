"""#2547: outcome cells at externalization off (release-manager ruling, 11-10).

A session holding a Hermes saved-output marker, ended off-current (engine off-current prefix route), and a delegated
child handed the marker three times and then ended (the child live hook and child end). The promise is the stored rows:
the final reply is stored once and the host's rows are held exactly once.
"""

import copy
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_issue_10_persisted_output_spillover import FULL_OUTPUT, _write_output
from tests.test_issue_1931_delegate_children import child_rows, host  # noqa: F401
from tests.test_issue_2482_interrupted_child import _setup, call

CALL = "m1"


@pytest.fixture(autouse=True)
def isolated_tempdir(tmp_path, monkeypatch):
    directory = tmp_path / "host-temp"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))


def _path(home, location, name=CALL):
    root = Path(home) / "cache" / "spillover" if location == "spillover" else Path(tempfile.gettempdir()) / "hermes-results"
    return root / f"{name}.txt"


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
def test_an_off_current_end_at_ext_off_stores_its_final_reply(tmp_path, location):
    home = tmp_path / "hermes"
    engine = LCMEngine(config=LCMConfig(database_path=str(home / "lcm.db"), large_output_externalization_enabled=False,
                                        large_output_externalization_path=str(home / "externalized")),
                       hermes_home=str(home))
    try:
        marker = _write_output(_path(home, location), FULL_OUTPUT)
        texts = [("user", "A"), ("assistant", "B"), ("user", "C")]
        history = [{"role": r, "content": t, "timestamp": float(i + 1)} for i, (r, t) in enumerate(texts)]
        history.append({"role": "assistant", "content": "", "timestamp": 4.0, "tool_calls": [
            {"id": CALL, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]})
        history.append({"role": "tool", "tool_call_id": CALL, "content": marker, "timestamp": 5.0})
        history += [{"role": r, "content": t, "timestamp": float(6 + i)} for i, (r, t) in
                    enumerate((("assistant", "D"), ("user", "E"), ("assistant", "F"), ("user", "G"), ("assistant", "H")))]
        engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
        engine.ingest(copy.deepcopy(history))
        engine.on_session_start("S2", platform="cli", context_length=200_000, conversation_id="conv2")
        engine.on_session_end("S1", copy.deepcopy(history) + [{"role": "assistant", "content": "FINAL", "timestamp": 11.0}])
        rows = engine._store.get_session_messages("S1")
        assert (len(rows), sum(r["content"] == "FINAL" for r in rows)) == (len(history) + 1, 1)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
def test_a_delegated_child_at_ext_off_stores_its_saved_result_once(host, monkeypatch, location):  # noqa: F811
    monkeypatch.setenv("LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED", "false")
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    for e in {id(x): x for x in (ctx.offered, parent, engine)}.values():
        e._config.large_output_externalization_enabled = False
    handed = [{"role": "user", "content": "Dump the ledger."}, call(1),
              {"role": "tool", "tool_call_id": "c1", "content": _write_output(_path(engine._hermes_home, location, "c1"))}]
    for n in range(3):
        child.api_call(ctx, handed, n + 1)
        rows = child_rows(parent)
        assert (len(rows), Counter(r.get("tool_call_id") for r in rows if r["role"] == "tool")["c1"]) == (3, 1)
    final = [*handed, {"role": "assistant", "content": "FINAL"}]
    child.end(engine, final)
    rows = child_rows(parent)
    assert (len(rows), sum(r["content"] == "FINAL" for r in rows)) == (4, 1)


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
def test_an_ended_bypass_lineage_session_at_ext_off_keeps_its_stored_prefix_route(tmp_path, location):
    """The bypass-lineage end check (engine `ended_lineage_bypasses`): a stored prefix holding the marker matches, so
    the end is not skipped and the final reply is stored once."""
    home = tmp_path / "hermes"
    engine = LCMEngine(config=LCMConfig(database_path=str(home / "lcm.db"), large_output_externalization_enabled=False,
                                        large_output_externalization_path=str(home / "externalized")),
                       hermes_home=str(home))
    try:
        marker = _write_output(_path(home, location), FULL_OUTPUT)
        history = [{"role": "user", "content": "A", "timestamp": 1.0},
                   {"role": "assistant", "content": "", "timestamp": 2.0, "tool_calls": [
                       {"id": CALL, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
                   {"role": "tool", "tool_call_id": CALL, "content": marker, "timestamp": 3.0},
                   {"role": "assistant", "content": "B", "timestamp": 4.0}]
        engine.on_session_start("bypassed", platform="cli", context_length=200_000, conversation_id="conv-b")
        engine.ingest(copy.deepcopy(history))
        engine.on_session_start("parent", platform="cli", context_length=200_000, conversation_id="conv-p")
        engine._mark_lcm_bypass_lineage_session("bypassed", platform="subagent")
        engine.on_session_end("bypassed", copy.deepcopy(history) + [{"role": "assistant", "content": "FINAL",
                                                                       "timestamp": 5.0}])
        rows = engine._store.get_session_messages("bypassed")
        assert (len(rows), sum(r["content"] == "FINAL" for r in rows)) == (len(history) + 1, 1)
    finally:
        engine.shutdown()


def _failing_payload_write(monkeypatch):
    from hermes_lcm import externalize

    def fails(*_a, **_k):
        raise OSError("injected payload write failure")
    monkeypatch.setattr(externalize, "_write_externalized_payload", fails)


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
def test_a_failed_payload_write_at_default_settings_keeps_the_final_reply(tmp_path, monkeypatch, location):
    """Default settings, payload file write fails: ingest keeps the marker inline with LCM-X's generation line (the
    #2547 stored form without the identity line). An off-current end still stores the final reply once."""
    home = tmp_path / "hermes"
    engine = LCMEngine(config=LCMConfig(database_path=str(home / "lcm.db"),
                                        large_output_externalization_path=str(home / "externalized")),
                       hermes_home=str(home))
    try:
        _failing_payload_write(monkeypatch)
        marker = _write_output(_path(home, location), FULL_OUTPUT)
        texts = [("user", "A"), ("assistant", "B"), ("user", "C")]
        history = [{"role": r, "content": t, "timestamp": float(i + 1)} for i, (r, t) in enumerate(texts)]
        history.append({"role": "assistant", "content": "", "timestamp": 4.0, "tool_calls": [
            {"id": CALL, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]})
        history.append({"role": "tool", "tool_call_id": CALL, "content": marker, "timestamp": 5.0})
        history += [{"role": r, "content": t, "timestamp": float(6 + i)} for i, (r, t) in
                    enumerate((("assistant", "D"), ("user", "E"), ("assistant", "F"), ("user", "G"), ("assistant", "H")))]
        engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
        engine.ingest(copy.deepcopy(history))
        stored = [r for r in engine._store.get_session_messages("S1") if r["role"] == "tool"][0]["content"]
        if not (stored.startswith("<persisted-output>") and "[LCM persisted-output file generation:" in stored):
            pytest.fail("setup: the failed write did not leave the inline marker with a generation line")
        engine.on_session_start("S2", platform="cli", context_length=200_000, conversation_id="conv2")
        engine.on_session_end("S1", copy.deepcopy(history) + [{"role": "assistant", "content": "FINAL", "timestamp": 11.0}])
        rows = engine._store.get_session_messages("S1")
        assert (len(rows), sum(r["content"] == "FINAL" for r in rows)) == (len(history) + 1, 1)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
def test_a_failed_payload_write_at_default_settings_adds_no_child_copy(host, monkeypatch, location):  # noqa: F811
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    _failing_payload_write(monkeypatch)
    handed = [{"role": "user", "content": "Dump the ledger."}, call(1),
              {"role": "tool", "tool_call_id": "c1", "content": _write_output(_path(engine._hermes_home, location, "c1"))}]
    for n in range(3):
        child.api_call(ctx, handed, n + 1)
        rows = child_rows(parent)
        assert (len(rows), Counter(r.get("tool_call_id") for r in rows if r["role"] == "tool")["c1"]) == (3, 1)
    child.end(engine, [*handed, {"role": "assistant", "content": "FINAL"}])
    rows = child_rows(parent)
    assert (len(rows), sum(r["content"] == "FINAL" for r in rows)) == (4, 1)


@pytest.mark.parametrize("location", ["spillover", "hermes-results"])
def test_a_delegate_child_turn_store_at_ext_off_stores_each_row_once(tmp_path, monkeypatch, location):
    """The delegated child's turn store (aux_session `_append_delegate_child_rows`): a re-sent turn holding the
    marker adds nothing, and the final reply is stored once."""
    from hermes_lcm import aux_session
    from tests.test_issue_1931_delegate_children import DELEGATE, Agent
    monkeypatch.setattr(aux_session, "_DELEGATE_AUXILIARY_IDS", set())
    home = tmp_path / "home"
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "store.db"), embeddings_enabled=False,
                                        large_output_externalization_enabled=False), hermes_home=str(home))
    try:
        engine.on_session_start("parent", platform="desktop")
        class Child(Agent):  # the store finds its delegate agent on the call stack, as in the host
            def store(self, engine, history):
                return engine._store_delegate_child_turn(self.session_id, copy.deepcopy(history), platform="subagent")

        child = Child("child", **DELEGATE)
        child.start(engine)
        marker = _write_output(_path(home, location, "c1"), FULL_OUTPUT)
        turn = [{"role": "user", "content": "Dump the ledger."}, call(1),
                {"role": "tool", "tool_call_id": "c1", "content": marker}]
        engine.ingest(copy.deepcopy(turn))
        assert child.store(engine, turn) == 3
        assert child.store(engine, turn) == 0
        final = turn + [{"role": "assistant", "content": "FINAL"}]
        assert child.store(engine, final) == 1
        rows = engine._store.get_session_messages("child")
        assert (len(rows), sum(r["content"] == "FINAL" for r in rows)) == (4, 1)
    finally:
        engine.shutdown()
