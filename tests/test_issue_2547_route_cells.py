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
