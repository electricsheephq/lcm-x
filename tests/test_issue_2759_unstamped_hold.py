"""A held call without a host stamp still reserves its stored occurrence (I1/I2/I8/I9)."""

import base64
import copy
import json

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _open(tmp_path, monkeypatch, uid_mode):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LCM_DATABASE_PATH", str(tmp_path / "lcm.db"))
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "0")
    monkeypatch.setenv("LCM_BACKGROUND_LEAF_PREP_ENABLED", "0")
    for name in ("LCM_IDENTITY_ANCHOR", "LCM_INGEST_LEDGER", "LCM_HOST_MESSAGE_UID"):
        monkeypatch.delenv(name, raising=False)
    if uid_mode is not None:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", uid_mode)
    engine = LCMEngine(config=LCMConfig.from_env(), hermes_home=str(tmp_path / "home"))
    monkeypatch.setattr(engine_module, "summarize_with_escalation",
                        lambda **_kwargs: ("Earlier turns.", 1))
    engine.on_session_start("parent", platform="cli", context_length=128000, conversation_id="conv")
    return engine


def _encoded(size):
    return base64.b64encode(bytes((i * 37 + 11) % 256 for i in range(size))).decode("ascii")


def _args(shape):
    if shape == "wrapped-base64":
        body = _encoded(3420)
        return {"path": "a.b64",
                "content": "hdr\n" + "\n".join(body[i:i + 76] for i in range(0, len(body), 76)) + "\nend"}
    return {"data:image\\/png;base64," + _encoded(1200): "v", "b": "x"}


def _view(shape):
    rows = [
        {"role": "user", "content": "parent-question-0", "timestamp": 1.0},
        {"role": "assistant", "content": "parent-reply-0", "timestamp": 2.0},
        {"role": "assistant", "content": "", "timestamp": 3.0, "tool_calls": [
            {"id": "call-0", "type": "function",
             "function": {"name": "write_file", "arguments": json.dumps(_args(shape))}}]},
        {"role": "tool", "content": "parent-result-0", "tool_call_id": "call-0", "timestamp": 4.0},
    ]
    for i, row in enumerate(rows):
        row["message_uid"] = f"parent-0-{i}"
    return rows


@pytest.mark.parametrize("shape", ["wrapped-base64", "data-uri-key"])
@pytest.mark.parametrize("uid", ["missing", "distinct"])
@pytest.mark.parametrize("uid_mode", [None, "off"])
def test_unstamped_hold_reserves_stored_call(tmp_path, monkeypatch, shape, uid, uid_mode):
    engine = _open(tmp_path, monkeypatch, uid_mode)
    try:
        view = _view(shape)
        engine.ingest(copy.deepcopy(view))
        call = copy.deepcopy(view[2])
        call.pop("message_uid")
        if uid == "distinct":
            call["message_uid"] = "second-call"
        view[2].pop("timestamp")
        engine.ingest(copy.deepcopy(view + [call]))
        rows = engine._store.get_session_messages("parent")
        assert len(rows) == 5, rows
        assert sum(bool(r.get("tool_calls")) for r in rows) == 2
        assert [r["observed_at"] for r in rows] == [1., 2., 3., 4., 3.]
    finally:
        engine.shutdown()
