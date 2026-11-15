"""#2759: a receipt cannot reuse a durable copy spent by the after-cursor scan."""

import base64
import copy
import json
from collections import Counter

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _bind(engine):
    engine.on_session_start("parent", platform="cli", context_length=128000, conversation_id="conv")


def _open(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LCM_DATABASE_PATH", str(tmp_path / "lcm.db"))
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "0")
    monkeypatch.setenv("LCM_BACKGROUND_LEAF_PREP_ENABLED", "0")
    for name in ("LCM_IDENTITY_ANCHOR", "LCM_INGEST_LEDGER", "LCM_HOST_MESSAGE_UID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **_kwargs: ("Earlier turns.", 1))
    engine = LCMEngine(config=LCMConfig.from_env(), hermes_home=str(tmp_path / "home"))
    _bind(engine)
    return engine


def _row(role, content, stamp, uid):
    return dict(role=role, content=content, timestamp=stamp, message_uid=uid)


def _call(cid, stamp, uid, key=None):
    encoded = base64.b64encode(bytes((i * 37 + 11) % 256 for i in range(1200))).decode("ascii")
    arguments = {key + encoded: "v", "b": "x"} if key is not None else {
        "path": "a.txt", "content": "plain body"}
    row = {"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": "write_file", "arguments": json.dumps(arguments)}}]}
    if stamp is not None:
        row["timestamp"] = stamp
    if uid is not None:
        row["message_uid"] = uid
    return row


def _result(cid, content, stamp, uid):
    return dict(_row("tool", content, stamp, uid), tool_call_id=cid)


def _counts(engine):
    counts = Counter()
    for row in engine._store.get_session_messages("parent"):
        calls = row.get("tool_calls")
        if calls:
            calls = json.loads(calls) if isinstance(calls, str) else calls
            counts[("call", tuple(call["id"] for call in calls))] += 1
        elif row["role"] == "tool":
            counts[("tool", row["tool_call_id"], row["content"])] += 1
        else:
            counts[(row["role"], row["content"])] += 1
    return counts


@pytest.mark.parametrize("variant", ["j-then-i-nouid", "j11-then-i-nouid", "j11u-then-i-nouid"])
def test_aftercursor_copy_cannot_also_back_receipt(tmp_path, monkeypatch, variant):
    seen = []
    original = LCMEngine._replayed_tool_segment_indexes_after_cursor

    def observe(self, messages, cursor, **kwargs):
        replayed = original(self, messages, cursor, **kwargs)
        seen.append(dict(cursor=cursor, replayed=sorted(replayed), backing=dict(kwargs.get("backing") or {})))
        return replayed

    monkeypatch.setattr(LCMEngine, "_replayed_tool_segment_indexes_after_cursor", observe)
    engine = _open(tmp_path, monkeypatch)
    try:
        u1, a1 = _row("user", "q1", 1.0, "u1"), _row("assistant", "a1", 2.0, "a1")
        a, ra = _call("cA", 3.0, "A"), _result("cA", "resA", 4.0, "rA")
        x, rx = _call("cX", 5.0, "X", "data:image\\/png;base64,"), _result("cX", "resX", 6.0, "rX")
        p, rp = _call("cP", 7.0, "P"), _result("cP", "resP", 8.0, "rP")
        engine.ingest(copy.deepcopy([u1, a1, a, ra, x, rx, p, rp, _row("assistant", "a2", 9.0, "a2")]))
        stored_x = next(row for row in engine._store.get_session_messages("parent") if row.get("tool_calls")
                        and "cX" in str(row["tool_calls"]))
        j = _call("cX", None, None, "data:image/png;base64,")
        rj = {"role": "tool", "content": "resX", "tool_call_id": "cX"}
        assert "Externalized LCM ingest payload:" in str(stored_x["tool_calls"])
        assert engine._message_replay_identity(stored_x, stored_row=True) == engine._message_replay_identity(j)
        engine.shutdown()
        engine = LCMEngine(config=engine._config, hermes_home=engine._hermes_home)
        _bind(engine)
        i, ri = copy.deepcopy(x), copy.deepcopy(rx)
        i.pop("message_uid")
        ri.pop("message_uid")
        if variant.startswith("j11"):
            j["timestamp"], rj["timestamp"] = 11.0, 12.0
        if variant.startswith("j11u"):
            j["message_uid"], rj["message_uid"] = "J", "rJ"
        view = [a, ra, _row("assistant", "NEW", 4.5, "new"), j, rj, p, rp, i, ri,
                _row("user", "q2", 10.0, "u2")]
        engine.ingest(copy.deepcopy(view))
        engine.ingest(copy.deepcopy(view))
        engine.shutdown()
        engine = LCMEngine(config=engine._config, hermes_home=engine._hermes_home)
        _bind(engine)
        engine.ingest(copy.deepcopy(view))
        actual = _counts(engine)
        engine.shutdown()
        clean_dir = tmp_path / "clean"
        clean_dir.mkdir()
        engine = _open(clean_dir, monkeypatch)
        engine.ingest(copy.deepcopy(view))
        clean = _counts(engine)
        print("AFTERCURSOR", json.dumps(dict(variant=variant, scans=seen, X=actual[("call", ("cX",))],
                                            cleanX=clean[("call", ("cX",))], rX=actual[("tool", "cX", "resX")],
                                            cleanrX=clean[("tool", "cX", "resX")])))
        call, result = ("call", ("cX",)), ("tool", "cX", "resX")
        # No loss: at least a clean ingest's copies survive (with the double claim the store kept 1 of 2).
        assert actual[call] >= clean[call] and actual[result] >= clean[result]
        # Known limit, equal to main (main keeps 3 here too): a later restart re-appends the escaped twin,
        # because the receipts cover only the first stored copy. Update this pin when that is fixed.
        assert (actual[call], actual[result]) == (3, 3)
        assert actual[("assistant", "NEW")] == 1
        assert actual[("user", "q2")] == 1
    finally:
        engine.shutdown()
