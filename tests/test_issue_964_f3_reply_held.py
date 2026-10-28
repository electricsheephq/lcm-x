"""#964: the observer's turn_end note names a previous reply only when the host held the scripted reply."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.instruments.reliability import process_cell as PC
from bench.instruments.reliability.scorers import continuity as CT


def _observer(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("_f3_observer", PC.OBSERVER / "rel_observer.py")
    obs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obs)
    monkeypatch.setattr(obs, "DIR", tmp_path)
    monkeypatch.setattr(obs, "_r1", types.SimpleNamespace(provenance=lambda cell, rows: {}))
    monkeypatch.setattr(obs, "ensure_tap", lambda: None)
    monkeypatch.setattr(obs, "snapshot", lambda **extra: None)
    (tmp_path / "cell.json").write_text("{}")
    (tmp_path / "turn.json").write_text(json.dumps({"t": 9, "prefix": "T", "kind": "normal",
                                                    "reply": "reply to T09: completed fixture"}))
    return obs


@pytest.mark.parametrize("user,reply,expected", [
    ("[LCM survival fit: projected user row]", "ok", None),  # completed without the scripted reply
    ("[T09] user turn 9: fixture", "reply to T09: completed fixture", "T09"),
])
def test_turn_end_tags_only_a_reply_the_host_held(tmp_path, monkeypatch, user, reply, expected):
    obs = _observer(tmp_path, monkeypatch)

    class AIAgent:
        session_id = "s"

        def run_conversation(self, *a, **k):
            return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": reply}],
                    "completed": True}
    obs.patch_run_agent(types.SimpleNamespace(AIAgent=AIAgent))
    object.__new__(AIAgent).run_conversation("fixture")

    [end] = [n for n in PC.read_jsonl(tmp_path / "observer.jsonl") if n["kind"] == "turn_end"]
    assert end["reply_tag"] == expected
    compaction = {"kind": "compaction_committed", "ts": end["ts"] + 1, "phase": "A", "turn": 10,
                  "compaction_kind": "in-place", "compacted_reply_tags": []}
    messages = [{"role": "system", "content": CT.NOTE}, {"role": "user", "content": "[T09] user turn 9: fixture"},
                {"role": "assistant", "content": reply}, {"role": "user", "content": "[T10] user turn 10: fixture"}]
    request = {"rid": 1, "ts": end["ts"] + 2, "role": "main", "turn": 10, "current_user_tag": "T10",
               "continuity": CT.markers(messages, previous=expected, current="T10")}
    [row] = CT.score({}, [request], [end, compaction])["rows"]
    assert row["previous_reply_tag"] == expected
    assert row["F3"] is (None if expected is None else True)
