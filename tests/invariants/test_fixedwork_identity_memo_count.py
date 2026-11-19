"""#2856 counting pin: live identity computations per ordinary turn on a warm engine (N=500, three turns).

Row shape copied from the #2856 probe (probe_turn.py; synthetic). Counts calls of the live identity body: the
uncached body where it exists, else (a tree without the memo) the identity method itself. Runs on both trees, so it
imports no memo symbol.
"""
import json
import os
import sqlite3
from copy import deepcopy
from pathlib import Path

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.reconcile import ReconcileMixin
from hermes_lcm.tokens import count_messages_tokens

WORD = " lorem ipsum dolor sit amet"
N, TURNS = 500, 3


class Rows:
    """Mixed agent-session shape: user, assistant(tool_calls), tool result (varied size), assistant reply."""

    def __init__(self):
        self.i = 0
        self.call = 0

    def _uid(self, row):
        if self.i % 10 < 7:  # 70% stamped with a host uid, 30% not
            row["message_uid"] = f"uid-{self.i}"
        self.i += 1
        return row

    def user(self, tag="r"):
        return self._uid({"role": "user", "content": f"[{tag}{self.i}] question" + WORD * 6,
                          "timestamp": 1_700_000_000.0 + self.i})

    def call_row(self, tag="r"):
        self.call += 1
        cid = f"call_{self.call}"
        args = json.dumps({"path": f"/work/file_{self.call}.py", "limit": 200})
        return cid, self._uid({"role": "assistant", "content": f"[{tag}{self.i}] checking",
                               "tool_calls": [{"id": cid, "type": "function",
                                               "function": {"name": "read_file", "arguments": args}}]})

    def tool(self, cid, tag="r"):
        size = (8, 60, 240)[self.call % 3]  # ~200 / ~1.6k / ~6.5k chars
        return self._uid({"role": "tool", "tool_call_id": cid, "name": "read_file",
                          "content": f"[{tag}{self.i}] result" + WORD * size})

    def reply(self, tag="r"):
        return self._uid({"role": "assistant", "content": f"[{tag}{self.i}] answer" + WORD * 12})

    def exchange(self, tag="r"):
        rows = [self.user(tag)]
        for _ in range(1 + self.call % 2):
            cid, c = self.call_row(tag)
            rows += [c, self.tool(cid, tag)]
        rows.append(self.reply(tag))
        return rows


def build(n, gen):
    rows = [{"role": "system", "content": "[r0x] system prompt" + WORD * 20}]
    while len(rows) < n:
        rows += gen.exchange()
    return rows[:n] if rows[n - 1].get("role") == "assistant" and not rows[n - 1].get("tool_calls") else rows


def state_db(path):
    conn = sqlite3.connect(path / "state.db")
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, "
                 "end_reason TEXT, source TEXT, model_config TEXT)")
    conn.execute("INSERT OR REPLACE INTO sessions VALUES ('S', NULL, NULL, NULL, NULL)")
    conn.commit()
    conn.close()


def one_turn(engine, history, gen):
    """The host's order on an ordinary turn (Hermes 0.23), every call with fresh dicts."""
    ex = gen.exchange("n")
    msgs = history + [ex[0]]
    tokens = count_messages_tokens(msgs)
    engine.should_compress(tokens)
    engine.should_compress_preflight(deepcopy(msgs))
    engine.update_from_response({"prompt_tokens": tokens, "completion_tokens": 50, "total_tokens": tokens + 50})
    full = msgs + ex[1:]
    engine.ingest(deepcopy(full))
    engine.note_turn_complete()
    engine.on_turn_complete(deepcopy(full), usage=None)
    return full, len(ex)


def test_a4_live_identity_computations_per_ordinary_turn(tmp_path, monkeypatch):
    name = ("_message_replay_identity_uncached" if hasattr(ReconcileMixin, "_message_replay_identity_uncached")
            else "_message_replay_identity")
    original, calls = getattr(ReconcileMixin, name), [0]

    def counted(self, msg, *args, **kwargs):
        calls[0] += not kwargs.get("stored_row")
        return original(self, msg, *args, **kwargs)
    monkeypatch.setattr(ReconcileMixin, name, counted)
    monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    state_db(tmp_path)
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), fresh_tail_count=64,
                                         embeddings_enabled=False, temporal_rollups_enabled=False),
                       hermes_home=str(tmp_path))
    engine.on_session_start("S", platform="telegram", context_length=50_000_000, conversation_id="conv")
    turns = []
    try:
        gen = Rows()
        history = build(N, gen)
        engine.should_compress_preflight(deepcopy(history))  # cold catch-up, excluded
        engine.ingest(deepcopy(history))
        engine.note_turn_complete()
        for _ in range(TURNS):
            calls[0] = 0
            history, new = one_turn(engine, history, gen)
            turns.append(dict(rows=len(history), new_rows=new, live_computations=calls[0]))
    finally:
        engine.shutdown()
    memo = name == "_message_replay_identity_uncached"
    out = os.environ.get("LCMX_FIXEDWORK_EVIDENCE")
    if out:
        Path(out, f"a4_counts_{'candidate' if memo else 'base'}.json").write_text(
            json.dumps(dict(tree="candidate" if memo else "base", counted=name, n=N, turns=turns), indent=2) + "\n")
    for turn in turns:
        bound = 4 * turn["new_rows"] + 16
        if memo:
            assert turn["live_computations"] <= bound, turns
        else:  # the pin discriminates: the tree without the memo computes about 4N per turn
            assert turn["live_computations"] > bound, turns
