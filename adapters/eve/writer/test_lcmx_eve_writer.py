"""Tests for the Eve capture writer: run with Python 3.11+ and pytest from the repo root:

    uv run --python 3.12.16 --with pytest==9.1.1 pytest -c adapters/eve/writer/pytest.ini adapters/eve/writer

Histories here have Eve's shapes (AI SDK messages; framework user messages carry ``kind``). Digests
are made here the way the TypeScript provider makes them (sha256 over canonical JSON); the writer only
compares digests, so any stable digest works.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lcmx_eve_writer as writer  # noqa: E402

writer.bootstrap(None)


def digest(message: dict) -> str:
    return hashlib.sha256(json.dumps(message, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def framework(kind: str, text: str) -> dict:
    return {"role": "user", "content": text, "kind": kind}


def assistant(text: str, calls: list[tuple[str, str, dict]] = ()) -> dict:
    parts = [{"type": "text", "text": text}] if text else []
    parts += [{"type": "tool-call", "toolCallId": ident, "toolName": name, "input": args} for ident, name, args in calls]
    return {"role": "assistant", "content": parts}


def tool(ident: str, name: str, value: str) -> dict:
    return {"role": "tool", "content": [{"type": "tool-result", "toolCallId": ident, "toolName": name,
                                         "output": {"type": "text", "value": value}}]}


class Agent:
    """Writes spool records the way the provider does: full snapshot, or a delta on what it sent."""

    def __init__(self, home: Path, session: str = "s1"):
        self.home = home
        self.session = session
        self.sent: list[str] = []
        self.counter = 0

    def capture(self, history: list[dict], *, phase: str = "turn.completed", turn: str = "t1",
                delta: bool = True, op: str | None = None, inputs: list[dict] = ()) -> Path:
        digests = [digest(message) for message in history]
        base = 0
        if delta and self.sent and digests[:len(self.sent)] == self.sent:
            base = len(self.sent)
        record = {"api": writer.API, "op": op or f"op-{self.counter}", "phase": phase,
                  "session": {"id": self.session}, "turn": {"id": turn, "sequence": int(turn.lstrip("t")), "input": [digest(m) for m in inputs]},
                  "at": "2026-11-09T12:00:00.000Z", "scope": {"key": "k", "slot": "lcmx"},
                  "base": {"count": base, "chain": writer.chain(digests[:base])}, "total": len(history),
                  "messages": [{"d": d, "m": m} for d, m in zip(digests[base:], history[base:])]}
        folder = self.home / "spool"
        folder.mkdir(parents=True, exist_ok=True)
        self.counter += 1
        path = folder / f"{1_000_000_000_000 + self.counter:013d}-1-{self.counter:06d}.json"
        path.write_text(json.dumps(record))
        self.sent = digests
        return path


def rows(home: Path, session: str = "s1") -> list[tuple[str, str]]:
    with closing(sqlite3.connect(home / "lcm.db")) as conn:
        return [(role, content) for role, content in conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY store_id", (f"eve:{session}",))]


def drain(home: Path) -> dict:
    report = writer.drain(home, embed=False)
    assert report["failed"] == 0, report
    return report


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "scope-a"


def test_turns_are_stored_once_with_tool_rows(home: Path) -> None:
    agent = Agent(home)
    h1 = [user("hello"), assistant("hi")]
    agent.capture(h1)
    h2 = h1 + [user("list files"), assistant("", [("c1", "bash", {"cmd": "ls"})]), tool("c1", "bash", "a.txt"),
               assistant("one file")]
    agent.capture(h2)
    drain(home)
    assert rows(home) == [("user", "hello"), ("assistant", "hi"), ("user", "list files"), ("assistant", ""),
                          ("tool", "a.txt"), ("assistant", "one file")]
    report = writer.audit(home)
    assert report["ok"], report
    assert report["positions"] == 6


def test_retry_and_replay_add_nothing(home: Path) -> None:
    agent = Agent(home)
    h1 = [user("hello"), assistant("hi")]
    agent.capture(h1, op="same")
    agent.capture(h1, op="same", delta=False)  # Eve re-ran the handler with the same operationId
    drain(home)
    agent.capture(h1[:1], delta=False)  # a stale, shorter snapshot
    drain(home)
    assert rows(home) == [("user", "hello"), ("assistant", "hi")]
    assert writer.audit(home)["ok"]


def test_identical_messages_in_one_generation_are_kept_apart(home: Path) -> None:
    agent = Agent(home)
    agent.capture([user("ok"), assistant("done"), user("ok"), assistant("done")])
    drain(home)
    assert [role for role, _ in rows(home)] == ["user", "assistant", "user", "assistant"]


def test_compaction_keeps_tail_once_and_stores_checkpoint(home: Path) -> None:
    agent = Agent(home)
    old = [user("task one"), assistant("", [("c1", "read", {"p": "x"})]), tool("c1", "read", "X" * 50),
           assistant("read it"), user("task two"), assistant("", [("c2", "read", {"p": "y"})]),
           tool("c2", "read", "Y" * 50), assistant("done two")]
    agent.capture(old, phase="compaction.requested", turn="t2")
    capped = tool("c2", "read", "Y" * 10 + "[capped]")
    new = [framework("context.compaction", "Summary of our conversation so far:"),
           {"role": "assistant", "content": "We did task one and two."},
           assistant("", [("c2", "read", {"p": "y"})]), capped, assistant("done two"),
           user("task three"), assistant("done three")]
    agent.capture(new, turn="t3")
    drain(home)
    stored = rows(home)
    assert len(stored) == len(old) + 4  # marker, summary, task three, done three
    assert stored[len(old)][0] == "system" and stored[len(old)][1].startswith("[eve context.compaction]")
    assert stored[len(old) + 1] == ("system", "[eve compaction checkpoint]\nWe did task one and two.")
    assert stored[-2:] == [("user", "task three"), ("assistant", "done three")]
    report = writer.audit(home)
    assert report["ok"], report
    with closing(sqlite3.connect(home / "lcm.db")) as conn:
        kinds = dict(conn.execute("SELECT disposition, COUNT(*) FROM capture_events WHERE generation = 1 "
                                  "GROUP BY disposition").fetchall())
    assert kinds == {"new": 4, "carried": 2, "carried_modified": 1}


def test_compaction_that_drops_tool_messages_and_replays_the_user_message(home: Path) -> None:
    agent = Agent(home)
    old = [user("first"), assistant("ok first"), user("fix the bug"),
           assistant("looking", [("c1", "grep", {"q": "bug"})]), tool("c1", "grep", "line 3")]
    agent.capture(old, phase="compaction.requested", turn="t2")
    new = [framework("context.compaction", "Summary of our conversation so far:"),
           {"role": "assistant", "content": "Summary."},
           {"role": "assistant", "content": "looking"},  # keepNonToolResultMessages: text only, tool gone
           user("fix the bug"),  # the turn's user message, replayed
           assistant("fixed")]
    agent.capture(new, turn="t2")
    drain(home)
    assert rows(home)[len(old):] == [("system", "[eve context.compaction]\nSummary of our conversation so far:"),
                                     ("system", "[eve compaction checkpoint]\nSummary."), ("assistant", "fixed")]
    assert writer.audit(home)["ok"]


def test_same_text_after_a_compaction_in_a_new_turn_is_new(home: Path) -> None:
    agent = Agent(home)
    old = [user("ok"), assistant("done")]
    agent.capture(old, phase="compaction.requested", turn="t1")
    new = [framework("context.compaction", "Summary of our conversation so far:"),
           {"role": "assistant", "content": "Summary."}, user("ok"), assistant("done again")]
    agent.capture(new, turn="t2")
    drain(home)
    assert rows(home)[-2:] == [("user", "ok"), ("assistant", "done again")]


def test_clear_starts_over_without_loss(home: Path) -> None:
    agent = Agent(home)
    agent.capture([user("a"), assistant("b")])
    agent.capture([user("a"), assistant("b"), user("z")], turn="t2", inputs=[user("z")])
    # history cleared, then the first question and its answer again, word for word
    agent.capture([user("a"), assistant("b")], turn="t3", inputs=[user("a")])
    drain(home)
    assert rows(home) == [("user", "a"), ("assistant", "b"), ("user", "z"), ("user", "a"), ("assistant", "b")]
    assert writer.audit(home)["ok"]


def test_base_mismatch_keeps_messages_and_fails_audit(home: Path) -> None:
    agent = Agent(home)
    agent.capture([user("a"), assistant("b")])
    drain(home)
    lost = agent.capture([user("a"), assistant("b"), user("c"), assistant("d")])
    lost.unlink()  # a record that never reached the writer
    agent.capture([user("a"), assistant("b"), user("c"), assistant("d"), user("e"), assistant("f")])
    drain(home)
    assert rows(home)[-2:] == [("user", "e"), ("assistant", "f")]
    report = writer.audit(home)
    assert report["base_mismatch"] == 2 and not report["ok"]


def test_bad_record_is_set_aside(home: Path) -> None:
    (home / "spool").mkdir(parents=True)
    (home / "spool" / "0000000000001-1-000001.json").write_text("{not json")
    Agent(home).capture([user("x")])
    report = writer.drain(home, embed=False)
    assert report["failed"] == 1 and report["drained"] == 1
    assert (home / "spool" / "0000000000001-1-000001.failed").exists()
    assert rows(home) == [("user", "x")]


def test_store_is_stamped_for_eve_and_refuses_claude_code(home: Path) -> None:
    Agent(home).capture([user("x")])
    drain(home)
    owner = json.loads((home / "capture-owner.json").read_text())
    assert owner["host"] == "eve"
    cc = writer._cc()
    with pytest.raises(cc.AdmissionError):
        cc.admit(home)  # the Claude Code unit never writes an Eve store


def test_projection_of_parts() -> None:
    message = {"role": "assistant", "content": [
        {"type": "reasoning", "text": "secret thoughts"}, {"type": "text", "text": "answer"},
        {"type": "tool-call", "toolCallId": "c9", "toolName": "web", "input": {"q": "x"}}]}
    [row] = writer.project(message)
    assert row["content"] == "answer" and row["tool_calls"][0]["function"]["name"] == "web"
    result = {"role": "tool", "content": [
        {"type": "tool-result", "toolCallId": "c9", "toolName": "web", "output": {"type": "json", "value": {"a": 1}}},
        {"type": "tool-result", "toolCallId": "c8", "toolName": "web", "output": {"type": "execution-denied"}}]}
    assert [row["content"] for row in writer.project(result)] == ['{"a": 1}', "[execution denied]"]
    picture = {"role": "user", "content": [{"type": "text", "text": "see"},
                                           {"type": "image", "image": {"$bytes": "AAA"}, "mediaType": "image/png"}]}
    assert writer.project(picture) == [{"role": "user", "content": "see\n\n[image image/png]"}]


def test_compaction_replays_the_turn_message_with_its_middle_cut(home: Path) -> None:
    """Seen with gpt-6.1-terra: Eve compacted inside a turn, kept no tail and re-sent the turn's large user
    message with its middle cut out. The cut copy is the same message, not a new one."""
    agent = Agent(home)
    report = "Field report 2.\n" + "\n".join(f"line {n}: reading {n * 37 % 101}" for n in range(400))
    old = [user("first"), assistant("ok first"), user(report)]
    agent.capture(old, phase="compaction.requested", turn="t2")
    head, tail = report[:1500], report[-1200:]
    cut = (head + f"\n\n[eve: {len(report) - 2700} characters from the middle of this message were cut during "
           "context compaction to fit the context window.]\n\n" + tail)
    new = [framework("context.compaction", "Summary of our conversation so far:"),
           {"role": "assistant", "content": "Summary."}, user(cut), assistant("Field report 2 acknowledged.")]
    agent.capture(new, turn="t2")
    drain(home)
    assert [content[:15] for _, content in rows(home)].count("Field report 2.") == 1
    assert writer.audit(home)["ok"]


def test_a_cut_note_on_a_different_message_is_new(home: Path) -> None:
    agent = Agent(home)
    agent.capture([user("first"), assistant("ok"), user("alpha " * 300)], phase="compaction.requested", turn="t2")
    other = "beta " * 50 + "\n\n[eve: 900 characters from the middle of this message were cut during context " \
            "compaction to fit the context window.]\n\n" + "gamma " * 50
    agent.capture([framework("context.compaction", "Summary of our conversation so far:"),
                   {"role": "assistant", "content": "Summary."}, user(other), assistant("done")], turn="t2")
    drain(home)
    assert any(content.startswith("beta") for _, content in rows(home))
