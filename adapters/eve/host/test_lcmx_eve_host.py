"""Tests for the LCM-X engine host (the real LCMEngine driven like Hermes). Run from the repo root:

    uv run --no-project --python 3.12.16 --with pytest==9.1.1 pytest -c adapters/eve/writer/pytest.ini adapters/eve/host
"""
from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "writer"))
import lcmx_eve_host as host_module  # noqa: E402
from test_lcmx_eve_writer import Agent, assistant, framework, tool, user  # noqa: E402

host_module.planner.bootstrap(None)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # deterministic: no auxiliary model in unit tests
    path = tmp_path / "scope"
    path.mkdir()
    return path


def new_host(home: Path, context_length: int = 200_000, summarize: bool = False) -> host_module.Host:
    return host_module.Host(home, context_length=context_length, model="test-model", summarize=summarize)


class FakeEveModel:
    """Stands in for the Eve process behind the bridge: answers LCM-X's summary contract."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, params: dict, timeout: float) -> dict:
        import re

        self.calls.append(params)
        system = next((m["content"] for m in params["messages"] if m["role"] == "system"), "")
        nonce = re.search(r'<lcm-summary nonce="([0-9a-f]+)">', system)
        body = ("The user stored numbered notes with item codes (ZX values) and the assistant confirmed each.\n"
                "Codes follow the pattern ZX-<7 x item>.\n"
                "Expand for details about: the individual item codes")
        content = f'<lcm-summary nonce="{nonce.group(1)}">\n{body}\n</lcm-summary>' if nonce else body
        return {"content": content, "finish_reason": "stop", "model": "fake-eve-model"}


@pytest.fixture
def eve_model(monkeypatch: pytest.MonkeyPatch) -> FakeEveModel:
    import agent.auxiliary_client as auxiliary

    model = FakeEveModel()
    monkeypatch.setattr(auxiliary, "BRIDGE", model)
    return model


def stored(home: Path, session: str = "s1") -> list[tuple[str, str]]:
    with closing(sqlite3.connect(home / "lcm.db")) as conn:
        return [(role, content) for role, content in conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY store_id", (f"eve:{session}",))]


def test_turns_go_through_the_engine_once(home: Path) -> None:
    agent = Agent(home)
    h1 = [user("hello"), assistant("hi")]
    agent.capture(h1)
    h2 = h1 + [user("list files"), assistant("", [("c1", "bash", {"cmd": "ls"})]), tool("c1", "bash", "a.txt"),
               assistant("one file")]
    agent.capture(h2)
    agent.capture(h2, op="retry", delta=False)  # Eve re-ran the handler
    host = new_host(home)
    try:
        report = host.drain()
    finally:
        host.close()
    assert report["failed"] == 0 and report["fed"] == 6
    assert [role for role, _ in stored(home)] == ["user", "assistant", "user", "assistant", "tool", "assistant"]
    result = host_module.audit(home)
    assert result["ok"], result


def test_eve_compaction_is_not_fed_and_tail_is_not_doubled(home: Path) -> None:
    agent = Agent(home)
    old = [user("task one"), assistant("done one"), user("task two"), assistant("done two")]
    agent.capture(old, phase="compaction.requested", turn="t2")
    agent.capture([framework("context.compaction", "Summary of our conversation so far:"),
                   {"role": "assistant", "content": "Eve's summary."}, user("task two"), assistant("done two"),
                   user("task three"), assistant("done three")], turn="t3", inputs=[user("task three")])
    host = new_host(home)
    try:
        host.drain()
    finally:
        host.close()
    contents = [content for _, content in stored(home)]
    assert contents == ["task one", "done one", "task two", "done two", "task three", "done three"]
    with closing(sqlite3.connect(home / host_module.LEDGER)) as conn:
        kept = conn.execute("SELECT COUNT(*) FROM events WHERE notes LIKE '%checkpoint%' AND raw IS NOT NULL").fetchone()
    assert kept == (1,)  # Eve's summary is kept raw in the ledger, not given to the engine
    assert host_module.audit(home)["ok"]


def test_restart_keeps_the_shadow_and_the_engine_dedupes_a_replay(home: Path) -> None:
    agent = Agent(home)
    h1 = [user("alpha"), assistant("beta")]
    agent.capture(h1)
    host = new_host(home)
    host.drain()
    host.close()
    agent.capture(h1 + [user("gamma"), assistant("delta")], turn="t2", inputs=[user("gamma")])
    host = new_host(home)  # a new Eve process: same store, shadow from the ledger
    try:
        host.drain()
        # A crash between engine ingest and ledger commit replays the record: same shadow again.
        shadow = host_module.load_json(host.ledger, "shadows", "eve:s1")
        host.engine.ingest(shadow)
    finally:
        host.close()
    assert [content for _, content in stored(home)] == ["alpha", "beta", "gamma", "delta"]
    assert host_module.audit(home)["ok"]


def notes(agent: Agent, count: int) -> list[dict]:
    history: list[dict] = []
    for index in range(count):
        history += [user(f"Note {index}: item {index} has code ZX-{index * 7}. " + "filler words " * 60),
                    assistant(f"Noted item {index}.")]
        agent.capture(list(history), turn=f"t{index + 1}", inputs=[history[-2]])
    return history


def test_engine_summarizes_through_the_eve_model_and_tools_answer(home: Path, eve_model: FakeEveModel) -> None:
    history = notes(Agent(home), 40)
    host = new_host(home, context_length=8_000, summarize=True)
    try:
        report = host.drain()
        assert report["compressed"] >= 1
        shadow = host_module.load_json(host.ledger, "shadows", "eve:s1")
        assert len(shadow) < len(history)  # the engine's own window, as a Hermes host would hold it
        names = {schema["name"] for schema in host.tool_schemas()}
        assert {"lcm_grep", "lcm_recall", "lcm_expand", "lcm_describe", "lcm_status", "lcm_doctor",
                "lcm_expand_query"} <= names
        found = json.loads(host.call_tool("s1", "lcm_grep", {"query": "ZX-21"}))
        assert found["total_results"] >= 1
        assert json.loads(host.call_tool("s1", "lcm_status", {}))
    finally:
        host.close()
    with closing(sqlite3.connect(home / "lcm.db")) as conn:
        nodes = conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0]
        written = conn.execute("SELECT summary FROM summary_nodes LIMIT 1").fetchone()[0]
    assert nodes >= 1 and "ZX" in written  # the Eve model's summary, accepted under the contract
    assert eve_model.calls and all(call["task"] == "compression" for call in eve_model.calls)
    # A separate call with LCM-X's prompt only: its policy and the transcript to summarize, no agent history.
    assert all([m["role"] for m in call["messages"]] == ["system", "user"] for call in eve_model.calls)
    result = host_module.audit(home)
    assert result["ok"], result
    assert result["store_rows"] == 80


def test_no_summaries_inside_a_turn_or_without_the_bridge(home: Path, eve_model: FakeEveModel) -> None:
    agent = Agent(home)
    history = notes(agent, 40)
    agent.capture(history + [user("one more")], phase="compaction.requested", turn="t41")
    host = new_host(home, context_length=8_000, summarize=False)  # a timer drain: no Eve process
    try:
        assert host.drain()["compressed"] == 0
    finally:
        host.close()
    assert eve_model.calls == []
    with closing(sqlite3.connect(home / "lcm.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == 0
    assert host_module.audit(home)["ok"]


def test_idle_time_summarizes_a_session_left_over_its_threshold(home: Path, eve_model: FakeEveModel) -> None:
    notes(Agent(home), 40)
    first = new_host(home, context_length=8_000, summarize=False)  # Eve ended: store only
    try:
        first.drain()
    finally:
        first.close()
    second = new_host(home, context_length=8_000, summarize=True)  # the next Eve process, idle
    try:
        second.turn_started_at = __import__("time").monotonic()  # an Eve turn runs: no summary now
        assert second.summarize_backlog() is None
        second.turn_started_at = None
        done = second.summarize_backlog()
        assert done and done["tokens"][1] < done["tokens"][0]
        assert second.summarize_backlog() is None  # nothing left over the threshold
    finally:
        second.close()
    with closing(sqlite3.connect(home / "lcm.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] >= 1
    assert host_module.audit(home)["ok"]


def test_call_llm_has_no_http_route_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent.auxiliary_client as auxiliary

    monkeypatch.setattr(auxiliary, "BRIDGE", None)
    monkeypatch.delenv("LCMX_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")  # an embeddings key must never reach chat
    with pytest.raises(auxiliary.AuxiliaryLLMError, match="no LLM route"):
        auxiliary.call_llm(messages=[{"role": "user", "content": "hi"}], task="compression")


def test_bridge_routes_replies_by_id() -> None:
    sent: list[dict] = []
    bridge = host_module.Bridge(sent.append)
    import threading

    def answer() -> None:
        while not sent:
            pass
        assert not bridge.deliver({"jsonrpc": "2.0", "id": 7, "result": {}})  # a numeric id is not ours
        bridge.deliver({"jsonrpc": "2.0", "id": sent[0]["id"], "result": {"content": "ok"}})

    threading.Thread(target=answer).start()
    assert bridge.complete({"task": "t"}, timeout=5)["content"] == "ok"
    assert sent[0]["method"] == "llm/complete"


def test_closed_bridge_fails_waiting_and_new_calls_at_once() -> None:
    import threading
    import time

    bridge = host_module.Bridge(lambda message: None)  # nobody answers: Eve is gone
    outcome: dict = {}

    def wait() -> None:
        started = time.monotonic()
        try:
            bridge.complete({"task": "t"}, timeout=60)
        except RuntimeError as error:
            outcome.update(error=str(error), seconds=time.monotonic() - started)

    worker = threading.Thread(target=wait)
    worker.start()
    time.sleep(0.2)
    bridge.close()
    worker.join(5)
    assert "ended" in outcome["error"] and outcome["seconds"] < 5
    with pytest.raises(RuntimeError, match="ended"):
        bridge.complete({"task": "t"}, timeout=60)


def test_embed_catchup_is_quiet_without_an_api_provider(home: Path) -> None:
    host = new_host(home)
    try:
        assert host.embed_catchup() == {"embedding": "local_or_off"}
    finally:
        host.close()


def test_serve_lock_is_exclusive(home: Path) -> None:
    first = host_module.Lock(home)
    second = host_module.Lock(home)
    try:
        assert first.try_acquire()
        assert not second.try_acquire()
        assert host_module.drain_once(home, context_length=200_000, model="m") == {"lock": "held"}
    finally:
        first.close()
        second.close()
