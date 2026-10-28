"""Subprocess payload for the real-host long-session regression.

Lane labels I1-I5 map to catalogue I1; I3/I4/I8/I9; I2; I14; I1,
respectively. Only the provider is synthetic; host dispatch, compression,
publication, durable resume and retrieval use their real implementations.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import sys
import time
from typing import Any

WINDOW = 32768
TAIL = 8
OBJECTIVE = "Objective: preserve the observatory inventory and its numbered records."
ROOT = Path(__file__).resolve().parents[1]


def user_text(turn: int) -> str:
    rng = random.Random(8347 + turn)
    words = [rng.choice(("amber", "birch", "cedar", "delta", "elm", "fjord")) for _ in range(400)]
    fact = f" Planted fact ORBIT{turn:04d} is inventory item {turn}." if turn % 6 == 0 else ""
    return f"[T{turn:03d}] user {OBJECTIVE if turn == 1 else 'Inventory update.'}{fact} " + " ".join(words)


def canonical(message: dict[str, Any]) -> str:
    """Stable wire identity; host stamps, clocks and scratch-home paths are not payload."""
    fields = {k: message.get(k) for k in ("role", "content", "tool_calls", "tool_call_id")}
    calls = fields["tool_calls"]
    if isinstance(calls, str):
        fields["tool_calls"] = json.loads(calls)
    fields["tool_calls"] = [{k: c[k] for k in ("id", "type", "function")} for c in (fields["tool_calls"] or [])] or None
    # Hermes serializes tool arguments compactly at the provider boundary.
    # Compare their JSON values, retaining exact tool IDs, names and content.
    for call in fields["tool_calls"] or []:
        function = call["function"]
        call["function"] = {**function, "arguments": json.loads(function["arguments"])}
    fields["tool_call_id"] = fields["tool_call_id"] or None
    fields["content"] = fields["content"] or ""
    return json.dumps(fields, sort_keys=True).replace(str(Path.home()), "<HOME>")


def assert_ordered_tail(messages: list[dict[str, Any]], protected: list[str],
                        carrier_remainder: str | None = None) -> None:
    """Match a suffix exactly; only its first row may carry a verified summary."""
    emitted = [canonical(m) for m in messages[-len(protected):]]
    if emitted and emitted != protected and carrier_remainder is not None:
        first = json.loads(emitted[0])
        if first["role"] == "user":
            emitted[0] = canonical({**first, "content": carrier_remainder})
    assert emitted == protected, "I2 assembly: ordered fresh tail changed"


def assert_restored_history(messages: list[dict[str, Any]], terminal: list[str]) -> None:
    assert [canonical(m) for m in messages] == terminal, "I5 continuity: restored history changed"


def paired(messages: list[dict[str, Any]]) -> None:
    pending: set[str] = set()
    for message in messages:
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            assert call_id in pending, f"I2 assembly: orphan tool result {call_id}"
            pending.remove(call_id)
        else:
            assert not pending, f"I2 assembly: unanswered tool calls {sorted(pending)}"
            pending = {c["id"] for c in message.get("tool_calls") or []}
    assert not pending, f"I2 assembly: unanswered tool calls {sorted(pending)}"


def stored(home: Path, engine: Any) -> list[dict[str, Any]]:
    with sqlite3.connect(home / "lcm.db") as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute("SELECT * FROM messages ORDER BY store_id")]
    tools = importlib.import_module(type(engine).__module__.rsplit(".", 1)[0] + ".tools")
    for row in rows:
        match = re.search(r'lcm_expand\(externalized_ref="([^"]+)"\)', row.get("content") or "")
        if match:
            expanded = json.loads(tools.lcm_expand({"externalized_ref": match[1], "max_tokens": 10000}, engine=engine))
            assert not expanded.get("error") and not expanded.get("has_more"), "I3 recall: externalized payload unavailable"
            row["content"] = expanded["content"]
    return rows


def digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(f"{r['store_id']}:{canonical(r)}" for r in rows).encode()).hexdigest()


def host_order(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ledger in host order: parallel tool results finish in any order, but the
    host appends them in the order of the assistant's tool calls."""
    ordered: list[dict[str, Any]] = []
    position: dict[str, int] = {}
    for message in ledger:
        if message.get("role") != "tool":
            position = {c["id"]: i for i, c in enumerate(message.get("tool_calls") or [])}
            ordered.append(message)
            continue
        group_start = len(ordered)
        while group_start and ordered[group_start - 1].get("role") == "tool":
            group_start -= 1
        group = ordered[group_start:] + [message]
        ordered[group_start:] = sorted(group, key=lambda m: position.get(m.get("tool_call_id"), len(position)))
    return ordered


def checkpoint(engine: Any, home: Path, turn: int, history: list[dict[str, Any]],
               ledger: list[dict[str, Any]]) -> dict[str, Any]:
    rows = stored(home, engine)
    got, want = Counter(map(canonical, rows)), Counter(map(canonical, ledger))
    assert got == want, (f"I1 exactly-once at turn {turn}: stored={len(rows)} expected={len(ledger)} "
                         f"missing={str(list((want - got).elements())[:1])[:250]} extra={str(list((got - want).elements())[:1])[:250]}")
    # The multiset check above cannot see reordering; store order must follow host order.
    got_order, want_order = [canonical(r) for r in rows], [canonical(m) for m in host_order(ledger)]
    if got_order != want_order:
        first = next(i for i, (a, b) in enumerate(zip(got_order, want_order)) if a != b)
        raise AssertionError(f"I1 chronology at turn {turn}: store order diverges from host order "
                             f"at position {first}: stored={got_order[first][:250]} expected={want_order[first][:250]}")
    paired(history)
    tools = importlib.import_module(type(engine).__module__.rsplit(".", 1)[0] + ".tools")
    recalled = 0
    for planted in range(6, turn + 1, 6):
        text = user_text(planted)
        if any(text in str(m.get("content") or "") for m in history):
            continue
        query = f"ORBIT{planted:04d}"
        hits = json.loads(tools.lcm_grep({"query": query, "mode": "full_text"}, engine=engine))
        hits = [h for h in hits.get("results", []) if h.get("store_id")]
        assert hits, f"I3 recall at turn {turn}: grep missed {query}"
        assert any(json.loads(tools.lcm_expand({"store_id": h["store_id"], "max_tokens": 4000},
                                             engine=engine)).get("content") == text for h in hits), \
            f"I3 recall at turn {turn}: expand changed {query}"
        if hasattr(tools, "lcm_recall"):
            recall = json.loads(tools.lcm_recall({"query": query, "include": "verbatim"}, engine=engine))
            assert any(h.get("store_id") in {x["store_id"] for x in hits}
                       for h in recall.get("hits", [])), f"I3 recall at turn {turn}: recall missed {query}"
        recalled += 1
    with sqlite3.connect(home / "lcm.db") as db:
        nodes = db.execute("SELECT node_id, depth, summary, source_ids, source_type FROM summary_nodes").fetchall()
        frontier = db.execute(
            "SELECT current_frontier_store_id FROM lcm_lifecycle_state WHERE conversation_id = ?",
            (engine._conversation_id,),
        ).fetchone()[0]
    row_ids = {r["store_id"] for r in rows}
    node_ids = {n[0] for n in nodes}
    covered: set[int] = set()
    for node_id, depth, summary, sources, kind in nodes:
        ids = set(json.loads(sources))
        assert summary.strip() and ids, f"I4 DAG: empty summary/sources at node {node_id}"
        assert ids <= (row_ids if kind == "messages" else node_ids), f"I4 DAG: dangling sources at node {node_id}"
        if kind == "messages":
            covered.update(ids)
    uncovered_ids = row_ids - covered
    # Hosts compact at different turns. Coverage follows the published frontier,
    # not the number of live rows observed on one reference host at this turn.
    assert uncovered_ids == {i for i in row_ids if i > frontier}, \
        f"I4 DAG at turn {turn}: source coverage disagrees with frontier {frontier}"
    live = []
    for message in history:
        remainder = engine._generated_context_carrier_remainder(message)
        live.append(canonical({**message, "content": remainder} if remainder is not None else message))
    pending = Counter(canonical(r) for r in rows if r["store_id"] in uncovered_ids)
    assert not pending - Counter(live), f"I4 DAG at turn {turn}: uncovered rows missing from live history"
    uncovered = len(uncovered_ids)
    return {"turn": turn, "rows": len(rows), "digest": digest(rows), "uncovered": uncovered,
            "frontier": frontier, "nodes": len(nodes),
            "max_depth": max((n[1] for n in nodes), default=-1), "recalled": recalled}


def run(first: int, last: int, out: Path) -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, os.getcwd())  # Hermes tools package must precede the plugin tools.py.
    reliability = importlib.import_module("bench.instruments.reliability.probe")
    reliability.guard_sockets(local_ok=True)
    fake = importlib.import_module("bench.instruments.reliability.fake_provider")
    home = Path(os.environ["HERMES_HOME"])
    ledger_path = out / "ledger.json"
    ledger: list[dict[str, Any]] = json.loads(ledger_path.read_text()) if first > 1 else []
    terminal_path = out / "terminal-history.json"
    terminal = json.loads(terminal_path.read_text()) if first > 1 else None
    resume_request_checked = False
    turn = first
    issued = False
    failures: list[str] = []
    timings: list[dict[str, Any]] = []
    compactions: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    tails: list[dict[str, Any]] = []
    pending_tail: list[str] | None = None

    def main_reply(messages: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal issued
        if turn % 10 == 0 and not issued:
            issued = True
            calls = [{"id": f"read_{turn}_{k}", "name": "read_file",
                      "arguments": json.dumps({"path": f"~/long-session/record-{turn}-{k}.txt"})}
                     for k in range(2 if turn % 20 == 0 else 1)]
            ledger.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in calls]})
            return {"tool_calls": calls}
        text = f"Acknowledged inventory turn {turn:03d}."
        ledger.append({"role": "assistant", "content": text})
        return {"content": text}

    provider = fake.FakeProvider(out / f"provider-{first}.jsonl", main=main_reply).start()
    original_reply = provider.reply_for

    def reply_for(role: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        reply: dict[str, Any] = original_reply(role, messages)
        # The kit preserves turn tags only. Keep this scenario's objective when
        # it occurs in the summary INPUT, including a higher-level condensation.
        if role == "lcm-summary" and any(OBJECTIVE in str(m.get("content") or "") for m in messages):
            reply["content"] = reply["content"].replace("Expand for details", OBJECTIVE + "\nExpand for details")
        return reply

    provider.reply_for = reply_for
    original_serve = provider.serve

    def serve(handler: Any, body: dict[str, Any], api: str) -> None:
        nonlocal pending_tail, resume_request_checked
        if fake.role_of(body.get("model", "")) == "main":
            messages = body["messages"]
            try:
                assert fake.usage_for(fake.normalize(body, api), 1.0) <= WINDOW, f"I2 assembly at turn {turn}: over window"
                assert any(user_text(turn) in str(m.get("content") or "") for m in messages), \
                    f"I2 assembly at turn {turn}: newest user changed"
                assert any(OBJECTIVE in str(m.get("content") or "") for m in messages), \
                    f"I2 assembly at turn {turn}: objective missing"
                paired(messages)
                if terminal is not None and not resume_request_checked:
                    # The leading system prompt is host-owned. Before answering
                    # the first new user turn, the entire restored conversation
                    # must still precede it, unchanged and in order.
                    assert canonical(messages[-1]) == canonical({"role": "user", "content": user_text(first)}), \
                        "I5 continuity: first request newest user changed"
                    restored = messages[-len(terminal["history"]) - 1:-1]
                    (out / "resume-request.json").write_text(json.dumps(restored, indent=2))
                    assert_restored_history(restored, terminal["history"])
                    assert_ordered_tail(restored, terminal["tail"])
                    resume_request_checked = True
                if pending_tail is not None:
                    # The provider must receive the very same emitted suffix,
                    # including any verified summary carrier at position zero.
                    assert_ordered_tail(messages, pending_tail)
                    tails[-1]["provider_checked"] = True
                    pending_tail = None
            except AssertionError as exc:
                failures.append(str(exc))
                (out / "provider-mismatch.json").write_text(json.dumps(
                    {"turn": turn, "messages": messages, "expected_tail": pending_tail}, indent=2))
        original_serve(handler, body, api)

    provider.serve = serve
    route = f"provider: custom\n    base_url: {provider.base_url}\n    api_key: synthetic-key\n"
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [hermes-lcm-x]\ncontext:\n  engine: lcm-x\n"
        "compression:\n  enabled: true\n  threshold: 0.5\n  in_place: true\n  target_ratio: 0.3\n"
        f"model:\n  default: rel/main\n  provider: custom\n  base_url: {provider.base_url}\n"
        "  api_key: synthetic-key\n  context_length: 64000\n"
        f"auxiliary:\n  compression:\n    {route}    model: rel/aux\n"
        f"  title_generation:\n    {route}    model: rel/aux-title\n"
        "memory:\n  memory_enabled: false\n  user_profile_enabled: false\n"
        "model_catalog:\n  enabled: false\n  excluded_providers: [opencode-free]\n"
    )
    plugins = importlib.import_module("hermes_cli.plugins")
    plugins.discover_plugins(force=True)
    state = importlib.import_module("hermes_state").SessionDB(db_path=home / "state.db")
    # Same durable history reader used by ACP SessionManager._restore, in a NEW process.
    history = state.get_messages_as_conversation("long-session", repair_alternation=True) if first > 1 else []
    if terminal is not None:
        (out / "restored-history.json").write_text(json.dumps(history, indent=2))
        assert_restored_history(history, terminal["history"])
        assert_ordered_tail(history, terminal["tail"])
    model_tools: Any = importlib.import_module("model_tools")
    dispatch = model_tools.handle_function_call

    def traced_dispatch(*args: Any, **kwargs: Any) -> Any:
        result = dispatch(*args, **kwargs)
        ledger.append({"role": "tool", "content": result, "tool_call_id": kwargs["tool_call_id"]})
        return result

    model_tools.handle_function_call = traced_dispatch
    agent = importlib.import_module("run_agent").AIAgent(
        api_key="synthetic-key", base_url=provider.base_url, model="rel/main", provider="custom",
        quiet_mode=True, session_db=state, session_id="long-session", skip_context_files=True,
        skip_memory=True, platform="acp", enabled_toolsets=["file", "context_engine"],
    )
    agent.tool_delay, agent.save_trajectories = 0, False
    agent.compression_in_place = True
    agent._compression_feasibility_checked = True
    engine = agent.context_compressor
    assert engine.name == "lcm-x", "I1 setup: real PluginManager did not load LCM-X"
    engine.update_model("rel/main", WINDOW, base_url=provider.base_url, api_key="synthetic-key", provider="custom")
    module: Any = importlib.import_module(type(engine).__module__.rsplit(".", 1)[0] + ".tokens")
    # Reuse the documented char-counter fallback, with no tokenizer download/race.
    module._encoder_ready, module._encoder, module._get_encoder = True, None, lambda: None
    original_compress = engine.compress

    def compress(messages: list[dict[str, Any]], *args: Any, **kwargs: Any) -> Any:
        nonlocal pending_tail
        protected = [canonical(m) for m in messages[-TAIL:]]
        started = time.perf_counter()
        result = original_compress(messages, *args, **kwargs)
        compactions.append({"turn": turn, "seconds": time.perf_counter() - started,
                            "status": engine._last_compression_status})
        emitted = result[-len(protected):]
        remainder = engine._generated_context_carrier_remainder(emitted[0]) if emitted else None
        try:
            assert_ordered_tail(result, protected, remainder)
        except AssertionError as exc:
            failures.append(f"{exc} at turn {turn}")
        pending_tail = [canonical(m) for m in emitted]
        tails.append({"turn": turn, "protected": protected, "emitted": pending_tail,
                      "carrier_remainder": remainder, "provider_checked": False})
        return result

    engine.compress = compress
    before = stored(home, engine) if first > 1 else []
    if first > 1:
        assert digest(before) == json.loads((out / "phase-1.json").read_text())["checkpoints"][-1]["digest"], \
            "I5 cursor: restart changed pre-existing stored ids"
    try:
        for turn in range(first, last + 1):
            issued = False
            ledger.append({"role": "user", "content": user_text(turn)})
            started = time.perf_counter()
            result = agent.run_conversation(user_message=user_text(turn), conversation_history=history,
                                            task_id="long-session", persist_user_message=user_text(turn))
            timings.append({"turn": turn, "seconds": time.perf_counter() - started})
            assert not result.get("error"), f"I1 host turn {turn}: {result.get('error')}"
            history = result["messages"]
            assert not failures, failures[0] if failures else ""
            if turn == first and first > 1:
                assert resume_request_checked, "I5 continuity: first provider request was not checked"
                after = stored(home, engine)
                assert [(r["store_id"], canonical(r)) for r in after[:len(before)]] == [
                    (r["store_id"], canonical(r)) for r in before], "I5 cursor: old stored ids changed"
                assert Counter(map(canonical, after)) == Counter(map(canonical, ledger)), \
                    "I5 cursor: restart re-ingested or skipped rows"
            if turn % 25 == 0:
                checks.append(checkpoint(engine, home, turn, history, ledger))
                print(json.dumps(checks[-1]), flush=True)
        if last == 300:
            assert checks[-1]["max_depth"] >= 1, "I4 DAG: no condensation above leaf level"
            assert list((home / "lcm_externalized").rglob("*.json")), "I3 recall: no large outputs externalized"
        if last == 150:
            terminal_path.write_text(json.dumps({"history": [canonical(m) for m in history],
                                                 "tail": [canonical(m) for m in history[-TAIL:]]}, indent=2))
    finally:
        (out / f"phase-{first}.json").write_text(json.dumps({"checkpoints": checks, "turns": timings,
                                                          "compactions": compactions, "tails": tails}, indent=2))
        ledger_path.write_text(json.dumps(ledger, indent=2))
        engine.shutdown()
        state.close()
        provider.stop()


if __name__ == "__main__":
    run(int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]))
