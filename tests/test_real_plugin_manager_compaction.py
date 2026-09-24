"""Real Hermes compaction commit sequence with LCM-X loaded by the real PluginManager (#483).

Runs a real Hermes checkout's ``AIAgent._compress_context`` (``_commit_compaction`` ->
``on_session_end`` -> ``_finish_compaction_boundary`` -> ``on_session_start``) and its
``repair_message_sequence`` role repair in a subprocess with an isolated ``HERMES_HOME``.
The engine is selected through ``context.engine: lcm-x``; only the LCM summarizer is
stubbed (turn tags are kept so coverage can be checked). Skipped unless a Hermes Python
is available: set ``LCM_REAL_HERMES_PYTHON`` (and ``LCM_REAL_HERMES_SRC`` when the source
tree is not importable from that interpreter's cwd).
"""

from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_PROBE = textwrap.dedent(
    """
    import io, json, logging, os, re, sqlite3, sys
    from pathlib import Path
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
    from hermes_cli import plugins as P
    P.discover_plugins(force=True)
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from agent.agent_runtime_helpers import repair_message_sequence
    home = Path(os.environ["HERMES_HOME"])
    in_place = os.environ["PROBE_IN_PLACE"] == "1"
    agent = AIAgent(
        api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
        session_db=SessionDB(db_path=home / "state.db"), session_id="S0",
        skip_context_files=True, skip_memory=True,
    )
    agent.compression_in_place = in_place
    # The host's in-place commit needs the state.db session row; without it the
    # commit fails ("Session not found") and rolls back to the input. Only the row
    # is created: the probe does not persist a transcript the agent loop would not.
    agent._ensure_db_session()
    agent._compression_feasibility_checked = True
    engine = agent.context_compressor
    engine_module = sys.modules[type(engine).__module__]
    n = {"c": 0}
    def stub(*a, **kw):
        n["c"] += 1
        text = kw.get("text") if "text" in kw else (a[0] if a else "")
        tags = sorted(set(re.findall(r"\\[T(\\d\\d)\\] user", text or "")) | set(re.findall(r"U(\\d\\d)", text or "")))
        return f"Stub summary #{n['c']} covers " + " ".join("U" + t for t in tags) + ".\\nExpand for details about: stub", 1
    engine_module.summarize_with_escalation = stub
    native = os.environ.get("LCM_NATIVE_RECOVERY") == "true"
    if native:
        import agent.context_compressor as host_cc
        class StandInNative(host_cc.ContextCompressor):
            # The real host class (its handoff helpers run); only the model call is replaced.
            def __init__(self, **kwargs):
                self.protect_last_n = kwargs["protect_last_n"]
                self._last_compress_aborted = False
                self._last_summary_fallback_used = False
                self._last_compression_made_progress = True
            def on_session_start(self, session_id, **kwargs):
                pass
            def compress(self, messages, **kwargs):
                head, tail = messages[:-self.protect_last_n], messages[-self.protect_last_n:]
                text = " ".join(str(m.get("content")) for m in head)
                tags = sorted(set(re.findall(r"\\[T(\\d\\d)\\] user", text)) | set(re.findall(r"U(\\d\\d)", text)))
                summary = host_cc.SUMMARY_PREFIX + "\\nNative stub covers " + " ".join("U" + t for t in tags) + "."
                return [{"role": "user", "content": summary}] + [dict(m) for m in tail]
        host_cc.ContextCompressor = StandInNative
    def turn(i):
        return ({"role": "user", "content": f"[T{i:02d}] user turn {i}: " + ("alpha beta gamma delta " * 40)},
                {"role": "assistant", "content": f"reply to T{i:02d}: noted item {i}."})
    def post_llm_call(history, i):
        P.invoke_hook("post_llm_call", session_id=agent.session_id, task_id="t", turn_id=f"turn-{i}",
                      user_message=history[-2]["content"], assistant_response=history[-1]["content"],
                      conversation_history=list(history), model="probe-model", platform="cli")
    host, t, statuses, sids, repairs, adopted = [], 0, [], [], [], []
    for cycle in range(1, 4):
        for _ in range(6 if cycle > 1 else 12):
            t += 1; u, a = turn(t)
            host.append(u); host.append(a); post_llm_call(host, t)
        t += 1; host.append(turn(t)[0])
        if native:
            engine._compression_cancelled_check = lambda: False
        compressed, _prompt = agent._compress_context(host, "sys", approx_tokens=100_000, force=True)
        statuses.append(engine._last_compression_status)
        adopted.append(len(compressed) < len(host))
        host = list(compressed)
        repairs.append(repair_message_sequence(agent, host))
        host.append(turn(t)[1]); post_llm_call(host, t)
        sids.append(agent.session_id)
    db = sqlite3.connect(str(home / "lcm.db"))
    rows = db.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    db.close()
    ut = [m.group(1) for r, c in rows if r == "user" for m in [re.match(r"\\[T(\\d\\d)\\] user", c or "")] if m]
    rt = [m.group(1) for r, c in rows if r == "assistant" for m in [re.match(r"reply to T(\\d\\d)", c or "")] if m]
    exp = {f"{i:02d}" for i in range(1, t + 1)}
    view = " ".join(str(m.get("content")) for m in host)
    vt = set(re.findall(r"\\[T(\\d\\d)\\] user", view)) | set(re.findall(r"U(\\d\\d)", view))
    print(json.dumps({
        "engine": getattr(engine, "name", None), "statuses": statuses, "session_ids": sids, "repairs": repairs,
        "host_adopted_output": adopted,
        "duplicate_rows": len(rows) - len(set(rows)),
        "scaffold_rows_stored": sum(1 for _r, c in rows if re.search(r"Summary \\(d\\d+, node \\d+\\)", c or "")),
        "missing_user_turns": sorted(exp - set(ut)), "missing_replies": sorted(exp - set(rt)),
        "context_missing": sorted(exp - vt),
        "commit_logged": buf.getvalue().count("as a compaction commit"),
        "errors": [l for l in buf.getvalue().splitlines() if "publication" in l.lower() and ("fail" in l.lower() or "conflict" in l.lower())][:5],
    }))
    """
)


def _hermes_python() -> tuple[str, str] | None:
    python = os.environ.get("LCM_REAL_HERMES_PYTHON")
    src = os.environ.get("LCM_REAL_HERMES_SRC", "")
    if python:
        return python, src
    if importlib.util.find_spec("hermes_cli") is not None:
        try:
            if importlib.util.find_spec("hermes_cli.plugins") is not None:
                return sys.executable, src
        except (ImportError, ValueError):
            return None
    return None


HERMES = _hermes_python()
pytestmark = pytest.mark.skipif(HERMES is None, reason="no real Hermes runtime available")


def _run_probe(tmp_path, *, in_place, tail, native=False) -> dict:
    home = tmp_path / "hermes-home"
    (home / "plugins").mkdir(parents=True)
    (home / "plugins" / "hermes-lcm-x").symlink_to(REPO_ROOT)
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [hermes-lcm-x]\ncontext:\n  engine: lcm-x\n",
        encoding="utf-8",
    )
    python, src = HERMES
    completed = subprocess.run(
        [python, "-c", _PROBE],
        cwd=src or None,
        env={
            "HOME": str(tmp_path / "home"),
            "PATH": "/usr/bin:/bin",
            "HERMES_HOME": str(home),
            "PROBE_IN_PLACE": "1" if in_place else "0",
            "LCM_FRESH_TAIL_COUNT": str(tail),
            "LCM_LEAF_CHUNK_TOKENS": "400",
            "PYTHONDONTWRITEBYTECODE": "1",
            **({"LCM_NATIVE_RECOVERY": "true"} if native else {}),
        },
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["engine"] == "lcm-x"
    assert result["host_adopted_output"] == [True] * 3, result  # a real commit, not a host rollback
    assert (len(set(result["session_ids"])) == 1) is in_place, result
    assert result["missing_user_turns"] == [], result
    assert result["missing_replies"] == [], result
    assert result["context_missing"] == [], result
    return result


@pytest.mark.parametrize("tail", [6, 7], ids=["assistant-leading-tail", "user-leading-tail"])
@pytest.mark.parametrize("in_place", [True, False], ids=["in-place", "rotation"])
def test_real_host_compaction_commits_publish_without_duplicates(tmp_path, in_place, tail):
    result = _run_probe(tmp_path, in_place=in_place, tail=tail)
    assert result["statuses"] == ["compacted"] * 3, result
    assert result["commit_logged"] == 3, result
    assert result["duplicate_rows"] == 0, result
    assert result["scaffold_rows_stored"] == 0, result
    assert result["errors"] == [], result


@pytest.mark.parametrize("in_place", [True, False], ids=["in-place", "rotation"])
def test_real_host_native_recovery_commits_keep_every_turn(tmp_path, in_place):
    """Native recovery (LCM_NATIVE_RECOVERY=true) through the real host commit: no
    commit proof exists, so the in-place start must reconcile (#484 round 1 item 1).
    The native compressor is a stand-in subclass of the real host class."""
    result = _run_probe(tmp_path, in_place=in_place, tail=6, native=True)
    assert result["statuses"] == ["host_native"] * 3, result


_NATIVE_PROOF_PROBE = textwrap.dedent(
    r"""
    import io, json, logging, os, re, sqlite3, sys
    from pathlib import Path
    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    os.environ.setdefault("OPENROUTER_API_KEY", "synthetic-test-key")
    from hermes_cli import plugins as P
    P.discover_plugins(force=True)
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from agent.agent_runtime_helpers import repair_message_sequence
    import agent.conversation_compression as host_compression

    def supports_unclaimed_nested_compressor():
        check = getattr(host_compression, "_caller_attempt_is_current", None)
        generation = getattr(host_compression, "_COMPRESSOR_ATTEMPT_GENERATION", None)
        if not callable(check) or generation is None:
            return True
        class UnclaimedCompressor:
            _compression_attempt_generation = 0
        token = generation.set(1)
        try:
            return bool(check(UnclaimedCompressor()))
        finally:
            generation.reset(token)

    nested_compressor_supported = supports_unclaimed_nested_compressor()
    home = Path(os.environ["HERMES_HOME"])
    mode, scenario = os.environ["PROBE_NATIVE_MODE"], os.environ["PROBE_SCENARIO"]
    in_place = os.environ["PROBE_IN_PLACE"] == "1"
    agent = AIAgent(
        api_key="synthetic-test-key", base_url="https://example.invalid/v1",
        model="test/model", quiet_mode=True,
        session_db=SessionDB(db_path=home / "state.db"), session_id="S0",
        skip_context_files=True, skip_memory=True,
    )
    agent.compression_in_place = in_place
    agent._ensure_db_session()
    agent._compression_feasibility_checked = True
    engine = agent.context_compressor
    engine_module = sys.modules[type(engine).__module__]
    def summary_text(text):
        tags = sorted(set(re.findall(r"\[T(\d\d)\] user", text or "")))
        return "Native summary covers " + " ".join("U" + tag for tag in tags) + "."
    engine_module.summarize_with_escalation = lambda *args, **kwargs: (
        summary_text(kwargs.get("text", "")), 1,
    )
    import agent.context_compressor as host_cc
    if mode == "standin":
        class Native(host_cc.ContextCompressor):
            def __init__(self, **kwargs):
                self.protect_last_n = kwargs["protect_last_n"]
                self._last_compress_aborted = False
                self._last_summary_fallback_used = False
                self._last_compression_made_progress = True
            def on_session_start(self, session_id, **kwargs):
                pass
            def compress(self, messages, **kwargs):
                head, tail = messages[:-self.protect_last_n], messages[-self.protect_last_n:]
                role = "assistant" if tail and tail[0].get("role") == "user" else "user"
                return [{"role": role, "content": host_cc.SUMMARY_PREFIX + "\n" + summary_text(
                    " ".join(str(row.get("content")) for row in head)
                )}] + [dict(row) for row in tail]
    else:
        class Native(host_cc.ContextCompressor):
            def _generate_summary(self, turns, **kwargs):
                return summary_text(" ".join(str(row.get("content")) for row in turns))
    host_cc.ContextCompressor = Native
    def turn_rows(index):
        call_id = f"call-{index:02d}"
        return [
            {"role": "user", "content": f"[T{index:02d}] user turn {index}: " + "alpha beta gamma delta " * 40},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": call_id, "name": "read_file",
             "content": "duplicate synthetic tool body " * 20},
            {"role": "assistant", "content": f"reply to T{index:02d}: noted item {index}."},
        ]
    def post_llm_call(history, index, user, reply):
        P.invoke_hook(
            "post_llm_call", session_id=agent.session_id, task_id="t",
            turn_id=f"turn-{index}", user_message=user["content"],
            assistant_response=reply["content"], conversation_history=list(history),
            model="probe-model", platform="cli",
        )
    host, turn, statuses, session_ids, repairs, adopted = [], 0, [], [], [], []
    for cycle in range(1, 4):
        for _ in range(6 if cycle > 1 else 12):
            turn += 1
            rows = turn_rows(turn)
            host.extend(rows)
            post_llm_call(host, turn, rows[0], rows[-1])
        turn += 1
        pending_user = turn_rows(turn)[0]
        host.append(pending_user)
        engine._compression_cancelled_check = lambda: False
        if scenario == "reject":
            engine.compress(list(host), current_tokens=250000, force=True)
            compressed = list(host)
        else:
            compressed, _prompt = agent._compress_context(
                host, "synthetic system", approx_tokens=250000, force=True,
            )
        statuses.append(engine._last_compression_status)
        adopted.append(len(compressed) < len(host))
        host = list(compressed)
        repairs.append(repair_message_sequence(agent, host))
        if scenario == "restart":
            engine._compress_commit_proof = None
            engine._ingest_cursor = 0
            engine._ingest_cursor_needs_reconcile = True
        pending_reply = {"role": "assistant", "content": f"reply to T{turn:02d}: noted item {turn}."}
        host.append(pending_reply)
        post_llm_call(host, turn, pending_user, pending_reply)
        session_ids.append(agent.session_id)
    db = sqlite3.connect(str(home / "lcm.db"))
    rows = db.execute(
        "SELECT role, content, COALESCE(tool_call_id, ''), "
        "COALESCE(tool_calls, ''), COALESCE(tool_name, '') "
        "FROM messages ORDER BY store_id"
    ).fetchall()
    db.close()
    users = {m.group(1) for role, content, *_ in rows if role == "user"
             for m in [re.match(r"\[T(\d\d)\] user", content or "")] if m}
    replies = {m.group(1) for role, content, *_ in rows if role == "assistant"
               for m in [re.match(r"reply to T(\d\d)", content or "")] if m}
    tool_ids = {tool_call_id for role, _content, tool_call_id, *_ in rows if role == "tool"}
    result = {
        "mode": mode, "scenario": scenario, "in_place": in_place,
        "nested_compressor_supported": nested_compressor_supported,
        "statuses": statuses, "session_ids": session_ids, "repairs": repairs,
        "adopted": adopted, "duplicate_rows": len(rows) - len(set(rows)),
        "users": len(users), "replies": len(replies), "tool_ids": len(tool_ids),
        "summary_rows": sum(
            str(content).startswith("Native summary covers")
            or "CONTEXT COMPACTION" in str(content)[:200]
            for _role, content, *_ in rows
        ),
    }
    print(json.dumps(result, sort_keys=True))
    """
)


def _run_native_proof_probe(tmp_path, *, mode, scenario, in_place):
    home = tmp_path / f"native-{mode}-{scenario}-{int(in_place)}"
    (home / "plugins").mkdir(parents=True)
    (home / "plugins" / "hermes-lcm-x").symlink_to(REPO_ROOT)
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [hermes-lcm-x]\ncontext:\n  engine: lcm-x\n",
        encoding="utf-8",
    )
    python, src = HERMES
    completed = subprocess.run(
        [python, "-c", _NATIVE_PROOF_PROBE], cwd=src or None,
        env={
            "HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin",
            "HERMES_HOME": str(home), "PROBE_NATIVE_MODE": mode,
            "PROBE_SCENARIO": scenario, "PROBE_IN_PLACE": "1" if in_place else "0",
            "LCM_NATIVE_RECOVERY": "true", "LCM_FRESH_TAIL_COUNT": "24",
            "LCM_FRESH_TAIL_MAX_TOKENS": "24000", "LCM_LEAF_CHUNK_TOKENS": "400",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True, text=True, timeout=600, check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("mode", ["real", "standin"])
@pytest.mark.parametrize("scenario", ["none", "restart", "reject"])
@pytest.mark.parametrize("in_place", [True, False], ids=["in-place", "rotation"])
def test_native_commit_proof_prevents_duplicate_storage(tmp_path, mode, scenario, in_place):
    result = _run_native_proof_probe(tmp_path, mode=mode, scenario=scenario, in_place=in_place)
    if (
        mode == "real"
        and scenario != "reject"
        and not result["nested_compressor_supported"]
    ):
        pytest.xfail(
            "#479: this Hermes host rejects an unclaimed nested ContextCompressor"
        )
    assert result["statuses"] == ["host_native"] * 3, result
    assert result["adopted"] == ([False] * 3 if scenario == "reject" else [True] * 3), result
    assert result["duplicate_rows"] == 0, result
    assert (result["users"], result["replies"], result["tool_ids"]) == (27, 27, 24), result
    assert result["summary_rows"] == 0, result
