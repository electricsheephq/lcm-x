"""Real Hermes turn loop, ACP-shaped: the host's persist override must not duplicate rows.

Hermes' ACP adapter calls ``run_conversation(user_message=<raw prompt>,
persist_user_message=<raw.strip()>)`` and feeds ``result["messages"]`` back as the
next turn's history. At turn end ``finalize_turn`` rewrites the current user dict in
place to that override (``_apply_persist_user_message_override``), AFTER a same-turn
compaction has already stored the raw row and recorded it in the commit proof. The
replay identity of a user row therefore ignores leading/trailing whitespace.

Runs the real ``AIAgent.run_conversation`` in a subprocess with an isolated
``HERMES_HOME`` and the engine selected through ``context.engine: lcm-x``. Only the
provider client (MagicMock), the host's auxiliary LLM and the LCM summarizer are
stubbed; sockets are blocked. Skipped unless a Hermes Python is available: set
``LCM_REAL_HERMES_PYTHON`` (and ``LCM_REAL_HERMES_SRC`` when the source tree is not
importable from that interpreter's cwd).
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
    import io, json, logging, os, re, socket, sqlite3, sys
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch
    def _blocked(*a, **k):
        raise OSError("network blocked by probe")
    socket.socket.connect = _blocked
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked
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
    home = Path(os.environ["HERMES_HOME"])
    trailing = os.environ["PROBE_TRAILING"] == "1"
    turns = int(os.environ["PROBE_TURNS"])
    def aux_llm(**kwargs):
        text = "## Goal\\nstub\\n## Progress\\nstub" if kwargs.get("task") == "compression" else "Title"
        msg = SimpleNamespace(content=text, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model="aux", usage=None)
    import agent.title_generator as host_tg
    import agent.context_compressor as host_cc
    host_tg.call_llm = aux_llm
    host_cc.call_llm = aux_llm
    with patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, session_db=SessionDB(db_path=home / "state.db"), session_id="S0",
            skip_context_files=True, skip_memory=True, platform="acp", enabled_toolsets=["todo"],
        )
    agent.client, agent.tool_delay, agent.save_trajectories = MagicMock(), 0, False
    agent.compression_in_place = os.environ["PROBE_IN_PLACE"] == "1"
    agent._compression_feasibility_checked = True
    engine = agent.context_compressor
    engine.update_model("test/model", 64000, base_url="https://openrouter.ai/api/v1", api_key="k", provider="openrouter")
    n = {"c": 0}
    def stub(*a, **kw):
        n["c"] += 1
        text = kw.get("text") if "text" in kw else (a[0] if a else "")
        tags = sorted(set(re.findall(r"\\[T(\\d\\d)\\] user", text or "")) | set(re.findall(r"U(\\d\\d)", text or "")))
        return f"Stub summary #{n['c']} covers " + " ".join("U" + t for t in tags) + ".\\nExpand for details about: stub", 1
    for name, module in list(sys.modules.items()):
        if name.startswith("hermes_plugins.hermes_lcm_x") and hasattr(module, "summarize_with_escalation"):
            module.summarize_with_escalation = stub
    rewrites = {"n": 0}
    original_override = type(agent)._apply_persist_user_message_override
    def counted_override(self, messages):
        idx = getattr(self, "_persist_user_message_idx", None)
        in_range = isinstance(idx, int) and 0 <= idx < len(messages)
        before = messages[idx].get("content") if in_range else None
        original_override(self, messages)
        if in_range and messages[idx].get("content") != before:
            rewrites["n"] += 1
    type(agent)._apply_persist_user_message_override = counted_override
    def reply(content, prompt_tokens):
        msg = SimpleNamespace(content=content, tool_calls=None)
        response = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model="test/model")
        response.usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=20, total_tokens=prompt_tokens + 20)
        return response
    continue_turns = {int(t) for t in os.environ.get("PROBE_CONTINUE_TURNS", "").split(",") if t}
    history = []
    for t in range(1, turns + 1):
        text = "continue" if t in continue_turns else f"[T{t:02d}] user turn {t}: " + ("alpha beta gamma delta " * 400) + "end."
        raw = text + ("\\n" if trailing else "")
        est = sum(len(str(m.get("content") or "")) for m in history) // 4 + len(raw) // 4 + 800
        agent.client.chat.completions.create.side_effect = [reply(f"reply to T{t:02d}: noted item {t}.", est)]
        # ACP shape (acp_adapter/server.py): raw prompt in, stripped prompt persisted.
        result = agent.run_conversation(user_message=raw, conversation_history=history, task_id="S0",
                                        persist_user_message=raw.strip())
        if isinstance(result.get("messages"), list):
            history = result["messages"]
    db = sqlite3.connect(str(home / "lcm.db"))
    stored = db.execute("SELECT store_id, session_id, role, content FROM messages ORDER BY store_id").fetchall()
    db.close()
    rows = [(r, c) for _s, _sid, r, c in stored]
    normalized = [(r, (c or "").rstrip()) for r, c in rows if (c or "").strip() != "continue"]
    state = sqlite3.connect(str(home / "state.db"))
    host_user_texts = {c for (c,) in state.execute("SELECT content FROM messages WHERE role = 'user'")}
    state.close()
    host_user_texts |= {m.get("content") for m in history if m.get("role") == "user"}
    # Every stored user row's replay identity is a form the host holds (state.db or live history):
    # the host-rewrite override form, or the stored form Hermes' state.db keeps for a commit turn.
    identity_mismatches = [
        store_id for store_id, sid, r, c in stored if r == "user" and not any(
            engine._message_replay_identity({"store_id": key, "session_id": sid, "role": r, "content": c},
                                            stored_row=True)[1] in host_user_texts
            for key in (store_id, 0))
    ]
    log = buf.getvalue()
    print(json.dumps({
        "engine": getattr(engine, "name", None),
        "stored_rows": len(rows),
        "duplicate_rows": len(normalized) - len(set(normalized)),
        "commit_logged": log.count("as a compaction commit"),
        "conflicts": log.count("publication_invariant_conflict"),
        "compactions_published": len(re.findall(r"LCM compaction #\\d+", log)),
        "raw_user_rows": sum(1 for r, c in rows if r == "user" and (c or "").endswith("\\n")),
        "continue_rows": sum(1 for r, c in rows if r == "user" and (c or "").strip() == "continue"),
        "identity_mismatches": identity_mismatches,
        "override_rewrites": rewrites["n"],
        "user_rows_by_turn": {
            f"{i:02d}": sum(1 for r, c in rows if r == "user" and (c or "").startswith(f"[T{i:02d}]"))
            for i in range(1, turns + 1)
        },
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


def _run_turn_loop(tmp_path, *, in_place, trailing, fresh_tail=24, continue_turns=()) -> dict:
    home = tmp_path / "hermes-home"
    (home / "plugins").mkdir(parents=True)
    (home / "plugins" / "hermes-lcm-x").symlink_to(REPO_ROOT)
    (home / "config.yaml").write_text(
        "context:\n  engine: lcm-x\n"
        "compression:\n  enabled: true\n  threshold: 0.8\n"
        f"  in_place: {'true' if in_place else 'false'}\n  target_ratio: 0.3\n"
        "lcm:\n  context_threshold: 0.5\n"
        "plugins:\n  enabled: [hermes-lcm-x]\n  disabled: []\n",
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
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENROUTER_API_KEY": "test-key",
            "PROBE_IN_PLACE": "1" if in_place else "0",
            "PROBE_TRAILING": "1" if trailing else "0",
            "PROBE_TURNS": "26",
            "PROBE_CONTINUE_TURNS": ",".join(str(t) for t in continue_turns),
            "LCM_CONTEXT_THRESHOLD": "0.5",
            "LCM_FRESH_TAIL_COUNT": str(fresh_tail),
            "LCM_FRESH_TAIL_MAX_TOKENS": "12000",
            "LCM_LEAF_CHUNK_TOKENS": "4000",
            "LCM_THRESHOLD_FULL_SWEEP_ENABLED": "true",
            "LCM_NATIVE_RECOVERY": "false",
        },
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["engine"] == "lcm-x", result
    assert result["commit_logged"] >= 1, result  # a same-turn compaction really committed
    # The host rewrite under test happens exactly when the prompt carries whitespace.
    assert (result["override_rewrites"] > 0) is trailing, result
    return result


def _assert_each_turn_stored_once(result, continue_turns=()) -> None:
    not_once = {
        turn: count
        for turn, count in result["user_rows_by_turn"].items()
        if count != (0 if int(turn) in continue_turns else 1)
    }
    assert not_once == {}, result
    assert result["continue_rows"] == len(continue_turns), result
    assert result["duplicate_rows"] == 0, result
    assert result["conflicts"] == 0, result
    assert result["identity_mismatches"] == [], result


@pytest.mark.parametrize(
    "in_place",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "separate rotation defect, not the persist rewrite: a 2nd compaction inside a "
                    "rotation child hits publication_invariant_conflict, the host still rotates the "
                    "no-progress result into a proof-less child, and that child re-stores its "
                    "context (reproduces with no whitespace; rotation-child conflict, #495)"
                ),
            ),
        ),
    ],
    ids=["in-place", "rotation"],
)
def test_acp_persist_override_after_same_turn_compaction_stores_no_duplicates(tmp_path, in_place):
    _assert_each_turn_stored_once(_run_turn_loop(tmp_path, in_place=in_place, trailing=True))


@pytest.mark.parametrize("continue_turns", [(), (17, 18, 19)], ids=["distinct-prompts", "repeated-continue"])
def test_acp_persist_override_after_preflight_ingest_stores_no_duplicates(tmp_path, continue_turns):
    """Hermes' sub-threshold preflight maintenance (len(messages) > 3 + fresh tail + 1)
    ingests the RAW prompt on most turns, with no compaction and no commit proof;
    the persist override then rewrites it in place."""
    result = _run_turn_loop(tmp_path, in_place=True, trailing=True, fresh_tail=8, continue_turns=continue_turns)
    assert result["raw_user_rows"] >= 10, result  # the preflight path really stored raw prompts
    assert result["compactions_published"] >= 2, result
    _assert_each_turn_stored_once(result, continue_turns)


def test_turn_loop_without_host_rewrite_is_clean(tmp_path):
    _assert_each_turn_stored_once(_run_turn_loop(tmp_path, in_place=True, trailing=False))
