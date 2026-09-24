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
    def turn(i):
        return ({"role": "user", "content": f"[T{i:02d}] user turn {i}: " + ("alpha beta gamma delta " * 40)},
                {"role": "assistant", "content": f"reply to T{i:02d}: noted item {i}."})
    def post_llm_call(history, i):
        P.invoke_hook("post_llm_call", session_id=agent.session_id, task_id="t", turn_id=f"turn-{i}",
                      user_message=history[-2]["content"], assistant_response=history[-1]["content"],
                      conversation_history=list(history), model="probe-model", platform="cli")
    host, t, statuses, sids, repairs = [], 0, [], [], []
    for cycle in range(1, 4):
        for _ in range(6 if cycle > 1 else 12):
            t += 1; u, a = turn(t)
            host.append(u); host.append(a); post_llm_call(host, t)
        t += 1; host.append(turn(t)[0])
        compressed, _prompt = agent._compress_context(host, "sys", approx_tokens=100_000, force=True)
        statuses.append(engine._last_compression_status)
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


@pytest.mark.parametrize("tail", [6, 7], ids=["assistant-leading-tail", "user-leading-tail"])
@pytest.mark.parametrize("in_place", [True, False], ids=["in-place", "rotation"])
def test_real_host_compaction_commits_publish_without_duplicates(tmp_path, in_place, tail):
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
        },
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["engine"] == "lcm-x"
    assert result["statuses"] == ["compacted"] * 3, result
    assert result["commit_logged"] == 3, result
    assert (len(set(result["session_ids"])) == 1) is in_place, result
    assert result["duplicate_rows"] == 0, result
    assert result["scaffold_rows_stored"] == 0, result
    assert result["missing_user_turns"] == [], result
    assert result["missing_replies"] == [], result
    assert result["context_missing"] == [], result
    assert result["errors"] == [], result
