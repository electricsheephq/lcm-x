"""#1160: real Hermes lifecycle recovery after the bounded end flush skips.

Uses the real-host interpreter discovery/PluginManager fixture pattern from
test_real_plugin_manager_identity; no host or engine lifecycle methods are faked.
Only network access and provider construction are blocked. The pending tail models
an interrupted turn: Hermes has persisted it, but post_llm_call has not ingested it.
"""

import json
from pathlib import Path
import sqlite3
import subprocess
import textwrap

import pytest

from tests.test_real_plugin_manager_identity import _hermes_python


HERMES = _hermes_python()
REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(HERMES is None, reason="no real Hermes runtime available")

_PROBE = textwrap.dedent(r'''
    import io, json, logging, os, socket, sqlite3
    from pathlib import Path
    from unittest.mock import patch

    def blocked(*args, **kwargs):
        raise OSError("network blocked by #1160 probe")
    socket.socket.connect = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    log = io.StringIO()
    logging.getLogger().addHandler(logging.StreamHandler(log))
    logging.getLogger().setLevel(logging.INFO)
    from hermes_cli import plugins as P
    P.discover_plugins(force=True)
    from cli import HermesCLI
    from run_agent import AIAgent
    from hermes_state import SessionDB
    home = Path(os.environ["HERMES_HOME"])
    mode = os.environ["PROBE_MODE"]
    history = [
        {"role": "user", "content": "stored question", "timestamp": 1760000001.0},
        {"role": "assistant", "content": "stored answer", "timestamp": 1760000002.0},
        {"role": "user", "content": "pending question", "timestamp": 1760000003.0},
        {"role": "assistant", "content": "pending answer", "timestamp": 1760000004.0},
    ]
    cli = HermesCLI(model="test/model", api_key="synthetic-test-key", provider="openrouter",
                    base_url="https://openrouter.ai/api/v1", toolsets=[])
    db = cli._session_db
    assert isinstance(db, SessionDB)
    if mode == "resume-process":
        cli.session_id = (home / "original-id").read_text()
    sid = cli.session_id
    with patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(
            api_key="synthetic-test-key", base_url="https://openrouter.ai/api/v1",
            model="test/model", quiet_mode=True, session_db=db, session_id=sid,
            skip_context_files=True, skip_memory=True, skip_background_review=True,
            enabled_toolsets=[], platform="cli",
        )
    cli.agent = agent
    agent._ensure_db_session()
    engine = agent.context_compressor
    assert engine.name == "lcm-x"

    def stored(session_id):
        return [(r["store_id"], r["role"], r["content"])
                for r in engine._store.get_session_messages(session_id)]

    def post_turn(messages):
        # Same real PluginManager hook fixture as the compaction/identity tests.
        P.invoke_hook("post_llm_call", session_id=agent.session_id,
                      context_compressor=engine, conversation_history=messages,
                      platform="cli", model="test/model")

    def later_turn():
        previous = list(cli.conversation_history)
        cli.conversation_history += [
            {"role": "user", "content": "later question"},
            {"role": "assistant", "content": "later answer"},
        ]
        agent._flush_messages_to_session_db(cli.conversation_history,
                                           conversation_history=previous)
        post_turn(cli.conversation_history)

    if mode == "resume-process":
        assert cli._load_resumed_history_late()
        loaded = [(m["role"], m["content"]) for m in cli.conversation_history]
        before = stored(sid)
        later_turn()
        post_turn(cli.conversation_history)
        result = {"loaded": loaded, "before": before, "after": stored(sid)}
    else:
        (home / "original-id").write_text(sid)
        cli.conversation_history = history
        assert agent._flush_messages_to_session_db(history)
        assert len(db.get_messages_as_conversation(sid)) == 4
        post_turn(history[:2])
        prefix = stored(sid)
        assert [(r, c) for _, r, c in prefix] == [(m["role"], m["content"]) for m in history[:2]]
        if mode == "resume":
            db.create_session(session_id="other-session", source="cli", model="test/model")
            db.append_message("other-session", "user", "other question")
            db.append_message("other-session", "assistant", "other answer")
        locker = sqlite3.connect(str(engine._store.db_path), isolation_level=None)
        locker.execute("BEGIN IMMEDIATE")
        skipped = []
        class ReleaseAfterSkip(logging.Handler):
            def emit(self, record):
                if "operation=message_store.append_protected_batch" in record.getMessage():
                    skipped.append(stored(sid))
                    locker.rollback()
        release = ReleaseAfterSkip()
        logging.getLogger().addHandler(release)
        try:
            # Release only once the real short timeout has failed the atomic append.
            # This lets the subsequent /new reset use its normal connection
            # timeout, without accidentally testing the reset-contention path.
            if mode == "new":
                cli.new_session(silent=True)
            elif mode == "resume":
                agent.commit_memory_session(history)
                cli._handle_resume_command("/resume other-session")
            else:
                agent.shutdown_memory_provider(history)
            assert skipped == [prefix]
        finally:
            logging.getLogger().removeHandler(release)
            locker.rollback()
            locker.close()
        result = {"prefix": prefix, "after_skip": skipped[0],
                  "host": [(m["role"], m["content"]) for m in db.get_messages_as_conversation(sid)],
                  "warning": [line for line in log.getvalue().splitlines()
                              if "raw-message ingest skipped" in line]}
        if mode in ("new", "resume"):
            assert cli.session_id != sid
            later_turn()
            result["after_other_work"] = stored(sid)
            cli._handle_resume_command("/resume " + sid)
            assert cli.session_id == sid
            result["after_bind_back"] = stored(sid)
            later_turn()
            post_turn(cli.conversation_history)
            result["after_resume_back"] = stored(sid)
    if mode != "exit-only":
        engine.shutdown()
    print("ISSUE1160=" + json.dumps(result))
''')


@pytest.fixture(scope="module")
def real_host(tmp_path_factory):
    """Load this worktree through real Hermes with a fresh, isolated profile."""
    tmp_path = tmp_path_factory.mktemp("issue1160-real-host")
    home = tmp_path / "hermes-home"
    (home / "plugins").mkdir(parents=True)
    (home / "plugins" / "hermes-lcm-x").symlink_to(REPO_ROOT)
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [hermes-lcm-x]\ncontext:\n  engine: lcm-x\n"
        "model:\n  default: test/model\n  provider: openrouter\n"
        "lcm:\n  embeddings_enabled: false\n  temporal_rollups_enabled: false\n"
    )
    python, src = HERMES
    results = {}

    def run(mode):
        if mode in results:
            return results[mode]
        completed = subprocess.run(
            [python, "-c", _PROBE], cwd=tmp_path,
            env={
                "HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin",
                "HERMES_HOME": str(home), "PYTHONPATH": src,
                "PYTHONDONTWRITEBYTECODE": "1", "HERMES_DISABLE_LAZY_INSTALLS": "1",
                "OPENROUTER_API_KEY": "synthetic-test-key", "PROBE_MODE": mode,
            }, capture_output=True, text=True, timeout=60, check=False,
        )
        (tmp_path / f"{mode}.stdout.txt").write_text(completed.stdout)
        (tmp_path / f"{mode}.stderr.txt").write_text(completed.stderr)
        assert completed.returncode == 0, completed.stderr[-6000:] + completed.stdout[-2000:]
        result = json.loads(next(line.removeprefix("ISSUE1160=")
                                 for line in completed.stdout.splitlines() if line.startswith("ISSUE1160=")))
        if mode in ("exit", "exit-only"):
            # Inspect after the child exited, using a new read-only connection.
            sid = (home / "original-id").read_text()
            with sqlite3.connect(f"file:{home / 'lcm.db'}?mode=ro", uri=True) as conn:
                result["after_process_exit"] = conn.execute(
                    "SELECT store_id, role, content FROM messages WHERE session_id=? ORDER BY store_id", (sid,),
                ).fetchall()
        (tmp_path / f"{mode}.result.json").write_text(json.dumps(result, indent=2))
        results[mode] = result
        return result

    return run


def _pairs(rows):
    return [(role, content) for _, role, content in rows]


EXPECTED = [("user", "stored question"), ("assistant", "stored answer"),
            ("user", "pending question"), ("assistant", "pending answer")]
LATER = [("user", "later question"), ("assistant", "later answer")]


@pytest.mark.parametrize("switch", ["new", "resume"])
def test_same_process_other_session_work_recovers_skipped_tail(real_host, switch):
    """Required recovery oracle: another session's work must not strand the tail."""
    result = real_host(switch)
    assert [tuple(row) for row in result["host"]] == EXPECTED
    assert _pairs(result["after_other_work"]) == EXPECTED, result


@pytest.mark.parametrize("switch", ["new", "resume"])
def test_same_process_resume_back_recovers_exactly_once_in_order(real_host, switch):
    result = real_host(switch)
    assert _pairs(result["after_bind_back"]) == EXPECTED  # the pending tail was already retried
    assert _pairs(result["after_resume_back"]) == EXPECTED + LATER
    assert result["after_resume_back"][:2] == result["prefix"]


def test_fresh_process_resume_recovers_exactly_once_in_order(real_host):
    original = real_host("exit")
    result = real_host("resume-process")
    assert [tuple(row) for row in result["loaded"]] == EXPECTED
    assert _pairs(result["before"]) == EXPECTED
    assert result["before"][:2] == original["prefix"]
    assert _pairs(result["after"]) == EXPECTED + LATER
    assert result["after"][:2] == original["prefix"]


def test_process_exit_without_resume_retains_tail_in_lcm_store(real_host):
    """I1 oracle: Hermes retaining a copy is not LCM-X storing the occurrence."""
    result = real_host("exit")
    assert [tuple(row) for row in result["host"]] == EXPECTED  # no global transcript deletion
    assert _pairs(result["after_process_exit"]) == EXPECTED, result


def test_atexit_without_explicit_engine_shutdown_retains_tail(real_host):
    result = real_host("exit-only")
    assert _pairs(result["after_process_exit"]) == EXPECTED
