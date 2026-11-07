"""Synthetic copies and fake executables only; never start a real gateway."""

from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from bench.instruments.d5_clone import gateway as g

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:12345/v1"
# Pinned host hermes_state_common.py CREATE text, with no store data.
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    created_source TEXT,
    user_id TEXT,
    session_key TEXT,
    chat_id TEXT,
    chat_type TEXT,
    thread_id TEXT,
    display_name TEXT,
    origin_json TEXT,
    expiry_finalized INTEGER DEFAULT 0,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    system_prompt_hash TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    cwd TEXT,
    git_branch TEXT,
    git_repo_root TEXT,
    git_metadata_generation INTEGER NOT NULL DEFAULT 0,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    title_source TEXT,
    last_activity_at REAL,
    last_activity_description TEXT,
    last_activity_provenance TEXT,
    api_call_count INTEGER DEFAULT 0,
    handoff_state TEXT,
    handoff_platform TEXT,
    handoff_error TEXT,
    compression_failure_cooldown_until REAL,
    compression_failure_error TEXT,
    compression_fallback_streak INTEGER NOT NULL DEFAULT 0,
    compression_ineffective_count INTEGER NOT NULL DEFAULT 0,
    compression_recovery_deadline REAL,
    compression_overload_streak INTEGER NOT NULL DEFAULT 0,
    profile_name TEXT,
    transport_profile TEXT,
    rewind_count INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    auto_archived INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    hidden INTEGER NOT NULL DEFAULT 0,
    last_read_at REAL,
    tool_names TEXT,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id),
    FOREIGN KEY (system_prompt_hash) REFERENCES system_prompts(hash)
);
CREATE TABLE IF NOT EXISTS gateway_routing (
    scope TEXT NOT NULL DEFAULT '',
    session_key TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (scope, session_key)
);
"""


def test_aux_tasks_ast_only(tmp_path):
    cli = tmp_path / "hermes_cli"
    cli.mkdir()
    config = cli / "config.py"
    config.write_text("raise RuntimeError('never import')\nDEFAULT_CONFIG = {'auxiliary': {'vision': _aux(3), 'review': {'model': ''}, 'free_only': False, 'retry': 2, 'urls': []}}")
    assert g.aux_tasks(tmp_path) == ["vision", "review"]
    (cli / "config_defaults.py").write_text(config.read_text())
    config.write_text("from hermes_cli.config_defaults import DEFAULT_CONFIG")
    assert g.aux_tasks(tmp_path) == ["vision", "review"]


def test_config_keys(tmp_path):
    settings = {"compression": {"enabled": True, "in_place": False}, "lcm": {"summary_model": "old", "context_threshold": 0.5}}
    before = deepcopy(settings)
    config = g.build_config(settings, provider_url=URL, home=tmp_path, aux=["vision", "compression"], context_length=32000,
                            stored_route_providers=["old-provider", "d5", "old-provider"])
    assert settings == before
    assert config["model"] == {"default": "d5-agent", "provider": "d5", "context_length": 32000}
    providers = config["custom_providers"]
    assert [p["name"] for p in providers] == ["d5", "old-provider"]
    assert {k: v for k, v in providers[0].items() if k != "name"} == {k: v for k, v in providers[1].items() if k != "name"}
    assert providers[0]["base_url"] == URL and providers[0]["api_key"] == "d5-dummy"
    assert providers[0]["api_mode"] == "chat_completions"
    assert providers[0]["models"] == {m: {"context_length": 32000} for m in ("d5-agent", "d5-aux", "d5-cron", "d5-summary")}
    assert config["auxiliary"] == {t: {"provider": "d5", "model": "d5-aux", "base_url": URL} for t in ("vision", "compression")}
    assert config["context"] == {"engine": "lcm-x"}
    assert config["compression"] == settings["compression"]
    assert config["lcm"] == settings["lcm"]  # No dead summary_model key: LCM-X takes it from the env.
    assert config["plugins"] == {"enabled": ["hermes-lcm-x", "d5-local"], "disabled": [], "isolation": "in_process"}
    assert config["platforms"] == {"d5-local": {"enabled": True, "extra": {}}}
    assert config["gateway"]["message_timestamps"]["enabled"] is False
    assert config["mcp_servers"] == {} and config["auth"] == {"adopt_external_logins": False}
    assert config["telemetry"] == {"shared_metrics": {"enabled": False, "send": False}}
    assert config["terminal"]["cwd"] == str(tmp_path / "workspace")


def test_write_config(tmp_path):
    yaml = pytest.importorskip("yaml")
    config = g.build_config({"compression": {}, "lcm": {}}, provider_url=URL, home=tmp_path, aux=[], context_length=32000)
    g.write_config(tmp_path, config)
    assert yaml.safe_load((tmp_path / "config.yaml").read_text()) == config


@pytest.mark.parametrize("provider,drop", [("openai", True), ("fastembed", False), ("ollama", False)])
def test_minimal_env(tmp_path, monkeypatch, provider, drop):
    monkeypatch.setenv("D5_SECRET_MARKER", "synthetic-secret")
    lcm = {"LCM_CONTEXT_THRESHOLD": "0.5", "LCM_DATABASE_PATH": "outside", "LCM_HERMES_BASE_DIR": "outside",
           "LCM_EMBEDDING_PROVIDER": provider, "LCM_EMBEDDING_MODEL": "synthetic-model", "OTHER_KEY": "ignored"}
    env, declaration = g.build_env(lcm, home=tmp_path, provider_url=URL, host_bin="/synthetic/bin", port_file=tmp_path / "port",
                                    lcm_tree=ROOT)
    assert "D5_SECRET_MARKER" not in env and "OTHER_KEY" not in env
    assert env["PATH"] == "/synthetic/bin:/usr/bin:/bin"
    for key, suffix in (("HOME", "user-home"), ("TMPDIR", "tmp"), ("LCM_DATABASE_PATH", "lcm.db")):
        assert env[key] == str(tmp_path / suffix)
    assert env["HERMES_HOME"] == env["LCM_HERMES_BASE_DIR"] == str(tmp_path)
    assert env["D5_LOCAL_PORT_FILE"] == str(tmp_path / "port")
    assert env["OPENAI_BASE_URL"] == URL and env["OPENAI_API_KEY"] == "d5-dummy"
    assert all(env[k] == "http://127.0.0.1:9" for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"))
    assert env["NO_PROXY"] == "127.0.0.1,localhost,::1"
    assert env["LCM_CONTEXT_THRESHOLD"] == "0.5"
    assert env["PYTHONDONTWRITEBYTECODE"] == env["PYTHONUNBUFFERED"] == "1"
    assert declaration["env_names"] == sorted(env)
    assert declaration["dropped_names"] == ["LCM_EMBEDDING_MODEL", *(["LCM_EMBEDDING_PROVIDER"] if drop else [])]
    assert ("LCM_EMBEDDING_PROVIDER" not in env) is drop
    assert all(isinstance(names, list) and all(isinstance(n, str) for n in names) for names in declaration.values())
    assert (tmp_path / "plugins/hermes-lcm-x").resolve() == ROOT
    assert (tmp_path / "plugins/d5-local/adapter.py").is_file()


def test_build_env_drops_every_redirecting_lcm_key(tmp_path):
    lcm = {"LCM_EMBEDDING_PROVIDER": "fastembed", "LCM_EMBEDDING_BASE_URL": "http://192.0.2.1", "LCM_SUMMARY_MODEL": "x",
           "LCM_SUMMARY_FALLBACK_MODELS": "x", "LCM_EXPANSION_MODEL": "x", "LCM_EXTRACTION_MODEL": "x",
           "LCM_OLLAMA_BASE_URL": "x", "LCM_CUSTOM_INSTRUCTIONS": "see /etc", "LCM_FRESH_TAIL_COUNT": "8",
           **{k: "/outside" for k in g.FORCED_LCM_PATHS}}
    env, declaration = g.build_env(lcm, home=tmp_path, provider_url=URL, host_bin="/b", port_file=tmp_path / "p", lcm_tree=ROOT)
    assert declaration["dropped_names"] == sorted(set(lcm) - {"LCM_EMBEDDING_PROVIDER", "LCM_FRESH_TAIL_COUNT", *g.FORCED_LCM_PATHS})
    assert {k: v for k, v in env.items() if k.startswith("LCM_") and k not in g.FORCED_LCM_PATHS} == {
        "LCM_EMBEDDING_PROVIDER": "fastembed", "LCM_FRESH_TAIL_COUNT": "8", "LCM_SUMMARY_MODEL": "d5-summary"}
    assert declaration["overridden_names"] == sorted([*g.FORCED_LCM_PATHS, "LCM_SUMMARY_MODEL"])
    assert all(Path(env[k]).is_relative_to(tmp_path) for k in g.FORCED_LCM_PATHS)
    # Every write-location key the LCM-X tree reads is forced.
    named = {m for f in ROOT.glob("*.py") for m in re.findall(r"LCM_[A-Z0-9_]*(?:PATH|DIR)\b", f.read_text())}
    assert named and named <= set(g.FORCED_LCM_PATHS)


@pytest.mark.parametrize("url", ["https://127.0.0.1/v1", "http://192.0.2.1/v1"])
def test_nonlocal_provider_refused(url):
    with pytest.raises(ValueError, match="loopback"):
        g.build_config({"compression": {}, "lcm": {}}, provider_url=url, home="unused", aux=[], context_length=1)


def test_copy_routing(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    path = run / "state.db"
    original = {"session_id": "session-a", "session_key": "old-key", "created_at": "2000-01-01T00:00:00+00:00",
                "updated_at": "2001-01-01T00:00:00+00:00", "platform": "old-platform", "chat_type": "dm",
                "origin": {"platform": "old-platform", "chat_id": "old-chat", "chat_type": "dm", "extra": "retain"},
                "metadata": {"watermark": 7}, "input_tokens": 123, "resume_pending": True, "resume_reason": "restart",
                "active_turn_token": "t", "active_turn_started_at": "2000-01-01T00:00:00", "suspended": True,
                "last_resume_marked_at": "2000-01-01T00:00:00",
                "model_override": {"model": "old-model", "provider": "override-provider", "base_url": "https://example.invalid"}}
    route = {"provider": "stored-provider", "base_url": "https://example.invalid", "api_mode": "old-mode",
             "gateway_runtime": {"provider": "runtime-provider", "model": "old-model"},
             "browser_model_lock": {"provider": "locked-provider", "model": "old-model", "confirmed": True,
                                    "model_options": {"temperature": 0.25}},
             "unrelated": {"model": "preserve-this"}}
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        for sid in ("session-a", "session-b"):
            db.execute("INSERT INTO sessions (id,source,started_at,model,model_config) VALUES (?,?,?,?,?)",
                       (sid, "synthetic", 946684800, "old-model", json.dumps(route)))
        db.execute("INSERT INTO gateway_routing VALUES (?,?,?,?)", ("old-scope", "old-key", json.dumps(original), 1))
        before = [db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("sessions", "gateway_routing")]
    conversations = [{"name": n, "session_id": s, "chat_id": c} for n, s, c in
                     (("a", "session-a", "chat-a"), ("b", "session-b", "chat-b"))]
    scope = run / "home/sessions/../sessions"
    result = g.edit_copy_routing(path, conversations, sessions_dir=scope, run_dir=run, provider_url=URL)
    assert result["sessions"] == result["routing_rows"] == 2
    assert result["stored_route_providers"] == ["locked-provider", "override-provider", "runtime-provider", "stored-provider"]
    assert "example.invalid" not in json.dumps(result)
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        after = [db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("sessions", "gateway_routing")]
        assert after == [before[0], before[1] + 2]
        assert json.loads(db.execute("SELECT entry_json FROM gateway_routing WHERE scope='old-scope'").fetchone()[0]) == original
        for conversation in conversations:
            row = db.execute("SELECT * FROM sessions WHERE id=?", (conversation["session_id"],)).fetchone()
            assert row["session_key"] == g.session_key(conversation["chat_id"])
            assert row["chat_id"] == conversation["chat_id"] and row["model"] == "d5-agent"
            origin = json.loads(row["origin_json"])
            assert origin["platform"] == "d5-local" and origin["user_id"] == "d5-user"
            assert origin["chat_id"] == conversation["chat_id"] and origin["chat_type"] == "dm"
            normalized = json.loads(row["model_config"])
            for block in (normalized, normalized["gateway_runtime"], normalized["browser_model_lock"]):
                assert block["provider"] == "d5" and block["base_url"] == URL and block["api_mode"] == "chat_completions"
            assert normalized["browser_model_lock"]["model"] == "d5-agent"
            assert normalized["browser_model_lock"]["model_options"] == {"temperature": 0.25}
            assert normalized["unrelated"] == route["unrelated"]
            entry = json.loads(db.execute("SELECT entry_json FROM gateway_routing WHERE scope=? AND session_key=?",
                                         (str(scope.resolve()), row["session_key"])).fetchone()[0])
            assert entry["session_id"] == conversation["session_id"] and entry["origin"] == origin
            assert set(("created_at", "updated_at", "session_key", "session_id")) <= entry.keys()
            if conversation["name"] == "a":
                changed = result["changes"][0]["routing_fields"]
                assert changed == sorted(["model_override", "origin", "platform", "session_key", *g._CLEARED_ENTRY])
                assert {k: entry[k] for k in g._CLEARED_ENTRY} == g._CLEARED_ENTRY
                assert {k: v for k, v in entry.items() if k not in changed} == {k: v for k, v in original.items() if k not in changed}
                assert entry["model_override"]["model"] == "d5-agent"
            else:  # A synthesized entry uses the host's naive local timestamps.
                assert entry["created_at"] == entry["updated_at"] == datetime.fromtimestamp(946684800).isoformat()


def test_copy_refusal_and_symlink(tmp_path):
    outside = tmp_path / "outside.db"
    outside.write_bytes(b"unchanged")
    run = tmp_path / "run"
    run.mkdir()
    link = run / "link.db"
    link.symlink_to(outside)
    for path in (outside, link):
        with pytest.raises(ValueError, match="ARM COPY"):
            g.edit_copy_routing(path, [], sessions_dir=run / "sessions", run_dir=run, provider_url=URL)
    assert outside.read_bytes() == b"unchanged"


def test_copy_collision_rolls_back(tmp_path):
    path = tmp_path / "state.db"
    scope = tmp_path / "sessions"
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        for sid in ("session-a", "session-b"):
            db.execute("INSERT INTO sessions (id,source,started_at,model) VALUES (?,?,?,?)", (sid, "synthetic", 0, "old-model"))
        db.execute("INSERT INTO gateway_routing VALUES (?,?,?,?)",
                   (str(scope.resolve()), g.session_key("chat-b"), json.dumps({"session_id": "someone-else"}), 0))
    conversations = [{"name": n, "session_id": f"session-{n}", "chat_id": f"chat-{n}"} for n in ("a", "b")]
    with pytest.raises(ValueError, match="belongs to another session"):
        g.edit_copy_routing(path, conversations, sessions_dir=scope, run_dir=tmp_path, provider_url=URL)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE sessions SET end_reason='user_exit' WHERE id='session-a'")
    with pytest.raises(ValueError, match="has ended"):
        g.edit_copy_routing(path, conversations[:1], sessions_dir=scope, run_dir=tmp_path, provider_url=URL)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT model FROM sessions ORDER BY id").fetchall() == [("old-model",), ("old-model",)]
        assert db.execute("SELECT count(*) FROM gateway_routing").fetchone()[0] == 1


@pytest.fixture
def fake_gateway(tmp_path, monkeypatch):
    host_bin, home = tmp_path / "bin", tmp_path / "home"
    host_bin.mkdir()
    executable = host_bin / "hermes"
    executable.write_text(f"#!{sys.executable}\n" + '''
import hashlib, json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
assert sys.argv[1:] == ['gateway', 'run', '--no-supervise']
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def respond(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self): self.respond({'connected': self.path == '/health', 'path': self.path})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.respond({'step': body['step'], 'completed': True, 'replies': 1,
                      'reply_sha256': [hashlib.sha256(body['text'].encode()).hexdigest()]})
import subprocess  # A group member that ignores SIGTERM.
child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])
Path(os.environ['D5_LOCAL_PORT_FILE'] + '.child').write_text(str(child.pid))
server = HTTPServer(('127.0.0.1', 0), Handler)
Path(os.environ['D5_LOCAL_PORT_FILE']).write_text(str(server.server_port))
server.serve_forever()
''')
    executable.chmod(0o700)
    env, _ = g.build_env({}, home=home, provider_url=URL, host_bin=host_bin, port_file=tmp_path / "port", lcm_tree=ROOT)
    monkeypatch.setattr(g.sandbox, "available", lambda: False)
    return g.Gateway(host_bin, home, env, tmp_path, provider_port=9, allow_unsandboxed=True)


@pytest.mark.parametrize("wrapped", [False, True])
def test_gateway_process(fake_gateway, monkeypatch, wrapped):
    calls = []
    monkeypatch.setattr(g.sandbox, "available", lambda: wrapped)
    monkeypatch.setattr(g.sandbox, "wrap", lambda argv, **kw: calls.append(kw) or argv)
    try:
        fake_gateway.start(3)
        pid = fake_gateway.process.pid
        assert pid in fake_gateway.pids()
        assert fake_gateway.turn("chat-a", "one", "synthetic\r\n  ", 2)["completed"] is True
        assert fake_gateway.replies("chat a&b")["path"] == "/replies?chat_id=chat+a%26b"
        assert os.getpgid(pid) == pid
        assert len(calls) == int(wrapped)
        if wrapped:
            assert calls[0] == {"ports": [9], "socket_roots": [fake_gateway.run_dir]}
        assert fake_gateway.stop(2) == -signal.SIGTERM
        assert fake_gateway.pids() == []
        child, deadline = int(Path(f"{fake_gateway.port_file}.child").read_text()), time.monotonic() + 5
        while time.monotonic() < deadline and subprocess.run(["ps", "-p", str(child)], capture_output=True).returncode == 0:
            time.sleep(0.01)
        assert subprocess.run(["ps", "-p", str(child)], capture_output=True).returncode == 1
        assert (fake_gateway.run_dir / "gateway-1.out").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        fake_gateway.stop(1)


def test_gateway_refuses_unsandboxed(fake_gateway):
    fake_gateway.allow_unsandboxed = False
    with pytest.raises(RuntimeError, match="refusing unsandboxed"):
        fake_gateway.start(1)
    assert fake_gateway.process is None


def test_gateway_start_failure_records_exit(fake_gateway):
    (fake_gateway.host_bin / "hermes").write_text(f"#!{sys.executable}\nraise SystemExit(23)\n")
    with pytest.raises(RuntimeError, match="exit_code=23"):
        fake_gateway.start(2)
    assert fake_gateway.exit_code == 23


def test_stop_kills_exact_pid_after_timeout(fake_gateway):
    # No sockets: a child confirms SIGTERM is ignored before the parent stops it.
    marker = fake_gateway.run_dir / "ready"
    script = "import signal,time,pathlib; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path(__import__('sys').argv[1]).touch(); time.sleep(30)"
    fake_gateway.process = subprocess.Popen([sys.executable, "-c", script, str(marker)], env=fake_gateway.env,
                                            start_new_session=True)
    try:
        import time
        deadline = time.monotonic() + 2
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        assert fake_gateway.stop(0.01) == -signal.SIGKILL
    finally:
        fake_gateway.stop(0.01)


def test_a_restarted_gateway_is_stopped(fake_gateway, monkeypatch):
    monkeypatch.setattr(g.sandbox, "available", lambda: False)
    try:
        fake_gateway.start(3)
        assert fake_gateway.stop(2) == -signal.SIGTERM
        fake_gateway.start(3)
        second = fake_gateway.process
        assert fake_gateway.stop(2) == -signal.SIGTERM
        assert second.poll() is not None and fake_gateway.pids() == []
    finally:
        fake_gateway.stop(1)


def test_a_second_stop_never_signals(fake_gateway, monkeypatch):
    fake_gateway.process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    assert fake_gateway.stop(5) == -signal.SIGTERM
    signals = []  # The exit is recorded: the pid and group id may now belong to another process.
    monkeypatch.setattr(g.os, "killpg", lambda pid, sig: signals.append(sig))
    assert fake_gateway.stop(0.01) == -signal.SIGTERM
    assert signals == []


def test_a_gateway_that_exits_by_itself_is_never_signalled_later(fake_gateway, monkeypatch):
    fake_gateway.process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"], start_new_session=True)
    deadline = time.monotonic() + 10
    while fake_gateway.pids() and time.monotonic() < deadline:  # pids() is the poll that reaps the exited gateway.
        time.sleep(0.02)
    signals = []  # Reaped: its pid and group id are free, so stop() must send nothing.
    monkeypatch.setattr(g.os, "killpg", lambda pid, sig: signals.append(sig))
    assert fake_gateway.stop(0.01) == 7
    assert signals == []


@pytest.mark.parametrize("value, passed", [("voyage", False), ("openai", False), ("fastembed", True), ("", True)])
def test_build_env_drops_a_remote_recall_provider(tmp_path, value, passed):
    env, declaration = g.build_env({"LCM_PROACTIVE_RECALL_PROVIDER": value}, home=tmp_path, provider_url=URL,
                                   host_bin="/b", port_file=tmp_path / "p", lcm_tree=ROOT)
    assert ("LCM_PROACTIVE_RECALL_PROVIDER" in env) is passed
    assert ("LCM_PROACTIVE_RECALL_PROVIDER" in declaration["dropped_names"]) is not passed


@pytest.mark.parametrize("lcm", [{}, {"LCM_SUMMARY_MODEL": "another-summary-model"}])
def test_build_env_pins_the_summary_model(tmp_path, lcm):
    env, declaration = g.build_env(lcm, home=tmp_path, provider_url=URL, host_bin="/b", port_file=tmp_path / "p",
                                   lcm_tree=ROOT)
    assert env["LCM_SUMMARY_MODEL"] == "d5-summary" and "LCM_SUMMARY_MODEL" in declaration["overridden_names"]


@pytest.mark.parametrize("provider, flag, expected", [
    ("voyage", "0", "false"), ("voyage", "1", "false"), ("voyage", None, "false"),
    ("fastembed", "0", "0")])  # The local row is a guard: a local provider's flag passes through as given.
def test_build_env_turns_embeddings_off_when_a_remote_provider_is_dropped(tmp_path, provider, flag, expected):
    lcm = {"LCM_EMBEDDING_PROVIDER": provider, **({"LCM_EMBEDDINGS_ENABLED": flag} if flag is not None else {})}
    env, declaration = g.build_env(lcm, home=tmp_path, provider_url=URL, host_bin="/b", port_file=tmp_path / "p",
                                   lcm_tree=ROOT)
    assert env["LCM_EMBEDDINGS_ENABLED"] == expected
    assert ("LCM_EMBEDDINGS_ENABLED" in declaration["overridden_names"]) is (provider == "voyage")


def test_nonlocal_connections(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="""COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME
python 11 user 1 IPv4 0 0 TCP 127.0.0.1:40->127.0.0.1:41 (ESTABLISHED)
python 11 user 2 IPv6 0 0 TCP [::1]:40->[::1]:41 (ESTABLISHED)
python 12 user 3 IPv4 0 0 TCP 127.0.0.1:40->192.0.2.1:443 (ESTABLISHED)
python 12 user 4 IPv6 0 0 TCP [::1]:40->[2001:db8::1]:443 (ESTABLISHED)
python 12 user 5 IPv4 0 0 TCP *:40 (LISTEN)
""")

    monkeypatch.setattr(g.subprocess, "run", run)
    assert g.nonlocal_connections([]) == 0 and calls == []
    assert g.nonlocal_connections([11, 12]) == 2
    assert calls == [["lsof", "-a", "-p", "11,12", "-i", "TCP", "-n", "-P"]]
    monkeypatch.setattr(g.subprocess, "run", lambda argv, **kwargs: SimpleNamespace(returncode=2, stdout=""))
    with pytest.raises(RuntimeError, match="lsof"):
        g.nonlocal_connections([11])


def test_build_env_links_the_arm_tree_and_refuses_a_non_plugin_tree(tmp_path):
    arm = tmp_path / "arm"
    arm.mkdir()
    (arm / "plugin.yaml").write_text("name: hermes-lcm-x\n", encoding="utf-8")
    g.build_env({}, home=tmp_path / "home", provider_url=URL, host_bin="/synthetic/bin", port_file=tmp_path / "port",
                lcm_tree=arm)
    assert (tmp_path / "home/plugins/hermes-lcm-x").resolve() == arm.resolve()
    with pytest.raises(ValueError):
        g.build_env({}, home=tmp_path / "home2", provider_url=URL, host_bin="/synthetic/bin",
                    port_file=tmp_path / "port", lcm_tree=tmp_path)
    with pytest.raises(ValueError):  # An existing link to another tree is never repointed.
        g.build_env({}, home=tmp_path / "home", provider_url=URL, host_bin="/synthetic/bin",
                    port_file=tmp_path / "port", lcm_tree=ROOT)
