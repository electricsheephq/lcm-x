"""Reliability harness R2 (bench/instruments/reliability): the fake provider, the ACP driver framing, the socket
guard, chronology and the process-cell plumbing. No Hermes: the ACP peer is a tiny stdio JSON-RPC script."""
from __future__ import annotations

import http.client
import json
import socket
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.instruments.reliability import acp_driver as AD, cells, ci, fake_provider as FP, hosts, plugin_tree, probe  # noqa: E402
from bench.instruments.reliability import process_cell as PC, run_matrix as RM  # noqa: E402
from bench.instruments.reliability.scorers import chronology  # noqa: E402


@pytest.fixture
def provider(tmp_path):
    calls = []

    def main(messages):
        calls.append(messages)
        last = messages[-1]["content"]
        if "fault429" in last:
            return {"status": 429}
        if "tool" in last:
            return {"tool_calls": [{"id": "call_1", "name": "lcm_grep", "arguments": '{"query": "alpha"}'}]}
        return {"content": "reply to T01: noted item 1."}
    p = FP.FakeProvider(tmp_path / "req.jsonl", main=main, usage_scale=2.0).start()
    p.calls = calls
    yield p
    p.stop()


def post(p, path, body, raw=False):
    con = http.client.HTTPConnection("127.0.0.1", p.port, timeout=10)
    con.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    resp = con.getresponse()
    data = resp.read().decode()
    con.close()
    return resp.status, (data if raw else json.loads(data))


def sse_data(text):
    return [line[6:] for line in text.splitlines() if line.startswith("data: ")]


MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "[T01] user turn 1: alpha end."}]


def test_roles_route_by_model_and_usage_is_computed_from_received_messages(provider):
    status, out = post(provider, "/v1/chat/completions", {"model": "rel/main", "messages": MSGS})
    assert status == 200 and out["choices"][0]["message"]["content"] == "reply to T01: noted item 1."
    assert out["usage"]["prompt_tokens"] == ((3 + len(MSGS[1]["content"])) // 4 + 800) * 2
    nonce = "ab12" * 8
    lcm = [{"role": "system", "content": f'policy <lcm-summary nonce="{nonce}">'}, MSGS[1]]
    _, out = post(provider, "/v1/chat/completions", {"model": "rel/lcm-summary", "messages": lcm})
    text = out["choices"][0]["message"]["content"]
    assert text.startswith(f'<lcm-summary nonce="{nonce}">') and "UT01" in text and "[T01]" not in text
    assert text.rstrip().endswith("Expand for details about: stub\n</lcm-summary>")
    _, out = post(provider, "/v1/chat/completions", {"model": "aux", "messages": MSGS})
    assert out["choices"][0]["message"]["content"].startswith("## Goal")
    assert post(provider, "/v1/chat/completions", {"model": "rel/nope", "messages": MSGS})[0] == 400
    con = http.client.HTTPConnection("127.0.0.1", provider.port, timeout=10)
    con.request("GET", "/v1/models")
    assert {m["id"] for m in json.loads(con.getresponse().read())["data"]} >= {"rel/main", "rel/lcm-summary"}
    log = [json.loads(x) for x in (provider.log_path).read_text().splitlines()]
    assert [r.get("role", r["route"]) for r in log] == ["main", "lcm-summary", "aux", "nope", "models"]
    assert len({r["rid"] for r in log}) == 5 and all(len(r["messages_sha256"]) == 64 for r in log[:4])


def test_every_request_is_logged_and_an_unrouted_request_fails_accounting(provider):
    """Regression (R2a review): GET /v1/models and unknown routes bypassed the log, so accounting passed on an
    incomplete request set."""
    post(provider, "/v1/chat/completions", {"model": "rel/main", "messages": MSGS})
    con = http.client.HTTPConnection("127.0.0.1", provider.port, timeout=10)
    con.request("GET", "/v1/models")
    con.getresponse().read()
    d, emit = provider.log_path.parent, [{"phase": "A", "event": "emit"}]
    (d / "provider-requests.jsonl").write_text(provider.log_path.read_text())
    acct = PC.accounting(d, emit)
    assert acct["ok"] and acct["requests_by_route"] == {"completion": 1, "models": 1}
    for method, path in (("GET", "/v1/models/rel/aux"), ("GET", "/v1/files"), ("POST", "/v1/embeddings")):
        con.request(method, path, "{}" if method == "POST" else None)
        resp = con.getresponse()
        assert resp.read() and resp.status == 404
    (d / "provider-requests.jsonl").write_text(provider.log_path.read_text())
    log = [json.loads(x) for x in provider.log_path.read_text().splitlines()]
    assert [(r["method"], r["path"]) for r in log[1:]] == [("GET", "/v1/models"), ("GET", "/v1/models/rel/aux"),
                                                           ("GET", "/v1/files"), ("POST", "/v1/embeddings")]
    acct = PC.accounting(d, emit)
    assert not acct["ok"] and [u["path"] for u in acct["unexpected_requests"]] == ["/v1/files", "/v1/embeddings"]


def test_openai_sse_framing_text_tools_usage_and_done(provider):
    _, text = post(provider, "/v1/chat/completions", {"model": "rel/main", "messages": MSGS, "stream": True,
                                                      "stream_options": {"include_usage": True}}, raw=True)
    data = sse_data(text)
    assert data[-1] == "[DONE]"
    chunks = [json.loads(d) for d in data[:-1]]
    assert chunks[0]["object"] == "chat.completion.chunk" and chunks[0]["choices"][0]["delta"]["content"].startswith("reply")
    assert chunks[1]["choices"][0]["finish_reason"] == "stop" and chunks[2]["choices"] == [] and chunks[2]["usage"]["prompt_tokens"] > 0
    tool = [{"role": "user", "content": "please tool"}]
    _, text = post(provider, "/v1/chat/completions", {"model": "rel/main", "messages": tool, "stream": True}, raw=True)
    chunks = [json.loads(d) for d in sse_data(text)[:-1]]
    call = chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert call["id"] == "call_1" and call["function"]["name"] == "lcm_grep" and call["index"] == 0
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls" and "usage" not in chunks[-1]


def test_anthropic_messages_json_and_sse_and_tool_result_normalisation(provider):
    body = {"model": "rel/main", "system": "sys", "max_tokens": 50, "messages": [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "R"},
                                     {"type": "text", "text": "[T01] user turn 1: alpha end."}]}]}
    _, out = post(provider, "/v1/messages", body)
    assert out["type"] == "message" and out["content"][0]["text"].startswith("reply") and out["stop_reason"] == "end_turn"
    assert provider.calls[-1][1] == {"role": "tool", "tool_call_id": "tu1", "content": "R"}
    _, text = post(provider, "/v1/messages", {**body, "stream": True}, raw=True)
    names = [line[7:] for line in text.splitlines() if line.startswith("event: ")]
    assert names == ["message_start", "content_block_start", "content_block_delta", "content_block_stop",
                     "message_delta", "message_stop"]


def test_fault_hooks_http_error_hold_until_killed_and_slow(tmp_path):
    fired = []
    replies = iter([{"status": 500}, {"hold_until_killed": True, "on_hold": lambda: fired.append("held")},
                    {"hold": 0.3, "content": "late"}])
    p = FP.FakeProvider(tmp_path / "req.jsonl", main=lambda m: next(replies), hold_cap=10).start()
    try:
        assert post(p, "/v1/chat/completions", {"model": "rel/main", "messages": MSGS})[0] == 500
        raw = json.dumps({"model": "rel/main", "messages": MSGS}).encode()
        sock = socket.create_connection(("127.0.0.1", p.port))
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(raw)).encode() + b"\r\n\r\n" + raw)
        assert p.in_flight.wait(5) and fired == ["held"]
        sock.close()  # the "host" dies while its request is held
        started = time.monotonic()
        assert post(p, "/v1/chat/completions", {"model": "rel/main", "messages": MSGS})[1]["choices"][0]["message"]["content"] == "late"
        assert time.monotonic() - started >= 0.3
        for _ in range(50):
            log = [json.loads(x) for x in p.log_path.read_text().splitlines()]
            if any(r.get("phase") == "client_closed" for r in log):
                break
            time.sleep(0.1)
        assert [r.get("fault") for r in log if r.get("phase") in (None, "held")][:2] == ["http_500", "hold_until_killed"]
        assert any(r.get("phase") == "client_closed" for r in log)
    finally:
        p.stop()


def test_driver_framing():
    assert AD.frame({"a": 1}) == b'{"a":1}\n'
    msgs, rest = AD.parse_frames(bytearray(b'{"id":1}\n\n{"method":"x"}\n{"par'))
    assert msgs == [{"id": 1}, {"method": "x"}] and rest == bytearray(b'{"par')
    with pytest.raises(AD.DriverError):
        AD.parse_frames(bytearray(b"not json\n"))
    upd = {"method": "session/update", "params": {"update": {"sessionUpdate": "agent_message_chunk", "content": {"text": "hi"}}}}
    assert AD.agent_text(upd) == "hi" and AD.agent_text({"method": "session/update", "params": {}}) is None


PEER = textwrap.dedent("""
    import json, sys
    for line in sys.stdin:
        m = json.loads(line)
        if "id" not in m or "method" not in m:
            continue
        if m["method"] == "session/prompt":
            print(json.dumps({"jsonrpc": "2.0", "id": 99, "method": "session/request_permission", "params": {}}), flush=True)
            reply = json.loads(sys.stdin.readline())
            assert reply["id"] == 99 and reply["error"]["code"] == -32601
            for part in ("he", "llo"):
                print(json.dumps({"jsonrpc": "2.0", "method": "session/update", "params": {"update": {
                    "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": part}}}}), flush=True)
            print(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": {"stopReason": "end_turn"}}), flush=True)
        elif m["method"] == "session/new":
            print(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": {"sessionId": "s-1"}}), flush=True)
        else:
            print(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": {}}), flush=True)
""")


def test_acp_process_against_a_stdio_peer(tmp_path):
    proc = AD.AcpProcess([sys.executable, "-c", PEER], {"PATH": "/usr/bin:/bin"}, tmp_path, tmp_path / "err")
    try:
        proc.initialize(10)
        sid = proc.new_session(tmp_path, 10)
        assert sid == "s-1" and proc.prompt(sid, "hi", 10) == ("hello", "end_turn")
    finally:
        proc.close()
    killed = AD.AcpProcess([sys.executable, "-c", "import time; time.sleep(30)"], {"PATH": "/usr/bin:/bin"}, tmp_path,
                           tmp_path / "err2")
    killed.kill()
    with pytest.raises(AD.ProcessGone):
        killed.request("initialize", {}, 10)
    assert killed.close() == -9


GUARD = textwrap.dedent("""
    import json, socket, sys
    sys.path.insert(0, sys.argv[1])
    import probe
    seen = []
    probe.guard_sockets(local_ok=sys.argv[2] == "1", on_refuse=lambda *a: seen.append(list(a[:2])))
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    out = {}
    for name, fn in [("local", lambda: socket.create_connection(srv.getsockname(), timeout=1)),
                     ("remote", lambda: socket.socket().connect(("192.0.2.1", 80))),
                     ("remote_ex", lambda: socket.socket().connect_ex(("192.0.2.1", 80))),
                     ("udp", lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("192.0.2.1", 53))),
                     ("dns", lambda: socket.getaddrinfo("example.com", 443)),
                     ("dns127", lambda: socket.getaddrinfo("127.attacker.example", 80))]:
        try:
            fn(); out[name] = "ok"
        except OSError:
            out[name] = "refused"
    print(json.dumps({"out": out, "seen": seen}))
""")


@pytest.mark.parametrize("local_ok", ["1", "0"])
def test_socket_guard_refuses_connect_ex_udp_literals_and_dns(local_ok):
    rel = str(Path(__file__).resolve().parent.parent / "bench" / "instruments" / "reliability")
    got = json.loads(subprocess.run([sys.executable, "-c", GUARD, rel, local_ok], capture_output=True, text=True,
                                    check=True).stdout)
    assert got["out"] == {"local": "ok" if local_ok == "1" else "refused", "remote": "refused", "remote_ex": "refused",
                          "udp": "refused", "dns": "refused", "dns127": "refused"}
    assert {"connect", "connect_ex", "sendto", "getaddrinfo"} <= {what for what, _host in got["seen"]}
    assert ["getaddrinfo", "127.attacker.example"] in got["seen"]  # R2a.3: refused and recorded


def test_loopback_is_exact_names_or_loopback_literals():
    """Regression (R2a.3, #570 thread): ``startswith("127.")`` let 127.attacker.example reach the real resolver."""
    for host in ("localhost", "::1", "127.0.0.1", "127.5.6.7", b"127.0.0.1"):
        assert probe.is_loopback(host), host
    for host in ("127.attacker.example", "127.0.0.1.nip.io", "localhost.example", "10.0.0.1", "", "0.0.0.0"):
        assert not probe.is_loopback(host), host


def test_chronology_is_reported_per_lineage():
    rows = [(1, "S0", "user", "[T01] user turn 1"), (2, "S0", "assistant", "r"), (3, "S0", "user", "[T03] user turn 3"),
            (4, "S0", "user", "[T02] user turn 2"), (5, "X", "user", "[K01] user turn 1")]
    rep = chronology.report(rows, lambda sid: "chat" if sid == "S0" else sid)
    assert rep["user_rows_checked"] == 4 and rep["violations"] == 1
    assert rep["examples"][0] == {"lineage": "chat", "store_id": 4, "tag": "T02", "after_tag": "T03", "after_store_id": 3}


def test_process_cells_are_selected_or_unsupported_and_config_is_localhost_only():
    by = {c["id"]: c for c in cells.registry()}
    for cid in ("baseline/in-place/acp", "crash-after-compaction/in-place/acp-history", "cancel-retry/in-place",
                "lcm-tool-mid-turn/in-place"):
        assert PC.unsupported(by[cid], "acp-process") is None, cid
    assert "in-process" in PC.unsupported(by["publication-failure/pass-3-in-place"], "acp-process")
    assert "gateway-process" in PC.unsupported(by["gateway-second-restart/in-place"], "acp-process")
    assert "cron" in PC.unsupported(by["multi-session-one-process/in-place"], "acp-process")
    plugin = {"engine": "lcm-x", "enabled": "hermes-lcm-x"}
    cfg = PC.config_yaml(by["baseline/in-place/acp"], plugin, "http://127.0.0.1:5/v1")
    urls = [line.split(":", 1)[1].strip().strip('"') for line in cfg.splitlines() if "base_url" in line]
    assert len(urls) == 3 and all(u == "http://127.0.0.1:5/v1" for u in urls)
    assert "context_length: 128000" in cfg and "model_catalog:\n  enabled: false" in cfg


def test_accounting_matches_requests_to_scripted_steps(tmp_path):
    reqs = [{"rid": 1, "role": "main", "reply": {}, "kind": "normal"},  # "kind" = the scenario's turn kind
            {"rid": 2, "role": "main", "phase": "held", "fault": "hold_until_killed"},
            {"rid": 2, "role": "main", "phase": "client_closed"}, {"rid": 3, "role": "lcm-summary", "reply": {}}]
    reqs = [{**r, "route": "completion", "method": "POST", "path": "/v1/chat/completions"} for r in reqs]
    (tmp_path / "provider-requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    events = [{"phase": "A", "event": "emit"}, {"phase": "A", "event": "crash", "fault": "crash_after_compaction_before_reply"}]
    acct = PC.accounting(tmp_path, events)
    assert acct["ok"] and acct["requests_by_role"] == {"main": 2, "lcm-summary": 1}
    assert not PC.accounting(tmp_path, events[:1])["ok"]
    rotation_kill = {"phase": "A", "event": "crash", "fault": "crash_after_rotation_before_child_row"}  # holds no request
    assert not PC.accounting(tmp_path, [events[0], rotation_kill])["ok"]


def test_rotation_killer_skips_a_lost_race_and_kills_on_the_next_empty_child(tmp_path, monkeypatch):
    states = iter([("S1", 5), ("S2", 3), ("S3", 0)])  # baseline, child already has rows (lost race), empty child
    monkeypatch.setattr(AD, "rotation_probe", lambda *_: next(states))
    killed, fired = [], []
    killer = AD.RotationKiller(types.SimpleNamespace(kill=lambda: killed.append(1)), tmp_path / "lcm.db", "c", fired.append)
    killer.check(kill=True)
    assert killer.event is None and killer.baseline == "S2" and not killed
    killer.check(kill=True)
    assert killed and fired[0]["child_session_id"] == "S3" and fired[0]["missed_rotations"] == 1 and fired[0]["killed"]


def test_plugins_root_symlink_or_escape_is_refused_before_any_cache_write(tmp_path):
    outside, out = tmp_path / "outside", tmp_path / "out"
    (outside / "keep").mkdir(parents=True)
    out.mkdir()
    (out / "plugins").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        plugin_tree.export(tmp_path, "HEAD", out / "plugins", out)
    with pytest.raises(ValueError):
        plugin_tree.export(tmp_path, "HEAD", outside, out)
    assert (outside / "keep").is_dir() and list(outside.iterdir()) == [outside / "keep"]


@pytest.mark.parametrize("broken", ["missing_other_module", "bad_name", "legacy_host"])
def test_final_check_falls_back_only_when_the_manual_module_is_missing(monkeypatch, broken):
    fallback = []
    mods = {"agent": types.SimpleNamespace(), "acp_adapter": types.SimpleNamespace(),
            "acp_adapter.commands": types.SimpleNamespace(_estimate_tokens=lambda *a: 1),
            "agent.conversation_compression_manual": types.SimpleNamespace(compress_now=None, parse_compress_args=None),
            "agent.conversation_compression": types.SimpleNamespace(finalize_context_engine_compression_notification=None)}
    if broken == "missing_other_module":
        mods["agent.conversation_compression"] = None  # ModuleNotFoundError, but for another module
    elif broken == "bad_name":
        mods["agent.conversation_compression"] = types.SimpleNamespace()  # plain ImportError: name not found
    else:
        mods["agent.conversation_compression_manual"] = None
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    engine = types.SimpleNamespace(_last_compression_status="compacted")
    agent = types.SimpleNamespace(context_compressor=engine, _compress_context=lambda *a, **k: fallback.append(1) or (a[0], None))
    out = probe.final_check(agent, [], probe.io.StringIO())
    if broken == "legacy_host":
        assert fallback and out["entry"] == "_compress_context(force=True)"
    else:
        assert out["outcome"] == "failed" and out["exception"] and not fallback


def test_host_processes_get_a_private_pycache_and_sourceless_pyc_is_refused(tmp_path, monkeypatch):
    seen = {}

    def fake_run(argv, **kw):
        seen["env"] = kw["env"]
        return types.SimpleNamespace(stdout='{"exit": "error", "reason": "x"}', stderr="", returncode=1)
    monkeypatch.setattr(RM.subprocess, "run", fake_run)
    tree = tmp_path / "tree"
    tree.mkdir()
    plugin = {"ref": "r", "sha": "a" * 40, "tree": str(tree), "dir": "p", "enabled": "p", "engine": "e"}
    cell = cells.select("baseline/in-place/acp")[0]
    RM.run_cell(cell, "h", {"python": sys.executable, "src": str(tmp_path), "sha": "s"}, plugin, tmp_path / "out", 5,
                False, identity={"method": "test"})
    assert seen["env"]["PYTHONPYCACHEPREFIX"].endswith("/pycache") and str(tmp_path / "out") in seen["env"]["PYTHONPYCACHEPREFIX"]
    run = PC.ProcessCell(cell, tmp_path / "c", {"python": sys.executable, "src": str(tmp_path)}, "acp-process", 5)
    try:
        assert run.env()["PYTHONPYCACHEPREFIX"] == str(tmp_path / "c" / "pycache")
    finally:
        run.proxy.stop()
    src = tmp_path / "src"
    (src / "__pycache__").mkdir(parents=True)
    (src / "m.py").write_text("")
    (src / "__pycache__" / "m.cpython-311.pyc").write_bytes(b"")
    assert hosts.sourceless_pyc(src) == []
    (src / "__pycache__" / "gone.cpython-311.pyc").write_bytes(b"")
    (src / "loose.pyc").write_bytes(b"")
    assert sorted(hosts.sourceless_pyc(src)) == ["__pycache__/gone.cpython-311.pyc", "loose.pyc"]
    with pytest.raises(ValueError, match="sourceless"):
        hosts.verify("h", {"src": str(src), "sha": "s"})


def test_extra_turns_only_when_a_pass_just_consumed_the_backlog(tmp_path):
    cell = {"turns": 5, "final_compaction_check": True}
    (tmp_path / "transcript.jsonl").write_text(json.dumps({"event": "compaction", "turn": 4, "compression_status": "compacted"}) + "\n")
    log = []
    assert list(probe.extend_turns(cell, 1, lambda t: probe.backlog_low(tmp_path, t, log))) == [1, 2, 3, 4, 5, 6, 7]
    assert [c["turns_since_pass"] for c in log] == [1, 2, 3]
    (tmp_path / "transcript.jsonl").write_text(json.dumps({"event": "compaction", "turn": 1, "compression_status": "compacted"}) + "\n")
    assert list(probe.extend_turns(cell, 4, lambda t: probe.backlog_low(tmp_path, t, []))) == [4, 5]
    assert list(probe.extend_turns({**cell, "final_compaction_check": False}, 1, lambda t: True)) == [1, 2, 3, 4, 5]


def full_set(transport=None, **over):
    """One PASS row per expected cell of one (host, transport, plugin sha), with ``over`` = {cell: row fields}."""
    extra = {"transport": transport} if transport else {}
    return [{"verdict": "PASS", "host": "h", "plugin_sha": "s", "cell": c, "targets": [], **extra, **over.get(c, {})}
            for c in ci.expected_cells(transport)]


def test_ci_gate_fails_on_error_and_on_untracked_g_rel_1_fail():
    rows = full_set(**{"crash-after-rotation/rotation": {"verdict": "FAIL", "targets": [519, 549]},
                       "multi-session-one-process/in-place": {"verdict": "FAIL"},
                       "baseline/in-place/acp": {"verdict": "INCONCLUSIVE"}})
    assert ci.gate(rows, {549}) == []  # an open targeted issue, a cell outside G-REL-1, a non-FAIL
    assert len(ci.gate(rows, {1})) == 1
    err = full_set(**{"long-80/in-place": {"verdict": "ERROR", "reason": "x"}, "crash-after-rotation/rotation": {}})
    assert len(ci.gate(err, {549})) == 1
    assert len(ci.gate(full_set(**{"baseline/rotation/acp": {"verdict": "FAIL"}}), {549})) == 1
    assert all(h["sha"] and h["python_version"] for h in json.loads(ci.CI_HOSTS.read_text())["hosts"].values())


def test_unverified_host_never_executes_its_interpreter(tmp_path):
    """Regression (R2a review): the anthropic-SDK probe ran the host python before the identity check."""
    marker, venv = tmp_path / "ran", tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text(f"#!/bin/sh\ntouch {marker}\n")
    (venv / "bin" / "python").chmod(0o755)
    host = {"python": str(venv / "bin" / "python"), "src": str(tmp_path / "src"), "sha": "0" * 40}
    cell = next(c for c in PC.R2_CELLS if c.get("api") == "anthropic")
    plugin = {"ref": "HEAD", "sha": "1" * 40, "dir": "lcm", "tree": str(tmp_path)}
    rec = PC.run_cell_process(cell, "h", host, plugin, tmp_path / "out", 5, False, identity={"error": "sha mismatch"})
    assert rec["verdict"] == "ERROR" and "identity" in rec["reason"] and not marker.exists()
    assert not PC.has_dist(host["python"], "anthropic")
    (venv / "lib" / "python3.11" / "site-packages" / "anthropic-0.87.0.dist-info").mkdir(parents=True)
    (venv / "lib" / "python3.11" / "site-packages" / "anthropic-0.87.0.dist-info" / "METADATA").write_text("Name: anthropic\n")
    assert PC.has_dist(host["python"], "anthropic") and not marker.exists()


def test_colliding_suffixes_are_unrouted_and_unexpected(provider):
    """Regression (R2a.1 delta review): any path ending in /models was served as the catalog and accepted."""
    con = http.client.HTTPConnection("127.0.0.1", provider.port, timeout=10)
    for path in ("/rogue/models", "/v1/x/models", "/v1/models/unknown-id", "/v1/models/"):
        con.request("GET", path)
        resp = con.getresponse()
        assert resp.read() and resp.status == 404, path
    for method, path in (("POST", "/rogue/chat/completions"), ("POST", "/v1/x/messages")):
        con.request(method, path, "{}")
        resp = con.getresponse()
        assert resp.read() and resp.status == 404, path
    d = provider.log_path.parent
    (d / "provider-requests.jsonl").write_text(provider.log_path.read_text())
    acct = PC.accounting(d, [])
    assert [u["path"] for u in acct["unexpected_requests"]] == ["/rogue/models", "/v1/x/models", "/v1/models/unknown-id",
                                                                "/v1/models/", "/rogue/chat/completions", "/v1/x/messages"]
    forged = {"rid": 99, "method": "GET", "path": "/rogue/models", "route": "models"}  # a log line cannot vouch for itself
    (d / "provider-requests.jsonl").write_text(json.dumps(forged) + "\n")
    assert not PC.accounting(d, [])["ok"]


def test_ci_gate_requires_a_complete_result_set():
    """Regression (R2a.3, #570 thread): an empty or incomplete result set passed the gate."""
    r1, r2 = full_set(), full_set("acp-process", **{"gateway-second-restart/in-place": {"verdict": "UNSUPPORTED"}})
    assert ci.gate(r1 + r2, set()) == [] and len(r2) == len(r1) + 1  # UNSUPPORTED counts as present
    assert ci.gate([], set()) == ["empty result set"]
    for broken in (r1[1:], r1 + r1[:1], r1 + [dict(r1[0], cell="made-up/cell")], r2[:-1] + r1):
        problems = ci.gate(broken, set())
        assert len(problems) == 1 and problems[0].startswith("INCOMPLETE h"), problems


def test_process_phase_deadline_is_an_error_with_its_cause(tmp_path):
    """R2a.3 item 7: run_matrix --timeout was unused for process cells; a host that never answers a prompt now
    ERRORs at the phase deadline instead of waiting out the per-request timeout."""
    silent = PEER.replace('if m["method"] == "session/prompt":', 'if m["method"] == "session/prompt":\n        continue')
    cell = next(c for c in cells.registry() if c["id"] == "baseline/in-place/acp")
    run = PC.ProcessCell(cell, tmp_path, {"src": str(tmp_path)}, "acp-process", 300.0, phase_timeout=1.5)
    run.work.mkdir(parents=True)
    run.argv, run.env = lambda: [sys.executable, "-c", silent], lambda: {"PATH": "/usr/bin:/bin"}  # no observer: tmp_path may be /tmp
    started = time.monotonic()
    try:
        last = run.run_phase(1)
    finally:
        run.proxy.stop()
        run.provider.server.server_close()
    assert time.monotonic() - started < 15
    assert last["exit"] == "error" and "PhaseDeadline" in last["reason"] and "1.5s deadline" in last["reason"], last


def test_process_phase_deadline_is_checked_before_a_done_return(tmp_path):
    """Regression (R2a.3 review): work after the last request could cross the deadline and still return done."""
    cell = {**next(c for c in cells.registry() if c["id"] == "baseline/in-place/acp"), "turns": 1,
            "final_compaction_check": False}
    run = PC.ProcessCell(cell, tmp_path, {"src": str(tmp_path)}, "acp-process", 300.0, phase_timeout=1.0)
    run.work.mkdir(parents=True)
    run.argv = lambda: [sys.executable, "-c", PEER]  # answers every request at once
    run.env = lambda: {"PATH": "/usr/bin:/bin"}  # no observer (it refuses a /tmp cell dir, as tmp_path is on Linux)
    event = run.event
    run.event = lambda **ev: (event(**ev), ev.get("event") == "acp_response" and time.sleep(1.3))
    try:
        last = run.run_phase(1)
    finally:
        run.proxy.stop()
        run.provider.server.server_close()
    assert any(json.loads(x).get("event") == "acp_response" for x in run.transcript.read_text().splitlines())
    assert last["exit"] == "error" and "PhaseDeadline" in last["reason"], last


def test_ci_gate_fails_a_missing_lane_or_host(tmp_path, capsys):
    """Regression (R2a.3 bot thread): the gate flattened its files, so an empty R2 file left no group to check."""
    r1, r2 = tmp_path / "r1.jsonl", tmp_path / "r2.jsonl"
    issues = tmp_path / "open-issues.txt"
    issues.write_text("")

    def run(a, b):
        r1.write_text("".join(json.dumps(r) + "\n" for r in a))
        r2.write_text("".join(json.dumps(r) + "\n" for r in b))
        rc = ci.main(["gate", str(r1), str(r2), "--open-issues", str(issues)])
        return rc, capsys.readouterr().out
    assert run(full_set(), full_set("acp-process"))[0] == 0
    rc, out = run(full_set(), [])
    assert rc == 1 and f"empty result file: {r2}" in out
    rc, out = run(full_set(), [dict(r, host="B") for r in full_set("acp-process")])
    assert rc == 1 and "has no rows for host(s) ['B']" in out and "has no rows for host(s) ['h']" in out
