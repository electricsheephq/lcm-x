"""Scratch-only gateway configuration, copied routing edits, and process control."""

import ast
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
import ipaddress
import json
from pathlib import Path
import sqlite3
import subprocess
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from . import sandbox


def aux_tasks(host_src) -> list[str]:
    path = Path(host_src) / "hermes_cli/config.py"
    tree = ast.parse(path.read_text())
    if any(isinstance(n, ast.ImportFrom) and n.module == "hermes_cli.config_defaults"
           and any(a.name == "DEFAULT_CONFIG" for a in n.names) for n in tree.body):
        tree = ast.parse(path.with_name("config_defaults.py").read_text())
    for node in tree.body:
        value = getattr(node, "value", None)
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if any(isinstance(t, ast.Name) and t.id == "DEFAULT_CONFIG" for t in targets) and isinstance(value, ast.Dict):
            for key, block in zip(value.keys, value.values):
                if isinstance(key, ast.Constant) and key.value == "auxiliary":
                    return [k.value for k, v in zip(block.keys, block.values)
                            if isinstance(k, ast.Constant) and isinstance(v, (ast.Dict, ast.Call))]
    raise ValueError("DEFAULT_CONFIG auxiliary task blocks not found")


def _local_url(url):
    parsed = urlsplit(url)
    if parsed.scheme != "http" or not (parsed.hostname == "localhost" or
                                      ipaddress.ip_address(parsed.hostname).is_loopback):
        raise ValueError("provider URL must use loopback HTTP")
    return url


def build_config(settings, *, provider_url, home, aux, context_length, stored_route_providers=()) -> dict:
    provider_url = _local_url(provider_url)
    provider = {"base_url": provider_url, "api_key": "d5-dummy", "api_mode": "chat_completions",
                "context_length": context_length, "models": {m: {"context_length": context_length}
                for m in ("d5-agent", "d5-aux", "d5-cron", "d5-summary")}}
    return {"model": {"default": "d5-agent", "provider": "d5", "context_length": context_length},
            "custom_providers": [{"name": n, **deepcopy(provider)} for n in dict.fromkeys(("d5", *stored_route_providers))],
            "auxiliary": {t: {"provider": "d5", "model": "d5-aux", "base_url": provider_url} for t in aux},
            "context": {"engine": "lcm-x"}, "compression": deepcopy(settings["compression"]),
            "lcm": {**deepcopy(settings["lcm"]), "summary_model": "d5-summary"},
            "plugins": {"enabled": ["hermes-lcm-x", "d5-local"], "disabled": [], "isolation": "in_process"},
            "platforms": {"d5-local": {"enabled": True, "extra": {}}},
            "gateway": {"message_timestamps": {"enabled": False}},
            "mcp_servers": {}, "auth": {"adopt_external_logins": False},
            "telemetry": {"shared_metrics": {"enabled": False, "send": False}},
            "terminal": {"cwd": str(Path(home) / "workspace")}}


def write_config(home, config):
    import yaml
    Path(home).mkdir(parents=True, exist_ok=True)
    (Path(home) / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def build_env(lcm_env, *, home, provider_url, host_bin, port_file, lcm_tree):
    """Prepare the isolated layout; return (child env, declaration of key names). ``lcm_tree`` is the arm's LCM-X
    checkout (the rc or the previous GA), never this instrument's own tree."""
    home = Path(home).resolve()
    if not (Path(lcm_tree) / "plugin.yaml").is_file():
        raise ValueError("lcm_tree is not an LCM-X plugin tree")
    for name in ("user-home", "tmp", "workspace", "plugins", "sessions"):
        (home / name).mkdir(parents=True, exist_ok=True)
    for name, target in (("hermes-lcm-x", Path(lcm_tree).resolve()), ("d5-local", Path(__file__).with_name("d5_local"))):
        link = home / "plugins" / name
        if not link.is_symlink():
            link.symlink_to(target, target_is_directory=True)
        elif link.resolve() != target.resolve():
            raise ValueError("scratch plugin link has a different target")
    remote = str(lcm_env.get("LCM_EMBEDDING_PROVIDER") or "").strip().lower() not in ("", "none", "fastembed", "fast-embed", "local", "ollama")
    dropped = sorted(k for k in lcm_env if k.startswith("LCM_EMBEDDING") and remote)
    env = {k: str(v) for k, v in lcm_env.items() if k.startswith("LCM_") and k not in dropped and v is not None}
    env.update(PATH=f"{host_bin}:/usr/bin:/bin", HOME=str(home / "user-home"), HERMES_HOME=str(home),
               TMPDIR=str(home / "tmp"), D5_LOCAL_PORT_FILE=str(port_file), D5_LOCAL_ALLOW_ALL_USERS="1",
               HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9", ALL_PROXY="http://127.0.0.1:9",
               NO_PROXY="127.0.0.1,localhost,::1", OPENAI_API_KEY="d5-dummy", OPENAI_BASE_URL=_local_url(provider_url),
               LCM_DATABASE_PATH=str(home / "lcm.db"), LCM_HERMES_BASE_DIR=str(home),
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    return env, {"env_names": sorted(env), "dropped_names": dropped,
                 "overridden_names": ["LCM_DATABASE_PATH", "LCM_HERMES_BASE_DIR"]}


def session_key(chat_id) -> str:
    return f"agent:main:d5-local:dm:{chat_id}"


def _route(route, url, providers):
    for key, value in list(route.items()):
        if isinstance(value, dict) and key in ("gateway_runtime", "browser_model_lock", "model_override", "route"):
            _route(value, url, providers)
        elif key in ("provider", "base_url", "api_mode", "model", "model_lock", "model_locked"):
            if key == "provider" and value:
                providers.add(value)
            route[key] = {"provider": "d5", "base_url": url, "api_mode": "chat_completions",
                          "model": "d5-agent", "model_lock": False, "model_locked": False}[key]
    route.update(provider="d5", base_url=url, api_mode="chat_completions")


def edit_copy_routing(host_db, conversations, *, sessions_dir, run_dir, provider_url) -> dict:
    """Normalize only an ARM COPY; provider_url is the caller's bound replay URL."""
    path = Path(host_db).resolve()
    if not path.is_relative_to(Path(run_dir).resolve()):
        raise ValueError("routing edits require an ARM COPY inside run_dir")
    provider_url = _local_url(provider_url)
    providers, changes = set(), []
    with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)) as db, db:
        db.row_factory = sqlite3.Row
        columns = {r[1] for r in db.execute("PRAGMA table_info(sessions)")}
        for conversation in conversations:
            sid, chat = conversation["session_id"], conversation["chat_id"]
            row = db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
            if row is None:
                raise ValueError("copied session missing")
            candidates = [json.loads(r[0]) for r in db.execute("SELECT entry_json FROM gateway_routing ORDER BY scope, session_key")]
            entry = next((e for e in candidates if e.get("session_id") == sid), None)
            stamp = datetime.fromtimestamp(row["started_at"], timezone.utc).isoformat()
            entry = entry or {"session_id": sid, "created_at": stamp, "updated_at": stamp, "metadata": {}}
            original = deepcopy(entry)
            existing = db.execute("SELECT entry_json FROM gateway_routing WHERE scope=? AND session_key=?",
                                  (str(Path(sessions_dir).resolve()), session_key(chat))).fetchone()
            if existing and json.loads(existing[0]).get("session_id") != sid:
                raise ValueError("copied routing destination belongs to another session")
            origin = {**(entry.get("origin") or {}), "platform": "d5-local", "chat_id": chat,
                      "chat_name": chat, "chat_type": "dm", "user_id": "d5-user", "user_name": "d5-user"}
            entry.update(session_key=session_key(chat), platform="d5-local", chat_type="dm", origin=origin)
            for key in ("model_override", "model_config", "gateway_runtime", "browser_model_lock"):
                if isinstance(entry.get(key), dict):
                    _route(entry[key], provider_url, providers)
            if any(k in entry for k in ("provider", "base_url", "api_mode", "model")):
                _route(entry, provider_url, providers)
            route = json.loads(row["model_config"] or "{}") if "model_config" in columns else {}
            _route(route, provider_url, providers)
            updates = {"session_key": session_key(chat), "chat_id": chat, "origin_json": json.dumps(origin),
                       "model": "d5-agent", "model_config": json.dumps(route)}
            updates = {k: v for k, v in updates.items() if k in columns}
            db.execute("UPDATE sessions SET " + ",".join(f"{k}=?" for k in updates) + " WHERE id=?", (*updates.values(), sid))
            db.execute("INSERT INTO gateway_routing (scope,session_key,entry_json,updated_at) VALUES (?,?,?,?) ON CONFLICT(scope,session_key) DO UPDATE SET entry_json=excluded.entry_json,updated_at=excluded.updated_at",
                       (str(Path(sessions_dir).resolve()), session_key(chat), json.dumps(entry), time.time()))
            changes.append({"name": conversation["name"], "session_fields": sorted(k for k in updates if row[k] != updates[k]),
                            "routing_fields": sorted(k for k in entry if entry[k] != original.get(k))})
    return {"sessions": len(changes), "routing_rows": len(changes), "changes": changes, "stored_route_providers": sorted(providers)}


class Gateway:
    def __init__(self, host_bin, home, env, run_dir, *, provider_port, allow_unsandboxed=False):
        """The gateway may reach only the provider's loopback port and unix sockets inside run_dir."""
        self.host_bin, self.home, self.env, self.run_dir = Path(host_bin), Path(home), dict(env), Path(run_dir)
        self.provider_port = provider_port
        self.allow_unsandboxed, self.process, self.exit_code, self.port = allow_unsandboxed, None, None, None
        self.port_file, self.starts = Path(env["D5_LOCAL_PORT_FILE"]), 0

    def _request(self, path, data=None, timeout=5):
        request = Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(data).encode() if data is not None else None,
                          headers={"Content-Type": "application/json"})
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            return json.load(response)

    def start(self, timeout):
        if self.process and self.process.poll() is None:
            raise RuntimeError("gateway already running")
        if not self.home.resolve().is_relative_to(self.run_dir.resolve()) or not self.port_file.resolve().is_relative_to(self.run_dir.resolve()):
            raise ValueError("gateway home and port file must be inside run_dir")
        argv = [str(self.host_bin / "hermes"), "gateway", "run", "--no-supervise"]
        if sandbox.available():
            argv = sandbox.wrap(argv, ports=[self.provider_port], socket_roots=[self.run_dir])
        elif not self.allow_unsandboxed:
            raise RuntimeError("sandbox unavailable; refusing unsandboxed gateway")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        previous = self.port_file.stat().st_mtime_ns if self.port_file.exists() else None
        self.starts += 1
        with (self.run_dir / f"gateway-{self.starts}.out").open("xb") as output:
            self.process = subprocess.Popen(["/usr/bin/nice", "-n", "19", *argv], env=self.env,
                                            cwd=self.home, stdout=output, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self.process.poll() is None:
            try:
                if self.port_file.stat().st_mtime_ns != previous:
                    self.port = int(self.port_file.read_text())
                    if self._request("/health", timeout=min(1, timeout)).get("connected") is True:
                        return self
            except (OSError, ValueError):
                pass
            time.sleep(0.05)
        self.stop(min(timeout, 5))
        raise RuntimeError(f"gateway failed readiness; exit_code={self.exit_code}")

    def turn(self, chat_id, step, text, timeout=600):
        return self._request("/turn", {"chat_id": chat_id, "step": step, "text": text, "timeout": timeout}, timeout + 1)

    def stop(self, timeout=10):
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.exit_code = self.process.wait()
        return self.exit_code

    def pids(self):
        if not self.process or self.process.poll() is not None:
            return []
        rows = subprocess.check_output(["ps", "-A", "-o", "pid=,ppid="], text=True)
        pairs, found = [tuple(map(int, line.split())) for line in rows.splitlines()], {self.process.pid}
        while (children := {pid for pid, parent in pairs if parent in found} - found):
            found.update(children)
        return sorted(found)


def nonlocal_connections(pids) -> int:
    if not pids:
        return 0
    result = subprocess.run(["lsof", "-a", "-p", ",".join(map(str, pids)), "-i", "TCP", "-n", "-P"], capture_output=True, text=True)
    count = 0
    for line in result.stdout.splitlines():
        if "->" in line:
            remote = line.split("->", 1)[1].split()[0].rsplit(":", 1)[0].strip("[]")
            count += remote not in ("127.0.0.1", "::1")
    return count
