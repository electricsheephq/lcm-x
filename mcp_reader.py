"""``lcm-x mcp``: a read-only stdio MCP server over ONE live Hermes LCM-X store (#1192).

Speaks newline-delimited JSON-RPC 2.0 (MCP stdio) and serves ``lcm_recall``,
``lcm_grep``, ``lcm_expand``, ``lcm_load_session`` and ``lcm_describe`` with the
production schemas and the production tool functions, called directly on a plain
engine shim (never ``LCMEngine``: its dispatch ingests). Every store object is a
``readonly_store`` view over ``mode=ro`` connections opened per tool call, so no
read snapshot outlives a call and nothing the reader does can write the store.

Launched by ``scripts/lcm_x_mcp.py`` in a dedicated process: the profile's
``LCM_*`` settings are loaded from ``<hermes-home>/.env`` into that process only,
and stdout carries nothing but protocol lines (everything else goes to a log file).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, MutableMapping, TextIO

from . import embedding_provider, retrieval_core, schemas, tokens, tools
from .command import _resolve_storage_dtype, _resolve_store_dim
from .config import LCMConfig
from .ingest_protection import embedding_provider_requires_privacy
from .readonly_store import (
    ReadOnlyLifecycleStateStore,
    ReadOnlyMessageStore,
    ReadOnlySummaryDAG,
    ReadOnlyVectorStore,
    SchemaRefusedError,
    StoreOfflineError,
    active_profile_hashes,
    derived_store_uuid,
    open_readonly_connection,
    probe_capabilities,
    store_uuid_for,
)
from .vector_store import _VECTOR_BYTEORDER, EmbeddingIdentity, VectorStore

logger = logging.getLogger(__name__)

SERVER_NAME = "lcm-x"
# Newest first; an unknown client version is answered with the newest we speak.
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
READER_TOOLS = {
    "lcm_recall": schemas.LCM_RECALL,
    "lcm_grep": schemas.LCM_GREP,
    "lcm_expand": schemas.LCM_EXPAND,
    "lcm_load_session": schemas.LCM_LOAD_SESSION,
    "lcm_describe": schemas.LCM_DESCRIBE,
}
READER_NOTE = (
    " Served read-only by lcm-x mcp, which has no active session: lcm_grep defaults to "
    "session_scope='all', a node_id without session_id resolves in the node's own session, "
    "externalized payloads resolve from any session of this store, and ids may carry a "
    "'<store_uuid>:' prefix."
)
PROFILE_ENV_EXTRA_KEYS = ("FASTEMBED_CACHE_PATH",)
TOKENIZER_WAIT_S = 30.0

_QUALIFIED_ID = re.compile(r"([0-9a-f]{32}):(.*)", re.DOTALL)
_SCALAR_ID_ARGS = ("store_id", "node_id", "session_id", "externalized_ref", "after_store_id")
_INTEGER_ID_ARGS = frozenset({"store_id", "node_id", "after_store_id"})
_LIST_ID_ARGS = ("seen_refs", "exclude_session_ids", "externalized_refs")


class ForeignStoreError(ValueError):
    """An input id is qualified with another store's uuid."""


# -- Profile configuration ------------------------------------------------------

def read_profile_env(hermes_home: str | Path) -> dict[str, str]:
    """Parse ONLY ``LCM_*`` (plus FASTEMBED_CACHE_PATH) assignments from ``<home>/.env``.

    The file also holds provider keys; every other line is skipped unparsed, so no other
    key is ever read into this process, logged or forwarded.
    """
    try:
        text = (Path(hermes_home) / ".env").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    found: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not (key.startswith("LCM_") or key in PROFILE_ENV_EXTRA_KEYS):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        found[key] = value
    return found


def apply_profile_environment(hermes_home: str | Path, environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Make this dedicated process's environment the profile's LCM view; return the keys set.

    The reader's own ``LCM_*`` values (``LCM_HERMES_BASE_DIR`` included) are dropped first:
    configuration comes from the profile, never from whoever launched the reader.
    """
    env = os.environ if environ is None else environ
    for key in [name for name in env if name.startswith("LCM_")]:
        del env[key]
    profile = read_profile_env(hermes_home)
    env.update(profile)
    env["HERMES_HOME"] = str(hermes_home)
    return sorted(profile)


def disabled_tool_names(environ: MutableMapping[str, str] | None = None) -> set[str]:
    """LCM_DISABLED_TOOLS, parsed as ``LCMEngine._disabled_tool_names`` does (engine.py is never imported)."""
    raw = (os.environ if environ is None else environ).get("LCM_DISABLED_TOOLS", "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def tool_definitions(disabled: set[str]) -> list[dict[str, Any]]:
    return [{"name": name, "description": schema.get("description", "") + READER_NOTE,
             "inputSchema": schema.get("parameters") or {"type": "object", "properties": {}}}
            for name, schema in READER_TOOLS.items() if name not in disabled]


# -- Engine shim ----------------------------------------------------------------

class ReaderEngine:
    """Exactly the engine attributes the five tools read; deliberately NOT an LCMEngine."""

    def __init__(self, *, config: LCMConfig, store: Any, dag: Any, hermes_home: str,
                 lifecycle: Any, provider_caches: dict[str, Any]) -> None:
        self._config = config
        self._store = store
        self._dag = dag
        self._hermes_home = hermes_home
        self.current_session_id = ""
        self.current_conversation_id = None
        self._lifecycle = lifecycle
        self._session_occurrence_dates: dict[str, Any] = {}
        # Product seams (tools.py): the KNN backend class and any-session payload reach.
        self._vector_store_cls = ReadOnlyVectorStore
        self._reader_any_session = True
        self._lcm_embedding_provider_cache = provider_caches.get("_lcm_embedding_provider_cache")
        self._lcm_proactive_provider_cache = provider_caches.get("_lcm_proactive_provider_cache")
        self._lcm_chunk_provider_cache = provider_caches.get("_lcm_chunk_provider_cache")

    def _rotation_predecessor_session_ids(self, session_id: str, max_hops: int = 32) -> list[str]:
        return []  # no current session; payload reach is any-session instead


def _unqualify(key: str, value: Any, store_uuid: str) -> Any:
    if not isinstance(value, str):
        return value
    match = _QUALIFIED_ID.fullmatch(value)
    if match is None:
        return value
    if match.group(1) != store_uuid:
        raise ForeignStoreError(
            f"foreign_store: {key} is qualified with store {match.group(1)}, "
            f"but this reader serves store {store_uuid}"
        )
    bare = match.group(2)
    if key in _INTEGER_ID_ARGS and re.fullmatch(r"-?\d+", bare):
        return int(bare)
    return bare


def unqualify_inputs(args: dict[str, Any], store_uuid: str) -> dict[str, Any]:
    """Accept bare or ``<store_uuid>:``-prefixed ids; refuse a prefix naming another store."""
    resolved = dict(args)
    for key in _SCALAR_ID_ARGS:
        if key in resolved:
            resolved[key] = _unqualify(key, resolved[key], store_uuid)
    for key in _LIST_ID_ARGS:
        if isinstance(resolved.get(key), list):
            resolved[key] = [_unqualify(key, item, store_uuid) for item in resolved[key]]
    return resolved


# -- Reader ---------------------------------------------------------------------

class StoreReader:
    """One store, five tools, per-call read-only connections."""

    def __init__(self, store_path: str | Path, hermes_home: str | Path, *, config: LCMConfig,
                 environ: MutableMapping[str, str] | None = None) -> None:
        self.store_path = Path(store_path).expanduser().resolve()
        self.hermes_home = str(Path(hermes_home).expanduser().resolve())
        self.config = copy.copy(config)
        self.config.database_path = str(self.store_path)
        self.disabled = disabled_tool_names(environ)
        self._provider: Any = None
        self._provider_problem = ""
        self._expected_identity: dict[str, str | None] = {}
        self._provider_caches: dict[str, Any] = {}
        self._last_uuid: tuple[str, str] | None = None
        self.tokenizer = "estimate"

    # -- startup --
    def prepare(self, *, tokenizer_wait_s: float = TOKENIZER_WAIT_S) -> None:
        self._prewarm_tokenizer(tokenizer_wait_s)
        self._prewarm_provider()

    def _prewarm_tokenizer(self, wait_s: float) -> None:
        started = time.monotonic()
        tokens._get_encoder()  # starts the background load; waits its own short bound
        thread = tokens._encoder_thread
        if thread is not None:
            thread.join(timeout=max(0.0, wait_s - (time.monotonic() - started)))
        self.tokenizer = self._tokenizer_state()

    @staticmethod
    def _tokenizer_state() -> str:
        return "exact" if tokens._encoder_ready and tokens._encoder is not None else "estimate"

    def _prewarm_provider(self) -> None:
        """Resolve and warm the query model WITHOUT downloading; record its full identity."""
        cfg = self.config
        if not bool(getattr(cfg, "embeddings_enabled", False)):
            return
        try:
            provider = embedding_provider.resolve_provider(cfg)
            if provider is None:
                raise embedding_provider.ProviderUnavailable("no embedding provider configured")
            if embedding_provider_requires_privacy(str(getattr(provider, "provider_id", ""))):
                # Cloud providers need keys the reader never loads.
                raise embedding_provider.ProviderUnavailable("cloud embedding provider")
            if isinstance(provider, embedding_provider.FastembedProvider):
                provider._ensure_local()  # local_files_only: a missing model is not downloaded
            vector = provider.embed_query("warmup")
        except Exception as exc:  # noqa: BLE001 - any failure degrades to full text
            self._provider_problem = "provider_unavailable"
            logger.warning("lcm-x mcp: embedding provider unavailable (%s)", type(exc).__name__)
            return
        dtype = _resolve_storage_dtype(cfg)
        dim = _resolve_store_dim(cfg, len(vector))
        # The same revision rule the writer's register_profile applies (prescreen tag).
        prescreen = SimpleNamespace(_write_binary_prescreen=bool(getattr(cfg, "embedding_binary_prescreen", False)))
        revision = VectorStore._prescreen_revision(prescreen, "", dtype=dtype)
        provider_id, model_id = str(provider.provider_id), str(provider.model_id)
        expected = {task: EmbeddingIdentity.canonical(provider_id, model_id, revision, dim, dtype,
                                                      _VECTOR_BYTEORDER, task).identity_hash
                    for task in ("summary", "chunk")}
        if embedding_provider.default_chunk_model(provider_id, model_id) != model_id:
            expected["chunk"] = None  # a separate chunk model is a cloud (Voyage) path: never matched
        self._expected_identity = expected
        self._provider = provider

    def _embedding_gate(self, conn: sqlite3.Connection) -> tuple[bool, list[str]]:
        """Vectors only when the store's active identity equals the reader's on every field."""
        if not bool(getattr(self.config, "embeddings_enabled", False)):
            return False, []
        if self._provider_problem or self._provider is None:
            return False, [self._provider_problem or "provider_unavailable"]
        active = active_profile_hashes(conn)
        if not active.get("summary"):
            return False, []  # no corpus; the capability flags say so
        if active["summary"] != self._expected_identity.get("summary") or (
            "chunk" in active and active["chunk"] != self._expected_identity.get("chunk")
        ):
            return False, ["embedding_identity_mismatch"]
        return True, []

    # -- per call --
    @contextlib.contextmanager
    def _open_call(self) -> Iterator[tuple[ReaderEngine, dict[str, Any]]]:
        conns: list[sqlite3.Connection] = []
        engine: ReaderEngine | None = None
        try:
            store_conn = open_readonly_connection(self.store_path)
            conns.append(store_conn)
            caps = probe_capabilities(store_conn, self.store_path)
            store_uuid, uuid_source = store_uuid_for(store_conn, self.store_path)
            self._last_uuid = (store_uuid, uuid_source)
            dag_conn = open_readonly_connection(self.store_path)
            conns.append(dag_conn)
            lifecycle = None
            if caps.lifecycle_state:
                lifecycle_conn = open_readonly_connection(self.store_path, row_factory=sqlite3.Row)
                conns.append(lifecycle_conn)
                lifecycle = ReadOnlyLifecycleStateStore(self.store_path, lifecycle_conn)
            vectors_on, degraded = self._embedding_gate(store_conn)
            config = self.config
            if bool(getattr(config, "embeddings_enabled", False)) and not vectors_on:
                config = copy.copy(config)
                config.embeddings_enabled = False
            engine = ReaderEngine(
                config=config,
                store=ReadOnlyMessageStore(self.store_path, store_conn, config=config, hermes_home=self.hermes_home,
                                           stem_usable=caps.stem, fts_usable=caps.fts_messages),
                dag=ReadOnlySummaryDAG(self.store_path, dag_conn, fts_usable=caps.fts_nodes,
                                       provenance_table=caps.summary_node_provenance),
                hermes_home=self.hermes_home, lifecycle=lifecycle, provider_caches=self._provider_caches,
            )
            if vectors_on:  # the cache key _lcm_grep_resolve_provider computes from this config
                engine._lcm_embedding_provider_cache = (
                    (str(getattr(config, "embedding_provider", "") or "").strip().lower(),
                     str(getattr(config, "embedding_model", "") or "").strip()),
                    self._provider,
                )
            yield engine, {"store_uuid": store_uuid, "store_uuid_source": uuid_source,
                           "degraded": degraded, "capabilities": caps.as_flags()}
        finally:
            if engine is not None:
                for name in ("_lcm_embedding_provider_cache", "_lcm_proactive_provider_cache",
                             "_lcm_chunk_provider_cache"):
                    self._provider_caches[name] = getattr(engine, name, None)
            for conn in conns:
                conn.close()  # never a checkpoint: these are mode=ro connections

    def _evict_pooled_vector_store(self) -> None:
        with retrieval_core._pool_lock:
            for key in [k for k in retrieval_core._vector_store_pool if k[0] == str(self.store_path)]:
                entry = retrieval_core._vector_store_pool.pop(key)
                with entry["lock"]:
                    entry["store"].close()

    @staticmethod
    def _apply_defaults(name: str, args: dict[str, Any], engine: ReaderEngine) -> dict[str, Any]:
        """Server-side defaults for a reader with no current session."""
        if name == "lcm_grep" and "session_scope" not in args:
            args["session_scope"] = "session" if args.get("session_id") else "all"
        if (name in {"lcm_expand", "lcm_describe"} and args.get("node_id") is not None
                and "session_id" not in args):
            try:
                node = engine._dag.get_node(args["node_id"])
            except (sqlite3.Error, TypeError, ValueError):
                node = None
            if node is not None and node.session_id:
                args["session_id"] = node.session_id
        return args

    def _uuid_for_envelope(self) -> tuple[str, str]:
        if self._last_uuid is not None:
            return self._last_uuid
        try:
            return derived_store_uuid(self.store_path), "derived"
        except OSError:
            return "", "unknown"

    def envelope(self, raw: str, context: dict[str, Any] | None) -> str:
        """Prefix the production JSON object with ``store_uuid`` and the reader flags.

        The production bytes follow the prefix unchanged, so stripping it restores them.
        """
        if context is None:
            store_uuid, source = self._uuid_for_envelope()
            context = {"store_uuid": store_uuid, "store_uuid_source": source, "degraded": []}
        reader: dict[str, Any] = {
            "read_only": True,
            "store_uuid_source": context["store_uuid_source"],
            "tokenizer": self._tokenizer_state(),
            "degraded": list(context["degraded"]),
        }
        if "capabilities" in context:
            reader["capabilities"] = context["capabilities"]
        prefix = json.dumps({"store_uuid": context["store_uuid"], "reader": reader})[:-1]
        if raw == "{}":
            return prefix + "}"
        if raw.startswith("{"):
            return prefix + ", " + raw[1:]
        return prefix + ", \"result\": " + json.dumps(raw) + "}"

    def call(self, name: Any, arguments: Any) -> tuple[str, bool]:
        """Run one tool; return ``(enveloped JSON text, is_error)``. Never raises."""
        started = time.monotonic()
        context: dict[str, Any] | None = None
        if name not in READER_TOOLS:
            raw = json.dumps({"error": f"Unknown LCM tool: {name}"})
        elif name in self.disabled:
            raw = json.dumps({"error": f"LCM tool {name} is disabled via LCM_DISABLED_TOOLS"})
        elif arguments is not None and not isinstance(arguments, dict):
            raw = json.dumps({"error": f"Invalid arguments for {name}: arguments must be a JSON object"})
        else:
            try:
                with self._open_call() as (engine, context):
                    args = unqualify_inputs(arguments or {}, context["store_uuid"])
                    args = self._apply_defaults(name, args, engine)
                    raw = getattr(tools, name)(args, engine=engine)
            except StoreOfflineError as exc:
                self._evict_pooled_vector_store()
                raw = json.dumps({"error": str(exc)})
            except (SchemaRefusedError, ForeignStoreError) as exc:
                raw = json.dumps({"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - a failure is this call's result, as in Hermes
                logger.exception("lcm-x mcp: %s failed", name)
                raw = json.dumps({"error": f"Error executing {name}: {exc}"}, ensure_ascii=False)
        try:
            is_error = "error" in json.loads(raw)
        except (TypeError, ValueError):
            is_error = False
        text = self.envelope(raw, context)
        args_hash = hashlib.sha256(json.dumps(arguments, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        logger.info("%s", json.dumps({"tool": str(name), "args_sha256": args_hash, "is_error": is_error,
                                      "latency_s": round(time.monotonic() - started, 4),
                                      "output_bytes": len(text.encode("utf-8"))}))
        return text, is_error


# -- JSON-RPC over stdio --------------------------------------------------------

def _error(ident: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}}


def handle(reader: StoreReader, request: Any) -> dict[str, Any] | None:
    """Answer one decoded JSON-RPC message; None for a notification."""
    if not isinstance(request, dict):
        return _error(None, -32600, "invalid request: expected a JSON object")
    ident = request.get("id")
    method = request.get("method")
    if not isinstance(method, str):
        return _error(ident, -32600, "invalid request: method must be a string")
    if "id" not in request:
        return None  # notifications (e.g. notifications/initialized) get no reply
    params = request.get("params")
    params = params if isinstance(params, dict) else {}
    if method == "initialize":
        requested = params.get("protocolVersion")
        version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        result: dict[str, Any] = {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": _plugin_version()},
        }
    elif method == "tools/list":
        result = {"tools": tool_definitions(reader.disabled)}
    elif method == "tools/call":
        text, is_error = reader.call(params.get("name"), params.get("arguments"))
        result = {"content": [{"type": "text", "text": text}], "isError": is_error}
    elif method == "ping":
        result = {}
    else:
        return _error(ident, -32601, f"method not found: {method}")
    return {"jsonrpc": "2.0", "id": ident, "result": result}


def serve(reader: StoreReader, stdin: TextIO, stdout: TextIO) -> None:
    for line in stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except ValueError:
            reply: dict[str, Any] | None = _error(None, -32700, "parse error")
        else:
            try:
                reply = handle(reader, request)
            except Exception:  # noqa: BLE001 - one bad message never ends the session
                logger.exception("lcm-x mcp: request handling failed")
                reply = _error(request.get("id") if isinstance(request, dict) else None,
                               -32603, "internal error")
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            stdout.flush()


def _plugin_version() -> str:
    try:
        text = (Path(__file__).resolve().parent / "plugin.yaml").read_text(encoding="utf-8")
        return re.search(r"^version:\s*['\"]?([^'\"\s]+)", text, re.MULTILINE).group(1)
    except (OSError, AttributeError):
        return "unknown"


def _isolate_stdio(log_path: Path) -> TextIO:
    """Keep the real stdout for protocol lines; send fds 1 and 2 to the log file."""
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)
    sys.stdout = sys.stderr = open(log_path, "a", encoding="utf-8", buffering=1)
    return protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lcm_x_mcp.py",
        description="Read-only MCP server (stdio) over one Hermes LCM-X store.",
    )
    parser.add_argument("--store", required=True, help="path to the profile's lcm.db")
    parser.add_argument("--hermes-home", required=True, help="the profile's HERMES_HOME")
    parser.add_argument("--log-dir", default="", help="directory for the server log (default: $TMPDIR)")
    options = parser.parse_args(argv)
    log_dir = Path(options.log_dir).expanduser() if options.log_dir else Path(tempfile.gettempdir())
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"lcm-x-mcp-{os.getpid()}.log"
    protocol = _isolate_stdio(log_path)
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    stdin = open(sys.stdin.fileno(), "r", encoding="utf-8", errors="replace", closefd=False)
    try:
        hermes_home = str(Path(options.hermes_home).expanduser().resolve())
        profile_keys = apply_profile_environment(hermes_home)
        config = LCMConfig.from_env()
        reader = StoreReader(options.store, hermes_home, config=config)
        reader.prepare()
        logger.info(
            "lcm-x mcp ready: profile LCM keys=%d tools=%d tokenizer=%s embeddings=%s",
            len(profile_keys), len(READER_TOOLS) - len(reader.disabled & set(READER_TOOLS)),
            reader.tokenizer, reader._provider_problem or ("on" if reader._provider else "off"),
        )
        serve(reader, stdin, protocol)
    except Exception:  # noqa: BLE001 - logged for the operator; the client sees the exit
        logger.exception("lcm-x mcp: fatal")
        return 1
    return 0
