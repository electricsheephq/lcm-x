"""Compact real-engine cursor traces; regenerate deliberately with this CLI.

The frozen data inventory is independent of source discovery. Observers wrap
store calls transparently (one real call, unchanged result), and SQL callbacks
include reads and transaction boundaries. Snapshot reads happen outside tracing.
Only sandbox paths, generated clocks/nonces and process identities normalize.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib
import json
import logging
import os
import re
import sqlite3
import sys
import tempfile
import time
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import pytest

# Differential runs use this identical harness with an independently selected
# product tree. Register that tree before the suite's directory-loader runs.
if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--repo")
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.repo == "":
        pre_parser.error("argument --repo: expected a non-empty path")
    selected = Path(pre_args.repo).resolve() if pre_args.repo is not None else Path(__file__).resolve().parents[2]
    package = ModuleType("hermes_lcm")
    package.__path__ = [str(selected)]
    sys.modules["hermes_lcm"] = package

import tests.conftest  # noqa: E402, F401
from tests.phase0_hygiene import product_env_keys
from tests.harness import engine_context, pin_char_counter  # noqa: E402

reconcile: Any = importlib.import_module("hermes_lcm.reconcile")
ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "fixtures" / "reconcile_trace.json"
TARGETS = ("_find_reconciled_cursor_for_store_tail", "_reconcile_ingest_cursor_from_store")
# Reviewed engine-run data, including indirect reconcile helpers and external
# helper side effects. New data lookups must fail completeness before recording.
OBSERVED_ATTRIBUTES = frozenset({
    "_compiled_ignore_message_patterns", "_compress_commit_proof", "_compress_occurrences",
    "_compress_run", "_config", "_conversation_id", "_current_compress_placeholder_identity_counts",
    "_dag", "_generated_ignored_active_replay_placeholder_message_ids", "_hermes_home",
    "_host_rewrite_overrides", "_host_rewrite_store", "_host_rewrite_watch",
    "_last_compacted_store_id", "_last_emission_descriptors", "_last_ingest_reconciliation",
    "_lifecycle", "_retired_durable_proof", "_session_id", "_store",
    "_generated_ignored_active_replay_placeholder_hashes",
    "_generated_ignored_active_replay_placeholder_hash_counts",
    "_generated_ignored_active_replay_placeholder_hash_ordinals",
    "_out_of_band_durable_rows",
    "_foreground_budget", "_ingest_cursor", "_consecutive_ingest_failures",
    "_pruned_tool_source_generation", "_pruned_tool_source_identities", "_pruned_tool_source_store",
})
Trace = list[str]
Message = dict[str, Any]
NOTE: Message = {"role": "system", "content": (
    "[Note: This conversation uses Lossless Context Management (LCM). "
    "Earlier turns have been compacted into hierarchical summaries below.]")}


TRACE_GUIDANCE = (
    "Regenerate from repo root: python -m tests.behaviour.reconcile_trace_harness --repo . --regenerate\n"
    "Attribute every changed event in the PR description to its commit or behaviour change.\n"
    "See docs/process/turn-path-probe-packet.md#golden-traces."
)


def encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fingerprint(value: str) -> str:
    return f"{hashlib.sha256(value.encode()).hexdigest()[:12]}/{len(value)}"


def compact(value: Any) -> str:
    text = encoded(value)
    return text if len(text) <= 110 else "sha12/len=" + fingerprint(text)


def engine_attributes(source: str) -> set[str]:
    """AST closure of self lookups, including literal getattr/setattr/hasattr.

    Traverse methods declared on ReconcileMixin only, plus statically named
    ReconcileMixin calls. Other mixins are the explicit external boundary.
    Callable lookups are control, never observed private helper qualnames.
    """
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ReconcileMixin")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    pending, visited, attributes = list(TARGETS), set(), set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        for node in ast.walk(methods[name]):
            attr = None
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id in {"self", "ReconcileMixin"}:
                    attr = node.attr
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in {"getattr", "setattr", "hasattr"} and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Name) and node.args[0].id == "self"):
                key = node.args[1]
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    raise AssertionError(
                        "Dynamic self lookup: update engine_attributes and OBSERVED_ATTRIBUTES.\n"
                        + TRACE_GUIDANCE)
                attr = key.value
            if attr is not None:
                if attr in methods:
                    pending.append(attr)
                else:
                    engine: Any = importlib.import_module("hermes_lcm.engine").LCMEngine
                    if not callable(getattr(engine, attr, None)):
                        attributes.add(attr)
    return attributes


@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str
    shape: str = "turns"
    mode: str = "restart"
    variant: str = "full"
    proof: str = ""
    count: int = 4
    allow_empty: bool = True
    session_end: bool = False
    fault: str = ""


SCENARIOS = (
    Scenario("no-session", variant="no-session"),
    Scenario("no-messages", variant="empty"),
    Scenario("empty-store", count=0),
    Scenario("count-error", fault="get_session_count"),
    Scenario("tail-error", fault="get_session_tail"),
    Scenario("head-error", variant="new", fault="get_session_messages"),
    Scenario("metadata-error", fault="read_metadata_json"),
    *[Scenario(f"matcher-boundary-{variant}", mode="matcher", variant=variant)
      for variant in ("no-tail-rows", "invalid-tail-json", "no-tail")],
    *[Scenario(f"bounded-tail-{n}", count=n, variant="last-pair") for n in (63, 64, 65, 80)],
    *[Scenario(f"{shape}-{variant}", shape=shape, variant=variant)
      for shape in ("turns", "system", "tools", "lossy", "lossy-arguments", "oob", "todo", "whitespace")
      for variant in ("full", "prefix", "suffix", "new", "delta", "changed")],
    *[Scenario(f"singleton-{role}-{mode}", shape=role, count=1, mode=mode)
      for role in ("user", "assistant", "system", "tool", "externalized", "marker", "quarantine")
      for mode in ("matcher", "restart")],
    *[Scenario(f"matcher-{variant}-{allow}", mode="matcher", variant=variant, allow_empty=allow)
      for variant in ("empty", "full", "new", "alias", "equal", "scaffold", "objective")
      for allow in (False, True)],
    *[Scenario(f"proof-{proof}-{variant}-{end}", proof=proof, variant=variant, session_end=end)
      for proof in ("native", "engine", "end") for variant in ("suffix", "changed") for end in (False, True)],
    *[Scenario(f"empty-native-{variant}", count=0, proof="native", variant=variant)
      for variant in ("full", "delta", "changed")],
    *[Scenario(f"commit-v{version}-{variant}", proof=f"v{version}", variant=variant)
      for version in (2, 3, 4) for variant in ("full", "delta", "changed", "empty-child")],
    *[Scenario(f"placeholder-{variant}", shape="placeholder", count=0, variant=variant)
      for variant in ("full", "gap", "duplicate", "exhausted", "new")],
    *[Scenario(f"carry-{variant}", count=0, proof="carry", variant=variant)
      for variant in ("full", "write-error")],
    *[Scenario(f"persisted-{variant}", shape="marker", variant=variant)
      for variant in ("full", "missing", "replaced", "generationless", "vanish", "appear")],
    Scenario("ignore-collapsed", shape="ignored"),
    Scenario("ignore-placeholder", shape="ignored", variant="placeholder"),
    Scenario("stale-no-overlap", shape="system", count=80, variant="prefix"),
    Scenario("cold-replay", mode="public"),
    Scenario("cold-replay-new-turn", mode="public", variant="delta"),
    Scenario("tool-middle-window", shape="tools", variant="middle"),
    Scenario("tool-user-after-anchor", shape="tools", variant="user-after"),
    Scenario("tool-reused-id-new-result", shape="tools", variant="changed-result"),
    Scenario("tool-missing-result", shape="tools", variant="missing-result"),
    Scenario("tool-duplicate-result", shape="tools", variant="duplicate-result"),
    Scenario("host-rewrite-head", shape="whitespace", variant="rewrite"),
    Scenario("commit-proof-extends-tail", proof="v3", variant="proof-extends"),
)


class Canonical:
    def __init__(self, root: Path, messages: list[Message]) -> None:
        self.root = str(root)
        self.identities = {id(row): f"message-{i}" for i, row in enumerate(messages)}
        self.nonces: dict[str, str] = {}

    def text(self, text: str, *, identity: bool = False) -> str:
        text = text.replace(self.root, "<sandbox>")
        # Only structurally identified message UIDs are generated nonces.
        # Content, SQL literals and durable numeric store IDs remain observable.
        if identity and re.fullmatch(r"[0-9a-f]{32}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", text):
            return self.nonces.setdefault(text, f"<nonce-{len(self.nonces)}>")
        # Filesystem generations are generated by the OS, not the replay bytes.
        return re.sub(r"(mtime_ns|ctime_ns)=\d+", r"\1=<file-clock>", text)

    def value(self, value: Any, path: str = "", *, in_payload: bool = False) -> Any:
        # Once inside message data, reserved-looking keys are ordinary content.
        in_payload = in_payload or path in {"content", "payload"}
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, (str, Path)):
            return self.text(str(value), identity=not in_payload and path in {
                "message_uid", "_absorbed_message_uids", "_tool_call_uids", "_tool_call_uid"})
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {f.name: self.value(getattr(value, f.name), f.name, in_payload=in_payload)
                    for f in dataclasses.fields(value)}
        if isinstance(value, dict):
            # UID maps use call IDs as keys and may hold lists for reused calls.
            return {self.identities.get(k, str(k)) if isinstance(k, int) else self.text(str(k)):
                    "<file-clock>" if k in {"mtime_ns", "ctime_ns"} else self.value(v,
                        path if path in {"_absorbed_message_uids", "_tool_call_uids"} else str(k),
                        in_payload=in_payload)
                    for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.value(v, path, in_payload=in_payload) for v in value]
        if isinstance(value, (set, frozenset)):
            return sorted((self.identities.get(v, v) if isinstance(v, int) else self.value(v, path, in_payload=in_payload)
                           for v in value), key=encoded)
        if isinstance(value, re.Pattern) or hasattr(value, "pattern"):
            return {"pattern": value.pattern}
        if hasattr(value, "__dict__") and not callable(value):
            return self.value(vars(value), path, in_payload=in_payload)
        raise TypeError(f"Unclassified trace value {path}: {type(value).__name__}")


def rows(case: Scenario, root: Path) -> list[Message]:
    if case.shape in {"user", "assistant", "system", "tool", "externalized", "marker", "quarantine"} and case.count == 1:
        role = case.shape if case.shape in {"user", "assistant", "system", "tool"} else "tool"
        result: list[Message] = [{"role": role, "content": "singleton result"}]
    else:
        result = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"row {i}: العربية 🪷"}
                  for i in range(case.count)]
    if case.shape == "system" and result:
        result[0]["role"] = "system"
    if case.shape in {"tools", "lossy-arguments"}:
        result = [{"role": "user", "content": "inspect"},
                  {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function",
                   "function": {"name": "inspect", "arguments": '{"path":"synthetic"}'}}]},
                  {"role": "tool", "tool_call_id": "call-1", "name": "inspect", "content": "durable output"},
                  {"role": "assistant", "content": "done"}]
        if case.shape == "lossy-arguments":
            result[1]["tool_calls"][0]["function"]["arguments"] = (
                "[LCM sensitive redaction: name=password_assignment; chars=8; bytes=8]")
    if case.shape == "lossy":
        result[0]["content"] = "[LCM sensitive redaction: name=password_assignment; chars=8; bytes=8]"
    if case.shape == "oob":
        result[-1] = {"role": "tool", "tool_call_id": "call-1", "content": (
            "result\n[OUT-OF-BAND USER MESSAGE]keep this instruction[/OUT-OF-BAND USER MESSAGE]")}
    if case.shape == "todo":
        result[0]["content"] = "[Your active task list was preserved across context compression]\n- [>] 1. inspect (in_progress)"
    if case.shape == "whitespace":
        result[0]["content"] = "  user spacing\n\n"
    if case.shape == "ignored":
        result[1]["content"] = "IGNORE THIS ROW"
    if case.shape == "externalized":
        externalize: Any = importlib.import_module("hermes_lcm.externalize")
        result[0]["content"] = externalize._build_externalized_placeholder({
            "kind": "tool_result", "tool_name": "inspect", "tool_call_id": "call-1",
            "ref": "a" * 64, "content_chars": 100, "content_bytes": 100})
    if case.shape == "quarantine":
        result[0] = {"role": "assistant", "content": (
            "[LCM sensitive redaction: kind=quarantined_assistant_output; reason=ignored; "
            "scope=ignored_message_pattern; field=content; chars=4; bytes=4; sha256=0123456789abcdef]")}
    if case.shape == "marker":
        payload = root / "hermes-results" / "payload.txt"
        payload.parent.mkdir(exist_ok=True)
        payload.write_text("line one\nline two\n" * 40)
        os.utime(payload, (1700000000, 1700000000))
        result[-1] = {"role": "tool", "tool_call_id": "call-1", "content": (
            f"<persisted-output>\nOutput too large (720 characters).\nFull output saved to: {payload}\n\n"
            "Preview (first 18 chars):\nline one\nline two\n</persisted-output>")}
        if case.variant == "generationless":
            result[-1]["content"] = result[-1]["content"].replace("</persisted-output>",
                "[LCM persisted-output file generation: size=720; mtime_ns=1; ctime_ns=1]\n</persisted-output>")
    for i, row in enumerate(result):
        row["timestamp"] = 100.0 + i
    return result


def seed_proof(engine: Any, case: Scenario, incoming: list[Message]) -> None:
    if case.proof == "native":
        engine._remember_native_recovery_replay_snapshot_digest(engine._native_recovery_replay_snapshot_digest(incoming))
    elif case.proof == "engine":
        engine._remember_compacted_active_replay_snapshot(incoming)
    elif case.proof == "end":
        engine._remember_session_end_replay_snapshot("S", incoming)
    elif case.proof.startswith("v") or case.proof == "carry":
        version = int(case.proof[1:]) if case.proof.startswith("v") else 3
        identities = [engine._message_replay_identity(m, strip_carrier=False) for m in incoming]
        if version != 2:
            identities = [reconcile._proof_user_identity(i) for i in identities]
        payload = {"version": version, "hermes_home": str(engine._hermes_home), "session_id": "S",
                   "conversation_id": "trace-conversation", "created_at": 1700001000.0,
                   "effective_sha256": [reconcile._commit_proof_identity_digest(i) for i in identities],
                   "last_store_id": engine._commit_proof_last_store_id(),
                   "carry_ranges": [["parent", 0, 2]] if case.proof == "carry" else []}
        if version == 4:
            payload.update(engine._emission_binding())
            payload["emissions"] = []
        engine._store.write_metadata_json(["compaction_commit_proof:S"], encoded(payload))


class Logs(logging.Handler):
    def __init__(self, events: Trace, canonical: Canonical) -> None:
        super().__init__()
        self.events, self.canonical = events, canonical

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append(f"log {record.levelname} {self.canonical.text(record.getMessage())}")


class LiteralPattern:
    """The configured pattern boundary without an optional regex dependency."""

    pattern = "IGNORE THIS ROW"

    def search(self, text: str, timeout: float | None = None) -> bool:
        return self.pattern in text


def snapshot(engine: Any, canonical: Canonical) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name in sorted(OBSERVED_ATTRIBUTES):
        value = getattr(engine, name, {"absent": True})
        if name == "_store":
            value = {table: [tuple(row) for row in engine._store._conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
                     for table in ("messages", "metadata")}
        elif name == "_dag":
            value = engine._dag.get_session_nodes("S", limit=10000)
        elif name == "_lifecycle":
            value = engine._lifecycle.get_by_conversation("trace-conversation")
        elif name == "_host_rewrite_store":
            # Owned SQLite service: its durable rows are observed via SQL and
            # MessageStore snapshots, never serialize a connection or lock.
            value = {"present": value is not None}
        state[name] = canonical.value(value, name)
    return state


def install_clock(patch: pytest.MonkeyPatch) -> None:
    clock = SimpleNamespace(**{name: getattr(time, name) for name in dir(time) if not name.startswith("__")})
    clock.time = lambda: 1700001000.0
    clock.monotonic = clock.perf_counter = lambda: 1000.0
    clock.time_ns = lambda: 1700001000000000000
    clock.gmtime = lambda: time.gmtime(1700001000)
    for name in ("engine", "store", "dag", "db_bootstrap", "lifecycle_state", "compaction", "externalize", "reconcile"):
        module = importlib.import_module("hermes_lcm." + name)
        if hasattr(module, "time"):
            patch.setattr(module, "time", clock)


def observe(case: Scenario, root: Path) -> Trace:
    root.mkdir(parents=True, exist_ok=True)
    events: Trace = []
    with pytest.MonkeyPatch.context() as patch:
        pin: Any = pin_char_counter
        context: Any = engine_context
        pin(patch)
        token_module = importlib.import_module("hermes_lcm.tokens")
        counter = token_module._count_tokens_cached

        def count(text: str, generation: int) -> int:
            # The sandbox's generated path length must not alter stored token
            # counts. All non-path payload bytes use the shared offline counter.
            return int(counter(text.replace(str(root), "<sandbox>"), generation))

        patch.setattr(token_module, "_count_tokens_cached", count)
        install_clock(patch)
        patch.setenv("HERMES_HOME", str(root / "home"))
        protection = importlib.import_module("hermes_lcm.ingest_protection")
        patch.setattr(protection.tempfile, "gettempdir", lambda: str(root))
        settings: dict[str, Any] = dict(embeddings_enabled=False, temporal_rollups_enabled=False,
            empty_lifecycle_gc_enabled=False, large_output_externalization_enabled=False,
            large_output_active_replay_stubbing_enabled=False,
            large_output_externalization_path=str(root / "home" / "payloads"))
        if case.shape == "ignored":
            settings["ignore_message_patterns"] = ["IGNORE THIS ROW"]
        with context(root, window=200000, conversation_id="trace-conversation", **settings) as engine:
            if case.shape == "ignored":
                engine._compiled_ignore_message_patterns = [LiteralPattern()]
            stored = rows(case, root)
            incoming = [dict(m) for m in stored] or [{"role": "user", "content": "fresh"}]
            if stored:
                if case.mode == "public" or case.shape == "marker":
                    engine.ingest(stored)
                    assert engine._consecutive_ingest_failures == 0, "scenario ingest setup failed"
                else:
                    engine._store.append_batch("S", stored, conversation_id="trace-conversation")
            if case.variant == "suffix":
                incoming = incoming[-1:]
            elif case.variant in {"prefix", "middle"}:
                incoming = incoming[:2] if case.variant == "prefix" else incoming[1:3]
            elif case.variant == "last-pair":
                incoming = incoming[-2:]
            elif case.variant == "new":
                incoming = [{"role": "user", "content": "new request"}]
            elif case.variant == "alias":
                incoming = [incoming[0], incoming[0]]
            elif case.variant == "equal":
                incoming = [dict(incoming[0]), dict(incoming[0])]
            elif case.variant == "scaffold":
                incoming = [dict(NOTE)]
            elif case.variant == "objective":
                incoming = [{"role": "user", "content": "[Current user objective preserved from compacted history]\ninspect"}]
            seed_proof(engine, case, incoming)
            if case.variant == "empty-child":
                engine._store._conn.execute("DELETE FROM messages WHERE session_id='S'")
                engine._store._conn.commit()
            if case.shape == "placeholder":
                placeholder = engine._ignored_active_replay_placeholder("ignored text")
                digest = engine._active_replay_placeholder_digest(placeholder)
                engine._remember_generated_ignored_placeholder_hash(digest)
                engine._write_generated_ignored_placeholder_hash_counts({digest: 1})
                engine._write_generated_ignored_placeholder_hash_ordinals({digest: {1}})
                incoming = [{"role": "assistant", "content": placeholder}]
                if case.variant in {"gap", "exhausted"}:
                    keys = engine._ignored_placeholder_ordinal_metadata_keys()
                    engine._store.write_metadata_json(keys, encoded({digest: [2]}))
                    engine._generated_ignored_active_replay_placeholder_hash_ordinals = {}
                if case.variant == "duplicate":
                    incoming *= 2
                if case.variant == "new":
                    incoming.insert(0, {"role": "user", "content": "fresh"})
            if case.proof == "carry":
                key = "compaction_commit_proof:S"
                proof = engine._store.read_metadata_json(key)
                proof["effective_sha256"] = ["0" * 64]
                engine._store.write_metadata_json([key], encoded(proof))
            if case.variant == "placeholder":
                incoming[1] = {"role": "assistant", "content": engine._ignored_active_replay_placeholder("IGNORE THIS ROW")}
                engine._remember_generated_ignored_placeholder_hash(engine._active_replay_placeholder_digest(incoming[1]["content"]))
            if case.variant == "empty":
                incoming = []
            if case.variant == "delta":
                incoming.append({"role": "user", "content": "fresh delta"})
            if case.variant == "changed":
                incoming[0] = {**incoming[0], "content": "different content"}
            if case.variant == "proof-extends":
                # A durable output can extend beyond content-only suffix proof.
                engine._store._conn.execute("UPDATE messages SET content='older raw form' WHERE store_id=1")
                engine._store._conn.commit()
            if case.variant == "rewrite":
                incoming[0]["content"] = incoming[0]["content"].strip()
                incoming.append({"role": "user", "content": "new request"})
                content = stored[0]["content"]
                engine._store.write_metadata_json(["host_rewrite_identity:1"], encoded({
                    "version": 1, "content": incoming[0]["content"],
                    "stored_sha256": hashlib.sha256(content.encode()).hexdigest()}))
            if case.variant == "user-after":
                incoming = incoming[1:] + [{"role": "user", "content": "next turn"}]
            if case.variant == "changed-result":
                incoming[2]["content"] = "new invocation result"
            if case.variant == "missing-result":
                del incoming[2]
            if case.variant == "duplicate-result":
                incoming.insert(3, dict(incoming[2]))
            if case.variant == "missing":
                (root / "hermes-results" / "payload.txt").unlink()
            if case.variant == "appear":
                (root / "hermes-results" / "payload.txt").unlink()
            if case.variant == "replaced":
                (root / "hermes-results" / "payload.txt").write_text("different file generation")
            if case.variant == "generationless":
                incoming[-1]["content"] = re.sub(r"\[LCM persisted-output file generation: [^\]]+\]\n", "", incoming[-1]["content"])
                (root / "hermes-results" / "payload.txt").unlink()
            if case.variant == "no-session":
                engine._session_id = ""
            if case.mode == "public":
                engine.shutdown()
                # The context manager shuts down the reopened instance at exit.
                engine.__init__(config=engine._config, hermes_home=str(root / "home"))
                engine.on_session_start("S", conversation_id="trace-conversation", context_length=200000)
            canonical = Canonical(root, incoming)
            events.append("input " + compact(canonical.value(incoming)))
            aliases = [next(i for i, row in enumerate(incoming) if row is m) for m in incoming]
            events.append("aliases " + encoded(aliases))
            before_input = encoded(canonical.value(incoming))
            # Avoid observer reads inside the trace: matcher inputs are prepared first.
            tail_rows = engine._store.get_session_messages("S")
            tail = [engine._message_replay_identity(row, stored_row=True) for row in tail_rows]
            if case.variant == "invalid-tail-json":
                tail = [(role, content, call_id, "{invalid", tool) for role, content, call_id, _calls, tool in tail]
            if case.variant == "no-tail":
                tail = []
            before = snapshot(engine, canonical)
            events.append("before attrs=" + fingerprint(encoded(before)))
            connections: list[tuple[str, sqlite3.Connection]] = []
            for name in ("_store", "_dag", "_lifecycle", "_host_rewrite_store"):
                conn = getattr(getattr(engine, name, None), "_conn", None)
                if isinstance(conn, sqlite3.Connection) and all(conn is not c for _, c in connections):
                    connections.append((name, conn))
            shapes: dict[str, int] = {}

            def sql_trace(name: str, sql: str) -> None:
                # SQLite-internal FTS5 statements vary by SQLite build.
                # Filter raw SQL before normalization can erase quoted schema names.
                if sql.lstrip().startswith("--") or re.search(r"(['\"])(main|temp)\1\s*\.", sql):
                    return
                normalized = canonical.text(sql.strip())
                # Keep literal whitespace; normalize only SQL layout.
                parts = re.split(r"('(?:''|[^'])*')", normalized)
                normalized = "".join(p if i % 2 else " ".join(p.split()) for i, p in enumerate(parts))
                shape = re.sub(r"'(?:''|[^'])*'|(?<![\w])\d+(?:\.\d+)?(?![\w])", "?", normalized)
                key = name + " " + shape
                if key not in shapes:
                    shapes[key] = len(shapes) + 1
                    events.append(f"sql shape={shapes[key]} {key}")
                event = f"sql #{shapes[key]} " + fingerprint(normalized)
                if events and events[-1] == event:
                    events[-1] += " x2"
                else:
                    events.append(event)

            def wrap(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
                def observed(*args: Any, **kwargs: Any) -> Any:
                    events.append(f"store {name} " + compact(canonical.value([args, kwargs])))
                    fail = case.fault == name or (case.variant == "write-error" and name == "write_metadata_json")
                    if fail:
                        engine._store._conn.set_authorizer(lambda *_: sqlite3.SQLITE_DENY)
                    try:
                        result = original(*args, **kwargs)
                    except Exception as exc:
                        events.append(f"store raises {type(exc).__name__} " + canonical.text(str(exc)))
                        raise
                    finally:
                        if fail:
                            engine._store._conn.set_authorizer(None)
                    events.append("store result " + compact(canonical.value(result)))
                    return result
                return observed

            log = logging.getLogger("hermes_lcm")
            handler, old_level = Logs(events, canonical), log.level
            log.addHandler(handler)
            log.setLevel(logging.DEBUG)
            with pytest.MonkeyPatch.context() as observers:
                real_recover = reconcile.recover_hermes_persisted_output_with_file_stat
                recovery_calls = 0

                def recover(text: str | None) -> Any:
                    nonlocal recovery_calls
                    recovery_calls += 1
                    events.append("file recover " + compact(canonical.value(text)))
                    recovered = real_recover(text)
                    events.append("file result " + compact(canonical.value(recovered)))
                    if recovery_calls == 1 and case.variant in {"vanish", "appear"}:
                        payload = root / "hermes-results" / "payload.txt"
                        if case.variant == "vanish":
                            payload.unlink()
                        else:
                            payload.write_text("line one\nline two\n" * 40)
                            os.utime(payload, (1700000000, 1700000000))
                        events.append("file " + case.variant)
                    return recovered

                observers.setattr(reconcile, "recover_hermes_persisted_output_with_file_stat", recover)
                for name in ("get_session_count", "get_session_tail", "get_session_messages",
                             "get_session_messages_after", "get_range", "get_message",
                             "get_batch", "get_session_rows_through", "find_session_rows_by_content",
                             "read_metadata_json", "write_metadata_json"):
                    if hasattr(engine._store, name):
                        observers.setattr(engine._store, name, wrap(name, getattr(engine._store, name)))
                for name, conn in connections:
                    conn.set_trace_callback(partial(sql_trace, name))
                try:
                    if case.mode == "matcher":
                        result = engine._find_reconciled_cursor_for_store_tail(incoming, tail,
                            stored_tail_rows=None if case.variant in {"no-tail-rows", "invalid-tail-json", "no-tail"} else tail_rows,
                            allow_empty_prefix=case.allow_empty,
                            session_count=len(tail), raw_session_count=len(tail),
                            allow_session_end_replay_proof=case.session_end)
                    elif case.mode == "public":
                        result = engine.ingest(incoming)
                    else:
                        result = engine._reconcile_ingest_cursor_from_store(incoming,
                            allow_session_end_replay_proof=case.session_end)
                    events.append("return " + encoded(result))
                except Exception as exc:
                    events.append(f"raise {type(exc).__name__} " + canonical.text(str(exc)))
                finally:
                    for _, conn in connections:
                        conn.set_trace_callback(None)
                    log.removeHandler(handler)
                    log.setLevel(old_level)
            after = snapshot(engine, canonical)
            events.extend(f"attr {name}=" + compact(value) for name, value in after.items())
            events.append("attrs sha256=" + hashlib.sha256(encoded(after).encode()).hexdigest())
            events.append("input-after " + compact(canonical.value(incoming)))
            after_aliases = [next(i for i, row in enumerate(incoming) if row is m) for m in incoming]
            events.append("aliases-after " + encoded(after_aliases))
            assert encoded(canonical.value(incoming)) == before_input, "target mutated caller input"
            assert after_aliases == aliases, "target changed caller aliases"
    return events


def first_difference(expected: Trace, actual: Trace) -> str | None:
    for index in range(max(len(expected), len(actual))):
        left = expected[index] if index < len(expected) else "<end>"
        right = actual[index] if index < len(actual) else "<end>"
        if left != right:
            return f"first divergence event {index}: expected {left!r}, got {right!r}"
    return None


def main() -> None:
    environment = dict(os.environ)
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--regenerate", action="store_true")
        parser.add_argument("--repo", type=Path, default=ROOT)
        parser.add_argument("--out", type=Path)
        args = parser.parse_args()
        for key in product_env_keys(ROOT) | product_env_keys(args.repo.resolve()):
            os.environ.pop(key, None)
        if not args.regenerate and args.out is None:
            parser.error("pass --out or explicitly --regenerate")
        assert Path(reconcile.__file__).resolve() == args.repo.resolve() / "reconcile.py", "wrong product tree"
        with tempfile.TemporaryDirectory(prefix="reconcile-trace-") as directory:
            observed = {case.name: observe(case, Path(directory) / case.name) for case in SCENARIOS}
        destination = args.out or GOLDEN
        destination.write_text(json.dumps(observed, sort_keys=True, indent=0, ensure_ascii=False) + "\n")
        print(f"Wrote {len(observed)} scenarios to {destination.name}")
    finally:
        os.environ.clear()
        os.environ.update(environment)


if __name__ == "__main__":
    main()
