"""Frozen, real-engine ingest traces; regenerate only with an explained delta.

SQL events preserve literal whitespace and transaction order. SQLite-internal
FTS statements use exactly the reconcile harness exclusion. Store observers
call the real method once; failures are injected only at the SQLite boundary.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import tempfile
import uuid
from functools import lru_cache, partial
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, cast
from contextlib import ExitStack, contextmanager

import pytest

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
from tests.behaviour.frontier_observer import legacy_frontier_fields, observed_attribute  # noqa: E402
from tests.behaviour.reconcile_trace_harness import (  # noqa: E402
    Canonical, LiteralPattern, NOTE, Scenario as ReplayScenario, encoded, fingerprint,
    first_difference as first_difference, install_clock as install_replay_clock, rows, seed_proof,
)
from tests.harness import ScriptedSummarizer, engine_context, pin_char_counter  # noqa: E402
from tests.phase0_hygiene import product_env_keys  # noqa: E402

engine_module: Any = importlib.import_module("hermes_lcm.engine")
ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "fixtures" / "ingest_trace.json"
TRACE_GUIDANCE = "Regenerate deliberately: python -m tests.behaviour.ingest_trace_harness --repo . --regenerate"
# FROZEN inventory, never generated from _ingest_messages at observation time.
OBSERVED_ATTRIBUTES = frozenset({
    "_session_id", "_session_ignored", "_session_stateless", "_session_platform", "_conversation_id",
    "_ingest_cursor", "_ingest_cursor_needs_reconcile", "_last_compacted_store_id",
    "_compress_commit_proof", "_last_emission_descriptors", "_compress_occurrences", "_compress_run",
    "_config", "_hermes_home", "_store", "_dag", "_lifecycle", "_host_rewrite_store",
    "_host_rewrite_overrides", "_host_rewrite_watch", "_last_active_replay_messages",
    "_last_active_replay_source_identities", "_last_ingest_reconciliation", "_retired_durable_proof",
    "_compiled_ignore_message_patterns", "_compression_boundary_ingest_pending",
    "_compression_boundary_active_placeholder_digest_budget", "_compression_boundary_active_placeholder_digest_ordinals",
    "_compression_boundary_stored_placeholder_digest_counts", "_ignored_message_count", "_ignore_pattern_dropped_count",
    "_generated_ignored_active_replay_placeholder_message_ids", "_generated_ignored_active_replay_placeholder_hashes",
    "_generated_ignored_active_replay_placeholder_hash_counts", "_generated_ignored_active_replay_placeholder_hash_ordinals",
    "_session_end_flush_active", "_pending_session_tails", "_out_of_band_durable_rows", "_foreground_budget",
    "_consecutive_ingest_failures", "_pruned_tool_source_generation", "_pruned_tool_source_identities",
    "_pruned_tool_source_store", "_LCM_SUMMARY_PART_HEADER_RE", "context_length",
    "_current_compress_placeholder_identity_counts", "_emitted_carriers_retired", "_emitted_carriers_seen",
    "_foreground_rebind_parent_session_id", "_foreground_rebind_previous_conversation_id",
    "_foreground_rebind_previous_platform", "_foreground_rebind_previous_session_id", "_foreground_rebind_session_id",
    "_generated_ignored_active_replay_placeholder_messages", "_host_uid_counters", "_host_uid_error_logged",
    "_host_uid_read_error", "_identity_anchor_chain_cache", "_identity_anchor_recent", "_identity_anchor_text_memo",
    "_identity_anchor_recoveries", "_identity_anchor_recovery", "_identity_anchor_user_generation",
    "_identity_anchor_versions", "_pressure_yield_tail_token_limit", "_survival_projected_fields",
    "_host_uid_lineage_cache", "_engine_uids_pending", "_engine_uids_recorded",
})
Trace = list[str]
Message = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    variant: str = "full"
    shape: str = "turns"
    proof: str = ""
    fault: str = ""
    warm: bool = False
    boundary: int = 0
    ordinal: int = 1
    stored: int = 4


SCENARIOS = (
    *[Scenario("admission-" + v, "P0", variant=v, stored=0)
      for v in ("no-session", "ignored", "stateless", "empty", "full", "aliases")],
    *[Scenario(f"{shape}-{v}", "P2/P3/P4/P5/W0/W1", shape=shape, variant=v)
      for shape in ("turns", "tools", "whitespace", "marker", "ignored")
      for v in ("full", "prefix", "suffix", "new", "delta", "changed", "warm", "rewrite")],
    *[Scenario(f"proof-{p}-{v}", "P1/P3", proof=p, variant=v, warm=True)
      for p in ("output", "native", "consumed")
      for v in ("full", "foreign-session", "foreign-conversation", "consulted", "zero", "short",
                "input", "divergent", "new-uid", "new-stamp", "missing-metadata", "merge", "repair")],
    *[Scenario(f"durable-{p}-{v}", "P3/P4", proof=p, variant=v)
      for p in ("v2", "v3", "v4", "end") for v in ("full", "delta", "new-uid", "allow-end")],
    *[Scenario(f"cache-{v}", "P1/P4/P5", variant=v, warm=True)
      for v in ("absent", "short-cache", "divergent-cache", "new-uid", "new-stamp", "rewrite", "empty-delta")],
    *[Scenario(f"placeholder-{budget}-{ordinal}-{v}", "F0/F1/W0", shape="placeholder", stored=0,
                boundary=budget, ordinal=ordinal, variant=v)
      for budget in (0, 1, 2) for ordinal in (1, 2)
      for v in ("single", "all", "summary", "mixed", "generated", "stored-counts")],
    *[Scenario(f"sqlite-{v}", "P3/P4/P5/W1", fault=v,
                stored=4 if v in {"prematch", "durable"} else 0, warm=v == "prematch",
                variant="changed" if v == "prematch" else "full")
      for v in ("probe", "append", "locked", "duplicate-tail", "metadata", "relation", "prematch", "durable")],
    *[Scenario(f"anchor-{v}", "P4/F1/W1", variant=v, warm=True)
      for v in ("disabled", "stamp-replay", "stamp-new-uid", "remainder", "version")],
    Scenario("reconcile-tool-after-new", "P3/F1", shape="tools", variant="tool-after-new"),
    Scenario("repeated-tool-id", "P3/F1/W1", shape="tools", variant="repeated-id"),
    Scenario("repeated-equal-occurrences", "Sequence", variant="equal", stored=0),
    Scenario("scaffold-metadata", "W1", variant="scaffold", stored=0),
    *[Scenario(f"opaque-raw-stub-{role}-{count}", "W0", shape="raw-stub", variant=role, stored=count)
      for role in ("assistant", "user", "tool") for count in (1, 2)],
    *[Scenario(f"tool-stub-{v}", "W0/P5", shape="tool-stub", variant=v, stored=0)
      for v in ("single", "pair", "warm", "host-held", "recovery")],
    *[Scenario(f"proof-native-{v}", "P1/P3", proof="native", variant=v, warm=True)
      for v in ("lossy", "remap", "remap-new-stamp")],
    Scenario("durable-metadata-conflict", "P3", proof="v3", variant="new-stamp"),
    Scenario("empty-child-placeholder-prefix", "F0", shape="placeholder", variant="prefix", stored=0),
    Scenario("boundary-exhausted-other-digest", "F1", shape="placeholder", variant="other-digest", boundary=1, stored=0),
    Scenario("sqlite-snapshot", "W1", fault="snapshot", variant="disabled", stored=0),
    Scenario("tool-segments-after-new", "P3/F1", shape="tools-many"),
    Scenario("empty-all-ordinals", "F1", shape="placeholder", variant="all-ordinals", stored=0),
    Scenario("boundary-stored-counts-mixed", "F0", shape="placeholder", variant="stored-counts-mixed", boundary=2, stored=0),
    Scenario("boundary-two-digests", "F0", shape="placeholder", variant="two-digests", boundary=2, stored=0),
    *[Scenario(f"sequence-{v}", "Sequence/P1/P4/W1", shape="sequence", variant=v, warm=True)
      for v in ("compaction", "host-merge", "rotation", "rewrite")],
    Scenario("uid-bindings-new", "W1", shape="uid", stored=0),
    Scenario("uid-bindings-replay", "P1/W1", shape="uid", warm=True),
    Scenario("uid-bindings-new-occurrence", "P1/W1", shape="uid", warm=True, variant="new-stamp"),
    Scenario("row-filter-ignored-append-error", "F1/W1", shape="ignored", variant="changed", fault="append"),
    Scenario("row-filter-generated-append-error", "F1/W1", shape="quarantine", variant="generated-append", fault="append"),
    Scenario("row-filter-ignore-ledger-lock", "F1/W1", shape="ignored-ledger",
             variant="ignore-ledger", fault="locked", stored=0),
    Scenario("replay-build-mapped-ignore", "P2", shape="ignored", variant="mapped-only"),
    Scenario("replay-build-quarantine-prefix", "P2", shape="quarantine-prefix", warm=True),
    Scenario("protect-after-filter-copy", "W0/F1", shape="ignored-ledger", variant="protect-ledger", fault="locked", stored=0),
    Scenario("protect-path-error", "W0", shape="protect-error", variant="single", fault="protect-path", stored=0),
    Scenario("sqlite-locked-existing-tail", "W1", variant="delta", fault="locked", stored=4),
    Scenario("sqlite-duplicate-existing-tail", "W1", variant="delta", fault="duplicate-tail", stored=4),
    Scenario("protect-large-assistant", "W0", shape="raw-large", variant="assistant", stored=0),
    Scenario("protect-large-user", "W0", shape="raw-large", variant="user", stored=0),
)


@lru_cache(maxsize=4)
def method_inventory(root: Path) -> tuple[dict[str, ast.FunctionDef | ast.AsyncFunctionDef], dict[str, Path]]:
    """Respect the real engine MRO, including overrides and super calls."""
    classes: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    class_paths: dict[str, Path] = {}
    for path in sorted(root.glob("*.py")):
        for cls in ast.parse(path.read_text()).body:
            if not isinstance(cls, ast.ClassDef) or not (cls.name == "LCMEngine" or cls.name.endswith("Mixin")):
                continue
            classes[cls.name] = {n.name: n for n in cls.body
                                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            class_paths[cls.name] = path
    methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    paths: dict[str, Path] = {}
    mro = [cls.__name__ for cls in engine_module.LCMEngine.__mro__]
    # Unknown synthetic classes follow the same self-call rules in guard tests.
    for cls in [*mro, *sorted(set(classes) - set(mro))]:
        for name, node in classes.get(cls, {}).items():
            methods[cls + "." + name], paths[cls + "." + name] = node, class_paths[cls]
            if name not in methods:
                methods[name], paths[name] = node, class_paths[cls]
    return methods, paths


def engine_attributes(root: Path) -> tuple[set[str], set[str]]:
    """Transitive direct-self-call closure, including future extracted phase methods."""
    methods, _ = method_inventory(root)
    pending, visited, attributes = ["_ingest_messages"], set(), set()
    while pending:
        name = pending.pop()
        if name in visited or name not in methods:
            continue
        visited.add(name)
        method = methods[name]
        attributes |= legacy_frontier_fields(method)
        owner = next((key.split(".")[0] for key, value in methods.items() if "." in key and value is method), "")
        mro = [cls.__name__ for cls in engine_module.LCMEngine.__mro__]
        for node in ast.walk(method):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                receiver = node.func.value
                if (isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Name)
                        and receiver.func.id == "super" and owner in mro):
                    for parent in mro[mro.index(owner) + 1:]:
                        qualified = parent + "." + node.func.attr
                        if qualified in methods:
                            pending.append(qualified)
                            break
                elif isinstance(receiver, ast.Name) and receiver.id != "self":
                    qualified = receiver.id + "." + node.func.attr
                    if qualified in methods:
                        pending.append(qualified)
            attr = None
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
                attr = node.attr
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in {"getattr", "setattr", "hasattr"} and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Name) and node.args[0].id == "self"):
                key = node.args[1]
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    attr = key.value
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "__dict__"
                    and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == "self"
                    and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                attributes.add(node.args[0].value)
            if attr in methods:
                pending.append(attr)
            elif attr is not None and attr not in {"_frontier", "__dict__"}:
                attributes.add(attr)
    return attributes, visited


class TraceConnection(sqlite3.Connection):
    failure_observer: Callable[[str, Any, Exception], None] | None = None

    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        try:
            return super().execute(sql, parameters)
        except Exception as exc:
            if self.failure_observer is not None:
                self.failure_observer(sql, parameters, exc)
            raise

    def executemany(self, sql: str, seq_of_parameters: Any, /) -> sqlite3.Cursor:
        parameters = list(seq_of_parameters)
        try:
            return super().executemany(sql, parameters)
        except Exception as exc:
            if self.failure_observer is not None:
                self.failure_observer(sql, parameters, exc)
            raise


class IngestCanonical(Canonical):
    def __init__(self, root: Path, messages: list[Message]) -> None:
        super().__init__(root, messages)
        self.path_hashes: dict[str, str] = {}
        # Labels outlive individual snapshots. Keep their objects alive too, so
        # a later placeholder cannot inherit a freed object's address/label.
        self.retained_messages = {id(row): row for row in messages}

    def text(self, text: str, *, identity: bool = False) -> str:
        # Host occurrence UIDs are evidence, including UUID-shaped values.
        # Engine nonces are pinned at generation, so no UID substitution is needed.
        text = text.replace(self.root, "<sandbox>")
        text = re.sub(r"\[LCM persisted-output file generation: [^\]]+\]",
                      lambda match: re.sub(r"(mtime_ns|ctime_ns)=\d+", r"\1=<file-clock>", match.group()), text)
        for original, normalized in self.path_hashes.items():
            text = text.replace(original, normalized)
        return text

    def value(self, value: Any, path: str = "", *, in_payload: bool = False) -> Any:
        if (in_payload or path in {"content", "payload"}) and isinstance(value, dict):
            return {k: self.value(v, str(k), in_payload=True) for k, v in value.items()}
        if callable(value):
            return {"callable": True}
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        return super().value(value, path, in_payload=in_payload)


def recovery_receipts(engine: Any, value: Any) -> Any:
    """#1505 receipts hold the live store; observe only whether it is the engine's own."""
    if isinstance(value, list):
        return [recovery_receipts(engine, item) for item in value]
    if isinstance(value, dict) and "store" in value:
        return {**value, "store": {"active": value["store"] is engine._store}}
    return value


def snapshot(engine: Any, canonical: IngestCanonical) -> dict[str, Any]:
    state: dict[str, Any] = {}
    # Match address-keyed ledgers to the actual retained message objects before
    # serializing a set/map. These labels preserve sharing, never addresses.
    def register(value: Any) -> None:
        if isinstance(value, dict):
            if "role" in value and "content" in value:
                canonical.identities.setdefault(id(value), f"retained-message-{len(canonical.identities)}")
                canonical.retained_messages[id(value)] = value
            for child in value.values():
                register(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                register(child)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                register(getattr(value, field.name))
    for name in sorted(OBSERVED_ATTRIBUTES - {"_store", "_dag", "_lifecycle", "_host_rewrite_store"}):
        register(observed_attribute(engine, name))
    for name in sorted(OBSERVED_ATTRIBUTES):
        value = observed_attribute(engine, name)
        if name in {"_store", "_host_rewrite_store"}:
            value = {"present": value is not None}
        elif name == "_dag":
            value = {session: engine._dag.get_session_nodes(session, limit=10000)
                     for session in sorted({"S", engine._session_id} - {""})}
        elif name == "_lifecycle":
            value = engine._lifecycle.get_by_conversation("trace-conversation")
        elif name in {"_identity_anchor_recoveries", "_identity_anchor_recovery"}:
            value = recovery_receipts(engine, value)
        elif name == "_pending_session_tails":
            value = [{**tail, "store": (
                "<absent-store>" if "store" not in tail else
                None if tail["store"] is None else
                "<owning-store>" if tail["store"] is engine._store else "<other-store>"
            )} for tail in value]
        state[name] = canonical.value(value, name)
    conn = engine._store._conn
    # Exact durable rows, including relation IDs, observed_at/source, UID bindings,
    # scaffold/carry/proof metadata. Searchability is observed via the store API,
    # separately from this snapshot; SQLite FTS internals stay excluded.
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    state["tables"] = {name: canonical.value([tuple(row) for row in conn.execute(f'SELECT * FROM "{name}" ORDER BY 1')])
                       for name in tables if not name.startswith(("messages_fts", "sqlite_"))}
    return state


def aliases(inputs: list[Message], outputs: Any) -> dict[str, Any]:
    return {"inputs": [[j for j, other in enumerate(inputs) if other is row] for row in inputs],
            "output-is-input": outputs is inputs,
            "outputs": [[j for j, other in enumerate(inputs) if other is row] for row in outputs]
            if isinstance(outputs, list) else []}


class Logs(logging.Handler):
    def __init__(self, events: Trace, canonical: IngestCanonical) -> None:
        super().__init__()
        self.events, self.canonical = events, canonical

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append("log " + encoded([record.name, record.levelname,
                            self.canonical.text(str(record.msg)), self.canonical.value(record.args)]))


def reopen(engine: Any, root: Path, engines: ExitStack) -> Any:
    config = engine._config
    session_id = engine._session_id
    binding = dict(platform=engine._session_platform, conversation_id=engine._conversation_id,
                   context_length=engine.context_length)
    engine.shutdown()
    fresh = engine_module.LCMEngine(config=config, hermes_home=str(root / "home"))
    engines.callback(fresh.shutdown)
    fresh.on_session_start(session_id, **binding)
    return fresh


def payload_observation(engine: Any) -> list[dict[str, Any]]:
    """Observe every durable reference and read it via the supported expansion API."""
    externalize = importlib.import_module("hermes_lcm.externalize")
    tools = importlib.import_module("hermes_lcm.tools")
    refs = sorted({ref for (content,) in engine._store._conn.execute(
        "SELECT content FROM messages ORDER BY store_id")
        for ref in externalize.extract_externalized_refs(content)})
    directory = externalize.get_large_output_storage_dir(
        engine._config, hermes_home=engine._hermes_home, create=False)
    manifest = []
    for ref in refs:
        path = directory / ref
        exists = path.exists()
        raw = path.read_bytes() if exists else None
        manifest.append(dict(ref=ref, exists=exists, bytes=len(raw) if raw is not None else None,
                             sha256=hashlib.sha256(raw).hexdigest() if raw is not None else None,
                             read=json.loads(tools.lcm_expand({"externalized_ref": ref}, engine=engine))))
    return manifest


def search_observation(engine: Any, incoming: list[Message]) -> list[Any]:
    """Search input and durable message bytes through the product retrieval API."""
    stored = engine._store.get_session_messages("S")
    if engine._session_id and engine._session_id != "S":
        stored += engine._store.get_session_messages(engine._session_id)
    queries = sorted({row["content"] for row in [*incoming, *stored]
                      if isinstance(row.get("content"), str) and row["content"].strip()})
    return [[query, sorted(([hit["store_id"], hit["session_id"], hit["content"]]
                            for hit in engine._store.search(query, conversation_id="trace-conversation", limit=10000)),
                           key=lambda hit: hit[0])] for query in queries]


def install_clock(patch: pytest.MonkeyPatch) -> None:
    install_replay_clock(patch)
    clock = engine_module.time
    # ForegroundBudget lives in escalation, while its consumer's deadline lives
    # in compaction. Both must use the same clock, as must survival/SQL timings.
    for name in ("escalation", "survival_fit", "sqlite_util"):
        module = importlib.import_module("hermes_lcm." + name)
        patch.setattr(module, "time", clock)
    patch.setattr(importlib.import_module("hermes_lcm.sqlite_util"), "_monotonic", clock.monotonic)


def prepare(engine: Any, case: Scenario, root: Path, engines: ExitStack) -> tuple[Any, list[Message]]:
    replay = ReplayScenario(case.name, shape=case.shape, count=case.stored)
    original = rows(replay, root)
    if case.shape == "uid":
        for index, row in enumerate(original):
            row["message_uid"] = f"host-uid-{index}"
    if case.shape == "sequence":
        original = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}: " + "alpha beta " * 50,
                     "timestamp": 100. + i, "message_uid": f"host-{i}"} for i in range(24)]
    elif case.shape == "quarantine-prefix":
        original = [{"role": "assistant", "content": text * 3000}
                    for text in ("first repeated segment. ", "second repeated segment. ")]
    elif case.shape in {"raw-stub", "raw-large"}:
        original = []
    elif case.shape == "tools-many":
        original = [{"role": "system", "content": "system"}, {"role": "user", "content": "u1"}]
        for index in range(4):
            pair = rows(ReplayScenario(case.name, shape="tools"), root)[1:3]
            pair[0]["tool_calls"][0]["id"] = f"call-{index}"
            pair[1]["tool_call_id"] = f"call-{index}"
            pair[1]["tool_name"] = pair[1].pop("name")
            pair[1]["content"] = f"output-{index} " * 50
            original.extend(pair)
        original.append({"role": "assistant", "content": "a1"})
    if case.variant in {"remainder", "version"} or case.fault == "relation":
        original += [{"role": "user", "content": "held head\n\n  exact whitespace \t", "timestamp": 500.}]
    incoming = [dict(m) for m in original] or [{"role": "user", "content": "fresh"}]
    if case.warm or case.variant == "warm" or case.proof in {"output", "native", "consumed"}:
        engine._ingest_messages(original)
    elif original:
        engine._store.append_batch("S", original, source="cli", conversation_id="trace-conversation")
        engine = reopen(engine, root, engines)
    if case.proof in {"output", "native", "consumed"}:
        proof: dict[str, Any] = {"session_id": "S", "conversation_id": "trace-conversation", "consulted": False,
                 "output": [engine._proof_replay_identity(m, strip_carrier=False) for m in incoming],
                 "output_metadata": [dict(m) for m in incoming],
                 "input": [engine._proof_replay_identity(m, strip_carrier=False) for m in incoming],
                 "native": case.proof == "native", "end_consumed": case.proof == "consumed"}
        for row, identity in zip(proof["output_metadata"], proof["output"]):
            row["identity_sha256"] = engine_module._commit_proof_identity_digest(engine_module._proof_user_identity(identity))
        proof["output_effective"] = [engine_module._proof_user_identity(i) for i in proof["output"]]
        engine._compress_commit_proof = proof
        engine._ingest_cursor = len(incoming)
        engine._ingest_cursor_needs_reconcile = False
        if case.variant == "foreign-session":
            proof["session_id"] = "other-session"
        elif case.variant == "foreign-conversation":
            proof["conversation_id"] = "other-conversation"
        elif case.variant == "consulted":
            proof["consulted"] = True
        elif case.variant == "zero":
            engine._ingest_cursor = 0
        elif case.variant == "short":
            incoming = incoming[:1]
        elif case.variant == "input":
            proof["output"] = [("user", "compacted output", "", "", "")] * len(incoming)
        elif case.variant == "merge":
            original = [{"role": "user", "content": "left"}, {"role": "user", "content": "right"}]
            proof["output"] = [engine._proof_replay_identity(m, strip_carrier=False) for m in original]
            proof["input"] = proof["output"]
            proof["output_metadata"] = original
            engine._ingest_cursor = 2
            incoming = [{"role": "user", "content": "left\n\nright"}]
        elif case.variant == "repair":
            proof.update(droppable=[False, False, False, True], skip_landing=[True] * 4, native_summary_index=0)
            incoming = incoming[:-1]
        elif case.variant in {"remap", "remap-new-stamp"}:
            incoming.insert(0, dict(NOTE))
            if case.variant == "remap-new-stamp":
                incoming[1]["timestamp"] = 999.
        elif case.variant == "lossy":
            identity = list(proof["output"][0])
            identity[1] = "[LCM sensitive redaction: name=password_assignment; chars=8; bytes=8]"
            proof["output"][0] = tuple(identity)
        elif case.variant == "divergent":
            incoming[0]["content"] = "new content"
    elif case.proof:
        seed_proof(engine, ReplayScenario(case.name, proof=case.proof), incoming)
        if case.proof.startswith("v"):
            raw = engine._store.read_metadata_json("compaction_commit_proof:S")
            metadata = [{**engine._host_occurrence_metadata(m), "identity_sha256":
                         engine_module._commit_proof_identity_digest(engine_module._proof_user_identity(
                             engine._message_replay_identity(m, strip_carrier=False)))} for m in incoming]
            engine._store.write_metadata_json(["host_occurrence_snapshot:S"], json.dumps({
                "proof_sha256": hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest(),
                "output_metadata": metadata}, sort_keys=True))
    if case.shape == "sequence":
        incoming = engine.compress(original, current_tokens=engine.threshold_tokens + 1, force=True)
        assert engine._last_compression_status == "compacted", engine._last_compression_noop_reason
        if case.variant == "rotation":
            engine.on_session_end("S", original)
            engine.on_session_start("child", conversation_id="trace-conversation", old_session_id="S",
                                    boundary_reason="compression", context_length=200000)
        elif case.variant == "host-merge":
            merged: list[Message] = []
            for message in incoming:
                if merged and merged[-1]["role"] == message["role"] == "user":
                    merged[-1] = {**merged[-1], "content": merged[-1]["content"] + "\n\n" + message["content"]}
                else:
                    merged.append(dict(message))
            incoming = merged
        elif case.variant == "rewrite":
            incoming[-1]["content"] += " host rewrite"
        incoming.append({"role": "user", "content": "after compaction", "timestamp": 999., "message_uid": "host-next"})
    elif case.shape == "tools-many":
        incoming = incoming[2:4] + [{"role": "assistant", "content": "new row"}] + incoming[4:-1]
        incoming.append({"role": "user", "content": "u2"})
    elif case.shape == "raw-large":
        incoming = [{"role": case.variant, "content": " ".join(f"word{index:05d}" for index in range(11000))}]
    elif case.shape == "raw-stub":
        incoming = [{"role": case.variant, "content": (
            f"[Externalized payload: kind=raw_payload; role={case.variant}; chars=100; bytes=100; ref=synthetic.json]")}
            for _ in range(case.stored)]
    elif case.shape in {"tool-stub", "protect-error"}:
        incoming = [{"role": "tool", "tool_call_id": "call-1", "content": "durable output words " * 100}]
        if case.variant in {"pair", "host-held"}:
            incoming += [{"role": "tool", "tool_call_id": "call-2", "content": "another output " * 100}]
        if case.variant == "recovery":
            incoming.insert(0, {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function",
                               "function": {"name": "lcm_expand", "arguments": '{"node_id":1}'}}]})
        if case.variant in {"warm", "host-held"}:
            active = engine._ingest_messages(incoming)
            if case.variant == "host-held":
                incoming = [dict(m) for m in active]
            incoming.append({"role": "user", "content": "next"})
    elif case.shape == "quarantine-prefix":
        # A missing old payload must not be recreated before the warm cursor;
        # the other old payload is reused while a fresh tail is externalized.
        first = engine._store.get_session_messages("S")[0]
        refs = engine_module.extract_ingest_externalized_refs(first["content"])
        assert len(refs) == 1
        (root / "home" / "payloads" / refs[0]).unlink()
        incoming.append({"role": "assistant", "content": "fresh repeated segment. " * 3000})
    elif case.variant == "mapped-only":
        incoming = [{"role": "assistant", "content": "host replacement", "store_id": 2},
                    {"role": "user", "content": "after mapped ignore"}]
    elif case.variant in {"ignore-ledger", "protect-ledger"}:
        # The SQLite writer lock prevents ledger writes. The second digest
        # displaces the first, and append failure exposes the fallback copies.
        incoming = [{"role": "user", "content": "IGNORE THIS ROW first"},
                    {"role": "assistant", "content": "IGNORE THIS ROW second"},
                    {"role": "user", "content": "ordinary row after ignored messages"}]
        if case.variant == "protect-ledger":
            incoming += [{"role": "user", "content": "[Externalized payload: kind=raw_payload; role=user; chars=100; bytes=100; ref=synthetic.json]"},
                         {"role": "tool", "tool_call_id": "call-1", "content": "durable output words " * 100}]
    elif case.variant == "empty":
        incoming = []
    elif case.variant in {"prefix", "suffix"}:
        incoming = incoming[:2] if case.variant == "prefix" else incoming[-2:]
    elif case.variant == "new":
        incoming = [{"role": "user", "content": "new request"}]
    elif case.variant == "delta":
        incoming.append({"role": "user", "content": "new delta", "timestamp": 200.})
    elif case.variant in {"changed", "divergent-cache"}:
        incoming[0]["content"] = "different prefix"
    elif case.variant == "rewrite":
        incoming[0]["content"] = incoming[0]["content"].strip()
        incoming.append({"role": "user", "content": "new after rewrite"})
    elif case.variant in {"new-uid", "stamp-new-uid"}:
        incoming[0]["message_uid"] = "fresh-occurrence"
        # Both known identities are required to prove a new occurrence.
        old = engine._last_active_replay_messages
        if old:
            old[0]["message_uid"] = "old-occurrence"
        if engine._compress_commit_proof:
            engine._compress_commit_proof["output_metadata"][0]["message_uid"] = "old-occurrence"
    elif case.variant == "new-stamp":
        incoming[0]["timestamp"] = 999.
    elif case.variant == "missing-metadata":
        incoming[0].pop("timestamp", None)
    elif case.variant == "absent":
        engine._last_active_replay_source_identities = None
        engine._last_active_replay_messages = None
    elif case.variant == "short-cache":
        engine._last_active_replay_source_identities = []
        engine._last_active_replay_messages = []
    elif case.variant == "aliases":
        incoming *= 2
    elif case.variant == "equal":
        incoming = [{"role": "user", "content": "equal", "timestamp": 10., "message_uid": "one"},
                    {"role": "user", "content": "equal", "timestamp": 11., "message_uid": "two"}]
    elif case.variant == "tool-after-new":
        incoming.insert(1, {"role": "user", "content": "new between prefix and tools"})
    elif case.variant == "repeated-id":
        incoming += [dict(m) for m in incoming[1:3]]
        incoming[-1]["content"] = "new result same call id"
    elif case.variant == "scaffold":
        incoming = [{"role": "user", "content": "[Current user objective preserved from compacted history]\ninspect"}]
    if case.variant == "remainder" or case.fault == "relation":
        incoming[-1]["content"] += "\n\n  fresh remainder \n\ntrailing \t"
        incoming.append({"role": "assistant", "content": "reply", "timestamp": 501.})
    elif case.variant == "version":
        incoming = original  # The host mutates the same watched object.
        incoming[-1]["content"] = "edited head with new bytes"
        incoming.append({"role": "assistant", "content": "reply", "timestamp": 501.})
    if case.fault == "metadata":
        incoming = [{"role": "user", "content": "[Current user objective preserved from compacted history]\ninspect"}]
    if case.shape == "placeholder":
        text = engine._ignored_active_replay_placeholder("synthetic ignored row")
        digest = engine._active_replay_placeholder_digest(text)
        assert digest
        engine._remember_generated_ignored_placeholder_hash(digest)
        engine._write_generated_ignored_placeholder_hash_counts({digest: 2})
        engine._write_generated_ignored_placeholder_hash_ordinals({digest: {case.ordinal}})
        incoming = [{"role": "assistant", "content": text}]
        if case.variant in {"all", "stored-counts", "mixed", "all-ordinals", "two-digests", "stored-counts-mixed"}:
            incoming += [dict(incoming[0]), dict(incoming[0])]
        if case.variant in {"mixed", "stored-counts-mixed"}:
            incoming.insert(0, {"role": "user", "content": "not a placeholder"})
        if case.variant == "summary":
            incoming.insert(0, {"role": "user", "content": "[Current user objective preserved from compacted history]\ninspect"})
        engine._compression_boundary_ingest_pending = bool(case.boundary)
        engine._compression_boundary_active_placeholder_digest_budget = {digest: case.boundary} if case.boundary else {}
        engine._compression_boundary_active_placeholder_digest_ordinals = {digest: {case.ordinal}}
        engine._compression_boundary_stored_placeholder_digest_counts = {digest: 1} if case.variant in {"stored-counts", "two-digests", "stored-counts-mixed"} else {}
        if case.variant == "all-ordinals":
            engine._write_generated_ignored_placeholder_hash_ordinals({digest: {1, 2, 3}})
        elif case.variant == "two-digests":
            other_text = engine._ignored_active_replay_placeholder("another row")
            other = engine._active_replay_placeholder_digest(other_text)
            incoming += [{"role": "assistant", "content": other_text}] * 3
            engine._compression_boundary_active_placeholder_digest_budget[other] = 2
            engine._compression_boundary_stored_placeholder_digest_counts[other] = 1
            engine._compression_boundary_active_placeholder_digest_budget["absent-digest"] = 2
        elif case.variant == "prefix":
            engine._ingest_cursor = 2
            engine._ingest_cursor_needs_reconcile = False
            incoming.insert(0, dict(incoming[0]))
            incoming.insert(0, {"role": "system", "content": "native retained prefix"})
        elif case.variant == "other-digest":
            other = engine._active_replay_placeholder_digest(engine._ignored_active_replay_placeholder("another row"))
            incoming += [{"role": "assistant", "content": engine._ignored_active_replay_placeholder("another row")}]
            engine._compression_boundary_active_placeholder_digest_ordinals[other] = {2}
            engine._compression_boundary_active_placeholder_digest_budget[other] = 1
            engine._compression_boundary_active_placeholder_digest_budget[digest] = 0
            engine._generated_ignored_active_replay_placeholder_message_ids.add(id(incoming[0]))
        if case.variant == "generated":
            engine._generated_ignored_active_replay_placeholder_message_ids.add(id(incoming[0]))
    if case.shape == "uid" and not case.stored:
        incoming[0]["message_uid"] = "host-uid-new"
    if case.variant == "generated-append":
        incoming = [{**original[0], "timestamp": 999.}, {"role": "user", "content": "after ignored quarantine"}]
        incoming[0]["content"] = incoming[0]["content"].replace(
            "LCM sensitive redaction:", "LCM active replay placeholder: assistant output quarantined;")
        digest = engine._active_replay_placeholder_digest(incoming[0]["content"])
        assert digest
        engine._remember_generated_ignored_placeholder_hash(digest)
        incoming.insert(0, {"role": "user", "content": "new row before generated placeholder"})
    if case.variant == "no-session":
        engine._session_id = ""
    return engine, incoming


@contextmanager
def sqlite_fault(engine: Any, fault: str, events: Trace) -> Iterator[None]:
    """Fail real SQL, without mocking ingest phases or store implementations."""
    conn = engine._store._conn
    if fault == "protect-path":
        # Exercise the real containment error at payload protection, then permit retry.
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("LCM_HERMES_BASE_DIR", str(Path(engine._hermes_home) / "contained"))
            yield
        return
    if fault == "snapshot":
        conn.execute("BEGIN")
        conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        with sqlite3.connect(engine._config.database_path) as publisher:
            publisher.execute("INSERT INTO metadata VALUES ('snapshot-publisher', 'committed')")
        try:
            yield
        finally:
            if conn.in_transaction:
                conn.rollback()
        return
    if fault in {"locked", "duplicate-tail"}:
        # SQLite itself raises SQLITE_BUSY at the real append boundary.
        other = sqlite3.connect(engine._config.database_path, timeout=0)
        conn.execute("PRAGMA busy_timeout=0")
        other.execute("BEGIN IMMEDIATE")
        engine._session_end_flush_active = True
        try:
            yield
        finally:
            other.rollback()
            other.close()
            engine._session_end_flush_active = False
        return
    seen = 0

    def authorize(action: int, table: str | None, column: str | None, db: str | None, trigger: str | None) -> int:
        nonlocal seen
        deny = ((fault == "append" and action == sqlite3.SQLITE_INSERT and table == "messages")
                or (fault == "metadata" and action == sqlite3.SQLITE_INSERT and table == "metadata")
                or (fault == "relation" and action == sqlite3.SQLITE_INSERT and table == "message_relations")
                or (fault == "durable" and action == sqlite3.SQLITE_READ and table == "metadata")
                )
        if deny:
            seen += 1
            events.append(f"sqlite denied {action} {table} {column} #{seen}")
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(authorize)
    try:
        yield
    finally:
        conn.set_authorizer(None)


def observe(case: Scenario, root: Path, *, payloads: list[str] | None = None) -> Trace:
    root.mkdir(parents=True, exist_ok=True)
    events: Trace = []
    with pytest.MonkeyPatch.context() as patch:
        pin: Any = pin_char_counter
        context: Any = engine_context
        pin(patch)
        install_clock(patch)
        patch.setattr(engine_module.uuid, "uuid4", lambda: uuid.UUID(int=1))
        original_connect = sqlite3.connect

        def connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
            kwargs.setdefault("factory", TraceConnection)
            conn = cast(sqlite3.Connection, original_connect(*args, **kwargs))
            # Pin SQLite's generated migration/default clocks too. No host
            # occurrence stamp or stored message timestamp is rewritten.
            conn.create_function("strftime", 2, lambda fmt, value:
                "1700001000" if (fmt, value) == ("%s", "now") else
                (_ for _ in ()).throw(AssertionError("unclassified SQL clock")))
            return conn

        patch.setattr(sqlite3, "connect", connect)
        token_module = importlib.import_module("hermes_lcm.tokens")
        counter = token_module._count_tokens_cached

        def count(text: str, generation: int) -> int:
            return int(counter(text.replace(str(root), "<sandbox>"), generation))

        patch.setattr(token_module, "_count_tokens_cached", count)
        patch.setenv("HERMES_HOME", str(root / "home"))
        patch.setenv("LCM_IDENTITY_ANCHOR", "0" if case.variant == "disabled" else "1")
        patch.setenv("LCM_HOST_UID_SHADOW", "1")
        settings: dict[str, Any] = dict(embeddings_enabled=False, temporal_rollups_enabled=False,
            empty_lifecycle_gc_enabled=False, large_output_externalization_enabled=False,
            large_output_active_replay_stubbing_enabled=False,
            large_output_externalization_path=str(root / "home" / "payloads"))
        if case.shape == "sequence":
            summarizer_factory: Any = ScriptedSummarizer
            summarizer = summarizer_factory()
            summarizer.install(patch)
            settings.update(leaf_chunk_tokens=300, fresh_tail_count=2, fresh_tail_max_tokens=1000)
        if case.shape in {"tool-stub", "protect-error", "raw-large"} or case.variant == "protect-ledger":
            settings.update(large_output_externalization_enabled=True,
                            large_output_externalization_threshold_chars=100,
                            large_output_active_replay_stubbing_enabled=True,
                            large_output_active_replay_stub_threshold_tokens=10)
        if case.shape in {"ignored", "ignored-ledger"}:
            settings["ignore_message_patterns"] = ["IGNORE THIS ROW"]
        if case.variant in {"ignored", "stateless"}:
            settings["ignore_session_patterns" if case.variant == "ignored" else "stateless_session_patterns"] = ["S"]
        if case.shape == "uid":
            (root / "home").mkdir(exist_ok=True)
            with sqlite3.connect(root / "home" / "state.db") as host:
                host.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, "
                             "end_reason TEXT, source TEXT, model_config TEXT)")
                host.execute("INSERT INTO sessions VALUES ('S', NULL, NULL, 'cli', NULL)")
        with context(root, window=200000, conversation_id="trace-conversation", **settings) as engine, ExitStack() as engines:
            if case.shape in {"ignored", "ignored-ledger"}:
                engine._compiled_ignore_message_patterns = [LiteralPattern()]
            engine, incoming = prepare(engine, case, root, engines)
            if case.shape in {"ignored", "ignored-ledger"}:
                engine._compiled_ignore_message_patterns = [LiteralPattern()]
            canonical = IngestCanonical(root, incoming)
            if case.shape == "uid":
                state_path = str(Path(engine._state_db_path()).resolve())
                tag = hashlib.sha256(state_path.encode()).hexdigest()[:16]
                normalized_tag = hashlib.sha256(canonical.text(state_path).encode()).hexdigest()[:16]
                canonical.path_hashes[tag] = normalized_tag
            proof_keys = [row[0] for row in engine._store._conn.execute(
                "SELECT key FROM metadata WHERE key LIKE 'compaction_commit_proof:%' ORDER BY key")]
            for key in proof_keys:
                raw_proof = engine._store.read_metadata_json(key)
                if isinstance(raw_proof, dict):
                    # Occurrence bindings checksum JSON containing the sandbox
                    # path. Recompute after path substitution; content/proof
                    # changes still produce a different checksum.
                    raw_hash = hashlib.sha256(json.dumps(raw_proof, sort_keys=True).encode()).hexdigest()
                    path_hash = hashlib.sha256(json.dumps(canonical.value(raw_proof), sort_keys=True).encode()).hexdigest()
                    canonical.path_hashes[raw_hash] = path_hash
            # All emitted values retain their exact canonical payload in the hash;
            # larger values use the established sha12+length display encoding.
            def emit(label: str, value: Any) -> None:
                text = encoded(canonical.value(value, label))
                if payloads is not None:
                    payloads.append(label + " " + text)
                events.append(label + " " + (text if len(text) <= 80 else "sha12/len=" + fingerprint(text)))

            emit("input-before", incoming)
            emit("aliases-before", aliases(incoming, incoming))
            before = snapshot(engine, canonical)
            emit("state-before", before)
            def sql_trace(name: str, sql: str) -> None:
                if sql.lstrip().startswith("--") or re.search(r"(['\"])(main|temp)\1\s*\.", sql):
                    return
                emit("sql " + name, canonical.text(sql))

            def wrap(name: str, original: Callable[..., Any], inject_faults: bool) -> Callable[..., Any]:
                def observed(*args: Any, **kwargs: Any) -> Any:
                    # Callback identity is a stable label; invocation stays inside
                    # the real store transaction and remains visible through SQL.
                    def argument(v: Any) -> Any:
                        if callable(v):
                            return "<metadata-callback>"
                        if isinstance(v, dict):
                            return {k: argument(x) for k, x in v.items()}
                        if isinstance(v, (list, tuple)):
                            return [argument(x) for x in v]
                        return v
                    emit("store " + name, argument([args, kwargs]))
                    callback = kwargs.get("metadata_factory")
                    if callback is not None:
                        def metadata(*values: Any) -> Any:
                            emit("metadata-callback input", values)
                            result = callback(*values)
                            emit("metadata-callback result", result)
                            return result
                        kwargs = {**kwargs, "metadata_factory": metadata}
                    scoped_failure = inject_faults and (
                        (case.fault == "probe" and name == "has_session_messages")
                        or (case.fault == "prematch" and name == "find_rows_by_observed_at"))
                    if scoped_failure:
                        engine._store._conn.set_authorizer(lambda action, table, *_:
                            sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ and table == "messages"
                            else sqlite3.SQLITE_OK)
                    try:
                        result = original(*args, **kwargs)
                    except Exception as exc:
                        emit("store raises " + name, [type(exc).__name__, str(exc)])
                        raise
                    finally:
                        if scoped_failure:
                            engine._store._conn.set_authorizer(None)
                    emit("store result " + name, result)
                    return result
                return observed

            @contextmanager
            def observing(*, inject_faults: bool = False) -> Iterator[None]:
                # Resolve connections anew for each engine, including cold
                # reopen. Snapshots and lifecycle setup stay outside tracing.
                connections: list[tuple[str, sqlite3.Connection]] = []
                for name in ("_store", "_dag", "_lifecycle", "_host_rewrite_store"):
                    conn = getattr(getattr(engine, name, None), "_conn", None)
                    if isinstance(conn, sqlite3.Connection) and all(conn is not c for _, c in connections):
                        connections.append((name, conn))
                log = logging.getLogger("hermes_lcm")
                handler, old_level = Logs(events, canonical), log.level
                log.addHandler(handler)
                log.setLevel(logging.DEBUG)
                with pytest.MonkeyPatch.context() as observers:
                    for name in ("append_batch", "_append_protected_batch", "has_session_messages", "get_session_count",
                                 "get_session_messages", "get_session_messages_after", "get_session_tail",
                                 "get_session_rows_through", "get_range", "get_batch", "get_message_relations", "get",
                                 "read_metadata_json", "read_metadata_json_many", "update_metadata_json", "write_metadata_json",
                                 "backfill_observed_at_many", "add_message_relations", "add_host_uid_bindings",
                                 "host_uid_bindings_for", "find_rows_by_observed_at", "find_session_rows_by_content",
                                 "load_session_page"):
                        if hasattr(engine._store, name):
                            observers.setattr(engine._store, name, wrap(name, getattr(engine._store, name), inject_faults))
                    for name, conn in connections:
                        conn.set_trace_callback(partial(sql_trace, name))
                        if isinstance(conn, TraceConnection):
                            conn.failure_observer = lambda sql, parameters, exc, name=name: emit(
                                "sql failed " + name, [sql, parameters, type(exc).__name__, str(exc)])
                    try:
                        yield
                    finally:
                        for _, conn in connections:
                            conn.set_trace_callback(None)
                            if isinstance(conn, TraceConnection):
                                conn.failure_observer = None
                        log.removeHandler(handler)
                        log.setLevel(old_level)

            with observing(inject_faults=True), sqlite_fault(engine, case.fault, events):
                for attempt in range(2 if case.fault == "duplicate-tail" else 1):
                    try:
                        result = engine._ingest_messages(incoming,
                            allow_session_end_replay_proof=case.variant == "allow-end")
                        emit(f"return-{attempt}", result)
                        emit(f"aliases-{attempt}", aliases(incoming, result))
                    except Exception as exc:
                        emit(f"raise-{attempt}", [type(exc).__name__, str(exc)])
            after = snapshot(engine, canonical)
            for name, value in after.items():
                if before.get(name) != value:
                    emit("attr " + name, value)
            emit("state-after", after)
            emit("search-after", search_observation(engine, incoming))
            manifest = payload_observation(engine)
            if manifest:
                emit("payloads-after", manifest)
            if case.shape == "quarantine-prefix":
                emit("quarantine-payloads-after", [
                    [path.name, hashlib.sha256(path.read_bytes()).hexdigest()]
                    for path in sorted((root / "home" / "payloads").glob("*.json"))])
            emit("input-after", incoming)
            emit("aliases-after", aliases(incoming, incoming))
            # Deferred effects are checked by another real ingest and a cold
            # lifecycle rebind, including a replay after the reopened cursor.
            engine._session_end_flush_active = False
            with observing():
                try:
                    result = engine._ingest_messages(incoming)
                    emit("next-return", result)
                    emit("next-aliases", aliases(incoming, result))
                except Exception as exc:
                    emit("next-raise", [type(exc).__name__, str(exc)])
            emit("next-state", snapshot(engine, canonical))
            engine = reopen(engine, root, engines)
            if case.shape in {"ignored", "ignored-ledger"}:
                engine._compiled_ignore_message_patterns = [LiteralPattern()]
            with observing():
                try:
                    emit("cold-return", engine._ingest_messages(incoming))
                except Exception as exc:
                    emit("cold-raise", [type(exc).__name__, str(exc)])
            emit("cold-state", snapshot(engine, canonical))
            manifest = payload_observation(engine)
            if manifest:
                emit("payloads-cold", manifest)
    return events


def main() -> None:
    environment = dict(os.environ)
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--regenerate", action="store_true")
        parser.add_argument("--repo", type=Path, default=ROOT)
        parser.add_argument("--out", type=Path)
        parser.add_argument("--payloads", type=Path)
        parser.add_argument("--scenario")
        args = parser.parse_args()
        for key in product_env_keys(ROOT) | product_env_keys(args.repo.resolve()):
            os.environ.pop(key, None)
        if not args.regenerate and args.out is None:
            parser.error("pass --out or explicitly --regenerate")
        assert Path(engine_module.__file__).resolve() == args.repo.resolve() / "engine.py", "wrong product tree"
        if args.regenerate and args.scenario:
            parser.error("cannot regenerate a partial inventory")
        traces, payloads = {}, {}
        with tempfile.TemporaryDirectory(prefix="ingest-trace-") as directory:
            for case in SCENARIOS:
                if args.scenario and case.name != args.scenario:
                    continue
                full: list[str] = []
                traces[case.name] = observe(case, Path(directory) / case.name, payloads=full)
                payloads[case.name] = full
        if args.payloads:
            args.payloads.write_text(json.dumps(payloads, sort_keys=True, ensure_ascii=False) + "\n")
        destination = args.out or GOLDEN
        destination.write_text(json.dumps(traces, sort_keys=True, indent=0, ensure_ascii=False) + "\n")
        print(f"Wrote {len(traces)} scenarios to {destination.name}")
    finally:
        os.environ.clear()
        os.environ.update(environment)


if __name__ == "__main__":
    main()
