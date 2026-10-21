#!/usr/bin/env python3
"""Read-only #1598 diagnosis of an operator-made, closed store copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
import types
from contextlib import closing, contextmanager
from pathlib import Path

sys.dont_write_bytecode = True
if "hermes_lcm" not in sys.modules:
    package = types.ModuleType("hermes_lcm")
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules["hermes_lcm"] = package

from hermes_lcm import tokens  # noqa: E402
from hermes_lcm.compaction import select_oldest_leaf_chunk  # noqa: E402
from hermes_lcm.config import LCMConfig  # noqa: E402
from hermes_lcm.db_bootstrap import SCHEMA_VERSION  # noqa: E402
from hermes_lcm.externalize import extract_externalized_ref  # noqa: E402
from hermes_lcm.fresh_tail import resolve_fresh_tail_boundary  # noqa: E402
from hermes_lcm.ingest_protection import (  # noqa: E402
    _is_hermes_persisted_output_marker, is_externalized_ingest_placeholder,
    quarantine_suspicious_assistant_messages, redact_sensitive_value,
)
from hermes_lcm.message_content import normalize_content_value  # noqa: E402
from hermes_lcm.message_patterns import compile_message_patterns, matches_message_pattern  # noqa: E402
from hermes_lcm.recognition_matching import _match_occurrences  # noqa: E402
from hermes_lcm.reconcile import ReconcileMixin, _HERMES_PRUNED_TOOL_PLACEHOLDER  # noqa: E402
from hermes_lcm.store_complete import StoreCompleteMixin  # noqa: E402


class InspectionRefused(ValueError):
    """Only fixed codes may cross the CLI boundary."""


def _require(condition, code):
    if not condition:
        raise InspectionRefused(code)


@contextmanager
def _quiet_helpers():
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def _digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _sidecars(path):
    _require(not any(Path(str(path) + suffix).exists()
                     for suffix in ("-wal", "-shm", "-journal")), "sidecar_present")


def _schema(connection):
    stamp = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
    _require(stamp is not None and str(stamp[0]) == str(SCHEMA_VERSION), "schema_mismatch")
    for table, required in {
        "messages": "store_id session_id source conversation_id role content tool_call_id tool_calls tool_name "
                    "timestamp token_estimate pinned ingested_at observed_at observed_at_source",
        "summary_nodes": "node_id session_id depth summary token_count source_token_count source_ids source_type",
        "lcm_lifecycle_state": "conversation_id current_session_id current_frontier_store_id",
    }.items():
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        _require(set(required.split()) <= columns, "schema_mismatch")


def _coverage(connection, rows, session_id):
    """Validate edges before counting coverage, including sources below the frontier."""
    owned = {row["store_id"] for row in rows}
    nodes = {row["node_id"]: dict(row) for row in connection.execute("SELECT * FROM summary_nodes")}
    done, visiting, covered = set(), set(), set()

    def visit(node_id):
        _require(node_id in nodes and node_id not in visiting, "invalid_dag")
        if node_id in done:
            return
        visiting.add(node_id)
        node = nodes[node_id]
        _require(node["session_id"] == session_id, "unverified_ancestry")
        ids = json.loads(node["source_ids"])
        _require(isinstance(ids, list) and all(type(i) is int and i > 0 for i in ids), "invalid_dag")
        _require(node["source_type"] in {"messages", "nodes"}, "invalid_dag")
        if node["source_type"] == "messages":
            _require(set(ids) <= owned, "invalid_dag")
            covered.update(ids)
        else:
            for child in ids:
                visit(child)
        visiting.remove(node_id)
        done.add(node_id)

    for node_id, node in nodes.items():
        if node["session_id"] == session_id:
            visit(node_id)
    return covered


class _ReadStore:
    """Only read methods used by the mapper; never a MessageStore constructor."""

    def __init__(self, connection, rows):
        self.connection, self.rows = connection, rows

    def get_session_messages_after(self, session_id, after_store_id):
        return [r for r in self.rows if r["session_id"] == session_id and r["store_id"] > after_store_id]

    @staticmethod
    def to_openai_msg(row):
        message = {key: row[key] for key in ("role", "content", "tool_calls", "tool_call_id")
                   if row.get(key) is not None}
        if row.get("tool_name"):
            message["name"] = row["tool_name"]
        return message


class _Probe(StoreCompleteMixin, ReconcileMixin):
    """Read-only facade for ordinary raw views; unavailable host proof stays unknown."""

    def __init__(self, connection, rows, session_id, frontier, config, covered):
        self._store = _ReadStore(connection, rows)
        self._session_id, self._last_compacted_store_id = session_id, frontier
        self._config, self.covered = LCMConfig(**config), covered
        self.patterns = compile_message_patterns(self._config.ignore_message_patterns)
        self._hermes_home = None

    def _message_replay_identity(self, message, **kwargs):
        content = normalize_content_value(message.get("content")) or ""
        # These identities need host/emission/payload evidence outside the selected copy.
        _require(not (extract_externalized_ref(content) or is_externalized_ingest_placeholder(content) or "[LCM" in content
                      or "[Expand for details:" in content or "[System Message]" in content
                      or _is_hermes_persisted_output_marker(content)
                      or content == _HERMES_PRUNED_TOOL_PLACEHOLDER
                      or self._is_context_summary_content(content)), "unsupported_view_evidence")
        _require(isinstance(message.get("content", ""), (str, type(None))), "unsupported_view_evidence")
        return super()._message_replay_identity(message, **kwargs)

    def _survival_projection_source(self, *_args):
        return None

    def _restore_ingest_payload_placeholders_in_content_identity(self, value, **_kwargs):
        return value

    def _restore_ingest_payload_placeholders_in_value(self, value, **_kwargs):
        return value

    def _out_of_band_blocks_are_durable_for_row(self, **_kwargs):
        raise InspectionRefused("unsupported_view_evidence")

    def _load_host_rewrite_overrides(self, _rows):
        pass

    def _host_rewrite_override_content(self, _row):
        return None

    def _active_folded_tail_identity_overrides(self, _messages):
        return {}

    def _load_folded_tail_lineage(self, _messages):
        return None

    def _replay_occurrences(self, _messages):
        return [], False

    def _load_compression_carry_ranges(self):
        return []

    def _content_has_externalized_placeholder_ref(self, _content):
        return False  # _message_replay_identity refuses these before matching.

    def _store_complete_node_covered(self, store_ids):
        return set(store_ids) & self.covered

    def _store_complete_copy_twins(self, _rows, _carry):
        return {}, set()  # Never infer a copy exclusion without host occurrence proof.

    def _matches_ignore_message_patterns(self, message, **_kwargs):
        return matches_message_pattern(normalize_content_value(message.get("content")) or "", self.patterns)

    def _store_complete_messages(self, rows):
        messages = quarantine_suspicious_assistant_messages(
            [self._store.to_openai_msg(row) for row in rows], config=self._config,
            hermes_home=None, session_id=self._session_id, externalize=[False] * len(rows),
        )
        return [{**m, **{k: redact_sensitive_value(m[k], self._config, parse_json_strings=k == "tool_calls")
                        for k in ("content", "tool_calls") if k in m}} for m in messages]

    @staticmethod
    def _is_context_summary_content(content):
        return isinstance(content, str) and any(s in content for s in (
            "CONTEXT COMPACTION", "CONTEXT SUMMARY", "Earlier turns have been compacted", "Earlier turns were compacted"))


def _pollution(connection, rows):
    groups, suspects = {}, set()
    for row in rows:
        key = tuple(json.dumps(row.get(k), sort_keys=True) for k in (
            "session_id", "conversation_id", "source", "role", "content", "tool_call_id", "tool_calls", "tool_name"))
        groups.setdefault(key, []).append(row["store_id"])
        if row["role"] == "tool" and row["content"] == _HERMES_PRUNED_TOOL_PLACEHOLDER:
            suspects.update(r["store_id"] for r in rows if r["role"] == "tool"
                            and r["tool_call_id"] and r["tool_call_id"] == row["tool_call_id"]
                            and r["tool_name"] == row["tool_name"] and r["store_id"] <= row["store_id"])
    for ids in groups.values():
        if len(ids) > 1:
            suspects.update(ids)
    confirmed = set()
    if connection.execute("SELECT 1 FROM sqlite_schema WHERE type='table' AND name='host_uid_bindings'").fetchone():
        bindings = {}
        for row in connection.execute("SELECT store_id,uid,lineage_key,binding_version,kind FROM host_uid_bindings"):
            if row[4] in {"canonical", "version"} and row[0] in suspects and row[1] and row[2] and row[3]:
                bindings.setdefault(tuple(row[1:4]), set()).add(row[0])
        for ids in bindings.values():
            for equals in groups.values():
                overlap = ids & set(equals)
                if len(overlap) > 1:
                    confirmed.update(overlap)
    return sorted(suspects), sorted(confirmed)


def _empty_input_kind(actual, blocker, reason, chunk, mapping):
    if actual == []:
        return "retained" if blocker else "contiguity" if "contiguous" in reason else "no_source"
    if actual is None and not any(id(m) in mapping for m in chunk):
        return "no_source"  # The host's subsequent raw-lineage guard also refuses this leaf.
    return "none"


def _view_probe(connection, rows, covered, frontier, session_id, config, snapshot):
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    host_keys = [f"{prefix}:{session_id}" for prefix in
                 ("identity_anchor_carry", "folded_tail_lineage", "retained_user_anchor")]
    host_keys += [f"host_rewrite_identity:{r['store_id']}" for r in rows]
    _require(not any(json.loads(metadata[k]) for k in host_keys if k in metadata), "unsupported_view_evidence")
    proof = json.loads(metadata.get(f"compaction_commit_proof:{session_id}", "null"))
    _require(not isinstance(proof, dict) or not proof.get("carry_ranges"), "unverified_ancestry")
    required = ("leaf_chunk_tokens", "selection_budget", "effective_trigger", "fresh_tail_count",
                "fresh_tail_max_tokens", "pressure_yield_limit", "context_window")
    _require(all(type(config.get(k)) is int and config[k] >= 0 for k in required), "config_incomplete")
    _require(config["leaf_chunk_tokens"] > 0 and config["selection_budget"] > 0
             and config["effective_trigger"] > 0 and config["context_window"] > 0, "config_incomplete")
    _require(isinstance(config.get("lcm_config"), dict)
             and set(LCMConfig.__dataclass_fields__) <= config["lcm_config"].keys()
             and config["lcm_config"].get("leaf_chunk_tokens")
             == config["leaf_chunk_tokens"], "config_incomplete")
    # Use the captured fallback counter only; never start a tokenizer download or change global state.
    _require(config.get("token_counter") == "fallback-v1" and tokens._encoder_ready
             and tokens._get_encoder() is None, "token_counter_unavailable")
    view = snapshot["messages"]
    _require(isinstance(view, list) and all(isinstance(m, dict) for m in view), "snapshot_invalid")
    probe = _Probe(connection, rows, session_id, frontier, config["lcm_config"], covered)
    # Check every row before entering helpers that could otherwise resolve an external payload.
    for message in [*rows, *view]:
        probe._message_replay_identity(message, stored_row=True)
    mapping = probe._get_store_id_map_for_messages(view)
    shown = _match_occurrences([r for r in rows if r["store_id"] not in mapping.values()], probe._stored_row_forms,
                              [(i, probe._message_replay_identity(m, strip_carrier=False))
                               for i, m in enumerate(view) if id(m) not in mapping])
    visible = set(mapping.values()) | set(shown)
    tail_cap = config["fresh_tail_max_tokens"]
    if config["pressure_yield_limit"]:
        tail_cap = min(tail_cap, config["pressure_yield_limit"]) if tail_cap else config["pressure_yield_limit"]
    tail = resolve_fresh_tail_boundary(view, fresh_tail_count=config["fresh_tail_count"], fresh_tail_max_tokens=tail_cap)
    leading = 0
    while leading < len(view) and view[leading].get("role") == "system":
        leading += 1
    raw = view[leading:max(leading, tail.start)]
    chunk = select_oldest_leaf_chunk(raw, config["selection_budget"])
    top = max((mapping[id(m)] for m in chunk if id(m) in mapping), default=0)
    end = top - 1 if top > frontier else min((sid - 1 for sid in mapping.values() if sid > frontier), default=None)
    scanned = ([r for r in rows if r["store_id"] > frontier and (end is None or r["store_id"] <= end)], False)

    def attribute(budget):
        return probe._store_complete_input(chunk, {}, mapping, view, chunk, frontier, [], budget, scanned=scanned)

    actual = attribute(config["leaf_chunk_tokens"])
    reason, actual_blocker = probe._store_complete_empty_reason, probe._store_complete_blocked
    # A budget cut can hide the later retained row. Probe the same selection without that cut
    # to name it; this does not relabel the actual attribution branch.
    all_cost = tokens.count_messages_tokens(probe._store_complete_messages(rows)) + tokens.count_messages_tokens(view)
    unlimited = attribute(max(all_cost + 1, config["leaf_chunk_tokens"]))
    blocker = actual_blocker or probe._store_complete_blocked
    blocker_id = blocker[0] if blocker else None
    positions = [i for i, m in enumerate(view) if blocker_id is not None and mapping.get(id(m)) == blocker_id]
    positions += [i for sid, i in shown.items() if sid == blocker_id]
    # Count the helper's protected replay inputs, keeping its exclusions and whole tool groups,
    # then include the named retained occurrence. Stored estimates alone cannot prove a wedge.
    prefix_messages = [m for m, _ids in unlimited or ()] if blocker else []
    if blocker:
        _require(bool(positions), "blocker_evidence_missing")
        prefix_messages.append(view[min(positions)])
    cost = tokens.count_messages_tokens(prefix_messages)
    first_group = select_oldest_leaf_chunk(prefix_messages, 1)
    first_group_exception = len(first_group) == len(prefix_messages)
    empty = _empty_input_kind(actual, actual_blocker, reason, chunk, mapping)
    missing = sorted({r["store_id"] for r in rows} - covered - visible)
    # Unmatched view occurrences and special host identity metadata cannot establish health.
    complete = len(visible) == len(view) and not any(
        k in m for m in view for k in ("message_uid", "_lcm", "store_id"))
    return dict(view_state="complete" if complete else "unknown", empty_input_kind=empty,
                prefix_over_leaf_budget=bool(blocker and cost > config["leaf_chunk_tokens"] and not first_group_exception),
                blocker_in_fresh_tail=bool(blocker and any(i >= tail.start for i in positions)),
                blocker_source_ids=[blocker_id] if blocker else [], prefix_tokens=cost,
                prefix_rows=len(prefix_messages), leaf_budget=config["leaf_chunk_tokens"],
                selection_budget=config["selection_budget"], unreachable_source_ids=missing,
                selection_failed=bool(empty != "none" or tail.tokens >= config["effective_trigger"]))


def inspect_store(copy, *, session_id, conversation_id, view_snapshot=None, effective_config=None):
    """Return fixed codes, counts and source IDs; never payloads or scope identifiers."""
    path = Path(copy).resolve(strict=True)
    _sidecars(path)
    before = _digest(path)
    with path.open("rb") as source:
        header = source.read(100)
    _require(header[:16] == b"SQLite format 3\x00", "schema_mismatch")
    # mode=ro can create sidecars for a WAL-mode header even after checkpointing.
    _require(header[18:20] == b"\x01\x01", "wal_mode_copy")
    result = dict(polluted_copies="unknown", polluted_source_ids=[], prefix_over_leaf_budget=None,
                  blocker_in_fresh_tail=None, empty_input_kind="unknown", unreachable_source_ids=None,
                  view_state="unknown", healthy=False)
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            _schema(connection)
            state = connection.execute("SELECT current_session_id,current_frontier_store_id FROM lcm_lifecycle_state "
                                       "WHERE conversation_id=?", (conversation_id,)).fetchone()
            _require(state is not None and state[0] == session_id, "scope_mismatch")
            rows = [dict(r) for r in connection.execute(
                "SELECT * FROM messages WHERE session_id=? AND TRIM(COALESCE(conversation_id,'')) IN ('',?) "
                "ORDER BY store_id", (session_id, conversation_id))]
            covered = _coverage(connection, rows, session_id)
            suspects, confirmed = _pollution(connection, rows)
            result.update(polluted_copies="confirmed" if confirmed else "suspected" if suspects else "none",
                          polluted_source_ids=suspects, confirmed_source_ids=confirmed,
                          source_rows=len(rows), covered_rows=len(covered), frontier=state[1])
            for row in rows:
                if row.get("tool_calls"):
                    row["tool_calls"] = json.loads(row["tool_calls"])
            if view_snapshot is not None and effective_config is not None:
                snapshot, config = view_snapshot, effective_config
                matching = (snapshot.get("source_sha256") == before and snapshot.get("session_id") == session_id
                            and snapshot.get("conversation_id") == conversation_id and bool(snapshot.get("capture_id"))
                            and snapshot.get("effective_config") == config)
                if matching:
                    try:
                        with _quiet_helpers():
                            result.update(_view_probe(connection, rows, covered, int(state[1]), session_id, config, snapshot))
                    except InspectionRefused as error:
                        result["evidence_code"] = str(error)
            result["healthy"] = (result["view_state"] == "complete" and not suspects
                                 and not result["unreachable_source_ids"] and not result["selection_failed"]
                                 and not result["prefix_over_leaf_budget"] and not result["blocker_in_fresh_tail"])
            connection.rollback()
    except (sqlite3.Error, ValueError, TypeError, KeyError, RecursionError) as error:
        if isinstance(error, InspectionRefused):
            raise
        raise InspectionRefused("invalid_store") from None
    finally:
        _require(_digest(path) == before, "input_changed")
        _sidecars(path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect"])
    parser.add_argument("--copy", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--view-snapshot", type=Path)
    parser.add_argument("--effective-config", type=Path)
    parser.add_argument("--json", action="store_true", help="JSON is the only output format")
    args = parser.parse_args(argv)
    try:
        snapshot = json.loads(args.view_snapshot.read_text()) if args.view_snapshot else None
        config = json.loads(args.effective_config.read_text()) if args.effective_config else None
        # Pin the fallback estimator without starting a background download.
        if config and config.get("token_counter") == "fallback-v1":
            tokens._encoder_ready, tokens._encoder = True, None
        result = inspect_store(args.copy, session_id=args.session_id, conversation_id=args.conversation_id,
                               view_snapshot=snapshot, effective_config=config)
    except InspectionRefused as error:
        result = {"error": str(error)}
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, RecursionError):
        result = {"error": "invalid_input"}
    print(json.dumps(result, sort_keys=True))
    return 2 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
