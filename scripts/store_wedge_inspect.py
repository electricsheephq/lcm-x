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
from hermes_lcm.db_bootstrap import SCHEMA_VERSION, _V5_CORE_OPTIONAL_COLUMNS, _V5_CORE_TABLE_COLUMNS  # noqa: E402
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
from hermes_lcm.store import _normalize_observed_at  # noqa: E402


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
    metadata_columns = {row[1] for row in connection.execute("PRAGMA table_info(metadata)")}
    _require(_V5_CORE_TABLE_COLUMNS["metadata"] <= metadata_columns, "schema_mismatch")
    stamp = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
    _require(stamp is not None and str(stamp[0]) == str(SCHEMA_VERSION), "schema_mismatch")
    missing_optional = set()
    for table, required in _V5_CORE_TABLE_COLUMNS.items():
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        # db_bootstrap._V5_CORE_OPTIONAL_COLUMNS permits pre-source-time v5 stores.
        optional = _V5_CORE_OPTIONAL_COLUMNS.get(table, frozenset())
        _require(required <= columns, "schema_mismatch")
        missing_optional.update(optional - columns)
    return missing_optional


def _coverage(connection, rows, session_id, session_rows, conversation_id):
    """Validate edges before counting coverage, including sources below the frontier."""
    owned = {row["store_id"] for row in rows}
    session_owned = {row["store_id"] for row in session_rows}
    # IdentityAnchorMixin._identity_anchor_carry_ranges / _identity_anchor_owned:
    # parent sources are legal inside persisted (start, end] ranges. Validate
    # their stored scope here; host ancestry remains unassessed by the view probe.
    record = connection.execute("SELECT value FROM metadata WHERE key=?",
                                (f"identity_anchor_carry:{session_id}",)).fetchone()
    carry = json.loads(record[0]) if record else []
    _require(isinstance(carry, list), "unverified_ancestry")
    carried = set()
    owners = {"", str(conversation_id or "").strip()}
    for span in carry:
        _require(isinstance(span, list) and len(span) == 3, "unverified_ancestry")
        source, start, end = span
        _require(isinstance(source, str) and source and source != session_id
                 and type(start) is int and type(end) is int and 0 <= start < end, "unverified_ancestry")
        parent = connection.execute("SELECT store_id,conversation_id FROM messages "
                                    "WHERE session_id=? AND store_id>? AND store_id<=? ORDER BY store_id",
                                    (source, start, end)).fetchall()
        _require(bool(parent) and parent[-1][0] == end
                 and all(str(r[1] or "").strip() in owners for r in parent), "unverified_ancestry")
        carried.update(r[0] for r in parent)
    session_owned.update(carried)
    nodes = {row["node_id"]: dict(row) for row in connection.execute("SELECT * FROM summary_nodes")}
    done, visiting, covered = {}, set(), set()

    def visit(node_id):
        _require(node_id in nodes and node_id not in visiting, "invalid_dag")
        if node_id in done:
            return done[node_id]
        visiting.add(node_id)
        node = nodes[node_id]
        _require(node["session_id"] == session_id, "unverified_ancestry")
        try:
            ids = json.loads(node["source_ids"])
        except (ValueError, TypeError):
            raise InspectionRefused("invalid_dag") from None
        _require(isinstance(ids, list) and bool(ids) and all(type(i) is int and i > 0 for i in ids), "invalid_dag")
        _require(len(ids) == len(set(ids)), "invalid_dag")
        _require(node["source_type"] in {"messages", "nodes"}, "invalid_dag")
        if node["source_type"] == "messages":
            _require(ids == sorted(ids), "invalid_dag")
            lineage = set(ids)
            _require(lineage <= session_owned, "invalid_dag")
        else:
            # SummaryDAG.get_uncondensed_at_depth orders condensation inputs by
            # created_at, not by node id (repairs can insert older nodes later).
            _require(all(child in nodes for child in ids), "invalid_dag")
            _require(ids == sorted(ids, key=lambda child: nodes[child]["created_at"]), "invalid_dag")
            lineage = set()
            for child in ids:
                lineage.update(visit(child))
        # Like db_bootstrap.select_conversation_range, coverage belongs only to
        # this conversation; a reused session's wholly foreign DAG is not ours.
        _require(not lineage & owned or lineage <= owned | carried, "invalid_dag")
        covered.update(lineage & owned)
        visiting.remove(node_id)
        done[node_id] = lineage
        return lineage

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
        # rows already share owned_conversation_values() ownership: legacy blank,
        # NULL and padded spellings must not split a copy group.
        key = tuple(json.dumps(row.get(k), sort_keys=True) for k in (
            "session_id", "role", "content", "tool_call_id", "tool_calls", "tool_name"))
        groups.setdefault(key, []).append(row["store_id"])
        if row["role"] == "tool" and row["content"] == _HERMES_PRUNED_TOOL_PLACEHOLDER:
            originals = {r["store_id"] for r in rows if r["role"] == "tool"
                         and r["tool_call_id"] and r["tool_call_id"] == row["tool_call_id"]
                         # MessageStore.find_session_tool_rows allows an absent name.
                         and (not r["tool_name"] or not row["tool_name"] or r["tool_name"] == row["tool_name"])
                         and r["store_id"] < row["store_id"]
                         and r["content"] != row["content"]}
            if originals:
                suspects.update(originals | {row["store_id"]})
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


def _candidate_composite(content, texts):
    """Exact joins, including constituents that themselves contain blank lines."""
    pending, seen = [(0, 0)], {0}
    while pending:
        offset, count = pending.pop()
        for text in texts:
            if not text or not content.startswith(text, offset):
                continue
            end = offset + len(text)
            if end == len(content) and count:
                return True
            if content[end:end + 2] == "\n\n" and end + 2 not in seen:
                seen.add(end + 2)
                pending.append((end + 2, count + 1))
    return False


def _view_probe(connection, rows, covered, frontier, session_id, config, snapshot, session_rows):
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    host_keys = [f"{prefix}:{session_id}" for prefix in
                 ("identity_anchor_carry", "folded_tail_lineage", "retained_user_anchor",
                  "ignored_dependent_reply_hashes")]
    host_keys += [f"host_rewrite_identity:{r['store_id']}" for r in rows]
    _require(not any(json.loads(metadata[k]) for k in host_keys if k in metadata), "unsupported_view_evidence")
    # StoreCompleteMixin._store_complete_copy_twins needs the active host UID
    # lineage/mode to prove exclusions. A binding table alone cannot reproduce it.
    if connection.execute("SELECT 1 FROM sqlite_schema WHERE type='table' AND name='host_uid_bindings'").fetchone():
        _require(connection.execute("SELECT 1 FROM host_uid_bindings LIMIT 1").fetchone() is None,
                 "copy_exclusions_unassessed")
    proof = json.loads(metadata.get(f"compaction_commit_proof:{session_id}", "null"))
    _require(not isinstance(proof, dict) or not proof.get("carry_ranges"), "unverified_ancestry")
    required = ("leaf_chunk_tokens", "selection_budget", "effective_trigger", "fresh_tail_count",
                "fresh_tail_max_tokens", "pressure_yield_limit", "context_window")
    _require(all(type(config.get(k)) is int and config[k] >= 0 for k in required), "config_incomplete")
    _require(config["leaf_chunk_tokens"] > 0 and config["selection_budget"] > 0
             and config["effective_trigger"] > 0 and config["context_window"] > 0, "config_incomplete")
    _require(isinstance(config.get("lcm_config"), dict)
             and set(LCMConfig.__dataclass_fields__) == config["lcm_config"].keys()
             and config["lcm_config"].get("leaf_chunk_tokens")
             == config["leaf_chunk_tokens"], "config_incomplete")
    # Use the captured fallback counter only; never start a tokenizer download or change global state.
    _require(config.get("token_counter") == "fallback-v1" and tokens._encoder_ready
             and tokens._get_encoder() is None, "token_counter_unavailable")
    view = snapshot.get("messages")
    _require(isinstance(view, list) and all(isinstance(m, dict) for m in view), "snapshot_invalid")
    _require(not any(k in m for m in view for k in ("message_uid", "_lcm", "store_id")),
             "unsupported_view_evidence")
    # IdentityAnchorMixin._identity_anchor_summary_input: the no-witness branch
    # may compose timestamped user rows and write a relation. Do not guess claims.
    texts = {r["content"] for r in rows if isinstance(r.get("content"), str)}
    _require(not any(m.get("role") == "user" and _normalize_observed_at(m.get("timestamp")) is not None
                     and isinstance(m.get("content"), str) and "\n\n" in m["content"]
                     and _candidate_composite(m["content"], texts) for m in view),
             "identity_anchor_unsupported")
    if connection.execute("SELECT 1 FROM sqlite_schema WHERE type='table' AND name='message_relations'").fetchone():
        # IdentityAnchorMixin._identity_anchor_summary_input can record new
        # composites. Never run it here. Conservatively refuse a user stamp with
        # an owned persisted witness (_identity_anchor_witnesses), even if text
        # matching might ultimately reject that witness in the engine.
        stamps = {_normalize_observed_at(m.get("timestamp")) for m in view if m.get("role") == "user"}
        owned = {r["store_id"] for r in rows}
        witnessed = any(r["store_id"] in owned and r["observed_at"] in stamps
                        for r in connection.execute("SELECT store_id,observed_at FROM message_relations "
                                                    "WHERE kind='composite' AND observed_at IS NOT NULL"))
        _require(not witnessed, "identity_anchor_unsupported")
    try:
        # ReconcileMixin._get_store_id_map_for_messages reads the whole session,
        # while scanned attribution and coverage remain conversation-scoped.
        probe = _Probe(connection, session_rows, session_id, frontier, config["lcm_config"], covered)
    except Exception:
        raise InspectionRefused("config_incomplete") from None
    # Check every row before entering helpers that could otherwise resolve an external payload.
    for message in [*session_rows, *view]:
        probe._message_replay_identity(message, stored_row=True)
    mapping = probe._get_store_id_map_for_messages(view)
    # The session-wide replay mapper can select a foreign conversation occurrence.
    _require(set(mapping.values()) <= {r["store_id"] for r in rows}, "unsupported_view_evidence")
    shown = _match_occurrences([r for r in rows if r["store_id"] not in mapping.values()], probe._stored_row_forms,
                              [(i, probe._message_replay_identity(m, strip_carrier=False))
                               for i, m in enumerate(view) if id(m) not in mapping])
    visible = set(mapping.values()) | set(shown)
    shown_messages = {sid: id(view[i]) for sid, i in shown.items()}
    tail_cap = config["fresh_tail_max_tokens"]
    if config["pressure_yield_limit"]:
        tail_cap = min(tail_cap, config["pressure_yield_limit"]) if tail_cap else config["pressure_yield_limit"]
    tail = resolve_fresh_tail_boundary(view, fresh_tail_count=config["fresh_tail_count"], fresh_tail_max_tokens=tail_cap)
    leading = 0
    while leading < len(view) and view[leading].get("role") == "system":
        leading += 1
    raw = view[leading:max(leading, tail.start)]
    # CompactionMixin._filter_leaf_backlog keeps dependent replies for selection;
    # _prepare_leaf_input removes them only from summarizer input, not coverage.
    filtered, excluded, dependent_ids, dependent = [], set(), set(), False
    by_id = {r["store_id"]: r for r in rows}
    for message in raw:
        # LCMEngine._mapped_stored_row_matches_ignore_message_patterns also
        # checks stored text before replay normalization removes notifications.
        stored = by_id.get(mapping.get(id(message)))
        ignored = probe._matches_ignore_message_patterns(message) or bool(
            stored and probe._matches_ignore_message_patterns(stored))
        role = message.get("role")
        if ignored:
            if id(message) in mapping:
                excluded.add(mapping[id(message)])
            dependent = True
            continue
        if dependent and role in {"assistant", "tool"}:
            dependent_ids.add(id(message))
        dependent = dependent and role not in {"user", "system"}
        filtered.append(message)
    view = view[:leading] + filtered + view[max(leading, tail.start):]
    tail = resolve_fresh_tail_boundary(view, fresh_tail_count=config["fresh_tail_count"], fresh_tail_max_tokens=tail_cap)
    raw = view[leading:max(leading, tail.start)]
    # CompactionMixin._select_leaf_candidate returns before selection below the
    # working threshold. Force/fit/sweep/deferred state is absent from V1 captures.
    _require(not raw or (not probe._config.dynamic_leaf_chunk_enabled
                        and tokens.count_messages_tokens(raw) >= config["leaf_chunk_tokens"]),
             "leaf_admission_unassessed")
    chunk = select_oldest_leaf_chunk(raw, config["selection_budget"])
    top = max((mapping[id(m)] for m in chunk if id(m) in mapping), default=0)
    end = top - 1 if top > frontier else min(
        (sid - 1 for sid in mapping.values() if sid > frontier and sid not in excluded), default=None)
    scanned = ([r for r in rows if r["store_id"] > frontier and (end is None or r["store_id"] <= end)], False)
    # CompactionMixin._select_leaf_candidate stops before attribution when there
    # is no candidate_raw and no hidden backlog (StoreCompleteMixin._store_complete_backlog).
    hidden = [r for r in scanned[0] if r["store_id"] not in visible | covered
              and r["role"] != "system" and not probe._matches_ignore_message_patterns(r)]
    noop = not raw and not hidden

    def attribute(budget):
        selected = [m for m in chunk if id(m) not in dependent_ids]
        if chunk and not selected:
            probe._store_complete_empty_reason, probe._store_complete_blocked = "", None
            return None  # IdentityAnchorMixin._identity_anchor_summary_input guard.
        return probe._store_complete_input(selected, {}, mapping, view, chunk, frontier, [], budget,
                                           accounted_ids=excluded, scanned=scanned)

    actual = None if noop else attribute(config["leaf_chunk_tokens"])
    reason = "" if noop else probe._store_complete_empty_reason
    actual_blocker = None if noop else probe._store_complete_blocked
    # A budget cut can hide the later retained row. Probe the same selection without that cut
    # to name it; this does not relabel the actual attribution branch.
    all_cost = tokens.count_messages_tokens(probe._store_complete_messages(rows)) + tokens.count_messages_tokens(view)
    unlimited = None if noop else attribute(max(all_cost + 1, config["leaf_chunk_tokens"]))
    blocker = None if noop else actual_blocker or probe._store_complete_blocked
    blocker_id = blocker[0] if blocker else None
    positions = [i for i, m in enumerate(view) if blocker_id is not None and mapping.get(id(m)) == blocker_id]
    positions += [i for i, message in enumerate(view) if shown_messages.get(blocker_id) == id(message)]
    # Count the helper's protected replay inputs, keeping its exclusions and whole tool groups,
    # then include the named retained occurrence. Stored estimates alone cannot prove a wedge.
    prefix_messages = [m for m, _ids in unlimited or ()] if blocker else []
    if blocker:
        _require(bool(positions), "blocker_evidence_missing")
        prefix_messages.append(view[min(positions)])
    cost = tokens.count_messages_tokens(prefix_messages)
    first_group = select_oldest_leaf_chunk(prefix_messages, 1)
    first_group_exception = len(first_group) == len(prefix_messages)
    empty = "none" if noop else _empty_input_kind(actual, actual_blocker, reason, chunk, mapping)
    missing = sorted({r["store_id"] for r in rows} - covered - visible)
    complete = len(visible) == len(snapshot["messages"])
    return dict(view_state="complete" if complete else "unknown", empty_input_kind=empty,
                prefix_over_leaf_budget=bool(blocker and cost > config["leaf_chunk_tokens"] and not first_group_exception),
                blocker_in_fresh_tail=bool(blocker and any(i >= tail.start for i in positions)),
                blocker_source_ids=[blocker_id] if blocker else [], prefix_tokens=cost,
                prefix_rows=len(prefix_messages), leaf_budget=config["leaf_chunk_tokens"],
                selection_budget=config["selection_budget"], unreachable_source_ids=missing,
                selection_failed=bool(not noop and (empty != "none" or tail.tokens >= config["effective_trigger"])))


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
            missing_optional = _schema(connection)
            state = connection.execute("SELECT current_session_id,current_frontier_store_id FROM lcm_lifecycle_state "
                                       "WHERE conversation_id=?", (conversation_id,)).fetchone()
            _require(state is not None and state[0] == session_id, "scope_mismatch")
            _require(type(state[1]) is int, "invalid_store")
            session_rows = [dict(r) for r in connection.execute(
                "SELECT * FROM messages WHERE session_id=? ORDER BY store_id", (session_id,))]
            # Mirror owned_conversation_values() without its process-global cache.
            owners = {"", str(conversation_id or "").strip()}
            rows = [r for r in session_rows if str(r.get("conversation_id") or "").strip() in owners]
            covered = _coverage(connection, rows, session_id, session_rows, conversation_id)
            suspects, confirmed = _pollution(connection, rows)
            result.update(polluted_copies="confirmed" if confirmed else "suspected" if suspects else "none",
                          polluted_source_ids=suspects, confirmed_source_ids=confirmed,
                          source_rows=len(rows), covered_rows=len(covered), frontier=state[1])
            for row in session_rows:
                if row.get("tool_calls"):
                    row["tool_calls"] = json.loads(row["tool_calls"])
            if view_snapshot is not None or effective_config is not None:
                try:
                    snapshot, config = view_snapshot, effective_config
                    _require(not missing_optional, "source_time_unavailable")
                    _require(isinstance(snapshot, dict), "snapshot_invalid")
                    _require(isinstance(config, dict), "config_incomplete")
                    _require(snapshot.get("source_sha256") == before and snapshot.get("session_id") == session_id
                             and snapshot.get("conversation_id") == conversation_id and bool(snapshot.get("capture_id"))
                             and snapshot.get("effective_config") == config, "snapshot_invalid")
                    with _quiet_helpers():
                        findings = _view_probe(connection, rows, covered, state[1], session_id, config, snapshot,
                                               session_rows)
                    result.update(findings)
                except InspectionRefused as error:
                    result["evidence_code"] = str(error)
                except Exception:
                    result["evidence_code"] = "snapshot_invalid"
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
        def load_evidence(path):
            if path is None:
                return None
            try:
                return json.loads(path.read_text())
            except (OSError, ValueError):
                return {}  # Invalid optional evidence must not discard database findings.

        snapshot, config = load_evidence(args.view_snapshot), load_evidence(args.effective_config)
        # Pin the fallback estimator without starting a background download.
        if isinstance(config, dict) and config.get("token_counter") == "fallback-v1":
            tokens._encoder_ready, tokens._encoder = True, None
        try:
            result = inspect_store(args.copy, session_id=args.session_id, conversation_id=args.conversation_id,
                                   view_snapshot=snapshot, effective_config=config)
        except InspectionRefused as error:
            result = {"error": str(error)}
        output = json.dumps(result, sort_keys=True)
    except Exception:
        result = {"error": "invalid_input"}
        output = '{"error": "invalid_input"}'
    print(output)
    return 2 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
