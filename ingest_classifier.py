"""P2 PR 1: read-only classification of one host view against one lineage.

No plan is applied here. PARTIAL and rank 5 are discovery-only until PR 2.
The caller owns the snapshot transaction and resolves NewRow references on insert.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import json
from types import MappingProxyType
from typing import NamedTuple, Protocol

from . import ingest_ledger as ledger
from .host_uid import _valid_uid
from .message_content import normalize_content_value
from .reconcile import ReconcileMixin, _strip_volatile_host_scaffolding
from .store import _normalize_observed_at, ledger_digest


class Descriptor(NamedTuple):
    view_seq: int
    output_index: int


class NewRow(NamedTuple):
    index: int


class Record(NamedTuple):
    kind: str
    view_seq: int


class Run(NamedTuple):
    record: Record
    i: int
    j: int
    k: int


class Head(NamedTuple):
    kind: str
    ordinal: int | None


class Coverage(NamedTuple):
    kind: str
    unbound: int = 0


class Relation(NamedTuple):
    kind: str
    target: int | Descriptor | None


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return deepcopy(value)


def _thaw(value):
    if isinstance(value, (dict, MappingProxyType)):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class IncomingRow:
    index: int
    role: str
    never_stored: bool
    identity_digest: bytes
    stamp_digest: bytes | None
    uidset_digest: bytes | None
    stamp: str | None
    uid: str | None
    absorbed_uids: tuple[str, ...]
    tool_call_id: str
    tool_segment: int | None
    message: object


def build_view(salt, messages, *, uid_mode, never_stored=None):
    """Host-only preparation, with the same raw digests in all three uid modes."""
    if uid_mode not in ("off", "shadow", "on"):
        raise ValueError("unknown uid mode")
    rows, calls = [], defaultdict(list)
    for i, message in enumerate(messages):
        role = str(message.get("role") or "unknown")
        segment = None
        if role == "assistant" and message.get("tool_calls"):
            segment = i
            for cid in {str(c.get("id") or "") for c in message["tool_calls"] if isinstance(c, dict)} - {""}:
                calls[cid].append(i)
        cid = str(message.get("tool_call_id") or "")
        if role == "tool" and calls[cid]:
            segment = calls[cid].pop()
        uid = message.get("message_uid")
        uid = uid if _valid_uid(uid) else None
        absorbed = message.get("_absorbed_message_uids")
        absorbed = tuple(u for u in absorbed if _valid_uid(u)) if isinstance(absorbed, (list, tuple)) else ()
        stamp = ledger.stamp_text(_normalize_observed_at(message.get("timestamp")))
        rows.append(IncomingRow(i, role, role == "system" if never_stored is None else bool(never_stored(message)),
                                ledger.identity_digest(salt, message), ledger.stamp_digest(salt, stamp),
                                ledger.uidset_digest(salt, uid, absorbed), stamp, uid, absorbed, cid,
                                segment, _freeze(message)))
    return tuple(rows)


class LedgerSnapshot(Protocol):
    """Transaction-bound read surface; implementations must never open writable storage."""
    lineage: str
    salt: bytes
    stamp_policy: str
    records: tuple
    emissions: tuple
    bindings: dict
    by_digest: dict
    by_stamp: dict
    uid_bindings: dict
    alt_stamp_counts: dict
    coverage: Coverage
    pending_digests: set
    query_count: int

    def payload(self, occ_id): ...

    def stamps(self, occ_id): ...

    def ordinals(self): ...


class SqliteLedgerSnapshot:
    """Read one lineage using existing ledger, message and uid indexes.

    Bulk reads are scoped to the lineage; no reads escape the caller's connection.
    The coverage anti-join is deliberately temporary (L2b owns its replacement).
    """
    def __init__(self, conn, lineage, *, session_id):
        self.connection = conn
        self.lineage, self.query_count = lineage, 0

        def read(sql, args=()):
            self.query_count += 1
            return ledger._rows(conn.execute(sql, args))

        meta = {r["key"]: r["value"] for r in read("SELECT * FROM ledger_meta")}
        self.salt = meta["digest_salt"]
        marks = read("SELECT * FROM lineage_marks WHERE lineage=?", (lineage,))
        self.stamp_policy = marks[0]["stamp_policy"] if marks else "normal"
        floor = int(meta.get("emission_provenance_floor", meta.get("view_seq", 0)))
        recovered = int(meta.get("emission_recovered:" + lineage, -1))
        records = read("SELECT * FROM ingest_views WHERE lineage=? ORDER BY view_seq DESC", (lineage,))
        retained = Counter()
        self.records = []
        for r in records:
            retained[r["kind"]] += 1
            if retained[r["kind"]] > 2:
                continue
            r["eligible"] = r["kind"] == "ingest" or (
                r["state"] in ("proposed", "adopted") and r["view_seq"] > max(floor, recovered))
            r["entries"] = tuple(ledger.decode_entries(r["entries"]))
            self.records.append(r)
        self.records = tuple(self.records)
        self.emissions = tuple(read("SELECT * FROM ledger_emissions WHERE lineage=?", (lineage,)))
        for e in self.emissions:
            e["eligible"] = e["state"] in ("proposed", "adopted") and e["view_seq"] > max(floor, recovered)
        bindings = read("SELECT b.*,m.role,m.content,m.tool_call_id,m.tool_calls,m.tool_name,m.observed_at,"
                        "m.session_id,m.conversation_id FROM occurrence_bindings b LEFT JOIN messages m "
                        "ON m.store_id=b.store_id WHERE b.lineage=?", (lineage,))
        self.bindings = {b["occ_id"]: b for b in bindings}
        self.by_digest, self.by_stamp = defaultdict(list), defaultdict(list)
        self._stamps, self._payloads = defaultdict(set), {}
        for b in bindings:
            calls = b["tool_calls"]
            if isinstance(calls, str):
                b["tool_calls"] = json.loads(calls)
            self._payloads[b["occ_id"]] = b
            b["uids"], b["constituents"] = set(), set()
            if b["stamp"] is not None:
                self._stamps[b["occ_id"]].add(b["stamp"])
            if b["disposition"] in ("stored", "migrated", "ambiguous") and b["role"] is not None:
                self.by_digest[b["identity_digest"]].append(b)
        self.alt_stamp_counts = Counter()
        for o in read("SELECT * FROM stamp_observations WHERE lineage=?", (lineage,)):
            self._stamps[o["occ_id"]].add(o["stamp"])
            if o["kind"] == "alt_stamp":
                self.alt_stamp_counts[o["occ_id"]] += 1
        tables = {r["name"] for r in read("SELECT name FROM sqlite_master WHERE type='table'")}
        self.uid_bindings = defaultdict(list)
        by_store = defaultdict(list)
        for b in bindings:
            by_store[b["store_id"]].append(b)
        if "host_uid_bindings" in tables:
            for u in read("SELECT * FROM host_uid_bindings WHERE lineage_key=? "
                          "AND kind IN ('canonical','version','constituent') ORDER BY store_id", (lineage,)):
                self.uid_bindings[u["uid"]].append(u)
                for b in by_store[u["store_id"]]:
                    b["constituents" if u["kind"] == "constituent" else "uids"].add(u["uid"])
        if "message_relations" in tables:
            for r in read("SELECT r.* FROM occurrence_bindings b JOIN message_relations r ON r.store_id=b.store_id "
                          "WHERE b.lineage=? AND r.kind IN ('alt_stamp','carried_stamp','supersedes')", (lineage,)):
                for b in by_store[r["store_id"]]:
                    if r["kind"] == "supersedes":
                        for old in by_store[r["related_store_id"]]:
                            old["superseded_by"] = b["occ_id"]
                    elif r["observed_at"] is not None:
                        self._stamps[b["occ_id"]].add(ledger.stamp_text(r["observed_at"]))
        for candidates in self.by_digest.values():
            candidates.sort(key=lambda b: (b["ordinal"], b["occ_id"]))
            for b in candidates:
                for stamp in self._stamps[b["occ_id"]]:
                    self.by_stamp[stamp].append(b)
        sessions = {session_id} | {b["session_id"] for b in bindings if b["session_id"]}
        sessions.update(k.removeprefix("session_lineage:") for k, v in meta.items()
                        if k.startswith("session_lineage:") and v == lineage)
        count = read("SELECT count(*) AS n FROM messages m WHERE session_id IN "
                     "(SELECT value FROM json_each(?)) AND NOT EXISTS "
                     "(SELECT 1 FROM occurrence_bindings b WHERE b.store_id=m.store_id)",
                     (json.dumps(sorted(sessions)),))[0]["n"]
        self.coverage = Coverage("partial" if count else "complete", count)
        # Registry lookup is read-only and shares the store identity used by P1.
        paths = read("PRAGMA database_list")
        path = next(r["file"] for r in paths if r["name"] == "main")
        from pathlib import Path
        self.pending_digests = ledger.pending_emission_digests((str(Path(path).resolve()), self.salt), lineage)

    def payload(self, occ_id):
        return self._payloads[occ_id]

    def stamps(self, occ_id):
        return frozenset(self._stamps[occ_id])

    def ordinals(self):
        return sorted({b["ordinal"] for b in self.bindings.values() if b["ordinal"] is not None})


class IdentityOracle(Protocol):
    read_sources: tuple

    def incoming_aliases(self, row, snapshot): ...

    def stored_aliases(self, row, snapshot): ...

    def has_generated_marker(self, row): ...


def _memo_key(value):
    """Exact, type-tagged JSON input, retaining object order and float signs."""
    kind = type(value)
    if kind is str:
        return value
    if kind is dict:
        return kind, tuple((_memo_key(k), _memo_key(v)) for k, v in value.items())
    if kind in (list, tuple):
        return kind, tuple(_memo_key(v) for v in value)
    if kind is float:
        return kind, value.hex()
    if kind in (int, bool, type(None)):
        return kind, value
    raise TypeError("not a JSON memo input")


class _IdentityMemo:
    """One pass's helper reuse, without modifying or retaining engine state.

    Bind only the replay method here; all store-dependent rules still call the
    real engine. Store/config/home are fixed for this adapter's lifetime.
    """
    _message_replay_identity = ReconcileMixin._message_replay_identity

    def __init__(self, engine):
        self.engine = engine
        self.calls, self.strings, self.contents = {}, {}, {}

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def _canonical(self, value):
        if isinstance(value, dict):
            return {k: self._canonical(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._canonical(v) for v in value]
        if type(value) is str:
            if value not in self.strings:
                self.strings[value] = ReconcileMixin._canonicalize_tool_call_identity_value(value)
            return self.strings[value]
        return ReconcileMixin._canonicalize_tool_call_identity_value(value)

    def _stable_tool_calls_identity(self, calls):
        if not calls:
            return ""
        try:
            key = _memo_key(calls)
        except TypeError:
            return self.engine._stable_tool_calls_identity(calls)
        if key not in self.calls:
            try:
                self.calls[key] = json.dumps(self._canonical(calls), sort_keys=True,
                                             separators=(",", ":"), ensure_ascii=False)
            except (TypeError, ValueError):
                self.calls[key] = str(calls)
        return self.calls[key]

    def _restore_ingest_payload_placeholders_in_content_identity(self, content, *, session_id, hermes_home=None):
        key = content, session_id, hermes_home
        if key not in self.contents:
            self.contents[key] = self.engine._restore_ingest_payload_placeholders_in_content_identity(
                content, session_id=session_id, hermes_home=hermes_home)
        return self.contents[key]


class EngineOracle:
    """Adapter for an already-bound engine. Never accesses the engine's store property.

    Read helpers may fill engine caches; L1 must establish shadow neutrality.
    Explicit store arguments prevent these helpers from opening another store.
    """
    read_sources = ("bound snapshot connection: messages, metadata, message_relations, summary DAG",
                    "externalized payload files", "Hermes persisted-output files")

    def __init__(self, engine):
        self.engine = engine
        self.store = engine._bound_store_unopened()
        if self.store is None:
            raise ValueError("EngineOracle requires an already opened bound store")

    def _for_pass(self):
        if getattr(self.engine._message_replay_identity, "__func__", None) is not ReconcileMixin._message_replay_identity:
            return self
        oracle = EngineOracle.__new__(EngineOracle)
        oracle.engine, oracle.store = _IdentityMemo(self.engine), self.store
        return oracle

    @staticmethod
    def identity_tuple(row):
        role = str(row.get("role") or "unknown")
        return (role, _strip_volatile_host_scaffolding(normalize_content_value(row.get("content")) or ""),
                str(row.get("tool_call_id") or ""),
                ReconcileMixin._stable_tool_calls_identity(row.get("tool_calls")),
                str(row.get("tool_name") or row.get("name") or "") if role == "tool" else "")

    def _identity(self, row, snapshot, **kwargs):
        if self.store._conn is not getattr(snapshot, "connection", self.store._conn):
            raise ValueError("oracle and snapshot must share their connection")
        identity = self.engine._message_replay_identity(
            row, store=self.store, hermes_home=self.engine._hermes_home,
            session_id=self.engine._session_id, conversation_id=self.engine._conversation_id, **kwargs)
        # Reconcile's helpers return the same field order, but not all strip scaffolding.
        role, content, cid, calls, name = identity
        return role, _strip_volatile_host_scaffolding(normalize_content_value(content) or ""), cid, calls, name

    def incoming_aliases(self, row, snapshot):
        return (self._identity(row, snapshot),)

    def stored_aliases(self, row, snapshot):
        # Host-rewrite overrides apply only to user rows. The other roles have
        # the same stored identity with either flag, including tool payloads.
        rewrites = (False, True) if row.get("role") == "user" else (False,)
        forms = {self._identity(row, snapshot, stored_row=True, with_host_rewrite=h) for h in rewrites}
        tool = row.get("role") == "tool"
        raw = self.identity_tuple(row) if tool else None
        # A digest-equal nameless form proves the ledger identity was nameless,
        # including when a stored stub expands to the original payload. Otherwise
        # a lost `name` cannot be reconstructed: use v5.1's witness fallback.
        nameless = tool and any(not t[4] and ledger_digest(snapshot.salt, *t) == row["identity_digest"]
                                for t in forms | {raw})
        needs_witness = tool and not nameless
        aliases = tuple((identity, needs_witness and not identity[4]) for identity in forms)
        return aliases + (((raw[:-1] + ("",), True),) if tool and raw[4] else ())

    def has_generated_marker(self, row):
        return self.engine._lcm_generated_marker(row) is not None


@dataclass
class RowVerdict:
    verdict: str = "new"
    evidence: str | None = None
    target: int | Descriptor | None = None
    reason: str | None = None
    relations: list[Relation] = field(default_factory=list)
    deferred: str | None = None


@dataclass
class WritePlan:
    uid_bindings: list = field(default_factory=list)
    alt_stamps: list = field(default_factory=list)
    emission_stamps: list = field(default_factory=list)
    adoptions: list = field(default_factory=list)
    stamp_unreliable: bool = False
    placements: list = field(default_factory=list)
    entries: list = field(default_factory=list)


@dataclass
class Classification:
    lineage: str
    base: Record | str
    previous: Record | None
    run: Run | None
    head: Head
    coverage: Coverage
    counters: Counter
    rows: tuple[RowVerdict, ...]
    writes: WritePlan


_REASONS = ("veto_uid", "veto_stamp", "veto_tool_call_id", "refused", "revision",
            "head_h4", "window", "order", "no_witness")


class _Pass:
    def __init__(self, view, snapshot, oracle, mode, ignore_stamps):
        oracle = oracle._for_pass() if type(oracle) is EngineOracle else oracle
        self.view, self.s, self.oracle, self.mode = view, snapshot, oracle, mode
        self.ignore_stamps = ignore_stamps
        self.rows = [RowVerdict("skip" if r.never_stored else "new") for r in view]
        self.rejected, self.first = defaultdict(set), {}
        self.candidates, self.target_rejections = defaultdict(list), defaultdict(set)
        self.used, self.counts = set(), Counter()
        self._placed = None
        self.aliases, self.incoming, self.messages = {}, {}, {}
        self.by_store, self.by_role_call = defaultdict(list), defaultdict(list)
        self.uid_revisions = {uid: [u for u in rows if u["kind"] in ("canonical", "version")]
                              for uid, rows in snapshot.uid_bindings.items()}
        sessions = defaultdict(list)
        for b in snapshot.bindings.values():
            self.by_store[b["store_id"]].append(b)
            sessions[b["session_id"]].append(b)
            if b["disposition"] in ("stored", "migrated", "ambiguous") and b["role"] is not None:
                self.by_role_call[b["role"], b["tool_call_id"] or ""].append(b)
        self.rank3_index, self.stored_segments, self.user_before = {}, {}, {}
        user = None
        for b in sorted((b for b in snapshot.bindings.values() if b["ordinal"] is not None),
                        key=lambda b: b["ordinal"]):
            self.user_before[b["occ_id"]] = user
            if b["role"] == "user":
                user = b
        for bindings in sessions.values():
            segment = None
            for b in sorted(bindings, key=lambda b: b["ordinal"] or 0):
                if b["role"] == "assistant" and b.get("tool_calls"):
                    segment = []
                    self.stored_segments[b["occ_id"]] = segment
                elif segment is not None:
                    segment.append(b)
        self.anchor_before = {}
        anchor = None
        for row in view:
            self.anchor_before[row.index] = anchor
            if row.role == "user":
                anchor = row.index
        self.head, self.run, self.base = Head("h4", None), None, "history"
        self.previous = next((r for r in snapshot.records if r["kind"] == "ingest"), None)
        self.segments = defaultdict(list)
        for row in view:
            if row.tool_segment is not None:
                self.segments[row.tool_segment].append(row.index)
        self.plan = WritePlan()

    def ordinal(self, target):
        return self.s.bindings[target]["ordinal"] if isinstance(target, int) else None

    def reject(self, i, target, reason):
        if target is not None:
            self.remember(i, target)
            self.rejected[i].add(reason)
            self.target_rejections[i, target].add(reason)

    def remember(self, i, target):
        self.first.setdefault(i, target)
        if target not in self.candidates[i]:
            self.candidates[i].append(target)

    def accept(self, i, target, evidence):
        if target in self.used:
            return False
        self.rows[i] = RowVerdict("replay", evidence, target)
        self.used.add(target)
        self._placed = None
        return True

    def known_stamp(self, row):
        return None if self.ignore_stamps else row.stamp

    def uid_rows(self, uid):
        return self.uid_revisions.get(uid, ())

    def latest(self, uid):
        bindings = self.uid_rows(uid)
        return bindings[-1] if bindings else None

    def eligible(self, u):
        return not u["disagree_seen"] and (u["proof_kind"] == "stored_new" or u["store_id"] in self.by_store)

    def current_uid(self, row, target):
        u = self.latest(row.uid)
        return (self.mode == "on" and u is not None and self.eligible(u)
                and isinstance(target, int) and u["store_id"] == self.s.bindings[target]["store_id"])

    def old_revision(self, b):
        return self.mode != "off" and any(self.latest(uid) is not None and
            self.latest(uid)["store_id"] != b["store_id"] for uid in b["uids"])

    def veto(self, row, b):
        if self.mode != "off" and row.uid and b["uids"]:
            if {row.uid, *row.absorbed_uids} != b["uids"] | b["constituents"]:
                return "veto_uid"
        stamps = set(self.s.stamps(b["occ_id"]))
        observed = ledger.stamp_text(b["observed_at"])
        if observed is not None:
            stamps.add(observed)
        if self.known_stamp(row) is not None and stamps and row.stamp not in stamps:
            return "veto_stamp"
        if row.tool_call_id and b["tool_call_id"] and row.tool_call_id != b["tool_call_id"]:
            return "veto_tool_call_id"
        if self.old_revision(b):
            return "revision"
        return None

    def positive(self, row, b):
        return (self.known_stamp(row) is not None and row.stamp in self.s.stamps(b["occ_id"]))

    def match(self, row, b):
        if row.identity_digest == b["identity_digest"]:
            return "identity"
        incoming = self.incoming_digests(row)
        self.stored_digests(b)
        if b["identity_digest"] in incoming:
            return "alias"
        matches = incoming & self.aliases[b["occ_id"]].keys()
        if matches:
            return "name" if all(self.aliases[b["occ_id"]][d] for d in matches) else "alias"
        return None

    def incoming_digests(self, row):
        if row.index not in self.incoming:
            self.incoming[row.index] = {ledger_digest(self.s.salt, *t) for t in
                                        self.oracle.incoming_aliases(self.message(row), self.s)} | {row.identity_digest}
        return self.incoming[row.index]

    def message(self, row):
        # The engine adapter only reads these structures; arbitrary injected
        # oracles retain the original independent-copy behaviour.
        if type(self.oracle) is not EngineOracle:
            return _thaw(row.message)
        if row.index not in self.messages:
            self.messages[row.index] = _thaw(row.message)
        return self.messages[row.index]

    def stored_digests(self, b):
        if b["occ_id"] not in self.aliases:
            self.aliases[b["occ_id"]] = {ledger_digest(self.s.salt, *t): witnessed for t, witnessed in
                                        self.oracle.stored_aliases(self.s.payload(b["occ_id"]), self.s)}
        return self.aliases[b["occ_id"]]

    def entry_target(self, record, entry):
        if entry.is_descriptor:
            emission = next((e for e in self.s.emissions if e["emission_id"] == entry.ref), None)
            return Descriptor(emission["view_seq"], emission["output_index"]) if emission else None
        return entry.ref

    def emission(self, target):
        return next((e for e in self.s.emissions if (e["view_seq"], e["output_index"]) == target), None)

    def run_search(self):
        starts = [r.index for r in self.view if not r.never_stored][:2]
        chosen = None
        for i in starts:
            runs = []
            for record in self.s.records:
                if not record["eligible"]:
                    continue
                entries = record["entries"]
                for j, entry in enumerate(entries):
                    if entry.identity_digest != self.view[i].identity_digest:
                        continue
                    k = 0
                    for row, old in zip(self.view[i:], entries[j:]):
                        if row.identity_digest != old.identity_digest:
                            break
                        if (self.mode != "off" and row.uidset_digest is not None and old.uidset_digest is not None
                                and row.uidset_digest != old.uidset_digest):
                            break
                        target = self.entry_target(record, old)
                        stamp = old.stamp_digest
                        if old.is_descriptor:
                            e = self.emission(target)
                            stamp = ledger.stamp_digest(self.s.salt, e["observed_stamp"]) if e else None
                        if (not self.ignore_stamps and row.stamp_digest and stamp and row.stamp_digest != stamp):
                            break
                        k += 1
                    if k:
                        runs.append((k, record["view_seq"], -j, record, j))
            if runs:
                chosen = max(runs, key=lambda r: r[:3])
                break
        if chosen is None:
            return
        k, _seq, _negative_j, record, j = chosen
        accepted = 0
        for offset in range(k):
            row, entry = self.view[i + offset], record["entries"][j + offset]
            if row.never_stored:
                accepted += 1
                continue
            target = self.entry_target(record, entry)
            if entry.is_descriptor:
                emission = self.emission(target)
                if emission and emission["eligible"]:
                    self.accept(row.index, target, "rank4")
                    accepted += 1
                    continue
            b = self.s.bindings.get(target)
            if b is None or b["role"] is None or target in self.used:
                self.counts["run.unresolved_ref"] += 1
                break
            if self.old_revision(b):
                self.reject(row.index, target, "revision")
                self.counts["run.revision_cut"] += 1
                break
            if (self.mode != "off" and row.uid and b["uids"]
                    and {row.uid, *row.absorbed_uids} != b["uids"] | b["constituents"]):
                self.reject(row.index, target, "veto_uid")
                self.counts["run.veto_uid"] += 1
                break
            if self.positive(row, b) or self.current_uid(row, target):
                evidence = "run_keyed"
            else:
                u = self.latest(row.uid)
                incoming_key = self.known_stamp(row) is not None or (
                    self.mode == "on" and u is not None and self.eligible(u))
                stored_key = (not self.ignore_stamps and bool(self.s.stamps(target))) or (
                    self.mode == "on" and any(self.eligible(u) for uid in b["uids"] for u in self.uid_rows(uid)
                                              if u["store_id"] == b["store_id"]))
                if incoming_key or stored_key:
                    self.reject(row.index, target, "no_witness")
                    self.counts["run.no_witness"] += 1
                    break
                evidence = "run_unkeyed"
            self.accept(row.index, target, evidence)
            accepted += 1
        if accepted:
            key = Record(record["kind"], record["view_seq"])
            self.run, self.base = Run(key, i, j, accepted), key
            self.run_record = record

    def choose_head(self):
        if self.run:
            for entry in self.run_record["entries"][self.run.j:]:
                target = self.entry_target(self.run_record, entry)
                ordinal = self.ordinal(target)
                if ordinal is not None:
                    self.head = Head("h1", ordinal)
                    return
            for e in self.s.emissions:
                if e["view_seq"] == self.run.record.view_seq and e["next_occ_id"] in self.s.bindings:
                    self.head = Head("h1", self.ordinal(e["next_occ_id"]))
                    return
        kept = sorted((b for bs in self.s.by_digest.values() for b in bs), key=lambda b: b["ordinal"])
        m = min(3, len(kept))
        eligible = [r for r in self.view if not r.never_stored and not self.oracle.has_generated_marker(self.message(r))]
        if m and len(eligible) >= m:
            last, hits = -1, []
            for row in eligible[:m]:
                hit = next((b for b in kept[:8] if b["ordinal"] > last and b["identity_digest"] == row.identity_digest), None)
                if hit is None:
                    break
                hits.append(hit)
                last = hit["ordinal"]
            if len(hits) == m:
                self.head = Head("h2", hits[0]["ordinal"])
                return
        if self.mode == "on":
            hits = []
            for row in self.view:
                u = self.latest(row.uid)
                if u and self.eligible(u):
                    hit = next((b for b in self.by_store.get(u["store_id"], ()) if self.match(row, b)), None)
                    if hit:
                        hits.append(hit)
            if hits and any(b["ordinal"] > hits[0]["ordinal"] for b in hits[1:]):
                self.head = Head("h3", hits[0]["ordinal"])
                return
            if hits:
                self.counts["head.demoted"] += 1
        if eligible and kept and any(b["identity_digest"] == eligible[0].identity_digest for b in kept[:8]):
            self.counts["head.demoted"] += 1
        self.counts["head.h4"] += 1

    def window(self, b):
        return self.head.ordinal is not None and b["ordinal"] is not None and b["ordinal"] >= self.head.ordinal

    def bounds(self, i, outside=()):
        # Placements do not change while weak pairs or deferred candidates are
        # collected. Cache that exact state; accept/cut invalidate it.
        if self._placed is None:
            self._placed = [(j, self.ordinal(v.target)) for j, v in enumerate(self.rows)
                            if v.verdict == "replay" and self.ordinal(v.target) is not None]
        placed = self._placed
        index = bisect_left(placed, (i,))
        left_index = index - 1
        while left_index >= 0 and placed[left_index][0] in outside:
            left_index -= 1
        left = placed[left_index][1] if left_index >= 0 else None
        if index < len(placed) and placed[index][0] == i:
            index += 1
        while index < len(placed) and placed[index][0] in outside:
            index += 1
        right = placed[index][1] if index < len(placed) else None
        return left, right

    def position(self, i, b, outside=()):
        if not self.window(b):
            return "head_h4" if self.head.ordinal is None else "window"
        left, right = self.bounds(i, outside)
        if (left is not None and b["ordinal"] <= left) or (right is not None and b["ordinal"] >= right):
            return "window"
        return None

    def defer(self, i, shape, targets):
        verdict = self.rows[i]
        verdict.deferred, verdict.reason = shape, "deferred"
        for target in targets:
            self.first.setdefault(i, target)
        if shape == "partial":
            verdict.relations = [Relation("possible_replay_of", t) for t in targets]

    def strong(self):
        for row, verdict in zip(self.view, self.rows):
            if verdict.verdict != "new":
                continue
            if self.mode != "off" and row.uid and (u := self.latest(row.uid)):
                stores = {v["store_id"] for v in self.uid_rows(row.uid)}
                candidates = [b for store in stores for b in self.by_store.get(store, ())]
                candidates.sort(key=lambda b: b["store_id"], reverse=True)
                latest = next((b for b in candidates if b["store_id"] == u["store_id"]), None)
                match = next((b for b in candidates if self.match(row, b)), None)
                hit = match or latest
                if hit is not None:
                    if not self.window(hit):
                        self.reject(row.index, hit["occ_id"], "head_h4" if self.head.ordinal is None else "window")
                        continue
                    if not self.eligible(u):
                        continue
                    if match is not None and latest is not None and match["occ_id"] != latest["occ_id"]:
                        self.reject(row.index, match["occ_id"], "revision")
                        verdict.relations.append(Relation("supersedes", latest["occ_id"]))
                        verdict.reason = "revision"
                        continue
                    if match is not None and self.mode == "on" and set(row.absorbed_uids) <= match["constituents"]:
                        if self.accept(row.index, match["occ_id"], "rank1"):
                            continue
                    content = normalize_content_value(self.message(row).get("content")) or ""
                    old = latest and (latest["content"] or "")
                    if match is None and old and content.startswith(old) and content != old:
                        if self.mode == "on":
                            self.defer(row.index, "partial", [latest["occ_id"]])
                            continue
                    elif match is None and latest:
                        self.reject(row.index, latest["occ_id"], "revision")
                        verdict.relations.append(Relation("supersedes", latest["occ_id"]))
                        verdict.reason = "revision"
                        continue
            if verdict.deferred or verdict.reason == "revision":
                continue
            for e in self.s.emissions:
                if e["output_index"] != row.index or e["descriptor_digest"] != row.identity_digest:
                    continue
                target = Descriptor(e["view_seq"], e["output_index"])
                if not e["eligible"]:
                    self.reject(row.index, target, "refused")
                elif self.known_stamp(row) and e["observed_stamp"] and row.stamp != e["observed_stamp"]:
                    self.reject(row.index, target, "veto_stamp")
                elif self.accept(row.index, target, "rank4"):
                    break

    def row_candidates(self, row):
        # Rank 2 discovers by positive stamp, not lineage-wide identity. Only
        # resolved segment rows can discover identity/alias candidates at rank 3.
        # Rank 5's identity lookup belongs to discovery(), inside the final gap.
        candidates = {b["occ_id"]: b for b in self.s.by_stamp.get(self.known_stamp(row), ())
                      if b["role"] == row.role}
        if row.tool_segment is not None:
            key = row.role, row.tool_call_id
            bindings = self.by_role_call.get(key, ())
            if len(bindings) > 1:
                # Index the complete identity/alias sets once, preserving every
                # candidate and witness rule even for a reused call id.
                if key not in self.rank3_index:
                    index = defaultdict(list)
                    for b in bindings:
                        for digest in self.stored_digests(b).keys() | {b["identity_digest"]}:
                            index[digest].append(b)
                    self.rank3_index[key] = index
                for digest in self.incoming_digests(row):
                    for b in self.rank3_index[key].get(digest, ()):
                        candidates[b["occ_id"]] = b
            else:
                for b in bindings:
                    if b["occ_id"] not in candidates and self.match(row, b):
                        candidates[b["occ_id"]] = b
        return sorted(candidates.values(), key=lambda b: b["ordinal"], reverse=True)

    def weak(self, *, segments):
        options = defaultdict(list)
        for row, verdict in zip(self.view, self.rows):
            if (row.tool_segment is not None) != segments:
                continue
            if verdict.verdict != "new" or verdict.deferred or verdict.reason == "revision":
                continue
            candidates = self.row_candidates(row)
            self.counts["row_candidate_pairs"] += len(candidates)
            for b in candidates:
                target = b["occ_id"]
                if target in self.used:
                    continue
                match = self.match(row, b)
                positive = self.positive(row, b)
                if not match and not positive:
                    continue
                self.remember(row.index, target)
                reason = self.veto(row, b)
                if reason:
                    self.reject(row.index, target, reason)
                    continue
                if not self.window(b):
                    self.reject(row.index, target, "head_h4" if self.head.ordinal is None else "window")
                    continue
                if positive and not match:
                    content = normalize_content_value(self.message(row).get("content")) or ""
                    if b["content"] and content.startswith(b["content"]) and content != b["content"]:
                        self.defer(row.index, "partial", [target])
                    else:
                        self.reject(row.index, target, "no_witness")
                    continue
                if match == "name" and not (positive or self.current_uid(row, target)):
                    self.reject(row.index, target, "no_witness")
                    continue
                rank3 = row.tool_segment is not None and match
                if rank3 and not positive:
                    anchor = self.anchor_before[row.index]
                    if anchor is not None:
                        placed = self.rows[anchor]
                        if placed.verdict != "replay":
                            rank3 = False
                        else:
                            user = self.user_before.get(target)
                            rank3 = user is not None and user["occ_id"] == placed.target
                    if self.known_stamp(row) and not self.s.stamps(target):
                        rank3 = False
                if positive or rank3:
                    if reason := self.position(row.index, b):
                        self.reject(row.index, target, reason)
                    else:
                        options[row.index].append((b, "alias" if match in ("alias", "name") else
                                                   "rank2" if positive else "rank3"))
                else:
                    self.reject(row.index, target, "no_witness")
        units, members = [], set()
        for start, indexes in (self.segments.items() if segments else ()):
            if len(indexes) < 2:
                continue
            members.update(indexes)
            fixed = {i: self.rows[i].target for i in indexes if self.rows[i].verdict == "replay"}
            assistants = ([self.s.bindings[fixed[start]]] if start in fixed and isinstance(fixed[start], int)
                          else [b for b, _ in options[start]])
            for assistant in assistants:
                mapping = {start: (assistant, self.rows[start].evidence if start in fixed else
                                   next(e for b, e in options[start] if b is assistant))}
                stored = self.stored_segments.get(assistant["occ_id"], ())
                answered = set()
                for i in indexes[1:]:
                    choices = ([self.s.bindings[fixed[i]]] if i in fixed and isinstance(fixed[i], int)
                               else [b for b, _ in options[i]])
                    hit = None
                    for b in stored:
                        if b in choices and b["tool_call_id"] == self.view[i].tool_call_id and b["occ_id"] not in answered:
                            hit = b
                            break
                    if hit is None:
                        break
                    answered.add(hit["occ_id"])
                    mapping[i] = (hit, self.rows[i].evidence if i in fixed else
                                  next(e for b, e in options[i] if b is hit))
                if len(mapping) == len(indexes) and not self.position(start, assistant, indexes):
                    units.append((start, assistant["ordinal"], mapping))
        for i, candidates in options.items():
            if i not in members:
                units.extend((i, b["ordinal"], {i: (b, evidence)}) for b, evidence in candidates)
        # Deduplicate complete mappings before the C11 iii uniqueness test.
        unique = {}
        for unit in units:
            unique[unit[0], tuple((i, b["occ_id"]) for i, (b, _) in sorted(unit[2].items()))] = unit
        units = sorted(unique.values(), key=lambda p: (p[0], -p[1]))
        self.counts["candidate_pairs"] += len(units)
        f, g = _lis_lengths([p[1] for p in units]), list(reversed(_lis_lengths([-p[1] for p in reversed(units)])))
        length = max(f, default=0)
        levels = Counter(a for a, b in zip(f, g) if a + b - 1 == length)
        for unit, a, b in zip(units, f, g):
            accepted = a + b - 1 == length and levels[a] == 1
            for i, (candidate, evidence) in unit[2].items():
                if self.rows[i].verdict == "replay":
                    continue
                if accepted:
                    self.accept(i, candidate["occ_id"], evidence)
                else:
                    self.reject(i, candidate["occ_id"], "order")

    def discovery(self):
        for row, verdict in zip(self.view, self.rows):
            if verdict.verdict != "new" or verdict.deferred or verdict.reason == "revision":
                continue
            content = normalize_content_value(self.message(row).get("content")) or ""
            for record in self.s.records:
                if not record["eligible"]:
                    continue
                if record["kind"] == "ingest" and record is not self.previous:
                    continue
                if row.index >= len(record["entries"]):
                    continue
                entry = record["entries"][row.index]
                target = self.entry_target(record, entry)
                old = None
                if entry.is_descriptor:
                    emission = self.emission(target)
                    # Descriptors carry digests, not payloads. The generated prefix
                    # is recognized by digesting host-only proper text boundaries.
                    if emission and emission["eligible"]:
                        for offset in range(1, len(content)):
                            prefix = content[:offset].rstrip("\n")
                            if ledger.identity_digest(self.s.salt, dict(self.message(row), content=prefix)) == entry.identity_digest:
                                targets = [target]
                                following = record["entries"][row.index + 1:row.index + 2]
                                if following and not following[0].is_descriptor and following[0].ref in self.s.bindings:
                                    donor = self.s.payload(following[0].ref)
                                    if content[offset:].lstrip("\n") == donor["content"]:
                                        targets.append(following[0].ref)
                                self.defer(row.index, "partial", targets)
                                break
                            suffix = content[offset:].lstrip("\n")
                            if ledger.identity_digest(self.s.salt, dict(self.message(row), content=suffix)) == entry.identity_digest:
                                self.defer(row.index, "partial", [target])
                                break
                elif target in self.s.bindings:
                    old = self.s.payload(target)["content"]
                if old and content != old and content.startswith(old):
                    self.defer(row.index, "partial", [target])
                if verdict.deferred:
                    break
            if verdict.deferred:
                continue
            for b in reversed(self.s.by_digest.get(row.identity_digest, ())):
                if ("order" not in self.rejected[row.index] and b["occ_id"] not in self.used
                        and not self.position(row.index, b) and not self.veto(row, b)):
                    self.defer(row.index, "rank5", [b["occ_id"]])
                    break

    def finish(self):
        cuts = []
        for start, indexes in self.segments.items():
            if any(self.rows[i].verdict != "replay" for i in indexes):
                cuts.append(start)
                for i in indexes:
                    self.cut(i)
        if self.run and any(start < self.run.i + self.run.k and
                            any(self.run.i <= i < self.run.i + self.run.k for i in self.segments[start])
                            for start in cuts):
            original = self.run.k
            end = min(start for start in cuts if any(
                self.run.i <= i < self.run.i + self.run.k for i in self.segments[start]))
            for i in range(max(end, self.run.i), self.run.i + self.run.k):
                self.cut(i)
            self.counts["run.segment_cut"] += original
            self.run = self.run._replace(k=end - self.run.i) if end > self.run.i else None
        record = self.run_record if self.base != "history" and self.head.kind != "h4" else self.previous
        a5 = defaultdict(list)
        if record:
            for entry in record["entries"]:
                target = self.entry_target(record, entry)
                if target not in self.used:
                    a5[entry.identity_digest].append(target)
        offsets, bound_uids = Counter(), set(self.s.uid_bindings)
        for row, verdict in zip(self.view, self.rows):
            if verdict.verdict == "skip":
                self.plan.entries.append(ledger.ViewEntry(row.identity_digest, row.stamp_digest, row.uidset_digest, None))
                continue
            if verdict.verdict == "new":
                # Candidate pairs are collected before weak placements commit.
                # A donor another row ultimately consumed is no longer a NEW
                # row's candidate (C4's second result); keep the next unbound
                # candidate in algorithm order, if there is one.
                if not verdict.deferred and self.first.get(row.index) in self.used:
                    self.first.pop(row.index)
                    candidate = next((t for t in self.candidates[row.index] if t not in self.used), None)
                    if candidate is not None:
                        self.first[row.index] = candidate
                if row.index not in self.first and a5[row.identity_digest]:
                    candidates = a5[row.identity_digest]
                    target = candidates[min(offsets[row.identity_digest], len(candidates) - 1)]
                    offsets[row.identity_digest] += 1
                    if target is None:
                        target = next((b["occ_id"] for b in reversed(self.s.by_digest.get(row.identity_digest, ()))), None)
                    if target is None:
                        self.counts["a5.no_target"] += 1
                    elif isinstance(target, int):
                        b = self.s.bindings.get(target)
                        if b:
                            self.reject(row.index, target, self.veto(row, b) or self.position(row.index, b) or "no_witness")
                    else:
                        e = self.emission(target)
                        self.reject(row.index, target, "refused" if not e or not e["eligible"] else "no_witness")
                if row.index not in self.first and self.head.kind == "h4":
                    candidates = self.s.by_digest.get(row.identity_digest, ())
                    if candidates:
                        self.reject(row.index, candidates[-1]["occ_id"], "head_h4")
                target = self.first.get(row.index)
                if target is not None and not any(r.kind == "possible_replay_of" for r in verdict.relations):
                    verdict.relations.insert(0, Relation("possible_replay_of", target))
                if target is not None and verdict.reason not in ("segment", "deferred"):
                    reasons = self.target_rejections[row.index, target]
                    verdict.reason = next((r for r in _REASONS if r in reasons), "no_witness")
                if self.oracle.has_generated_marker(self.message(row)) or row.identity_digest in self.s.pending_digests:
                    descriptor = next((r.target for r in verdict.relations if isinstance(r.target, Descriptor)), None)
                    verdict.relations.append(Relation("generated_unproven", descriptor))
                ref = NewRow(row.index)
                if self.mode != "off" and row.uid:
                    self.plan.uid_bindings.append((ref, row.uid, "version" if row.uid in bound_uids else "canonical",
                                                   "version_new" if row.uid in bound_uids else "stored_new"))
                    bound_uids.add(row.uid)
                left, right = self.bounds(row.index)
                if left is not None and right is not None and left >= right:
                    left, right = max(left, right), None
                if self.plan.placements:
                    previous = self.plan.placements[-1][1]
                    left = max(left or 0, previous.left or 0) or None
                    right = None if right is None or previous.right is None else max(right, previous.right)
                supersedes = next((self.ordinal(r.target) for r in verdict.relations if r.kind == "supersedes"), None)
                self.plan.placements.append((row.index, ledger.Placement(left, right, supersedes)))
            else:
                ref = verdict.target
                if isinstance(ref, Descriptor):
                    e = self.emission(ref)
                    if row.stamp is not None and e["observed_stamp"] is None:
                        self.plan.emission_stamps.append((ref, row.stamp))
                    if e["state"] == "proposed" and ref.view_seq not in self.plan.adoptions:
                        self.plan.adoptions.append(ref.view_seq)
                else:
                    b = self.s.bindings[ref]
                    if row.stamp is not None and row.stamp not in self.s.stamps(ref) and verdict.evidence in (
                            "run_keyed", "run_unkeyed", "rank1") and self.s.alt_stamp_counts.get(ref, 0) < 4:
                        self.plan.alt_stamps.append((ref, row.stamp))
                    if self.mode != "off" and row.uid and row.uid not in bound_uids:
                        self.plan.uid_bindings.append((b["store_id"], row.uid, "canonical", "anchor_replay"))
                        bound_uids.add(row.uid)
            if self.mode != "off":
                store_ref = ref if isinstance(ref, NewRow) else (
                    self.s.bindings[ref]["store_id"] if isinstance(ref, int) else ref)
                self.plan.uid_bindings.extend((store_ref, uid, "constituent", "absorbed") for uid in row.absorbed_uids)
            self.plan.entries.append(ledger.ViewEntry(row.identity_digest, row.stamp_digest, row.uidset_digest,
                                                       ref, isinstance(ref, Descriptor)))

    def cut(self, i):
        verdict = self.rows[i]
        if verdict.verdict == "replay":
            self.used.discard(verdict.target)
            self._placed = None
            self.first.setdefault(i, verdict.target)
            self.rows[i] = RowVerdict(reason="segment")

    def execute(self):
        self.run_search()
        self.choose_head()
        self.strong()
        # Step 5: non-segment rank-2 anchors must survive C11 before rank 3
        # evaluates their segments. The segment pass uses those final gaps.
        self.weak(segments=False)
        self.weak(segments=True)
        self.discovery()
        self.finish()
        return Classification(self.s.lineage, self.base,
                              Record("ingest", self.previous["view_seq"]) if self.previous else None,
                              self.run, self.head, self.s.coverage, self.counts, tuple(self.rows), self.plan)


def _lis_lengths(values):
    tails, result = [], []
    for value in values:
        index = bisect_left(tails, value)
        if index == len(tails):
            tails.append(value)
        else:
            tails[index] = value
        result.append(index + 1)
    return result


def classify(view, snapshot: LedgerSnapshot, oracle: IdentityOracle, *, uid_mode):
    """Return a plan, never a mutation. M14 replaces the first pass exactly once."""
    if uid_mode not in ("off", "shadow", "on"):
        raise ValueError("unknown uid mode")
    unreliable = snapshot.stamp_policy == "stamp_unreliable"
    first = _Pass(view, snapshot, oracle, uid_mode, unreliable)
    result = first.execute()
    proven, contributing_rows = set(), set()
    if not unreliable:
        for row in view:
            u = first.latest(row.uid)
            if uid_mode == "on" and u and first.eligible(u):
                for b in first.by_store.get(u["store_id"], ()):
                    if first.match(row, b) and row.stamp and snapshot.stamps(b["occ_id"]) and (
                            row.stamp not in snapshot.stamps(b["occ_id"])):
                        proven.add(b["occ_id"])
                        contributing_rows.add(row.index)
            for e in snapshot.emissions:
                if (e["eligible"] and e["output_index"] == row.index and e["descriptor_digest"] == row.identity_digest
                        and row.stamp and e["observed_stamp"] and row.stamp != e["observed_stamp"]):
                    proven.add(Descriptor(e["view_seq"], e["output_index"]))
                    contributing_rows.add(row.index)
        if len(proven) >= 3 and len(contributing_rows) >= 3:
            result = _Pass(view, snapshot, oracle, uid_mode, True).execute()
            result.writes.stamp_unreliable = True
            result.counters.update(first.counts)
            result.counters["m14.rerun"] = 1
    result.counters["queries"] = snapshot.query_count
    return result
