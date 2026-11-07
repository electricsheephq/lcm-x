"""Cold-ingest seam and independent occurrence oracle for the #2377 v7 proofs.

Only cursor/ledger selection is pinned. Storage, replay selection and relations
run through the real engine. ``new`` always contains HOST indexes, not keys: an
identical new occurrence must not be hidden by a stored replay of the same key.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import json

from hermes_lcm.ingest_ledger import stripped_replay_content
from tests.invariants.test_issue_2163_compacted_cursor import _poll

U = {"role": "user", "content": "start", "timestamp": 0.5}
V = {"role": "user", "content": "next", "timestamp": 1000.0}
X = {"role": "tool", "tool_call_id": "c", "content": "X"}
D = {"role": "tool", "tool_call_id": "d", "content": "D"}


def call(stamp=None, ids=("c",), args="{}"):
    return _poll(ids, args, stamp)


def key(row):
    """v7 scorer key, including tool fields and conversation; stamps excluded."""
    calls = row.get("tool_calls") or None
    if isinstance(calls, str):
        calls = json.loads(calls)
    content = row.get("content") or ""
    stripped = stripped_replay_content(content)
    return (row.get("conversation_id") or "conversation", row["role"],
            stripped if stripped is not None else content,
            row.get("tool_call_id") or None,
            json.dumps(calls, sort_keys=True), row.get("tool_name") or None)


def relations(engine):
    conn = engine._store._conn
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='message_relations'").fetchone():
        return []
    return list(conn.execute("SELECT store_id,kind,related_store_id FROM message_relations "
                             "WHERE kind IN ('possible_replay_of','link_pending')"))


@dataclass
class Run:
    engine: object
    before: list
    added: list
    inserted: dict
    skips: set
    host: list
    bound: frozenset = frozenset()  # ledger-bound HOST indexes (distinguishable by proof)

    def violations(self, new):
        """Return both invariant failures without short-circuiting the batch."""
        missing = sorted(set(new) - self.inserted.keys())
        by_id = {r["store_id"]: r for r in self.before + self.added}
        rels = relations(self.engine)
        links = {}
        pending = set()
        invalid = []
        for source, kind, target in rels:
            if kind == "link_pending":
                pending.add(source)
                continue
            links.setdefault(source, []).append(target)
            if (source not in by_id or target not in by_id or source == target
                    or key(by_id[source]) != key(by_id[target])):
                invalid.append((source, target, "cross-content/identity/conversation"))
        invalid += [(s, ts, "multiple links") for s, ts in links.items() if len(ts) > 1]
        originals = {key(r) for r in self.before}
        genuinely_new_ids = {self.inserted[i] for i in new if i in self.inserted}
        # Attribution. A replay copy the product CAN tell apart from its key-mates is checked per row:
        # a ledger-bound host occurrence, or a stamped copy that is unique among the added rows of its
        # key AND has an equal-stamp original (the evidence §2.5's first tier uses). It must carry the
        # link itself, and must link to its own equal-stamp original when exactly one exists. A stamped
        # copy that shares its stamp with other added key-mates AND has an equal-stamp original is
        # counted per (key, stamp) group: its members are interchangeable, but the group as a whole is
        # distinguishable from rows of other stamps, so another stamp's link never covers it. On a host
        # that re-stamps (originals X@2 and X@None, copies [@None, @2, @2] with host 0 new) this can flag a
        # correct plan; it fails closed (it flags, never hides) and no current test reaches it. Every
        # other replay row pools with its scorer KEY, across stamps: a stamp no original carries is no
        # evidence, so WHICH row of the key carries the link is not observable (NOTE §3 I-link) and the
        # scorer counts per key. Pooling's blind spot, shared with the gate scorer: within one pooled
        # group in one batch, a dropped new row and an unlinked replay cancel in the count (no text is
        # missing: the content is identical).
        bound_ids = {self.inserted[i] for i in self.bound if i in self.inserted}
        silent = []
        groups = {}
        for r in self.added:
            if key(r) not in originals:
                continue
            sid, stamp = r["store_id"], r.get("observed_at")
            own = [o["store_id"] for o in self.before
                   if stamp is not None and key(o) == key(r) and o.get("observed_at") == stamp]
            unique = stamp is not None and sum(
                key(o) == key(r) and o.get("observed_at") == stamp for o in self.added) == 1
            if sid in bound_ids or (unique and own):
                if sid not in genuinely_new_ids and sid not in links and sid not in pending:
                    silent.append(sid)
                if sid not in genuinely_new_ids and len(own) == 1 and links.get(sid, own) != own:
                    invalid.append((sid, links[sid], "replay not linked to its own equal-stamp original"))
                continue
            groups.setdefault((key(r), stamp) if own else (key(r),), []).append(sid)
        for rows in groups.values():
            replays = sum(sid not in genuinely_new_ids for sid in rows)
            attributed = [sid for sid in rows if sid in links or sid in pending]
            unattributed = [sid for sid in rows if sid not in links and sid not in pending]
            silent += sorted(unattributed)[:max(0, replays - len(attributed))]
        silent.sort()
        return dict(missing=missing, silent=silent, invalid=invalid)

    def check(self, new):
        assert_ceiling(self.engine)
        failures = self.violations(new)
        assert not any(failures.values()), failures

    def linked_copy_rate(self, indexes):
        linked = {s for s, kind, _target in relations(self.engine) if kind == "possible_replay_of"}
        return sum(self.inserted.get(i) in linked for i in indexes) / len(indexes)

    def floor(self, indexes):
        """A6/A7 allow either a skipped replay or a fully attributed copy."""
        stored = set(indexes) & self.inserted.keys()
        assert stored | (set(indexes) & self.skips) == set(indexes)
        return "linked" if stored else "skipped"


def assert_ceiling(engine):
    # This hook must expose the frozen, independently evaluated main predicate.
    # Never manufacture a ceiling by calling the candidate predicate twice.
    if hasattr(engine, "_last_tool_skip_sets"):
        main_skips, v7_skips = engine._last_tool_skip_sets
        assert set(v7_skips) <= set(main_skips)


def ingest_shape(build, monkeypatch, source, host, cursor=1, bindings=None,
                 prepare=None, configure=None):
    engine = build()
    if configure:
        configure(engine)
    engine.ingest(deepcopy(source))
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    before = engine._store.get_session_messages(engine._session_id)
    assert len(before) == len(source), "initial history must contain every true occurrence"
    if prepare:
        prepare(engine, before)
    engine.shutdown()
    engine = build()
    if configure:
        configure(engine)
    pairs = {i: before[j] for i, j in (bindings or {}).items()}
    monkeypatch.setattr(engine, "_reconcile_ingest_cursor_from_store", lambda *a, **kw: cursor)
    monkeypatch.setattr(engine, "_ledger_cold_cursor", lambda *a, **kw: dict(k=cursor, pairs=pairs))
    scans, inserted = [], {}
    scan, write = engine._replayed_tool_segment_indexes_after_cursor, engine._store_ingest_rows

    def tracked(messages, at, **kwargs):
        result = scan(messages, at, **kwargs)
        scans.append(set(result))
        return result

    def capture(**kwargs):
        ids = write(**kwargs)
        inserted.update((idx, sid) for (idx, _row), sid in zip(kwargs["messages_to_store_with_index"], ids))
        return ids

    monkeypatch.setattr(engine, "_replayed_tool_segment_indexes_after_cursor", tracked)
    monkeypatch.setattr(engine, "_store_ingest_rows", capture)
    engine.ingest(deepcopy(host))
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    after = engine._store.get_session_messages(engine._session_id)
    assert len(after) >= len(before), "durable history shrank"
    for old, current in zip(before, after):
        # Existing cold reconciliation may enrich an unknown host stamp. It must
        # preserve all payload/ownership fields and every already known stamp.
        enriched = {"observed_at", "observed_at_source"} if old["observed_at"] is None else set()
        assert {k: v for k, v in old.items() if k not in enriched} == {
            k: v for k, v in current.items() if k not in enriched}, "durable prefix changed"
    added = after[len(before):]
    assert scans, "the cold ingest never ran the tool-segment scan"
    assert Counter(inserted.values()) == Counter(r["store_id"] for r in added), "capture missed inserted rows"
    return Run(engine, before, added, inserted, scans[0] if scans else set(), host,
               frozenset(bindings or ()))
