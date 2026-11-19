"""#2856: replay identities of STORED rows that read no state are remembered too, and equal the unmemoised ones.

S1 oracle: rows inserted through the real store and read back (the store's own dict shape), every flag combination,
memoised == uncached on a miss and on a hit, one cell per kind of write between the rounds. Each state-write cell
also asserts that its write changed at least one (impure) row's uncached identity, so a cell cannot pass because
its write touched nothing. S2: an admitted stored row reads no state on a miss and on a hit. S3's mutants are
selected with LCMX_FIXEDWORK_MUTANT (see _MUTANTS); the default run applies none. Synthetic data only.
"""
import base64
import copy
import dataclasses
import gc
import inspect
import itertools
import json
import os
import pathlib
import random
import sqlite3
import sys
import textwrap
import threading

import pytest

import hermes_lcm.reconcile as reconcile
from hermes_lcm.dag import SummaryNode
from hermes_lcm.reconcile import ReconcileMixin, _HERMES_PRUNED_TOOL_PLACEHOLDER
from tests.invariants.test_fixedwork_identity_memo import (
    NEAR_MISSES, PROJECTION, SWITCH, TODO, WORD, _Raiser, _Touched, _engine, _memo, _raising_io, _evidence,
)
from tests.test_issue_10_persisted_output_spillover import _write_output
from tests.test_reconcile_oob_message_block import OOB_BLOCK

FLAGS = [dict(stored_row=s, with_host_rewrite=w, strip_carrier=c, allow_pruned_tool_alias=a)
         for s, w, c, a in itertools.product((True, False), repeat=4)]
INGEST = "[Externalized LCM ingest payload: 900 chars; ref={ref}]"
EXTERNAL = "[Externalized tool output: tool=read_file; tool_call_id={cid}; chars=900; ref={ref}]"
GC = "[GC'd externalized tool output: tool_call_id={cid}; chars=9; ref={ref}]"
SUMMARY = "[Recent Summary (d0, node {node})]\nEarlier turns.\n[Expand for details: turns]\n\nthe next request {i}"
MUTANT = os.environ.get("LCMX_FIXEDWORK_MUTANT", "")


# --- S3 mutants: each replaces one piece of the memo, in this process only --------------------------------------
class _KeyDropping(dict):
    """A memo entry table that ignores some key slots: the memo's key without those fields."""
    drop = ()

    def _k(self, key):
        return tuple(None if i in self.drop else part for i, part in enumerate(key))

    def get(self, key, default=None):
        return super().get(self._k(key), default)

    def __setitem__(self, key, value):
        super().__setitem__(self._k(key), value)


def _drop_key_slots(patch, slots):
    original = reconcile._ReplayIdentityMemo.__init__

    def init(self, epoch):
        original(self, epoch)
        self.entries = type("_Dropping", (_KeyDropping,), {"drop": slots})()
    patch.setattr(reconcile._ReplayIdentityMemo, "__init__", init)


def _gate_without(patch, kind):
    patch.setattr(reconcile, "_REPLAY_IDENTITY_MEMO_GATE",
                  tuple(entry for entry in reconcile._REPLAY_IDENTITY_MEMO_GATE if entry[0] != kind))


def _walk_without_dict_keys(patch):
    def walk(frozen):
        strings, size, stack = [], 0, [frozen]
        while stack:
            node = stack.pop()
            tag, size = node[0], size + sys.getsizeof(node)
            if tag == "d":
                stack.extend(value for _key, value in node[1])  # the mutation: keys are not scanned
            elif tag in ("l", "t"):
                stack.extend(node[1])
            elif tag == "s":
                strings.append(node[1])
        return strings, size
    patch.setattr(reconcile, "_frozen_replay_identity_walk", walk)


def _wrapper_admitting_host_rewrite(patch):
    """The wrapper's own source with with_host_rewrite neither declined nor keyed (it still computes with it)."""
    source = textwrap.dedent(inspect.getsource(ReconcileMixin._message_replay_identity))
    mutated = source.replace("if (with_host_rewrite or ", "if (", 1).replace(
        "stored_row=stored_row, strip_carrier=strip_carrier", "stored_row=stored_row, with_host_rewrite=with_host_rewrite,"
        " strip_carrier=strip_carrier")
    assert mutated.count("with_host_rewrite=with_host_rewrite,") > source.count("with_host_rewrite=with_host_rewrite,")
    assert "if (with_host_rewrite or " in source and "if (with_host_rewrite or " not in mutated
    namespace = {}
    exec(compile(mutated, reconcile.__file__, "exec"), vars(reconcile), namespace)
    patch.setattr(ReconcileMixin, "_message_replay_identity", namespace["_message_replay_identity"])


_MUTANTS = {
    "gate_without_ingest_placeholder": lambda patch: _gate_without(patch, "ingest_placeholder"),
    "gate_without_dict_key_scan": _walk_without_dict_keys,
    "gate_without_externalized": lambda patch: _gate_without(patch, "externalized"),
    "key_without_stored_row": lambda patch: _drop_key_slots(patch, (0,)),
    "key_without_content": lambda patch: _drop_key_slots(patch, (4,)),
    "memo_admits_with_host_rewrite": _wrapper_admitting_host_rewrite,
    # Combined: the masking kind removed too, to show what the single mutant's kind guards on its own.
    "gate_without_ingest_placeholder_and_externalized": lambda patch: (
        _gate_without(patch, "ingest_placeholder"), _gate_without(patch, "externalized")),
    "key_without_stored_row_and_gate_without_oob": lambda patch: (
        _drop_key_slots(patch, (0,)), _gate_without(patch, "oob")),
}


@pytest.fixture(autouse=True)
def _mutant(monkeypatch):
    if MUTANT:
        _MUTANTS[MUTANT](monkeypatch)
    yield


# --- rows through the real store -------------------------------------------------------------------------------
class _Rows:
    """Appends rows of every class through the engine's store and remembers their store ids and kinds."""

    def __init__(self, engine, rng, tmp_path):
        self.engine, self.rng, self.tmp_path, self.ids, self.kinds, self.i = engine, rng, tmp_path, [], {}, 0
        self.refs = []  # (ref, kind of payload) a payload write may create

    def add(self, kind, msg, session="S", conversation="conv"):
        try:
            store_id = self.engine._store.append(session, msg, conversation_id=conversation)
        except Exception as error:  # a row the store refuses (recorded, never silently skipped)
            self.kinds.setdefault("refused:" + kind, []).append(type(error).__name__)
            return None
        self.ids.append(store_id)
        self.kinds[store_id] = kind
        return store_id

    def ref(self, payload_kind):
        self.i += 1
        ref = f"lcm_000000{self.i:06d}_pr3.json"
        self.refs.append((ref, payload_kind))
        return ref

    def text(self):
        self.i += 1
        return f"row {self.i % 40}" + WORD * self.rng.randrange(0, 4)

    def fill(self, n):
        rng = self.rng
        for shape in [rng.randrange(22) for _ in range(n)] + list(range(22)):  # every shape at least once
            text = self.text()
            if shape == 0:
                self.add("pure", {"role": rng.choice(["user", "assistant"]), "content": text})
            elif shape == 1:
                self.add("structured", {"role": "user", "content": [{"type": "text", "text": text},
                                                                   {"type": "image_url", "image_url": "x"}]})
            elif shape == 2:
                args = rng.choice([True, 1, 1.0, None, -0.0, "x"])
                calls = [{"id": f"c{self.i}", "type": "function", "function": {"name": "f", "arguments": rng.choice(
                    [json.dumps({"a": [1, {"b": args}]}), {"k": args, "nested key": {"x y": args}}, "{bad"])}}]
                self.add("calls", {"role": "assistant", "content": text, "tool_calls": calls})
            elif shape == 3:
                self.add("tool", {"role": "tool", "tool_call_id": f"t{self.i % 5}", "tool_name": rng.choice(
                    ["read_file", "grep"]), "content": text})
            elif shape == 4:
                self.add("scaffold", {"role": "user", "content": rng.choice([TODO, SWITCH])})
            elif shape == 5:
                self.add("near_miss", {"role": rng.choice(["user", "tool", "assistant"]), "tool_call_id": "t1",
                                       "content": rng.choice(NEAR_MISSES) + " " + text})
            elif shape == 6:
                self.add("surrogate", {"role": "user", "content": text + " \ud800 lone"})
            elif shape == 7:
                self.add("ingest_content", {"role": "user", "content": "see " + INGEST.format(
                    ref=self.ref("ingest_payload"))})
            elif shape == 8:  # a marker in a tool-call dict KEY (the restore rewrites keys)
                key = INGEST.format(ref=self.ref("ingest_payload"))
                self.add("ingest_dict_key", {"role": "assistant", "content": text, "tool_calls": [
                    {"id": f"k{self.i}", "type": "function", "function": {"name": "f", "arguments": {key: 1}}}]})
            elif shape == 9:
                self.add("ingest_call_string", {"role": "assistant", "content": "", "tool_calls": [
                    {"id": f"v{self.i}", "type": "function", "function": {"name": "write_file", "arguments": json.dumps(
                        {"content": INGEST.format(ref=self.ref("ingest_payload"))})}}]})
            elif shape == 10:
                cid = f"e{self.i}"
                self.add("externalized", {"role": "tool", "tool_call_id": cid, "tool_name": "read_file",
                                          "content": EXTERNAL.format(cid=cid, ref=self.ref("tool_result"))})
            elif shape == 11:
                cid = f"g{self.i}"
                self.add("gc", {"role": "tool", "tool_call_id": cid, "content": GC.format(
                    cid=cid, ref=self.ref("tool_result"))})
            elif shape == 12:
                self.add("oob", {"role": "tool", "tool_call_id": f"o{self.i}", "content": text + OOB_BLOCK})
            elif shape == 13:
                self.add("pruned", {"role": "tool", "tool_call_id": f"p{self.i}", "tool_name": "read_file",
                                    "content": _HERMES_PRUNED_TOOL_PLACEHOLDER})
            elif shape == 14:
                self.add("summary", {"role": "user", "content": SUMMARY.format(node=rng.randrange(1, 4), i=self.i)})
            elif shape == 15:
                marker = _write_output(self.tmp_path / "home" / "cache" / "spillover" / f"s{self.i}.txt")
                self.add("persisted", {"role": "tool", "tool_call_id": f"s{self.i}", "content": marker})
            elif shape == 16:
                self.add("projection", {"role": "user", "content": PROJECTION.format(sid=self.ids[0] if self.ids else 1)})
            elif shape == 17:  # a row whose session falls back to the bound one (the restore's session check)
                self.add("ingest_unbound_session", {"role": "user", "content": INGEST.format(
                    ref=self.ref("ingest_payload_S"))}, session="")
            elif shape == 18:  # large: the store's own protection externalizes it (a real ingest placeholder)
                long = base64.b64encode(bytes((j * 37 + self.i) % 256 for j in range(3420))).decode("ascii")
                self.add("real_ingest", {"role": "assistant", "content": "", "tool_calls": [
                    {"id": f"w{self.i}", "type": "function", "function": {"name": "write_file", "arguments": json.dumps(
                        {"content": "hdr\n" + "\n".join(long[j:j + 76] for j in range(0, len(long), 76))})}}]})
            elif shape == 19:  # rows differing only in non-content keys: session, conversation, timestamp, store id
                msg = {"role": "user", "content": f"twin {self.i % 7}", "timestamp": rng.choice([1.0, 2.0])}
                for session, conversation in (("S", "conv"), ("S2", "conv"), ("S", "conv2")):
                    self.add("twin", dict(msg), session=session, conversation=conversation)
            elif shape == 20 and self.ids:  # a duplicate of an earlier row (a new store id, same bytes)
                row = self.engine._store.get(rng.choice(self.ids))
                self.add("duplicate", {key: row[key] for key in ("role", "content", "tool_call_id", "tool_calls",
                                                                  "tool_name") if row.get(key) is not None})
            else:
                self.add("host_rewrite_target", {"role": "user", "content": "rewritable " + text})

    def read(self):
        """Every row read back twice over: by store id (get) and in session order (get_session_messages)."""
        store = self.engine._store
        rows = [store.get(store_id) for store_id in self.ids]
        for session in ("S", "S2", ""):
            rows += store.get_session_messages(session)
        return [row for row in rows if row]


def _admitted(row, stored_row):
    if type(row) is not dict:
        return False
    try:
        key, frozen = reconcile._replay_identity_memo_key(row, True, True, stored_row)
    except (TypeError, RecursionError):
        return False
    return not reconcile._replay_identity_memo_gate(key[3], key[4], frozen)


class _Calls:
    """Counts uncached computations (stored and live), wrapping the attribute the memo wrapper calls."""

    def __init__(self, monkeypatch):
        self.n, self.original = 0, ReconcileMixin._message_replay_identity_uncached

        def counted(engine, msg, **kwargs):
            self.n += 1
            return self.original(engine, msg, **kwargs)
        monkeypatch.setattr(ReconcileMixin, "_message_replay_identity_uncached", counted)


def _expected(calls, engine, row, flags):
    return calls.original(engine, copy.deepcopy(row), **flags)


def _check(calls, engine, rows, stats, label):
    """memoised == uncached for every row x flag, twice; an admitted call's second call is a hit."""
    out = {}
    for index, row in enumerate(rows):
        for f, flags in enumerate(FLAGS):
            expected = _expected(calls, engine, row, flags)
            assert engine._message_replay_identity(row, **flags) == expected, (label, "miss", row, flags)
            before = calls.n
            assert engine._message_replay_identity(row, **flags) == expected, (label, "hit", row, flags)
            if not flags["with_host_rewrite"] and _admitted(row, flags["stored_row"]):
                assert calls.n == before, (label, "admitted row computed twice", row, flags)
                stats["hits"] += 1
            stats["checks"] += 1
            out[(index, f)] = expected
    return out


def _db_path(engine):
    return next(row[2] for row in engine._store._conn.execute("PRAGMA database_list") if row[1] == "main")


def _payload(engine, ref, payload_kind, content):
    directory = pathlib.Path(engine._config.large_output_externalization_path)
    directory.mkdir(parents=True, exist_ok=True)
    body = {"kind": "ingest_payload" if payload_kind.startswith("ingest") else "tool_result", "content": content,
            "field_path": "content"}
    if payload_kind == "ingest_payload_S":
        body["session_id"] = "S"
    (directory / ref).write_text(json.dumps(body), encoding="utf-8")


def _update(engine, sql, args):
    store = engine._store
    with store._write_lock:
        store._conn.execute(sql, args)
        store._conn.commit()


def _new_content(rows, rng):
    return rng.choice([f"updated {rng.randrange(9)}", "updated " + OOB_BLOCK, INGEST.format(ref=rows.refs[0][0]),
                       f"updated twin {rng.randrange(3)}"])


# Each write: (rows, rng, round) -> None. Writes aimed at the rows' identity inputs or at state an identity reads.
def _w_content_update(rows, rng, r):
    for store_id in rng.sample(rows.ids, 6):
        _update(rows.engine, "UPDATE messages SET content = ? WHERE store_id = ?", (_new_content(rows, rng), store_id))


def _w_delete_reinsert(rows, rng, r):
    for store_id in rng.sample(rows.ids, 6):
        old = rows.engine._store.get(store_id)
        _update(rows.engine, "DELETE FROM messages WHERE store_id = ?", (store_id,))
        _update(rows.engine, "INSERT INTO messages (store_id, session_id, conversation_id, role, content, tool_call_id,"
                " tool_calls, tool_name, timestamp, token_estimate, pinned, ingested_at) VALUES"
                " (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 1.0)",
                (store_id, old["session_id"], old["conversation_id"], old["role"], _new_content(rows, rng),
                 old["tool_call_id"], None, old["tool_name"], old["timestamp"]))


def _w_tool_calls_rewrite(rows, rng, r):
    key = INGEST.format(ref=rows.refs[0][0])
    for store_id in rng.sample(rows.ids, 6):
        calls = rng.choice([[{"id": "r", "type": "function", "function": {"name": "f", "arguments": {key: r}}}],
                            [{"id": "r", "type": "function", "function": {"name": "f", "arguments": {"plain": r}}}],
                            [{"id": "r", "type": "function", "function": {"name": "g", "arguments": "{}"}}]])
        _update(rows.engine, "UPDATE messages SET tool_calls = ? WHERE store_id = ?", (json.dumps(calls), store_id))


def _w_payload_written(rows, rng, r):
    for ref, payload_kind in rows.refs[r::3]:
        _payload(rows.engine, ref, payload_kind, f"restored payload {ref}")


def _w_payload_changed(rows, rng, r):
    for ref, payload_kind in rows.refs:
        _payload(rows.engine, ref, payload_kind, f"changed payload {ref} round {r}")


def _w_payload_deleted(rows, rng, r):
    directory = pathlib.Path(rows.engine._config.large_output_externalization_path)
    for ref, _kind in rows.refs[r::3]:
        if (directory / ref).exists():
            (directory / ref).rename(directory.parent / f"gone-{ref}")  # gone from the loader's view


def _w_dag_insert(rows, rng, r):
    node = rows.engine._dag.add_node(SummaryNode(session_id="S", depth=0, summary="Earlier turns.", source_ids=[
        rows.ids[0]], source_type="messages", expand_hint="turns", created_at=1.0))
    assert node == r + 1  # the summary rows name nodes 1 to 3


def _w_host_rewrite_override(rows, rng, r):
    engine = rows.engine
    for store_id in [i for i in rows.ids if rows.kinds.get(i) == "host_rewrite_target"]:
        stored = reconcile.normalize_content_value(engine._store.get(store_id)["content"]) or ""
        payload = {"version": 1, "content": f"host form {r} {store_id}", "stripped": ["", ""],
                   "stored_sha256": reconcile.hashlib.sha256(stored.encode()).hexdigest()}
        engine._store.write_metadata_json([f"host_rewrite_identity:{store_id}"], json.dumps(payload, sort_keys=True))
        engine._host_rewrite_state()[1].pop(store_id, None)  # the engine reloads it on the next read


def _w_oob_durable(rows, rng, r):
    for store_id in [i for i in rows.ids if rows.kinds.get(i) == "oob"]:
        row = rows.engine._store.get(store_id)
        rows.engine._store.append(rows.engine._session_id, {"role": "tool", "tool_call_id": row["tool_call_id"],
                                                            "content": row["content"]}, conversation_id="conv")


def _w_pruned_source(rows, rng, r):
    for store_id in [i for i in rows.ids if rows.kinds.get(i) == "pruned"]:
        row = rows.engine._store.get(store_id)
        rows.engine._store.append(rows.engine._session_id, {"role": "tool", "tool_call_id": row["tool_call_id"],
                                  "tool_name": "read_file", "content": f"source {r}" + WORD * 3}, conversation_id="conv")


def _w_config_swap(rows, rng, r):
    engine = rows.engine
    other = rows.tmp_path / f"ext-{r}"  # another payload directory, holding different payloads for the same refs
    engine._config = dataclasses.replace(engine._config, large_output_externalization_path=str(other))
    for ref, payload_kind in rows.refs:
        _payload(engine, ref, payload_kind, f"payload of directory {r} {ref}")


def _w_rebind(rows, rng, r):
    rows.engine.on_session_start(rows.engine._session_id, platform="cli", context_length=200_000,
                                 conversation_id="conv")


def _w_session_switch(rows, rng, r):
    other = "S2" if rows.engine._session_id == "S" else "S"
    rows.engine.on_session_start(other, platform="cli", context_length=200_000, conversation_id="conv")


def _w_second_connection(rows, rng, r):
    conn = sqlite3.connect(_db_path(rows.engine), timeout=30)
    try:
        for store_id in rng.sample(rows.ids, 6):
            conn.execute("UPDATE messages SET content = ? WHERE store_id = ?", (_new_content(rows, rng), store_id))
        target = next(i for i in rows.ids if rows.kinds.get(i) == "host_rewrite_target")
        stored = reconcile.normalize_content_value(rows.engine._store.get(target)["content"]) or ""
        conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (
            f"host_rewrite_identity:{target}", json.dumps({"version": 1, "content": f"other connection {r}",
                                                           "stripped": ["", ""], "stored_sha256": reconcile.hashlib.sha256(
                                                               stored.encode()).hexdigest()}, sort_keys=True)))
        conn.commit()
    finally:
        conn.close()
    rows.engine._host_rewrite_state()[1].pop(target, None)


def _w_rollback(rows, rng, r):
    """A write the reads see inside its transaction, then rolled back: identities are taken inside the transaction
    (they may fill the memo with the uncommitted rows' identities) and must match the rolled-back rows after."""
    store = rows.engine._store
    with store._write_lock:
        store._conn.execute("BEGIN IMMEDIATE")
        for store_id in rng.sample(rows.ids, 6):
            store._conn.execute("UPDATE messages SET content = ? WHERE store_id = ?", (_new_content(rows, rng), store_id))
        inside = [store.get(store_id) for store_id in rows.ids]
        for row in inside:
            for flags in FLAGS[:4]:
                rows.engine._message_replay_identity(row, **flags)
        store._conn.execute("ROLLBACK")


WRITES = {
    "content_update": (_w_content_update, True), "delete_reinsert": (_w_delete_reinsert, True),
    "tool_calls_rewrite": (_w_tool_calls_rewrite, True), "payload_written": (_w_payload_written, True),
    "payload_changed": (_w_payload_changed, True), "payload_deleted": (_w_payload_deleted, True),
    "dag_insert": (_w_dag_insert, True), "host_rewrite_override": (_w_host_rewrite_override, True),
    "oob_durable": (_w_oob_durable, True), "pruned_source": (_w_pruned_source, True),
    "config_swap": (_w_config_swap, True), "rebind": (_w_rebind, False), "session_switch": (_w_session_switch, True),
    "second_connection": (_w_second_connection, True), "rollback": (_w_rollback, False),
}


@pytest.mark.parametrize("write", sorted(WRITES))
def test_s1_stored_identity_equals_the_uncached_one_across_writes(tmp_path, monkeypatch, write):
    apply, changes_state = WRITES[write]
    calls, rng = _Calls(monkeypatch), random.Random(f"pr3-{write}")
    stats = dict(checks=0, hits=0, changed=0, rows=0)
    with _engine(tmp_path) as engine:
        rows = _Rows(engine, rng, tmp_path)
        rows.fill(160)
        for kind in ("host_rewrite_target", "oob", "pruned", "summary", "ingest_dict_key", "ingest_content"):
            assert kind in rows.kinds.values(), kind  # every cell's target exists
        if write in ("payload_changed", "payload_deleted"):
            for ref, payload_kind in rows.refs:
                _payload(engine, ref, payload_kind, f"first payload {ref}")
        for r in range(3):
            current = rows.read()
            stats["rows"] += len(current)
            before = _check(calls, engine, current, stats, (write, r, "before"))
            apply(rows, rng, r)
            # The same dicts again (the state they read may have changed under them), then fresh reads.
            after = _check(calls, engine, current, stats, (write, r, "same dicts"))
            stats["changed"] += sum(after[k] != before[k] for k in before)
            _check(calls, engine, rows.read(), stats, (write, r, "re-read"))
            stats["changed"] += sum(1 for a, b in zip(current, rows.read()) if a != b)
    if changes_state:
        assert stats["changed"] > 0, ("the write changed no row's identity inputs or identity", write)
    else:
        assert stats["changed"] == 0, write
    assert stats["hits"] > 0
    _evidence(f"s1_{write}.json", dict(stats, kinds=sorted({str(k) for k in rows.kinds.values()})))


def test_s1_live_row_with_a_marker_in_a_tool_call_dict_key_is_not_remembered(tmp_path, monkeypatch):
    """The gate scans tool-call dict keys for live rows too."""
    calls = _Calls(monkeypatch)
    key = INGEST.format(ref="lcm_000000000001_k.json")
    row = {"role": "assistant", "content": "x", "tool_calls": [{"id": "k", "type": "function", "function": {
        "name": "f", "arguments": {key: 1}}}]}
    with _engine(tmp_path) as engine:
        assert "ingest_placeholder" in reconcile._replay_identity_memo_gate(
            "assistant", "x", reconcile._freeze_replay_identity_value(row["tool_calls"]))
        for flags in FLAGS:
            assert engine._message_replay_identity(row, **flags) == _expected(calls, engine, row, flags)
        assert not any(key[6] == reconcile._freeze_replay_identity_value(row["tool_calls"]) for key in _memo(engine))


def test_s1_stored_call_whose_key_does_not_build_keeps_its_stored_identity(tmp_path, monkeypatch):
    """Fallbacks pass stored_row on: an OOB tool row is stripped as a stored row and kept as a live one."""
    calls = _Calls(monkeypatch)

    class Text(str):
        pass
    with _engine(tmp_path) as engine:
        engine._message_replay_identity({"role": "user", "content": "warm"})
        for row in ({"role": "tool", "tool_call_id": "o1", "content": "out" + OOB_BLOCK, "tool_calls": [{"f": {1}}]},
                    {"role": "tool", "tool_call_id": "o1", "content": Text("out" + OOB_BLOCK)},
                    {"role": "tool", "tool_call_id": "o1", "content": "out" + OOB_BLOCK, "session_id": Text("S")}):
            stored = engine._message_replay_identity(row, stored_row=True)
            assert stored == _expected(calls, engine, row, dict(stored_row=True))
            assert stored != engine._message_replay_identity(row, stored_row=False)  # the two identities differ here


GATE_KINDS = {"ingest_content": "ingest_placeholder", "ingest_dict_key": "ingest_placeholder",
              "ingest_call_string": "ingest_placeholder", "ingest_unbound_session": "ingest_placeholder",
              "real_ingest": "ingest_placeholder", "externalized": "externalized", "gc": "externalized",
              "oob": "oob", "pruned": "pruned", "summary": "summary", "persisted": "persisted",
              "projection": "projection"}


def test_s1_structure_gate_names_each_stored_kind_and_keys_stored_and_live_apart(tmp_path, monkeypatch):
    """Structural cells (they pin the shape, not a result): each impure stored row carries its own kind in the
    gate's verdict, and a live fill never serves a stored call (``stored_row`` is part of the key)."""
    calls = _Calls(monkeypatch)
    with _engine(tmp_path) as engine:
        rows = _Rows(engine, random.Random("pr3-structure"), tmp_path)
        rows.fill(300)
        by_id = {row["store_id"]: row for row in rows.read()}
        for store_id, kind in rows.kinds.items():
            if kind in GATE_KINDS and store_id in by_id:
                row = by_id[store_id]
                key, frozen = reconcile._replay_identity_memo_key(row, True, True, True)
                # A row the store itself externalized on insert (over the 200-char line) carries that kind instead.
                expected = "externalized" if key[4].startswith("[Externalized tool output:") else GATE_KINDS[kind]
                assert expected in reconcile._replay_identity_memo_gate(key[3], key[4], frozen), (kind, row)
        separate, seen = 0, set()
        for row in by_id.values():
            if _admitted(row, True) and reconcile._replay_identity_memo_key(row, True, True, True)[0] not in seen:
                seen.add(reconcile._replay_identity_memo_key(row, True, True, True)[0])
                engine._message_replay_identity(row, stored_row=False)
                before = calls.n
                engine._message_replay_identity(row, stored_row=True)
                assert calls.n == before + 1, row
                separate += 1
        assert separate > 50


# --- S2: reads no state ----------------------------------------------------------------------------------------
def _in_wrapper(owner):
    """True on the identity's thread while the memo wrapper (and so a hit) is on the stack."""
    if threading.get_ident() != owner:
        return False
    codes = {ReconcileMixin._message_replay_identity.__code__, _Calls.original_code}
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code in codes:
            return True
        frame = frame.f_back
    return False


_Calls.original_code = ReconcileMixin._message_replay_identity_uncached.__code__


class _WrapperRaiser(_Raiser):
    """PR1's raiser, scoped to the memo wrapper instead of the uncached body: a hit's reads are caught too."""

    def __getattribute__(self, attr):
        if not _in_wrapper(object.__getattribute__(self, "_owner")):
            return getattr(object.__getattribute__(self, "_real"), attr)
        object.__getattribute__(self, "_log").append((object.__getattribute__(self, "_name"), attr, "wrapper"))
        raise _Touched(attr)


def test_s2_admitted_stored_rows_read_no_state_on_a_miss_and_on_a_hit(tmp_path, monkeypatch):
    rng = random.Random("pr3-s2")
    with _engine(tmp_path) as engine:
        rows = _Rows(engine, rng, tmp_path)
        rows.fill(600)
        for ref, payload_kind in rows.refs:  # payloads exist: an impure row the gate admitted would read them
            _payload(engine, ref, payload_kind, f"payload {ref}")
        flags = [f for f in FLAGS if f["stored_row"] and not f["with_host_rewrite"]]
        admitted = [row for row in rows.read() if _admitted(row, True)]
        normal = [[ReconcileMixin._message_replay_identity_uncached(engine, row, **f) for f in flags]
                  for row in admitted]
        engine.__dict__.pop("_replay_identity_memo", None)
        engine._message_replay_identity({"role": "user", "content": "warm"})
        store, dag, config, log, computed = engine._store, engine._dag, engine._config, [], []
        with monkeypatch.context() as patch:
            owner = threading.get_ident()
            # The raisers go in before the fill: the epoch holds the config object's identity, so a config swapped
            # in after the fill would drop the memo and turn the hit into a miss.
            engine._store, engine._dag, engine._config = (
                _WrapperRaiser(n, log, real, owner) for n, real in (("store", store), ("dag", dag), ("config", config)))
            _raising_io(patch, log, owner)
            original = ReconcileMixin._message_replay_identity_uncached

            def counted(engine_, msg, **kwargs):
                computed.append(1)
                return original(engine_, msg, **kwargs)
            patch.setattr(ReconcileMixin, "_message_replay_identity_uncached", counted)
            gc.disable()
            try:
                misses = [[engine._message_replay_identity(row, **f) for f in flags] for row in admitted]
                filled = len(computed)
                hits = [[engine._message_replay_identity(copy.deepcopy(row), **f) for f in flags] for row in admitted]
            finally:
                gc.enable()
                engine._store, engine._dag, engine._config = store, dag, config
        assert log == []
        assert misses == normal and hits == normal
        assert len(computed) == filled  # every second call was a hit: nothing computed, nothing read
    assert len(admitted) >= 300
    _evidence("s2_reads_no_state.json", dict(admitted_rows=len(admitted), flag_sets=len(flags), state_reads=len(log),
                                             miss_computations=filled, hit_computations=len(computed) - filled))
