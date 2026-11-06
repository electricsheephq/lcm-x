"""Store-wide P1 crash oracle, shared B2 surplus definition and content-free receipts."""

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

from bench.instruments.reliability.scorers.multiset import split_surplus
from hermes_lcm.message_content import normalize_content_value


def _value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _calls(value):
    if isinstance(value, str):
        value = json.loads(value)
    return _value(value) if value else None


def expect_from_view(rows, lineage):
    return {lineage: [(row["role"], normalize_content_value(row.get("content")), row.get("tool_call_id"),
                      _calls(row.get("tool_calls"))) for row in rows]}


def store_outcome(conn, expected):
    wanted = Counter((lineage, role, normalize_content_value(content), call_id, _calls(calls))
                     for lineage, rows in expected.items() for role, content, call_id, calls in rows)
    ids, stored = defaultdict(list), {}
    for sid, session, conversation, role, content, call_id, calls in conn.execute(
            "SELECT store_id,session_id,conversation_id,role,content,tool_call_id,tool_calls FROM messages"):
        ids[(conversation or session, role, content, call_id, _calls(calls))].append(sid)
        stored[sid] = (session, role, content)  # verifies stripped-content replay links, as the B2 scorer does
    relations = (list(conn.execute("SELECT store_id,kind,related_store_id FROM message_relations"))
                 if conn.execute("SELECT 1 FROM sqlite_master WHERE name='message_relations'").fetchone() else [])
    missing = wanted - Counter({key: len(rows) for key, rows in ids.items()})
    groups = [(key, max(0, len(rows) - wanted[key]), sorted(rows)) for key, rows in ids.items()]
    split = split_surplus(groups, relations, stored)
    return dict(loss=sum(missing.values()),
                loss_keys=[dict(sha256=hashlib.sha256(repr(key).encode()).hexdigest(), count=n)
                           for key, n in sorted(missing.items(), key=lambda item: repr(item[0]))],
                visible=split["visible_surplus_rows"], silent=split["silent_surplus_rows"],
                silent_store_ids=sorted({row["store_id"] for row in split["silent_surplus_examples"]}))


class GateFailure(AssertionError):
    """A gate cell that is expected to fail raises this; gate_result counts any strict-xfail cell as FAIL."""


def gate_result(receipts, *, strict_xfail_cells=()):
    """Pytest success with a strict xfail never makes the P1 gate pass."""
    return "FAIL" if strict_xfail_cells or any(
        row.get("result") == "FAIL" or row["outcome"]["loss"] or row["outcome"]["silent"]
        or row["cell"].get("strict_xfail", False) for row in receipts) else "PASS"


def record(cell, outcome, engine):
    output = os.environ.get("LCMX_P1_GATE_OUTPUT")
    if output:
        with engine._ingest_lock_guard:
            holds = [{key: sample[key] for key in ("route", "attempt", "outcome", "hold_ms", "view_rows")}
                     for sample in engine._ingest_lock_records]
        with Path(output).open("a") as stream:
            stream.write(json.dumps(dict(cell=cell, outcome=outcome,
                                         result=gate_result([dict(cell=cell, outcome=outcome)]), lock_records=holds,
                                         ingest_lock=engine.get_status()["ingest_lock"]), sort_keys=True) + "\n")


def install_former_commit_faults(engine, patch, boundary, fire):
    if boundary == "m":
        original = engine._rekey_durable_commit_proof

        def rekey(messages):
            before = engine._durable_commit_proof_payload()
            result = original(messages)
            assert before is not None and engine._durable_commit_proof_payload() != before
            fire("m")
            return result

        patch(engine, "_rekey_durable_commit_proof", rekey)
        return
    store = engine._store
    seed = store.get_session_messages("S")[0]["store_id"]
    host = [engine._identity_anchor_recovery["row"]]
    def wrap(target, name, point, after=None):
        original = getattr(target, name)

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            if after:
                after(*args, **kwargs)
            fire(point(*args, **kwargs) if callable(point) else point)
            return result

        patch(target, name, wrapped)

    capture = engine._capture_host_rewrites

    def rewrites(messages):
        capture(messages)
        if boundary == "b":
            row = store.get(seed)
            changed = {**messages[0], "content": " seed"}
            engine._host_rewrite_state()[0][seed] = (engine._message_replay_identity(row), row["content"], [changed])
            capture([changed])

    patch(engine, "_capture_host_rewrites", rewrites)
    wrap(engine, "_capture_host_rewrite", "b")
    reconcile = engine._reconcile_ingest_cursor_from_store
    rewrite_carry = store.write_metadata_json
    in_c = []

    def metadata(keys, *args, **kwargs):
        result = rewrite_carry(keys, *args, **kwargs)
        if in_c and any(key.startswith("compaction_commit_proof:") for key in keys):
            fire("c")
        return result

    patch(store, "write_metadata_json", metadata)

    def reconciled(*args, **kwargs):
        result = reconcile(*args, **kwargs)
        key = engine._replay_snapshot_metadata_key("compaction_commit_proof")
        store.write_metadata_json([key], json.dumps({"carry_ranges": []}))
        in_c.append(True)
        try:
            engine._rewrite_own_carry_ranges([], "transaction probe", [])
        finally:
            in_c.clear()
        return result

    patch(engine, "_reconcile_ingest_cursor_from_store", reconciled)
    backfill = store.backfill_observed_at_many

    def backfilled(pairs):
        result = backfill([*pairs, (seed, 10.)])
        fire("d")
        return result

    patch(store, "backfill_observed_at_many", backfilled)
    prematch = engine._identity_anchor_prematch

    def prematched(*args, **kwargs):
        if boundary == "e":
            engine._capture_merged_user_head({**host[0], "content": "seed\n\nextra", "_merged_turn_prefix": "seed"},
                                             [(store.get(seed), ())], set())
        return prematch(*args, **kwargs)

    patch(engine, "_identity_anchor_prematch", prematched)
    wrap(engine, "_record_ws_host_rewrite", "e")

    def anchor_evidence(plan, remainder_ids=None):
        store.add_message_relations([[(seed, "alt_stamp", None, None, 10.)]])
        store.write_metadata_json([engine._identity_anchor_carry_key()], "[]")

    wrap(engine, "_identity_anchor_commit", lambda plan, remainder_ids=None: "f" if remainder_ids is None else "i",
         anchor_evidence)
    wrap(store, "_append_protected_batch", "INSERT")
    for point, name in (("h", "_identity_anchor_mark_carried"), ("j", "_identity_anchor_record_versions")):
        def evidence(*args, _point=point, **kwargs):
            newest = store.get_session_tail("S", limit=1)[0]["store_id"]
            store.add_message_relations([[(newest, "carried_stamp" if _point == "h" else "supersedes",
                                          seed, None, 10.)]])
        wrap(engine, name, point, evidence)
    wrap(store, "_host_uid_write", "k")
    wrap(engine, "_persist_ingest_replay_evidence", "l")
    wrap(engine, "_rekey_durable_commit_proof", "m")


def prepare_rekey_case(root, shape="insert"):
    """#1286's real cleanup proof, then a row inserted below the proven end."""
    from tests.test_issue_1286_cleanup_rewrite_proof import CID, Host, _change, _commit_cleanup, _turns

    host = Host(root)
    _turns(host, "reasoning")
    _commit_cleanup(host, "reasoning")
    if shape == "field_chain_replace":
        host.turn()
        host.turn()
        expected = {CID: host.rows()}
        changed = _change(host, "replace", insert_at=0, replace_at=len(host.live) - 4)
        view = [dict(row) for row in host.live]
        added = [next(row for row in view if row["content"] == changed)]
    else:
        expected = {CID: host.rows()}
        view = [dict(row) for row in host.live]
        added = [{"role": "user", "content": "insert before cursor", "timestamp": host.stamp()}]
        view.insert(2, added[0])
    added += [{"role": "user", "content": "next turn", "timestamp": host.stamp()},
              {"role": "assistant", "content": "next reply"}]
    view += added[1:]
    expected[CID] += expect_from_view(added, CID)[CID]
    return host, view, expected

