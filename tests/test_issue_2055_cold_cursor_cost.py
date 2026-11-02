"""P1d T10: bounded indexed reads and the whole-ingest lock budget."""
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

from hermes_lcm import ingest_ledger as ledger
from tests.test_host_uid_shadow import _m
from tests.test_issue_2055_cold_cursor import candidate, force_today, prepare
from tests.test_issue_2055_ingest_ledger_writes import engines as _engines

engines = _engines


def quantiles(values):
    values = sorted(values)
    return dict(n=len(values), p50_ms=values[len(values) // 2], p90_ms=values[int(len(values) * .9)],
                p99_ms=values[min(len(values) - 1, int(len(values) * .99))], max_ms=max(values))


def test_t10_indexed_candidate_and_whole_ingest_lock_hold(engines, monkeypatch):
    engine = engines()
    view = [_m("user" if i % 2 == 0 else "assistant", f"synthetic row {i}", float(i + 1)) for i in range(2000)]
    attempt = prepare(engine, view)
    conn, lineage = engine._store._conn, attempt["lineage"]
    with engine._store.atomic(immediate=True):
        conn.executemany("INSERT INTO messages(store_id,session_id,conversation_id,role,content,timestamp,observed_at) "
                         "VALUES (?,'S','conv',?,?,?,?)",
                         [(i + 1, view[i]["role"], view[i]["content"], float(i + 1), float(i + 1)) if i < 2000
                          else (i + 1, "assistant", "old synthetic history", float(i + 1), float(i + 1))
                          for i in range(50000)])
        ledger.insert_bindings(conn, lineage, [dict(
            ordinal=(i + 1) * ledger.ORDINAL_STEP, store_id=i + 1,
            identity_digest=attempt["entries"][i].identity_digest if i < 2000 else b"z" * 16,
            stamp=ledger.stamp_text(float(i + 1)), disposition="stored", view_seq=1) for i in range(50000)])
        ledger.write_view(conn, attempt["salt"], lineage, "ingest", None,
                          [e._replace(ref=i + 1) for i, e in enumerate(attempt["entries"])])
    times = []
    for _ in range(50):
        with engine._store.atomic(immediate=True):
            start = perf_counter()
            run, _counts = ledger.cold_cursor_candidate(
                conn, session_id="S", attempt=attempt, ignored=[False] * 2000)
            times.append((perf_counter() - start) * 1000)
        assert run["k"] == 2000

    holds, raw_holds = {}, {}
    forbidden_calls = []
    for enabled in (False, True):
        measurements = []
        with monkeypatch.context() as patch:
            force_today(engine, patch)
            if not enabled:
                patch.setattr(engine, "_ledger_cold_cursor", lambda *a: dict(k=0, pairs={}))
            else:
                def forbidden(*a, **kw):
                    forbidden_calls.append((a, kw))
                    raise AssertionError("today=0 must not activate tool suppression or prefix backfill")
                patch.setattr(engine, "_replayed_tool_segment_indexes_after_cursor", forbidden)
                patch.setattr(engine, "_identity_anchor_backfill_prefix", forbidden)
            for _ in range(50):
                engine._ingest_cursor_needs_reconcile = True
                engine._ingest_messages([dict(row) for row in view])
                measurements.append(engine._last_ingest_attribution["lock_hold_ms"])
        label = "on" if enabled else "off"
        holds[label], raw_holds[label] = quantiles(measurements), measurements
    assert not forbidden_calls

    result = dict(candidate_read=quantiles(times), whole_ingest=holds, view_rows=2000, lineage_bindings=50000,
                  raw_candidate_ms=times, raw_whole_ingest_ms=raw_holds,
                  targets=dict(candidate_p90_ms=10, whole_p90_ms=50, whole_p99_ms=250))
    result["within_targets"] = (result["candidate_read"]["p90_ms"] <= 10
                                and holds["on"]["p90_ms"] <= 50 and holds["on"]["p99_ms"] <= 250)
    print(json.dumps(result, sort_keys=True))
    output = os.environ.get("LCMX_P1D_TIMING_OUTPUT")
    if output:
        Path(output).write_text(json.dumps(result, indent=2) + "\n")
    assert engine._store.get_session_count("S") == 50000


def test_t10_repeated_digest_and_raw_part_rows_exhaust_one_budget(engines):
    engine = engines()
    view = [_m("user", "ok", 1.)]
    engine._ingest_messages(view)
    attempt, conn = prepare(engine, view), engine._store._conn
    lineage, digest = attempt["lineage"], attempt["entries"][0].identity_digest
    with engine._store.atomic(immediate=True):
        ledger.insert_bindings(conn, lineage, [dict(
            ordinal=(i + 2) * ledger.ORDINAL_STEP, store_id=1, identity_digest=digest, stamp="1.0",
            disposition="stored", view_seq=1) for i in range(5000)])
        ledger.write_view(conn, attempt["salt"], lineage, "ingest", None, attempt["entries"])
    run, counts = candidate(engine, view)
    assert run["k"] == 0 and counts["stop.budget"] == 1
    with engine._store.atomic(immediate=True):
        # Composite/split rows must count BEFORE eligibility filtering and LIMIT.
        conn.execute("UPDATE occurrence_bindings SET disposition='composite_part' WHERE occ_id>1")
    run, counts = candidate(engine, view)
    assert run["k"] == 0 and counts["stop.budget"] == 1
    assert Counter(row["disposition"] for row in ledger.bindings_by_digest(conn, lineage, digest))["composite_part"] == 5000
