"""Receipt reads stay bounded when a provider reuses one call id and inline arguments."""

import base64
import copy
import json
from collections import Counter

import pytest

from tests.invariants.test_issue_2742_warm_tail_refinal import _open
from tests.test_host_uid_shadow import _state_db


def _turns(k):
    encoded = base64.b64encode(bytes((i * 37) % 256 for i in range(1200))).decode("ascii")
    arguments = json.dumps({"data:image\\/png;base64," + encoded: "v", "b": "same"})
    rows = []
    for n in range(k):
        stamp = float(n * 10 + 1)
        rows += [
            dict(role="user", content=f"q-{n}", timestamp=stamp, message_uid=f"u{n}0"),
            dict(role="assistant", content="", timestamp=stamp + 1, message_uid=f"u{n}1", tool_calls=[
                dict(id="c0", type="function", function=dict(name="write_file", arguments=arguments))]),
            dict(role="tool", content=f"r-{n}", tool_call_id="c0", timestamp=stamp + 2, message_uid=f"u{n}2"),
            dict(role="assistant", content=f"a-{n}", timestamp=stamp + 3, message_uid=f"u{n}3"),
        ]
    return rows


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_receipt_queries_once_per_digest_and_record(tmp_path, monkeypatch, reverse):
    home = tmp_path / "home"
    home.mkdir()
    _state_db(home, [("parent", None, None)])
    engine = _open(tmp_path, monkeypatch, "shadow")
    conn = engine._store._conn
    try:
        view = _turns(40)
        for end in range(100, len(view) + 100, 100):
            engine.ingest(copy.deepcopy(view[:end]))
        stored = [row for row in engine._store.get_session_messages("parent") if row.get("tool_calls")]
        assert len(stored) == 40
        assert all("Externalized LCM ingest payload:" in str(row["tool_calls"]) for row in stored)
        calls = view[1::4]
        if reverse:
            calls = calls[::-1]  # UID vetoes must be reconsidered for later host rows.
            stored = stored[::-1]
        entries, _stamps = engine._ledger_view_entries(engine._store.ingest_ledger_salt(), calls)
        assert len({entry.identity_digest for entry in entries}) == 1

        for with_uid in (True, False):  # A fresh attempt; UID-less rows exercise repeated stamp vetoes.
            host_calls = calls if with_uid else [{k: v for k, v in row.items() if k != "message_uid"} for row in calls]
            queries, consumed, matched = [], set(), {}
            plan = {"cursor": 0, "replayed": set()}
            changes = conn.total_changes
            conn.set_trace_callback(queries.append)
            try:
                engine._identity_anchor_receipts(host_calls, 0, consumed, matched, plan)
            finally:
                conn.set_trace_callback(None)
            assert conn.total_changes == changes
            assert [matched[i][0]["store_id"] for i in range(40)] == [row["store_id"] for row in stored]
            assert consumed == {row["store_id"] for row in stored}
            assert plan["replayed"] == set(range(40))
            bindings = [sql for sql in queries if sql.startswith("SELECT b.*,m.* FROM occurrence_bindings")]
            assert len(bindings) == 1, f"binding query ran {len(bindings)} times for one digest"
            for prefix, expected in (("SELECT uid FROM host_uid_bindings WHERE store_id=", 40 if with_uid else 0),
                                     ("SELECT stamp FROM stamp_observations WHERE occ_id=", 40)):
                counts = Counter(sql for sql in queries if sql.startswith(prefix))
                assert len(counts) == expected, (prefix, counts)
                assert all(count == 1 for count in counts.values()), (prefix, counts)
    finally:
        conn.set_trace_callback(None)
        engine.shutdown()
