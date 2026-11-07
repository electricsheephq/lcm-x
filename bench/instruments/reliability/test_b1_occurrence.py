"""B1 host-occurrence binding, compared with the previous store-adjacency predicate."""
import json

import pytest

import externalize
from bench.instruments.reliability.scorers import bars
from tests import test_reliability_scorers as fixture


@pytest.mark.parametrize("externalized_role", [None, "user", "assistant"])
@pytest.mark.parametrize("case,order,old,new", [
    ("out_of_order_head", [0, 1, 3, 4, 5, 2], "FAIL", "PASS"),
    ("missing_reply", [0, 1, 2, 4, 5], "FAIL", "FAIL"),
    ("duplicate_reply", [0, 1, 2, 3, 3, 4, 5], "FAIL", "FAIL"),
    ("in_order", [0, 1, 2, 3, 4, 5], "PASS", "PASS"),
])
def test_b1_old_vs_new(tmp_path, monkeypatch, case, order, old, new, externalized_role):
    rows, events = fixture.clean_rows(), fixture.clean_events()
    rows[2] = ("user", "continue")
    events[3]["content"] = events[3]["persist"] = events[5]["held"] = "continue"
    host = [("S0", r, c, None, None, i * 10.0) for i, (r, c) in enumerate(rows, 1)]
    if externalized_role:
        index = 2 if externalized_role == "user" else 3
        payload = {"content": rows[index][1], "kind": "raw_payload", "role": externalized_role}
        for version in ("old", "new"):
            storage = tmp_path / version / "cell" / "hermes-home" / externalize.DEFAULT_LARGE_OUTPUT_DIRNAME
            storage.mkdir(parents=True)
            (storage / "reply.json").write_text(json.dumps(payload))
        rows[index] = (externalized_role, externalize._build_externalized_placeholder({**payload, "ref": "reply.json"}))
    kw = dict(rows=[rows[i] for i in order], events=events, host_occurrences=host,
              observed=[host[i][-1] for i in order], bars=["B1"])
    current = fixture.make(tmp_path / "new", **kw)
    # The exact previous have_c predicate; all other B1 deficit/surplus checks are unchanged.
    monkeypatch.setattr(bars, "continue_occurrences", lambda rows, _state, _group, pat:
                        bars.continue_positions([(r[2], r[3]) for r in rows if r[2] in ("user", "assistant")], pat))
    previous = fixture.make(tmp_path / "old", **kw)
    old_verdict = "FAIL" if "B1" in previous["failed_bars"] else "PASS"
    new_verdict = "FAIL" if "B1" in current["failed_bars"] else "PASS"
    print(f"{case}: old={old_verdict} new={new_verdict}")
    assert (old_verdict, new_verdict) == (old, new)
    if case == "duplicate_reply":
        assert current["failed_bars"]["B1"]["reply T02"] == {"expected": 1, "stored": 2}


@pytest.mark.parametrize("change", ["reply_timestamp", "continue_timestamp", "reply_content", "tool_calls",
                                    "tool_call_id", "lineage", "missing_host", "missing_timestamp"])
def test_b1_requires_full_occurrence_identity(tmp_path, change):
    rows = [(1, "S0", "user", "continue", None, None, 10.0),
            (2, "S0", "assistant", "reply to T01: exact reply", None, None, 20.0)]
    import sqlite3
    state = tmp_path / "state.db"
    con = sqlite3.connect(state)
    con.execute("create table messages (id integer, session_id text, role text, content text, tool_calls text,"
                " tool_call_id text, timestamp real)")
    con.executemany("insert into messages values (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    changed = list(rows[0 if change == "continue_timestamp" else 1])
    column, value = {"reply_timestamp": (6, 21.0), "continue_timestamp": (6, 11.0),
                     "reply_content": (3, "reply to T01: altered reply"), "tool_calls": (4, "[]"),
                     "tool_call_id": (5, "other-call"), "lineage": (1, "other"),
                     "missing_host": (6, 20.0), "missing_timestamp": (6, None)}[change]
    changed[column] = value
    rows[0 if change == "continue_timestamp" else 1] = tuple(changed)
    if change == "missing_host":
        state = tmp_path / "absent.db"
    assert bars.continue_occurrences(rows, state, lambda sid: sid, r"reply to (T\d+)\b") == {None: 1}
