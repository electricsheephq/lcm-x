"""B1 host-occurrence binding, compared with the previous store-adjacency predicate."""
import json
import sqlite3

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
    monkeypatch.setattr(bars, "continue_occurrences", lambda rows, _state, _group, pat, **_kw:
                        bars.continue_positions([(r[2], r[3]) for r in rows if r[2] in ("user", "assistant")], pat))
    previous = fixture.make(tmp_path / "old", **kw)
    old_verdict = "FAIL" if "B1" in previous["failed_bars"] else "PASS"
    new_verdict = "FAIL" if "B1" in current["failed_bars"] else "PASS"
    print(f"{case}: old={old_verdict} new={new_verdict}")
    assert (old_verdict, new_verdict) == (old, new)
    if case == "duplicate_reply":
        assert current["failed_bars"]["B1"]["reply T02"] == {"expected": 1, "stored": 2}


@pytest.mark.parametrize("change", ["reply_timestamp", "continue_timestamp", "reply_content", "tool_calls",
                                    "tool_call_id", "lineage", "missing_timestamp"])
def test_b1_requires_full_occurrence_identity(tmp_path, change):
    rows = [(1, "S0", "user", "continue", None, None, 10.0),
            (2, "S0", "assistant", "reply to T01: exact reply", None, None, 20.0)]
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
                     "missing_timestamp": (6, None)}[change]
    changed[column] = value
    rows[0 if change == "continue_timestamp" else 1] = tuple(changed)
    assert bars.continue_occurrences(rows, state, lambda sid: sid, r"reply to (T\d+)\b") == {None: 1}


def continue_fixture():
    rows, events = fixture.clean_rows(), fixture.clean_events()
    rows[2] = ("user", "continue")
    events[3]["content"] = events[3]["persist"] = events[5]["held"] = "continue"
    host = [("S0", r, c, None, None, i * 10.0) for i, (r, c) in enumerate(rows, 1)]
    return rows, events, host


@pytest.mark.parametrize("rotation", [False, True])
def test_b1_duplicate_host_copy_ignores_empty_successor(tmp_path, monkeypatch, rotation):
    rows, events, host = continue_fixture()
    if rotation:
        host = host[:3] + [("child", *r[1:]) for r in host[2:]]
    else:
        host = host[:3] + [("S0", "user", "summary", None, None, 35.0)] + host[2:]
    kw = dict(rows=rows, events=events, host_occurrences=host, observed=[10., 20., 30., 40., 50., 60.],
              parents={"S0": None, "child": "S0"} if rotation else None, bars=["B1"])
    current = fixture.make(tmp_path / "new", **kw)
    monkeypatch.setattr(bars, "continue_occurrences", lambda rows, _state, _group, pat, **_kw:
                        bars.continue_positions([(r[2], r[3]) for r in rows if r[2] in ("user", "assistant")], pat))
    previous = fixture.make(tmp_path / "old", **kw)
    assert previous["verdict"] == current["verdict"] == "PASS"
    assert current["numbers"]["B1"]["continue_fallbacks"] == []


def test_f3_with_host_evidence_uses_occurrence_not_store_order(tmp_path):
    rows, events, host = continue_fixture()
    order = [0, 1, 3, 2, 4, 5]
    out = fixture.make(tmp_path, rows=[rows[i] for i in order], events=events,
                       host_occurrences=host, observed=[host[i][-1] for i in order], bars=["B1"])
    assert out["verdict"] == "PASS"
    assert out["numbers"]["B1"]["continue_fallbacks"] == []


@pytest.mark.parametrize("cause,reason", [
    ("missing_host", "state.db missing"), ("unreadable_host", "state.db unreadable"),
    ("no_observed_at", "store has no observed_at column"),
    ("store_timestamp", "stored continue has no timestamp"),
    ("host_timestamp", "host continue copy has no timestamp"),
])
@pytest.mark.parametrize("ordered", [True, False])
def test_b1_unavailable_evidence_falls_back_neutrally(tmp_path, cause, reason, ordered):
    rows, events, host = continue_fixture()
    order = list(range(6)) if ordered else [0, 1, 3, 2, 4, 5]
    if cause == "host_timestamp":
        host.insert(2, (*host[2][:-1], None))  # even a second, timestamped copy cannot fill this gap
    stamps = [float(i + 1) * 10 for i in order]
    if cause == "store_timestamp":
        stamps[order.index(2)] = None
    out = fixture.make(tmp_path, rows=[rows[i] for i in order], events=events,
                       host_occurrences=None if cause == "missing_host" else host,
                       observed=None if cause == "no_observed_at" else stamps, bars=["B1"])
    if cause == "unreadable_host":
        state = tmp_path / "cell/db/state.db"
        with sqlite3.connect(state) as con:
            con.execute("drop table sessions")
            con.execute("drop table messages")
        out = bars.score({"bars": ["B1"]}, tmp_path / "cell")
    seq = [(rows[i][0], rows[i][1]) for i in order]
    old_passes = bars.continue_positions(rows, r"reply to (T\d+)\b") == bars.continue_positions(seq, r"reply to (T\d+)\b")
    assert (out["verdict"] == "PASS") == old_passes == ordered
    assert out["numbers"]["B1"]["continue_fallbacks"] == [{"lineage": "chat", "reason": reason}]


def test_b1_fallback_is_per_lineage(tmp_path):
    rows, events, host = continue_fixture()
    order = [0, 1, 3, 2, 4, 5]
    rows = [rows[i] for i in order] + [("user", "continue"), ("assistant", "reply to K01: cron")]
    events += [{"event": "user_sent", "tag": "K01", "session_prefix": "K", "content": "continue"},
               {"event": "emit", "tag": "K01", "text": "reply to K01: cron"},
               {"event": "turn_end", "tag": "K01", "held": "continue", "session": "cron_job_01"}]
    host += [("cron_job_01", "user", "continue", None, None, None)]
    out = fixture.make(tmp_path, rows=rows, events=events, host_occurrences=host,
                       sids=["S0"] * 6 + ["cron_job_01"] * 2,
                       observed=[float(i + 1) * 10 for i in order] + [None, None], bars=["B1"])
    assert out["verdict"] == "PASS"  # chat still requires occurrence binding; cron requires store order
    assert out["numbers"]["B1"]["continue_fallbacks"] == [
        {"lineage": "cron_job_01", "reason": "stored continue has no timestamp"}]


@pytest.mark.parametrize("ambiguous", [False, True])
def test_b1_nonempty_binding_stays_unambiguous_and_single_use(tmp_path, ambiguous):
    rows = [(1, "S0", "user", "continue", None, None, 10.),
            (2, "S0", "assistant", "reply to T01: reply", None, None, 20.)]
    host = rows + [(3, "S0", "user", "continue", None, None, 10.),
                   (4, "S0", "assistant", "reply to T02: other reply", None, None, 30.)] if ambiguous else rows
    state = tmp_path / "state.db"
    with sqlite3.connect(state) as con:
        con.execute("create table messages (id integer, session_id text, role text, content text, tool_calls text,"
                    " tool_call_id text, timestamp real)")
        con.executemany("insert into messages values (?,?,?,?,?,?,?)", host)
    out = bars.continue_occurrences(rows if ambiguous else rows + [rows[0]], state, lambda sid: sid, r"reply to (T\d+)\b")
    assert out == ({None: 1} if ambiguous else {"T01": 1, None: 1})
