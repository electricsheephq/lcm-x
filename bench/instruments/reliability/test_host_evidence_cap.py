"""Unreadable host state must not turn a host-dependent proof into PASS."""
import sqlite3

import pytest

from bench.instruments.reliability.scorers import bars
from bench.instruments.reliability.test_b1_occurrence import continue_fixture
from bench.instruments.reliability import test_delegated_child as child_fixture
from tests import test_reliability_scorers as fixture


def break_host(tmp_path):
    with sqlite3.connect(tmp_path / "cell/db/state.db") as con:
        con.execute("drop table sessions")
        con.execute("drop table messages")


def test_b1_restamped_reply_with_unreadable_host_is_inconclusive(tmp_path):
    rows, events, host = continue_fixture()
    out = fixture.make(tmp_path, rows=rows, events=events, host_occurrences=host,
                       observed=[10., 20., 30., 41., 50., 60.], bars=["B1"])
    assert out["verdict"] == "FAIL" and "continue" in out["failed_bars"]["B1"]
    break_host(tmp_path)
    out = bars.score({"bars": ["B1"]}, tmp_path / "cell")
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["failed_bars"] == {} and set(out["inconclusive_bars"]) == {"B1"}


def test_b4_lost_descendant_link_with_unreadable_host_is_inconclusive(tmp_path):
    rows = fixture.clean_rows() + [("user", "uncovered descendant row")]
    out = fixture.make(tmp_path, rows=rows, events=fixture.clean_events(), nodes=[[1, 2, 3, 4]],
                       sids=["S0"] * 6 + ["S1"], parents={"S0": None, "S1": "S0"}, host=[],
                       phase={"final_check": fixture.drained_final()}, bars=["B4"])
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["numbers"]["B4"]["coverage"]["uncovered_ids"] == [7]
    break_host(tmp_path)
    out = bars.score({"bars": ["B4"]}, tmp_path / "cell")
    assert out["numbers"]["B4"]["coverage"]["uncovered_ids"] == []
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["failed_bars"] == {} and set(out["inconclusive_bars"]) == {"B4"}


@pytest.mark.parametrize("bar", ["B1", "B2", "B4", "B5"])
@pytest.mark.parametrize("failure", [False, True])
def test_host_dependent_bars_cap_only_pass(tmp_path, bar, failure):
    kw = {"rows": fixture.clean_rows(), "events": fixture.clean_events(), "parents": {"S0": None},
          "host": [], "bars": [bar]}
    if failure:
        kw.update(rows=fixture.clean_rows()[:-1], passes=0, phase={"final_check": {"published": False}})
    before = fixture.make(tmp_path, **kw)
    assert before["verdict"] == ("FAIL" if failure else "PASS")
    break_host(tmp_path)
    out = bars.score({"bars": [bar], "min_compactions": 2}, tmp_path / "cell")
    assert out["verdict"] == ("FAIL" if failure else "INCONCLUSIVE")
    assert out["failed_bars"] == before["failed_bars"]
    if failure:
        assert bar not in out["inconclusive_bars"]
        assert "verdict_capped" not in out["numbers"]["diagnostic"]
    else:
        assert out["numbers"]["diagnostic"]["verdict_capped"] == {
            "bars": [bar], "reason": "host evidence unreadable: state.db unreadable (OperationalError)"}


@pytest.mark.parametrize("error", ["schema", "corrupt"])
def test_cap_records_only_applicable_passing_bars(tmp_path, error):
    fixture.make(tmp_path, rows=fixture.clean_rows(), events=fixture.clean_events(),
                 parents={"S0": None}, host=[])
    if error == "schema":
        break_host(tmp_path)
    else:
        (tmp_path / "cell/db/state.db").write_bytes(b"not a sqlite database")
    out = bars.score({"bars": ["B1", "B2", "B3", "B4", "B5", "B8"], "min_compactions": 2}, tmp_path / "cell")
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["numbers"]["diagnostic"]["verdict_capped"]["bars"] == ["B1", "B2", "B4", "B5"]
    assert set(out["inconclusive_bars"]) == {"B1", "B2", "B4", "B5"}
    independent = bars.score({"bars": ["B3", "B8"]}, tmp_path / "cell")
    assert independent["verdict"] == "PASS"


@pytest.mark.parametrize("missing", [False, True])
def test_optional_parity_unavailable_is_not_a_requirement(tmp_path, missing):
    kw = {} if missing else {"parents": {"S0": None}, "host": []}
    out = fixture.make(tmp_path, rows=fixture.clean_rows(), events=fixture.clean_events(), **kw)
    assert out["verdict"] == "PASS"
    assert "unavailable" in out["numbers"]["B1"]["host_parity_licensed"]
    assert "verdict_capped" not in out["numbers"]["diagnostic"]


@pytest.mark.parametrize("failed", [False, True])
def test_delegated_host_error_caps_b10_but_keeps_child_failure(tmp_path, failed):
    events = child_fixture.CHILD_EVENTS[:2] + [dict(child_fixture.CHILD_EVENTS[2], failed=failed)]
    child_fixture.make(tmp_path, events=events)
    with sqlite3.connect(tmp_path / "cell/db/state.db") as con:
        con.execute("alter table sessions rename to unreadable_sessions")
    out = bars.score({"bars": ["B10"], "delegate_plan": [{"turns": [2]}]}, tmp_path / "cell")
    assert out["verdict"] == ("FAIL" if failed else "INCONCLUSIVE")
    if failed:
        assert out["failed_bars"]["B10"]["children_failed"] == ["D02"]
        assert "B10" not in out["inconclusive_bars"]
    else:
        assert set(out["inconclusive_bars"]) == {"B10"}


def test_missing_required_host_state_cannot_pass_b10(tmp_path):
    out = fixture.make(tmp_path, rows=fixture.clean_rows(), events=fixture.clean_events(),
                       bars=["B10"], delegate_plan=[{"turns": []}])
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["inconclusive_bars"] == {"B10": "B10 requires host state.db: missing"}


def test_b9_only_early_unsupported_is_unchanged(tmp_path):
    before = fixture.make(tmp_path, rows=fixture.clean_rows(), events=fixture.clean_events(),
                          parents={"S0": None}, host=[], bars=["B9"])
    break_host(tmp_path)
    out = bars.score({"bars": ["B9"], "min_compactions": 2}, tmp_path / "cell")
    assert out == before
    assert out["verdict"] == "UNSUPPORTED" and "diagnostic" not in out["numbers"]


@pytest.mark.parametrize("ordered", [True, False], ids=["would-pass", "fails"])
def test_unreadable_messages_table_with_readable_sessions_is_never_pass(tmp_path, ordered):
    """#2382: the sessions read succeeds, so lineage sets no error; the messages read fails. Any failed host read
    must set host_evidence_error: a passing B1 is capped at INCONCLUSIVE, and a failure stays a failure."""
    rows, events, host = continue_fixture()
    order = list(range(6)) if ordered else [0, 1, 3, 2, 4, 5]
    fixture.make(tmp_path, rows=[rows[i] for i in order], events=events, host_occurrences=host,
                 observed=[float(i + 1) * 10 for i in order], bars=["B1"])
    with sqlite3.connect(tmp_path / "cell/db/state.db") as con:
        con.execute("drop table messages")
    out = bars.score({"bars": ["B1"]}, tmp_path / "cell")
    assert out["numbers"]["diagnostic"]["host_evidence_error"]
    assert out["numbers"]["B1"]["continue_fallbacks"] == [{"lineage": "chat", "reason": "state.db unreadable"}]
    if ordered:
        assert out["verdict"] == "INCONCLUSIVE" and set(out["inconclusive_bars"]) == {"B1"}
    else:
        assert out["verdict"] == "FAIL" and "B1" in out["failed_bars"]


@pytest.mark.parametrize("link", ["missing", "S0"], ids=["missing-parent-link", "linked"])
def test_unreadable_messages_keeps_b10_session_checks(tmp_path, monkeypatch, link):
    """Review of #2382: readable sessions with a failed messages read (here the host parity read) must keep B10's
    session-backed failures (a missing parent link stays FAIL); an otherwise-passing B10 is capped at INCONCLUSIVE."""
    child_fixture.make(tmp_path, link=None if link == "missing" else link)
    monkeypatch.setattr(bars.host_parity, "load",
                        lambda *_a, **_k: (None, "host state.db unreadable: OperationalError('no such table')"))
    out = bars.score({"bars": ["B10"], "delegate_plan": [{"turns": [2]}]}, tmp_path / "cell")
    assert out["numbers"]["diagnostic"]["host_evidence_error"]
    if link == "missing":
        assert out["verdict"] == "FAIL"
        assert out["failed_bars"]["B10"]["children_without_parent_link"] == ["C1"]
    else:
        assert out["verdict"] == "INCONCLUSIVE" and set(out["inconclusive_bars"]) == {"B10"}
