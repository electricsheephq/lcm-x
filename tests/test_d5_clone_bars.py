"""Bars 1-6 and the verdict on hand-built synthetic rows and receipts: each bar both ways."""

import pytest

from bench.instruments.d5_clone import bars as b


def identity(row):
    return row["role"], row["content"]


def whitespace(row, host):  # A synthetic stand-in for the pinned rewrite credits.
    return row["role"] == host["role"] == "user" and row["content"].rstrip() == host["content"].rstrip()


def swap(row, host):
    return row.get("ref") is not None and row["ref"] == host.get("ref")


def store(store_id, session, role, content, **extra):
    return {"store_id": store_id, "session_id": session, "role": role, "content": content, **extra}


def host(host_id, session, role, content, **extra):
    return {"id": host_id, "session_id": session, "role": role, "content": content, **extra}


SNAPSHOT = [store(1, "S", "assistant", "old"), store(2, "S", "user", "older")]
HOSTS = [host(51, "S", "user", "a"), host(52, "C", "assistant", "cron"), host(53, "S", "assistant", "b"),
         host(54, "S", "tool", "big", ref="x1")]
BIND = {"bind": "e1-L-0", "session_id": "S", "t0_max": 9, "h8_max": 13, "dup_store_ids": [10]}


def rc_rows():
    return [store(10, "S", "assistant", "old"),  # a copy at the bind
            store(11, "S", "user", "a  \r\n"),  # whitespace rewrite of host 51
            store(12, "C", "assistant", "cron"),  # cron session, ordered within its own session
            store(13, "S", "assistant", "b"), store(14, "S", "tool", "[stub]", ref="x1"),  # payload swap
            store(15, "S", "assistant", "digest")]  # class 3: no host or snapshot row; GA stores it too


def bar1(rows, ga=(store(99, "S", "assistant", "digest"),), bind2=b.PASS):
    return b.bar1(rows, HOSTS, SNAPSHOT, list(ga), [BIND], {"e1-L-0": bind2}, identity, (whitespace, swap))


def test_bar1_assigns_every_row_once():
    result = bar1(rc_rows())
    assert result["verdict"] == b.PASS
    assert (result["matched"], result["copies_by_bind"], result["class3_rows"], result["lost"]) == (
        4, {"e1-L-0": 1}, 1, 0)


@pytest.mark.parametrize("change, expected", [
    (lambda r: [x for x in r if x["store_id"] != 13], b.FAIL),  # a lost host row
    (lambda r: r[:1] + [r[3], r[1]] + r[2:3] + r[4:], b.FAIL),  # reorder within the session
    (lambda r: r + [store(16, "S", "assistant", "b")], b.FAIL),  # a duplicate of a host row outside bar 2
    (lambda r: r + [store(16, "S", "assistant", "digest")], b.FAIL),  # class 3 stored more often than GA
    (lambda r: r + [store(16, "T", "user", "a")], b.FAIL),  # a wrong-session row GA does not store
    (lambda r: r + [store(16, "S", "user", "older")], b.FAIL),  # matches a snapshot row: never class 3
])
def test_bar1_failures(change, expected):
    assert bar1(change(rc_rows()))["verdict"] == expected


def test_bar1_class3_needs_ga_and_inconclusive_bind():
    assert bar1(rc_rows(), ga=())["verdict"] == b.FAIL
    leftover = rc_rows()[:4] + [store(12.5, "S", "assistant", "b")] + rc_rows()[4:]  # inside the bind range
    assert bar1(leftover, bind2=b.INCONCLUSIVE)["verdict"] == b.INCONCLUSIVE
    assert bar1(leftover, bind2=b.PASS)["verdict"] == b.FAIL
    # One class-3 identity both outside the bind and inside an open bind: GA's one occurrence absorbs the settled
    # row, so only the bind's excess is left, and that is INCONCLUSIVE, never FAIL.
    split = rc_rows()[:4] + [store(12.5, "S", "assistant", "digest")] + rc_rows()[4:]
    assert bar1(split, bind2=b.INCONCLUSIVE)["verdict"] == b.INCONCLUSIVE
    assert bar1(split, bind2=b.PASS)["verdict"] == b.FAIL
    assert bar1(split, ga=(), bind2=b.INCONCLUSIVE)["verdict"] == b.FAIL  # No allowance: the settled row fails.


def test_bar1_reserves_exactly_the_copies_bar2_judged():
    # Probe A: an old row re-stored after the bind's user row is outside the edition's set, so it is a leftover.
    hosts = [host(51, "S", "user", "u1"), host(52, "S", "assistant", "r1")]
    rows = [store(10, "S", "user", "u1"), store(11, "S", "assistant", "r1"), store(12, "S", "assistant", "old")]
    bind = {"bind": "B", "session_id": "S", "t0_max": 9, "h8_max": 20, "dup_store_ids": []}
    assert b.bar1(rows, hosts, SNAPSHOT, [], [bind], {"B": b.PASS}, identity, (whitespace,))["verdict"] == b.FAIL
    # Unknown copies at the bind (no dup_store_ids) or an INCONCLUSIVE bar 2: never FAIL.
    unknown = {**bind, "dup_store_ids": None}
    assert b.bar1(rows, hosts, SNAPSHOT, [], [unknown], {"B": b.PASS}, identity, ())["verdict"] == b.INCONCLUSIVE
    assert b.bar1(rows, hosts, SNAPSHOT, [], [bind], {"B": b.INCONCLUSIVE}, identity, ())["verdict"] == b.INCONCLUSIVE
    # Probe B: a whitespace-credited copy the edition put in its set is reserved, not failed.
    hosts = [host(50, "S", "user", "a"), host(51, "S", "assistant", "x"), host(52, "S", "user", "u2"),
             host(53, "S", "assistant", "r2")]
    rows = [store(1, "S", "user", "a  \r\n"), store(2, "S", "assistant", "x"), store(3, "S", "user", "a"),
            store(4, "S", "user", "u2"), store(5, "S", "assistant", "r2")]
    bind = {"bind": "B2", "session_id": "S", "t0_max": 2, "h8_max": 20, "dup_store_ids": [3]}
    result = b.bar1(rows, hosts, [], rows, [bind], {"B2": b.PASS}, identity, (whitespace,))
    assert (result["verdict"], result["copies_by_bind"]) == (b.PASS, {"B2": 1})
    assert b.dup_store_ids({"bind_identity": {"dup_store_ids": [3]}}) == [3]
    assert b.dup_store_ids({"bind_identity": {}}) is None and b.dup_store_ids(None) is None


def test_bar1_a_bind_without_its_replay_receipt_is_never_fail_or_pass():
    lost = [x for x in rc_rows() if x["store_id"] != 13]
    for rows in (rc_rows(), lost):
        result = b.bar1(rows, HOSTS, SNAPSHOT, [store(99, "S", "assistant", "digest")], [BIND], {"e1-L-0": b.PASS},
                        identity, (whitespace, swap), missing_receipts=["e1-L-0"])
        assert (result["verdict"], result["binds_missing_receipt"]) == (b.INCONCLUSIVE, ["e1-L-0"])


def receipt(validity="PASS", comparison="PASS", **extra):
    checks = {"validity": {"verdict": validity}, **{k: {"verdict": comparison} for k in b.COMPARISONS}}
    return {"checks": checks, "bind_identity": {"dup_store_ids": [10]}, "range_new_rows": 0, **extra}


def test_bar2_pass_and_each_inconclusive_or_fail():
    assert b.bar2_bind(receipt())["verdict"] == b.PASS
    assert b.bar2_bind(receipt())["copies"] == 1
    assert b.bar2_bind(receipt(comparison="FAIL"))["verdict"] == b.FAIL
    for case in (receipt(validity="NOT SHOWN"), receipt(validity="FAIL"), None, receipt(range_new_rows=1),
                 receipt(range_new_rows=None), receipt(comparison="NOT SHOWN"),
                 {"checks": {"validity": {"verdict": "PASS"}}, "range_new_rows": 0}):
        assert b.bar2_bind(case)["verdict"] == b.INCONCLUSIVE
    assert b.bar2_bind(receipt(), error=True)["verdict"] == b.INCONCLUSIVE
    assert b.bar2_bind(receipt(range_new_rows=None), range_new_rows=0)["verdict"] == b.PASS
    for odd in (False, True, 0.0, "0"):  # Only an int 0 is zero.
        assert b.bar2_bind(receipt(range_new_rows=odd))["verdict"] == b.INCONCLUSIVE
    assert b.bar2([b.bar2_bind(receipt())])["verdict"] == b.PASS
    assert b.bar2([])["verdict"] == b.INCONCLUSIVE


def test_bar3_preservation():
    final = [store(1, "S", "assistant", "old", observed_at=5), store(2, "S", "user", "older"), store(3, "S", "x", "y")]
    result = b.bar3(SNAPSHOT, final, identity)
    assert (result["verdict"], result["observed_at_filled"]) == (b.PASS, 1)
    for bad in ([final[1]], [store(1, "S", "assistant", "new"), final[1]], [store(1, "T", "assistant", "old"), final[1]],
                [final[1], final[0]]):
        assert b.bar3(SNAPSHOT, bad, identity)["verdict"] == b.FAIL


def traceback(root, kind="KeyError"):
    return ["Traceback (most recent call last):", f'  File "{root}/engine.py", line 3, in ingest', "    x()",
            '  File "/usr/lib/python3/json.py", line 9, in loads', f"{kind}: synthetic"]


def test_bar4_signatures_and_conflicts():
    rc, ga = traceback("/rc/tree"), traceback("/ga/tree") * 2
    assert b.traceback_signatures(rc, ["/rc/tree"]) == {"KeyError @ engine.py:ingest": 1}
    assert b.bar4(rc, ga, ["/rc/tree"], ["/ga/tree"])["verdict"] == b.PASS
    assert b.bar4(traceback("/other"), [], ["/rc/tree"], [])["verdict"] == b.PASS  # not an LCM-X frame
    for rc_lines in (rc + ["publication_invariant_conflict"], traceback("/rc/tree", "ValueError"), rc * 3):
        assert b.bar4(rc_lines, ga, ["/rc/tree"], ["/ga/tree"])["verdict"] == b.FAIL


def checks(*extra):
    return [{"kind": k, "check": k, "status": "ok"} for k in b.CHECK_KINDS] + list(extra)


def test_bar5_checks_and_warnings():
    warn = {"kind": "doctor", "check": "survival_fit", "status": "warn"}
    assert b.bar5(checks(warn), checks(warn))["verdict"] == b.PASS
    assert b.bar5(checks(warn, warn), checks(warn))["verdict"] == b.FAIL
    assert b.bar5(checks({"kind": "schema", "check": "core", "status": "fail"}), checks())["verdict"] == b.FAIL
    assert b.bar5(checks()[1:], checks())["verdict"] == b.INCONCLUSIVE


def test_bar5_a_failed_ga_check_is_never_pass():
    ga_fail = checks({"kind": "integrity", "check": "fk", "status": "fail"})
    first = b.bar5(checks(), ga_fail)
    assert first["verdict"] == b.INCONCLUSIVE and first["ga_rerun_needed"] and first["ga_failed"] == ["fk"]
    again = b.bar5(checks(), ga_fail, ga_fail)
    assert again["verdict"] == b.INCONCLUSIVE and again["ga_disposition_needed"] and again["ga_rerun_failed"] == ["fk"]
    assert b.bar5(checks(), ga_fail, checks())["verdict"] == b.PASS
    rc_fail = checks({"kind": "schema", "check": "core", "status": "fail"})
    assert b.bar5(rc_fail, ga_fail, ga_fail)["verdict"] == b.FAIL  # An rc failure is FAIL whatever the GA arm shows.


def test_bar5_unknown_status_fails_and_a_rerun_must_show_every_kind():
    assert b.bar5(checks({"kind": "doctor", "check": "d", "status": "error"}), checks())["verdict"] == b.FAIL
    ga_fail = checks({"kind": "integrity", "check": "fk", "status": "fail"})
    for rerun in ([], [{"kind": "quick_check", "check": "q", "status": "ok"}]):
        result = b.bar5(checks(), ga_fail, rerun)
        assert result["verdict"] == b.INCONCLUSIVE and result["ga_rerun_kinds_missing"]
    warn = {"kind": "doctor", "check": "w", "status": "warn"}
    assert b.bar5(checks(warn), ga_fail + [warn, warn], checks())["verdict"] == b.FAIL  # The clean re-run's baseline.
    assert b.check_counts(checks(warn)) == {"by_kind_status": {"doctor:ok": 1, "quick_check:ok": 1,
                                            "integrity:ok": 1, "schema:ok": 1, "doctor:warn": 1}, "warnings": {"w": 1}}


def test_bar5_warnings_are_not_judged_without_a_clean_ga_baseline():
    warn = {"kind": "doctor", "check": "survival_fit", "status": "warn"}
    ga_fail = checks({"kind": "integrity", "check": "fk", "status": "fail"})
    for rerun, complete in ((None, True), (ga_fail, True), (checks(), False), (checks()[1:], True)):
        result = b.bar5(checks(warn), ga_fail, rerun, complete)  # A healthy rc with an ordinary warning.
        assert result["verdict"] == b.INCONCLUSIVE and result["warnings_judged"] is False
        assert result["ga_disposition_needed"] is (rerun is not None)  # Every unclean re-run needs a disposition.



def test_bar5_a_kind_the_ga_arm_does_not_show_is_judged_on_the_rc_alone():
    warn = {"kind": "doctor", "check": "survival_fit", "status": "warn"}
    for ga in ([], checks()[1:]):  # checks()[0] is the doctor row: both miss the doctor kind.
        ok = b.bar5(checks(), ga)
        assert ok["verdict"] == b.PASS and "doctor" in ok["ga_kinds_missing"] and not ok["ga_rerun_needed"]
        warned = b.bar5(checks(warn), ga)  # No baseline to show the warning already existed: never a FAIL.
        assert warned["verdict"] == b.INCONCLUSIVE and warned["ga_kinds_missing_rc_warned"] == ["doctor"]
        assert warned["warnings_judged"] is False and warned["warnings_over_ga"] == []
        assert b.bar5(checks({"kind": "doctor", "check": "d", "status": "fail"}), ga)["verdict"] == b.FAIL
    no_schema = [c for c in checks(warn, warn) if c["kind"] != "schema"]
    assert b.bar5(checks(warn, warn), no_schema)["verdict"] == b.PASS  # Warnings of a kind GA shows still line up.
    assert b.bar5(checks(warn, warn, warn), no_schema)["verdict"] == b.FAIL
    ga_fail = checks({"kind": "integrity", "check": "fk", "status": "fail"})[1:]  # Failed and missing a kind.
    assert b.bar5(checks(warn), ga_fail, checks(warn))["verdict"] == b.PASS  # A clean re-run is the baseline.
    assert b.bar5(checks(), checks())["ga_kinds_missing"] == []

def workload(**change):
    names = ["L", "S1", "S2"]
    return {"conversation_names": names, "large_conversation_rows": 1000, "cron_fired": 1, "tool_heavy_turns": 2,
            "restarts": [{"epoch": e, "cold_binds": names} for e in (1, 2, 3)], "payload_files_created": 1,
            "compactions_published": 1, "whitespace_rewrites": 1, "payload_swaps": 1, "replay_misses": 0,
            "turns_planned": 36, "turns_completed": 36, **change}


def test_bar6_items_per_arm():
    assert b.bar6({"rc": workload(), "ga": workload()})["verdict"] == b.PASS
    short = [{"epoch": e, "cold_binds": ["L", "S1"] if e == 2 else ["L", "S1", "S2"]} for e in (1, 2, 3)]
    for bad in (workload(replay_misses=1), workload(cron_fired=None), workload(restarts=short),
                workload(payload_swaps=0), workload(large_conversation_rows=999), workload(turns_completed=35),
                workload(turns_planned=0, turns_completed=0)):
        assert b.bar6({"rc": workload(), "ga": bad})["verdict"] == b.INCONCLUSIVE
    assert b.bar6({"rc": workload()})["verdict"] == b.INCONCLUSIVE


def test_verdict_rerun_and_instrument_defect():
    passing = {name: {"verdict": b.PASS} for name in b.BARS}
    assert b.verdict(passing)["raw"] == b.PASS
    assert b.verdict({**passing, "bar3": {"verdict": b.FAIL}, "bar6": {"verdict": b.INCONCLUSIVE}})["raw"] == b.FAIL
    assert b.verdict({k: v for k, v in passing.items() if k != "bar5"})["raw"] == b.INCONCLUSIVE
    first = b.verdict({**passing, "bar2": {"verdict": b.INCONCLUSIVE}})
    assert (first["raw"], first["rerun_allowed"], first["instrument_defect"]) == (b.INCONCLUSIVE, True, False)
    second = b.verdict({**passing, "bar2": {"verdict": b.INCONCLUSIVE}}, [b.INCONCLUSIVE])
    assert (second["rerun_allowed"], second["instrument_defect"]) == (False, True)
    assert b.verdict(passing, [b.INCONCLUSIVE])["raw"] == b.PASS
    assert b.may_run([b.FAIL], None) and b.may_run([b.INCONCLUSIVE] * 2, None)
    assert b.may_run([], None) is None and b.may_run([b.INCONCLUSIVE] * 2, "fix") is None


def test_bar5_a_session_that_started_flagged_is_not_judged_and_never_passes():
    """#2471: a driven session whose store already had a flagged summary node keeps summary_quality warning, so a
    further flag in that session would not show. Its warnings are not judged: INCONCLUSIVE, never PASS or a FAIL from
    that check alone; the release manager judges the node counts."""
    flag = {"kind": "doctor", "check": "summary_quality@abc", "status": "warn"}
    other = {"kind": "doctor", "check": "survival_fit", "status": "warn"}
    assert b.bar5(checks(flag, flag), checks(flag))["verdict"] == b.FAIL  # Not inherited: judged as before.
    result = b.bar5(checks(flag, flag), checks(flag), inherited=["summary_quality@abc"])
    assert result["verdict"] == b.INCONCLUSIVE and result["inherited_flags_not_judged"] == ["summary_quality@abc"]
    assert result["warnings_over_ga"] == []
    assert b.bar5(checks(), checks(), inherited=["summary_quality@abc"])["verdict"] == b.INCONCLUSIVE
    assert b.bar5(checks(other, other), checks(other), inherited=["summary_quality@abc"])["verdict"] == b.FAIL
    assert b.bar5(checks(), checks())["inherited_flags_not_judged"] == []
