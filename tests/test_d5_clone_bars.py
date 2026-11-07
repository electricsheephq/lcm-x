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
BIND = {"bind": "e1-L-0", "session_id": "S", "t0_max": 9, "h8_max": 13, "t0_keys": b.keys(SNAPSHOT, identity)}


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


def workload(**change):
    names = ["L", "S1", "S2"]
    return {"conversation_names": names, "large_conversation_rows": 1000, "cron_fired": 1, "tool_heavy_turns": 2,
            "restarts": [{"epoch": e, "cold_binds": names} for e in (1, 2, 3)], "payload_files_created": 1,
            "compactions_published": 1, "whitespace_rewrites": 1, "payload_swaps": 1, "replay_misses": 0, **change}


def test_bar6_items_per_arm():
    assert b.bar6({"rc": workload(), "ga": workload()})["verdict"] == b.PASS
    short = [{"epoch": e, "cold_binds": ["L", "S1"] if e == 2 else ["L", "S1", "S2"]} for e in (1, 2, 3)]
    for bad in (workload(replay_misses=1), workload(cron_fired=None), workload(restarts=short),
                workload(payload_swaps=0), workload(large_conversation_rows=999)):
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
