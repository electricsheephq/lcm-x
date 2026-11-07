"""#1179 R4: absorbed replacement identity and UID-only unchanged suffixes."""
from copy import deepcopy

import pytest

pytest.importorskip("hypothesis")
from hermes_lcm import host_uid_emit
from tests.invariants.test_exactly_once_machine import ExactlyOnceMachine, assert_occurrences, identity


@pytest.fixture
def machine():
    machine = ExactlyOnceMachine()
    yield machine
    machine.teardown()


@pytest.mark.parametrize("anchor", [False, True])
@pytest.mark.parametrize("fresh", [False, True])
def test_r4_1_replaced_user_behind_unchanged_scaffold(machine, anchor, fresh):
    machine.patch.setenv("LCM_IDENTITY_ANCHOR", str(int(anchor)))
    machine.patch.setattr(host_uid_emit, "_host_uid_capability", True)
    for text in ("alpha", "same text"):
        machine.new_turn(text, False, 0, False, "lookup")
    machine.compress()
    assert machine.engine._last_compression_status == "compacted"
    output = deepcopy(machine.host)
    replacement = deepcopy(output[-2])
    assert replacement["role"] == "user"
    if fresh:
        replacement["message_uid"] = "replacement-user"
    output[-2] = replacement
    host = []
    for row in output:
        if host and host[-1]["role"] == row["role"] == "user":
            host_uid_emit.record_absorbed_message(host[-1], row)
            host[-1]["content"] += "\n\n" + row["content"]
        else:
            host.append(deepcopy(row))
    merged = host[-2]
    assert len(host) < len(output)
    assert merged.get("message_uid") == output[-3].get("message_uid")
    assert merged.get("timestamp") == output[-3].get("timestamp")
    assert merged["_absorbed_message_uids"] == [replacement["message_uid"]]
    # A fresh merged occurrence follows the existing lossless ingest path:
    # persist the complete host row, rather than stripping its generated prefix.
    expected = machine.occurrences + ([merged] if fresh else [])
    machine.engine.ingest(host)
    assert_occurrences(expected, machine.rows(), engine=machine.engine)
    machine.engine.ingest(deepcopy(host))
    assert_occurrences(expected, machine.rows(), engine=machine.engine)


@pytest.mark.parametrize("anchor", [False, True])
@pytest.mark.parametrize("compacted", [False, True])
def test_r4_2_uid_only_replacement_keeps_unchanged_suffix(machine, anchor, compacted):
    machine.patch.setenv("LCM_IDENTITY_ANCHOR", str(int(anchor)))
    machine.patch.setattr(machine, "stamp", lambda row: dict(
        row, message_uid=f"uid-{len(machine.occurrences)}-{row['role']}"))
    for text in ("alpha", "same text"):
        machine.new_turn(text, False, 0, False, "lookup")
    if compacted:
        machine.compress()
        assert machine.engine._last_compression_status == "compacted"
    host = deepcopy(machine.host)
    assert all("timestamp" not in row for row in host)
    replacement = dict(host[-2], message_uid="replacement-user")
    host[-2] = replacement
    # Isolate suffix duplication from main's known #1179 replacement loss.
    # The prefix-retention regressions in test_identity_seeds cover that loss.
    expected = [row for row in machine.occurrences if row["role"] == "assistant"]
    for batch in (host, deepcopy(host)):
        machine.engine.ingest(batch)
        assert_occurrences(expected, [row for row in machine.rows() if row["role"] == "assistant"], engine=machine.engine)


def _merge_absorbed(output):
    host = []
    for row in output:
        if host and host[-1]["role"] == row["role"] == "user":
            host_uid_emit.record_absorbed_message(host[-1], row)
            host[-1]["content"] += "\n\n" + row["content"]
        else:
            host.append(deepcopy(row))
    return host


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1428: after the unchanged merged output consumes the compaction proof, the "
                                                              "cached-prefix check compares the cached copy, which lacks the absorbed user's "
                                                              "uid, so a uid-only replacement is not stored")
@pytest.mark.parametrize("anchor", [False, True])
def test_r4_1_uid_only_replacement_after_the_proof_is_consumed(machine, anchor):
    """#1428 (a v0.30.0 named loss limit): compress, ingest the unchanged merged output, then replace only the
    absorbed user row's host uid and ingest again. The new occurrence must be stored."""
    machine.patch.setenv("LCM_IDENTITY_ANCHOR", str(int(anchor)))
    machine.patch.setattr(host_uid_emit, "_host_uid_capability", True)
    for text in ("alpha", "same text"):
        machine.new_turn(text, False, 0, False, "lookup")
    machine.compress()
    if machine.engine._last_compression_status != "compacted":
        pytest.fail(f"setup: the compress was {machine.engine._last_compression_status!r}, not compacted")
    output = deepcopy(machine.host)
    if output[-2]["role"] != "user":
        pytest.fail(f"setup: the absorbed row is a {output[-2]['role']} row")
    machine.engine.ingest(_merge_absorbed(output))  # the unchanged merged output consumes the proof
    if list(map(identity, machine.rows())) != list(map(identity, machine.occurrences)):
        pytest.fail("setup: the unchanged merged output changed the stored rows")
    output[-2] = dict(output[-2], message_uid="replacement-user")
    host = _merge_absorbed(output)
    if host[-2]["_absorbed_message_uids"] != ["replacement-user"]:
        pytest.fail(f"setup: the merged row absorbed {host[-2]['_absorbed_message_uids']}")
    machine.engine.ingest(host)
    observed = list(map(identity, machine.rows()))
    _expect_stored_shape(observed, list(map(identity, [*machine.occurrences, host[-2]])),
                         list(map(identity, machine.occurrences)),
                         "#1428 known shape: the uid-only replacement was not stored")


def _expect_stored_shape(observed, fixed, known, message, error_type=AssertionError):
    """Accept only the complete fixed list or the complete named-limit list, in store order."""
    if observed == fixed:
        return
    if observed == known:
        raise error_type(message)
    differences = []
    for label, expected in (("fixed", fixed), ("known", known)):
        index = next((i for i, (actual, wanted) in enumerate(zip(observed, expected)) if actual != wanted),
                     min(len(observed), len(expected)))
        differences.append(f"{label}: first difference={index}, lengths={len(observed)}/{len(expected)}")
    pytest.fail("unexpected stored shape; " + "; ".join(differences))


@pytest.mark.parametrize("case", ["known", "fixed", "doubled", "swapped"])
def test_stored_shape_comparison(case):
    fixed = [("user", "first"), ("assistant", "second")]
    known = [fixed[0]]
    observed = {"known": known, "fixed": fixed, "doubled": [fixed[0], *fixed], "swapped": fixed[::-1]}[case]
    if case == "known":
        with pytest.raises(AssertionError, match="synthetic known shape"):
            _expect_stored_shape(observed, fixed, known, "synthetic known shape", AssertionError)
    elif case == "fixed":
        _expect_stored_shape(observed, fixed, known, "synthetic known shape", AssertionError)
    else:
        with pytest.raises(pytest.fail.Exception, match="first difference=.*lengths="):
            _expect_stored_shape(observed, fixed, known, "synthetic known shape", AssertionError)
