"""#2672: recover changed proof occurrences without losing suffix order on resume."""
from __future__ import annotations

import copy
from collections import Counter

import pytest

from tests import test_issue_519_replaced_tail as rotation
from tests.test_issue_2624_host_row_recovery import (
    C, P, case as _case_fixture, ident, pair, rows,
)

case = _case_fixture


def _stamp_turns(monkeypatch):
    original = rotation._turn

    def stamped(index):
        return tuple({**message, "timestamp": 1800000000 + 2 * index + offset,
                      "message_uid": f"prefix-{index}-{offset}"}
                     for offset, message in enumerate(original(index)))

    monkeypatch.setattr(rotation, "_turn", stamped)
    monkeypatch.setattr(rotation, "T16", stamped(16))


@pytest.fixture
def stamped_case(monkeypatch, request):
    # Install occurrence metadata before the shared fixture commits its proof.
    _stamp_turns(monkeypatch)
    return request.getfixturevalue("case")


def _history(case, position, kind, shape):
    prefix = copy.deepcopy(case.prefix)
    users = [i for i, row in enumerate(prefix) if row["role"] == "user" and "timestamp" in row]
    assistants = [i for i, row in enumerate(prefix) if row["role"] == "assistant"]
    positions = {"last-user": [users[-1]], "middle-user": [users[len(users) // 2]],
                 "first-assistant": [assistants[0]], "two": [assistants[0], users[-1]], "none": []}
    changed = positions[position]
    for index in changed:
        if kind == "timestamp":
            prefix[index]["timestamp"] += 10000
        else:
            prefix[index]["message_uid"] += "-resent"
    suffix = pair(uids=True)[1:] if shape == "reply" else pair(uids=True)
    return prefix + suffix, changed, suffix


def _resume(case, history, check):
    child = case.start(C, conversation=True)
    child.ingest(history)
    check("first ingest")
    more = [{**row, "content": "2672 next " + row["content"],
             "timestamp": row["timestamp"] + 100, "message_uid": row["message_uid"] + "-next"}
            for row in pair(uids=True)]
    for end in (1, 2):
        child.ingest(history + more[:end])
        check(f"next turn row {end}")
    return ident(rows(case))


@pytest.mark.parametrize("arm", ["never-resumed", "resumed"])
@pytest.mark.parametrize("position", ["last-user", "middle-user", "first-assistant", "two", "none"])
@pytest.mark.parametrize("kind", ["timestamp", "message_uid"])
@pytest.mark.parametrize("shape", ["pair", "reply"])
def test_changed_proof_prefix(stamped_case, arm, position, kind, shape, record_property):
    case = stamped_case
    label = f"{arm}/{position}/{kind}/{shape}"
    history, changed, suffix = _history(case, position, kind, shape)
    case.host.lists[C] = history
    parent = ident(rows(case, P))

    def no_duplicates(stage, target=case):
        actual = ident(rows(target))
        assert len(actual) == len(set(actual)), f"{label}: duplicate at {stage}: {actual}"
        assert all(row[1] == P for row in actual), f"{label}: wrong conversation at {stage}"

    case.run()
    recovered = ident(rows(case))
    expected_suffix = [(C, P, row["role"], row["content"]) for row in suffix]
    assert Counter(expected_suffix) <= Counter(recovered), f"{label}: lost today's recovery suffix"
    allowed = [(C, P, history[i]["role"], history[i]["content"]) for i in changed] + expected_suffix
    assert Counter(recovered) <= Counter(allowed), f"{label}: recovered an unchanged carried row"
    no_duplicates("recovery")
    case.run()  # Retry the detached worker against its own durable writes.
    assert ident(rows(case)) == recovered, f"{label}: worker retry stored rows again"
    no_duplicates("worker retry")
    assert ident(rows(case, P)) == parent, f"{label}: recovery changed parent rows"

    final, control = recovered, None
    if arm == "resumed":
        final = _resume(case, history, no_duplicates)
        ctl = case.control()
        control_history, _, _ = _history(ctl, position, kind, shape)
        ctl.host.lists[C] = control_history
        control = _resume(ctl, control_history, lambda stage: no_duplicates(stage, ctl))
    record_property("2672", dict(arm=arm, position=position, kind=kind, shape=shape,
                                 recovered=recovered, final=final, control=control))
    if control is not None:
        assert Counter(final) == Counter(control), f"{label}: extra/missing rows against recovery-off control"
        assert final == control, f"{label}: row order differs from recovery-off control"
