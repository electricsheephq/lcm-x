"""#2672 round 2: keep main's suffix selection when prefix selection is unsafe."""
from collections import Counter

import pytest

from hermes_lcm import host_row_recovery as recovery, message_patterns
from hermes_lcm.engine import LCMEngine
from tests import test_issue_519_replaced_tail as rotation
from tests.engine_helpers import _FakeTimeoutRegexEngine
from tests.test_issue_2624_host_row_recovery import C, P, assert_suffix, case as _case_fixture, ident, rows
from tests.test_issue_2672_recovery_changed_prefix import (
    _history, _resume, _stamp_turns, stamped_case as _stamped_case_fixture,
)

case = _case_fixture
stamped_case = _stamped_case_fixture


@pytest.fixture
def ignored_case(monkeypatch, request):
    try:
        import regex
    except ImportError:
        regex = _FakeTimeoutRegexEngine
    monkeypatch.setattr(message_patterns, "_regex_engine", regex)
    config = rotation._config
    monkeypatch.setattr(rotation, "_config", lambda path, **kw: config(
        path, ignore_message_patterns=[r"^reply to T11:"], **kw))
    _stamp_turns(monkeypatch)
    return request.getfixturevalue("case")


@pytest.mark.parametrize("arm", ["never-resumed", "resumed"])
@pytest.mark.parametrize("position", ["ign-a", "ign-a+u", "none"])
@pytest.mark.parametrize("kind", ["timestamp", "message_uid"])
@pytest.mark.parametrize("shape", ["pair", "reply"])
def test_ignored_prefix(ignored_case, arm, position, kind, shape):
    case = ignored_case
    # A11 is already an LCM placeholder in the compacted host output; only
    # ign-a+u changes a remaining real occurrence, just like the review witness.
    history, changed, suffix = _history(
        case, "last-user" if position == "ign-a+u" else "none", kind, shape)
    placeholders = [i for i, row in enumerate(history[:len(case.prefix)])
                    if case.live._is_ignored_active_replay_placeholder(row, row["content"])]
    assert placeholders and (not changed or max(placeholders) < min(changed))
    case.host.lists[C] = history
    parent = ident(rows(case, P))
    assert not case.run()
    recovered = ident(rows(case))
    main_rows = [(C, P, row["role"], row["content"]) for row in suffix]
    assert Counter(main_rows) <= Counter(recovered)  # Never lose main's rows.
    assert len(recovered) == len(set(recovered))
    case.run()
    assert ident(rows(case)) == recovered
    assert ident(rows(case, P)) == parent

    def no_duplicates(stage, target=case):
        actual = ident(rows(target))
        assert len(actual) == len(set(actual)), stage

    if arm == "resumed":
        final = _resume(case, history, no_duplicates)
        ctl = case.control()
        control_history, _, _ = _history(
            ctl, "last-user" if changed else "none", kind, shape)
        control = _resume(ctl, control_history, lambda stage: no_duplicates(stage, ctl))
        assert Counter(final) == Counter(control)  # No placeholder, extra or missing row.
        # Accepted fallback: main stores Q/R (or R), then U16, then NQ/NR.
        # The recovery-off control stores U16 before Q/R (or R).
        expected = control[len(changed):len(changed) + len(suffix)] + control[:len(changed)]
        expected += control[len(changed) + len(suffix):]
        assert final == expected
        assert (final != control) == bool(changed)
    else:
        assert_suffix(case, suffix)


def test_ordinary_rotation_prefix_keeps_changed_selection(stamped_case):
    case = stamped_case
    assert any(case.live._is_replayed_context_scaffold_message(m) for m in case.prefix)
    assert not any(case.live._is_ignored_active_replay_placeholder(m, m["content"])
                   for m in case.prefix)
    history, changed, suffix = _history(case, "last-user", "timestamp", "pair")
    case.host.lists[C] = history
    assert not case.run()
    assert_suffix(case, [history[i] for i in changed] + suffix)


@pytest.mark.parametrize("position", ["middle-user", "two"])
def test_mid_prefix_system_keeps_main_suffix(stamped_case, position):
    case = stamped_case
    history, changed, suffix = _history(case, position, "timestamp", "pair")
    at = len(case.prefix) - 2
    history.insert(at, dict(role="system", content=(
        "[Note: This conversation uses Lossless Context Management (LCM). "
        "Earlier turns have been compacted into hierarchical summaries below.]")))
    assert min(changed) < at
    assert case.live._is_replayed_context_scaffold_message(history[at])
    case.host.lists[C] = history
    before = recovery.host_row_recovery_counts()
    assert not case.run()
    assert_suffix(case, suffix)
    after = recovery.host_row_recovery_counts()
    assert after["recovered_rows"] == before["recovered_rows"] + len(suffix)
    assert after["suffix_rejected"] == before["suffix_rejected"]


@pytest.mark.parametrize("position", ["none", "last-user"])
@pytest.mark.parametrize("helper", ["_remapped_output_metadata", "_host_occurrence_changed",
                                    "_durable_commit_proof_payload"])
def test_selection_failure_keeps_main_result(stamped_case, monkeypatch, position, helper):
    case = stamped_case
    history, _, suffix = _history(case, position, "timestamp", "pair")
    case.host.lists[C] = history
    calls = []

    def raising(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("injected metadata/proof read failure")

    monkeypatch.setattr(LCMEngine, helper, raising)
    before = recovery.host_row_recovery_counts()
    warnings = case.run()
    assert calls
    after = recovery.host_row_recovery_counts()
    assert after["worker_error"] == before["worker_error"]
    if helper == "_durable_commit_proof_payload":
        assert not rows(case)
        assert len(warnings) == 1 and "reason=proof" in warnings[0]
        assert after["proof"] == before["proof"] + 1
        assert after["unproven"] == before["unproven"] + 1
    else:
        assert not warnings
        assert_suffix(case, suffix)
        assert after["recovered_rows"] == before["recovered_rows"] + len(suffix)
