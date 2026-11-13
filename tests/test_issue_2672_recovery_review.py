"""#2672 rounds 2/3: keep main's suffix selection when prefix selection is unsafe."""
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


_CARRIED = {
    "todo": dict(role="user", content=(
        "[Your active task list was preserved across context compression]\n"
        "- [>] 1. finish probe (in_progress)")),
    "objective": dict(role="user", content=(
        "[Current user objective preserved from compacted history]\nfinish probe")),
    "note": dict(role="system", content=(
        "[Note: This conversation uses Lossless Context Management (LCM). "
        "Earlier turns have been compacted into hierarchical summaries below.]")),
    "guard": dict(role="user", content="[T11]"),
}
_ANCHOR_VARIANTS = {
    "ptodo@T10+u16": ("todo", 10, None, "last-user", False),
    "ptodo@T12+u16": ("todo", 12, None, "last-user", False),
    "ptodo@T10+a09": ("todo", 10, None, "first-assistant", False),
    "pobjective@T10+u16": ("objective", 10, None, "last-user", False),
    "pnote@T10+u16": ("note", 10, None, "last-user", False),
    "todo@T10+u16": ("todo", None, "[T10]", "last-user", False),
    "note@T10+u16": ("note", None, "[T10]", "last-user", False),
    "todo@T10+a09": ("todo", None, "[T10]", "first-assistant", False),
    "pguard@T10+u16": ("guard", 10, None, "last-user", False),
    "base+u16": (None, None, None, "last-user", True),
    "todo@end+u16": ("todo", None, "end", "last-user", True),
    "todo@1+u16": ("todo", None, "lead", "last-user", True),
}


@pytest.fixture
def carried_case(monkeypatch, request):
    row_kind, after, _, _, _ = _ANCHOR_VARIANTS[request.node.callspec.params["variant"]]
    _stamp_turns(monkeypatch)
    if row_kind == "guard":
        turn = rotation._turn

        def unstamped_turn(index):
            messages = list(turn(index))
            if index == 11:
                messages[0] = {k: v for k, v in messages[0].items() if k not in ("timestamp", "message_uid")}
            return tuple(messages)

        monkeypatch.setattr(rotation, "_turn", unstamped_turn)
    elif after is not None:
        ingest = LCMEngine.ingest
        inserted = set()

        def with_carried_row(engine, messages, *args, **kwargs):
            result = ingest(engine, messages, *args, **kwargs)
            if (engine._session_id == P and id(engine) not in inserted and messages
                    and messages[-1].get("content", "").startswith(f"reply to T{after:02d}:")):
                inserted.add(id(engine))
                messages.append(dict(_CARRIED[row_kind]))
                ingest(engine, messages)
            return result

        monkeypatch.setattr(LCMEngine, "ingest", with_carried_row)
    return request.getfixturevalue("case")


@pytest.mark.parametrize("arm", ["never-resumed", "resumed"])
@pytest.mark.parametrize("variant", _ANCHOR_VARIANTS)
@pytest.mark.parametrize("kind", ["timestamp", "message_uid"])
@pytest.mark.parametrize("shape", ["pair", "reply"])
def test_carried_anchor(carried_case, arm, variant, kind, shape, record_property):
    case = carried_case
    row_kind, after, before, position, positive = _ANCHOR_VARIANTS[variant]

    def history_for(target):
        history, _, suffix = _history(target, "none", kind, shape)
        prefix = "reply to T09:" if position == "first-assistant" else "[T16]"
        changed = [next(i for i, m in enumerate(history) if m["content"].startswith(prefix))]
        if kind == "timestamp":
            history[changed[0]]["timestamp"] += 10000
        else:
            history[changed[0]]["message_uid"] += "-resent"
        if before is not None:
            cursor = len(target.prefix)
            at = cursor if before == "end" else (
                next(i for i, m in enumerate(history[:cursor])
                     if not target.live._is_replayed_context_scaffold_message(m)) if before == "lead" else
                next(i for i, m in enumerate(history[:cursor]) if m["content"].startswith(before)))
            history.insert(at, dict(_CARRIED[row_kind]))
            changed = [i + (i >= at) for i in changed]
            if before == "end":
                suffix = [dict(_CARRIED[row_kind])] + suffix
        if row_kind is not None:
            row = next(m for m in history if m["content"].startswith(_CARRIED[row_kind]["content"]))
            assert target.live._is_replayed_context_scaffold_message(row) == (row_kind != "guard")
            assert not target.live._is_ignored_active_replay_placeholder(row, row["content"])
            if row_kind == "guard":
                key = target.live._identity_anchor_view_key(row)
                assert key is not None and key[0] is None
        return history, changed, suffix

    history, changed, suffix = history_for(case)
    case.host.lists[C] = history
    parent = ident(rows(case, P))
    assert not case.run()
    recovered = ident(rows(case))
    case.run()
    assert ident(rows(case)) == recovered
    assert ident(rows(case, P)) == parent
    main_rows = [(C, P, m["role"], m["content"]) for m in suffix]
    assert Counter(main_rows) <= Counter(recovered)
    assert len(recovered) == len(set(recovered))
    expected = ([history[i] for i in changed] if positive else []) + suffix
    if arm == "never-resumed":
        assert_suffix(case, expected)
    final = control = None
    if arm == "resumed":
        final = _resume(case, history, lambda stage: None)
        ctl = case.control()
        control_history, _, _ = history_for(ctl)
        control = _resume(ctl, control_history, lambda stage: None)
        extra = list((Counter(final) - Counter(control)).elements())
        missing = list((Counter(control) - Counter(final)).elements())
        record_property("2672-r3", dict(variant=variant, arm=arm, kind=kind, shape=shape,
                                       extra=extra, missing=missing, final=final, control=control))
        assert not extra and not missing, f"extra={len(extra)} missing={len(missing)}"
        assert len(final) == len(set(final))
        if positive:
            assert final == control
        else:
            # Main's suffix-first order is the accepted fallback, with the same rows.
            assert final == main_rows + [row for row in control if row not in main_rows]
            assert final != control

    assert recovered == [(C, P, m["role"], m["content"]) for m in expected]
