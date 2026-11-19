"""Cursor and row preservation with the #2924 walk memo enabled and bypassed."""

from copy import deepcopy
import subprocess
import sys

import pytest

from tests.test_issue_10_persisted_output_spillover import (
    FULL_OUTPUT, SESSION_ID, _engine, _marker, _messages, _write_output,
)
from tests.test_issue_1090_replay_prefix_scaling import NOTE
from tests.test_issue_772_replay_keeps_host_stubs import CLEARED


class NoMemo(dict):
    def __setitem__(self, key, value):
        pass


def _compare(tmp_path, monkeypatch, stored, incoming, project=None, reset=None, pin_literals=False):
    writer, _ = _engine(tmp_path)
    try:
        writer._ingest_messages(stored)
        rows = writer._store.get_session_messages(SESSION_ID)
    finally:
        writer.shutdown()
    before = deepcopy(incoming)
    observations, literal_counts = [], []
    for memo in (False, True):
        if reset is not None:
            reset()
        reader, _ = _engine(tmp_path)
        try:
            with monkeypatch.context() as patch:
                prepare = reader._prepare_tail_replay_run
                project_candidate = reader._project_replay_candidate
                collect = reader._collect_persisted_replay_evidence
                identity = reader._message_replay_identity
                visits, recovery, cleanup = [], [], []
                literals = 0

                def count_identity(msg, **kwargs):
                    nonlocal literals
                    literals += kwargs.get("allow_pruned_tool_alias") is False and msg.get("content") == CLEARED
                    return identity(msg, **kwargs)

                def prepare_run(run):
                    prepare(run)
                    if not memo:
                        run.literal_pruned_identities = NoMemo()
                        run.persisted_marker_texts = NoMemo()
                        run.cleanup_identities = NoMemo()

                def visit(run, candidate):
                    visits.append(candidate.cursor)
                    if project is not None:
                        project(candidate, recovery, patch)
                    project_candidate(run, candidate)

                def observe_recovery(run, candidate):
                    collect(run, candidate)
                    recovery.append((candidate.cursor, candidate.candidate_has_unrecoverable_persisted_marker))
                    cleanup.append((candidate.cursor, candidate.raw_suffix_needs_cleanup_equivalence))

                patch.setattr(reader, "_prepare_tail_replay_run", prepare_run)
                patch.setattr(reader, "_project_replay_candidate", visit)
                patch.setattr(reader, "_collect_persisted_replay_evidence", observe_recovery)
                patch.setattr(reader, "_message_replay_identity", count_identity)
                tail = [reader._message_replay_identity(row, stored_row=True, allow_pruned_tool_alias=False)
                        for row in rows]
                cursor = reader._find_reconciled_cursor_for_store_tail(
                    incoming, tail, stored_tail_rows=rows, allow_empty_prefix=True,
                    session_count=len(rows), raw_session_count=len(rows))
            observations.append((cursor, visits, recovery, cleanup))
            literal_counts.append(literals)
            assert visits
            assert incoming == before
            assert reader._store.get_session_messages(SESSION_ID) == rows
        finally:
            reader.shutdown()
    assert observations[0] == observations[1]
    if pin_literals:
        assert 0 < literal_counts[1] < literal_counts[0]
    return observations[0]


@pytest.mark.parametrize("shape", ["pure", "pruned", "externalized", "marker", "missing", "scaffold"])
@pytest.mark.parametrize("cut", ["top", "middle", "none"])
def test_cursor_oracle(tmp_path, monkeypatch, shape, cut):
    stored = [{"role": "user", "content": "Persisted question"},
              {"role": "assistant", "content": "Persisted answer"}]
    incoming = deepcopy(stored)
    if shape in {"marker", "missing"}:
        spill = tmp_path / "hermes" / "cache" / "spillover" / "call_spill.txt"
        incoming = _messages(_write_output(spill))
        stored = deepcopy(incoming)
        if shape == "missing":
            spill.rename(spill.with_suffix(".moved"))
    elif shape in {"pruned", "externalized"}:
        stored += [msg for i in range(4) for msg in _messages(FULL_OUTPUT, call_id=f"call-{i}")]
        incoming = deepcopy(stored)
        if shape == "pruned":
            incoming[3]["content"] = CLEARED
            incoming[5].update(content=CLEARED, tool_calls=[])
    elif shape == "scaffold":
        incoming.insert(0, NOTE)
    expected = len(incoming) if shape != "missing" and cut != "none" else 0
    if cut == "none":
        incoming.insert(0, {"role": "user", "content": "Different opening"})
    if cut != "top":
        incoming.append({"role": "user", "content": "Fresh turn"})
    cursor, *_ = _compare(tmp_path, monkeypatch, stored, incoming, pin_literals=shape == "pruned" and cut != "top")
    assert cursor == expected


@pytest.mark.parametrize("change", ["none", "move", "rewrite", "digit-limit"])
def test_reconcile_marker_between_candidates(tmp_path, monkeypatch, change):
    spill = tmp_path / "hermes" / "cache" / "spillover" / "call_spill.txt"
    moved = spill.with_suffix(".moved")
    stored = _messages(_write_output(spill))
    incoming = deepcopy(stored) + [{"role": "user", "content": "Fresh turn"}]
    original_limit = sys.get_int_max_str_digits()
    if change == "digit-limit":
        incoming[1]["content"] = stored[1]["content"].replace("too large (", "too large (" + "0" * 640)

    def project(candidate, recovery, patch):
        if candidate.cursor == 3 and change == "digit-limit":
            sys.set_int_max_str_digits(640)
        if candidate.cursor == 2 and change != "none":
            assert recovery == [(3, False)]
            if change == "rewrite":
                patch.setitem(incoming[1], "content", _marker(spill.with_name("missing.txt")))
            else:
                sys.set_int_max_str_digits(0 if change == "digit-limit" else original_limit)
                subprocess.run(["mv", str(spill), str(moved)], check=True)

    def reset():
        sys.set_int_max_str_digits(0 if change == "digit-limit" else original_limit)
        if moved.exists():
            moved.rename(spill)

    try:
        cursor, visits, recovery, _ = _compare(tmp_path, monkeypatch, stored, incoming, project, reset)
    finally:
        sys.set_int_max_str_digits(original_limit)
    assert visits[:2] == [3, 2]
    assert recovery[:2] == [(3, False), (2, change != "none")]
    assert cursor == (2 if change == "none" else 0)


@pytest.mark.parametrize("case", ["literal-call-id", "pruned-with-calls", "rewrite-call-id", "rewrite-content"])
def test_literal_pruned_cursor_oracle(tmp_path, monkeypatch, case):
    older = _messages(FULL_OUTPUT, "a") + _messages(FULL_OUTPUT, "b")
    suffix = _messages("durable anchor", "anchor") + _messages(CLEARED, "a") + _messages(CLEARED, "b")
    if case == "pruned-with-calls":
        suffix[-1]["tool_calls"] = suffix[-2]["tool_calls"]
    incoming = deepcopy(suffix) + _messages("fresh", "x") + _messages("fresh", "y")
    rewrite = {"rewrite-call-id": ("tool_call_id", "new-call"),
               "rewrite-content": ("content", "New tool output")}.get(case)

    def project(candidate, recovery, patch):
        if rewrite and candidate.cursor == len(incoming) - 1:
            assert recovery == [(len(incoming), False)]
            patch.setitem(incoming[len(suffix) - 1], *rewrite)

    # A role-keyed cache with full validation would still thrash between a and b.
    cursor, *_ = _compare(
        tmp_path, monkeypatch, older + suffix, incoming, project, pin_literals=case == "literal-call-id",
    )
    assert cursor == len(suffix) - bool(rewrite)


def test_raw_cleanup_cursor_oracle(tmp_path, monkeypatch):
    clean = [{"role": "assistant", "content": text} for text in ("First", "Second")]
    stored = clean + [{"role": "assistant", "content": "<think>Private reasoning</think>"}] + clean
    cursor, _, _, cleanup = _compare(tmp_path, monkeypatch, stored, [NOTE, *deepcopy(stored)])
    assert cursor == 1
    assert cleanup[0] == (6, True)
    assert (3, False) in cleanup


def test_raw_cleanup_digit_limit_between_candidates(tmp_path, monkeypatch):
    # Both matching suffixes contain the same reasoning-only assistant identity.
    # The 641-digit number keeps the first cleanup from decoding it as a dict.
    reasoning = {"role": "assistant", "content": '{"number": ' + "1" * 641 + ', "type": "thinking"}'}
    answer = {"role": "assistant", "content": "Visible answer"}
    stored = [reasoning, answer] * 2
    incoming = [NOTE, *deepcopy(stored)]
    original_limit = sys.get_int_max_str_digits()

    def project(candidate, recovery, patch):
        if candidate.cursor == 5:
            sys.set_int_max_str_digits(640)
        elif candidate.cursor == 3:
            sys.set_int_max_str_digits(0)

    try:
        sys.set_int_max_str_digits(0)
        cursor, _, _, cleanup = _compare(
            tmp_path, monkeypatch, stored, incoming, project, reset=lambda: sys.set_int_max_str_digits(0),
        )
    finally:
        sys.set_int_max_str_digits(original_limit)
    assert cleanup[:3] == [(5, False), (4, False), (3, True)]
    assert cursor == 1
