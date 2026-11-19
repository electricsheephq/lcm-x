"""Cursor and row preservation with the #2924 walk memo enabled and bypassed."""

from copy import deepcopy
import subprocess

import pytest

from tests.test_issue_10_persisted_output_spillover import (
    FULL_OUTPUT, SESSION_ID, _engine, _messages, _write_output,
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
                visits, recovery = [], []
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
                        project(candidate, recovery)
                    project_candidate(run, candidate)

                def observe_recovery(run, candidate):
                    collect(run, candidate)
                    recovery.append((candidate.cursor, candidate.candidate_has_unrecoverable_persisted_marker))

                patch.setattr(reader, "_prepare_tail_replay_run", prepare_run)
                patch.setattr(reader, "_project_replay_candidate", visit)
                patch.setattr(reader, "_collect_persisted_replay_evidence", observe_recovery)
                patch.setattr(reader, "_message_replay_identity", count_identity)
                tail = [reader._message_replay_identity(row, stored_row=True) for row in rows]
                cursor = reader._find_reconciled_cursor_for_store_tail(
                    incoming, tail, stored_tail_rows=rows, allow_empty_prefix=True,
                    session_count=len(rows), raw_session_count=len(rows))
                observations.append((cursor, visits, recovery))
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
    cursor, _, _ = _compare(tmp_path, monkeypatch, stored, incoming, pin_literals=shape == "pruned" and cut != "top")
    assert cursor == expected


@pytest.mark.parametrize("move_spill", [False, True])
def test_reconcile_file_moves_between_candidates(tmp_path, monkeypatch, move_spill):
    spill = tmp_path / "hermes" / "cache" / "spillover" / "call_spill.txt"
    moved = spill.with_suffix(".moved")
    stored = _messages(_write_output(spill))
    incoming = stored + [{"role": "user", "content": "Fresh turn"}]

    def project(candidate, recovery):
        if candidate.cursor == 2 and move_spill:
            assert recovery == [(3, False)]
            subprocess.run(["mv", str(spill), str(moved)], check=True)

    # Restore at preparation so the memo-off and memo-on arms see the same
    # file when they build the restart's occurrence identities.
    def restore_after_move():
        if moved.exists():
            moved.rename(spill)
    cursor, visits, recovery = _compare(tmp_path, monkeypatch, stored, incoming, project, restore_after_move)
    assert visits[:2] == [3, 2]
    assert recovery[:2] == [(3, False), (2, move_spill)]
    assert cursor == (0 if move_spill else 2)
