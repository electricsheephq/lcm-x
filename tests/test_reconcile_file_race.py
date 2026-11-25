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


def _compare(
    tmp_path, monkeypatch, stored, incoming, project=None, reset=None, pin_literals=False,
    *, configure=None, observe_projection=None,
):
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
                if configure is not None:
                    configure(reader, patch)
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
                        if hasattr(run, "projection_rows"):
                            run.projection_rows = NoMemo()

                def visit(run, candidate):
                    visits.append(candidate.cursor)
                    if project is not None:
                        project(candidate, recovery, patch)
                    project_candidate(run, candidate)
                    if observe_projection is not None:
                        observe_projection(candidate)

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


VOLATILE = (
    "[LCM active replay placeholder: assistant output quarantined; "
    "kind=quarantined_assistant_output; reason=fixture; "
    "scope=ignored_message_pattern; field=content; chars=7; bytes=7; sha256=0123456789abcdef]"
)


@pytest.mark.parametrize("change", ["role", "content-and-call-id", "structured", "in-place-structured"])
def test_projection_revalidates_rewritten_placeholder(tmp_path, monkeypatch, change):
    stored = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]
    target = {"role": "assistant", "content": VOLATILE, "tool_call_id": "previous"}
    if change == "in-place-structured":
        target["content"] = [{"type": "text", "text": VOLATILE}]
    incoming = [*deepcopy(stored), target, {"role": "user", "content": "Fresh question"}]

    def project(candidate, recovery, patch):
        if candidate.cursor == 3:
            if change == "role":
                patch.setitem(target, "role", "user")
            elif change == "in-place-structured":
                patch.setitem(target["content"][0], "text", "New output")
            else:
                content = [{"type": "text", "text": "New output"}] if change == "structured" else "New output"
                patch.setitem(target, "content", content)
                patch.setitem(target, "tool_call_id", "new-call")

    def observe(candidate):
        if candidate.cursor in (4, 3):
            assert any(row is target for row in candidate.candidate_identity_messages) == (candidate.cursor == 3)

    cursor, visits, _, _ = _compare(
        tmp_path, monkeypatch, stored, incoming, project, observe_projection=observe,
    )
    assert cursor == 2
    assert visits == [4, 3, 2]


def test_projection_miss_uses_the_role_from_its_key(tmp_path, monkeypatch):
    from hermes_lcm import reconcile

    stored = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]
    target = {"role": "assistant", "content": VOLATILE}
    incoming = [*deepcopy(stored), target, {"role": "user", "content": "Fresh question"}]
    original_text = reconcile.text_content_for_pattern_matching

    def project(candidate, recovery, patch):
        if candidate.cursor == 4:
            def change_role_during_miss(content):
                if content is VOLATILE:
                    patch.setitem(target, "role", "user")
                return original_text(content)

            patch.setattr(reconcile, "text_content_for_pattern_matching", change_role_during_miss)
        elif candidate.cursor == 3:
            patch.setattr(reconcile, "text_content_for_pattern_matching", original_text)
            patch.setitem(target, "role", "assistant")

    def observe(candidate):
        if candidate.cursor == 3:
            assert all(row is not target for row in candidate.candidate_identity_messages)

    cursor, visits, _, _ = _compare(
        tmp_path, monkeypatch, stored, incoming, project, observe_projection=observe,
    )
    assert cursor == 3
    assert visits == [4, 3]


@pytest.mark.parametrize("change", ["digit-limit", "file-move", "ledger", "regex-timeout"])
def test_projection_reads_live_placeholder_evidence(tmp_path, monkeypatch, change):
    from hermes_lcm import engine as engine_module

    regex_times_out = False

    class IgnoreFixtureText:
        def search(self, text, *, timeout):
            if "ignore this projection" not in text.splitlines():
                return False
            if regex_times_out:
                raise TimeoutError
            return True

    stored = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]
    quarantine = (
        "[Externalized LCM ingest payload: assistant output quarantined; "
        "kind=quarantined_assistant_output; reason=fixture; field=content; chars=22; bytes=22; ref=fixture.json]"
    )
    target = {"role": "assistant", "content": quarantine}
    if change == "regex-timeout":
        target["content"] = "ignore this projection"
    incoming = [*deepcopy(stored), target, {"role": "user", "content": "Fresh question"}]
    payload = tmp_path / "payload.txt"
    moved = payload.with_suffix(".moved")
    payload.write_text("ignore this projection")
    original_limit = sys.get_int_max_str_digits()
    mapped = {}

    def reset():
        nonlocal regex_times_out
        regex_times_out = False
        mapped.clear()
        if moved.exists():
            moved.rename(payload)
        sys.set_int_max_str_digits(original_limit)

    def configure(reader, patch):
        if change == "ledger":
            marker = reader._ignored_active_replay_placeholder("ignore this projection")
            patch.setitem(target, "content", marker)
            digest = reader._active_replay_placeholder_digest(marker)
            # A matched row's store-ID admission can change between candidates.
            patch.setattr(reader, "_load_generated_ignored_placeholder_hashes", lambda: {digest})
            patch.setattr(reader._compress_run, "store_ids_by_message_id", mapped)
        else:
            patch.setattr(reader, "_compiled_ignore_message_patterns", [IgnoreFixtureText()])

        def load_payload(ref, **kwargs):
            assert ref == "fixture.json"
            return {"content": payload.read_text()} if payload.exists() else None

        if change == "file-move":
            patch.setattr(engine_module, "load_externalized_payload", load_payload)

    def project(candidate, recovery, patch):
        nonlocal regex_times_out
        if change == "ledger" and candidate.cursor == 3:
            mapped[id(target)] = 123
        if change == "regex-timeout" and candidate.cursor == 3:
            regex_times_out = True
        if change == "digit-limit":
            if candidate.cursor == 4:
                sys.set_int_max_str_digits(0)
                patch.setitem(
                    target, "content", '[{"n": ' + "1" * 641 + ', "text": "ignore this projection", "type": "text"}]',
                )
            elif candidate.cursor == 3:
                sys.set_int_max_str_digits(640)
        elif change == "file-move" and candidate.cursor == 3:
            payload.rename(moved)

    def observe(candidate):
        if candidate.cursor in (4, 3):
            assert any(row is target for row in candidate.candidate_identity_messages) == (candidate.cursor == 3)

    try:
        cursor, visits, _, _ = _compare(
            tmp_path, monkeypatch, stored, incoming, project, reset,
            configure=configure, observe_projection=observe,
        )
    finally:
        sys.set_int_max_str_digits(original_limit)
    assert cursor == 2
    assert visits == [4, 3, 2]


def test_projection_build_work_is_linear(tmp_path, monkeypatch):
    from hermes_lcm import reconcile

    reader, _ = _engine(tmp_path)
    messages = [
        {"role": "assistant" if i % 2 else "user", "content": f"Row {i}"}
        for i in range(64)
    ]
    counts = dict(text=0, regex=0, identity=0, built=0)
    text = reconcile.text_content_for_pattern_matching
    volatile = reader._is_volatile_ignored_quarantine_placeholder
    project = reader._project_replay_candidate
    fields = (
        ("projection_visible", "candidate_visible_messages"),
        ("projection_identity_messages", "candidate_identity_messages"),
        ("projection_visible_prefix", "candidate_visible_prefix"),
        ("projection_prefix", "candidate_prefix"),
    )

    class CountedList(list):
        def append(self, value):
            counts["built"] += 1
            super().append(value)

    def count_text(content):
        counts["text"] += 1
        return text(content)

    def count_volatile(msg, text):
        counts["regex"] += msg["role"] == "assistant"
        return volatile(msg, text)

    def project_candidate(run, candidate):
        project(run, candidate)
        for run_field, candidate_field in fields:
            result = getattr(candidate, candidate_field)
            if result is not getattr(run, run_field, None):
                counts["built"] += len(result)

    def identity(msg):
        counts["identity"] += 1
        return msg["role"], msg["content"], "", "", ""

    try:
        run = reconcile._TailReplayRun(messages, [], None, False, 0)
        run.scaffold_ids = set()
        run.scaffold_positions = []
        run.active_identity = identity
        for run_field, _ in fields:
            setattr(run, run_field, CountedList())
        monkeypatch.setattr(reconcile, "text_content_for_pattern_matching", count_text)
        monkeypatch.setattr(reader, "_is_volatile_ignored_quarantine_placeholder", count_volatile)
        for cursor in range(len(messages), 0, -1):
            project_candidate(run, reconcile._TailReplayCandidate(cursor, messages[:cursor]))
    finally:
        reader.shutdown()
    assert counts == {"text": 64, "regex": 32, "identity": 64, "built": 256}
