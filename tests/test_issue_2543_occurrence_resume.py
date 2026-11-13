"""#2543: bounded occurrence resume on normal off-current session ends (I1/I2/I6)."""

import copy
import json
import sqlite3
import time
from contextlib import contextmanager

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm import engine as engine_module
from tests.test_compression_boundary import _stub_summarizer, _turn
from hermes_lcm.session_patterns import compile_session_patterns
from tests.invariants.test_issue_2543_cold_inline_block import _b64, _call
from tests.invariants.test_p1e_cold_tails import fail_end_read
from tests.test_issue_10_persisted_output_spillover import _write_output


SHAPES = ("T7-wrapped", "T7-unwrapped", "T7K", "T6i", "T6ii", "plain")
FINAL = {"role": "assistant", "content": "FINAL"}


def shape_row(shape):
    if shape in ("T7-wrapped", "T7K", "plain"):
        return _call({"T7-wrapped": "wrapped-base64", "T7K": "data-uri-key", "plain": "plain"}[shape])
    long = _b64(3420)
    wrapped = "\n".join(long[i:i + 76] for i in range(0, len(long), 76))
    if shape == "T7-unwrapped":
        row = _call("plain")
        row["tool_calls"][0]["function"]["arguments"] = json.dumps({"content": long}).replace("/", r"\u002f")
        return row
    if shape == "T6i":  # Below whole-media externalization's size threshold.
        return {"role": "user", "content": [{"type": "text", "text": "see"},
                {"type": "image_url", "image_url": {"url": "data:image\\/png;base64," + _b64(1200)}}]}
    return {"role": "user", "content": [{"type": "text", "text": "hdr\n" + wrapped + "\nend"}]}


def history(shape="plain"):
    rows = [{"role": ("user", "assistant")[i % 2], "content": f"pad {i}"} for i in range(10)]
    rows[3] = shape_row(shape)
    if rows[3].get("tool_calls"):
        rows[4] = {"role": "tool", "tool_call_id": "m1", "content": "ok 1"}
    return rows


class Lane:
    def __init__(self, root):
        self.root = root
        self.engines = []
        self.open()

    def open(self):
        self.engine = LCMEngine(config=LCMConfig(database_path=str(self.root / "lcm.db"),
            large_output_externalization_path=str(self.root / "externalized")), hermes_home=str(self.root))
        self.engines.append(self.engine)
        self.engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")

    def seed(self, rows):
        self.engine.ingest(copy.deepcopy(rows))
        assert len(self.rows()) == len(rows)
        self.foreground()

    def foreground(self):
        self.engine.on_session_start("foreground", platform="cli", context_length=200_000, conversation_id="fg")

    def restart(self):
        self.engine.shutdown()
        self.open()
        self.foreground()

    def rows(self):
        return self.engine._store.get_session_messages("S1")

    def forms(self):
        return self.engine._session_end_host_forms(self.rows(), store=self.engine._store, session_id="S1")

    def end(self, view, monkeypatch, busy=False):
        if busy:
            with monkeypatch.context() as patch:
                patch.setattr(self.engine._store, "get_range", fail_end_read(self.engine))
                self.engine.on_session_end("S1", copy.deepcopy(view))
            assert len(self.engine._pending_session_tails) == 1
            tail = self.engine._pending_session_tails[0]
            self.engine._retry_pending_session_tails(deadline=time.monotonic() + 60)
            assert not self.engine._pending_session_tails
            assert self.engine._cold_tail_absent_count(tail) == 0
        else:
            self.engine.on_session_end("S1", copy.deepcopy(view))

    def assert_present(self, view):
        rows, forms = self.rows(), self.forms()
        for message in view:
            # Present: equal by the session-end compare, by the recorded host form, or by the replay identity the
            # current-session route dedupes on (#2543 review CR-3).
            identity = self.engine._message_replay_identity(message)
            assert any(self.engine._session_end_rows_equal(row, message, session_id="S1") or
                       forms.get(row["store_id"]) == self.engine._session_end_host_form(message) or
                       self.engine._message_replay_identity(row, stored_row=True) == identity for row in rows)


@pytest.fixture
def lanes(tmp_path, monkeypatch):
    for name in ("_schedule_rollup_maintenance", "_schedule_embedding_maintenance"):
        monkeypatch.setattr(LCMEngine, name, lambda *args: None)
    made = []

    def create(name="lane"):
        root = tmp_path / name
        root.mkdir()
        lane = Lane(root)
        made.append(lane)
        return lane

    yield create
    for lane in made:
        for engine in reversed(lane.engines):
            engine.shutdown()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("busy", [False, True], ids=["direct", "busy"])
@pytest.mark.parametrize("restart", [False, True])
def test_matrix_three_ends(lanes, monkeypatch, shape, busy, restart):
    lane = lanes()
    first = history(shape)
    lane.seed(first)
    n = len(first)
    view = first + [FINAL]
    lane.end(view, monkeypatch, busy)
    # T6i, T6ii and unwrapped T7 have the stored row's replay identity, so the ladder stores only FINAL (CR-3); a later
    # bind of the session then finds its history in order.
    copied = int(shape not in ("plain", "T6i", "T6ii", "T7-unwrapped"))
    assert len(lane.rows()) == n + 1 + copied
    lane.assert_present(view)
    assert len(lane.forms()) == (0 if shape == "plain" else 1 + copied)
    if restart:
        lane.restart()
    before = lane.rows()
    lane.end(view, monkeypatch, busy)
    assert lane.rows() == before
    if restart:
        lane.restart()
    lane.end(view + [{"role": "user", "content": "Thanks."}], monkeypatch, busy)
    assert len(lane.rows()) == n + 2 + copied
    lane.assert_present(view + [{"role": "user", "content": "Thanks."}])
    if shape == "plain":
        assert not lane.forms()


@pytest.mark.parametrize("busy", [False, True])
@pytest.mark.parametrize("change", ["edit", "insert", "delete", "repeat"])
def test_plain_occurrences(lanes, monkeypatch, busy, change):
    lane = lanes()
    first = history()
    if change == "repeat":
        first[5:8] = [{"role": "user", "content": "ok"} for _ in range(3)]
    lane.seed(first)
    view = copy.deepcopy(first)
    if change == "edit":
        view[1]["content"] = "host edit"
    elif change == "insert":
        view.insert(2, {"role": "user", "content": "inserted"})
    elif change == "delete":
        view.pop(2)
    else:
        view.append({"role": "user", "content": "ok"})
    view.append(FINAL)
    lane.end(view, monkeypatch, busy)
    assert len(lane.rows()) == len(first) + (1 if change == "delete" else 2)
    lane.assert_present(first + view)
    before = lane.rows()
    lane.end(view, monkeypatch, busy)
    assert lane.rows() == before


def test_L1_copy_order_and_assembly_stub(lanes, monkeypatch):
    """L-1: one out-of-order transformed-call copy; assembly repeats its id with a missing-result stub."""
    lane = lanes()
    first = history("T7-wrapped")
    lane.seed(first)
    before = lane.rows()
    lane.end(first + [FINAL], monkeypatch)
    rows = lane.rows()
    assert rows[:len(before)] == before
    assert rows[-2]["tool_calls"] and rows[-1]["content"] == "FINAL"
    lane.engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
    assembled = lane.engine._assemble_context({"role": "system", "content": "system"}, rows)
    calls = [call for row in assembled for call in (row.get("tool_calls") or []) if call["id"] == "m1"]
    assert len(calls) == 2
    stub = lane.engine._missing_tool_result_stub("m1")
    assert any(all(row.get(key) == value for key, value in stub.items()) for row in assembled)


def test_L2_drop_then_equal_new_occurrence(lanes, monkeypatch):
    """L-2: a later equal new row consumes a dropped occurrence, with all content retained."""
    lane = lanes()
    first = history()
    lane.seed(first)
    view = first[:2] + first[3:] + [FINAL]
    lane.end(view, monkeypatch)
    before = lane.rows()
    lane.end(view + [first[2]], monkeypatch)
    assert lane.rows() == before
    lane.assert_present(first + view)


@pytest.mark.parametrize("busy", [False, True], ids=["direct", "busy"])
def test_L3_compacted_summary_then_bind(lanes, monkeypatch, busy):
    """L-3 r2.3: a commit proof gates out both ladders; the next bind adds only FINAL and the prompt."""
    lane = lanes()
    monkeypatch.setattr(engine_module, "summarize_with_escalation", _stub_summarizer())
    lane.engine._config.fresh_tail_count = 6
    lane.engine._config.leaf_chunk_tokens = 400
    first = [row for i in range(1, 13) for row in _turn(i)] + [_turn(13)[0]]
    lane.engine.ingest(copy.deepcopy(first))
    compacted = lane.engine.compress(copy.deepcopy(first), force=True)
    assert lane.engine._last_compression_status == "compacted"
    summary = next(row for row in compacted if lane.engine._is_verified_replay_scaffold_message(row))
    before = lane.rows()
    assert len(before) == len(first) == 25
    assert lane.engine._store.read_metadata_json("compaction_commit_proof:S1") is not None
    lane.foreground()
    view = compacted + [FINAL]
    with monkeypatch.context() as patch:
        patch.setattr(lane.engine, "_session_end_occurrence_resume", lambda *a, **k: pytest.fail("proof entered ladder"))
        if busy:
            with monkeypatch.context() as blocked:
                blocked.setattr(lane.engine._store, "get_range", fail_end_read(lane.engine))
                lane.engine.on_session_end("S1", copy.deepcopy(view))
            assert len(lane.engine._pending_session_tails) == 1
            lane.engine._retry_pending_session_tails(deadline=time.monotonic() + 60)
            assert len(lane.engine._pending_session_tails) == 1
        else:
            lane.end(view, monkeypatch)
        assert lane.rows() == before
        assert not lane.forms()
    lane.engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
    prompt = {"role": "user", "content": "new prompt after bind"}
    lane.engine.ingest(copy.deepcopy(view + [prompt]))
    after = lane.rows()
    assert after[:len(before)] == before
    assert len(after) == 27
    assert [row["content"] for row in after[len(before):]] == [FINAL["content"], prompt["content"]]
    assert not lane.forms()
    # Summary text alone is no gate: without a proof it is an unknown host occurrence, copied once.
    unproven = lanes("unproven")
    unproven.seed(history())
    pasted = history()[:2] + [summary] + history()[2:] + [FINAL]
    unproven.end(pasted, monkeypatch, busy)
    assert len(unproven.rows()) == len(history()) + 2
    assert len(unproven.forms()) == 2
    unproven.assert_present(pasted)
    copied = unproven.rows()
    unproven.end(pasted, monkeypatch, busy)
    assert unproven.rows() == copied


def test_L4_unchanged_route_gates(lanes, monkeypatch):
    """L-4: same-id bypass, direct bypass and auxiliary-reused sessions retain main's answer."""
    for route in ("same-id", "direct", "auxiliary"):
        lane = lanes(route)
        first = history()
        lane.seed(first)
        e = lane.engine
        if route == "same-id":
            e._compiled_ignore_session_patterns = compile_session_patterns(["cron"])
            e.on_session_start("S1", platform="cron", context_length=200_000)
            e.ingest(copy.deepcopy(first))
            e.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
        elif route == "direct":
            e._auxiliary_foreground_reused_session_ids.add("S1")
            e._compiled_ignore_session_patterns = compile_session_patterns(["S1"])
        else:
            e._auxiliary_foreground_reused_session_ids.add("S1")
        with monkeypatch.context() as patch:
            patch.setattr(e, "_session_end_occurrence_resume", lambda *a, **k: pytest.fail("gate entered ladder"))
            lane.end(first + [FINAL], monkeypatch)
        assert len(lane.rows()) == len(first) + int(route != "same-id")
        assert not lane.forms()
        if route == "auxiliary":
            changed = copy.deepcopy(first + [FINAL])
            changed[1]["content"] = "auxiliary host edit"
            before = lane.rows()
            with monkeypatch.context() as patch:
                patch.setattr(e, "_session_end_occurrence_resume", lambda *a, **k: pytest.fail("auxiliary entered ladder"))
                lane.end(changed, monkeypatch)
            assert lane.rows() == before


def test_L5_absent_count_form_and_stamp_rule(lanes, monkeypatch):
    """L-5: a ladder copy is present by form, while a known host stamp still needs the same stored stamp."""
    lane = lanes()
    first = history("T7-wrapped")
    lane.seed(first)
    view = first + [FINAL]
    lane.end(view, monkeypatch)
    tail = dict(view=view, session_id="S1", conversation_id="conv1", hermes_home=str(lane.root), store=lane.engine._store)
    assert lane.engine._cold_tail_absent_count(tail) == 0
    timed = copy.deepcopy(view)
    timed[3]["timestamp"] = 900.0
    assert lane.engine._cold_tail_absent_count(dict(tail, view=timed)) == 1


@pytest.mark.parametrize("busy", [False, True])
def test_two_transforms_and_storage_longer_than_view(lanes, monkeypatch, busy):
    lane = lanes()
    first = history("T7-wrapped")
    first[6] = shape_row("T6ii")
    lane.seed(first)
    # Witness A: a current-route ambiguous re-store, using the ordinary append path with no forms.
    lane.engine._append_off_current_session_end_suffix("S1", [first[0]], source="cli", conversation_id="conv1")
    lane.end(first + [FINAL], monkeypatch, busy)
    assert len(lane.rows()) == len(first) + 3  # the re-stored first row, the T7 copy and FINAL; T6ii matches (CR-3)
    assert len(lane.forms()) == 2
    # Witness B: the two ladder copies themselves make storage longer than the view.
    before = lane.rows()
    lane.end(first + [FINAL], monkeypatch, busy)
    assert lane.rows() == before
    lane.end(first + [FINAL, {"role": "assistant", "content": "later"}], monkeypatch, busy)
    assert len(lane.rows()) == len(before) + 1


def test_marker_bridge_after_earlier_edit(lanes, monkeypatch):
    lane = lanes()
    first = history()
    path = lane.root / "cache" / "spillover" / "m1.txt"
    first[4]["content"] = _write_output(path)
    lane.seed(first)
    path.rename(lane.root / "pruned-output")
    view = copy.deepcopy(first)
    view[0]["content"] = "earlier edit"
    lane.end(view, monkeypatch)
    assert len(lane.rows()) == len(first) + 1
    assert lane.rows()[-1]["content"] == "earlier edit"


def test_stamps_three_passes_and_dual_keys(lanes, monkeypatch):
    lane = lanes()
    first = history()
    first[6:9] = [{"role": "user", "content": "ok"}, {"role": "user", "content": "ok", "timestamp": 7.0},
                  {"role": "user", "content": "ok", "timestamp": 8.0}]
    lane.seed(first)
    view = copy.deepcopy(first)
    view[0]["timestamp"] = 1.0
    view[1]["content"] = "edited"
    view[6:9] = [{"role": "user", "content": "ok"}, {"role": "user", "content": "ok", "timestamp": 8.0},
                 {"role": "user", "content": "ok", "timestamp": 7.0}]
    def resume(v):
        return lane.engine._session_end_occurrence_resume(v, session_id="S1", conversation_id="conv1",
                    store=lane.engine._store, hermes_home=str(lane.root), stamps=True)
    assert resume(view + [FINAL]) == [view[1], FINAL]
    retyped = copy.deepcopy(first)
    retyped[7]["timestamp"] = 9.0
    assert resume(retyped + [FINAL]) == [retyped[7], FINAL]
    # One occurrence's form and counted identity cannot satisfy two host rows.
    lane.engine._append_off_current_session_end_suffix("S1", [{"role": "user", "content": "one"}],
        source="cli", conversation_id="conv1", record_host_forms=True)
    one = {"role": "user", "content": "one"}
    assert resume([one, one]) == [one]
    # A missing versus empty result id changes the raw form but keeps the compared identity.
    call = _call("plain")
    call["tool_calls"][0]["id"] = "dual-key"
    lane.engine._append_off_current_session_end_suffix("S1", [call], source="cli", conversation_id="conv1", record_host_forms=True)
    changed = copy.deepcopy(call)
    changed["tool_call_id"] = ""
    assert lane.engine._session_end_host_form(call) != lane.engine._session_end_host_form(changed)
    assert lane.engine._session_end_prefix_compare_identity(call, session_id="S1") == lane.engine._session_end_prefix_compare_identity(changed, session_id="S1")
    assert resume([changed]) == []


def gc_copy(lane):
    """Use the existing payload-GC rewrite on a copied tool result; retain the recorded host form."""
    for row in lane.rows():
        if row["role"] == "tool" and row["store_id"] in lane.forms():
            lane.engine._store.gc_externalized_tool_result(row["store_id"], "[Externalized tool output: payload GC]")


def test_gc_repeated_ends(lanes, monkeypatch):
    lane = lanes()
    first = history()
    first[4]["content"] = shape_row("T6ii")["content"]
    lane.seed(first)
    lane.end(first + [FINAL], monkeypatch)
    assert len(lane.rows()) == len(first) + 1  # T6ii matches its stored row by replay identity (CR-3)
    for _ in range(3):
        gc_copy(lane)
        lane.restart()
        before = lane.rows()
        lane.end(first + [FINAL], monkeypatch)
        assert lane.rows() == before
        lane.assert_present(first + [FINAL])


def test_plain_redaction_no_copy_or_form(lanes, monkeypatch):
    lane = lanes()
    lane.engine._config.sensitive_patterns_enabled = True
    lane.engine._config.sensitive_patterns = ["api_key"]
    first = history()
    first[2]["content"] = "api_key=synthetic2543value"
    lane.seed(first)
    lane.end(first + [FINAL], monkeypatch)
    assert len(lane.rows()) == len(first) + 1
    assert not lane.forms()
    lane.assert_present(first + [FINAL])


def test_helper_busy_fallback_ignore_and_metadata_validation(lanes, monkeypatch):
    lane = lanes()
    lane.seed(history())
    e = lane.engine
    kwargs = dict(session_id="S1", conversation_id="conv1", store=e._store, hermes_home=str(lane.root), stamps=False)
    for error in (sqlite3.OperationalError("database is locked"), ValueError("injected")):
        with monkeypatch.context() as patch:
            def fail(*a, **k):
                raise error
            patch.setattr(e._store, "read_metadata_json_many", fail)
            if isinstance(error, sqlite3.OperationalError):
                with pytest.raises(sqlite3.OperationalError):
                    e._session_end_occurrence_resume([FINAL], **kwargs)
            else:
                assert e._session_end_occurrence_resume([FINAL], **kwargs) is None
    with monkeypatch.context() as patch:
        patch.setattr(e, "_cold_tail_ingest_view", lambda tail: [m for m in tail["view"] if m.get("content") != "ignored"])
        assert e._session_end_occurrence_resume([FINAL, {"role": "user", "content": "ignored"}], **kwargs) == [FINAL]
    row = lane.rows()[0]
    record = dict(version=1, form="form", session_id="S1", role=row["role"], tool_call_id="")
    for field, bad in (("version", 2), ("session_id", "foreign"), ("role", "tool"), ("tool_call_id", "other")):
        with monkeypatch.context() as patch:
            patch.setattr(e._store, "read_metadata_json_many", lambda keys: {keys[0]: dict(record, **{field: bad})})
            assert e._session_end_host_forms([row], store=e._store, session_id="S1") == {}


def test_2000_row_lock_and_transaction_timing(lanes, monkeypatch, record_property):
    lane = lanes()
    first = [{"role": ("user", "assistant")[i % 2], "content": f"row {i}"} for i in range(2000)]
    lane.seed(first)
    view = copy.deepcopy(first)
    view[1]["content"] = "host edit"
    view.append(FINAL)
    e, samples, states, transactions = lane.engine, [], [], []
    helper = e._session_end_occurrence_resume
    diagnostic_commit = e._cold_tail_diagnostic_commit

    @contextmanager
    def measured_commit(*args):
        with diagnostic_commit(*args):
            started = time.perf_counter()
            yield
        transactions.append(time.perf_counter() - started)

    def measured(*args, **kwargs):
        store = kwargs["store"]
        started = time.perf_counter()
        locked, transaction = store._write_lock._is_owned(), store._conn.in_transaction
        result = helper(*args, **kwargs)
        elapsed = time.perf_counter() - started
        samples.append(dict(seconds=elapsed, lock_seconds=elapsed if locked else 0.0,
                            transaction_seconds=elapsed if transaction else 0.0))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(e, "_session_end_occurrence_resume", measured)
        patch.setattr(e, "_cold_tail_diagnostic_commit", measured_commit)
        for name in ("get_range", "read_metadata_json_many", "read_metadata_json"):
            original = getattr(e._store, name)
            def read(*args, _name=name, _original=original, **kwargs):
                states.append((_name, e._store._write_lock._is_owned(), e._store._conn.in_transaction))
                return _original(*args, **kwargs)
            patch.setattr(e._store, name, read)
        lane.end(view, monkeypatch)
        print("2543 direct helper timing:", json.dumps(samples), "read states:", sorted(set(states)))
        assert not any(locked or transaction for _, locked, transaction in states)
        samples.clear()
        states.clear()
        view[1]["content"] = "second edit"
        lane.end(view, monkeypatch, busy=True)
        assert any(name == "read_metadata_json_many" and locked and transaction for name, locked, transaction in states)
    # Captured separately from pytest's duration table; no timing bound is asserted.
    record_property("timing", json.dumps(samples))
    record_property("transaction_seconds", json.dumps(transactions))
    print("2543 cold helper timing:", json.dumps(samples), "read states:", sorted(set(states)),
          "whole retry lock/transaction seconds:", transactions)


_REBIND_COUNTS = {  # (after the end, after the rebind ingest); main and v0.30.0 end at 12 / 22 / 22 / 22 / 23
    "plain": (11, 12), "T6i": (11, 12), "T6ii": (11, 12), "T7-unwrapped": (11, 12),
    "T7-wrapped": (12, 24), "T7K": (12, 24), "edit": (12, 24), "insert": (12, 25)}


@pytest.mark.parametrize("shape", list(_REBIND_COUNTS))
def test_rebind_after_an_off_current_end_adds_only_the_new_turn(lanes, monkeypatch, shape):
    """#2543 review CR-3 and named limit L-6 (RM T1-23). After an off-current end, the session's next bind stores only
    the new prompt wherever v0.30.0 does. Where v0.30.0 itself stores the whole view again at the next bind (T7-wrapped,
    T7K, a host edit, an insert: 10 -> 22 / 23), the rows the end stored stay beside that copy, so FINAL appears twice
    (L-6). Either way a later turn adds one row, the next off-current end adds one, and a repeated end adds nothing."""
    lane = lanes()
    first = history("plain" if shape in ("edit", "insert") else shape)
    lane.seed(first)
    view = copy.deepcopy(first)
    if shape == "edit":
        view[1]["content"] = "host edit"
    elif shape == "insert":
        view.insert(2, {"role": "user", "content": "inserted"})
    view.append(FINAL)
    lane.engine.on_session_end("S1", copy.deepcopy(view))
    after_end = len(lane.rows())
    engine = lane.engine
    engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
    prompt, reply = {"role": "user", "content": "new prompt after bind"}, {"role": "assistant", "content": "reply"}
    engine.ingest(copy.deepcopy(view + [prompt]))
    assert (after_end, len(lane.rows())) == _REBIND_COUNTS[shape]
    finals = sum(r["content"] == "FINAL" for r in lane.rows())
    assert finals == (1 if _REBIND_COUNTS[shape][1] == 12 else 2)
    engine.ingest(copy.deepcopy(view + [prompt, reply]))
    assert len(lane.rows()) == _REBIND_COUNTS[shape][1] + 1
    lane.foreground()
    later = view + [prompt, reply, {"role": "user", "content": "bye"}]
    lane.engine.on_session_end("S1", copy.deepcopy(later))
    assert len(lane.rows()) == _REBIND_COUNTS[shape][1] + 2
    lane.engine.on_session_end("S1", copy.deepcopy(later))
    assert len(lane.rows()) == _REBIND_COUNTS[shape][1] + 2


@pytest.mark.parametrize("busy", [False, True], ids=["direct", "busy"])
def test_auxiliary_reused_session_never_resumes_by_occurrence(lanes, monkeypatch, busy):
    """#2543 review CR-2 (L-4): an auxiliary session reused as a normal one keeps main's answer on both routes; its
    busy tail stays pending as on main."""
    lane = lanes()
    first = history()
    lane.seed(first)
    engine = lane.engine
    engine._auxiliary_foreground_reused_session_ids.add("S1")
    calls = []
    original = engine._session_end_occurrence_resume
    monkeypatch.setattr(engine, "_session_end_occurrence_resume", lambda *a, **k: calls.append(1) or original(*a, **k))
    foreign = [{"role": "user", "content": "Generate a short title for this chat"},
               {"role": "assistant", "content": "Title: pads"}]
    before = lane.rows()
    if busy:
        with monkeypatch.context() as patch:
            patch.setattr(engine._store, "get_range", fail_end_read(engine))
            engine.on_session_end("S1", copy.deepcopy(foreign))
        engine._retry_pending_session_tails(deadline=time.monotonic() + 60)
        assert len(engine._pending_session_tails) == 1
    else:
        engine.on_session_end("S1", copy.deepcopy(foreign))
    assert lane.rows() == before and not calls


def test_a_malformed_commit_proof_keeps_mains_answer(lanes, monkeypatch):
    """#2543 review CR-5: an unreadable compaction commit proof never raises out of the session end; the session gets
    main's answer (nothing new stored), as when a proof exists."""
    lane = lanes()
    first = history("T7-wrapped")
    lane.seed(first)
    engine = lane.engine
    key = engine._replay_snapshot_metadata_key(engine_module._COMPACTION_COMMIT_PROOF_METADATA_PREFIX, "S1")
    with engine._store._write_lock:
        engine._store._conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", (key, "{not json"))
        engine._store._conn.commit()
    before = lane.rows()
    engine.on_session_end("S1", copy.deepcopy(first + [FINAL]))
    assert lane.rows() == before and not lane.forms()
