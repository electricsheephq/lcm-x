"""#535: the host merges a new user row behind the RETAINED last user row of an adopted compaction.

Hermes' ``_merge_consecutive_users`` turns the last output row R into ``R + "\\n\\n" + NEW``. The
composite is stored whole (one new row), R's own row is consumed with it (its bytes are inside), and
the #524 / #457 alignments treat the stored pair (R, composite) as the one row the host replays.
Every negative control stores exactly what the base stores (duplicates, never loss).
"""
from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy

import pytest

import hermes_lcm.engine as lcm_engine_module
import hermes_lcm.reconcile as reconcile_module
from hermes_lcm.engine import LCMEngine
from tests.test_compression_boundary import _config, _host_merge_consecutive_users, _stub_summarizer, _turn
from tests.test_issue_524_gen2_replay import (
    REASON,
    SID,
    _adopt,
    _compress,
    _engine,
    _identities_lost,
    _more,
    _Provider,
    _rows,
    _transcript,
)
from tests.test_reconcile_todo_annotation import _TODO_ANNOTATION_V2

NEW = "NEW-0 typed after the compaction"
ANNOTATIONS = {"plain": "", "annotated": _TODO_ANNOTATION_V2.rstrip("\n")}


class _Counting:
    def __init__(self):
        self.calls, self._stub = 0, _stub_summarizer()

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._stub(*args, **kwargs)


def _base_behaviour(monkeypatch):
    """The base: the #535 predicate never holds (N1, N2 and N3 all read it)."""
    monkeypatch.setattr(reconcile_module, "_merge_append_cut", lambda *args, **kwargs: False, raising=False)
    monkeypatch.setattr(lcm_engine_module, "_merge_append_cut", lambda *args, **kwargs: False, raising=False)


def _all_rows(engine):
    return engine._store._conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()


def _dups(engine):
    return sum(count - 1 for count in Counter(_all_rows(engine)).values() if count > 1)


def _raw_count(engine, text):
    """Lossless bar, raw bytes: an identity check is blind to a cut that drops merged text."""
    return sum(text in (content or "") for _role, content in _all_rows(engine))


class _Session:
    """Compact once with fresh tail ``tail``; the host adopts the output (in place or in a rotation child)."""

    def __init__(self, tmp_path, monkeypatch, *, tail=6, mode="inplace", reply=False):
        self.summarizer = _Counting()
        monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", self.summarizer)
        self.tmp_path, self.tail = tmp_path, tail
        self.child = "S0" if mode == "inplace" else "S1"
        self.engine, self.host = self.make("S0"), []
        for i in range(1, 13):
            self.add(_turn(i)[0])
            self.add(_turn(i)[1])
        self.add(_turn(13)[0])
        if reply:
            self.add(_turn(13)[1])
        pre = list(self.host)
        out = self.engine.compress(list(self.host), force=True)
        assert self.engine._last_compression_status == "compacted"
        self.engine.on_session_end("S0", pre)
        self.engine.on_session_start(self.child, boundary_reason="compression", old_session_id="S0", platform="acp")
        self.host[:] = [dict(m) for m in out]

    def make(self, session_id):
        engine = LCMEngine(config=_config(self.tmp_path, fresh_tail_count=self.tail), hermes_home=str(self.tmp_path / "home"))
        engine.on_session_start(session_id, platform="acp", context_length=200_000)
        return engine

    def add(self, message):
        self.host.append(dict(message))
        self.host[:] = _host_merge_consecutive_users(self.host)
        self.engine.ingest(self.host)

    def turns(self, first, n=5):
        for i in range(first, first + n):
            self.add(_turn(i)[0])
            self.add(_turn(i)[1])
        self.add(_turn(first + n)[0])

    def compact(self, adopt=True):
        out = self.engine.compress(list(self.host), force=True)
        if adopt and self.engine._last_compression_status == "compacted":
            self.host[:] = [dict(m) for m in out]
        return self.engine._last_compression_status, _rows(out)


@pytest.mark.parametrize("annotation", sorted(ANNOTATIONS))
@pytest.mark.parametrize("mode", ["inplace", "rotation"])
@pytest.mark.parametrize("tail", [0, 1, 6])
def test_merge_behind_the_retained_row_is_stored_once_and_compacts_again(tmp_path, monkeypatch, tail, mode, annotation):
    """Tests 1-3: tail 6 ends the output in a retained user row (RED on the base: 5 duplicates, then
    every later compaction errors); tail 1 is #499's carrier extension (RED: errors); tail 0 is #499."""
    s = _Session(tmp_path, monkeypatch, tail=tail, mode=mode)
    try:
        if annotation and s.host[-1]["role"] == "user":
            s.host[-1]["content"] += ANNOTATIONS[annotation]  # the host's todo fold, then its merge
        before = len(_all_rows(s.engine))
        s.add({"role": "user", "content": NEW})
        rows = _all_rows(s.engine)
        assert len(rows) == before + 1 and rows[-1] == ("user", s.host[-1]["content"])  # the composite, whole
        assert NEW in s.host[-1]["content"] and _dups(s.engine) == 0
        s.add({"role": "assistant", "content": "reply to NEW-0"})
        s.turns(40)
        attempt2 = s.compact(adopt=not tail)  # tail 0: #499's emitted-head merge, whose retry is not #535
        assert attempt2[0] == "compacted", s.engine._last_compression_noop_reason
        if tail:  # test 4: the host cancelled attempt 2 after its commit and retries its list (#524)
            rows, calls = len(_all_rows(s.engine)), s.summarizer.calls
            assert s.compact() == attempt2 and s.summarizer.calls == calls and len(_all_rows(s.engine)) == rows
            # A rotation child stores the tail-1 composite as its first row: the durable tail proves it first.
            assert s.engine._last_ingest_reconciliation["reason"] == (
                "replayed durable tail" if (tail, mode) == (1, "rotation") else REASON
            )
        s.turns(50)
        assert s.compact()[0] == "compacted", s.engine._last_compression_noop_reason
        assert _dups(s.engine) == 0 and _raw_count(s.engine, NEW) == 1
    finally:
        s.engine.shutdown()


@pytest.mark.parametrize("cut", ["pre-516", "current"])
def test_lossless_bar_positive_control(tmp_path, monkeypatch, cut):
    """The raw-count bar can see a loss: with the pre-#516 identity cut (everything after the todo
    prefix) the merged text is dropped and the raw count goes red; with the current cut it is 1."""
    if cut == "pre-516":
        monkeypatch.setattr(reconcile_module, "_todo_annotation_span", lambda content, start: len(content))
    s = _Session(tmp_path, monkeypatch)
    try:
        s.host[-1]["content"] += ANNOTATIONS["annotated"]
        s.add({"role": "user", "content": NEW})
        assert _raw_count(s.engine, NEW) == (0 if cut == "pre-516" else 1)
    finally:
        s.engine.shutdown()


@pytest.mark.parametrize("mode", ["inplace", "rotation"])
def test_restart_before_the_merge_binds_by_the_durable_proof(tmp_path, monkeypatch, mode):
    """The durable walk (digests only) probes the ``\\n\\n`` cuts of the last output position."""
    s = _Session(tmp_path, monkeypatch, mode=mode)
    try:
        s.engine.ingest(list(s.host))
        before = len(_all_rows(s.engine))
        s.engine.shutdown()
        s.engine = s.make(s.child)
        s.add({"role": "user", "content": NEW})
        assert s.engine._last_ingest_reconciliation["reason"].startswith("replayed proven post-compaction continuation")
        assert len(_all_rows(s.engine)) == before + 1 and _dups(s.engine) == 0
        s.add({"role": "assistant", "content": "reply to NEW-0"})
        s.turns(40)
        assert s.compact()[0] == "compacted", s.engine._last_compression_noop_reason
    finally:
        s.engine.shutdown()


def test_legacy_v3_proof_restart_stores_the_composite_once(tmp_path, monkeypatch):
    """Hardening (v): under a v3 proof (no emissions, the rc4 ``occurrences is None`` path)."""
    s = _Session(tmp_path, monkeypatch)
    try:
        s.engine.ingest(list(s.host))
        key = "compaction_commit_proof:S0"
        payload = s.engine._store.read_metadata_json(key)
        payload["version"] = 3
        payload.pop("emissions", None)
        s.engine._store.write_metadata_json([key], json.dumps(payload, sort_keys=True))
        before = len(_all_rows(s.engine))
        s.engine.shutdown()
        s.engine = s.make("S0")
        assert s.engine._active_emission_proof()["version"] == 3
        s.add({"role": "user", "content": NEW})
        assert len(_all_rows(s.engine)) == before + 1 and _dups(s.engine) == 0
        s.add({"role": "assistant", "content": "reply to NEW-0"})
        s.turns(40)
        assert s.compact()[0] == "compacted", s.engine._last_compression_noop_reason
        s.engine.shutdown()
        s.engine = s.make("S0")
        s.engine.ingest(list(s.host))  # a later restart re-stores nothing
        assert _dups(s.engine) == 0 and _raw_count(s.engine, NEW) == 1
    finally:
        s.engine.shutdown()


# -- the #524 gen-2 retry in the transcript harness (the eva [pure] shape without a real host) --------


@pytest.mark.parametrize("cold", [False, True], ids=["same-engine", "restart-before-retry"])
def test_gen2_retry_after_a_merge_behind_the_retained_row(tmp_path, monkeypatch, cold):
    provider = _Provider()
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", provider)
    engine = _engine(tmp_path)
    try:
        first = _transcript()
        first += [] if first[-1]["role"] == "user" else [{"role": "user", "content": "the retained last user row"}]
        engine.ingest(first)
        out1 = deepcopy(_compress(engine, first))
        assert out1[-1]["role"] == "user"
        _adopt(engine, first)
        host = _host_merge_consecutive_users(deepcopy(out1) + _more(0, 120))
        assert len(host) == len(out1) + 119 and host[len(out1) - 1]["content"].endswith("\n\n" + _more(0, 1)[0]["content"])
        before = engine._store.get_session_count(SID)
        engine.ingest(host)
        assert engine._store.get_session_count(SID) == before + 1 + 119 and _dups(engine) == 0  # the composite, then 119
        out2 = _rows(_compress(engine, host))
        assert engine._last_compression_status == "compacted"
        rows, calls = engine._store.get_session_count(SID), provider.calls
        if cold:
            engine.shutdown()
            engine = _engine(tmp_path)
        retry = _rows(_compress(engine, deepcopy(host)))
        assert engine._last_ingest_reconciliation["reason"] == REASON
        assert (engine._last_compression_status, provider.calls - calls) == ("compacted", 0)
        assert engine._store.get_session_count(SID) == rows and retry == out2
        assert _identities_lost(engine, host) == [] and _raw_count(engine, _more(0, 1)[0]["content"]) == 1
    finally:
        engine.shutdown()


# -- negative controls: the predicate must not fire; each stores what the base stores ---------------


def _suffix(text):
    def mutate(host):
        host[-1] = {**host[-1], "content": host[-1]["content"] + text}
        return host
    return mutate


def _non_last(host):
    host[-3] = {**host[-3], "content": host[-3]["content"] + "\n\n" + NEW}  # a retained user row, not the last
    return host


def _edited_prefix(host):
    host[2] = {**host[2], "content": host[2]["content"] + " (edited by host)"}
    return _suffix("\n\n" + NEW)(host)


def _tool_row(host):
    host[-1] = {**host[-1], "content": host[-1]["content"] + "\n\n" + NEW, "tool_call_id": "call_535"}
    return host


def _unmerged(host):  # a host that does not merge: R, then R + NEW as its own row (B stays mapped)
    return host + [{"role": "user", "content": host[-1]["content"] + "\n\n" + NEW}]


def _replaced(host):  # Hermes' persist step replaced the merged last row in place: NEW alone (#519's case)
    host[-1] = {"role": "user", "content": NEW}
    return host


NEGATIVE = {
    "non_last_position": (_non_last, False),
    "assistant_last_row": (_suffix("\n\n" + NEW), True),
    "whitespace_suffix": (_suffix("\n\n  \n "), False),
    "single_newline_joiner": (_suffix("\n" + NEW), False),
    "empty_suffix": (_suffix("\n\n"), False),
    "tool_call_id": (_tool_row, False),
    "edited_prefix_row": (_edited_prefix, False),
    "unmerged_base_is_mapped": (_unmerged, False),
    "replaced_last_row": (_replaced, False),
}


def _negative_run(tmp_path, monkeypatch, case, *, fixed):
    if not fixed:
        _base_behaviour(monkeypatch)
    mutate, reply = NEGATIVE[case]
    s = _Session(tmp_path, monkeypatch, reply=reply)
    try:
        assert s.host[-1]["role"] == ("assistant" if reply else "user")
        s.host[:] = mutate(deepcopy(s.host))
        s.engine.ingest(list(s.host))
        stored = _all_rows(s.engine)
        s.add({"role": "assistant", "content": "a reply"} if s.host[-1]["role"] == "user" else _turn(39)[0])
        s.turns(40)
        return stored, s.compact(), _all_rows(s.engine), s.summarizer.calls
    finally:
        s.engine.shutdown()


@pytest.mark.parametrize("case", sorted(NEGATIVE))
def test_negative_controls_store_what_the_base_stores(tmp_path, monkeypatch, case):
    fixed = _negative_run(tmp_path / "fixed", monkeypatch, case, fixed=True)
    monkeypatch.undo()
    base = _negative_run(tmp_path / "base", monkeypatch, case, fixed=False)
    assert fixed == base


def test_n2_adds_only_an_unmapped_base_above_the_frontier(tmp_path, monkeypatch):
    s = _Session(tmp_path, monkeypatch)
    try:
        s.add({"role": "user", "content": NEW})
        (base, _), (composite, _) = s.engine._store._conn.execute(
            "SELECT store_id, content FROM messages ORDER BY store_id DESC LIMIT 2"
        ).fetchall()[::-1]
        assert s.engine._with_merge_append_bases([composite], set()) == [base, composite]
        assert s.engine._with_merge_append_bases([composite], {base}) == [composite]  # B mapped by the list
        assert s.engine._with_merge_append_bases([base, composite], set()) == [base, composite]  # already consumed
        assert s.engine._with_merge_append_bases([base], set()) == [base]  # a row with no base behind it
        s.engine._last_compacted_store_id = base
        assert s.engine._with_merge_append_bases([composite], set()) == [composite]  # B <= F
    finally:
        s.engine.shutdown()


def test_n2_skips_a_base_the_publication_does_not_own(tmp_path, monkeypatch):
    """N2 inserts only a row the publication owns (this session, or a proven carry range of the
    same conversation): an unowned id would raise at the publication's ownership check."""
    s = _Session(tmp_path, monkeypatch, mode="rotation")
    try:
        s.add({"role": "user", "content": NEW})
        (base, base_sid), (composite, child) = s.engine._store._conn.execute(
            "SELECT store_id, session_id FROM messages ORDER BY store_id DESC LIMIT 2"
        ).fetchall()[::-1]
        assert (base_sid, child) == ("S0", "S1")  # B is the parent's retained row, carried into the child
        assert s.engine._with_merge_append_bases([composite], set()) == [base, composite]
        s.engine._store._conn.execute("UPDATE messages SET conversation_id = 'another-conversation' WHERE store_id = ?", (base,))
        assert s.engine._with_merge_append_bases([composite], set()) == [composite]
        s.engine._store._conn.execute("UPDATE messages SET conversation_id = NULL WHERE store_id = ?", (base,))
        monkeypatch.setattr(LCMEngine, "_load_compression_carry_ranges", lambda self, session_id=None: [])
        assert s.engine._with_merge_append_bases([composite], set()) == [composite]  # no proven carry range
    finally:
        s.engine.shutdown()


def _persisted(tmp_path, monkeypatch, mode):
    """Hermes' persist step: the model saw ``R + "\\n\\n" + NEW`` (ingested), then the host replaced
    that last carried row in place, so the next ingest sees ``NEW`` alone at that position."""
    s = _Session(tmp_path, monkeypatch, mode=mode)
    before = len(_all_rows(s.engine))
    s.engine.ingest([*map(dict, s.host[:-1]), {"role": "user", "content": s.host[-1]["content"] + "\n\n" + NEW}])
    s.host[-1] = {"role": "user", "content": NEW}
    s.add({"role": "assistant", "content": "reply to NEW-0"})
    return s, before


@pytest.mark.parametrize("mode", ["inplace", "rotation"])
def test_persist_step_after_the_merge_stores_nothing_twice(tmp_path, monkeypatch, mode):
    s, before = _persisted(tmp_path, monkeypatch, mode)
    try:  # the composite (whole, once) and the reply; NEW is not stored again beside it
        assert len(_all_rows(s.engine)) == before + 2 and _dups(s.engine) == 0 and _raw_count(s.engine, NEW) == 1
    finally:
        s.engine.shutdown()


@pytest.mark.xfail(strict=True, reason="#519: after the persist step the replaced last row (NEW alone) is not "
                   "mapped to the stored composite, so the next compaction cannot prove contiguous coverage")
@pytest.mark.parametrize("mode", ["inplace", "rotation"])
def test_persist_step_after_the_merge_compacts_again(tmp_path, monkeypatch, mode):
    s, _before = _persisted(tmp_path, monkeypatch, mode)
    try:
        s.turns(40)
        assert s.compact()[0] == "compacted", s.engine._last_compression_noop_reason
    finally:
        s.engine.shutdown()


def _pair_run(tmp_path, monkeypatch, *, fixed):
    """N3 raw-first: a non-merging host stored [X, X + "\\n\\n" + Y] and replays both; today's bind holds."""
    if not fixed:
        _base_behaviour(monkeypatch)
    provider = _Provider()
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", provider)
    engine = _engine(tmp_path)
    try:
        first = _transcript()
        engine.ingest(first)
        out1 = deepcopy(_compress(engine, first))
        _adopt(engine, first)
        more = _more(0, 120)
        x = more[100]["content"]
        host = deepcopy(out1) + more[:101] + [{"role": "user", "content": x + "\n\nY typed right behind it"}] + more[101:]
        engine.ingest(host)
        _compress(engine, host)
        calls = provider.calls
        retry = _rows(_compress(engine, deepcopy(host)))
        return (engine._store.get_session_count(SID), engine._last_compression_status, provider.calls - calls,
                engine._last_ingest_reconciliation["reason"], retry)
    finally:
        engine.shutdown()


def test_n3_raw_rows_first_keep_the_bind(tmp_path, monkeypatch):
    fixed = _pair_run(tmp_path / "fixed", monkeypatch, fixed=True)
    monkeypatch.undo()
    base = _pair_run(tmp_path / "base", monkeypatch, fixed=False)
    assert fixed == base and fixed[1:4] == ("compacted", 0, REASON)
