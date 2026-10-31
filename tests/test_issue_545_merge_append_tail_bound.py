"""#545: after a restart the durable merge-append cut probe aligns a retained last user row that
itself carries 64+ paragraph breaks (the probe now also walks back from the row end)."""
from __future__ import annotations

import pytest

import tests.test_issue_535_retained_row_merge as M
from hermes_lcm.reconcile import _merge_append_cut, _proof_user_identity

NEW = "NEW-545 typed after the restart"
REPLAYED = "replayed proven post-compaction continuation"


def _paragraphs(n, tag="para"):
    return "".join(f"\n\n{tag} {k}" for k in range(n))


def _user(text):
    return ("user", text, "", "", "")


def _base_cut(identity, is_base, start=0):
    """Frozen copy of the pre-#545 predicate (the first 64 cuts at or after ``start``)."""
    cut = start - 1 if identity[0] == "user" and tuple(identity[2:]) == ("", "", "") else None
    for _ in range(64 if cut is not None else 0):
        cut = identity[1].find("\n\n", cut + 1)
        if cut < 0:
            return False
        head = (identity[0], identity[1][:cut], *identity[2:])
        if head[1].strip() and identity[1][cut + 2:].strip() and is_base(head):
            return True
    return False


@pytest.mark.parametrize("breaks", [64, 65, 200])
def test_long_base_row_aligns_from_the_row_end(breaks):
    base = "R" + _paragraphs(breaks)
    assert _merge_append_cut(_user(base + "\n\n" + NEW), lambda head: head[1] == base)


def test_probe_work_is_bounded_and_each_cut_is_probed_once():
    probed = []
    row = "R" + _paragraphs(300) + "\n\n" + "N" + _paragraphs(300)
    assert not _merge_append_cut(_user(row), lambda head: probed.append(len(head[1])) or False)
    assert 64 < len(probed) <= 128 and len(set(probed)) == len(probed)


@pytest.mark.parametrize("trail", ["", " ", "\n", "\n\n", " \n\n"])
@pytest.mark.parametrize("new_breaks", [0, 10, 70])
@pytest.mark.parametrize("base_breaks", [0, 10, 70])
def test_start_anchored_callers_see_the_base_predicate(base_breaks, new_breaks, trail):
    """The in-process walk (engine.py) and ``_merged_pair_row`` pass ``start``: unchanged results."""
    base = "R" + _paragraphs(base_breaks) + trail
    target = _proof_user_identity(_user(base))
    bases = (_user(base), _user(base.rstrip()))
    for row in (base + "\n\n" + NEW + _paragraphs(new_breaks), "X" + base + "\n\n" + NEW, base + "\n\n  "):
        stripped = _proof_user_identity(_user(row))
        in_process = lambda head: _proof_user_identity(head) == target  # noqa: E731
        assert _merge_append_cut(stripped, in_process, len(target[1])) == _base_cut(stripped, in_process, len(target[1]))
        pair = lambda prefix: prefix in bases  # noqa: E731
        start = len(base.rstrip())
        assert _merge_append_cut(_user(row), pair, start) == _base_cut(_user(row), pair, start)


def _session(tmp_path, monkeypatch, breaks, mode, stamped):
    original = M._turn

    def turn(i):
        user, assistant = original(i)
        if i == 13:  # the retained last user row of the compaction output
            user = {**user, "content": user["content"] + _paragraphs(breaks)}
        if stamped:
            user, assistant = {**user, "timestamp": 1.7e9 + 10 * i}, {**assistant, "timestamp": 1.7e9 + 10 * i + 1}
        return user, assistant

    monkeypatch.setattr(M, "_turn", turn)
    return M._Session(tmp_path, monkeypatch, tail=6, mode=mode)


def _stamp(stamped, offset):
    return {"timestamp": 1.7e9 + offset} if stamped else {}


@pytest.mark.parametrize("stamped", [False, True], ids=["unstamped", "stamped"])
@pytest.mark.parametrize("mode", ["inplace", "rotation"])
@pytest.mark.parametrize("breaks", [63, 64, 65, 200])
def test_restart_merge_behind_a_long_retained_row_is_stored_once(tmp_path, monkeypatch, breaks, mode, stamped):
    s = _session(tmp_path, monkeypatch, breaks, mode, stamped)
    try:
        assert s.host[-1]["role"] == "user" and s.host[-1]["content"].count("\n\n") >= breaks
        s.engine.shutdown()
        s.engine = s.make(s.child)
        before = len(M._all_rows(s.engine))
        s.add({"role": "user", "content": NEW, **_stamp(stamped, 999)})
        assert s.engine._last_ingest_reconciliation["reason"].startswith(REPLAYED)
        rows = M._all_rows(s.engine)
        assert len(rows) == before + 1 and M._dups(s.engine) == 0 and M._raw_count(s.engine, NEW) == 1
        if not stamped:  # unstamped: the composite, whole (#535); a stamped merge stores its remainder (identity anchor)
            assert rows[-1] == ("user", s.host[-1]["content"])
        s.add({"role": "assistant", "content": "reply to NEW-545", **_stamp(stamped, 1000)})
        for k in range(3):
            s.turns(40 + 10 * k)
            assert s.compact()[0] == "compacted", s.engine._last_compression_noop_reason
        assert M._dups(s.engine) == 0 and M._raw_count(s.engine, NEW) == 1
    finally:
        s.engine.shutdown()


def test_in_process_merge_behind_a_long_retained_row_is_unchanged(tmp_path, monkeypatch):
    s = _session(tmp_path, monkeypatch, 200, "inplace", False)
    try:
        before = len(M._all_rows(s.engine))
        s.add({"role": "user", "content": NEW})
        rows = M._all_rows(s.engine)
        assert len(rows) == before + 1 and rows[-1] == ("user", s.host[-1]["content"]) and M._dups(s.engine) == 0
    finally:
        s.engine.shutdown()
