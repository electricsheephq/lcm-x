"""#7: unstamped replacements before the steady-state cursor (#497/#493) — the acceptance set for a fix.

The loss cases are strict xfails until a fix lands; see PR #742 for the shapes a fix must also keep.

New host objects exercise the prefix audit, not the R6 same-object rewrite path.
The #561 duplicate remains open; its guard proves preservation only.
"""
from __future__ import annotations

from difflib import SequenceMatcher

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION = "s7-anchor-audit"
PAD = " alpha beta gamma delta" * 40


def _engine(tmp_path, **overrides) -> LCMEngine:
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"),
                       large_output_externalization_enabled=False, **overrides)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SESSION, platform="cli", context_length=1_000_000)
    return engine


def _stored(engine) -> list[dict]:
    return [row for row in engine._store.get_session_messages(SESSION) if row.get("role") != "system"]


def _u(text, ts) -> dict:
    return {"role": "user", "content": text} if ts is None else {"role": "user", "content": text, "timestamp": ts}


def _a(text, ts) -> dict:
    return {"role": "assistant", "content": text} if ts is None else {"role": "assistant", "content": text, "timestamp": ts}


def _repair(stamped=False):
    ts = (lambda n: float(n)) if stamped else (lambda n: None)
    host = [_u("U1" + PAD, ts(100)), _a("A1_1" + PAD, ts(101)), _u("U5" + PAD, ts(500)),
            _a("A1" + PAD, ts(501)), _a("A2" + PAD, ts(503))]
    merged = _a(host[3]["content"] + "\n\n" + host[4]["content"], ts(501))
    return host, [*host[:3], merged, _u("U6" + PAD, ts(600)), _a("A6" + PAD, ts(601))]


@pytest.mark.xfail(strict=True, reason="#7: an unstamped user row the host replaced before the ingest cursor is not audited")
def test_497_repair_merge_before_cursor_unstamped_stores_the_new_user_row_once(tmp_path):
    engine = _engine(tmp_path)
    try:
        host, repaired = _repair()
        engine.ingest(host)
        before = len(_stored(engine))
        engine.ingest(repaired)
        texts = [row["content"] for row in _stored(engine)]
        assert texts.count(repaired[4]["content"]) == 1
        assert texts.count(repaired[5]["content"]) == 1
        assert repaired[3]["content"] not in texts
        assert len(texts) == before + 2
    finally:
        engine.shutdown()


@pytest.mark.xfail(strict=True, reason="#7: an unstamped user row the host replaced before the ingest cursor is not audited")
def test_497_repair_merge_without_appended_reply_stores_the_new_user_row(tmp_path):
    engine = _engine(tmp_path)
    try:
        host, repaired = _repair()
        engine.ingest(host)
        before = len(_stored(engine))
        engine.ingest(repaired[:-1])
        texts = [row["content"] for row in _stored(engine)]
        assert texts.count(repaired[4]["content"]) == 1
        assert repaired[3]["content"] not in texts
        assert len(texts) == before + 1
    finally:
        engine.shutdown()


@pytest.mark.xfail(strict=True, reason="#7: an unstamped user row the host replaced before the ingest cursor is not audited")
@pytest.mark.parametrize("reply", [False, True])
@pytest.mark.parametrize("turn_stamps", ["none", "host"])
def test_493_rollback_and_replace_unstamped_stores_the_replacement_once(tmp_path, reply, turn_stamps):
    engine = _engine(tmp_path)
    try:
        ts = (lambda n: float(n)) if turn_stamps == "host" else (lambda n: None)
        host = [_u("U1" + PAD, ts(100)), _a("A1_1" + PAD, ts(101)), _u("U_last" + PAD, None)]
        engine.ingest(host)
        before = len(_stored(engine))
        prime = _u("U_prime" + PAD, None)
        engine.ingest([*host[:2], prime] + ([_a("reply" + PAD, ts(201))] if reply else []))
        texts = [row["content"] for row in _stored(engine)]
        assert texts.count(prime["content"]) == 1
        assert texts.count(host[2]["content"]) == 1
        assert len(texts) == before + 1 + int(reply)
    finally:
        engine.shutdown()


@pytest.mark.xfail(strict=True, reason="#7: an unstamped user row the host replaced before the ingest cursor is not audited")
def test_493_merge_onto_the_last_row_as_a_new_object_stores_the_new_text(tmp_path):
    engine = _engine(tmp_path)
    try:
        host = [_u("U1" + PAD, None), _a("A1_1" + PAD, None), _u("U_last" + PAD, None)]
        engine.ingest(host)
        new = "NEW instruction" + PAD
        composite = _u(host[2]["content"] + "\n\n" + new, None)
        engine.ingest([*host[:2], composite, _a("reply" + PAD, None)])
        texts = [row["content"] for row in _stored(engine)]
        assert sum(new in text for text in texts) == 1
        assert texts.count(composite["content"]) == 1  # #259: whole composite, no remainder splitting
        assert texts.count(host[2]["content"]) == 1
    finally:
        engine.shutdown()


def test_497_stamped_control_stores_the_new_user_row_once(tmp_path, record_property):
    engine = _engine(tmp_path)
    try:
        host, repaired = _repair(stamped=True)
        engine.ingest(host)
        before = len(_stored(engine))
        engine.ingest(repaired)
        texts = [row["content"] for row in _stored(engine)]
        assert texts.count(repaired[4]["content"]) == 1
        assert texts.count(repaired[5]["content"]) == 1
        record_property("duplicates", len(texts) - before - 2)
    finally:
        engine.shutdown()


def test_493_stamped_control_stores_the_replacement_once(tmp_path):
    engine = _engine(tmp_path)
    try:
        host = [_u("U1" + PAD, 100.0), _a("A1_1" + PAD, 101.0), _u("U_last" + PAD, 200.0)]
        engine.ingest(host)
        before = len(_stored(engine))
        prime = _u("U_prime" + PAD, 300.0)
        engine.ingest([*host[:2], prime, _a("reply" + PAD, 301.0)])
        texts = [row["content"] for row in _stored(engine)]
        assert texts.count(prime["content"]) == 1
        assert texts.count(host[2]["content"]) == 1
        assert len(texts) == before + 2
    finally:
        engine.shutdown()


def test_unstamped_merge_of_two_ingested_user_rows_adds_no_row(tmp_path):
    engine = _engine(tmp_path)
    try:
        host = [_u("U1" + PAD, None), _a("A1_1" + PAD, None),
                _u("Ua" + PAD, None), _u("Ub" + PAD, None)]
        engine.ingest(host)
        before = len(_stored(engine))
        assert [row["content"] for row in _stored(engine)].count(host[3]["content"]) == 1
        composite = _u(host[2]["content"] + "\n\n" + host[3]["content"], None)
        engine.ingest([*host[:2], composite])
        texts = [row["content"] for row in _stored(engine)]
        assert len(texts) == before
        assert composite["content"] not in texts
    finally:
        engine.shutdown()


def test_unstamped_stored_user_row_reshown_with_edge_whitespace_is_not_stored_again(tmp_path):
    engine = _engine(tmp_path)
    try:
        host = [_u("U1" + PAD, None), _a("A1_1" + PAD, None),
                _u("Ua" + PAD, None), _a("A_x" + PAD, None)]
        engine.ingest(host)
        before = len(_stored(engine))
        reshown = [*host[:2], _u(host[2]["content"] + "\n", None),
                   _a("rewritten " + host[3]["content"], None)]
        def identities(rows):
            return [engine._message_replay_identity(row, strip_carrier=False) for row in rows]

        assert ("replace", 2, 4, 2, 4) in SequenceMatcher(
            None, identities(host), identities(reshown), autojunk=False).get_opcodes()
        engine.ingest(reshown)
        texts = [row["content"].strip() for row in _stored(engine)]
        assert texts.count(host[2]["content"].strip()) == 1
        assert len(texts) == before
    finally:
        engine.shutdown()


@pytest.fixture
def summaries(monkeypatch):
    captured = []

    def summarize(**kwargs):
        captured.append(kwargs["text"])
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return captured


def test_561_dangling_retained_row_merge_persist_is_not_a_loss(tmp_path, summaries, record_property):
    engine = _engine(tmp_path, fresh_tail_count=2, leaf_chunk_tokens=1)
    head = []
    for turn in range(1, 4):
        head += [_u(f"head U{turn}" + PAD, float(turn * 100)),
                 _a(f"head A{turn}" + PAD, float(turn * 100 + 1))]
    retained = _u("retained R" + PAD, 500.0)
    user = _u("follow-up U" + PAD, 600.0)
    try:
        engine.ingest([*head, retained])
    finally:
        engine.shutdown()
    engine = _engine(tmp_path, fresh_tail_count=2, leaf_chunk_tokens=1)
    try:
        live = engine.compress([*head, retained, user])
        assert engine._last_compression_status == "compacted"
        # R may be glued behind the DAG-verified summary carrier or stand alone.
        assert [engine._message_replay_identity(row)[1] for row in live[-2:]] == [
            retained["content"], user["content"]]
        live = [*live[:-2], _u(user["content"], retained["timestamp"])]
        engine.ingest(live)  # host persist: U alone under the retained row's stamp
        statuses = []
        for turn in range(7, 10):
            live += [_u(f"later U{turn}" + PAD, float(turn * 100)),
                     _a(f"later A{turn}" + PAD, float(turn * 100 + 1))]
            engine.ingest(live)
            live = engine.compress(live)
            statuses.append(engine._last_compression_status)
        texts = [row["content"] for row in _stored(engine)]
        copies = texts.count(user["content"])
        record_property("u_copies", copies)
        record_property("compaction_statuses", statuses)
        assert texts.count(retained["content"]) == 1
        assert copies >= 1
        assert "error" not in statuses
        assert statuses == ["compacted"] * 3
    finally:
        engine.shutdown()


@pytest.mark.xfail(strict=True, reason="#7: LCM_IDENTITY_ANCHOR=0 has no prefix audit")
def test_anchor_off_is_out_of_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "0")
    engine = _engine(tmp_path)
    try:
        host, repaired = _repair()
        engine.ingest(host)
        engine.ingest(repaired)
        assert any(row["content"] == repaired[4]["content"] for row in _stored(engine))
    finally:
        engine.shutdown()


@pytest.mark.xfail(strict=True, reason="#7: a replacement after the host truncated its list is not audited")
def test_truncated_then_replaced_row_stores_the_new_row(tmp_path):
    """Review r4 shape 1: the host truncates its list, then puts a new row where an old one was."""
    engine = _engine(tmp_path)
    try:
        host = [_u("U1" + PAD, None), _a("A1" + PAD, None), _u("U2" + PAD, None), _a("A2" + PAD, None)]
        engine.ingest(host)
        engine.ingest(host[:2])
        new = _u("NEW question" + PAD, None)
        engine.ingest([host[0], new])
        texts = [row["content"] for row in _stored(engine)]
        assert texts.count(new["content"]) == 1
    finally:
        engine.shutdown()


@pytest.mark.xfail(strict=True, reason="#7: a short user row equal to an older row's last paragraph is not audited")
def test_short_reply_matching_an_earlier_paragraph_is_stored(tmp_path):
    """Review r4 shape 2: a new short user row equals the last paragraph of an older stored row."""
    engine = _engine(tmp_path)
    try:
        host = [_u("old paragraph" + PAD + "\n\nyes", None), _a("A1" + PAD, None),
                _u("U2" + PAD, None), _a("A2" + PAD, None)]
        engine.ingest(host)
        replay = [_u("new question" + PAD, None), _a("new answer" + PAD, None),
                  _u("yes", None), _a("new final" + PAD, None)]
        engine.ingest(replay)
        texts = [row["content"] for row in _stored(engine)]
        assert texts.count("yes") == 1
    finally:
        engine.shutdown()
