"""#659 round 5: a rotated head is normalised only on positive evidence that it is the head LCM emitted.

Ported from the fourth cross-model review's probes (R4-N1): a pending compression boundary alone is not occurrence
provenance. A lone user turn that quotes, or equals, the emitted context is stored whole; the emitted head re-sent
inside its window still stores in base shape (F1).
"""

import hashlib

import pytest

import hermes_lcm.engine as engine_module
import tests.test_issue_659_user_carry_packet as carry_packet
from tests.test_issue_659_carry_r4 import BASE_HEADS, _generated
from tests.test_issue_659_user_carry_packet import CARRY, SEP, _compact, _history, _host_merge, _tools

make = carry_packet.make  # the shared fixture

PREFIX = "Please inspect this copied context:" + SEP
PRE = [*_history(8), {"role": "user", "content": "latest"}, {"role": "assistant", "content": "ok"}]


def _rotate(engine, hops=1):
    for i in range(1, hops + 1):
        engine.on_session_start(f"P{i}", platform="cli", context_length=128000, conversation_id="conv",
                                boundary_reason="compression", old_session_id=f"P{i - 1}")
    assert engine._compression_boundary_ingest_pending


def _stored(engine, text):
    return [r["content"] for r in engine._store.get_session_messages(engine._session_id)
            if r["content"].encode("utf-8") == text.encode("utf-8")]


@pytest.mark.parametrize("form,size", [("prefixed", 913), ("exact", 871)])
def test_new_only_gateway_quote_of_the_emitted_context_is_stored_whole(make, form, size):
    engine = make(session="P0")
    _out, pure = _generated(engine)
    engine.on_session_end("P0", PRE)
    _rotate(engine)
    text = PREFIX + pure if form == "prefixed" else pure
    assert len(text.encode("utf-8")) == size
    engine.ingest([{"role": "user", "content": text}])
    assert _stored(engine, text) == [text]


@pytest.mark.parametrize("turn", ["answered", "tool-loop"])
def test_new_only_gateway_exact_copy_with_its_own_turn_is_stored_whole(make, turn):
    """The gateway's post-call batch carries the new turn's own rows, not LCM's emitted window."""
    engine = make(session="P0")
    _out, pure = _generated(engine)
    engine.on_session_end("P0", PRE)
    _rotate(engine)
    rows = [{"role": "assistant", "content": "answer"}] if turn == "answered" else [*_tools(500, 5),
                                                                                   {"role": "assistant", "content": "done"}]
    engine.ingest([{"role": "user", "content": pure}, *rows])
    assert _stored(engine, pure) == [pure]


def test_first_row_manifest_quote_is_stored_whole(make):
    engine = make(session="P0")
    _out, pure = _generated(engine)
    base = pure.split(SEP + CARRY)[0]
    node = engine._dag.get_node(int(engine._LCM_SUMMARY_PART_HEADER_RE.match(base).group(2)))
    text = "Please inspect this omitted node:" + SEP + base + SEP + engine_module._render_omitted_summaries([node], 0)
    assert len(text.encode("utf-8")) == 320
    _rotate(engine)
    engine.ingest([{"role": "user", "content": text}])
    assert _stored(engine, text) == [text]


def test_a_quote_behind_a_leading_system_row_is_stored_whole(make):
    engine = make(session="P0")
    _out, pure = _generated(engine)
    _rotate(engine)
    engine.ingest([{"role": "system", "content": "FIRST ROW"}, {"role": "user", "content": pure}])
    assert _stored(engine, pure) == [pure]


def test_public_empty_ingest_neither_spends_nor_misapplies_the_evidence(make):
    engine = make(session="P0")
    _out, pure = _generated(engine)
    _rotate(engine)
    engine.ingest([])
    engine.ingest([{"role": "user", "content": PREFIX + pure}])
    assert _stored(engine, PREFIX + pure) == [PREFIX + pure]


@pytest.mark.parametrize("stage", ["protect", "append", "after-append"])
def test_a_quote_after_a_failed_first_ingest_is_stored_whole(make, monkeypatch, stage):
    engine = make(session="P0")
    _out, pure = _generated(engine)
    _rotate(engine)
    target, attribute = ((engine_module, "protect_messages_for_ingest") if stage == "protect"
                         else (engine._store, "_append_protected_batch") if stage == "append"
                         else (engine, "_watch_stored_user_rows"))
    original = getattr(target, attribute)

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic ingest failure")

    monkeypatch.setattr(target, attribute, fail)
    engine.ingest([{"role": "user", "content": "FIRST ATTEMPT"}])
    monkeypatch.setattr(target, attribute, original)
    engine.ingest([{"role": "user", "content": PREFIX + pure}])
    assert _stored(engine, PREFIX + pure) == [PREFIX + pure]


# -- controls: green before and after the fix --------------------------------------------------------------

def test_host_merged_emitted_window_stores_the_new_turn_and_no_carry(make):
    engine = make(session="P0")
    pre = [*_history(8), {"role": "user", "content": "TAIL-USER: one more thing."},
           *_tools(90, 50), {"role": "assistant", "content": "tail reply"}]
    out = _compact(engine, pre)
    engine.on_session_end("P0", pre)
    _rotate(engine)
    engine.ingest(_host_merge([*out, {"role": "user", "content": "NEW MERGED WINDOW REQUEST"}]))
    rows = [r["content"] for r in engine._store.get_session_messages("P1")]
    assert "NEW MERGED WINDOW REQUEST" in rows and not any(CARRY in r for r in rows)
    assert not engine._compression_boundary_ingest_pending


@pytest.mark.parametrize("empty_first", [False, True])
def test_emitted_window_then_new_turn_stores_the_base_head(make, empty_first):
    engine = make(session="P0")
    out, pure = _generated(engine)
    assert out[0]["content"] == pure
    _rotate(engine)
    if empty_first:
        engine.ingest([])
    engine.ingest([*out, {"role": "user", "content": "NEW REQUEST"}])
    rows = [r["content"] for r in engine._store.get_session_messages("P1")]
    heads = [r.encode("utf-8") for r in rows if r.startswith("[Recent Summary")]
    assert [(hashlib.sha256(h).hexdigest(), len(h)) for h in heads] == [BASE_HEADS["summary"]]
    assert "NEW REQUEST" in rows and not any(CARRY in r for r in rows)
