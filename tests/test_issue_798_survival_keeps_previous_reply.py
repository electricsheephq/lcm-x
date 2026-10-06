"""#798: only leftover survival budget keeps a replay-safe completed reply."""

import hashlib
import json

import pytest

import hermes_lcm.host_uid_emit as emit
import hermes_lcm.survival_fit as survival_fit
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from tests.test_host_uid_shadow import _state_db


@pytest.fixture
def rough_counts(monkeypatch):
    """Use #916's deterministic chars / 4 harness, without a provider or model call."""
    def count(message):
        return len(str(message.get("content") or "")) // 4

    def measure(messages):
        return sum(count(message) for message in messages)

    monkeypatch.setattr(survival_fit, "count_message_tokens", count)
    monkeypatch.setattr(survival_fit, "count_messages_tokens", measure)
    monkeypatch.setattr(survival_fit, "_host_estimate", measure)
    return measure


def _engine(tmp_path):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), survival_reserve=0.0))
    engine._hermes_home = str(tmp_path)
    engine.on_session_start("S", platform="telegram", conversation_id="conv", context_length=20_000)
    return engine


@pytest.fixture
def engine(tmp_path):
    obj = _engine(tmp_path)
    yield obj
    obj.shutdown()


def _view(reply="[A1] completed reply", *, stamped=True, newest="[U2] newest request"):
    previous = {"role": "assistant", "content": reply}
    if stamped:
        previous["timestamp"] = 2.0
    return [{"role": "system", "content": "system prompt"},
            {"role": "user", "content": "[U1] " + "old " * 2000, "timestamp": 1.0},
            previous, {"role": "user", "content": newest, "timestamp": 3.0}]


def _rows(engine):
    return [row for row in engine._store.get_session_messages("S") if row["role"] != "system"]


def _fit(engine, view, budget):
    engine.ingest(view)
    return engine._survival_fit(view, view, 0, "issue_798", request_cap=budget)


def _base_without_reply(view, ids):
    """Exact base shape when only the newest whole turn fits (including notice bytes)."""
    notice = survival_fit._NOTICE.format(n=2, first=ids[0], last=ids[1])
    return [{**view[0], "content": survival_fit.SurvivalFitMixin._survival_with_notice(
        view[0]["content"], notice)}, view[-1]]


def test_1_verbatim_reply_object_and_notice_excludes_reply(engine, rough_counts):
    view = _view()
    fitted = _fit(engine, view, 100)
    assert fitted[1] is view[2] and fitted[2] is view[3]
    ids = [row["store_id"] for row in _rows(engine)]
    assert engine._last_survival_fit["dropped_rows"] == 1
    assert engine._last_survival_fit["notice"] == survival_fit._NOTICE.format(n=1, first=ids[0], last=ids[0])
    assert rough_counts(fitted) <= 100


def test_2_head_tail_reply_recognizes_source_and_adds_only_new_rows(tmp_path, rough_counts):
    """Also covers amendment 2 test 12: stamped head/tail replay on a new engine."""
    engine = _engine(tmp_path)
    try:
        view = _view("[A1] opening " + "long " * 2000 + " ending")
        fitted = _fit(engine, view, 600)
        reply = fitted[1]
        assert reply is not view[2] and reply["content"].startswith(view[2]["content"][:1200])
        assert reply["content"].endswith(view[2]["content"][-600:])
        source = engine._survival_projection_source(reply, "assistant", reply["content"])
        before = _rows(engine)
        assert source is not None and source["store_id"] == before[1]["store_id"]
        assert engine._last_survival_fit["dropped_rows"] == 1
        counter = engine._store.read_metadata_json(survival_fit.SURVIVAL_FIT_COUNTER_KEY)
        assert counter["projected_count"] == 1  # only the reply was projected in this cut.
    finally:
        engine.shutdown()
    new = [{"role": "assistant", "content": "[A2] answer", "timestamp": 4.0},
           {"role": "user", "content": "[U3] next", "timestamp": 5.0}]
    cold = _engine(tmp_path)
    try:
        cold.ingest(json.loads(json.dumps([*fitted, *new])))
        after = _rows(cold)
        assert after[:-2] == before and [row["content"] for row in after[-2:]] == [m["content"] for m in new]
        assert sum(row["content"] == view[2]["content"] for row in after) == 1
    finally:
        cold.shutdown()


@pytest.mark.parametrize("length,budget", [(1201, 100), (2400, 100), (8000, 60)])
def test_3_middle_band_and_no_room_keep_base_bytes(engine, rough_counts, length, budget):
    view = _view("a" * length)
    fitted = _fit(engine, view, budget)
    ids = [row["store_id"] for row in _rows(engine)]
    assert json.dumps(fitted, sort_keys=True) == json.dumps(_base_without_reply(view, ids), sort_keys=True)
    assert not any(row["role"] == "assistant" for row in fitted)


@pytest.mark.parametrize("case", ["tool-result", "tool-call", "empty", "scaffold"])
def test_4_ineligible_rows_never_keep_an_orphan(engine, rough_counts, case):
    view = _view()
    call = {"id": "call", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    if case in ("tool-result", "tool-call"):
        view[2].update(content="", tool_calls=[call])
        if case == "tool-result":
            view.insert(3, {"role": "tool", "content": "done", "tool_call_id": "call"})
    elif case == "empty":
        view[2]["content"] = "  "
    else:
        view[2]["content"] = "CONTEXT SUMMARY: earlier stored rows"
        assert engine._survival_generated(view[2])
    fitted = _fit(engine, view, 100)
    assert fitted[-1] is view[-1]
    assert [row["role"] for row in fitted] == ["system", "user"]


def test_5_reply_does_not_change_newest_user_projection(engine, rough_counts):
    view = _view(newest="[U2] opening " + "word " * 2000)
    fitted = _fit(engine, view, 650)
    without = [view[0], view[1], view[3]]
    baseline = engine._survival_fit(without, without, 0, "issue_798", request_cap=650)
    assert fitted[1] is view[2]  # the reply uses only space left after the newest projection.
    assert fitted[-1] == baseline[-1] and fitted[-1] is not view[-1]
    assert fitted[-1]["content"].startswith(view[-1]["content"][:1200])
    assert rough_counts(fitted) <= 650


@pytest.mark.parametrize("kind", ["summary", "carrier"])
def test_6_prefix_first_and_summary_uid_shadow(engine, tmp_path, rough_counts, monkeypatch, kind):
    monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    monkeypatch.setattr(emit, "_host_uid_capability", True)
    _state_db(tmp_path, [("S", None, None)])
    view = _view()
    engine.ingest(view)
    ids = [row["store_id"] for row in _rows(engine)]
    node = engine._dag.add_node(SummaryNode(session_id="S", summary="s", token_count=1,
                                            source_token_count=1, source_ids=[ids[0]], expand_hint="turns"))
    summary = f"[Recent Summary (d0, node {node})]\ns\n[Expand for details: turns]"
    prefix = {"role": "user", "content": summary}
    if kind == "carrier":
        prefix["content"] += "\n\n" + view[1]["content"]
        active = [prefix, view[2], view[3]]
        assert engine._generated_context_carrier_remainder(prefix) == view[1]["content"]
        uid_kind = "survival_summary"
    else:
        engine._mint_engine_uids([(prefix, "summary", None, "summary")])
        engine._host_uid_record_engine([prefix])
        active = [prefix, *view[1:]]
        uid_kind = "summary"
    assert engine._is_verified_replay_scaffold_message(prefix) or kind == "carrier"
    fitted = engine._survival_fit(active, active, 0, "issue_798", request_cap=100)
    assert [row["role"] for row in fitted] == ["user", "assistant", "user"]
    assert fitted[0]["content"] == summary and fitted[1] is view[2] and fitted[2] is view[3]
    lineage = engine._host_uid_lineage_key()[0]
    expected = emit.engine_uid(lineage, uid_kind, hashlib.sha256(summary.encode()).hexdigest(), 0)
    assert fitted[0]["message_uid"] == expected
    engine._host_uid_record_engine(fitted)
    records = engine._store._conn.execute(
        "SELECT store_id, uid, kind, proof_kind FROM host_uid_bindings WHERE kind = 'engine'").fetchall()
    assert (0, expected, "engine", uid_kind) in records
    assert engine._last_survival_fit["dropped_rows"] == 1


def test_7_cold_resume_with_host_stamps_adds_zero_rows(tmp_path, rough_counts):
    engine = _engine(tmp_path)
    try:
        fitted = _fit(engine, _view(), 100)
        before = _rows(engine)
        engine.on_session_end("S", fitted)
    finally:
        engine.shutdown()
    cold = _engine(tmp_path)
    try:
        cold.ingest(json.loads(json.dumps(fitted)))
        assert _rows(cold) == before
    finally:
        cold.shutdown()


def test_8_exit_cut_coverage_and_keep_from(engine, rough_counts):
    view = _view()
    second = [{"role": "user", "content": "[middle] request", "timestamp": 2.2},
              {"role": "assistant", "content": "[middle] answer", "timestamp": 2.5}]
    view[3:3] = second
    engine.ingest(view)
    ids = [row["store_id"] for row in _rows(engine)]
    # The immediately previous reply is deliberately uncovered; it must stay in the kept set.
    engine._dag.add_node(SummaryNode(session_id="S", summary="covered", source_ids=ids[:3]))
    mapping = engine._get_store_id_map_for_messages(view[1:])
    early = engine._survival_cut(view, 1, 100, True, "exit_fit:test", mapping, True, keep_from=3)
    assert early is not None and early[0][1:] == view[3:]  # an earlier cut keeps whole turns as before.
    assert early[1] == 2 and early[2] == ids[:2]
    newest_budget = rough_counts(early[0]) - rough_counts(view[3:4])
    blocked = engine._survival_cut(view, 1, newest_budget, True, "exit_fit:test", mapping, True, keep_from=3)
    assert blocked is None  # #668 never crosses the protected middle user row.
    last = engine._survival_cut(view, 1, newest_budget, True, "exit_fit:test", mapping, True, keep_from=5)
    assert last is not None and last[0][1:] == view[4:]
    assert last[1] == 3 and last[2] == ids[:3]  # #738 excludes the kept, uncovered reply.


@pytest.mark.parametrize("stamp", [None, "", "invalid"])
def test_9_unstamped_short_reply_is_byte_identical_to_base(engine, rough_counts, stamp):
    view = _view(stamped=False)
    if stamp is not None:
        view[2]["timestamp"] = stamp
    fitted = _fit(engine, view, 100)
    ids = [row["store_id"] for row in _rows(engine)]
    assert json.dumps(fitted, sort_keys=True) == json.dumps(_base_without_reply(view, ids), sort_keys=True)


@pytest.mark.parametrize("budget", [600, 3000])
def test_10_unstamped_long_reply_is_byte_identical_to_base(engine, rough_counts, budget):
    view = _view("[A1] " + "long " * 2000, stamped=False)
    fitted = _fit(engine, view, budget)
    ids = [row["store_id"] for row in _rows(engine)]
    assert json.dumps(fitted, sort_keys=True) == json.dumps(_base_without_reply(view, ids), sort_keys=True)


def test_11_stamped_short_verbatim_reply_survives_cold_resume(tmp_path, rough_counts):
    engine = _engine(tmp_path)
    try:
        view = _view()
        fitted = _fit(engine, view, 100)
        assert fitted[1] is view[2] and survival_fit._PROJECTED_PREFIX not in fitted[1]["content"]
        before = _rows(engine)
    finally:
        engine.shutdown()
    cold = _engine(tmp_path)
    try:
        cold.ingest(fitted)
        assert _rows(cold) == before
        assert sum(row["content"] == view[2]["content"] for row in _rows(cold)) == 1
    finally:
        cold.shutdown()
