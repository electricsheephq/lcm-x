"""Current projection bytes and selection policy, before codec extraction."""

from copy import deepcopy
import json

import pytest

import hermes_lcm.survival_fit as fit
from hermes_lcm.dag import SummaryNode
from hermes_lcm.message_content import normalize_content_value
from hermes_lcm.tokens import count_message_tokens as count
from tests.harness import (
    ScriptedSummarizer, assistant, call, engine_context, pin_char_counter, stored_rows, tool, user,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    pin_char_counter(monkeypatch)
    ScriptedSummarizer(fn=lambda *a, **kw: pytest.fail("unexpected model call"),
                       mode="scripted").install(monkeypatch)


@pytest.fixture
def engine(tmp_path):
    with engine_context(tmp_path, window=32_768, conversation_id="contract", temporal_rollups_enabled=False) as engine:
        yield engine


def message(kind, content):
    if kind == "user":
        return user(content)
    if kind == "tool":
        return tool("c", content)
    return assistant(content, calls=[call("c")] if kind == "assistant-calls" else [])


def marker(row, tokens):
    return (f"[LCM survival fit: this {row['role']} message ({tokens} tokens) is stored verbatim "
            f"as store id {row['store_id']}; lcm_expand / lcm_grep reach the full text. "
            "(projection 1200/600)]")


@pytest.mark.parametrize("length", [0, 1199, 1200, 1201, 2399, 2400, 2401, 6000])
@pytest.mark.parametrize("kind", ["user", "assistant", "assistant-calls", "tool"])
@pytest.mark.parametrize("structured", [False, True], ids=["text", "parts"])
def test_length_role_content_cells(engine, length, kind, structured):
    content = "x" * length
    if structured:
        content = [{"type": "text", "text": content}]
    source = message(kind, content)
    original = deepcopy(source)
    store_id = engine._store.append("S", source, conversation_id="contract")
    stored = engine._store.get(store_id)
    text = normalize_content_value(stored["content"]) or ""
    tokens = count(source)
    mark = marker(stored, tokens)
    # Independently render the pinned codec; the driver accepts only shrinking forms.
    head = (text[:1200] + "\n...\n" + mark + "\n...\n" + text[-600:] if len(text) > 2400
            else text[:1200] + "\n...\n" + mark if len(text) > 1200 else content)
    expected = {**source, "content": head}
    if tokens < 256 or count(expected) >= tokens:
        expected = source
    if (tokens >= 256 and 0 < len(text) <= 2400
            and count({**expected, "content": mark}) < count(expected)):
        expected = {**expected, "content": mark}
    result = engine._survival_projection([source], {id(source): store_id}, 0)
    assert result == [expected]
    assert source == original
    assert engine._store.get(store_id) == stored
    if result[0] is not source:
        assert count(result[0]) < tokens
    # rc8 permits user marks as the last resort when the zero budget remains unmet.


@pytest.mark.parametrize("tokens", [255, 256, 257])
@pytest.mark.parametrize("kind", ["user", "assistant", "assistant-calls", "tool"])
def test_exact_eligibility_floor(engine, monkeypatch, tokens, kind):
    source = message(kind, "x" * 6000)
    store_id = engine._store.append("S", source)
    monkeypatch.setattr(fit, "count_message_tokens", lambda row: tokens if row is source else 100)
    result = engine._survival_projection([source], {id(source): store_id}, 0)
    assert (result[0] is source) == (tokens == 255)
    if tokens >= 256:
        assert marker(engine._store.get(store_id), tokens) in result[0]["content"]


@pytest.mark.parametrize("kind", ["user", "assistant", "assistant-calls", "tool"])
def test_unstored_and_placeholder_rows(engine, kind):
    source = message(kind, "x" * 6000)
    before = deepcopy(source)
    assert engine._survival_projection([source], {}, 0) == [source]
    placeholder = "[Externalized payload: kind=raw_payload; synthetic placeholder]"
    store_id = engine._store.append("S", message(kind, placeholder))
    result = engine._survival_projection([source], {id(source): store_id}, 100)
    assert result == [{**source, "content": placeholder}]
    assert source == before


def test_head_pass_precedes_short_marks_and_tools_precede_larger_prose(engine):
    short = tool("short", "s" * 1199)
    long = tool("long", "l" * 6000)
    prose = assistant("p" * 8000)
    rows = [short, prose, long]
    ids = {id(row): engine._store.append("S", row) for row in rows}
    expected_long = {**long, "content": "l" * 1200 + "\n...\n"
                     + marker(engine._store.get(ids[id(long)]), count(long)) + "\n...\n" + "l" * 600}
    limit = engine._survival_measure([short, prose, expected_long])
    result = engine._survival_projection(rows, ids, limit)
    assert result == [short, prose, expected_long]
    assert result[0] is short and result[1] is prose
    exhausted = engine._survival_projection(rows, ids, 0)
    assert exhausted[0]["content"] == marker(engine._store.get(ids[id(short)]), count(short))
    assert exhausted[2] == expected_long


@pytest.mark.parametrize("kind,length", [(k, n) for k in ("user", "assistant", "assistant-calls", "tool")
                                         for n in (1199, 2400, 2401, 6000)])
def test_cold_replay_and_reprojection_identity(tmp_path, kind, length):
    with engine_context(tmp_path, window=32_768, conversation_id="contract", temporal_rollups_enabled=False) as engine:
        source = message(kind, "x" * length)
        source["timestamp"] = 2.0
        # Tool results need their call to be present at ingest, too.
        prefix = [user("request"), assistant("", [call("c")])] if kind == "tool" else []
        engine.ingest([*prefix, source])
        before = stored_rows(engine)
        mapping = engine._get_store_id_map_for_messages([source])
        result = engine._survival_projection([source], mapping, 0)
        again = engine._survival_projection(result, {id(result[0]): mapping[id(source)]}, 0)
        assert count(again[0]) <= count(result[0])
        if fit._PROJECTED_PREFIX in str(result[0]["content"]):
            for emitted in (result[0], again[0]):
                proof = engine._survival_projection_source(emitted, emitted["role"], emitted["content"])
                assert proof["store_id"] == mapping[id(source)]
        else:
            assert again == result
    with engine_context(tmp_path, window=32_768, conversation_id="contract", temporal_rollups_enabled=False) as cold:
        cold.ingest(deepcopy([*prefix, *result]))
        assert stored_rows(cold) == before


def test_pasted_marks_and_bounded_arguments(engine):
    pasted = marker({"role": "assistant", "store_id": 999999}, 585)
    source = assistant((pasted + "\n") * 4, [call("c", arguments="word " * 3000)])
    engine.ingest([user("request"), source])
    before = stored_rows(engine)
    mapping = engine._get_store_id_map_for_messages([source])
    result = engine._survival_projection([source], mapping, 0)[0]
    assert result["tool_calls"] != source["tool_calls"]
    assert source["tool_calls"][0]["function"]["arguments"] == "word " * 3000
    assert engine._survival_projection_source(result, "assistant", result["content"])["store_id"] == before[-1]["store_id"]
    assert engine._survival_projection([result], {id(result): mapping[id(source)]}, 0) == [result]


@pytest.mark.parametrize("site", ["content", "arguments", "emitted"])
@pytest.mark.parametrize("position", [4, 5], ids=["fourth-accepted", "fifth-ignored"])
def test_projection_candidate_scan_cap_on_cold_ingest(tmp_path, monkeypatch, site, position):
    """Each scan gets its own valid candidate, without another scan rescuing it."""
    with engine_context(tmp_path, window=32_768, conversation_id="contract",
                        temporal_rollups_enabled=False) as engine:
        # Fresh isolated stores start at 1; assert the reservation after appending.
        # Use the shared emitter seam so these fixtures also exercise pre-extraction code.
        def mark(store_id, head):
            row = {"role": "assistant", "store_id": store_id, "content": "raw"}
            return engine._survival_projected_fields(row, 1000, head, 100, legacy=True)["content"]

        fake = mark(900000, 2000)
        if site == "content":
            # The emitted mark is at 2005, away from the fixed-position shortcut,
            # and has a tail after it, so the ending shortcut cannot rescue it.
            prefix = (fake + "\n") * (position - 1)
            source = assistant(prefix.ljust(6000, "x"))
            head = 2000
        elif site == "arguments":
            # Content's own mark is beyond its scan cap. The quoted argument is
            # retained verbatim and is not a bounded notice (its value starts "quoted").
            source = assistant(((fake + "\n") * 4).ljust(6000, "x"), [call("quoted", arguments=json.dumps({
                "lcm_survival_fit": "quoted " + (fake + "\n") * (position - 1) + mark(1, 2000),
            }))])
            head = 2000
        else:
            # Saturate the argument scan before the distinct bounded notices.
            # Repeated marks do not count as distinct emitted candidates.
            quoted = call("quoted", arguments=json.dumps({"lcm_survival_fit": "quoted " + (fake + "\n") * 4}))
            notices = [engine._survival_bounded_call(call(f"decoy-{i}", arguments="x" * 6000),
                                                    mark(900001 + i, 1200)) for i in range(position - 1)]
            source = assistant("", [quoted, *notices, call("real", arguments="x" * 6000)])
            head = 1200
        source["timestamp"] = 9.0  # retain occurrence proof across the cold restart
        store_id = engine._store.append("S", source, conversation_id="contract")
        assert store_id == 1
        before = stored_rows(engine)
        projected = {**source, **engine._survival_projected_fields(before[0], 1000, head, 100)}
        assert projected != source

    with engine_context(tmp_path, window=32_768, conversation_id="contract",
                        temporal_rollups_enabled=False) as cold:
        # Observe the real verifier while invoking only the public replay entry point.
        # Other reconciliation fallbacks can still deduplicate a byte-identical view;
        # they must not conceal whether this candidate was accepted as its source row.
        recognized = []
        verify = cold._survival_projection_source

        def record(message, role, content):
            row = verify(message, role, content)
            if message == projected:
                recognized.append(None if row is None else row["store_id"])
            return row

        monkeypatch.setattr(cold, "_survival_projection_source", record)
        cold.ingest([deepcopy(projected)])
        assert recognized
        assert set(recognized) == ({store_id} if position == 4 else {None})
        after = stored_rows(cold)
        assert after[:1] == before  # replay never changes the verbatim source
        if position == 4:
            assert after == before
        else:
            assert len(after) == 2


@pytest.mark.parametrize("covered", [False, True])
def test_exit_fit_drops_only_covered_whole_turns(engine, covered):
    engine._config.fresh_tail_count = 2
    rows = [user("old " * 2000), assistant("old reply"), user("new request"), assistant("new reply")]
    engine.ingest(rows)
    before = stored_rows(engine)
    if covered:
        engine._dag.add_node(SummaryNode(session_id="S", summary="covered", source_type="messages",
                                         source_ids=[row["store_id"] for row in before[:2]]))
    original = deepcopy(rows)
    fitted = engine._survival_fit(rows, rows, 0, "exit_fit:contract", request_cap=200)
    assert fitted == rows[2:] if covered else fitted is rows
    assert rows == original and stored_rows(engine) == before
    if covered:
        assert fitted[0] is rows[2] and fitted[1] is rows[3]
        assert engine._last_survival_fit["uncovered_rows"] == 0


@pytest.mark.parametrize("length,kept", [(20, True), (1201, True), (2400, False), (6000, True)])
def test_previous_reply_keep_and_fixed_1227_boundary(engine, length, kept):
    rows = [user("old " * 6000), {**assistant("r" * length), "timestamp": 2.0}, user("new request")]
    engine.ingest(rows)
    before = stored_rows(engine)
    fitted = engine._survival_fit(rows, rows, 0, "contract", request_cap=600)
    replies = [row for row in fitted if row["role"] == "assistant"]
    assert bool(replies) == kept  # #798: middle-band replies are not projected here.
    assert fitted[-1] is rows[-1]
    if replies:
        assert replies[0] is rows[1] if length <= 1201 else replies[0]["content"].endswith("r" * 600)
    assert stored_rows(engine) == before


@pytest.mark.parametrize("stored", [False, True])
def test_impossible_fit_reports_failure_without_losing_raw_rows(engine, caplog, stored):
    rows = [user("x" * 6000)]
    if stored:
        engine.ingest(rows)
    before = stored_rows(engine)
    fitted = engine._survival_fit(rows, rows, 0, "contract-impossible", request_cap=10)
    assert fitted and engine._survival_measure(fitted) > 10
    record = engine._store.read_metadata_json(fit.SURVIVAL_FIT_COUNTER_KEY)
    assert record["count"] == record["unreached_budget_count"] == 1
    assert record["last_reached_budget"] is False
    assert record["last_shortened"] is stored
    assert record["projected_count"] == int(stored)
    assert engine._last_survival_fit["reached_budget"] is False
    assert ("could not reach budget" if stored else "could not shorten the list") in caplog.text
    assert "budget=10" in caplog.text and "reason=contract-impossible" in caplog.text
    assert stored_rows(engine) == before


def test_fixed_1227_short_assistant_fit_and_cold_replay(tmp_path):
    with engine_context(tmp_path, window=32_768, conversation_id="contract",
                        temporal_rollups_enabled=False) as engine:
        rows = [user("request"), *[{**assistant(str(i) + "x" * 1198), "timestamp": float(i + 1)}
                                   for i in range(4)]]
        engine.ingest(rows)
        raw = stored_rows(engine)
        original = deepcopy(rows)
        fitted = engine._survival_fit(rows, rows, 0, "fixed-1227", request_cap=500)
        assert fitted[0] is rows[0]
        assert len(fitted) == len(rows)
        assert engine._survival_measure(fitted) <= 500
        assert sum(bool(fit._PROJECTED_RE.fullmatch(row["content"])) for row in fitted) == 3
        assert engine._last_survival_fit["reached_budget"] is True
        assert rows == original and stored_rows(engine) == raw
    with engine_context(tmp_path, window=32_768, conversation_id="contract",
                        temporal_rollups_enabled=False) as cold:
        cold.ingest(deepcopy(fitted))
        assert stored_rows(cold) == raw
