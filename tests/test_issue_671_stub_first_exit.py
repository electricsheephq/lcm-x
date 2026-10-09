"""#671: stub tiers (10k first sight, 2k aged at compaction) and the stub-first exit."""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_tokens

STUB = "[Externalized tool output:"


def payload_of(tokens: int, word: str = "lorem") -> str:
    """About ``tokens`` tokens by the plugin's own counter (tiktoken or the character estimate)."""
    per_word = count_tokens((word + " ") * 1_000) / 1_000
    text = (word + " ") * max(1, round(tokens / per_word))
    assert abs(count_tokens(text) - tokens) <= max(10, tokens // 20)
    return text


@pytest.fixture
def make_engine(tmp_path):
    engines = []

    def build(**overrides):
        settings = dict(
            database_path=str(tmp_path / f"issue-671-{len(engines)}.db"),
            fresh_tail_count=2,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=1_000_000,
            large_output_active_replay_stubbing_enabled=True,
        )
        settings.update(overrides)
        engine = LCMEngine(config=LCMConfig(**settings), hermes_home=str(tmp_path / "hermes"))
        engine.on_session_start(
            f"issue-671-{len(engines)}",
            conversation_id=f"issue-671-conversation-{len(engines)}",
            context_length=200_000,
        )
        engines.append(engine)
        return engine

    yield build
    for engine in engines:
        engine.shutdown()


def tool_pair(call_id, payload, tool_name="read_file"):
    return [
        {
            "role": "assistant",
            "content": "running tool",
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": tool_name, "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": payload},
    ]


def tool_content(result, call_id):
    return next(
        m["content"] for m in result if m.get("role") == "tool" and m.get("tool_call_id") == call_id
    )


# -- commit 1: stub tiers -------------------------------------------------------------------------


def test_defaults_are_10k_first_sight_and_2k_aged(monkeypatch):
    config = LCMConfig()
    assert config.large_output_active_replay_stub_threshold_tokens == 10_000
    assert config.large_output_active_replay_stub_aged_threshold_tokens == 2_000
    monkeypatch.setenv("LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_AGED_THRESHOLD_TOKENS", "3000")
    assert LCMConfig.from_env().large_output_active_replay_stub_aged_threshold_tokens == 3_000


def test_aged_tier_stubs_3k_outside_the_fresh_tail_and_keeps_it_inside(make_engine):
    engine = make_engine()
    aged = payload_of(3_000)
    tail = tool_pair("old-call", aged) + tool_pair("fresh-call", aged)

    result = engine._assemble_context({"role": "system", "content": "system"}, tail)

    assert tool_content(result, "old-call").startswith(STUB)
    assert tool_content(result, "fresh-call") == aged  # the fresh tail never gets an aged stub


def test_aged_tier_zero_uses_the_first_sight_threshold(make_engine):
    engine = make_engine(large_output_active_replay_stub_aged_threshold_tokens=0)
    aged = payload_of(3_000)
    tail = tool_pair("old-call", aged) + tool_pair("fresh-call", "fresh")

    result = engine._assemble_context({"role": "system", "content": "system"}, tail)

    assert tool_content(result, "old-call") == aged


def test_aged_tier_never_exceeds_the_first_sight_threshold(make_engine):
    engine = make_engine(
        large_output_active_replay_stub_threshold_tokens=50,
        large_output_active_replay_stub_aged_threshold_tokens=2_000,
    )
    tail = tool_pair("old-call", payload_of(300)) + tool_pair("fresh-call", "fresh")

    result = engine._assemble_context({"role": "system", "content": "system"}, tail)

    assert tool_content(result, "old-call").startswith(STUB)


def test_first_sight_keeps_6k_whole_and_stubs_11k(make_engine):
    engine = make_engine(fresh_tail_count=32)
    small, large = payload_of(6_000), payload_of(11_000, word="ipsum")
    messages = [{"role": "system", "content": "system"}, *tool_pair("six", small), *tool_pair("eleven", large)]

    replay = engine._ingest_messages(messages)

    assert tool_content(replay, "six") == small
    assert tool_content(replay, "eleven").startswith(STUB)


def test_with_both_opt_in_flags_off_nothing_is_stubbed(make_engine):
    engine = make_engine(
        large_output_externalization_enabled=False,
        large_output_active_replay_stubbing_enabled=False,
    )
    aged = payload_of(3_000)
    tail = tool_pair("old-call", aged) + tool_pair("fresh-call", "fresh")

    result = engine._assemble_context({"role": "system", "content": "system"}, tail)

    assert tool_content(result, "old-call") == aged


# -- commit 2: the stub-first exit ----------------------------------------------------------------------------------

import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402

import hermes_lcm.engine as lcm_engine  # noqa: E402
import hermes_lcm.tools as lcm_tools  # noqa: E402
from hermes_lcm.command import _doctor_text  # noqa: E402
from hermes_lcm.tokens import count_messages_tokens  # noqa: E402

STOP = re.compile(r"LCM compaction stop: reason=(\S+) leaves=(\d+) .* backlog_tokens=\d+"
                  r"(?: tokens_before=(\d+) tokens_after=(\d+) target=(\d+) backlog_rows=(\d+))?")
THRESHOLD = 12_000
TURN = " alpha beta gamma delta" * 40


@pytest.fixture
def summaries(monkeypatch):
    calls: list[str] = []

    def summarize(**kwargs):
        calls.append(kwargs["text"])
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return calls


def exit_engine(make_engine, **overrides):
    engine = make_engine(**{"leaf_chunk_tokens": 1_000, "fresh_tail_pressure_yield_enabled": False, **overrides})
    engine.threshold_tokens = THRESHOLD
    return engine


def tool_view(pairs: int = 5, tokens: int = 3_000, prefix=()) -> list[dict]:
    """A tool-heavy list over the threshold whose aged-tier stubs bring it far under the target."""
    rows = [{"role": "system", "content": "system prompt"}, *prefix]
    for i in range(pairs):
        rows += [{"role": "user", "content": f"turn {i}"}, *tool_pair(f"call-{i}", payload_of(tokens, f"w{i}x"))]
    return rows + [{"role": "user", "content": "latest ask"}, {"role": "assistant", "content": "latest answer"}]


def text_view(turns: int = 14) -> list[dict]:
    rows = [{"role": "system", "content": "system prompt"}]
    for i in range(turns):
        rows += [{"role": "user", "content": f"[T{i}]" + TURN * 5}, {"role": "assistant", "content": f"r{i}" + TURN * 5}]
    return rows


def run(engine, view, caplog, **kwargs):
    current = kwargs.pop("current_tokens", count_messages_tokens(view))
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        return engine.compress(view, current_tokens=current, **kwargs)


def stop_line(caplog):
    found = [STOP.search(record.getMessage()) for record in caplog.records]
    return [m for m in found if m][-1]


def test_free_cuts_that_reach_the_target_exit_with_no_model_call(make_engine, summaries, caplog):
    engine = exit_engine(make_engine, deferred_maintenance_enabled=True)
    view = tool_view()
    assert count_messages_tokens(view) >= THRESHOLD

    result = run(engine, view, caplog)

    assert summaries == []  # no leaf, no condensation, no model call
    assert len(engine._dag.get_session_nodes(engine._session_id)) == 0
    assert engine._store.get_session_count(engine._session_id) == len(view)  # every row stored
    target = min(THRESHOLD - 1_000, int(THRESHOLD * 0.95))
    assert count_messages_tokens(result) <= target
    assert all(tool_content(result, f"call-{i}").startswith(STUB) for i in range(5))
    assert result[-2:] == view[-2:]  # the fresh tail is never aged
    record = engine.get_status()["last_stub_first_exit"]
    assert record["target_tokens"] == target and record["tokens_after"] <= target
    assert record["tokens_before"] == count_messages_tokens(view) and record["backlog_rows"] > 0
    state = engine._lifecycle.get_by_conversation(engine._conversation_id)
    assert state.debt_kind == "raw_backlog" and state.debt_size_estimate > 0  # the backlog is recorded
    line = stop_line(caplog)
    assert line.group(1) == "stub_first_exit" and line.group(2) == "0"
    assert int(line.group(4)) == record["tokens_after"] and int(line.group(5)) == target
    assert int(line.group(6)) == record["backlog_rows"]
    assert engine.last_compression_status == "sanitized" and engine._ingest_cursor == len(result)
    assert engine._last_survival_fit is None  # under 0.95 x threshold: the #668 exit fit is a no-op after it


def test_the_exit_is_a_partial_stop_in_lcm_status_and_the_doctor(make_engine, summaries, caplog):
    engine = exit_engine(make_engine, threshold_full_sweep_enabled=True)
    run(engine, tool_view(), caplog)

    status = json.loads(lcm_tools.lcm_status({}, engine=engine))
    assert status["threshold_full_sweep"]["stop_reason"] == "stub_first_exit"
    assert status["threshold_full_sweep"]["status"] == "partial"
    assert status["last_stub_first_exit"]["backlog_rows"] > 0
    assert "stub_first_exit: " in _doctor_text(engine)
    assert stop_line(caplog).group(1) == "stub_first_exit"
    assert summaries == []


def test_the_summary_prefix_stays_first_after_an_exit(make_engine, summaries, caplog):
    engine = exit_engine(make_engine, threshold_full_sweep_enabled=False)  # #1013: one summary row, as before
    first = run(engine, text_view(), caplog)
    assert summaries and engine.last_compression_status == "compacted"
    calls = len(summaries)
    summary_row = first[1]
    assert summary_row["role"] == "user" and "Earlier turns." in str(summary_row["content"])

    view = tool_view(prefix=first[1:])
    result = run(engine, view, caplog)

    assert len(summaries) == calls  # no new model call
    assert engine._stub_first_exit_now is not None
    assert result[0] == view[0] and result[1]["content"] == summary_row["content"]
    assert sum("Earlier turns." in str(m.get("content")) for m in result) == 1  # the prefix is not duplicated


def test_free_cuts_that_miss_the_target_run_the_same_calls_as_without_stubbing(make_engine, summaries, caplog):
    off = exit_engine(make_engine, large_output_active_replay_stubbing_enabled=False)
    run(off, text_view(), caplog)
    base_calls = len(summaries)
    summaries.clear()

    on = exit_engine(make_engine)
    run(on, text_view(), caplog)

    assert base_calls > 0 and len(summaries) == base_calls
    assert on._last_stub_first_exit is None
    assert stop_line(caplog).group(1) != "stub_first_exit"


@pytest.mark.parametrize("kwargs", [
    {"bypass_cooldown": True},  # a recovery attempt (#608)
    {"force": True},
    {"current_tokens": THRESHOLD - 1},  # below the threshold (#651)
])
def test_no_exit_on_recovery_forced_or_below_threshold(make_engine, summaries, caplog, kwargs):
    engine = exit_engine(make_engine)
    run(engine, tool_view(), caplog, **kwargs)

    assert engine._last_stub_first_exit is None
    assert stop_line(caplog).group(1) != "stub_first_exit"


def test_no_exit_with_both_opt_in_flags_off(make_engine, summaries, caplog):
    engine = exit_engine(
        make_engine, large_output_externalization_enabled=False, large_output_active_replay_stubbing_enabled=False
    )
    result = run(engine, tool_view(), caplog)

    assert engine._last_stub_first_exit is None and summaries
    assert not any(str(m.get("content", "")).startswith(STUB) for m in result)


def test_an_assembly_cap_never_counts_as_a_free_cut(make_engine, summaries, caplog):
    view = tool_view()
    engine = exit_engine(make_engine, max_assembly_tokens=200)  # under the stubbed list
    # A cap under the input is forced overflow; the guard covers an assembly larger than its input.
    engine._should_force_overflow_recovery = lambda **kwargs: False
    run(engine, view, caplog)

    assert engine._last_stub_first_exit is None


def test_host_overhead_counts_against_the_target(make_engine, summaries, caplog):
    engine = exit_engine(make_engine)
    view = tool_view()
    run(engine, view, caplog, current_tokens=count_messages_tokens(view) + 11_000)  # a large system/tools overhead

    assert engine._last_stub_first_exit is None


def _trial(engine, view, current_tokens):
    working = engine._ingest_messages(view)
    engine._compress_occurrences = None
    return engine._stub_first_exit(view, working, list(working), working, current_tokens)


def test_a_trial_that_misses_leaves_no_emission_candidates_behind(make_engine):
    engine = exit_engine(make_engine)
    view = tool_view()
    engine._pending_emission_candidates = ["before the trial"]

    from pathlib import Path

    from hermes_lcm.reconcile import _COMPACTED_ACTIVE_REPLAY_METADATA_PREFIX as prefix
    key = engine._replay_snapshot_metadata_key(prefix)
    digests = engine._store.read_metadata_json(key)

    assert _trial(engine, view, count_messages_tokens(view) + 50_000) is None  # the overhead misses the target
    assert engine._pending_emission_candidates == ["before the trial"]
    assert engine._last_stub_first_exit is None
    assert engine._store.read_metadata_json(key) == digests  # no snapshot digest (#726 round 2: no write at all)
    assert not list((Path(engine._hermes_home) / "lcm-large-outputs").glob("*.json"))  # no payload file


def test_a_resumed_committed_prefix_takes_todays_path(make_engine, monkeypatch):
    engine = exit_engine(make_engine)
    view = tool_view()
    monkeypatch.setattr(engine, "_committed_replay_drops", lambda working, start: ([1], None, 0, 0))

    assert _trial(engine, view, count_messages_tokens(view)) is None
    monkeypatch.setattr(engine, "_committed_replay_drops", lambda working, start: ([], None, 0, 0))
    assert _trial(engine, view, count_messages_tokens(view)) is not None


def test_the_exit_needs_no_summary_route(make_engine, summaries, caplog, monkeypatch):
    engine = exit_engine(make_engine)
    monkeypatch.setattr(engine, "_summary_route_available", lambda: False)  # #628: every route refused
    monkeypatch.setattr(engine, "_summary_route_stop_applies", lambda *args, **kwargs: True)
    run(engine, tool_view(), caplog)

    assert engine._stub_first_exit_now is not None and summaries == []
    assert stop_line(caplog).group(1) == "stub_first_exit"


def test_the_fit_only_hold_at_the_ceiling_runs_first(make_engine, summaries, caplog, monkeypatch):
    engine = exit_engine(make_engine)
    monkeypatch.setattr(engine, "_hold_fit_only_applies", lambda tokens: True)  # #618: held at the survival ceiling
    run(engine, tool_view(), caplog)

    assert engine._last_stub_first_exit is None and summaries == []
    assert engine.last_compression_noop_reason == "held"


# -- #726 round 2 ---------------------------------------------------------------------------------------------------

import copy  # noqa: E402
import time  # noqa: E402

from hermes_lcm.dag import SummaryNode  # noqa: E402


def _fold_engine(tmp_path, name="fold-671"):
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "fold.db"), fresh_tail_count=2, leaf_chunk_tokens=1_000, l3_truncate_tokens=1,
        fresh_tail_pressure_yield_enabled=False, large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=1_000_000, large_output_active_replay_stubbing_enabled=True,
    ), hermes_home=str(tmp_path / "hermes"))
    engine.on_session_start(name, conversation_id=name + "-conversation", context_length=200_000)
    engine.threshold_tokens = 12_000
    return engine


def test_a_missed_exit_leaves_the_folded_row_lineage_alone(tmp_path, summaries):
    """F1 (review of #726): retained sole user, a folded assistant/tool-call row, refused routes, a missed target."""
    engine = _fold_engine(tmp_path)
    system, anchor = {"role": "system", "content": "system"}, {"role": "user", "content": "sole retained question"}
    ids = engine._store.append_batch(engine._session_id, [system, anchor, {"role": "assistant", "content": "older"}])
    engine._last_compacted_store_id = ids[2]
    engine._dag.add_node(SummaryNode(
        session_id=engine._session_id, depth=0, summary="Folded carrier summary.", token_count=6,
        source_token_count=8, source_ids=[ids[2]], source_type="messages", created_at=time.time(),
        expand_hint="folded carrier summary"))
    tail = tool_pair("fold-call", "short result") + [{"role": "assistant", "content": "historical answer"}]
    engine._store.append_batch(engine._session_id, tail)
    engine._prepare_retained_user_anchor([system, anchor, *tail])
    view = engine._assemble_committed_compaction_context([system, anchor, *tail], [system, anchor, *tail], None)
    engine._last_compression_status, engine._ingest_cursor = "compacted", len(view)
    mapped = engine._get_store_ids_for_messages(view)
    engine._summary_route_available = lambda: False
    engine._summary_route_stop_applies = lambda *args, **kwargs: True

    out = engine.compress(view, current_tokens=12_000)  # the host overhead misses the target

    assert out == view and engine._last_stub_first_exit is None and summaries == []
    assert engine._get_store_ids_for_messages(out) == mapped  # the fold still maps to its durable source
    stored = engine._store.get_session_count(engine._session_id)
    engine.shutdown()
    cold = _fold_engine(tmp_path)
    cold._ingest_messages(copy.deepcopy(out) + [{"role": "user", "content": "fresh-new-question"}])
    assert cold._store.get_session_count(cold._session_id) - stored == 1
    cold.shutdown()


@pytest.mark.parametrize(("aged", "tokens"), [(2_000, 3_001), (0, 11_000)])
def test_no_stub_inside_a_protected_tool_group(make_engine, aged, tokens):
    """F2 (review of #726): one assistant opens 33 calls; the resolved fresh tail protects the whole group."""
    engine = make_engine(fresh_tail_count=32, large_output_active_replay_stub_aged_threshold_tokens=aged)
    calls = [{"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}} for i in range(33)]
    rows = [{"role": "user", "content": "ask"}, {"role": "assistant", "content": "", "tool_calls": calls}]
    rows += [{"role": "tool", "tool_call_id": f"c{i}", "content": payload_of(tokens) if i == 0 else "ok"} for i in range(33)]
    rows.append({"role": "assistant", "content": "done"})

    result = engine._assemble_context({"role": "system", "content": "system"}, rows)

    assert not tool_content(result, "c0").startswith(STUB)


def test_proactive_recall_runs_once_on_a_taken_exit_and_never_in_the_trial(make_engine, summaries, caplog, monkeypatch):
    built = []
    engine = exit_engine(make_engine, proactive_recall_enabled=True, l3_truncate_tokens=1)
    monkeypatch.setattr(engine, "_build_proactive_recall_message", lambda *args: built.append(1))
    run(engine, tool_view(), caplog)
    assert engine._stub_first_exit_now is not None and built == [1]

    built.clear()
    missed = exit_engine(make_engine, proactive_recall_enabled=True, l3_truncate_tokens=1)
    monkeypatch.setattr(missed, "_build_proactive_recall_message", lambda *args: built.append(1))
    missed._summary_route_available = lambda: False
    missed._summary_route_stop_applies = lambda *args, **kwargs: True
    view = tool_view()
    run(missed, view, caplog, current_tokens=count_messages_tokens(view) + 11_000)
    assert missed._last_stub_first_exit is None and built == []


def test_the_exit_record_describes_only_the_latest_compaction(make_engine, summaries, caplog):
    engine = exit_engine(make_engine)
    result = run(engine, tool_view(), caplog)
    assert engine.get_status()["last_stub_first_exit"] is not None

    run(engine, result, caplog, force=True)  # a later non-exit compaction

    assert engine.get_status()["last_stub_first_exit"] is None
    assert "stub_first_exit: " not in _doctor_text(engine)
    run(engine, tool_view(), caplog)
    engine._reset_session_counters()  # a reset or rebind
    assert engine.get_status()["last_stub_first_exit"] is None


def test_a_missed_trial_writes_no_ingest_payload_through_a_preserved_objective(make_engine):
    """G1 (#726 round 3): a preserved-objective scaffold with inline media after an ordinary tool result."""
    from pathlib import Path

    from hermes_lcm.reconcile import _PRESERVED_OBJECTIVE_CONTEXT_PREFIX as prefix
    engine = exit_engine(make_engine)
    view = tool_view(pairs=1)
    view.insert(4, {"role": "user", "content": prefix + "\ndata:image/png;base64," + "YWJjZGVmZ2hp" * 300})
    files = lambda: set(Path(engine._hermes_home).rglob("*.json"))  # noqa: E731
    working = engine._ingest_messages(view)
    engine._compress_occurrences = None
    before = files()

    assert engine._stub_first_exit(view, working, list(working), working, count_messages_tokens(view) + 50_000) is None
    assert sum(str(m.get("content", "")).startswith(prefix) for m in working) == 1  # the scaffold reached the trial
    assert files() == before  # no ingest-payload file


def test_a_final_assembly_that_misses_the_target_takes_the_normal_path(make_engine, summaries, caplog, monkeypatch):
    """G2 (#726 round 3): the trial fits, every payload write of the final assembly fails, the survival fit is off."""
    import hermes_lcm.externalize as externalize

    engine = exit_engine(make_engine, survival_fit=False)
    engine.context_length = 16_000
    keys = [engine._folded_tail_lineage_metadata_key(), engine._active_replay_snapshot_metadata_key()]
    for key, value in zip(keys, [{"version": 1, "source_store_id": 0}, {"version": 1, "digests": ["a" * 64]}]):
        engine._store.write_metadata_json([key], json.dumps(value, sort_keys=True))
    saved = [engine._store.read_metadata_json(key) for key in keys]
    seen, trial = [], engine._stub_first_exit

    def recorded(*args):
        exited = trial(*args)
        seen.append((exited, [engine._store.read_metadata_json(key) for key in keys], engine._pending_emission_candidates))
        return exited

    def refuse(*args, **kwargs):
        raise OSError("synthetic ENOSPC")

    monkeypatch.setattr(engine, "_stub_first_exit", recorded)
    monkeypatch.setattr(externalize, "_write_externalized_payload", refuse)
    result = run(engine, tool_view(pairs=6), caplog)

    assert seen == [(None, saved, [])]  # lineage, snapshot and emissions are their pre-final values
    assert engine._last_stub_first_exit is None and summaries
    assert "the final assembly missed the target" in caplog.text
    assert count_messages_tokens(result) <= engine.context_length
