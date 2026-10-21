"""#787 slice 1: background leaf preparation (default off) and promotion at the threshold crossing.

The provider stands in at ``escalation._invoke_summary_llm``, so the real escalation chain, its breaker keys and
spend guards run on both the worker and the turn thread. Every test uses its own process-wide preparation state.
"""

from __future__ import annotations

import contextvars
import importlib
import os
import sys
import threading
import time

import pytest

import hermes_lcm.escalation as escalation
import hermes_lcm.leaf_prep as leaf_prep
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.escalation import SummarySpendGuard
from tests.test_issue_622_slow_load import _Ctx, _Manager, _install_host, _load_plugin

WINDOW = 60_000  # T = 45,000; P = 36,000; L = 45,000 - max(2 x 2,000, 6,000) = 39,000
CHUNK = 2_000
PAD = " alpha beta gamma delta epsilon" * 64
SUMMARY = "Earlier turns.\nExpand for details about: turns"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("LCM_DATABASE_PATH", raising=False)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(
        tokens,
        "count_tokens",
        lambda text: (
            0
            if not text
            else (
                tokens._fallback_token_estimate(text)
                if isinstance(text, str)
                else len(text) // 4 + 1
            )
        ),
    )
    state = leaf_prep._LeafPrep()
    monkeypatch.setattr(leaf_prep, "_LEAF_PREP", state)
    yield state
    state.wait_idle(10)


class _Provider:
    """Counts provider calls per thread; thread-safe; optional per-call hook."""

    def __init__(self, result=SUMMARY):
        self.result, self.calls, self.lock, self.hook = (
            result,
            [],
            threading.Lock(),
            None,
        )

    def __call__(self, prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        with self.lock:
            self.calls.append(threading.current_thread().name)
        if self.hook is not None:
            self.hook()
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result(prompt) if callable(self.result) else self.result

    def on(self, thread_name: str) -> int:
        with self.lock:
            return sum(1 for name in self.calls if name == thread_name)


@pytest.fixture
def provider(monkeypatch):
    fake = _Provider()
    monkeypatch.setattr(escalation, "_invoke_summary_llm", fake)
    return fake


def _engine(tmp_path, *, window: int = WINDOW, **config) -> LCMEngine:
    settings = {
        "database_path": str(tmp_path / "lcm.db"),
        "background_leaf_prep_enabled": True,
        "background_leaf_prep_max_calls": 64,  # chunks end at host-unit boundaries: more, smaller leaves
        "fresh_tail_count": 2,
        "leaf_chunk_tokens": CHUNK,
        "context_threshold": 0.75,
        "threshold_full_sweep_enabled": True,
        "large_output_externalization_enabled": False,
        "large_output_active_replay_stubbing_enabled": False,
        "temporal_rollups_enabled": False,
        "l3_truncate_tokens": 2,
        "incremental_max_depth": 0,
        **config,
    }
    if os.environ.get(
        "LCM_787_RED_RECEIPT"
    ):  # the red receipt: every engine runs with the feature off
        settings["background_leaf_prep_enabled"] = False
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start(
        "S", platform="telegram", context_length=window, conversation_id="conv"
    )
    return engine


def _view(turns: int) -> list[dict]:
    rows = [{"role": "system", "content": "system prompt"}]
    for i in range(turns):
        rows.append(
            {
                "role": "user",
                "content": f"[T{i}] user turn{PAD}",
                "timestamp": float(i + 1),
            }
        )
        rows.append(
            {
                "role": "assistant",
                "content": f"reply to T{i}{PAD}",
                "timestamp": float(i + 1),
            }
        )
    return rows


def _turn_end(engine, view, *, tokens_now: int | None = None) -> None:
    """The post_llm_call work: ingest, then the enqueue; the worker runs to completion."""
    engine.ingest(view)
    engine.last_prompt_tokens = (
        tokens.count_messages_tokens(view) if tokens_now is None else tokens_now
    )
    engine._prepare_leaves_after_turn(view)
    assert leaf_prep._LEAF_PREP.wait_idle(10)


def _leaf_sources(engine) -> list[list[int]]:
    return [
        node.source_ids
        for node in engine._dag.get_session_nodes("S", limit=100_000)
        if node.depth == 0
    ]


def _frontier(engine) -> int:
    return int(engine._lifecycle.get_by_conversation("conv").current_frontier_store_id)


@pytest.fixture
def engine(tmp_path, provider):
    instance = _engine(tmp_path)
    try:
        yield instance
    finally:
        instance.shutdown()


def _crossing_view(engine) -> list[dict]:
    """A list just over the threshold (T = 45,000)."""
    turns = 1
    while tokens.count_messages_tokens(_view(turns)) <= engine.threshold_tokens:
        turns += 1
    return _view(turns)


# 1. off ------------------------------------------------------------------------------------------------------


def test_setting_defaults_off_and_parses_like_the_other_booleans(monkeypatch):
    assert LCMConfig().background_leaf_prep_enabled is False
    assert (
        LCMConfig().background_leaf_prep_max_calls,
        LCMConfig().background_leaf_prep_window_s,
    ) == (8, 600.0)
    monkeypatch.setenv("LCM_BACKGROUND_LEAF_PREP_ENABLED", "true")
    monkeypatch.setenv("LCM_BACKGROUND_LEAF_PREP_MAX_CALLS", "3")
    monkeypatch.setenv("LCM_BACKGROUND_LEAF_PREP_WINDOW_S", "60")
    config = LCMConfig.from_env()
    assert config.background_leaf_prep_enabled is True
    assert (
        config.background_leaf_prep_max_calls,
        config.background_leaf_prep_window_s,
    ) == (3, 60.0)


def test_off_starts_no_thread_enqueues_nothing_and_never_promotes(
    tmp_path, provider, monkeypatch, _isolated
):
    engine = _engine(tmp_path, background_leaf_prep_enabled=False)
    monkeypatch.setattr(
        leaf_prep,
        "ready_chain",
        lambda *a: pytest.fail("promotion consulted while off"),
    )
    try:
        view = _crossing_view(engine)
        _turn_end(engine, view)
        assert _isolated._worker is None and _isolated.counters["jobs_enqueued"] == 0
        engine.compress(view, current_tokens=tokens.count_messages_tokens(view))
        assert provider.on("MainThread") >= 1
        assert "background_leaf_prep" not in engine.get_status()
    finally:
        engine.shutdown()


# 2. prepare, then promote --------------------------------------------------------------------------------------


def test_prepared_leaves_publish_at_the_crossing_with_no_model_call(
    engine, provider, _isolated
):
    view = _crossing_view(engine)
    _turn_end(engine, view)
    prepared = _isolated.counters["prepared"]
    assert (
        prepared >= 4
        and provider.on("lcm-leaf-prep") == prepared
        and provider.on("MainThread") == 0
    )

    host_tokens = tokens.count_messages_tokens(view)
    result = engine.compress(view, current_tokens=host_tokens)

    assert provider.on("MainThread") == 0  # 0 summariser calls on the turn
    leaves = _leaf_sources(engine)
    assert len(leaves) == _isolated.counters["promoted"] >= 4
    assert engine.last_compression_status == "compacted"
    sweep = engine._last_threshold_full_sweep
    assert sweep["stop_reason"] == "prepared_leaves_promoted" and sweep[
        "promoted_leaves"
    ] == len(leaves)
    low_water = leaf_prep.levels(engine.threshold_tokens, engine.context_length, CHUNK)[
        1
    ]
    freed = sum(
        tokens.count_messages_tokens(view[1 + sum(map(len, leaves[:i])) :][: len(ids)])
        for i, ids in enumerate(leaves)
    )
    assert host_tokens - freed + len(leaves) * tokens.count_tokens(SUMMARY) <= low_water
    assert tokens.count_messages_tokens(result) < engine.threshold_tokens
    assert _frontier(engine) == leaves[-1][-1]
    status = engine.get_status()["background_leaf_prep"]
    assert status["promoted"] == len(leaves) and status["prepared"] == prepared


def _prepared(engine, _isolated, view) -> list:
    _turn_end(engine, view)
    chain = leaf_prep.ready_chain(engine._store.db_path, "conv", _frontier(engine))
    assert len(chain) >= 4
    return chain


def _second_crossing_view(engine, provider):
    view = _crossing_view(engine)[1:]
    engine.ingest(view)
    engine._config.background_leaf_prep_enabled = False
    view = engine.compress(view, current_tokens=tokens.count_messages_tokens(view))
    engine._config.background_leaf_prep_enabled = True
    assert provider.on("MainThread") > 0
    assert engine._generated_context_carrier_remainder(view[0]) is not None
    for turn in range(100):
        _turn_end(engine, view)
        if tokens.count_messages_tokens(view) > engine.threshold_tokens:
            return view
        view = view + [
            {"role": "user", "content": f"[N{turn}] new user{PAD}"},
            {"role": "assistant", "content": f"new reply N{turn}{PAD}"},
        ]
    pytest.fail("second crossing was never reached")


def test_summary_carrier_with_unmapped_glued_row_rejects_prepared_chain(
    engine, provider, _isolated
):
    view = _second_crossing_view(engine, provider)
    assert leaf_prep.ready_chain(engine._store.db_path, "conv", _frontier(engine))
    remainder = engine._generated_context_carrier_remainder(view[0])
    carrier = {"role": "user", "content": view[0]["content"][:-len(remainder)] + "unread host row"}
    assert engine._generated_context_carrier_remainder(carrier) == "unread host row"
    assert engine._get_store_ids_for_messages([carrier]) == []
    engine._current_compress_store_ids_by_message_id = engine._get_store_id_map_for_messages(view)
    view = [carrier, *view[1:]]

    match = engine._match_prepared_leaf(
        view, view, [], set(), engine.threshold_tokens + 1,
        tokens.count_messages_tokens(view),
    )

    assert match is None
    assert _isolated.counters["rejected_host_unit_split"] > 0
    assert _isolated.counters["promoted"] == 0


def test_rotate_restart_full_transcript_never_promotes_unread_start(
    tmp_path, provider, _isolated
):
    engine = _engine(tmp_path)
    try:
        view = _crossing_view(engine)
        engine.ingest(view)
        assert engine.rotate_active_session(apply=True)["ok"]
    finally:
        engine.shutdown()

    engine = _engine(tmp_path)
    try:
        for turn in range(3):
            view = view + [
                {"role": "user", "content": f"[N{turn}] new user{PAD}"},
                {"role": "assistant", "content": f"new reply N{turn}{PAD}"},
            ]
            _turn_end(engine, view)
        assert leaf_prep.ready_chain(engine._store.db_path, "conv", _frontier(engine))

        result = engine.compress(view)

        assert not _isolated.counters.get("promoted")
        output_contents = {message.get("content") for message in result}
        assert all(
            message["content"] in output_contents
            for message in view
            if str(message.get("content", "")).startswith("[T")
        )
    finally:
        engine.shutdown()


# 3. each reject reason -------------------------------------------------------------------------------------------


def test_row_content_update_rejects_with_coverage_digest_mismatch(
    engine, provider, _isolated
):
    view = _crossing_view(engine)
    chain = _prepared(engine, _isolated, view)
    changed = chain[0].covered_store_ids[1]
    engine._store.connection.execute(
        "UPDATE messages SET content = 'rewritten in place' WHERE store_id = ?",
        (changed,),
    )
    engine._store.connection.commit()
    view = [dict(m) for m in view]
    view[changed - 1]["content"] = (
        "rewritten in place"  # store ids follow the view (system prompt = 1)
    )

    engine.compress(view, current_tokens=tokens.count_messages_tokens(view))

    assert _isolated.counters["rejected_coverage_digest_mismatch"] == len(chain)
    assert _isolated.counters["promoted"] == 0
    assert provider.on("MainThread") >= 1  # the foreground path ran instead
    assert all(
        changed not in ids or ids != list(chain[0].covered_store_ids)
        for ids in _leaf_sources(engine)
    )


def test_summary_model_change_rejects_with_route_fingerprint_mismatch(
    engine, provider, _isolated
):
    view = _crossing_view(engine)
    chain = _prepared(engine, _isolated, view)
    engine._config.summary_model = "another-summariser"

    engine.compress(view, current_tokens=tokens.count_messages_tokens(view))

    assert _isolated.counters["rejected_route_fingerprint_mismatch"] == len(chain)
    assert _isolated.counters["promoted"] == 0


def test_foreground_compaction_in_between_rejects_with_frontier_moved(
    engine, provider, _isolated
):
    view = _crossing_view(engine)
    chain = _prepared(engine, _isolated, view)
    engine._config.background_leaf_prep_enabled = False
    result = engine.compress(
        view, current_tokens=tokens.count_messages_tokens(view)
    )  # the foreground wins
    engine._config.background_leaf_prep_enabled = True
    assert _frontier(engine) > 0

    engine.compress(result + view[-2:], current_tokens=engine.threshold_tokens + 1)

    assert _isolated.counters["rejected_frontier_moved"] == len(chain)
    assert _isolated.counters["promoted"] == 0


def test_session_rebind_rejects_with_binding_changed(engine, provider, _isolated):
    view = _crossing_view(engine)
    chain = _prepared(engine, _isolated, view)
    engine._session_id = "S-next"  # the leaves were prepared for S
    engine._current_compress_store_ids_by_message_id = {}

    match = engine._match_prepared_leaf(
        view[1:],
        view,
        [],
        set(),
        engine.threshold_tokens + 1,
        tokens.count_messages_tokens(view),
    )

    assert match is None
    assert _isolated.counters["rejected_binding_changed"] == len(chain)


def _emission_view(turns: int) -> list[dict]:
    """Turns whose assistant emission is stored as two rows (a short text, then its tool call) and a result."""
    rows = [{"role": "system", "content": "system prompt"}]
    for i in range(turns):
        call = {
            "id": f"C{i}",
            "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }
        rows += [
            {"role": "user", "content": f"[T{i}] user turn{PAD}"},
            {"role": "assistant", "content": f"notes on T{i}"},
            {
                "role": "assistant",
                "content": f"Read the fixture for C{i}.",
                "tool_calls": [call],
            },
            {"role": "tool", "tool_call_id": f"C{i}", "content": f"result {i}{PAD}"},
        ]
    return rows


def _emission_crossing(engine) -> list[dict]:
    turns = 1
    while (
        tokens.count_messages_tokens(_emission_view(turns)) <= engine.threshold_tokens
    ):
        turns += 1
    return _emission_view(turns)


def _merged(engine, view) -> list[dict]:
    """The host's list once LCM's assembly merged each emission's adjacent assistant rows into one message."""
    host = engine._sanitize_active_context_messages([dict(m) for m in view])
    assert len(host) < len(view) and any(
        m.get("tool_calls") and "notes on" in str(m.get("content")) for m in host
    )
    return host


def _stored_count(engine) -> int:
    return engine._store.connection.execute("SELECT COUNT(*) FROM messages").fetchone()[
        0
    ]


def test_a_merged_assistant_emission_inside_a_prepared_leaf_is_promoted(
    engine, provider, _isolated
):
    view = _emission_crossing(engine)
    _turn_end(engine, view)
    chain = leaf_prep.ready_chain(engine._store.db_path, "conv", 0)
    stored = _stored_count(engine)

    engine.compress(
        _merged(engine, view), current_tokens=tokens.count_messages_tokens(view)
    )

    assert _stored_count(engine) == stored  # the merged message is no new row
    assert _isolated.counters["promoted"] >= 1
    leaves = _leaf_sources(engine)
    assert leaves[0] == list(chain[0].covered_store_ids)
    roles = dict(
        engine._store.connection.execute(
            "SELECT store_id, role FROM messages WHERE session_id = 'S'"
        ).fetchall()
    )
    assert any(
        roles[a] == roles[a + 1] == "assistant" for a in leaves[0] if a + 1 in leaves[0]
    )  # the emission's text row and its tool-call row, both covered
    flat = [store_id for ids in leaves for store_id in ids]
    assert len(flat) == len(set(flat))


def test_a_prepared_chunk_ending_inside_an_emission_is_rejected_never_published(
    engine, provider, _isolated, monkeypatch
):
    # A worker that ends chunks only inside an emission (between its text row and its tool-call row).
    monkeypatch.setattr(
        leaf_prep,
        "host_unit_boundary",
        lambda previous, following: bool(
            previous
            and following
            and previous.get("role") == following.get("role") == "assistant"
        ),
    )
    view = _emission_crossing(engine)
    _turn_end(engine, view)
    chain = leaf_prep.ready_chain(engine._store.db_path, "conv", 0)
    assert chain

    engine.compress(
        _merged(engine, view), current_tokens=tokens.count_messages_tokens(view)
    )

    assert _isolated.counters["rejected_host_unit_split"] == len(chain)
    assert _isolated.counters["promoted"] == 0
    assert provider.on("MainThread") >= 1  # the leaf path summarised its own chunks
    published = _leaf_sources(engine)
    assert all(
        ids != list(leaf.covered_store_ids) for leaf in chain for ids in published
    )


def test_a_chain_whose_first_row_is_glued_into_the_summary_carrier_is_promoted(
    engine, provider, _isolated
):
    view = _crossing_view(engine)
    enabled, engine._config.background_leaf_prep_enabled = (
        engine._config.background_leaf_prep_enabled,
        False,
    )
    result = engine.compress(view, current_tokens=tokens.count_messages_tokens(view))
    engine._config.background_leaf_prep_enabled = (
        enabled  # off again for the red receipt
    )
    longer = result + _view(40)[len(view) :]
    _turn_end(engine, longer, tokens_now=engine.threshold_tokens - 1_000)
    chain = leaf_prep.ready_chain(engine._store.db_path, "conv", _frontier(engine))
    at = next(
        i
        for i, m in enumerate(longer)
        if str(m.get("content")).startswith("[Recent Summary")
    )
    glued = longer[at + 1]
    assert glued["role"] == "user"
    # The host repairs role alternation: the summary and the next user row become one message (the carrier).
    carrier = {
        "role": "user",
        "content": f"{longer[at]['content']}\n\n{glued['content']}",
    }
    host = longer[:at] + [carrier] + longer[at + 2 :]
    assert engine._generated_context_carrier_remainder(carrier) == glued["content"]
    stored = _stored_count(engine)

    engine.compress(host, current_tokens=engine.threshold_tokens + 1)

    assert _stored_count(engine) == stored
    assert _isolated.counters["promoted"] >= 1
    first = chain[0].covered_store_ids
    assert list(first) in _leaf_sources(engine)  # the glued row opens the promoted leaf
    flat = [store_id for ids in _leaf_sources(engine) for store_id in ids]
    assert len(flat) == len(set(flat))


def test_cache_bound_evicts_the_least_recently_used_conversation(
    tmp_path, provider, _isolated, monkeypatch
):
    monkeypatch.setattr(leaf_prep, "_PER_PROCESS", 2)
    first = _engine(tmp_path)
    second = _engine(tmp_path / "b", database_path=str(tmp_path / "b.db"))
    try:
        _turn_end(first, _crossing_view(first))
        kept = _isolated.status()["cached_leaves"]
        assert kept >= 2
        _turn_end(second, _crossing_view(second))
        assert _isolated.counters["rejected_stale_evicted"] == kept
        assert leaf_prep.ready_chain(first._store.db_path, "conv", 0) == []
        assert leaf_prep.ready_chain(second._store.db_path, "conv", 0)
    finally:
        first.shutdown()
        second.shutdown()


# 4. no double publication ----------------------------------------------------------------------------------------


def test_rows_already_claimed_are_rejected_by_the_fence_never_published_twice(
    engine, provider, _isolated
):
    view = _crossing_view(engine)
    chain = _prepared(engine, _isolated, view)
    claimed = list(chain[0].covered_store_ids)
    # A competing publication claims the first leaf's rows; the lifecycle frontier has not moved yet.
    engine._dag.add_node(
        SummaryNode(
            session_id="S",
            depth=0,
            summary="competing leaf",
            token_count=3,
            source_token_count=100,
            source_ids=claimed,
            source_type="messages",
        )
    )

    engine.compress(view, current_tokens=tokens.count_messages_tokens(view))

    assert _isolated.counters["rejected_fence_conflict"] == len(chain)
    assert _isolated.counters["promoted"] == 0
    for store_id in claimed:
        assert sum(store_id in ids for ids in _leaf_sources(engine)) == 1


def test_promotion_then_foreground_never_covers_a_row_twice(
    engine, provider, _isolated
):
    view = _crossing_view(engine)
    _prepared(engine, _isolated, view)
    result = engine.compress(view, current_tokens=tokens.count_messages_tokens(view))
    longer = result + _view(40)[len(view) :]
    engine._config.background_leaf_prep_enabled = False
    engine.compress(longer, current_tokens=engine.threshold_tokens + 1)

    sources = _leaf_sources(engine)
    flat = [store_id for ids in sources for store_id in ids]
    assert len(flat) == len(set(flat)) and _isolated.counters["promoted"] >= 4


# 5. budget isolation ---------------------------------------------------------------------------------------------


def test_worker_failures_never_open_the_live_breaker_key(engine, provider, _isolated):
    provider.result = RuntimeError("provider down")
    _turn_end(engine, _crossing_view(engine))
    breaker = engine._summary_circuit_breaker
    assert provider.on("lcm-leaf-prep") >= 2
    assert not breaker.allows(leaf_prep.ROUTE_KEY_PREFIX)  # the worker's own key opened
    assert engine._summary_route_available()  # the live route did not
    assert _isolated.counters["prepared"] == 0


def test_worker_skips_while_the_live_route_is_open(engine, provider, _isolated):
    engine._summary_circuit_breaker.record_config_error("")
    view = _crossing_view(engine)
    _turn_end(engine, view)
    assert provider.calls == [] and _isolated.counters["prepared"] == 0
    engine._summary_circuit_breaker.record_success(
        ""
    )  # control: the route closes, the next turn prepares
    _turn_end(engine, view)
    assert _isolated.counters["prepared"] >= 1


@pytest.mark.parametrize("used,prepares", [(2, True), (3, False)])
def test_worker_needs_two_free_foreground_slots_and_never_records_into_them(
    engine, provider, _isolated, used, prepares
):
    engine._summary_spend_guard = SummarySpendGuard(max_calls=4, window_seconds=600.0)
    for _ in range(used):
        engine._summary_spend_guard.record_call()
    _turn_end(engine, _crossing_view(engine))
    assert bool(_isolated.counters["prepared"]) is prepares
    assert len(engine._summary_spend_guard._calls) == used


def test_worker_never_touches_foreground_estimates_or_leaf_provenance(
    engine, provider, _isolated
):
    engine._last_leaf_summary_model, engine._last_leaf_content_filter_error = (
        "sentinel",
        "sentinel",
    )
    engine._last_leaf_level_3_verbatim = "sentinel"
    _turn_end(engine, _crossing_view(engine))
    assert _isolated.counters["prepared"] >= 1
    assert (
        engine._foreground_estimates.calls == {}
        and engine._foreground_estimates.finalize == []
    )
    assert (
        engine._last_leaf_summary_model,
        engine._last_leaf_content_filter_error,
        engine._last_leaf_level_3_verbatim,
    ) == ("sentinel", "sentinel", "sentinel")
    assert engine._foreground_budget is None


def test_worker_has_its_own_spend_guard(engine, provider, _isolated):
    engine._config.background_leaf_prep_max_calls = 2
    _turn_end(engine, _crossing_view(engine))
    assert (
        provider.on("lcm-leaf-prep") == 2 and _isolated.counters["prepared"] == 2
    )  # then the guard is spent
    assert (
        engine._summary_spend_guard.remaining()
        == engine._config.summary_spend_max_calls
    )


def test_host_single_aux_slot_turns_the_worker_off(
    tmp_path, engine, provider, _isolated
):
    (tmp_path / "config.yaml").write_text(
        "auxiliary:\n  compression:\n    max_concurrency: 1\n"
    )
    view = _crossing_view(engine)
    _turn_end(engine, view)
    assert provider.calls == [] and _isolated.counters["prepared"] == 0
    (tmp_path / "config.yaml").write_text(
        "auxiliary:\n  compression:\n    max_concurrency: 2\n"
    )  # control
    _turn_end(engine, view)
    assert _isolated.counters["prepared"] >= 1


# 6. concurrency --------------------------------------------------------------------------------------------------


def _foreground_run(tmp_path, view):
    control = _engine(
        tmp_path / "control",
        database_path=str(tmp_path / "control.db"),
        background_leaf_prep_enabled=False,
    )
    try:
        control.ingest(view)
        result = control.compress(
            view, current_tokens=tokens.count_messages_tokens(view)
        )
        return (
            result,
            _leaf_sources(control),
            control._last_compress_leaves,
            dict(control._foreground_estimates.calls),
        )
    finally:
        control.shutdown()


def test_worker_during_foreground_compress_changes_neither_result_nor_budget(
    tmp_path, engine, provider, _isolated
):
    view = _crossing_view(engine)
    control = _foreground_run(tmp_path, view)
    started, release = threading.Event(), threading.Event()

    def hold_worker():
        if threading.current_thread().name == "lcm-leaf-prep":
            started.set()
            assert release.wait(10)

    provider.hook = hold_worker
    engine.ingest(view)
    engine.last_prompt_tokens = tokens.count_messages_tokens(view)
    engine._prepare_leaves_after_turn(view)
    assert started.wait(10)
    try:
        result = engine.compress(
            view, current_tokens=tokens.count_messages_tokens(view)
        )
    finally:
        release.set()
    assert _isolated.wait_idle(10)

    assert result == control[0] and _leaf_sources(engine) == control[1]
    assert engine._last_compress_leaves == control[2]
    assert {
        key: len(value) for key, value in engine._foreground_estimates.calls.items()
    } == {key: len(value) for key, value in control[3].items()}
    assert (
        _isolated.counters["prepared"] == 0
    )  # its frontier moved during the call: discarded


def test_at_most_one_worker_model_call_in_flight(engine, provider, _isolated):
    assert _isolated._call_slot.acquire(blocking=False)
    view = _crossing_view(engine)
    try:
        _turn_end(engine, view)
    finally:
        _isolated._call_slot.release()
    assert provider.calls == [] and _isolated.counters["prepared"] == 0
    _turn_end(engine, view)  # control: the slot is free
    assert _isolated.counters["prepared"] >= 1


# 7. context ------------------------------------------------------------------------------------------------------

_SCOPE = contextvars.ContextVar("lcm_787_scope", default="unscoped")


def test_worker_job_runs_in_the_enqueue_time_context(engine, provider, _isolated):
    seen = []
    provider.hook = lambda: seen.append((threading.current_thread().name, _SCOPE.get()))
    token = _SCOPE.set("profile-a")
    try:
        _turn_end(engine, _crossing_view(engine))
    finally:
        _SCOPE.reset(token)
    assert seen and set(seen) == {("lcm-leaf-prep", "profile-a")}


def test_post_llm_call_hook_enqueues_after_ingest_and_skips_interrupted_turns(
    tmp_path, monkeypatch, provider, _isolated
):
    monkeypatch.setenv("LCM_BACKGROUND_LEAF_PREP_ENABLED", "1")
    _install_host(monkeypatch, tmp_path)
    module = _load_plugin("hermes_lcm_787_hook")
    jobs = []
    hook_prep = importlib.import_module(
        f"{module.__name__}.leaf_prep"
    )  # this load's own module state
    monkeypatch.setattr(hook_prep._LEAF_PREP, "enqueue", jobs.append)
    ctx = _Ctx(_Manager())
    module.register(ctx)
    engine = ctx.offered
    try:
        engine.on_session_start(
            "S", platform="telegram", conversation_id="conv", context_length=WINDOW
        )
        history = _crossing_view(engine)
        engine.last_prompt_tokens = tokens.count_messages_tokens(history)
        hook = ctx.hooks["post_llm_call"][0]
        hook(
            session_id="S",
            conversation_id="conv",
            platform="telegram",
            conversation_history=history,
            interrupted=True,
        )
        assert jobs == []
        hook(
            session_id="S",
            conversation_id="conv",
            platform="telegram",
            conversation_history=history,
        )
        assert [
            (job.session_id, job.conversation_id, job.frontier) for job in jobs
        ] == [("S", "conv", 0)]
        assert engine._store.get_session_count("S") == len(history)
    finally:
        engine.shutdown()


# 8. lifetime -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ending", ["shutdown", "rebind"])
def test_shutdown_and_rebind_clear_the_cache_and_discard_a_late_job(
    tmp_path, provider, _isolated, ending
):
    engine = _engine(tmp_path)
    view = _crossing_view(engine)
    _turn_end(engine, view)
    assert _isolated.status()["cached_leaves"] >= 4
    started, release = threading.Event(), threading.Event()

    def hold_worker():
        started.set()
        assert release.wait(10)

    provider.hook = hold_worker
    engine.ingest(
        view + _view(60)[len(view) :]
    )  # more rows: the next job prepares further
    engine.last_prompt_tokens = WINDOW  # far above L: the chain must grow
    engine._prepare_leaves_after_turn(view)
    assert started.wait(10)
    try:
        if ending == "shutdown":
            engine.shutdown()
        else:
            engine._rebind_storage_for_home(str(tmp_path / "other-home"))
    finally:
        release.set()
    assert _isolated.wait_idle(10)
    status = _isolated.status()
    assert status["cached_leaves"] == 0 and status["queued_jobs"] == 0
    if ending == "rebind":
        engine.shutdown()


# 9. levels -------------------------------------------------------------------------------------------------------


def test_levels_turn_off_on_small_windows_and_low_thresholds():
    assert leaf_prep.levels(45_000, 60_000, 2_000) == (36_000, 39_000)
    assert leaf_prep.levels(24_000, 32_000, 24_000) is None  # L <= 0 on a 32K window
    assert (
        leaf_prep.levels(24_000, 60_000, 2_000) is None
    )  # P = 0.25 x window <= 0.30 x window
    assert leaf_prep.levels(204_000, 272_000, 24_000) == (163_200, 156_000)


def test_small_window_prepares_nothing(tmp_path, provider, _isolated):
    engine = _engine(tmp_path, window=32_000, leaf_chunk_tokens=24_000)
    control = _engine(
        tmp_path / "c", database_path=str(tmp_path / "c.db"), leaf_chunk_tokens=CHUNK
    )
    try:
        _turn_end(engine, _crossing_view(engine))
        assert provider.calls == [] and _isolated.counters["prepared"] == 0
        _turn_end(
            control, _crossing_view(control)
        )  # control: the 60K window with a 2K chunk prepares
        assert _isolated.counters["prepared"] >= 1
    finally:
        engine.shutdown()
        control.shutdown()


def test_never_prepares_inside_the_fresh_tail(tmp_path, provider, _isolated):
    engine = _engine(tmp_path, fresh_tail_count=20)
    try:
        view = _crossing_view(engine)
        _turn_end(engine, view)
        chain = leaf_prep.ready_chain(engine._store.db_path, "conv", 0)
        assert chain
        tail_ids = set(
            range(len(view) - 20 + 1, len(view) + 1)
        )  # store ids follow the view
        assert not tail_ids & {
            store_id for leaf in chain for store_id in leaf.covered_store_ids
        }
    finally:
        engine.shutdown()


def test_sweep_chunk_matches_the_foreground_choice(engine):
    for raw in (0, 5_000, 50_000):
        assert leaf_prep.working_leaf_chunk_tokens(
            engine._config, engine.context_length, raw
        ) == engine._working_leaf_chunk_tokens(raw)
    engine._config.dynamic_leaf_chunk_enabled = True
    for raw in (0, 5_000, 50_000):
        assert leaf_prep.working_leaf_chunk_tokens(
            engine._config, engine.context_length, raw
        ) == engine._working_leaf_chunk_tokens(raw)


def test_spend_guard_remaining_records_nothing():
    guard = SummarySpendGuard(max_calls=3, window_seconds=600.0, backoff_seconds=60.0)
    assert guard.remaining() == 3
    guard.record_call()
    assert guard.remaining() == 2 and len(guard._calls) == 1
    guard.record_call()
    guard.record_call()  # fills the window: backoff
    assert guard.remaining() == 0
    assert SummarySpendGuard(max_calls=0).remaining() == float("inf")


def test_preparation_serializer_writes_no_payload_file(tmp_path):
    from hermes_lcm.engine import LeafPrepUnavailable, _serialize_leaf_input

    config = LCMConfig(
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=100,
        large_output_externalization_path=str(tmp_path / "payloads"),
    )
    rows = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "large tool output " * 40},
    ]

    def files():
        return (
            sorted(p.name for p in (tmp_path / "payloads").glob("*.json"))
            if (tmp_path / "payloads").exists()
            else []
        )

    with pytest.raises(LeafPrepUnavailable):
        _serialize_leaf_input(
            rows,
            config=config,
            hermes_home=str(tmp_path),
            session_id="S",
            externalize_write=False,
        )
    assert files() == []
    text, _clipped = _serialize_leaf_input(
        rows, config=config, hermes_home=str(tmp_path), session_id="S"
    )
    written = files()
    assert len(written) == 1  # the foreground writes the payload, as before
    again, _clipped = _serialize_leaf_input(
        rows,
        config=config,
        hermes_home=str(tmp_path),
        session_id="S",
        externalize_write=False,
    )
    assert (
        again == text and files() == written
    )  # on disk now: the same input, no new file


def test_a_newer_turn_supersedes_the_queued_job_of_its_conversation(
    engine, provider, _isolated
):
    started, release = threading.Event(), threading.Event()

    def hold_worker():
        started.set()
        assert release.wait(10)

    provider.hook = hold_worker
    other = _engine(
        engine._store.db_path.parent / "o",
        database_path=str(engine._store.db_path.parent / "o.db"),
    )
    try:
        _view_other = _crossing_view(other)
        other.ingest(_view_other)
        other.last_prompt_tokens = tokens.count_messages_tokens(_view_other)
        other._prepare_leaves_after_turn(_view_other)  # occupies the worker
        assert started.wait(10)
        view = _crossing_view(engine)
        engine.ingest(view)
        engine.last_prompt_tokens = tokens.count_messages_tokens(view)
        engine._prepare_leaves_after_turn(view)
        engine._prepare_leaves_after_turn(view)
        assert (
            _isolated.status()["queued_jobs"] == 1
            and _isolated.counters["jobs_superseded"] == 1
        )
    finally:
        release.set()
        assert _isolated.wait_idle(10)
        other.shutdown()


# Review round 1 (PR #1142) ------------------------------------------------------------------------------------------


def test_r1_f1_a_durable_dependent_reply_is_never_published_in_a_prepared_summary(
    engine, provider, _isolated
):
    view = _crossing_view(engine)
    marker = (
        "reply to T6 alpha"  # T6's reply only (T60 would read "reply to T60 alpha")
    )
    reply = 1 + next(
        i for i, m in enumerate(view) if str(m.get("content")).startswith(marker)
    )  # store ids follow the view (system prompt = 1)
    engine.ingest(view)
    assert engine._store.get(reply)["content"].startswith(marker)
    # No ignore pattern is configured; the reply stays marked as an ignored-dependent reply.
    engine._write_generated_ignored_dependent_reply_records(
        [
            {
                "store": engine._ignored_dependent_reply_store_fingerprint(
                    {"role": "assistant", "store_id": reply}
                )
            }
        ]
    )
    provider.result = lambda prompt: (
        SUMMARY + (" [dependent reply text]" if marker in prompt else "")
    )
    _turn_end(engine, view)
    assert _isolated.counters["prepared"] >= 1  # leaves before the reply are prepared

    engine.compress(view, current_tokens=tokens.count_messages_tokens(view))

    summaries = [
        node.summary for node in engine._dag.get_session_nodes("S", limit=100_000)
    ]
    assert summaries and not any("[dependent reply text]" in s for s in summaries)


def test_r1_f1_promotion_needs_the_summary_source_as_well_as_the_consumption(
    engine, provider, _isolated, monkeypatch
):
    # A worker without the stop: its leaf summarises the marked reply the leaf loop consumes but never reads.
    monkeypatch.setattr(
        leaf_prep,
        "excluded_from_summary_input",
        lambda *args: False,
        raising=False,
    )
    view = _crossing_view(engine)
    marker = "reply to T1 alpha"
    reply = 1 + next(
        i for i, m in enumerate(view) if str(m.get("content")).startswith(marker)
    )
    engine.ingest(view)
    engine._write_generated_ignored_dependent_reply_records(
        [
            {
                "store": engine._ignored_dependent_reply_store_fingerprint(
                    {"role": "assistant", "store_id": reply}
                )
            }
        ]
    )
    provider.result = lambda prompt: (
        SUMMARY + (" [dependent reply text]" if marker in prompt else "")
    )
    _turn_end(engine, view)
    chain = leaf_prep.ready_chain(engine._store.db_path, "conv", 0)
    holding = next(i for i, leaf in enumerate(chain) if reply in leaf.covered_store_ids)

    engine.compress(view, current_tokens=tokens.count_messages_tokens(view))

    # The leaves before it are promoted; the one holding the reply ends the chain.
    assert _isolated.counters["promoted"] == holding
    assert _isolated.counters["rejected_host_unit_split"] == len(chain) - holding
    summaries = [
        node.summary for node in engine._dag.get_session_nodes("S", limit=100_000)
    ]
    assert summaries and not any("[dependent reply text]" in s for s in summaries)


def test_r1_f2_each_profile_keeps_its_worker_spend_guard_across_interleaved_jobs(
    tmp_path, provider, _isolated
):
    first = _engine(tmp_path, background_leaf_prep_max_calls=2)
    second = _engine(
        tmp_path / "b",
        database_path=str(tmp_path / "b.db"),
        background_leaf_prep_max_calls=3,
    )
    try:
        view = _crossing_view(first)
        _turn_end(first, view)
        assert provider.on("lcm-leaf-prep") == 2  # the first profile's window is spent
        _turn_end(second, _crossing_view(second))  # another profile, another guard
        before = provider.on("lcm-leaf-prep")
        assert before > 2
        first.ingest(view + _view(60)[len(view) :])
        first.last_prompt_tokens = WINDOW  # far above L: the chain would grow
        first._prepare_leaves_after_turn(view)
        assert _isolated.wait_idle(10)
        assert (
            provider.on("lcm-leaf-prep") == before
        )  # still no slot for the first profile
    finally:
        first.shutdown()
        second.shutdown()


def test_r1_f3_the_worker_route_check_leaves_the_live_breaker_untouched(
    engine, provider, _isolated
):
    live = engine._summary_circuit_breaker
    key = live._key(engine._config.summary_model)
    live._open_until[key] = (
        time.monotonic() - 1
    )  # expired: only the foreground prunes it
    _turn_end(engine, _crossing_view(engine))
    assert _isolated.counters["prepared"] >= 1
    assert key in live._open_until


def test_r1_f4_the_preparation_serializer_creates_no_payload_directory(tmp_path):
    from hermes_lcm.engine import LeafPrepUnavailable, _serialize_leaf_input

    storage = tmp_path / "payloads"
    config = LCMConfig(
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=100,
        large_output_externalization_path=str(storage),
    )
    rows = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "large tool output " * 40},
    ]
    with pytest.raises(LeafPrepUnavailable):
        _serialize_leaf_input(
            rows,
            config=config,
            hermes_home=str(tmp_path),
            session_id="S",
            externalize_write=False,
        )
    assert not storage.exists()


def test_r1_f5_off_publication_records_progress_after_the_host_list_trim(
    tmp_path, provider, _isolated
):
    from tests.harness import make_engine

    engine = make_engine(
        tmp_path, window=WINDOW, session_id="S", platform="telegram", conversation_id="conv",
        background_leaf_prep_enabled=False, background_leaf_prep_max_calls=64,
        fresh_tail_count=2, leaf_chunk_tokens=CHUNK, context_threshold=0.75,
        threshold_full_sweep_enabled=True, large_output_externalization_enabled=False,
        large_output_active_replay_stubbing_enabled=False, temporal_rollups_enabled=False,
        l3_truncate_tokens=2, incremental_max_depth=0,
    )
    seen: dict[str, int] = {}

    def lines(frame, event, arg):
        if event == "line":
            local = frame.f_locals
            if frame.f_code.co_name == "_advance_after_leaf":
                budget, working = local["state"].budget, local["state"].working_messages
            else:
                budget, working = local.get("budget"), local.get("working_messages")
            if (
                "published" not in seen
                and engine._last_compacted_store_id
                and working is not None
            ):
                seen["published"] = len(
                    working
                )  # the host list when the frontier marker moved
            if "progress" not in seen and getattr(budget, "leaves", 0) >= 1:
                seen["progress"] = len(
                    working
                )  # the host list when the leaf counted as progress
        return lines

    def calls(frame, event, arg):
        return lines if frame.f_code.co_name in {"_compress_impl", "_advance_after_leaf"} else None

    try:
        view = _crossing_view(engine)
        sys.settrace(calls)
        try:
            engine.compress(view, current_tokens=tokens.count_messages_tokens(view))
        finally:
            sys.settrace(None)
        # Base order (09e32580): marker, counters, rollups, host-list trim, then progress and holds.
        assert seen["progress"] < seen["published"]
    finally:
        engine.shutdown()


# Review round 2 (PR #1142) ------------------------------------------------------------------------------------------


def test_r2_a_the_off_stub_trial_keeps_base_storage_resolution(tmp_path, provider):
    blocker = tmp_path / "blocker"
    blocker.write_text("a regular file: no directory can be created under it")
    engine = _engine(
        tmp_path,
        background_leaf_prep_enabled=False,
        large_output_externalization_enabled=True,
        large_output_active_replay_stubbing_enabled=True,
        large_output_externalization_path=str(blocker / "payloads"),
    )
    try:
        message = {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "large tool output " * 400,
        }
        # The #671/#726 no-write trial: storage cannot be created, so no stub (as at 09e32580).
        assert (
            engine._maybe_stub_active_tool_result(
                message, is_recovery_tool_result=False, threshold_tokens=1, write=False
            )
            is None
        )
    finally:
        engine.shutdown()


def _guard_job(name: str, max_calls: int = 2):
    from types import SimpleNamespace

    return SimpleNamespace(
        config=LCMConfig(background_leaf_prep_max_calls=max_calls),
        hermes_home=f"home-{name}",
        db_path=f"db-{name}",
    )


def test_r2_b_a_guard_with_a_live_obligation_is_never_evicted(_isolated):
    tripped = _isolated._guard(_guard_job("A"))
    for _ in range(2):
        tripped.record_call()
    assert tripped.remaining() == 0  # A is in its backoff
    for index in range(leaf_prep._PER_PROCESS):  # 64 other profiles arrive
        assert _isolated._guard(_guard_job(f"other-{index}")) is not None

    assert _isolated._guard(_guard_job("A")) is tripped and tripped.remaining() == 0
    assert len(_isolated._spend_guards) <= leaf_prep._PER_PROCESS


def test_r2_b_a_full_table_of_live_guards_skips_the_new_profile(_isolated):
    for index in range(leaf_prep._PER_PROCESS):
        _isolated._guard(
            _guard_job(f"busy-{index}")
        ).record_call()  # a call inside each window

    assert _isolated._guard(_guard_job("new")) is None
    assert _isolated.counters["prep_guard_full"] == 1
    assert len(_isolated._spend_guards) == leaf_prep._PER_PROCESS


# Review round 5: preserve the stop-bar evidence without changing opt-in policy.


@pytest.mark.xfail(
    strict=True,
    reason="Accepted opt-in follow-up: D-4 permits preparation-time focus at promotion",
)
def test_r5_changed_focus_would_require_a_new_foreground_summary(
    engine, provider, _isolated, tmp_path
):
    old_focus, new_focus = "orchard-budget", "marina-layout"
    old_summary = f"Earlier {old_focus}.\nExpand for details about: turns"
    new_summary = f"Earlier {new_focus}.\nExpand for details about: turns"
    provider.result = lambda prompt: (
        new_summary if f"Primary focus: {new_focus}" in prompt else old_summary
    )
    view = _crossing_view(engine)
    view[-2]["content"] = old_focus + PAD
    _turn_end(engine, view, tokens_now=engine.threshold_tokens + CHUNK)
    chain = leaf_prep.ready_chain(engine._store.db_path, "conv", 0)
    assert chain and chain[0].summary == old_summary

    foreground = _engine(tmp_path / "foreground", background_leaf_prep_enabled=False)
    try:
        foreground.compress(
            view, current_tokens=foreground.threshold_tokens + CHUNK,
            focus_topic=new_focus,
        )
        assert any(
            node.summary == new_summary
            for node in foreground._dag.get_session_nodes("S", limit=100_000)
        )
    finally:
        foreground.shutdown()

    engine.compress(
        view, current_tokens=engine.threshold_tokens + CHUNK, focus_topic=new_focus
    )
    assert _isolated.counters["promoted"] > 0
    published = [
        node for node in engine._dag.get_session_nodes("S", limit=100_000)
        if node.source_ids == list(chain[0].covered_store_ids)
    ]
    assert len(published) == 1
    # Expected RED: the current policy publishes old_summary despite the new focus.
    assert published[0].summary == new_summary


@pytest.mark.parametrize("field", ["role", "tool_call_id", "tool_calls", "tool_name"])
def test_r5_reconcile_metadata_change_does_not_rewrite_prepared_rows(
    engine, provider, _isolated, field
):
    from copy import deepcopy

    view = _emission_crossing(engine)
    _turn_end(engine, view)
    chain = leaf_prep.ready_chain(engine._store.db_path, "conv", 0)
    assert chain
    before = engine._store.get_session_messages("S", limit=100_000)
    original_ids = [row["store_id"] for row in before]
    before_map = engine._get_store_id_map_for_messages(view)
    changed = deepcopy(view)
    index = 3 if field == "tool_calls" else 4
    original_id = before_map[id(view[index])]
    assert original_id in chain[0].covered_store_ids
    if field == "role":
        changed[index]["role"] = "assistant"
    elif field == "tool_call_id":
        changed[index]["tool_call_id"] = "corrected-call"
    elif field == "tool_calls":
        changed[index][field][0]["function"]["name"] = "corrected-tool"
    else:
        changed[index]["tool_name"] = "corrected-tool"

    engine.ingest(changed)

    after = engine._store.get_session_messages("S", limit=100_000)
    # Reconciliation retains every original row, including all four metadata fields.
    assert [row for row in after if row["store_id"] in original_ids] == before
    mapping = engine._get_store_id_map_for_messages(changed)
    assert mapping.get(id(changed[index])) != original_id
    # A changed host occurrence cannot consume the old prepared row under its identity.
    assert engine._message_replay_identity(changed[index]) != (
        engine._message_replay_identity(before[index], stored_row=True)
    )
