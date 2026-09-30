"""#669: a rollup stores no level 3 cut, starts no build while the summary route is refused, and records its
circuit results under its own breaker keys.

The rollup builder, the escalation chain and the breaker are the real ones; only the provider helper under
_invoke_summary_llm is stubbed, so the levels, the route keys and the log lines are the production ones."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.escalation as escalation
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.escalation import SummaryCircuitBreaker
from hermes_lcm.rollup_builder import build_day, run_rollup_maintenance
from hermes_lcm.rollup_store import RollupStore
from hermes_lcm.tokens import count_tokens

DAY = date(2026, 7, 15)
SCOPE = "S"
LONG_SOURCE = "source material with durable decisions " * 40
ACCEPTED = "Earlier turns.\nExpand for details about: turns"
NOT_STORED = "LCM rollup not stored: summary at level 3"
SKIPPED = "LCM rollup maintenance skipped"
PAD = " alpha beta gamma delta" * 30


class _Provider:
    """Stands in for _call_llm_for_summary: each call takes the next scripted answer (the last one repeats).
    None is a provider failure; "" is an empty (rejected) result."""

    def __init__(self, *script):
        self.script = list(script) or [ACCEPTED]
        self.calls: list[str] = []

    def __call__(self, prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        answer = self.script[min(len(self.calls), len(self.script) - 1)]
        self.calls.append(model)
        return answer


class _SpyBreaker(SummaryCircuitBreaker):
    """The real breaker, recording every key it is asked about."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events: list[tuple[str, str]] = []

    def allows(self, model, *, now=None):
        self.events.append(("allows", model))
        return super().allows(model, now=now)

    def record_success(self, model):
        self.events.append(("success", model))
        super().record_success(model)

    def record_failure(self, model, *, now=None):
        self.events.append(("failure", model))
        super().record_failure(model, now=now)

    def record_rejection(self, model, *, now=None):
        self.events.append(("rejection", model))
        super().record_rejection(model, now=now)


def _provider(monkeypatch, *script) -> _Provider:
    provider = _Provider(*script)
    monkeypatch.setattr(escalation, "_call_llm_for_summary", provider)
    return provider


def _timestamp(hour: int) -> float:
    return datetime(DAY.year, DAY.month, DAY.day, hour, tzinfo=timezone.utc).timestamp()


def _add_node(dag: SummaryDAG, summary: str, scope: str = SCOPE) -> int:
    return dag.add_node(SummaryNode(
        session_id=scope, depth=0, summary=summary, token_count=count_tokens(summary),
        source_token_count=count_tokens(summary) * 2, source_ids=[1], source_type="messages",
        created_at=_timestamp(18), earliest_at=_timestamp(8), latest_at=_timestamp(22)))


@pytest.fixture
def parts(tmp_path):
    db_path = tmp_path / "rollup-669.db"
    dag = SummaryDAG(db_path)
    store = RollupStore(db_path)
    config = LCMConfig(database_path=str(db_path), rollup_daily_target_tokens=12, rollup_daily_max_tokens=20,
                       rollup_aggregate_max_tokens=30, rollup_builds_per_pass=4)
    try:
        yield store, dag, config
    finally:
        store.close()
        dag.close()


def _engine(tmp_path, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "database_path": str(tmp_path / "lcm.db"),
                "rollup_daily_target_tokens": 12, "rollup_daily_max_tokens": 20, "rollup_builds_per_pass": 4,
                **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    return engine


def _chunk() -> list[dict]:
    return [{"role": "user", "content": f"[T{i}] user turn{PAD}"} for i in range(3)]


def _count(caplog, text: str, level: int | None = None) -> int:
    return sum(text in record.getMessage() and (level is None or record.levelno == level)
               for record in caplog.records)


def _status(store: RollupStore, day: date = DAY) -> str | None:
    row = store.get_rollup("day", day.isoformat(), SCOPE)
    return None if row is None else str(row["status"])


# -- R1: an empty-output route stores no level 3 cut --------------------------------------------------------

def test_r1_empty_output_route_stores_no_rollup_and_keeps_it_pending(parts, monkeypatch, caplog):
    store, dag, config = parts
    _add_node(dag, LONG_SOURCE)
    store.mark_stale_for_day(DAY, SCOPE)
    provider = _provider(monkeypatch, "")
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        assert run_rollup_maintenance(dag, config, SCOPE, circuit_breaker=SummaryCircuitBreaker()) >= 1
    assert provider.calls == ["", ""]  # level 1 and level 2 were asked; level 3 came back
    row = store.get_rollup("day", DAY.isoformat(), SCOPE)
    assert row["status"] == "stale" and not row["summary"]
    assert _count(caplog, NOT_STORED, logging.WARNING) == 1

    # The next scheduled build retries it.
    _provider(monkeypatch, ACCEPTED)
    assert run_rollup_maintenance(dag, config, SCOPE, circuit_breaker=SummaryCircuitBreaker()) >= 1
    row = store.get_rollup("day", DAY.isoformat(), SCOPE)
    assert row["status"] == "ready" and row["summary"] == ACCEPTED


# -- R2: a verbatim level 3 (the whole source) is still stored ----------------------------------------------

def test_r2_verbatim_level_3_is_stored_as_before(parts, monkeypatch, caplog):
    store, dag, config = parts
    node_id = _add_node(dag, "short leaf")
    _provider(monkeypatch, "")
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        built = build_day(store, dag, config, SCOPE, DAY)
    assert built is not None and built["status"] == "ready"
    assert built["summary"] == f"[Summary node {node_id}]\nshort leaf"
    assert _count(caplog, NOT_STORED) == 0


# -- R3: no build while the summary route stop applies ------------------------------------------------------

def test_r3_refused_route_starts_no_build_and_no_provider_call(tmp_path, monkeypatch, caplog):
    engine = _engine(tmp_path)
    provider = _provider(monkeypatch, ACCEPTED)
    store = RollupStore(engine._dag.db_path)
    try:
        _add_node(engine._dag, LONG_SOURCE)
        second = date(2026, 7, 16)
        engine._dag.add_node(SummaryNode(
            session_id=SCOPE, depth=0, summary=LONG_SOURCE, token_count=1, source_token_count=2, source_ids=[1],
            source_type="messages", created_at=_timestamp(18) + 86_400, earliest_at=_timestamp(8) + 86_400,
            latest_at=_timestamp(22) + 86_400))
        store.mark_stale_for_day(DAY, SCOPE)
        store.mark_stale_for_day(second, SCOPE)
        breaker = engine._summary_circuit_breaker
        for _ in range(breaker.failure_threshold):
            breaker.record_failure(engine._config.summary_model)
        assert engine._summary_route_stop_applies(False)
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            started = run_rollup_maintenance(engine._dag, engine._config, SCOPE, circuit_breaker=breaker,
                                             spend_guard=engine._summary_spend_guard)
        assert started == 0 and provider.calls == []
        assert _status(store) == "stale" and _status(store, second) == "stale"
        assert _count(caplog, SKIPPED, logging.INFO) == 1  # at most one per maintenance run
    finally:
        store.close()
        engine.shutdown()


def test_r3_refused_rollup_keys_also_start_no_build(parts, monkeypatch):
    store, dag, config = parts
    _add_node(dag, LONG_SOURCE)
    store.mark_stale_for_day(DAY, SCOPE)
    provider = _provider(monkeypatch, None)
    breaker = SummaryCircuitBreaker()
    assert run_rollup_maintenance(dag, config, SCOPE, circuit_breaker=breaker) >= 1  # two failures open it
    assert breaker.allows(config.summary_model)  # the live route is untouched
    store.mark_stale_for_day(DAY, SCOPE)
    calls = len(provider.calls)
    assert run_rollup_maintenance(dag, config, SCOPE, circuit_breaker=breaker) == 0
    assert len(provider.calls) == calls


# -- R4: a failing rollup route leaves the live circuit closed ----------------------------------------------

def test_r4_failing_rollup_route_leaves_live_circuit_closed(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    store = RollupStore(engine._dag.db_path)
    try:
        _add_node(engine._dag, LONG_SOURCE)
        breaker = engine._summary_circuit_breaker
        provider = _provider(monkeypatch, None)
        for _ in range(breaker.failure_threshold):
            store.mark_stale_for_day(DAY, SCOPE)
            build_day(store, engine._dag, engine._config, SCOPE, DAY, circuit_breaker=breaker)
        assert len(provider.calls) >= breaker.failure_threshold
        assert breaker.allows(engine._config.summary_model)
        assert not engine._summary_route_stop_applies(False)

        live = _provider(monkeypatch, ACCEPTED)
        _chunk_out, _tokens, summary, level, _attempt = engine._summarize_leaf_chunk_with_rescue(_chunk())
        assert live.calls == [engine._config.summary_model]
        assert (summary, level) == (ACCEPTED, 1)
    finally:
        store.close()
        engine.shutdown()


# -- R5: live compaction is unchanged ----------------------------------------------------------------------

def test_r5_live_leaf_calls_route_keys_and_result_are_unchanged(tmp_path, monkeypatch):
    engine = _engine(tmp_path, summary_model="m1", summary_fallback_models=["m2"])
    try:
        breaker = _SpyBreaker()
        engine._summary_circuit_breaker = breaker
        seen_kwargs: list[dict] = []
        real = escalation.summarize_with_escalation

        def spy(*args, **kwargs):
            seen_kwargs.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", spy)
        provider = _provider(monkeypatch, None, ACCEPTED)
        _chunk_out, _tokens, summary, level, _attempt = engine._summarize_leaf_chunk_with_rescue(_chunk())

        assert provider.calls == ["m1", "m2"]
        assert breaker.events == [("allows", "m1"), ("failure", "m1"), ("allows", "m2"), ("success", "m2")]
        assert (summary, level) == (ACCEPTED, 1)
        assert len(seen_kwargs) == 1 and "route_key_prefix" not in seen_kwargs[0]
    finally:
        engine.shutdown()


# -- R6: rollup and live keys do not reset each other -------------------------------------------------------

def test_r6_rollup_success_does_not_reset_live_failures_and_the_reverse(parts, monkeypatch):
    store, dag, config = parts
    _add_node(dag, LONG_SOURCE)
    breaker = SummaryCircuitBreaker(failure_threshold=2)

    # A live failure, then a rollup success, then a live failure: the live route opens.
    breaker.record_failure(config.summary_model)
    _provider(monkeypatch, ACCEPTED)
    assert build_day(store, dag, config, SCOPE, DAY, circuit_breaker=breaker)["status"] == "ready"
    breaker.record_failure(config.summary_model)
    assert not breaker.allows(config.summary_model)

    # The reverse: a rollup failure, then a live success, then a rollup failure: the rollup keys open.
    other = SummaryCircuitBreaker(failure_threshold=3)
    provider = _provider(monkeypatch, None, None, ACCEPTED)
    store.mark_stale_for_day(DAY, SCOPE)
    assert build_day(store, dag, config, SCOPE, DAY, circuit_breaker=other) is None  # L1 + L2 fail: two
    live_summary, live_level = escalation.summarize_with_escalation(
        LONG_SOURCE, source_tokens=count_tokens(LONG_SOURCE), token_budget=20, circuit_breaker=other)
    assert (live_summary, live_level) == (ACCEPTED, 1)
    provider.script = [None]
    store.mark_stale_for_day(DAY, SCOPE)
    calls = len(provider.calls)
    assert build_day(store, dag, config, SCOPE, DAY, circuit_breaker=other) is None
    assert len(provider.calls) == calls + 1  # the third rollup failure opened the rollup keys; L2 was skipped
    assert other.allows(config.summary_model)
