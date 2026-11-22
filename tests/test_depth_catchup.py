"""Admission and publication under the depth target with a controlled clock."""

import pytest

import hermes_lcm.escalation as escalation
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.tokens import count_messages_tokens, count_tokens
from tests.test_depth_target_ratio import (
    SUMMARY,
    WINDOW,
    _leaves,
    _stored_row_hashes,
    _sweep,
    _sweep_engine,
    _target_after,
    _view,
)
from tests.test_depth_target_ratio import _pin_lcm_char_counter as _pin_lcm_char_counter
from tests.test_depth_target_ratio import clock as clock
from tests.test_depth_target_ratio import make_engine as make_engine
from tests.test_issue_671_stub_first_exit import payload_of


class ProviderFailure(Exception):
    status_code = 503


class Provider:
    def __init__(self, clock, engine, script):
        self.clock, self.engine, self.script = clock, engine, iter(script)
        self.calls = []
        self.budget = None

    def __call__(self, prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        assert not self.engine._store.connection.in_transaction
        self.budget = self.engine._foreground_call_budget()
        start = self.clock() - self.budget.t0
        deadline = escalation._summary_call.stream_deadline - self.budget.t0
        self.calls.append((start, timeout, deadline, model))
        wall, value = next(self.script)
        if wall > timeout:
            self.clock.offset += timeout
            raise TimeoutError("provider timed out")
        self.clock.offset += wall
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture
def install_provider(clock, monkeypatch):
    clock._real = lambda: 1000.0
    monkeypatch.setattr(escalation, "_host_stream_deadline_seam", lambda: True)

    def install(engine, *script):
        provider = Provider(clock, engine, script)
        monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
        return provider

    return install


def _case(make_engine, monkeypatch, ratio="depth", *, view=None, **settings):
    view = _view(8) if view is None else view
    if ratio is None:
        monkeypatch.delenv("LCM_DEPTH_TARGET_RATIO", raising=False)
    else:
        monkeypatch.setenv("LCM_DEPTH_TARGET_RATIO", str(_target_after(view, 5) / WINDOW) if ratio == "depth" else ratio)
    config = LCMConfig.from_env()
    engine = _sweep_engine(
        make_engine, view, depth_target_ratio=config.depth_target_ratio,
        foreground_soft_seconds=config.foreground_soft_seconds, foreground_hard_seconds=config.foreground_hard_seconds,
        **settings,
    )
    engine.ingest(view)
    return engine, view


def _compress(engine, view, **kwargs):
    before = _stored_row_hashes(engine)
    result = engine.compress(view, current_tokens=count_messages_tokens(view), **kwargs)
    assert _stored_row_hashes(engine) == before
    return result


def _assert_pass(engine, provider, calls, stop, met, count, *, route="", circuits=None):
    assert provider.calls == [(*call, route) for call in calls]
    status = _sweep(engine)
    assert status["stop_reason"] == stop
    assert status.get("depth_target_met") is met
    assert status["leaf_passes"] == count
    assert (engine._no_progress_hold, engine._sweep_budget_hold_until, engine._hung_summary_strike) == (None, 0.0, None)
    breaker = engine._summary_circuit_breaker
    for model, (state, error) in (circuits or {route: ("closed", None)}).items():
        actual = breaker.route_status([model])
        assert (actual["state"], actual["last_error_class"]) == (state, error)
        assert not breaker._failures.get(breaker._key(model))
    assert not provider.budget.summary_failure


TWO_BASE = [(0, 115, 115), (25, 90, 115)]
FIVE_CATCHUP = [*TWO_BASE, (50, 102, 152), (75, 97.75, 172.75), (100, 76.5, 176.5)]


def test_k1_five_leaves_are_published_in_one_compress(make_engine, monkeypatch, install_provider, clock):
    engine, view = _case(make_engine, monkeypatch)
    provider = install_provider(engine, *((25, SUMMARY),) * 5)

    result = _compress(engine, view)

    _assert_pass(engine, provider, FIVE_CATCHUP, "depth_target_reached", True, 5)
    assert clock.offset == 125 < 210 - 20
    leaves = _leaves(engine)
    assert len(leaves) == 5 and result[-6:] == view[-6:]
    covered = [row for node in leaves for row in node.source_ids]
    assert covered == list(_stored_row_hashes(engine))[1:11]
    assert provider.budget.estimates.calls == {"": [25, 25]}


def _publication_time(engine, monkeypatch, clock):
    publish = engine._publish_leaf_node

    def publish_with_work(*args, **kwargs):
        result = publish(*args, **kwargs)
        clock.offset += 6
        return result

    monkeypatch.setattr(engine, "_publish_leaf_node", publish_with_work)


def test_k2_estimate_must_fit_before_the_catchup_bound(make_engine, monkeypatch, install_provider, clock):
    engine, view = _case(make_engine, monkeypatch)
    _publication_time(engine, monkeypatch, clock)
    provider = install_provider(engine, (60, SUMMARY), (60, SUMMARY))

    _compress(engine, view)

    _assert_pass(engine, provider, [(0, 115, 115), (66, 102, 168)], "time_budget_exhausted", False, 2)
    assert clock.offset == 132 and 190 - clock.offset < provider.budget.estimate("")


def test_k10_estimate_must_fit_the_extension_timeout(make_engine, monkeypatch, install_provider, clock):
    engine, view = _case(make_engine, monkeypatch)
    provider = install_provider(engine, *((60, SUMMARY),) * 3)

    _compress(engine, view)

    _assert_pass(engine, provider, [(0, 115, 115), (60, 102, 162)], "time_budget_exhausted", False, 2)
    assert clock.offset == 120 and len(_leaves(engine)) == 2


@pytest.mark.parametrize("ratio", [None, "depth"])
def test_k3_force_uses_only_the_base_hard_budget(ratio, make_engine, monkeypatch, install_provider):
    engine, view = _case(make_engine, monkeypatch, ratio)
    provider = install_provider(engine, *((25, SUMMARY),) * 4)

    _compress(engine, view, force=True)

    calls = [(start, 115 - start, 115) for start in (0, 25, 50, 75)]
    _assert_pass(engine, provider, calls, "time_budget_exhausted", None, 4)
    assert provider.budget.soft == 0 and "catchup_open" not in vars(provider.budget)


@pytest.mark.parametrize("ratio", [None, "depth"])
@pytest.mark.parametrize("cooldown_elapsed", [False, True])
def test_k4_previous_hang_uses_only_the_base_budget(
    ratio, cooldown_elapsed, make_engine, monkeypatch, install_provider, clock,
):
    engine, view = _case(make_engine, monkeypatch, ratio, summary_model="primary", summary_fallback_models=["fallback"])
    first = install_provider(engine, (float("inf"), SUMMARY))
    _compress(engine, view)
    assert first.calls == [(0, 115, 115, "primary")]
    assert engine._hung_summary_strike == (engine._hold_conversation_key(), "primary")
    if cooldown_elapsed:
        clock.offset += engine._summary_circuit_breaker.cooldown_seconds + 1
    provider = install_provider(engine, (25, SUMMARY), (25, SUMMARY))

    _compress(engine, view)

    if cooldown_elapsed:
        _assert_pass(engine, provider, TWO_BASE[:1], "time_budget_exhausted",
                     False if ratio else None, 1, route="primary",
                     circuits={"primary": ("closed", None), "fallback": ("closed", None)})
    else:
        _assert_pass(engine, provider, TWO_BASE, "soft_target_reached", None, 2, route="fallback",
                     circuits={"primary": ("open", "hung"), "fallback": ("closed", None)})
    assert not getattr(provider.budget, "catchup_open", False)


def test_k5_extension_timeout_is_a_clean_budget_cut(make_engine, monkeypatch, install_provider, clock):
    engine, view = _case(make_engine, monkeypatch)
    provider = install_provider(engine, (25, SUMMARY), (25, SUMMARY), (float("inf"), SUMMARY))

    result = _compress(engine, view)

    _assert_pass(engine, provider, FIVE_CATCHUP[:3], "time_budget_exhausted", False, 2)
    assert clock.offset == 152
    assert provider.budget.catchup_ended and provider.budget.hung_route is None
    assert provider.budget.estimates.calls == {"": [25, 25]}
    assert len(_leaves(engine)) == 2

    clock.offset += 601
    grown = result + _view(10)[17:]
    before = _stored_row_hashes(engine)
    engine.ingest(grown)
    after = _stored_row_hashes(engine)
    assert before.items() <= after.items() and len(after) == len(before) + 4
    retry = install_provider(engine, *((25, SUMMARY),) * 6)
    _compress(engine, grown)
    _assert_pass(engine, retry, [*FIVE_CATCHUP, (125, 55.25, 180.25)], "depth_target_reached", True, 6)


def test_k6_extension_failure_ends_the_route_chain(make_engine, monkeypatch, install_provider):
    engine, view = _case(make_engine, monkeypatch, summary_model="primary", summary_fallback_models=["fallback"])
    provider = install_provider(engine, (25, SUMMARY), (25, SUMMARY), (3, ProviderFailure("HTTP 503")))

    _compress(engine, view)

    _assert_pass(engine, provider, FIVE_CATCHUP[:3], "soft_target_reached", False, 2, route="primary",
                 circuits={"primary": ("closed", None), "fallback": ("closed", None)})
    assert provider.budget.catchup_ended
    assert provider.budget.estimates.calls == {"primary": [25, 25]}


def _seed_prefix(engine, sizes):
    for index, size in enumerate(sizes):
        summary = f"Earlier group {index}. " + payload_of(size)
        engine._dag.add_node(SummaryNode(
            session_id=engine._session_id, depth=0, summary=summary, token_count=count_tokens(summary),
            source_token_count=2 * size, source_ids=[], source_type="messages", created_at=float(index + 1),
        ))


def test_k7_first_condensation_closes_catchup_above_the_prefix_budget(make_engine, monkeypatch, install_provider):
    view = _view(2)
    engine, _ = _case(make_engine, monkeypatch, "0.01", view=view,
                      summary_prefix_target_tokens=510, condensation_fanin=2)
    _seed_prefix(engine, [80] * 6)
    leaf = payload_of(250) + SUMMARY
    condensation = payload_of(115) + SUMMARY
    estimated_after_leaf = count_messages_tokens(view) - (count_messages_tokens(view[1:3]) - count_tokens(leaf))
    engine._config.depth_target_ratio = (estimated_after_leaf - 20) / WINDOW
    assert engine._summary_frontier_tokens() <= 510
    provider = install_provider(engine, (35, leaf), (35, condensation))

    _compress(engine, view)

    _assert_pass(engine, provider, [(0, 115, 115), (35, 102, 137)], "soft_target_reached", True, 1)
    status = _sweep(engine)
    assert status["pre_leaf_condensation_passes"] == 0 and status["condensation_passes"] == 1
    assert status["summary_prefix_tokens_after"] > 510
    assert len([node for node in engine._summary_frontier_nodes() if node.depth == 0]) == 5
    assert provider.budget.catchup_ended


def test_depth_stop_survives_post_leaf_condensation_deadline(make_engine, monkeypatch, install_provider, clock):
    engine, view = _case(make_engine, monkeypatch, summary_prefix_target_tokens=10, condensation_fanin=2)
    provider = install_provider(engine, *((25, SUMMARY),) * 5)

    _compress(engine, view)

    _assert_pass(engine, provider, FIVE_CATCHUP, "depth_target_reached", True, 5)
    status = _sweep(engine)
    assert clock.offset == 125 and status["condensation_passes"] == 0
    assert status["summary_prefix_tokens_after"] > status["summary_prefix_target_tokens"]


def test_k8_first_call_is_base_admitted_before_progress(make_engine, monkeypatch, install_provider):
    engine, view = _case(make_engine, monkeypatch)
    provider = install_provider(engine, *((25, SUMMARY),) * 5)
    attempted = []
    invoke = escalation._invoke_summary_attempt

    def record_progress(*args, budget, **kwargs):
        attempted.append((budget.progress, getattr(budget, "catchup_open", False),
                          getattr(budget, "catchup_call_deadline", None)))
        return invoke(*args, budget=budget, **kwargs)

    monkeypatch.setattr(escalation, "_invoke_summary_attempt", record_progress)
    _compress(engine, view)

    _assert_pass(engine, provider, FIVE_CATCHUP, "depth_target_reached", True, 5)
    assert attempted[0] == ("", False, None)
    assert attempted[1] == ("leaf", True, None)
    assert all(kind == "leaf" and opened and deadline is not None for kind, opened, deadline in attempted[2:])


@pytest.mark.parametrize("soft", ["90", "60"])
def test_k9_resolved_foreground_defaults_control_engagement(soft, make_engine, monkeypatch, install_provider):
    monkeypatch.setenv("LCM_FOREGROUND_SOFT_SECONDS", soft)
    monkeypatch.setenv("LCM_FOREGROUND_HARD_SECONDS", "120")
    engine, view = _case(make_engine, monkeypatch)
    if soft == "60":
        calls, stop, met = FIVE_CATCHUP, "depth_target_reached", True
    else:
        calls, stop, met = [*TWO_BASE, (50, 65, 115)], "soft_target_reached", False
    provider = install_provider(engine, *((25, SUMMARY),) * len(calls))

    _compress(engine, view)

    _assert_pass(engine, provider, calls, stop, met, len(calls))
    assert ("catchup_open" in vars(provider.budget)) == met


@pytest.mark.parametrize("ratio", [None, "0"])
@pytest.mark.parametrize("shape,script,calls,stop", [
    ("K1", [(25, SUMMARY)] * 5, TWO_BASE, "soft_target_reached"),
    ("K2", [(60, SUMMARY)] * 2, [(0, 115, 115)], "time_budget_exhausted"),
    ("K5", [(25, SUMMARY), (25, SUMMARY), (float("inf"), SUMMARY)], TWO_BASE, "soft_target_reached"),
])
def test_p1_base_expectations_for_unset_and_zero(
    ratio, shape, script, calls, stop, make_engine, monkeypatch, install_provider, clock,
):
    """Call sequences and stops observed on a04b1cd6d before the change."""
    engine, view = _case(make_engine, monkeypatch, ratio)
    if shape == "K2":
        _publication_time(engine, monkeypatch, clock)
    provider = install_provider(engine, *script)

    _compress(engine, view)

    _assert_pass(engine, provider, calls, stop, None, len(calls))
    assert "catchup_open" not in vars(provider.budget) and "catchup_ended" not in vars(provider.budget)


def test_pre_leaf_condensation_credits_its_saving_before_the_first_leaf(make_engine, monkeypatch, install_provider):
    engine, earlier = _case(make_engine, monkeypatch, "0", view=_view(3), condensation_fanin=2)
    long_summary = payload_of(1300) + SUMMARY
    seed = install_provider(engine, (10, long_summary), (10, long_summary))
    prefix_view = _compress(engine, earlier)
    _assert_pass(engine, seed, [(0, 115, 115), (10, 105, 115)], "summary_prefix_target_reached", None, 2)
    assert [node.source_ids for node in _leaves(engine)] == [[2, 3], [4, 5]]

    raw = _view(11)
    appended = raw[len(earlier):]
    view = prefix_view + appended
    before = _stored_row_hashes(engine)
    engine.ingest(view)
    after = _stored_row_hashes(engine)
    assert before.items() <= after.items() and len(after) == len(before) + len(appended)
    new_rows = [row["content"] for row in engine._store.get_session_messages(engine._session_id)
                if row["store_id"] not in before]
    assert new_rows == [message["content"] for message in appended]

    engine._config.summary_prefix_target_tokens = 100
    observed = count_messages_tokens(view)
    target = observed - 2_100
    engine.threshold_tokens = observed - 250
    engine._config.depth_target_ratio = target / WINDOW
    assert engine._summary_frontier_tokens() - count_tokens(SUMMARY) > 2_100
    assert count_messages_tokens(earlier[-2:]) - count_tokens(SUMMARY) < 2_100
    provider = install_provider(engine, *((10, SUMMARY),) * 3)

    _compress(engine, view)

    _assert_pass(engine, provider, [(0, 100, 100), (10, 105, 115)], "depth_target_reached", True, 1)
    status = _sweep(engine)
    assert status["pre_leaf_condensation_passes"] == 1
    assert status["whole_prompt_tokens_after"] <= status["depth_target_tokens"]
