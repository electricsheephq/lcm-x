"""``LCM_STUB_FIRST_EXIT_TARGET_RATIO``: a share of the threshold that lowers the #671 stub-first exit's target.

Unset (0), the target is the built-in min(threshold - leaf chunk, 0.95 x threshold); set, it is
min(that, int(ratio x threshold)). Stub cuts that miss the lowered target take the leaf path as before.
"""

import json

import pytest

import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.tokens import count_messages_tokens
from tests.test_issue_671_stub_first_exit import (  # noqa: F401 (fixtures)
    STUB,
    THRESHOLD,
    _trial,
    exit_engine,
    make_engine,
    run,
    stop_line,
    summaries,
    tool_content,
    tool_view,
)

ENV = "LCM_STUB_FIRST_EXIT_TARGET_RATIO"


def built_in_target(threshold: int, leaf_chunk_tokens: int = 1_000) -> int:
    return min(threshold - leaf_chunk_tokens, int(threshold * 0.95))


# -- config ---------------------------------------------------------------------------------------------------------


def test_unset_is_zero_from_the_default_source_with_no_warning(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    c = LCMConfig.from_env()
    assert LCMConfig().stub_first_exit_target_ratio == 0.0
    assert c.stub_first_exit_target_ratio == 0.0
    assert c.config_sources["stub_first_exit_target_ratio"] == "default"
    assert not any(ENV in warning for warning in c.config_source_warnings)


@pytest.mark.parametrize("raw, value", [("0.2", 0.2), ("1", 1.0), ("0", 0.0)])
def test_a_value_in_range_is_read_from_the_env(monkeypatch, raw, value):
    monkeypatch.setenv(ENV, raw)
    c = LCMConfig.from_env()
    assert c.stub_first_exit_target_ratio == value
    assert c.config_sources["stub_first_exit_target_ratio"] == f"env:{ENV}"
    assert not any(ENV in warning for warning in c.config_source_warnings)


@pytest.mark.parametrize("raw", ["-0.1", "1.5", "nan", "inf", "abc"])
def test_an_invalid_value_keeps_the_built_in_target_with_a_warning(monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    c = LCMConfig.from_env()
    assert c.stub_first_exit_target_ratio == 0.0
    assert c.config_sources["stub_first_exit_target_ratio"] == "default"
    assert any(ENV in warning for warning in c.config_source_warnings)


# -- unset: today's target ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("threshold", [THRESHOLD, 20_000, 100_000])  # the leaf-chunk side, both equal, the 0.95 side
@pytest.mark.parametrize("ratio", [0.0, 0.95, 1.0, -0.5, float("nan")])  # unset, no lower than today, not a share
def test_unset_or_a_ratio_at_or_above_the_built_in_share_keeps_todays_target(make_engine, threshold, ratio):  # noqa: F811
    engine = exit_engine(make_engine, stub_first_exit_target_ratio=ratio)
    engine.threshold_tokens = threshold
    view = tool_view()

    assert _trial(engine, view, count_messages_tokens(view)) is not None
    assert engine._last_stub_first_exit["target_tokens"] == built_in_target(threshold)


# -- set: the lowered target decides ---------------------------------------------------------------------------------


def test_stub_cuts_over_the_lowered_target_take_the_leaf_path(make_engine, summaries, caplog):  # noqa: F811
    view = tool_view()
    engine = exit_engine(make_engine, stub_first_exit_target_ratio=0.01)  # target 120: under the stubbed list
    assert _trial(exit_engine(make_engine), view, count_messages_tokens(view)) is not None  # unset, the cuts exit

    result = run(engine, view, caplog)

    assert engine._last_stub_first_exit is None and engine.get_status()["last_stub_first_exit"] is None
    line = stop_line(caplog)
    assert line.group(1) != "stub_first_exit" and int(line.group(2)) > 0  # at least one leaf
    assert summaries  # the leaf path made its model calls
    assert engine.last_compression_status == "compacted"
    assert len(engine._dag.get_session_nodes(engine._session_id)) > 0
    assert engine._store.get_session_count(engine._session_id) == len(view)  # every row stored
    assert count_messages_tokens(result) < count_messages_tokens(view)


def test_stub_cuts_under_the_lowered_target_exit_and_report_it(make_engine, summaries, caplog):  # noqa: F811
    engine = exit_engine(make_engine, stub_first_exit_target_ratio=0.5, threshold_full_sweep_enabled=True)
    target = int(THRESHOLD * 0.5)
    assert target < built_in_target(THRESHOLD)  # a real lowering
    view = tool_view()

    result = run(engine, view, caplog)

    assert summaries == []  # no leaf, no condensation, no model call
    assert len(engine._dag.get_session_nodes(engine._session_id)) == 0
    assert count_messages_tokens(result) <= target
    assert all(tool_content(result, f"call-{i}").startswith(STUB) for i in range(5))
    record = engine.get_status()["last_stub_first_exit"]
    assert record["target_tokens"] == target and record["tokens_after"] <= target
    line = stop_line(caplog)  # the stop line carries the lowered target as is
    assert line.group(1) == "stub_first_exit" and line.group(2) == "0"
    assert int(line.group(4)) == record["tokens_after"] and int(line.group(5)) == target
    status = json.loads(lcm_tools.lcm_status({}, engine=engine))
    assert status["last_stub_first_exit"]["target_tokens"] == target
    assert status["threshold_full_sweep"]["stop_reason"] == "stub_first_exit"
    assert status["config"]["stub_first_exit_target_ratio"] == 0.5


def test_a_ratio_that_rounds_the_target_to_zero_takes_the_leaf_path(make_engine):  # noqa: F811
    engine = exit_engine(make_engine, stub_first_exit_target_ratio=1e-6)  # int(12_000 * 1e-6) == 0
    view = tool_view()

    assert _trial(engine, view, count_messages_tokens(view)) is None
    assert engine._last_stub_first_exit is None
