"""Whole-envelope snippets caps, with fitting responses pinned to unchanged main."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from hermes_lcm import tools
from tests.test_recall_extra_entries import extra_reader
from tests.test_recall_snippet_windows import make_delivery, recall_engine as recall_engine


GOLDEN = Path(__file__).parent / "fixtures/recall_snippets_envelope_main.json"
MODES = ("plain", "windows", "window-deadline", "window-error", "extras", "extra-error")


def options(d, mode):
    if mode == "windows":
        return dict(snippet_window_hits=5)
    if mode.startswith("window-"):
        reason = "deadline" if mode == "window-deadline" else "read_error"
        return dict(snippet_window_hits=5, read_snippet_windows=Mock(return_value=({}, reason)))
    if mode == "extras":
        return dict(extra_entries=2, read_extra_entries=extra_reader(d))
    if mode == "extra-error":
        return dict(extra_entries=2, read_extra_entries=Mock(return_value=({}, "read_error")))
    return {}


@pytest.mark.parametrize("guidance", [False, True])
@pytest.mark.parametrize("order", ["score", "time"])
@pytest.mark.parametrize("mode", MODES)
def test_fitting_responses_match_main_bytes(guidance, order, mode):
    d = make_delivery((1300,) * 6)
    raw = d.invoke(limit=4, guidance_v2=guidance, delivery_order=order, **options(d, mode))
    assert raw == json.loads(GOLDEN.read_text())[f"{guidance}-{order}-{mode}"]
    assert "query_truncated" not in json.loads(raw)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("order", ["score", "time"])
@pytest.mark.parametrize("character", ["q", "🌿"])
def test_long_query_fits_all_snippets_paths(mode, order, character):
    d = make_delivery((1300,) * 6)
    query = character * 70_000
    original_entries, original_rows = deepcopy(d.entries), deepcopy(d.rows)
    raw = d.invoke(query=query, limit=4, delivery_order=order, **options(d, mode))
    result = json.loads(raw)
    assert len(raw) <= 64_000
    assert result["query"] == query[:4096]
    assert result["query_truncated"] is True
    assert result["total_results"] == 4
    assert [hit["store_id"] for hit in result["hits"][:4]] == [1, 2, 3, 4]
    assert "response_cap_truncated" not in result
    assert d.entries == original_entries and d.rows == original_rows


def test_long_query_without_hits():
    raw = make_delivery(()).invoke(query="q" * 70_000)
    result = json.loads(raw)
    assert len(raw) <= 64_000
    assert result["query"] == "q" * 4096 and result["query_truncated"] is True
    assert result["hits"] == [] and result["total_results"] == 0


def test_session_delivery_receives_the_original_envelope():
    query = "q" * 70_000
    session = Mock(return_value="session delivery owns its cap")
    raw = make_delivery((1300,)).invoke(query=query, session_delivery=session)
    assert raw == "session delivery owns its cap"
    response = session.call_args.args[1]
    assert response["query"] == query
    assert "query_truncated" not in response


@pytest.mark.parametrize("mode", ["plain", "windows", "window-error"])
@pytest.mark.parametrize("deficit", [0, 1])
def test_exact_boundary_and_one_character_over(mode, deficit):
    d = make_delivery((1300,) * 4)
    kwargs = dict(query="q" * 5000, **options(d, mode))
    baseline = d.invoke(**kwargs)
    cap = len(baseline) - deficit
    raw = d.invoke(**kwargs, response_char_cap=cap)
    result = json.loads(raw)
    assert len(raw) <= cap
    if not deficit:
        assert raw == baseline
    else:
        assert result["hits"] == json.loads(baseline)["hits"][:-1]
        assert result["total_results"] == 3
        assert result["response_cap_truncated"] is True
    assert result["query"] == kwargs["query"]
    assert "query_truncated" not in result


def test_production_cap_counts_mid_size_envelope_and_keeps_rank_prefix():
    d = make_delivery((1300,) * 3)
    # Exactly one character above the production cap, although the hit bodies
    # alone fit comfortably. The no-hit envelope fits, so keep the query echo.
    query = "q" * (64_001 - len(d.invoke(query="")))
    raw = d.invoke(query=query)
    result = json.loads(raw)
    assert len(raw) <= 64_000
    assert [hit["store_id"] for hit in result["hits"]] == [1, 2]
    assert result["total_results"] == 2 and result["response_cap_truncated"] is True
    assert result["query"] == query and "query_truncated" not in result


def test_hit_bodies_just_below_production_cap_still_budget_the_envelope():
    d = make_delivery((1300,) * 25)
    baseline = json.loads(d.invoke(query="q" * 1000))
    hit_chars = sum(len(json.dumps(hit)) for hit in baseline["hits"])
    # Legal provenance can be much larger than the snippets themselves. Make
    # the sum of hit bodies exactly 63,950; the envelope puts it over 64,000.
    d.entries[-1]["hit"]["expand_hint"] += " " * (63_950 - hit_chars)
    raw = d.invoke(query="q" * 1000)
    result = json.loads(raw)
    assert len(raw) <= 64_000
    assert [hit["store_id"] for hit in result["hits"]] == list(range(1, 25))
    assert result["total_results"] == 24 and result["response_cap_truncated"] is True
    assert result["query"] == "q" * 1000 and "query_truncated" not in result


@pytest.mark.parametrize("mode", ["plain", "windows", "window-error", "extra-error"])
def test_optional_time_and_extra_markers_never_evict_a_hit(mode):
    d = make_delivery((1300,) * 6)
    kwargs = dict(limit=4, **options(d, mode))
    baseline = json.loads(d.invoke(**kwargs))
    baseline.pop("extra_entries_skipped", None)
    cap = len(json.dumps(baseline))
    raw = d.invoke(**kwargs, delivery_order="time", response_char_cap=cap)
    assert len(raw) <= cap
    assert json.loads(raw) == baseline


@pytest.mark.parametrize("reason", ["deadline", "read_error"])
def test_optional_event_time_marker_never_evicts_a_hit(reason):
    d = make_delivery((1300,) * 3)
    baseline = d.invoke()
    raw = d.invoke(response_char_cap=len(baseline), read_event_time=Mock(return_value=({}, reason)))
    assert raw == baseline


def test_fitting_refreshes_provenance_to_a_fixed_point():
    d = make_delivery((1300,) * 12)
    kwargs = dict(session_floor_info=dict(requested=12, achieved=12, promoted=0, delivered=None),
                  adjacent_reply=[dict(source_store_id=1, store_id=12)],
                  fts_anchor_info=dict(fired=True, position=12, delivered=None),
                  arm_order=["chunk", "fts"], arm_weights=[1.0, 1.0],
                  arm_hits={"fts": [d.entries[-1]["hit"]]})
    baseline = d.invoke(**deepcopy(kwargs))
    raw = d.invoke(response_char_cap=len(baseline) - 1, **deepcopy(kwargs))
    result = json.loads(raw)
    assert len(raw) <= len(baseline) - 1
    assert result["total_results"] == len(result["hits"]) == 11
    assert result["provenance"]["session_floor"]["delivered"] == 11
    assert result["provenance"]["fts_anchor"]["delivered"] is False
    assert "adjacent_reply" not in result["provenance"]
    assert result["response_cap_truncated"] is True


@pytest.mark.parametrize("raw_limit", ["1e999", "-1e999"])
def test_nonfinite_limit_uses_unparseable_default(recall_engine, monkeypatch, raw_limit):
    recall_engine._config.embeddings_enabled = False
    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", lambda *a, **kw: ([], None))
    args = dict(query="ANSWER_TOKEN", limit="unparseable")
    baseline = tools.lcm_recall(args, engine=recall_engine)
    raw = tools.lcm_recall({**args, "limit": json.loads(raw_limit)}, engine=recall_engine)
    assert raw == baseline
    assert json.loads(raw)["limit"] == tools._LCM_RECALL_DEFAULT_LIMIT
    assert tools._parse_int_value(json.loads(raw_limit), 17) == 17
