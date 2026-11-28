"""Whole-envelope snippets caps, with fitting responses pinned to unchanged main."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from hermes_lcm import tools
from tests.test_recall_extra_entries import extra_reader
from tests.test_recall_snippet_windows import make_delivery, recall_engine as recall_engine


GOLDEN = Path(__file__).parent / "fixtures/recall_snippets_envelope_main.json"
MODES = ("plain", "windows", "window-deadline", "window-error", "extras", "extra-error")


MAIN_BASE = "9d9f0e8c97b1b6e408300f8cfe64227b3e2afe6c"
# Main's windows outputs at MAIN_BASE, pinned as sha256 and length. Regenerate from the repo root with
#   python -m pytest -p no:cacheprovider tests/fixtures/capture_recall_snippets_envelope_main_windows.py
MAIN_WINDOWS = Path(__file__).parent / "fixtures/recall_snippets_envelope_main_windows_9d9f0e8c.json"


def pinned(raw):
    return {"sha256": hashlib.sha256(raw.encode()).hexdigest(), "chars": len(raw)}


def windows_cases(invoke):
    """The windows inputs whose main outputs MAIN_WINDOWS pins; invoke(delivery, **kwargs) runs one."""
    cases = {}
    for query_chars in (5000, 4097):
        d = make_delivery((1300,) * 6)
        kwargs = dict(query="q" * query_chars, snippet_window_hits=5)
        cap = len(invoke(d, **kwargs, delivery_order="score")) - 1
        for order in ("score", "time"):
            cases[f"tail-{query_chars}-{order}"] = invoke(d, **kwargs, delivery_order=order, response_char_cap=cap)
    for order in ("score", "time"):
        d = dated_delivery()
        cases[f"cut-55000-{order}"] = invoke(d, query="q" * 55_000, **cut_kwargs(d, "windows", order))
    return cases


@pytest.fixture(scope="module")
def main_windows():
    # A missing or unreadable pin fails the test; it never skips.
    return json.loads(MAIN_WINDOWS.read_text())["cases"]


@pytest.mark.parametrize("query_chars", [5000, 4097])
@pytest.mark.parametrize("order", ["score", "time"])
def test_windows_tail_drop_preserves_main_query_bytes(main_windows, query_chars, order):
    cases = windows_cases(lambda d, **kwargs: d.invoke(**kwargs))
    assert set(cases) == set(main_windows)
    expected = cases[f"tail-{query_chars}-{order}"]
    assert pinned(expected) == main_windows[f"tail-{query_chars}-{order}"]
    result = json.loads(expected)
    assert result["total_results"] == 5
    assert result["query"] == "q" * query_chars
    marker_chars = len(', "delivery_order": "time"') if order == "time" else 0
    assert len(expected) == 14_044 + query_chars - 5000 + marker_chars


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



def dated_delivery():
    d = make_delivery((1300,) * 12)
    for entry in d.entries:
        entry["hit"]["snippet"] = d.rows[entry["hit"]["store_id"]]["content"]
    for row in d.rows.values():
        row.update(observed_at=1_700_000_000, observed_at_source="host_message_timestamp")
    return d


def cut_kwargs(d, mode, order):
    return dict(limit=8, snippet_chars=1300, delivery_order=order, **options(d, mode))

@pytest.mark.parametrize("guidance", [False, True])
@pytest.mark.parametrize("order", ["score", "time"])
@pytest.mark.parametrize("mode", MODES)
def test_fitting_responses_match_main_bytes(guidance, order, mode):
    d = make_delivery((1300,) * 6)
    raw = d.invoke(limit=4, guidance_v2=guidance, delivery_order=order, **options(d, mode))
    assert raw == json.loads(GOLDEN.read_text())[f"{guidance}-{order}-{mode}"]
    assert "query_truncated" not in json.loads(raw)


@pytest.mark.parametrize("guidance", [False, True])
@pytest.mark.parametrize("order", ["score", "time"])
@pytest.mark.parametrize("mode", MODES)
def test_fitting_5000_character_query_matches_main_bytes(guidance, order, mode):
    d = make_delivery((1300,) * 6)
    expected = json.loads(json.loads(GOLDEN.read_text())[f"{guidance}-{order}-{mode}"])
    expected["query"] = "q" * 5000
    raw = d.invoke(query=expected["query"], limit=4, guidance_v2=guidance,
                   delivery_order=order, **options(d, mode))
    assert len(raw) < 64_000
    assert raw == json.dumps(expected)
    assert "query_truncated" not in json.loads(raw)


@pytest.mark.parametrize("order", ["score", "time"])
@pytest.mark.parametrize("mode", MODES)
def test_query_cut_precedes_event_time_strip_and_hit_drop(mode, order, main_windows):
    d = dated_delivery()
    kwargs = cut_kwargs(d, mode, order)
    # Main's full response exceeds the cap, but its no-hit envelope fits.
    full = d.invoke(query="q" * 55_000, response_char_cap=100_000, **kwargs)
    assert len(full) > 64_000
    envelope = json.loads(full)
    envelope.update(hits=[], total_results=0)
    assert len(json.dumps(envelope)) < 64_000
    raw = d.invoke(query="q" * 55_000, **kwargs)
    result = json.loads(raw)
    assert len(raw) <= 64_000
    if mode == "windows":
        # Main already fitted windows by dropping the ranked tail. Preserve its
        # query and dates whenever the no-hit envelope (with markers) fits.
        assert pinned(raw) == main_windows[f"cut-55000-{order}"]
        assert result["query"] == "q" * 55_000
        assert "query_truncated" not in result
        assert result["response_cap_truncated"] is True
        assert 0 < len(result["hits"]) < 8
        assert all("event_time" in hit for hit in result["hits"])
        return
    assert result["query"] == "q" * 4096
    assert result.pop("query_truncated") is True
    expected = json.loads(full)
    expected["query"] = "q" * 4096
    assert result == expected  # Includes every hit, date, and optional marker.
    assert all("event_time" in hit for hit in result["hits"][:8])
    assert "response_cap_truncated" not in result


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("order", ["score", "time"])
def test_emoji_query_cut_still_fits_when_hits_must_be_dropped(mode, order):
    d = make_delivery((1300,) * 25)
    for entry in d.entries:
        entry["hit"]["snippet"] = d.rows[entry["hit"]["store_id"]]["content"]
    query = "🌿" * 55_000
    assert len(json.dumps(query[:4096])) == 49_154  # Quotes plus 12 chars/code point.
    raw = d.invoke(query=query, snippet_chars=1300, delivery_order=order, **options(d, mode))
    result = json.loads(raw)
    assert len(raw) <= 64_000
    assert result["query"] == query[:4096] and result["query_truncated"] is True
    assert result["response_cap_truncated"] is True
    assert 0 < len(result["hits"]) < 25
    assert [hit["store_id"] for hit in result["hits"]] == list(range(1, len(result["hits"]) + 1))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("order", ["score", "time"])
@pytest.mark.parametrize("character", ["q", "🌿"])
def test_long_query_fits_all_snippets_paths(mode, order, character):
    d = make_delivery((1300,) * 12)
    for row in d.rows.values():
        row.update(observed_at=1_700_000_000, observed_at_source="host_message_timestamp")
    kwargs = options(d, mode)
    if mode in {"extras", "extra-error"}:
        kwargs["extra_entries"] = 4
    query = character * 70_000
    original_entries, original_rows = deepcopy(d.entries), deepcopy(d.rows)
    raw = d.invoke(query=query, limit=8, delivery_order=order, **kwargs)
    result = json.loads(raw)
    assert len(raw) <= 64_000
    assert result["query"] == query[:4096]
    assert result["query_truncated"] is True
    assert result["total_results"] == 8
    assert [hit["store_id"] for hit in result["hits"][:8]] == list(range(1, 9))
    assert all("event_time" in hit for hit in result["hits"][:8])
    assert "event_time_unavailable" not in result
    if order == "time":
        assert result["provenance"]["delivery_order"] == "time"
    else:
        assert "delivery_order" not in result["provenance"]
    if mode in {"extras", "extra-error"}:
        kwargs["read_extra_entries"].assert_called_once()
        assert kwargs["read_extra_entries"].call_args.kwargs["expanded_limit"] == 4
    if mode == "extras":
        assert len(result["hits"]) == 12
        assert all(hit["extra"] for hit in result["hits"][8:])
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
    kwargs = dict(query="q" * 4000, **options(d, mode))
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


def test_production_cap_cuts_long_query_before_dropping_ranked_hits():
    d = make_delivery((1300,) * 3)
    # Exactly one character above the production cap, although the hit bodies
    # alone fit comfortably. Cut the query even though the no-hit envelope fits.
    query = "q" * (64_001 - len(d.invoke(query="")))
    raw = d.invoke(query=query)
    result = json.loads(raw)
    assert len(raw) <= 64_000
    assert [hit["store_id"] for hit in result["hits"]] == [1, 2, 3]
    assert result["total_results"] == 3 and "response_cap_truncated" not in result
    assert result["query"] == query[:4096] and result["query_truncated"] is True


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


@pytest.mark.parametrize("mode", ["plain", "window-error", "extra-error"])
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


@pytest.mark.parametrize("mode", ["plain", "windows", "window-deadline", "window-error"])
@pytest.mark.parametrize("order", ["score", "time"])
@pytest.mark.parametrize("slack", [0, 1, 31])
@pytest.mark.parametrize("observed", [False, True])
def test_no_hit_envelope_reserves_truncation_markers(mode, order, slack, observed):
    d = make_delivery((1300,) * 3)
    if observed:
        for row in d.rows.values():
            row.update(observed_at=1_700_000_000, observed_at_source="host_message_timestamp")
    kwargs = dict(delivery_order=order, **options(d, mode))
    envelope = json.loads(d.invoke(query="", **kwargs))
    envelope.update(hits=[], total_results=0)
    envelope["provenance"].pop("delivery_order", None)
    query = "q" * (64_000 - slack - len(json.dumps(envelope)))
    raw = d.invoke(query=query, **kwargs)
    result = json.loads(raw)
    assert len(raw) <= 64_000
    assert result["query_truncated"] is True
    assert result["query"] == query[:4096]
    assert len(result["hits"]) == 3


@pytest.mark.parametrize("case", ["time-marker", "time-marker-long-query", "event-time"])
def test_windows_boundary_matches_main_bytes(case):
    d = make_delivery((1300,) * 3)
    if case == "event-time":
        for row in d.rows.values():
            row.update(observed_at=1_700_000_000, observed_at_source="host_message_timestamp")
    kwargs = dict(snippet_window_hits=5,
                  delivery_order="time" if case.startswith("time-marker") else "score")
    if case == "time-marker-long-query":
        kwargs["query"] = "q" * 5000
    full = json.loads(d.invoke(**kwargs))
    # Main budgets the time marker, then drops the ranked tail without an
    # envelope-level event-time strip. Pin the complete resulting JSON bytes.
    cap = len(json.dumps(full)) - 1
    if case.startswith("time-marker"):
        unmarked = deepcopy(full)
        unmarked["provenance"].pop("delivery_order")
        assert len(json.dumps(unmarked)) <= cap < len(json.dumps(full))
    expected = deepcopy(full)
    expected["hits"].pop()
    expected.update(total_results=2, response_cap_truncated=True)
    assert d.invoke(**kwargs, response_char_cap=cap) == json.dumps(expected)
    if case == "event-time":
        assert all("event_time" in hit for hit in expected["hits"])
        assert "event_time_unavailable" not in expected


@pytest.mark.parametrize("mode", ["plain", "window-deadline", "window-error"])
def test_plain_item_loop_drop_reports_truncation(mode):
    d = make_delivery((1300,) * 2)
    d.entries[1]["hit"]["expand_hint"] += "x" * 5000
    cap = len(d.invoke(limit=1, **options(d, mode))) + len(', "response_cap_truncated": true')
    raw = d.invoke(limit=2, response_char_cap=cap, **options(d, mode))
    result = json.loads(raw)
    assert len(raw) <= cap
    assert [hit["store_id"] for hit in result["hits"]] == [1]
    assert result["response_cap_truncated"] is True
