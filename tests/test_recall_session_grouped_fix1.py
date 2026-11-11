"""Regressions for the grouped view's detail-specific fitting contracts."""
import copy
import json
from pathlib import Path

import pytest

from hermes_lcm import recall_delivery, recall_session, tokens, tools
from tests import test_recall_later_statements as later
from tests import test_recall_session_grouped as grouped

grouped_engine = grouped.grouped_engine
recall_engine = grouped.recall_engine
invoke = grouped.invoke


def grouped_without_times(engine, detail):
    response = json.loads(invoke(engine, detail, group_by_session=True))
    response["hits"] = [recall_delivery._without_event_time(h) for h in response["hits"]]
    return response


def test_snippets_preserve_long_query_echo(grouped_engine):
    args = {"query": "orchard " + "x" * 70000, "limit": 10, "detail": "snippets"}
    off = json.loads(tools.lcm_recall(args, engine=grouped_engine, group_by_session=False))
    on = json.loads(tools.lcm_recall(args, engine=grouped_engine, group_by_session=True))
    assert on["query"] == off["query"] == args["query"]
    assert len(on["hits"]) == len(off["hits"]) == 7


def test_snippets_budget_hits_and_group_fields_not_envelope(grouped_engine, monkeypatch):
    cap = len(json.dumps(grouped_without_times(grouped_engine, "snippets"))) + 5
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", cap)
    off = json.loads(invoke(grouped_engine, group_by_session=False))
    on = json.loads(invoke(grouped_engine, group_by_session=True))
    assert len(on["hits"]) == len(off["hits"]) == 7
    assert on["provenance"]["session_groups_dropped"] == 0


def test_snippets_report_actual_additional_drops(grouped_engine, monkeypatch):
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", 2000)
    off = json.loads(invoke(grouped_engine, group_by_session=False))
    on = json.loads(invoke(grouped_engine, group_by_session=True))
    assert 0 < len(on["hits"]) < len(off["hits"])
    assert on["response_cap_truncated"] is True
    assert on["provenance"]["session_groups_dropped"] == len(off["hits"]) - len(on["hits"])


@pytest.mark.parametrize("gap", [5, 120])
def test_answer_ready_strips_optional_times_before_evidence(grouped_engine, monkeypatch, gap):
    cap = len(json.dumps(grouped_without_times(grouped_engine, "answer_ready"), ensure_ascii=False)) + gap
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", cap)
    off = json.loads(invoke(grouped_engine, "answer_ready", group_by_session=False))
    raw = invoke(grouped_engine, "answer_ready", group_by_session=True)
    on = json.loads(raw)
    assert len(on["hits"]) == len(off["hits"]) == 7
    assert any("event_time" not in h for h in on["hits"])
    if gap == 120:
        assert on["event_time_unavailable_reason"] == "response_cap"
    assert len(raw) <= cap


def test_delta_refs_keep_original_rank_order(grouped_engine):
    args = {"query": "orchard", "limit": 10, "detail": "answer_ready", "seen_refs": []}
    off = json.loads(tools.lcm_recall(args, engine=grouped_engine, group_by_session=False))
    on = json.loads(tools.lcm_recall(args, engine=grouped_engine, group_by_session=True))
    assert on["delta"] == off["delta"]


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
@pytest.mark.parametrize("wide_ids", [True, False])
def test_session_grouping_reconciles_hitless_metadata_first(recall_engine, monkeypatch, detail, wide_ids):
    e = recall_engine
    e._config.embeddings_enabled = False
    e._config.recall_delivery_unit = "session"
    e._config.large_output_externalization_enabled = False
    e._config.recall_query_timeout_s = 60
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(tools.time, "time", lambda: 1700000000)
    entries = []
    for i in range(25):
        session = f"{i:02d}" + ("漢" * 480 if wide_ids else "a" * 480) + "a" * 1120
        store_id = e._store.append(session, {
            "role": "user", "content": "needle " + "x" * (16000 if i == 0 else 70000),
        })
        entries.append(later.entry(e, store_id))
    later.ranked(monkeypatch, entries)
    args = {"query": "needle", "limit": 25, "detail": detail}
    off = json.loads(tools.lcm_recall(args, engine=e, group_by_session=False))
    assert [h["store_id"] for h in off["hits"]] == [1]
    # A concrete bounded response exists with just the last hitless record gone.
    candidate = copy.deepcopy(off)
    candidate["provenance"]["delivery"]["sessions"].pop()
    recall_delivery.session_group_view(candidate)()
    meta = candidate["provenance"]["delivery"]
    for _ in range(10):
        raw = json.dumps(candidate, ensure_ascii=False)
        meta.update(response_chars=len(raw), response_tokens=tokens.count_tokens_with_encoder(raw, encoder=None))
    assert len(raw) <= recall_session.CHAR_BUDGET
    assert meta["response_tokens"] <= recall_session.TOKEN_BUDGET
    on_raw = tools.lcm_recall(args, engine=e, group_by_session=True)
    on = json.loads(on_raw)
    assert "error" not in on
    assert [h["store_id"] for h in on["hits"]] == [1]
    assert on["provenance"]["session_groups_dropped"] == 0
    delivery = on["provenance"]["delivery"]
    assert len(delivery["sessions"]) == len(off["provenance"]["delivery"]["sessions"]) - 1
    assert delivery["response_chars"] == len(on_raw) <= delivery["char_budget"]
    assert delivery["response_tokens"] <= delivery["token_budget"]


def test_ranking_consumers_pin_grouping_off(monkeypatch, tmp_path):
    from benchmarking import longmemeval as lme
    from tests.test_longmemeval_harness import _synthetic_dataset

    calls = []
    monkeypatch.setattr(tools, "lcm_recall", lambda args, **kw: calls.append(kw) or '{"hits": []}')
    lme.production_recall_hits(
        _synthetic_dataset()[0], grouped_config := type("Config", (), {"recall_group_by_session": True})(),
        None, None, None, provider_name="stub", tmp_dir=tmp_path, embeddings_enabled=False, limit=10,
    )
    assert grouped_config.recall_group_by_session
    assert calls[0].get("group_by_session") is False
    # The other consumers use the same explicit override at their call sites.
    for name in ("benchmarks/qa-harness/src/providers/hermes-lcm/bridge/hermes_lcm_bridge.py",
                 "bench/instruments/scale389/phase1a.py"):
        assert "group_by_session=False" in (Path(__file__).parents[1] / name).read_text()


def test_grouping_release_note_and_session_budget_tradeoff_documented():
    root = Path(__file__).parents[1]
    fragment = root / "changelog.d/1188-session-grouped.md"
    assert fragment.is_file()
    assert fragment.read_text().startswith("- New:")
    docs = (root / "docs/retrieval-tools.md").read_text()
    assert "whole session groups" in docs
    assert "provenance.session_groups_dropped" in docs


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_floor_delivered_refreshed_when_grouping_drops_hits(grouped_engine, monkeypatch, detail):
    full = json.loads(invoke(grouped_engine, detail, group_by_session=True, min_sessions=3))
    cap = 2000 if detail == "snippets" else len(json.dumps(full, ensure_ascii=False)) - 900
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", cap)
    on = json.loads(invoke(grouped_engine, detail, group_by_session=True, min_sessions=3))
    assert 0 < len(on["hits"]) < len(full["hits"])
    sessions = {h["session_id"] for h in on["hits"] if not h.get("extra")} - {None, ""}
    assert len(sessions) < full["provenance"]["session_floor"]["delivered"]
    assert on["provenance"]["session_floor"]["delivered"] == len(sessions)
