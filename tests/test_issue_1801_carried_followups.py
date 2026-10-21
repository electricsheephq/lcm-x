"""Read-only carried-time follow-ups through the public auto compiler."""
import importlib.util
import inspect
import json
from pathlib import Path
import sqlite3

import pytest

from hermes_lcm import tools
from tests.harness import engine_context

REASON = "uncertain_source_observation_time"
STAMP = 1676462400.0  # 2023-02-15: available before either question boundary.


def plant(engine, texts, carried_index=0):
    ids = [engine._store.append(str(i), dict(role="user", content=text,
        timestamp=STAMP, source="cli")) for i, text in enumerate(texts)]
    engine._store.add_message_relations([[(ids[carried_index], "carried_stamp", None, 0, STAMP)]])
    return [f"lcm:{sid}:0-{len(text)}" for sid, text in zip(ids, texts)]


def args(refs, question="What is the project status?", question_date=None):
    return dict(mode="auto", question=question, question_date=question_date,
        baseline_refs=refs, budgets={"max_retrieval_calls": 0})


@pytest.mark.parametrize("entry", ["tool", "engine"])
@pytest.mark.parametrize("site", ["_candidate_inventory", "_retrieve_sources", "_finite_enumeration", "get"])
def test_auto_sqlite_boundary(tmp_path, monkeypatch, entry, site):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        refs = plant(engine, ["My project status is green."])
        request = args(refs)
        if site == "_finite_enumeration":
            request = args([], "How many vacations did I take this year?", "2024-03-22")
        elif site == "_retrieve_sources":
            request = args([])
            request["budgets"]["max_retrieval_calls"] = 1
            monkeypatch.setattr(tools, "lcm_recall", lambda *a, **k: json.dumps({"hits": [{"exact_ref": refs[0]}]}))
        method = "get" if site == "get" else "get_carried_stamp_ids"
        original = getattr(engine._store, method)
        injected = []

        def fail(*a, **k):
            if site == "get" or inspect.currentframe().f_back.f_code.co_name == site:
                injected.append(site)
                raise sqlite3.OperationalError("synthetic read failure")
            return original(*a, **k)

        monkeypatch.setattr(engine._store, method, fail)
        before = engine._store.connection.total_changes
        raw = (tools.lcm_compile_evidence(request, engine=engine) if entry == "tool"
            else engine.handle_tool_call("lcm_compile_evidence", request))
        result = json.loads(raw)
        assert injected == [site]
        assert result["status"] == "fallback"
        assert result["reason"] == REASON
        assert result["reason_code"] == REASON
        assert result["next_path"] == "evidence_only"
        assert not result.get("context") and not result.get("computation")
        assert not result.get("evidence") and not result.get("query_view")
        assert engine._store.connection.total_changes == before


@pytest.mark.parametrize("entry", ["tool", "engine"])
def test_auto_does_not_swallow_non_sqlite(tmp_path, monkeypatch, entry):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        refs = plant(engine, ["My project status is green."])

        def fail(*a, **k):
            raise RuntimeError("synthetic programming error")

        monkeypatch.setattr(engine._store, "get", fail)
        with pytest.raises(RuntimeError, match="synthetic programming error"):
            if entry == "tool":
                tools.lcm_compile_evidence(args(refs), engine=engine)
            else:
                engine.handle_tool_call("lcm_compile_evidence", args(refs))


@pytest.mark.parametrize("form", ["February 10, 2023", "Feb 10, 2023", "February 10 2023", "Feb 10th, 2023", "February 10th 2023"])
@pytest.mark.parametrize("operation", ["order", "date_filter", "finite"])
@pytest.mark.parametrize("window", ["outside", "inside"])
def test_month_name_matches_iso_window(tmp_path, form, operation, window):
    suffix = "this year" if window == "outside" else "in February"
    question = {"order": f"What is the order of events {suffix}?",
        "date_filter": f"Which vacation did I take {suffix}?",
        "finite": f"How many vacations did I take {suffix}?"}[operation]
    question_date = "2024-03-22" if window == "outside" else "2023-03-22"
    results = []
    for arm, spelling in enumerate(["2023-02-10", form]):
        with engine_context(tmp_path / str(arm), embeddings_enabled=False) as engine:
            refs = plant(engine, [f"I took a vacation to Bali on {spelling}.",
                "I took a vacation to Kyoto on 2023-02-12."])
            result = json.loads(tools.lcm_compile_evidence(args(refs, question, question_date), engine=engine))
            results.append(result)
    iso, named = results
    assert iso["contract"]["operation"] == ("scalar" if operation == "finite" else operation)
    assert named["state"] == iso["state"]
    assert named["reason_code"] == iso["reason_code"]
    assert named["metrics"]["candidate_count"] == iso["metrics"]["candidate_count"]
    assert [r["reason_code"] for r in named.get("exclusions", [])] == [r["reason_code"] for r in iso.get("exclusions", [])]
    if window == "inside" and operation == "date_filter":
        assert named["direct_fact"]["time_basis"] == "occurred_at"
        assert named["direct_fact"]["time_uncertain"] == "carried"


@pytest.mark.parametrize("carried_first", [False, True])
def test_date_filter_checks_only_selected_row(tmp_path, carried_first):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        texts = ["I took a vacation to Bali on 2023-02-10.", "I took a vacation to Kyoto yesterday."]
        if carried_first:
            texts.reverse()
        refs = plant(engine, texts, carried_index=0 if carried_first else 1)
        before = engine._store.connection.total_changes
        result = json.loads(tools.lcm_compile_evidence(args(refs,
            "Which vacation did I take in February?", "2023-03-22"), engine=engine))
        if carried_first:
            assert result["reason_code"] == REASON
            assert result.get("direct_fact") is None
            assert {"exact_ref": refs[0], "time_uncertain": "carried", "reason_code": REASON} in result["exclusions"]
        else:
            assert result["direct_fact"]["exact_ref"] == refs[0]
            assert "Bali" in result["direct_fact"]["value"]
        assert engine._store.connection.total_changes == before


def test_preanswer_hook_fails_open_for_fallback_and_sqlite(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("hermes_lcm.issue1801_plugin", Path(__file__).parents[1] / "__init__.py")
    plugin = importlib.util.module_from_spec(spec)
    plugin.__package__ = "hermes_lcm"
    spec.loader.exec_module(plugin)
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        refs = plant(engine, ["My project status is green."])
        engine._config.preanswer_evidence_enabled = True
        engine._config.preanswer_evidence_mode = "requirements_v1"
        monkeypatch.setattr(plugin, "_answer_ready_baseline", lambda *a: refs)
        payload = {"user_message": "What is the project status?"}
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr("hermes_lcm.evidence_compiler.compile_preanswer_evidence",
                lambda *a, **k: {"status": "fallback", "reason": REASON, "next_path": "evidence_only"})
            assert plugin._pre_llm_context(engine, payload) is None

        def fail(*a, **k):
            raise sqlite3.OperationalError("synthetic read failure")

        monkeypatch.setattr(engine._store, "get_carried_stamp_ids", fail)
        assert plugin._pre_llm_context(engine, payload) is None
