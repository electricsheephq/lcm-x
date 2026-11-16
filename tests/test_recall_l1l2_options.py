"""Offline contracts for the opt-in recency floor and stemmed grep options."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from hermes_lcm import config as config_module, db_bootstrap as boot, retrieval_core, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.store import MessageStore, build_message_stem_fts_spec

NOW = 1_800_000_000.0
# Captured from unchanged 62a8137a with the fixture and clocks below.
BASELINE = {
    'recall': (
        '{"query": "cake", '
        '"limit": 3, '
        '"scope_bias": 0.5, '
        '"include": "all", '
        '"total_results": 3, '
        '"hits": [{"kind": "message_excerpt", '
        '"session_id": "fresh", '
        '"timestamp": 1800000000.0, '
        '"snippet": ">>>cake<<< garden", '
        '"score": 0.008197, '
        '"expand_hint": "lcm_expand(store_id=2, '
        'content_offset=0)", '
        '"from_current_session": false, '
        '"arms": ["fts"], '
        '"store_id": 2}, '
        '{"kind": "message_excerpt", '
        '"session_id": "plural", '
        '"timestamp": 1800000000.0, '
        '"snippet": ">>>cakes<<< for the party", '
        '"score": 0.007937, '
        '"expand_hint": "lcm_expand(store_id=3, '
        'content_offset=0)", '
        '"from_current_session": false, '
        '"arms": ["fts"], '
        '"store_id": 3}, '
        '{"kind": "message_excerpt", '
        '"session_id": "old", '
        '"timestamp": 1768464000.0, '
        '"snippet": ">>>cake<<< orchard orchard", '
        '"score": 0.004032, '
        '"expand_hint": "lcm_expand(store_id=1, '
        'content_offset=0)", '
        '"from_current_session": false, '
        '"arms": ["fts"], '
        '"store_id": 1}], '
        '"provenance": {"arms_run": ["fts"], '
        '"arm_weights": {"fts": 0.5}, '
        '"coverage": {"fts": "ok", '
        '"summary": "disabled", '
        '"chunk": "disabled"}, '
        '"fts_index": "stem", '
        '"rerank": "disabled", '
        '"ordering": "rrf-fusion -> scope/recency prior -> rerank reorder (top window); the reported score is the scope/recency-adjusted RRF score, '
        'and rerank (when applied) only permutes the top window without replacing that score"}, '
        '"metrics": {"embedding_query_calls": 0, '
        '"embedding_query_tokens": 0, '
        '"embedding_query_tokens_complete": true, '
        '"embedding_queries": []}, '
        '"degraded": true, '
        '"degraded_reason": "semantic retrieval is disabled"}'
    ),
    'grep': (
        '{"query": "cake", '
        '"sort": "recency", '
        '"session_scope": "all", '
        '"content_scope": "history", '
        '"source": null, '
        '"conversation_id": null, '
        '"limit": 10, '
        '"total_results": 2, '
        '"results": [{"type": "message", '
        '"depth": "raw", '
        '"store_id": 2, '
        '"session_id": "fresh", '
        '"source": "unknown", '
        '"conversation_id": "", '
        '"role": "user", '
        '"timestamp": 1800000000.0, '
        '"snippet": ">>>cake<<< garden", '
        '"from_current_session": false}, '
        '{"type": "message", '
        '"depth": "raw", '
        '"store_id": 1, '
        '"session_id": "old", '
        '"source": "unknown", '
        '"conversation_id": "", '
        '"role": "user", '
        '"timestamp": 1768464000.0, '
        '"snippet": ">>>cake<<< orchard orchard", '
        '"from_current_session": false}]}'
    ),
}


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "dispatch_background_stem_backfill", lambda *a: None)
    monkeypatch.setattr(tools.time, "time", lambda: NOW)
    monkeypatch.setattr(tools.time, "monotonic", lambda: 100.0)
    config = LCMConfig(database_path=str(tmp_path / "recall.db"), embeddings_enabled=False)
    store = MessageStore(config.database_path, ingest_protection_config=config)
    dag = SummaryDAG(config.database_path)
    old = store.append("old", {"role": "user", "content": "cake orchard orchard"})
    fresh = store.append("fresh", {"role": "user", "content": "cake garden"})
    plural = store.append("plural", {"role": "user", "content": "cakes for the party"})
    store.connection.execute("UPDATE messages SET timestamp=? WHERE store_id=?", (NOW - 365 * 86400, old))
    store.connection.commit()
    boot._run_stem_backfill(str(store.db_path), build_message_stem_fts_spec(), "l1l2-test")
    assert store._stem_state() == "ready"
    obj = SimpleNamespace(_config=config, _store=store, _dag=dag,
                          _hermes_home=str(tmp_path), current_session_id="active",
                          old=old, fresh=fresh, plural=plural)
    try:
        yield obj
    finally:
        dag.close()
        store.close()


def answers(engine):
    return {
        "recall": tools.lcm_recall({"query": "cake", "limit": 3}, engine=engine),
        "grep": tools.lcm_grep({"query": "cake", "session_scope": "all"}, engine=engine),
    }


def test_1_defaults_byte_identical(engine):
    absent = answers(engine)
    engine._config = LCMConfig(database_path=engine._config.database_path, embeddings_enabled=False,
                               recall_recency_floor=0.5, recall_grep_stemmed=False)
    assert absent == answers(engine) == BASELINE
    assert all(isinstance(value, str) for value in absent.values())


def test_2_recency_floor_ranking(engine, monkeypatch):
    # Deterministic ranked arms isolate the reported inversion from vector providers.
    hits = [{"kind": "message_excerpt", "store_id": sid, "session_id": session,
             "timestamp": stamp, "snippet": "cake", "content_offset": 0}
            for sid, session, stamp in ((engine.old, "old", NOW - 365 * 86400),
                                        (engine.fresh, "fresh", NOW))]
    arms = {"fts": [hits[0]], "chunk": hits}
    fuse = retrieval_core.fuse_candidates
    monkeypatch.setattr(tools, "fuse_candidates", lambda _arms, weights: fuse(arms, weights))
    for floor, expected in ((0.5, [engine.fresh, engine.old]), (1.0, [engine.old, engine.fresh])):
        engine._config.recall_recency_floor = floor
        result = json.loads(tools.lcm_recall({"query": "cake", "limit": 2}, engine=engine))
        assert [hit["store_id"] for hit in result["hits"]] == expected
    for floor in (0.0, 0.5, 1.0):
        for stamp in (None, 0, "invalid"):
            assert retrieval_core._lcm_recall_recency_boost(stamp, now=NOW, floor=floor) == floor
        assert retrieval_core._lcm_recall_recency_boost(NOW + 1, now=NOW, floor=floor) == 1.0
    assert retrieval_core._lcm_recall_recency_boost(NOW - 30 * 86400, now=NOW, floor=0.25) == 0.5


@pytest.mark.parametrize("value", [-0.1, 1.1, "x", True, None, float("nan"), float("inf")])
@pytest.mark.parametrize("source", ["config", "environment"])
def test_3_recency_floor_validation(monkeypatch, source, value):
    env = {"LCM_RECALL_RECENCY_FLOOR": str(value)} if source == "environment" else {}
    yaml = {"lcm": {"recall_recency_floor": value}} if source == "config" else {}
    monkeypatch.setattr(config_module, "lcm_env", lambda: env)
    monkeypatch.setattr(config_module, "_load_hermes_config_yaml_for_env", lambda _env: yaml)
    config = LCMConfig.from_env()
    label = "env LCM_RECALL_RECENCY_FLOOR" if source == "environment" else "config.yaml lcm.recall_recency_floor"
    assert config.config_source_warnings == [
        f"invalid {label}={value!r} ignored; using recall_recency_floor=0.5"
        if source == "config" else
        f"invalid {label}={str(value)!r} ignored; using recall_recency_floor=0.5"
    ]
    assert config.recall_recency_floor == 0.5
    assert config.config_sources["recall_recency_floor"] == "default"


@pytest.mark.parametrize("state", ["ready", "backfilling", "absent", "error"])
def test_4_grep_stemmed(engine, monkeypatch, state):
    args = {"query": "cake", "session_scope": "session", "session_id": "plural"}
    engine._config.recall_grep_stemmed = False
    assert json.loads(tools.lcm_grep({**args, "_stemmed": True}, engine=engine))["results"] == []
    engine._config.recall_grep_stemmed = True
    result = json.loads(tools.lcm_grep(args, engine=engine))
    assert [hit["store_id"] for hit in result["results"]] == [engine.plural]
    assert result["fts_index"] == "stem"
    if state == "backfilling":
        engine._store.connection.execute("UPDATE metadata SET value='backfilling' WHERE key='fts_stem_state'")
    elif state == "absent":
        engine._store.connection.execute("DELETE FROM metadata WHERE key='fts_stem_state'")
    elif state == "error":
        # A broken stem index uses the same plain-index fallback as recall.
        connection = engine._store.connection

        class ErrorConnection:
            def execute(self, sql, *args):
                if "messages_fts_stem MATCH" in sql:
                    raise sqlite3.OperationalError("injected stem failure")
                return connection.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(connection, name)

        monkeypatch.setattr(engine._store, "_conn", ErrorConnection())
    engine._store.connection.commit()
    if state != "ready":
        result = json.loads(tools.lcm_grep(args, engine=engine))
        assert result["results"] == []
        assert result["fts_index"] == "plain"


def test_5_grep_or_relevance_and_explicit_sort(engine):
    args = {"query": "cake OR orchard", "session_scope": "all", "limit": 2}
    engine._config.recall_grep_stemmed = False
    off = json.loads(tools.lcm_grep(args, engine=engine))
    assert off["sort"] == "recency"
    assert off["results"][0]["store_id"] == engine.fresh
    engine._config.recall_grep_stemmed = True
    on = json.loads(tools.lcm_grep(args, engine=engine))
    assert on["sort"] == "relevance"
    assert on["results"][0]["store_id"] == engine.old
    for sort in ("recency", None, ""):
        explicit = json.loads(tools.lcm_grep({**args, "sort": sort}, engine=engine))
        assert explicit["sort"] == "recency"
        assert explicit["results"][0]["store_id"] == engine.fresh
    for query in ('"cake OR orchard"', "cake or orchard", "(cake OR orchard)", "OR cake", "cake"):
        result = json.loads(tools.lcm_grep({**args, "query": query}, engine=engine))
        assert result["sort"] == "recency"


def test_6_fmp_options_reach_engine(tmp_path, monkeypatch):
    # Stores imports delivery only for build-time question_clock. Neither the parser
    # nor engine opening uses it; isolate the unrelated optional HTTP backends.
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("bench.fmp._l1l2_stores", root / "bench/fmp/stores.py")
    stores = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "bench.fmp.delivery", SimpleNamespace(question_clock=None))
        spec.loader.exec_module(stores)
    options = stores.parse_recall_options(["recall_recency_floor=1.0", "recall_grep_stemmed=true"])
    assert options == {"recall_grep_stemmed": True, "recall_recency_floor": 1.0}
    monkeypatch.setattr(stores, "commit", lambda _: "62a8137a2c46b0dd4bfa51102238d1c96e9523cd")
    product = stores.Stores(tmp_path / "stores", "off", root,
                            "synthetic", None, recall_options=options)
    question = SimpleNamespace(qid="l1l2")
    with product.engine(question, path=tmp_path) as (engine, _vectors):
        assert engine._config.recall_recency_floor == 1.0
        assert engine._config.recall_grep_stemmed is True
    for value in ("-0.1", "1.1", "x"):
        with pytest.raises(stores.RecallOptionError, match="invalid --recall-option"):
            stores.parse_recall_options([f"recall_recency_floor={value}"])
    for floor, stemmed in ((0.0, False), (0.25, True), (0.5, False), (1.0, True)):
        assert stores.parse_recall_options([
            f"recall_recency_floor={floor}", f"recall_grep_stemmed={stemmed}",
        ]) == {"recall_recency_floor": floor, "recall_grep_stemmed": stemmed}
        monkeypatch.setattr(config_module, "lcm_env", lambda: {})
        monkeypatch.setattr(config_module, "_load_hermes_config_yaml_for_env", lambda _env: {
            "lcm": {"recall_recency_floor": floor, "recall_grep_stemmed": stemmed},
        })
        config = LCMConfig.from_env()
        assert (config.recall_recency_floor, config.recall_grep_stemmed) == (floor, stemmed)
        assert config.config_sources["recall_recency_floor"] == "config_yaml:lcm.recall_recency_floor"
        assert config.config_sources["recall_grep_stemmed"] == "config_yaml:lcm.recall_grep_stemmed"
        assert config.config_source_warnings == []
        assert config.ignored_config_yaml_lcm_keys == []
        monkeypatch.setattr(config_module, "lcm_env", lambda: {
            "LCM_RECALL_RECENCY_FLOOR": "0.5", "LCM_RECALL_GREP_STEMMED": "false",
        })
        config = LCMConfig.from_env()
        assert (config.recall_recency_floor, config.recall_grep_stemmed) == (0.5, False)
        assert config.config_sources["recall_recency_floor"] == "env:LCM_RECALL_RECENCY_FLOOR"
        assert config.config_sources["recall_grep_stemmed"] == "env:LCM_RECALL_GREP_STEMMED"


@pytest.mark.parametrize("raw", ["", "   ", "maybe"])
def test_r2_stemmed_env_source(monkeypatch, raw):
    monkeypatch.setattr(config_module, "lcm_env", lambda: {"LCM_RECALL_GREP_STEMMED": raw})
    monkeypatch.setattr(config_module, "_load_hermes_config_yaml_for_env", lambda _env: {
        "lcm": {"recall_grep_stemmed": True},
    })
    config = LCMConfig.from_env()
    assert config.recall_grep_stemmed is (raw != "maybe")
    assert config.config_sources["recall_grep_stemmed"] == (
        "default" if raw == "maybe" else "config_yaml:lcm.recall_grep_stemmed")
    assert config.config_source_warnings == (
        ["invalid env LCM_RECALL_GREP_STEMMED ignored; expected bool"] if raw == "maybe" else [])


@pytest.mark.parametrize("consumer", ["proactive", "baseline", "requirements", "pack", "proposal", "auto"])
def test_r2_internal_callers_keep_recency_floor(engine, monkeypatch, consumer):
    from hermes_lcm import answer_contract, evidence_compiler, evidence_pack
    from hermes_lcm.engine import LCMEngine

    hits = [{"kind": "message_excerpt", "store_id": sid, "session_id": session,
             "timestamp": stamp, "snippet": "cake", "content_offset": 0}
            for sid, session, stamp in ((engine.old, "old", NOW - 365 * 86400),
                                        (engine.fresh, "fresh", NOW))]
    fuse = retrieval_core.fuse_candidates
    monkeypatch.setattr(tools, "fuse_candidates", lambda _arms, weights:
                        fuse({"fts": [hits[0]], "chunk": hits}, weights))
    args = {"query": "cake", "limit": 2}
    original = tools.lcm_recall
    expected = [hit["store_id"] for hit in json.loads(original(args, engine=engine))["hits"]]
    assert expected == [engine.fresh, engine.old]
    engine._config.recall_recency_floor = 1.0
    assert [hit["store_id"] for hit in json.loads(original(args, engine=engine))["hits"]] == [engine.old, engine.fresh]
    captured = []

    def capture(recall_args, **kwargs):
        result = original(recall_args, **kwargs)
        captured.append(json.loads(result))
        return result

    def consume(*a, retrieve, **kwargs):
        return {"context": retrieve(args)}

    monkeypatch.setattr(tools, "lcm_recall", capture)
    monkeypatch.setattr(evidence_pack, "build_evidence_pack", consume)
    monkeypatch.setattr(evidence_compiler, "compile_evidence", consume)
    monkeypatch.setattr(evidence_compiler, "compile_preanswer_evidence", consume)
    engine._disabled_tool_names = lambda: set()
    engine.handle_tool_call = lambda name, recall_args, **kw: LCMEngine.handle_tool_call(engine, name, recall_args, **kw)
    if consumer == "proactive":
        engine._proactive_recall_eligible = lambda: True
        engine._newest_user_message_text = LCMEngine._newest_user_message_text
        engine._wrap_relevant_memories = LCMEngine._wrap_relevant_memories
        engine._proactive_recall_skipped_count = engine._proactive_recall_injected_count = 0
        engine._config.proactive_recall_min_score = 0.0
        assert LCMEngine._build_proactive_recall_message(
            engine, [{"role": "user", "content": "cake"}], "user", set())
    elif consumer in {"baseline", "requirements"}:
        spec = importlib.util.spec_from_file_location("hermes_lcm.l1l2_hook", Path(tools.__file__).with_name("__init__.py"))
        spec.submodule_search_locations = None
        plugin = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plugin)
        if consumer == "baseline":
            plugin._answer_ready_baseline(engine, "cake", {})
        else:
            engine._config.preanswer_evidence_enabled = True
            engine._config.preanswer_evidence_mode = "requirements_v1"
            monkeypatch.setattr(answer_contract, "compile_answer_contract", lambda *a: SimpleNamespace(status="planned"))
            plugin._pre_llm_context(engine, {"user_message": "cake", "baseline_refs": []})
    elif consumer == "pack":
        tools.lcm_evidence_pack({}, engine=engine)
    else:
        tools.lcm_compile_evidence({"mode": consumer}, engine=engine)
    assert len(captured) == 1
    assert [hit["store_id"] for hit in captured[0]["hits"]] == expected


def test_r2_externalized_scope_omits_stale_fts_index(engine, monkeypatch):
    engine._config.recall_grep_stemmed = True
    history = json.loads(tools.lcm_grep({"query": "cake", "session_scope": "all"}, engine=engine))
    assert history["fts_index"] == engine._store._last_fts_index == "stem"

    def unexpected_search(*a, **kw):
        pytest.fail("externalized-only scope searched history")

    monkeypatch.setattr(engine._store, "search", unexpected_search)
    result = json.loads(tools.lcm_grep({"query": "cake", "content_scope": "externalized"}, engine=engine))
    assert "error" not in result
    assert result["content_scope"] == "externalized"
    assert "fts_index" not in result
