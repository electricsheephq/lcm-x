"""Session grouping is a response-only view of the same selected recall hits."""
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm import config, schemas, tools
from hermes_lcm.engine import LCMEngine
from tests import test_lcm_recall, test_recall_later_statements, test_recall_session_delivery

recall_engine = test_lcm_recall.recall_engine
internal_recall = test_recall_session_delivery.internal_recall
statements = test_recall_later_statements.statements
GOLDEN = Path(__file__).with_name("fixtures") / "recall_session_grouped_main_golden.json"


@pytest.fixture
def grouped_engine(recall_engine, monkeypatch):
    e = recall_engine
    e._config.embeddings_enabled = False
    e._config.recall_reference_strict = False
    e._omitted_tool_names = lambda: set()
    entries = []
    # Ranked order intentionally differs from session order and timestamp order.
    for i, (sid, stamp) in enumerate([("b", 300), ("a", 200), ("b", 100),
                                      ("a", 200), (None, 50), ("", 10), ("c", 100)]):
        monkeypatch.setattr(tools.time, "time", lambda stamp=stamp: stamp)
        store_id = e._store.append(sid or "unowned", {
            "role": "user", "content": f"orchard café 🍎 row {i}", "timestamp": stamp})
        entries.append(test_recall_later_statements.entry(e, store_id, session_id=sid))
    monkeypatch.setattr(tools.time, "time", lambda: 1700000000)
    test_recall_later_statements.ranked(monkeypatch, entries)
    return e


def invoke(e, detail="snippets", **kw):
    return tools.lcm_recall({"query": "orchard", "limit": 10, "detail": detail}, engine=e, **kw)


def snapshots(e):
    result = {}
    for guidance in ("v1", "v2"):
        e._config.recall_guidance = guidance
        for detail in ("snippets", "answer_ready"):
            result[f"{guidance}/{detail}"] = invoke(e, detail)
        result[f"{guidance}/schemas"] = json.dumps(LCMEngine.get_tool_schemas(e), sort_keys=True)
    return result


def iso(stamp):
    return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat()


def test_off_golden_and_schemas(grouped_engine):
    e = grouped_engine
    assert config.LCMConfig().recall_group_by_session is False
    expected = json.loads(GOLDEN.read_text())
    assert snapshots(e) == expected
    e._config.recall_group_by_session = False
    assert snapshots(e) == expected
    e._config.recall_group_by_session = True
    for guidance in ("v1", "v2"):
        e._config.recall_guidance = guidance
        assert json.dumps(LCMEngine.get_tool_schemas(e), sort_keys=True) == expected[f"{guidance}/schemas"]
    assert "group_by_session" not in json.dumps(schemas.LCM_RECALL)


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
@pytest.mark.parametrize("guidance", ["v1", "v2"])
def test_grouped_hits_ranks_content_ties_and_sessionless(grouped_engine, detail, guidance):
    e = grouped_engine
    e._config.recall_guidance = guidance
    off = json.loads(invoke(e, detail))
    e._config.recall_group_by_session = True
    before = e._store.connection.total_changes
    on = json.loads(invoke(e, detail))
    assert e._store.connection.total_changes == before
    assert [h["rank"] for h in on["hits"]] == [3, 1, 7, 2, 4, 5, 6]
    assert [{k: v for k, v in h.items() if k != "rank"}
            for h in sorted(on["hits"], key=lambda h: h["rank"])] == off["hits"]
    assert on["session_groups"] == [
        dict(session_id="b", first_at=iso(100), last_at=iso(300), hit_ranks=[3, 1]),
        dict(session_id="c", first_at=iso(100), last_at=iso(100), hit_ranks=[7]),
        dict(session_id="a", first_at=iso(200), last_at=iso(200), hit_ranks=[2, 4]),
        dict(session_id=None, first_at=iso(10), last_at=iso(50), hit_ranks=[5, 6]),
    ]
    assert on["sessions_covered"] == f"3 sessions; dates {iso(10)} to {iso(300)}"
    assert on["provenance"]["session_groups_dropped"] == 0


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_later_statement_keeps_own_group_and_links(statements, detail, monkeypatch):
    e, source, child = statements
    # Distinct write times are what response timestamp exposes; do not update rows.
    e._config.recall_later_statements = 1
    off = json.loads(test_recall_later_statements.invoke(e, detail))
    e._config.recall_group_by_session = True
    on = json.loads(test_recall_later_statements.invoke(e, detail))
    assert [h["store_id"] for h in on["hits"]] == [source, child]
    assert [h["rank"] for h in on["hits"]] == [1, 2]
    assert on["hits"][1]["later_statement_of"] == source
    assert on["hits"][0]["has_later_statements"] == [child]
    assert [g["session_id"] for g in on["session_groups"]] == ["old", "new"]
    assert [{k: v for k, v in h.items() if k != "rank"} for h in on["hits"]] == off["hits"]


# Whole-envelope cap: answer_ready only. Snippets keep main's hits-only budget; their grouped cap,
# forced-drop and query-echo cases are in tests/test_recall_session_grouped_fix1.py.
@pytest.mark.parametrize("detail", ["answer_ready"])
@pytest.mark.parametrize("forced", [False, True])
def test_cap_counts_drops_and_refreshes_groups(grouped_engine, monkeypatch, detail, forced):
    e = grouped_engine
    off_raw = invoke(e, detail)
    cap = len(off_raw) if forced else len(off_raw) + 1500
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", cap)
    off = json.loads(invoke(e, detail))
    e._config.recall_group_by_session = True
    raw = invoke(e, detail)
    on = json.loads(raw)
    assert len(raw) <= cap
    dropped = len(off["hits"]) - len(on["hits"])
    assert on["provenance"]["session_groups_dropped"] == dropped
    assert (dropped > 0) is forced
    assert [r for g in on["session_groups"] for r in g["hit_ranks"]] == [h["rank"] for h in on["hits"]]
    assert on["total_results"] == len(on["hits"])


@pytest.mark.parametrize("yaml", [False, True])
@pytest.mark.parametrize("raw,expected", [(True, True), (False, False), ("bad", False)])
def test_config_env_yaml_and_precedence(tmp_path, monkeypatch, yaml, raw, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if yaml:
        (tmp_path / "config.yaml").write_text("lcm:\n  recall_group_by_session: " + json.dumps(raw))
    else:
        monkeypatch.setenv("LCM_RECALL_GROUP_BY_SESSION", str(raw))
    cfg = config.LCMConfig.from_env()
    assert cfg.recall_group_by_session is expected
    assert "recall_group_by_session" not in cfg.ignored_config_yaml_lcm_keys
    source = ("default" if raw == "bad" else "config_yaml:lcm.recall_group_by_session"
              if yaml else "env:LCM_RECALL_GROUP_BY_SESSION")
    assert cfg.config_sources["recall_group_by_session"] == source
    monkeypatch.setenv("LCM_RECALL_GROUP_BY_SESSION", "true")
    assert config.LCMConfig.from_env().recall_group_by_session is True


@pytest.mark.parametrize("consumer", ["pack", "proposal", "auto", "requirements", "adaptive"])
def test_excluded_evidence_callers(internal_recall, monkeypatch, consumer):
    from hermes_lcm import answer_contract, evidence_compiler, evidence_pack
    e, plugin = internal_recall
    args = {"query": "needle", "detail": "answer_ready"}
    def consume(*a, retrieve, **kw):
        return {"context": retrieve(args)}
    monkeypatch.setattr(evidence_pack, "build_evidence_pack", consume)
    monkeypatch.setattr(evidence_compiler, "compile_evidence", consume)
    monkeypatch.setattr(evidence_compiler, "compile_preanswer_evidence", consume)
    monkeypatch.setattr(answer_contract, "compile_answer_contract", lambda *a: SimpleNamespace(status="planned"))
    e._config.preanswer_evidence_enabled = True
    e._config.preanswer_evidence_mode = "requirements_v1"
    e._adaptive_retrieval = SimpleNamespace(search=lambda dispatch, **kw:
        consume(retrieve=lambda recall_args: dispatch("lcm_recall", recall_args)))
    def call():
        if consumer == "adaptive":
            return json.loads(tools.lcm_retrieve({"action": "search"}, engine=e))
        if consumer == "requirements":
            return plugin._pre_llm_context(e, {"user_message": "needle", "baseline_refs": []})
        if consumer == "pack":
            return tools.lcm_evidence_pack({}, engine=e)
        return json.loads(tools.lcm_compile_evidence({"question": "needle", "mode": consumer}, engine=e))
    expected = call()
    e._config.recall_group_by_session = True
    assert call() == expected
    assert "session_groups" in json.loads(e.handle_tool_call("lcm_recall", args))


def test_proactive_and_baseline_pin_off(grouped_engine, monkeypatch):
    e = grouped_engine
    e._config.recall_group_by_session = True
    e._config.proactive_recall_enabled = True
    e._config.embeddings_enabled = True
    calls = []
    monkeypatch.setattr(tools, "lcm_recall", lambda args, **kw: calls.append(kw) or '{"hits": []}')
    fake = SimpleNamespace(_config=e._config, _proactive_recall_eligible=lambda: True,
        _newest_user_message_text=LCMEngine._newest_user_message_text, _proactive_recall_skipped_count=0)
    LCMEngine._build_proactive_recall_message(fake, [{"role": "user", "content": "orchard"}], "user", set())
    spec = importlib.util.spec_from_file_location("hermes_lcm.grouped_hook", Path(tools.__file__).with_name("__init__.py"))
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    e._disabled_tool_names = lambda: set()
    e.handle_tool_call = lambda *a, **kw: LCMEngine.handle_tool_call(e, *a, **kw)
    plugin._answer_ready_baseline(e, "orchard", {})
    assert len(calls) == 2
    assert all(call.get("group_by_session") is False for call in calls)


def test_grep_load_and_empty_unchanged(grouped_engine):
    e = grouped_engine
    args = {"query": "orchard"}
    expected = tools.lcm_grep(args, engine=e), tools.lcm_load_session({"session_id": "a"}, engine=e)
    e._config.recall_group_by_session = True
    assert (tools.lcm_grep(args, engine=e), tools.lcm_load_session({"session_id": "a"}, engine=e)) == expected
    assert tools.lcm_recall({"query": ""}, engine=e) == '{"error": "No query provided"}'


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_session_delivery_counts_grouping_in_own_budget(recall_engine, monkeypatch, detail):
    from hermes_lcm import recall_session
    e = recall_engine
    e._config.embeddings_enabled = False
    e._config.recall_delivery_unit = "session"
    call = {"id": "c", "function": {"name": "read", "arguments": "{}"}}
    monkeypatch.setattr(tools.time, "time", lambda: 1700000000)
    a = e._store.append("one", {"role": "assistant", "content": "orchard café", "tool_calls": [call]})
    b = e._store.append("one", {"role": "tool", "content": "orchard result", "tool_call_id": "c"})
    e._store.append("two", {"role": "user", "content": "orchard other"})
    off = json.loads(invoke(e, detail))
    e._config.recall_group_by_session = True
    on = json.loads(invoke(e, detail))
    assert [{k: v for k, v in h.items() if k != "rank"} for h in on["hits"]] == off["hits"]
    assert on["provenance"]["delivery"]["response_chars"] == len(json.dumps(on, ensure_ascii=False))
    ranks = {h["store_id"]: i for i, h in enumerate(off["hits"], 1)}
    assert {g["session_id"]: g["hit_ranks"] for g in on["session_groups"]}["one"] == [ranks[a], ranks[b]]
    monkeypatch.setattr(recall_session, "TOKEN_BUDGET", off["provenance"]["delivery"]["response_tokens"])
    fitted = json.loads(invoke(e, detail))
    meta = fitted["provenance"]["delivery"]
    ids = {h["store_id"] for h in fitted["hits"]}
    assert ({a, b} <= ids) or not ({a, b} & ids)
    assert meta["response_tokens"] <= meta["token_budget"]
    assert meta["response_chars"] == len(json.dumps(fitted, ensure_ascii=False))
    assert fitted["provenance"]["session_groups_dropped"] == len(off["hits"]) - len(fitted["hits"])
