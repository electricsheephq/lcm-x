"""Later statements are bounded read-side additions, with the default unchanged."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_lcm import config, schemas, tools
from tests import test_lcm_recall

recall_engine = test_lcm_recall.recall_engine


@pytest.fixture
def statements(recall_engine, monkeypatch):
    e = recall_engine
    e._config.embeddings_enabled = False
    earlier = e._store.append("old", {"role": "user", "content": "orchidpassport My favorite hobby is pottery.", "timestamp": 1700000000})
    later = e._store.append("new", {"role": "user", "content": "My favorite hobby is now watercolor painting.", "timestamp": 1700000100})
    monkeypatch.setattr(tools.time, "time", lambda: 1700000200)
    return e, earlier, later


def invoke(e, detail="snippets", **kw):
    return tools.lcm_recall({"query": "orchidpassport", "limit": 4, "detail": detail}, engine=e, **kw)


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_off_matches_unset_and_origin_golden(statements, monkeypatch, detail):
    e, _, _ = statements
    assert config.LCMConfig().recall_later_statements == 0
    e._config = SimpleNamespace(**{k: v for k, v in vars(e._config).items() if k != "recall_later_statements"})
    reader = Mock(side_effect=AssertionError("off must not search"))
    monkeypatch.setattr(tools, "_lcm_recall_later_statements", reader)
    unset = invoke(e, detail)
    e._config.recall_later_statements = 0
    assert invoke(e, detail) == unset == invoke(e, detail, later_statements=0)
    golden = json.loads((Path(__file__).parent / "fixtures/recall_later_statements_golden.json").read_text())
    assert unset == golden[detail]
    reader.assert_not_called()
    encoded = json.dumps(schemas.LCM_RECALL, sort_keys=True)
    assert hashlib.sha256(encoded.encode()).hexdigest() == "2489582d63ae6365e7814027170a946f02f72d2485bcc38584ad442a00cf6013"
    assert "later_statements" not in encoded


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_on_places_later_row_with_links_and_receipt(statements, monkeypatch, detail):
    e, earlier, later = statements
    e._config.recall_later_statements = 2
    changes = e._store.connection.total_changes
    connect = tools.sqlite3.connect
    connections, sql = [], []

    def track(path, **kw):
        connections.append((str(path), kw))
        conn = connect(path, **kw)
        conn.set_trace_callback(sql.append)
        return conn

    monkeypatch.setattr(tools.sqlite3, "connect", track)
    payload = json.loads(invoke(e, detail))
    assert [h["store_id"] for h in payload["hits"]] == [earlier, later]
    source, child = payload["hits"]
    assert source["has_later_statements"] == [later]
    assert child["later_statement_of"] == earlier
    assert source["event_time"] < child["event_time"]
    if detail == "answer_ready":
        assert child["content"] == "My favorite hobby is now watercolor painting."
    receipt = payload["provenance"]["later_statements"]
    assert {k: receipt[k] for k in ("requested", "searched", "found", "delivered")} == dict(requested=2, searched=1, found=1, delivered=1)
    assert receipt["ms"] >= 0
    assert "superseded" not in json.dumps(payload)
    assert e._store.connection.total_changes == changes
    assert all("mode=ro" in path and kw.get("uri") for path, kw in connections)
    assert all(not stmt.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "ALTER")) for stmt in sql)


def test_already_retrieved_is_not_placed_twice(statements):
    e, earlier, later = statements
    e._config.recall_later_statements = 2
    payload = json.loads(tools.lcm_recall({"query": "favorite hobby", "limit": 4}, engine=e))
    assert sorted(h["store_id"] for h in payload["hits"]) == [earlier, later]
    assert payload["provenance"]["later_statements"]["found"] == 0


@pytest.mark.parametrize("stamp", [1699999900, 1700000000, 1700000300])
def test_never_older_equal_or_future(statements, monkeypatch, stamp):
    e, _, later = statements
    original = tools._lcm_recall_event_time_rows

    def times(engine, ids, **kw):
        rows, reason = original(engine, ids, **kw)
        if later in rows:
            rows[later]["observed_at"] = stamp
        return rows, reason

    monkeypatch.setattr(tools, "_lcm_recall_event_time_rows", times)
    payload = json.loads(invoke(e, later_statements=2))
    assert len(payload["hits"]) == 1
    assert payload["provenance"]["later_statements"]["delivered"] == 0


@pytest.mark.parametrize("consumer", ["pack", "proposal", "auto"])
def test_internal_callers_pin_off(statements, monkeypatch, consumer):
    from hermes_lcm import evidence_compiler, evidence_pack
    e, _, _ = statements
    e._config.recall_later_statements = 2
    original = tools.lcm_recall
    calls = []

    def capture(*a, **kw):
        calls.append(kw)
        return original(*a, **kw)

    def consume(*a, retrieve, **kw):
        return {"context": retrieve({"query": "orchidpassport"})}

    monkeypatch.setattr(tools, "lcm_recall", capture)
    monkeypatch.setattr(evidence_pack, "build_evidence_pack", consume)
    monkeypatch.setattr(evidence_compiler, "compile_evidence", consume)
    monkeypatch.setattr(evidence_compiler, "compile_preanswer_evidence", consume)
    result = tools.lcm_evidence_pack({}, engine=e) if consumer == "pack" else json.loads(tools.lcm_compile_evidence({"mode": consumer}, engine=e))
    assert calls and all(c["later_statements"] == 0 for c in calls)
    assert "later_statements" not in json.loads(result["context"])["provenance"]
    assert json.loads(invoke(e))["provenance"]["later_statements"]["delivered"] == 1


def test_exhausted_deadline_is_normal_result_with_marker(statements, monkeypatch):
    e, _, _ = statements
    original = tools._lcm_recall_later_statements

    def expired(*a, **kw):
        return original(*a, **{**kw, "deadline": -1})

    monkeypatch.setattr(tools, "_lcm_recall_later_statements", expired)
    base = json.loads(invoke(e))
    result = json.loads(invoke(e, later_statements=2))
    assert result["provenance"].pop("later_statements_unavailable") == "deadline"
    receipt = result["provenance"].pop("later_statements")
    assert receipt["searched"] == receipt["found"] == receipt["delivered"] == 0
    assert result == base


@pytest.mark.parametrize("yaml", [False, True])
@pytest.mark.parametrize("raw,expected", [(0, 0), (2, 2), (5, 5), (40, 5), (-1, 0), (True, 0), (1.5, 0), ("bad", 0), ("2", 0)])
def test_config_parsing(monkeypatch, tmp_path, yaml, raw, expected):
    for key in tuple(config.os.environ):
        if key.startswith("LCM_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if yaml:
        (tmp_path / "config.yaml").write_text("lcm:\n  recall_later_statements: " + json.dumps(raw))
    else:
        monkeypatch.setenv("LCM_RECALL_LATER_STATEMENTS", str(raw))
        expected = 2 if raw == "2" else expected
    cfg = config.LCMConfig.from_env()
    assert cfg.recall_later_statements == expected
    invalid = isinstance(raw, (bool, float)) or raw in (-1, "bad") or (yaml and raw == "2")
    assert bool([w for w in cfg.config_source_warnings if "recall_later_statements" in w]) == invalid
    source = "default" if invalid else "config_yaml:lcm.recall_later_statements" if yaml else "env:LCM_RECALL_LATER_STATEMENTS"
    assert cfg.config_sources["recall_later_statements"] == source
    assert "recall_later_statements" not in cfg.ignored_config_yaml_lcm_keys
    monkeypatch.setenv("LCM_RECALL_LATER_STATEMENTS", "3")
    assert config.LCMConfig.from_env().recall_later_statements == 3


def test_all_existing_internal_pins_include_later_statements():
    root = Path(__file__).resolve().parents[1]
    for path in ["engine.py", "__init__.py", "evidence_tools.py", "bench/instruments/scale389/phase1a.py",
                 "benchmarks/qa-harness/src/providers/hermes-lcm/bridge/hermes_lcm_bridge.py"]:
        calls = [n for n in ast.walk(ast.parse((root / path).read_text())) if isinstance(n, ast.Call)
                 and any(k.arg == "adjacent_reply" and isinstance(k.value, ast.Constant) and k.value.value is False for k in n.keywords)]
        assert calls
        assert all(any(k.arg == "later_statements" and isinstance(k.value, ast.Constant) and k.value.value == 0 for k in n.keywords) for n in calls)


def test_read_error_rolls_back(statements, monkeypatch):
    e, _, _ = statements
    base = json.loads(invoke(e))
    original = tools._lcm_recall_fts_arm
    calls = 0

    def fail_followup(*a, **kw):
        nonlocal calls
        calls += 1
        return original(*a, **kw) if calls == 1 else ([], {"error": "private failure"})

    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", fail_followup)
    result = json.loads(invoke(e, later_statements=2))
    assert result["provenance"].pop("later_statements_unavailable") == "read_error"
    result["provenance"].pop("later_statements")
    assert result == base


def test_session_delivery_skips_option(statements, monkeypatch):
    e, _, _ = statements
    reader = Mock(side_effect=AssertionError("session search"))
    monkeypatch.setattr(tools, "_lcm_recall_later_statements", reader)
    assert invoke(e, delivery_unit="session", later_statements=2) == invoke(e, delivery_unit="session", later_statements=0)
    reader.assert_not_called()
