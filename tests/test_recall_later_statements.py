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
    monkeypatch.setattr(tools.time, "time", lambda: 1700000200)
    earlier = e._store.append("old", {"role": "user", "content": "orchidpassport My favorite hobby is pottery.", "timestamp": 1700000000})
    later = e._store.append("new", {"role": "user", "content": "My favorite hobby is now watercolor painting.", "timestamp": 1700000100})
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


def entry(e, store_id, **fields):
    row = e._store.get(store_id)
    hit = {"kind": "message_excerpt", "store_id": store_id, "session_id": row["session_id"],
           "timestamp": row["timestamp"], "snippet": row["content"], "role": row["role"], **fields}
    return {"hit": hit, "rrf_score": .02, "_final_score": .02, "ranks": {0: 1}}


def ranked(monkeypatch, entries):
    """Freeze ranking; all delivery and exact-row reads stay real."""
    monkeypatch.setattr(tools, "fuse_candidates", lambda *a, **kw: (entries, ["fts"], [1.0]))
    monkeypatch.setattr(tools, "apply_prior", lambda entries, *a: entries)


def assert_links(payload):
    hits = {h["store_id"]: h for h in payload["hits"]}
    children = [h for h in hits.values() if "later_statement_of" in h]
    for h in children:
        assert h["later_statement_of"] in hits
        assert h["store_id"] in hits[h["later_statement_of"]]["has_later_statements"]
    for h in hits.values():
        for sid in h.get("has_later_statements", []):
            assert hits[sid]["later_statement_of"] == h["store_id"]
    assert payload["provenance"]["later_statements"]["delivered"] == len(children)
    assert payload["total_results"] == len(payload["hits"])


@pytest.mark.parametrize("filter_kind", ["delta", "citation", "per_session"])
def test_filtered_source_is_never_expanded(statements, monkeypatch, filter_kind):
    e, source, child = statements
    entries = [entry(e, source)]
    args = {"query": "orchidpassport", "detail": "answer_ready", "limit": 4}
    if filter_kind == "delta":
        args["seen_refs"] = [f"lcm:{source}:0-{len(e._store.get(source)['content'])}"]
    elif filter_kind == "citation":
        get_batch = e._store.get_batch
        monkeypatch.setattr(e._store, "get_batch", lambda ids: {sid: row for sid, row in get_batch(ids).items() if sid != source})
    else:
        other = e._store.append("old", {"role": "user", "content": "first old-session row"})
        entries.insert(0, entry(e, other))
        monkeypatch.setattr(tools, "_LCM_RECALL_ANSWER_READY_PER_SESSION_LIMIT", 1)
    ranked(monkeypatch, entries)
    place = tools._lcm_recall_later_statements
    windows = []

    def capture(engine, ordered, **kw):
        windows.append([v["hit"]["store_id"] for v in ordered])
        return place(engine, ordered, **kw)

    monkeypatch.setattr(tools, "_lcm_recall_later_statements", capture)
    payload = json.loads(tools.lcm_recall(args, engine=e, later_statements=1))
    assert source not in [h["store_id"] for h in payload["hits"]]
    assert child not in [h["store_id"] for h in payload["hits"]]
    assert len(windows) == 1 and source not in windows[0]
    assert_links(payload)


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_later_hydrates_own_deep_fts_match(statements, monkeypatch, detail):
    e, source, _ = statements
    text = "unrelated filler " * 500 + "My favorite hobby is now silver jewelry making."
    child = e._store.append("deep", {"role": "user", "content": text, "timestamp": 1700000150})
    original = tools._lcm_recall_fts_arm

    def search(engine, query, **kw):
        hits, error = original(engine, query, **kw)
        return ([h for h in hits if h["store_id"] in (source, child)], error)

    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", search)
    hydrate = Mock(wraps=tools._lcm_recall_answer_ready_content)
    monkeypatch.setattr(tools, "_lcm_recall_answer_ready_content", hydrate)
    payload = json.loads(invoke(e, detail, later_statements=1, snippet_window_hits=2))
    hit = next(h for h in payload["hits"] if h["store_id"] == child)
    assert "later_statement" in hit["arms"]
    assert "silver jewelry making" in hit.get("content", hit["snippet"])
    if detail == "answer_ready":
        batches = [call.args[1] for call in hydrate.call_args_list]
        assert [[v["hit"]["store_id"] for v in batch] for batch in batches] == [[source], [child]]
        assert hydrate.call_args.kwargs["expanded_limit"] == 1
        assert hit["content_offset"] > 6000
        assert text[hit["content_offset"]:hit["content_offset"] + hit["content_returned_chars"]] == hit["content"]
        assert hit["evidence_span"]["char_start"] > 6000
    assert_links(payload)


def test_retrieve_nested_recall_pins_off(statements):
    from hermes_lcm import evidence_tools
    e, _, _ = statements
    e._config.recall_later_statements = 2
    calls = []

    def dispatch(name, args, **kw):
        calls.append(kw)
        return tools.lcm_recall(args, engine=e, **kw)

    def search(*, dispatch, **kw):
        return json.loads(dispatch("lcm_recall", {"query": "orchidpassport"}))

    e.handle_tool_call = dispatch
    e._adaptive_retrieval = SimpleNamespace(search=search)
    result = json.loads(evidence_tools.lcm_retrieve({"action": "search"}, engine=e))
    assert calls[0].get("later_statements") == 0
    assert "later_statements" not in result["provenance"]


@pytest.mark.parametrize("shape", ["summary", "node", "adjacent", "delta", "beyond_window"])
def test_only_eligible_window_sources_spend_searches(statements, monkeypatch, shape):
    e, source, child = statements
    ignored = e._store.append("ignored", {"role": "user", "content": "ignored source text", "timestamp": 1699999900})
    skip = entry(e, ignored)
    args = {"query": "orchidpassport", "limit": 4, "detail": "answer_ready"}
    if shape == "summary":
        skip["hit"].update(kind="summary", node_id=123)
    elif shape == "node":
        skip["hit"]["node_id"] = 123
    elif shape == "adjacent":
        skip["hit"]["adjacent_to"] = source
    elif shape == "delta":
        args["seen_refs"] = [f"lcm:{ignored}:0-{len(e._store.get(ignored)['content'])}"]
    else:
        args["limit"] = 1
        skip["hit"]["adjacent_to"] = ignored
    entries = [skip, entry(e, source)]
    ranked(monkeypatch, entries)
    original = tools._lcm_recall_fts_arm
    searches = []

    def search(engine, query, **kw):
        if query != "orchidpassport":
            searches.append(query)
        return original(engine, query, **kw)

    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", search)
    result = json.loads(tools.lcm_recall(args, engine=e, later_statements=1))
    assert searches == ([] if shape == "beyond_window" else [e._store.get(source)["content"]])
    assert result["provenance"]["later_statements"]["searched"] == len(searches)
    if shape != "beyond_window":
        assert any(h["store_id"] == child and h.get("later_statement_of") == source for h in result["hits"])


@pytest.mark.parametrize("source_observed,candidate_observed", [(1700000000, None), (None, 1700000100)])
def test_no_mixed_observation_write_time_basis(statements, source_observed, candidate_observed):
    e, source, child = statements
    e._store.connection.executemany("UPDATE messages SET observed_at=?, timestamp=? WHERE store_id=?",
                                  [(source_observed, 1699999900, source), (candidate_observed, 1700000100, child)])
    e._store.connection.commit()
    payload = json.loads(invoke(e, later_statements=1))
    assert payload["provenance"]["later_statements"]["found"] == 0
    assert [h["store_id"] for h in payload["hits"]] == [source]


@pytest.mark.parametrize("failure", ["identity", "event_time"])
def test_failed_time_identity_read_skips_expansion(statements, monkeypatch, failure):
    e, source, _ = statements
    if failure == "identity":
        monkeypatch.setattr(tools, "duplicate_identity_rows", lambda *a, **kw: {})
    else:
        monkeypatch.setattr(tools, "_lcm_recall_event_time_rows", lambda *a, **kw: ({}, "read_error"))
    payload = json.loads(invoke(e, later_statements=1))
    assert payload["provenance"].get("later_statements_unavailable") == "read_error"
    assert [h["store_id"] for h in payload["hits"]] == [source]
    assert_links(payload)


def test_write_time_fallback_when_both_observations_absent(statements):
    e, source, child = statements
    e._store.connection.executemany("UPDATE messages SET observed_at=NULL, timestamp=? WHERE store_id=?",
                                  [(1700000000, source), (1700000100, child)])
    e._store.connection.commit()
    payload = json.loads(invoke(e, later_statements=1))
    assert [h["store_id"] for h in payload["hits"]] == [source, child]
    assert_links(payload)


def test_candidate_beyond_window_is_moved_once(statements, monkeypatch):
    e, source, child = statements
    fillers = [e._store.append(f"filler-{i}", {"role": "user", "content": "unrelated"}) for i in range(3)]
    ranked(monkeypatch, [entry(e, sid) for sid in [source, *fillers, child]])
    result = json.loads(invoke(e, later_statements=1))
    ids = [h["store_id"] for h in result["hits"]]
    assert ids[:2] == [source, child]
    assert ids.count(child) == 1
    assert result["provenance"]["later_statements"]["delivered"] == 1
    assert_links(result)


def test_normalized_full_text_copy_is_skipped(statements, monkeypatch):
    e, source, child = statements
    copy = "ＯＲＣＨＩＤＰＡＳＳＰＯＲＴ  my FAVORITE\n hobby is pottery."
    copy_id = e._store.append("copy", {"role": "user", "content": copy, "timestamp": 1700000050})
    original = tools._lcm_recall_fts_arm

    def search(engine, query, **kw):
        hits, error = original(engine, query, **kw)
        if query == "orchidpassport":
            return [h for h in hits if h["store_id"] == source], error
        by_id = {h["store_id"]: h for h in hits}
        return [by_id[copy_id], by_id[child]], error

    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", search)
    result = json.loads(invoke(e, later_statements=1))
    assert [h["store_id"] for h in result["hits"]] == [source, child]
    assert_links(result)


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_time_order_keeps_intervening_hit(statements, monkeypatch, detail):
    e, source, child = statements
    middle = e._store.append("middle", {"role": "user", "content": "orchidpassport intervening observation", "timestamp": 1700000050})
    ranked(monkeypatch, [entry(e, source), entry(e, middle)])
    original = tools._lcm_recall_fts_arm

    def search(engine, query, **kw):
        hits, error = original(engine, query, **kw)
        return [h for h in hits if h["store_id"] != middle], error

    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", search)
    result = json.loads(invoke(e, detail, later_statements=1, delivery_order="time"))
    assert [h["store_id"] for h in result["hits"]] == [source, middle, child]
    assert_links(result)


def test_engine_dispatch_forwards_pin_off(statements):
    from hermes_lcm import engine as engine_module, evidence_tools
    e, _, _ = statements
    e._config.recall_later_statements = 2
    e._session_id = None
    e._disabled_tool_names = lambda: set()
    e.handle_tool_call = engine_module.LCMEngine.handle_tool_call.__get__(e)
    args = {"query": "orchidpassport", "limit": 4}
    assert json.loads(e.handle_tool_call("lcm_recall", args))["provenance"]["later_statements"]["delivered"] == 1
    assert "later_statements" not in json.loads(e.handle_tool_call("lcm_recall", args, later_statements=0))["provenance"]
    results = []

    def search(*, dispatch, **kw):
        results.append(json.loads(dispatch("lcm_recall", dict(args))))
        return results[-1]

    e._adaptive_retrieval = SimpleNamespace(search=search)
    evidence_tools.lcm_retrieve({"action": "search"}, engine=e)
    assert results and "later_statements" not in results[0]["provenance"]


def observe(e, stamps):
    e._store.connection.executemany("UPDATE messages SET observed_at=? WHERE store_id=?", [(t, sid) for sid, t in stamps.items()])
    e._store.connection.commit()


def followups(monkeypatch, e, answers):
    """Pin each follow-up search (keyed by its source row) to the given rows."""
    original = tools._lcm_recall_fts_arm
    by_text = {e._store.get(src)["content"]: rows for src, rows in answers.items()}

    def search(engine, query, **kw):
        hits, error = original(engine, query, **kw)
        if query not in by_text:
            return hits, error
        found, _ = original(engine, "watercolor", **{**kw, "candidate_limit": 50})
        by_id = {h["store_id"]: h for h in found}
        return [by_id[sid] for sid in by_text[query]], error

    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", search)


def test_replay_copies_found_by_two_sources_take_one_slot(statements, monkeypatch):
    e, source, child = statements
    second = e._store.append("old2", {"role": "user", "content": "orchidpassport My favorite hobby was pottery.", "timestamp": 1700000010})
    copy = e._store.append("new", {"role": "user", "content": "My favorite hobby is now watercolor painting.", "timestamp": 1700000100})
    observe(e, {source: 1700000000, second: 1700000010, child: 1700000100, copy: 1700000100})
    ranked(monkeypatch, [entry(e, source), entry(e, second)])
    followups(monkeypatch, e, {source: [child], second: [copy]})
    result = json.loads(invoke(e, later_statements=2))
    ids = [h["store_id"] for h in result["hits"]]
    assert len({child, copy} & set(ids)) == 1
    assert result["provenance"]["later_statements"]["found"] == 1
    assert_links(result)


def test_searched_group_keeps_aliases(statements, monkeypatch):
    e, source, child = statements
    second = e._store.append("old2", {"role": "user", "content": "orchidpassport My favorite hobby was pottery.", "timestamp": 1700000010})
    alias = e._store.append("new", {"role": "user", "content": "My favorite hobby is now watercolor painting.", "timestamp": 1700000100})
    fillers = [e._store.append(f"filler-{i}", {"role": "user", "content": "unrelated"}) for i in range(2)]
    observe(e, {source: 1700000000, second: 1700000010, child: 1700000100, alias: 1700000100})
    ranked(monkeypatch, [entry(e, source), entry(e, second), *[entry(e, f) for f in fillers],
                         entry(e, child, duplicate_store_ids=[alias])])
    followups(monkeypatch, e, {source: [child, alias], second: [alias]})
    result = json.loads(invoke(e, later_statements=2))
    ids = [h["store_id"] for h in result["hits"]]
    assert child in ids and alias not in ids
    assert next(h for h in result["hits"] if h["store_id"] == child).get("duplicate_store_ids") == [alias]
    assert result["provenance"]["later_statements"]["found"] == 1
    assert_links(result)

def test_anchor_child_is_ordinary_when_source_not_selected(statements, monkeypatch):
    e, source, child = statements
    fillers = [e._store.append(f"filler-{i}", {"role": "user", "content": "orchidpassport unrelated"}) for i in range(3)]
    ranked(monkeypatch, [*[entry(e, f) for f in fillers], entry(e, source), entry(e, child)])

    def anchor(ordered, *a, **kw):
        # #950 places the best FTS hit in the last window slot, here the child.
        moved = next(x for x in ordered if x["hit"]["store_id"] == child)
        rest = [x for x in ordered if x is not moved]
        return [*rest[:3], moved, *rest[3:]], None

    monkeypatch.setattr(tools, "place_fts_anchor", anchor)
    result = json.loads(invoke(e, later_statements=4))
    assert result["provenance"]["later_statements"]["found"] == 0
    ids = [h["store_id"] for h in result["hits"]]
    assert ids == [*fillers, child]
    assert "later_statement" not in result["hits"][-1]["arms"]
    assert "_later_pooled" not in json.dumps(result)
    assert_links(result)


def test_per_session_quota_keeps_full_window(statements, monkeypatch):
    e, source, child = statements
    prefix = [e._store.append("old", {"role": "user", "content": f"NeonDiff context {i}", "timestamp": 1699999900 + i}) for i in range(4)]
    quota_child = e._store.append("old", {"role": "user", "content": "watercolor NeonDiff update", "timestamp": 1700000100})
    tail = [e._store.append(f"tail-{i}", {"role": "user", "content": f"citable tail {i}"}) for i in range(3)]
    ranked(monkeypatch, [entry(e, sid) for sid in [*prefix, source, *tail]])
    followups(monkeypatch, e, {prefix[0]: [quota_child], **{sid: [] for sid in prefix[1:]}, source: [child]})
    result = json.loads(tools.lcm_recall({"query": "orchidpassport", "detail": "answer_ready", "limit": 8}, engine=e, later_statements=5))
    assert len(result["hits"]) == 8
    assert source in [h["store_id"] for h in result["hits"]]
    assert_links(result)


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_last_plain_hit_displaced_to_tail(statements, monkeypatch, detail):
    e, source, child = statements
    fillers = [e._store.append(f"tail-{i}", {"role": "user", "content": f"plain hit {i}"}) for i in range(4)]
    ranked(monkeypatch, [entry(e, sid) for sid in [source, *fillers]])
    result = json.loads(invoke(e, detail, later_statements=1, extra_entries=1))
    assert [h["store_id"] for h in result["hits"][:4]] == [source, child, *fillers[:2]]
    assert result["provenance"]["later_statements"]["displaced"] == 1
    if detail == "snippets":
        assert result["hits"][4]["store_id"] == fillers[2]
        assert result["hits"][4]["extra"] is True
        result["hits"] = result["hits"][:4]
    assert_links(result)


def test_pair_cannot_evict_its_source(statements):
    e, source, _ = statements
    result = json.loads(tools.lcm_recall({"query": "orchidpassport", "limit": 1}, engine=e, later_statements=1))
    assert [h["store_id"] for h in result["hits"]] == [source]
    assert result["provenance"]["later_statements"]["displaced"] == 0
    assert_links(result)


@pytest.mark.parametrize("mode", ["delta", "reference_strict"])
def test_child_must_have_novel_citable_ref(statements, monkeypatch, mode):
    e, source, child = statements
    ranked(monkeypatch, [entry(e, source)])
    args = {"query": "orchidpassport", "detail": "answer_ready", "limit": 4}
    hydrate = tools._lcm_recall_answer_ready_content
    batches = []

    def capture(engine, entries, **kw):
        ids = [v["hit"]["store_id"] for v in entries]
        batches.append(ids)
        if mode == "reference_strict":
            entries = [v for v in entries if v["hit"]["store_id"] != child]
        return hydrate(engine, entries, **kw)

    monkeypatch.setattr(tools, "_lcm_recall_answer_ready_content", capture)
    if mode == "delta":
        args["seen_refs"] = [f"lcm:{child}:0-{len(e._store.get(child)['content'])}"]
    result = json.loads(tools.lcm_recall(args, engine=e, later_statements=1))
    assert [h["store_id"] for h in result["hits"]] == [source]
    assert [ids for ids in batches if child in ids] == [[child]]
    assert_links(result)


def test_anchor_source_expands_without_splitting_pair(statements, monkeypatch):
    e, source, child = statements
    fillers = [e._store.append(f"tail-{i}", {"role": "user", "content": f"plain hit {i}"}) for i in range(4)]
    ranked(monkeypatch, [entry(e, sid) for sid in [*fillers, source]])
    anchor = tools.place_fts_anchor

    def two_arms(ordered, arm_hits, arm_order, limit, **kw):
        return anchor(ordered, {"fts": [entry(e, source)["hit"]]}, ["fts", "chunk"], limit, arm_weights=[.5, .5])

    monkeypatch.setattr(tools, "place_fts_anchor", two_arms)
    result = json.loads(invoke(e, later_statements=4))
    assert [h["store_id"] for h in result["hits"]] == [*fillers[:2], source, child]
    assert_links(result)


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_fitting_pops_child_before_source(statements, monkeypatch, detail):
    e, source, _ = statements
    ranked(monkeypatch, [entry(e, source)])
    base = json.loads(invoke(e, detail, later_statements=1, snippet_window_hits=2))
    child = base["hits"][-1]
    # Keep room for the envelope and one source, but not its child.
    cap = len(json.dumps(base, ensure_ascii=detail != "answer_ready")) - len(json.dumps(child)) + 80
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", cap)
    result = json.loads(invoke(e, detail, later_statements=1, snippet_window_hits=2))
    assert [h["store_id"] for h in result["hits"]] == [source]
    assert "has_later_statements" not in result["hits"][0]
    assert result["provenance"]["later_statements"]["displaced"] == 0
    assert_links(result)
