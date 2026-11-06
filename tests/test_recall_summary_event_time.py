"""Opt-in summary event-time span (LCM_RECALL_SUMMARY_EVENT_TIME): read path only."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_lcm import config, presets, recall_delivery, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from tests import test_lcm_recall, test_recall_session_delivery

recall_engine = test_lcm_recall.recall_engine
internal_recall = test_recall_session_delivery.internal_recall
FLAG = "recall_summary_event_time"


def iso(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def msg(e, text, observed=None, sid="s1"):
    payload = {"role": "user", "content": text}
    if observed is not None:
        payload["timestamp"] = observed
    return e._store.append(sid, payload)


def node(e, ids, *, source_type="messages", sid="s1", depth=0, **kw):
    return e._dag.add_node(SummaryNode(session_id=sid, depth=depth, summary="needle summary",
                                       source_ids=list(ids), source_type=source_type, **kw))


def summary_entry(e, node_id, score=2.0):
    n = e._dag.get_node(node_id)
    hit = {"kind": "summary", "node_id": node_id, "session_id": n.session_id,
           "store_id": int(n.source_ids[0]) if n.source_type == "messages" else None,
           "timestamp": n.latest_at or n.created_at or 0, "snippet": n.summary,
           "from_current_session": False}
    hit["expand_hint"] = tools._lcm_recall_summary_expand_hint(hit)
    return {"hit": hit, "ranks": {0: 1}, "_final_score": score}


def inject(e, monkeypatch):
    """Ranking is not under test: put the given summary entries ahead of the FTS hits."""
    e.injected = []
    original = tools._lcm_recall_rerank

    def rank(*args, **kwargs):
        entries, status, scores = original(*args, **kwargs)
        return [*e.injected, *entries], status, scores

    monkeypatch.setattr(tools, "_lcm_recall_rerank", rank)


@pytest.fixture
def engine(recall_engine, monkeypatch):
    recall_engine._config.embeddings_enabled = False
    inject(recall_engine, monkeypatch)
    return recall_engine


def recall(e, **args):
    return json.loads(tools.lcm_recall({"query": "needle", "limit": 10, **args}, engine=e))


def summaries(payload):
    return [h for h in payload["hits"] if h["kind"] == "summary"]


def test_flag_on_gives_min_max_span_and_keeps_timestamp(engine):
    e = engine
    times = [1_600_000_300.0, 1_600_000_100.0, 1_600_000_200.0]
    n = node(e, [msg(e, f"needle {i}", t) for i, t in enumerate(times)])
    e.injected = [summary_entry(e, n)]
    off = summaries(recall(e))[0]
    setattr(e._config, FLAG, True)
    on = summaries(recall(e))[0]
    assert on.pop("event_time_start") == iso(1_600_000_100)
    assert on.pop("event_time_end") == iso(1_600_000_300)
    assert on == off  # timestamp and every other field unchanged


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
@pytest.mark.parametrize("strict", [True, False])
def test_flag_off_is_byte_identical(engine, monkeypatch, detail, strict):
    e = engine
    e._config.recall_reference_strict = strict
    n = node(e, [msg(e, "needle a", 1_600_000_000), msg(e, "needle b", 1_600_000_900)])
    e.injected = [summary_entry(e, n)]
    args = {"query": "needle", "detail": detail}
    monkeypatch.setattr(tools, "_lcm_recall_summary_event_time_rows",
                        Mock(side_effect=AssertionError("read while off")))
    real_config = e._config
    e._config = SimpleNamespace(**{k: v for k, v in vars(real_config).items() if k != FLAG})
    unset = tools.lcm_recall(args, engine=e)
    e._config = real_config
    assert getattr(e._config, FLAG) is False
    assert tools.lcm_recall(args, engine=e) == unset
    setattr(e._config, FLAG, True)
    assert tools.lcm_recall(args, engine=e, summary_event_time=False) == unset
    assert "summary_event_time_unavailable" not in json.loads(unset)["provenance"]


def test_positive_control_flag_on_changes_snippets(engine):
    e = engine
    n = node(e, [msg(e, "needle a", 1_600_000_000)])
    e.injected = [summary_entry(e, n)]
    before = tools.lcm_recall({"query": "needle"}, engine=e)
    setattr(e._config, FLAG, True)
    assert tools.lcm_recall({"query": "needle"}, engine=e) != before


def test_null_observed_everywhere_omits_fields(engine):
    e = engine
    n = node(e, [msg(e, "needle a"), msg(e, "needle b")])
    e.injected = [summary_entry(e, n)]
    setattr(e._config, FLAG, True)
    payload = recall(e)
    hit = summaries(payload)[0]
    assert "event_time_start" not in hit and "event_time_end" not in hit
    assert "summary_event_time_unavailable" not in payload["provenance"]


def test_mixed_null_and_set_uses_only_set_values(engine):
    e = engine
    ids = [msg(e, "needle a"), msg(e, "needle b", 1_650_000_000), msg(e, "needle c"),
           msg(e, "needle d", 1_640_000_000)]
    n = node(e, ids)
    e.injected = [summary_entry(e, n)]
    setattr(e._config, FLAG, True)
    hit = summaries(recall(e))[0]
    assert (hit["event_time_start"], hit["event_time_end"]) == (iso(1_640_000_000), iso(1_650_000_000))


def test_condensed_multilevel_walk_is_uncapped(engine):
    e = engine
    # Store order runs newest-first, so the cap-4 lineage helper sees neither extreme.
    times = [1_700_000_000 - 1000 * i for i in range(9)]
    ids = [msg(e, f"needle {i}", t) for i, t in enumerate(times)]
    leaves = [node(e, ids[0:3]), node(e, ids[3:6]), node(e, ids[6:8]), node(e, ids[8:9])]
    mid = [node(e, leaves[:2], source_type="nodes", depth=1), node(e, leaves[2:3], source_type="nodes", depth=1)]
    top = node(e, [mid[0], mid[1], leaves[3]], source_type="nodes", depth=2)
    capped = e._dag.source_message_ids(top, limit=4)
    assert ids[-1] not in capped
    e.injected = [summary_entry(e, top)]
    setattr(e._config, FLAG, True)
    hit = summaries(recall(e))[0]
    assert (hit["event_time_start"], hit["event_time_end"]) == (iso(min(times)), iso(max(times)))


def test_text_form_observed_at_sorts_by_time_not_lexically(engine):
    e = engine
    ids = [msg(e, "needle a", 1_600_000_000), msg(e, "needle b", 1_700_000_000),
           msg(e, "needle c", 1_650_000_000), msg(e, "needle d", 1_650_000_000)]
    conn = e._store._conn
    # ISO text stays TEXT under REAL affinity; SQLite orders any TEXT after every number.
    conn.execute("UPDATE messages SET observed_at = ? WHERE store_id = ?", ("2019-01-01T00:00:00+00:00", ids[2]))
    conn.execute("UPDATE messages SET observed_at = ? WHERE store_id = ?", ("2001-01-01T00:00:00", ids[3]))  # naive: rejected
    conn.commit()
    assert conn.execute("SELECT typeof(observed_at) FROM messages WHERE store_id = ?", (ids[2],)).fetchone()[0] == "text"
    n = node(e, ids)
    e.injected = [summary_entry(e, n)]
    setattr(e._config, FLAG, True)
    hit = summaries(recall(e))[0]
    start = datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp()
    assert (hit["event_time_start"], hit["event_time_end"]) == (iso(start), iso(1_700_000_000))


def test_cold_start_resume_ingest_shows_past_event_span_not_ingest_time(tmp_path, monkeypatch):
    """RM-required: the real engine ingest path stamps timestamp=now, observed_at=host time.

    Rows come from LCMEngine._ingest_messages (a first ingest, then the restart
    "persisted ambiguous delta"). The summary node is constructed as compaction
    writes it: earliest_at/latest_at = MIN/MAX(messages.timestamp) of its sources.
    """
    import time as clock
    started = clock.time()
    cfg = LCMConfig(database_path=str(tmp_path / "cold.db"), embeddings_enabled=False)
    host_times = [1_500_000_000 + 3600 * i for i in range(4)]
    first = LCMEngine(config=cfg)
    first.on_session_start("cold", platform="cli", conversation_id="cold-conv", context_length=200000)
    first._ingest_messages([{"role": "user", "content": f"needle history {i}", "timestamp": t}
                            for i, t in enumerate(host_times[:3])])
    for closer in (first._store, first._dag, first._lifecycle):
        closer.close()
    e = LCMEngine(config=cfg)
    e.on_session_start("cold", platform="cli", conversation_id="cold-conv", context_length=200000)
    e._ingest_messages([{"role": "user", "content": "needle resumed", "timestamp": host_times[3]}])
    try:
        assert e.get_status()["ingest_reconciliation"]["reason"] == "persisted ambiguous delta"
        rows = e._store.get_session_messages("cold", limit=10)
        assert [r["observed_at"] for r in rows] == host_times
        assert all(r["timestamp"] >= started for r in rows)  # write time = ingest time
        written = [r["timestamp"] for r in rows]
        n = node(e, [r["store_id"] for r in rows], sid="cold",
                 earliest_at=min(written), latest_at=max(written))
        inject(e, monkeypatch)
        e.injected = [summary_entry(e, n)]
        e._config.recall_summary_event_time = True
        hit = summaries(recall(e))[0]
        assert hit["timestamp"] == max(written) >= started
        assert (hit["event_time_start"], hit["event_time_end"]) == (iso(host_times[0]), iso(host_times[-1]))
    finally:
        for closer in (e._store, e._dag, e._lifecycle):
            closer.close()


def test_reader_deadline_and_read_error(engine, tmp_path, monkeypatch):
    e = engine
    n = node(e, [msg(e, "needle", 1_600_000_000)])
    import time as clock
    assert tools._lcm_recall_summary_event_time_rows(e, [n], deadline=clock.monotonic() - 1) == ({}, "deadline")
    missing = SimpleNamespace(_store=SimpleNamespace(db_path=str(tmp_path / "missing.db")))
    assert tools._lcm_recall_summary_event_time_rows(missing, [n], deadline=clock.monotonic() + 5) == ({}, "read_error")
    # The budget runs out during the read: the progress handler interrupts it.
    ticks = iter([0.0])
    monkeypatch.setattr(tools.time, "monotonic", lambda: next(ticks, 100.0))
    assert tools._lcm_recall_summary_event_time_rows(e, [n], deadline=50.0) == ({}, "deadline")
    monkeypatch.undo()
    spans, reason = tools._lcm_recall_summary_event_time_rows(e, [n], deadline=clock.monotonic() + 5)
    assert reason is None and spans[n]["event_time_start"] == iso(1_600_000_000)


@pytest.mark.parametrize("reason", ["deadline", "read_error"])
def test_failed_read_omits_all_summary_fields_with_marker(engine, monkeypatch, reason):
    e = engine
    a = node(e, [msg(e, "needle a", 1_600_000_000)])
    b = node(e, [msg(e, "needle b", 1_600_000_500)])
    e.injected = [summary_entry(e, a), summary_entry(e, b, 1.5)]
    real = tools._lcm_recall_summary_event_time_rows
    partial_read = {a: {"event_time_start": "x", "event_time_end": "y"}}
    monkeypatch.setattr(tools, "_lcm_recall_summary_event_time_rows",
                        Mock(side_effect=lambda *a_, **kw: (partial_read, reason)))
    setattr(e._config, FLAG, True)
    payload = recall(e)
    assert len(summaries(payload)) == 2
    assert all("event_time_start" not in h and "event_time_end" not in h for h in summaries(payload))
    assert payload["provenance"]["summary_event_time_unavailable"] == reason
    assert all("event_time" in h for h in payload["hits"] if h["kind"] != "summary")
    tools._lcm_recall_summary_event_time_rows.assert_called_once()
    assert sorted(tools._lcm_recall_summary_event_time_rows.call_args.args[1]) == sorted([a, b])
    assert real is not tools._lcm_recall_summary_event_time_rows


@pytest.fixture
def delivery():
    # Six summaries, each with a span; non-strict answer_ready delivers them as hits.
    entries = [{"hit": {"kind": "summary", "node_id": i, "store_id": None, "session_id": f"s{i}",
                        "timestamp": 500 + i, "snippet": f"needle {i} " + "x" * 300,
                        "expand_hint": f"lcm_expand(node_id={i})"},
                "ranks": {0: i}, "_final_score": 1 / i} for i in range(1, 7)]
    spans = {i: {"event_time_start": iso(1_600_000_000 + i), "event_time_end": iso(1_600_100_000 + i)}
             for i in range(1, 7)}
    read = Mock(return_value=(spans, None))
    options = dict(
        query="needle", limit=6, requested_limit=6, scope_bias=0.0, include="all",
        detail="snippets", reference_strict=False, delta_requested=False, seen_refs=set(),
        include_occurrence_time=False, deadline=float("inf"), fts_arm_index=0,
        arm_order=["fts"], arm_weights=[1.0], arm_hits={}, coverage={"fts": "full"},
        served_fts_index="stem", rerank_status="disabled", rerank_scores=[],
        embedding_query_metrics=[], degraded_reasons=[], timed_out=False,
        fts_anchor_info=None, summary_leads=[], limit_cap=25, snippet_chars=400,
        response_char_cap=64000, per_session_limit=3, expanded_hit_limit=6,
        content_chars=4000, strict_selector_factory=Mock(), hydrate=Mock(return_value={}),
        read_event_time=Mock(return_value=({}, None)),
        event_time_fields=tools._lcm_recall_event_time,
        read_occurrence_source=Mock(side_effect=AssertionError("extra read")),
        rerank_enabled=lambda: False,
    )

    def invoke(**kwargs):
        return recall_delivery.deliver_recall(entries, **{**options, **kwargs})

    return invoke, read


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_response_cap_holds_with_fields_on(delivery, detail):
    invoke, read = delivery
    off = invoke(detail=detail)
    full = invoke(detail=detail, read_summary_event_time=read)
    read.assert_called_with([1, 2, 3, 4, 5, 6], deadline=float("inf"))
    assert len(json.loads(full)["hits"]) == 6 and len(full) > len(off)
    bare = [{k: v for k, v in h.items() if not k.startswith("event_time")} for h in json.loads(full)["hits"]]
    # answer_ready caps the whole envelope; snippets budgets the hits only.
    cap = len(off) + 300 if detail == "answer_ready" else sum(len(json.dumps(h)) for h in bare) + 300
    raw = invoke(detail=detail, response_char_cap=cap, read_summary_event_time=read)
    payload = json.loads(raw)
    if detail == "answer_ready":
        assert len(raw) <= cap
    assert [h["node_id"] for h in payload["hits"]] == [1, 2, 3, 4, 5, 6]  # fields never evict a hit
    kept = [h for h in payload["hits"] if "event_time_start" in h]
    assert 0 < len(kept) < 6 and [h["node_id"] for h in kept] == list(range(1, len(kept) + 1))
    assert payload["event_time_unavailable_reason"] == "response_cap"


def test_nonstrict_answer_ready_edge(engine):
    e = engine
    e._config.recall_reference_strict = False
    s1 = msg(e, "needle first", 1_600_000_100)
    s2 = msg(e, "needle second", 1_600_000_500)
    n = node(e, [s1, s2])
    e.injected = [summary_entry(e, n)]
    args = {"detail": "answer_ready"}
    off = recall(e, **args)
    summary = summaries(off)[0]
    shared = next(h for h in off["hits"] if h.get("store_id") == s1 and h["kind"] != "summary")
    # Pinned (off): the summary shares s1's store_id and takes s1's event time.
    assert summary["store_id"] == s1
    assert summary["event_time"] == shared["event_time"] == iso(1_600_000_100)
    setattr(e._config, FLAG, True)
    on = summaries(recall(e, **args))[0]
    assert "event_time" not in on and "event_time_source" not in on and "time_uncertain" not in on
    assert (on["event_time_start"], on["event_time_end"]) == (iso(1_600_000_100), iso(1_600_000_500))


def _internal_fixture(e, monkeypatch):
    e._config.recall_reference_strict = False
    n = node(e, [msg(e, "needle dated", 1_600_000_000)], sid="other")
    inject(e, monkeypatch)
    e.injected = [summary_entry(e, n)]


@pytest.mark.parametrize("consumer", ["pack", "proposal", "auto", "requirements", "adaptive", "baseline"])
def test_internal_consumers_pinned_off(internal_recall, monkeypatch, consumer):
    from hermes_lcm import answer_contract, evidence_compiler, evidence_pack
    e, plugin = internal_recall
    _internal_fixture(e, monkeypatch)
    args = {"query": "needle", "detail": "answer_ready"}

    def consume(*a, retrieve, **kw):
        return {"context": retrieve(args)}

    monkeypatch.setattr(evidence_pack, "build_evidence_pack", consume)
    monkeypatch.setattr(evidence_compiler, "compile_evidence", consume)
    monkeypatch.setattr(evidence_compiler, "compile_preanswer_evidence", consume)
    monkeypatch.setattr(answer_contract, "compile_answer_contract",
                        lambda *a: SimpleNamespace(status="planned"))
    e._config.preanswer_evidence_enabled = True
    e._config.preanswer_evidence_mode = "requirements_v1"
    e._adaptive_retrieval = SimpleNamespace(search=lambda dispatch, **kw:
        consume(retrieve=lambda recall_args: dispatch("lcm_recall", recall_args)))

    def invoke():
        if consumer == "baseline":
            return plugin._answer_ready_baseline(e, "needle", {})
        if consumer == "adaptive":
            return json.loads(tools.lcm_retrieve({"action": "search"}, engine=e))
        if consumer == "requirements":
            return plugin._pre_llm_context(e, {"user_message": "needle", "baseline_refs": []})
        if consumer == "pack":
            return tools.lcm_evidence_pack({}, engine=e)
        return json.loads(tools.lcm_compile_evidence({"question": "needle", "mode": consumer}, engine=e))

    before = invoke()
    setattr(e._config, FLAG, True)
    # Positive control: on this fixture the model's own call does add the span.
    own = summaries(json.loads(tools.lcm_recall(args, engine=e)))
    assert own and own[0]["event_time_start"] == iso(1_600_000_000)
    assert invoke() == before


def test_proactive_pinned_off(recall_engine, monkeypatch):
    e = recall_engine
    e._config.embeddings_enabled = False
    e._config.proactive_recall_min_score = 0
    _internal_fixture(e, monkeypatch)
    msg(e, "needle plain", 1_600_000_100)
    monkeypatch.setattr(tools.time, "time", lambda: 1700000000)
    monkeypatch.setattr(e, "_proactive_recall_eligible", lambda: True, raising=False)
    monkeypatch.setattr(e, "_newest_user_message_text", lambda _: "needle", raising=False)
    monkeypatch.setattr(e, "_proactive_recall_privacy_warned", False, raising=False)
    monkeypatch.setattr(e, "_proactive_recall_trust_warned", False, raising=False)
    e._wrap_relevant_memories = LCMEngine._wrap_relevant_memories
    e._proactive_recall_injected_count = 0
    e._proactive_recall_skipped_count = 0
    original = tools.lcm_recall
    responses = []

    def capture(*args, **kwargs):
        responses.append(original(*args, **kwargs))
        return responses[-1]

    monkeypatch.setattr(tools, "lcm_recall", capture)
    before = LCMEngine._build_proactive_recall_message(e, [], "system", set())
    setattr(e._config, FLAG, True)
    assert LCMEngine._build_proactive_recall_message(e, [], "system", set()) == before
    assert len(responses) == 2 and responses[0] == responses[1]
    assert summaries(json.loads(responses[0]))
    own = summaries(json.loads(original({"query": "needle"}, engine=e)))
    assert own[0]["event_time_start"] == iso(1_600_000_000)


def test_benchmark_callers_pin_off():
    root = Path(__file__).resolve().parents[1]
    for path in ["bench/instruments/scale389/phase1a.py",
                 "benchmarks/qa-harness/src/providers/hermes-lcm/bridge/hermes_lcm_bridge.py"]:
        calls = [node for node in ast.walk(ast.parse((root / path).read_text()))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "lcm_recall"]
        assert calls
        assert all(any(kw.arg == "summary_event_time" and isinstance(kw.value, ast.Constant)
                       and kw.value.value is False for kw in call.keywords) for call in calls)


@pytest.fixture
def clean_env(monkeypatch):
    for key in tuple(config.os.environ):
        if key.startswith("LCM_"):
            monkeypatch.delenv(key)
    yaml = {}
    monkeypatch.setattr(config, "_load_hermes_config_yaml", lambda *a, **kw: yaml)
    return yaml


@pytest.mark.parametrize("env,yaml_value,expected,warns", [
    (None, None, False, False),
    ("true", None, True, False),
    (" ON ", None, True, False),
    ("garbage", None, False, True),
    (None, True, True, False),
    (None, "yes", True, False),
    (None, False, False, False),
    (None, "maybe", False, True),
    ("false", True, False, False),   # the environment wins
    ("1", False, True, False),
])
def test_config_env_yaml_precedence(clean_env, monkeypatch, env, yaml_value, expected, warns):
    if yaml_value is not None:
        clean_env["lcm"] = {FLAG: yaml_value}
    if env is not None:
        monkeypatch.setenv("LCM_RECALL_SUMMARY_EVENT_TIME", env)
    cfg = LCMConfig.from_env()
    assert getattr(cfg, FLAG) is expected
    assert FLAG not in cfg.ignored_config_yaml_lcm_keys
    warnings = [w for w in cfg.config_source_warnings if "SUMMARY_EVENT_TIME" in w or FLAG in w]
    assert bool(warnings) == warns
    if yaml_value is not None and env is None and not warns:
        assert cfg.config_sources[FLAG] == f"config_yaml:lcm.{FLAG}"


def test_config_default_and_constructor():
    assert getattr(LCMConfig(), FLAG) is False
    assert getattr(LCMConfig(**{FLAG: True}), FLAG) is True


def test_no_preset_reaches_the_flag():
    assert FLAG not in presets._PRESET_FIELDS
    assert FLAG not in config._PRESET_ENV_FIELDS
    for preset in presets.shipped_presets():
        assert FLAG not in preset.runtime_env
        assert FLAG not in preset.unsupported_runtime_fields
        assert "LCM_RECALL_SUMMARY_EVENT_TIME" not in json.dumps(
            {**preset.runtime_env, **preset.unsupported_runtime_fields})


def test_session_delivery_summary_leads(engine, monkeypatch):
    from hermes_lcm import tokens
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    e = engine
    e._config.recall_delivery_unit = "session"
    n = node(e, [msg(e, "needle lead a", 1_600_000_000), msg(e, "needle lead b", 1_600_000_700)])
    e.injected = [summary_entry(e, n)]
    e._dag.source_message_ids = lambda *a, **kw: []  # unresolved: the node comes back as a lead
    off = tools.lcm_recall({"query": "needle"}, engine=e)
    lead = json.loads(off)["provenance"]["delivery"]["summary_leads"][0]
    assert lead["node_id"] == n and "event_time_start" not in lead
    assert tools.lcm_recall({"query": "needle"}, engine=e, summary_event_time=False) == off
    setattr(e._config, FLAG, True)
    payload = json.loads(tools.lcm_recall({"query": "needle"}, engine=e))
    lead = payload["provenance"]["delivery"]["summary_leads"][0]
    assert (lead["event_time_start"], lead["event_time_end"]) == (iso(1_600_000_000), iso(1_600_000_700))
    monkeypatch.setattr(tools, "_lcm_recall_summary_event_time_rows", lambda *a, **kw: ({}, "read_error"))
    payload = json.loads(tools.lcm_recall({"query": "needle"}, engine=e))
    assert payload["provenance"]["summary_event_time_unavailable"] == "read_error"
    assert "event_time_start" not in payload["provenance"]["delivery"]["summary_leads"][0]


def test_ten_thousand_message_summary_stays_inside_deadline(recall_engine, capsys):
    import time as clock
    e = recall_engine
    base = 1_500_000_000
    ids: list[int] = []
    for chunk in range(10):
        ids += e._store.append_batch("big", [{"role": "user", "content": f"m{chunk}-{i}",
                                              "timestamp": base + chunk * 1000 + i} for i in range(1000)])
    level = [node(e, ids[i:i + 25], sid="big") for i in range(0, len(ids), 25)]
    depth = 1
    while len(level) > 1:
        level = [node(e, level[i:i + 4], source_type="nodes", sid="big", depth=depth)
                 for i in range(0, len(level), 4)]
        depth += 1
    budget = e._config.recall_query_timeout_s
    started = clock.monotonic()
    spans, reason = tools._lcm_recall_summary_event_time_rows(e, level, deadline=started + budget)
    elapsed = clock.monotonic() - started
    with capsys.disabled():
        print(f"\nsummary_event_time 10k messages, depth {depth - 1}: {elapsed * 1000:.1f} ms "
              f"(budget {budget:.1f} s)")
    assert reason is None and elapsed < budget
    assert spans[level[0]] == {"event_time_start": iso(base), "event_time_end": iso(base + 9999)}
