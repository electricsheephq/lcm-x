"""Decided round-two instrument regressions; synthetic/offline only."""
import json
import math
import time
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace

import pytest

from bench.fmp import dev, dev_cache, dev_panel, delivery, judges, score
from bench.fmp.tests.test_dev import Backend, cached_call
from bench.fmp.tests.test_r3 import arm, row


def paired(items, gains=(), losses=(), identical=False):
    return {(a, e["id"]): dict(correct=(e["id"] in gains if a == "candidate" else e["id"] in losses),
        retrieval={"delivered_sha256": "same" if identical else a})
        for a in ("baseline", "candidate") for e in items}


def frozen():
    return json.loads((dev_panel.PANELS / "dev100.json").read_bytes())["items"]


def test_stratified_locomo_only_twenty_gain():
    items = frozen()
    # Both frozen conversations contribute +10. Concentrating all +20 in one
    # conversation legitimately permits zero-gain draws even after stratifying.
    gains = [e["id"] for cluster in ("locomo-1", "locomo-2")
             for e in [e for e in items if e["cluster"] == cluster][:10]]
    result = dev.compare(items, paired(items, gains))
    assert result["correct_count_difference"] == 20
    assert result["ci95"][0] > 0
    assert result["strata_clusters"]["locomo"] == 2
    assert "2 conversations" in result["interval_note"] and "approximate" in result["interval_note"]
    concentrated = [e["id"] for e in items if e["dataset"] == "locomo"][:20]
    assert dev.compare(items, paired(items, concentrated))["ci95"][0] == 0
    all_locomo = [e["id"] for e in items if e["dataset"] == "locomo"]
    assert dev.compare(items, paired(items, all_locomo))["ci95"] == pytest.approx([.81, .81])


@pytest.mark.parametrize("losses,margin,expected", [(1, 2, "NO-SIGNAL"), (2, 2, "GUARD-RISK"),
    (2, 3, "NO-SIGNAL"), (3, 3, "GUARD-RISK")])
def test_guard_margin_uses_net_items(losses, margin, expected):
    items = frozen()
    guard = [dict(id=f"guard-{i}", dataset="lme-s", cluster=f"g-{i}") for i in range(5)]
    rows = paired(items + guard, losses=[e["id"] for e in guard[:losses]])
    report = dev.evaluate(items, guard, rows, min_gain=1, guard_margin=margin)
    assert report["verdict"] == expected and report["guard_margin"] == margin
    assert report["guard"]["ci95"] is not None
    assert dev.parser().parse_args([]).guard_margin == 2


def test_guard_rule_combines_strata_before_margin():
    guard = [dict(id=f"g{i}", dataset="locomo" if i < 3 else "lme-s", cluster=f"c{i}") for i in range(5)]
    rows = paired(guard, gains=["g3", "g4"], losses=["g0", "g1", "g2"])
    result = dev.evaluate([], guard, rows, min_gain=1, guard_margin=2)
    assert result["guard"]["correct_count_difference"] == -1
    assert result["verdict"] == "NO-SIGNAL"


def test_discordance_mde_and_rules_disclosed():
    items = frozen()
    result = dev.evaluate(items, [], paired(items, gains=[e["id"] for e in items[:8]],
        losses=[e["id"] for e in items[8:15]]), min_gain=2, guard_margin=2)
    assert result["primary"]["discordant"] == 15
    assert result["primary"]["minimum_detectable_effect_items"] == pytest.approx(2.8 * math.sqrt(15))
    assert result["screen"] == "kill/futility screen, not a promotion instrument"
    assert result["lme_abstention_guard_items"] == 5
    for token in ("DIRECTIONAL-WIN", "GUARD-RISK", "INCOMPLETE"):
        assert token in result["rules"]
    assert "no-op-arm" in result["rules"]["INCOMPLETE"]


def test_identical_delivered_evidence_is_incomplete():
    items = frozen()
    result = dev.evaluate(items, [], paired(items, identical=True), min_gain=1, guard_margin=2)
    assert result["verdict"] == "INCOMPLETE" and result["reason"] == "no-op-arm"


@pytest.mark.parametrize("key", ["leaf_chunk_tokens", "extraction_enabled", "assertion_extraction_enabled", "unknown_key"])
def test_ingestion_or_unknown_override_refused(key):
    with pytest.raises(ValueError, match=f"unsupported recall-time override.*{key}"):
        dev.validate_overrides({key: 1})


def test_recall_allowlist_matches_documented_source():
    keys = {"recall_scan_max_rows", "recall_scan_budget_s", "recall_scan_rows", "recall_reference_strict",
        "recall_query_timeout_s", "recall_arm_weights", "recall_fts_anchor", "rerank_window_limit",
        "rerank_enabled", "rerank_model", "rerank_margin", "embeddings_enabled"}
    dev.validate_overrides(dict.fromkeys(keys, 1) | {"recall_depth": 50})
    assert dev.RECALL_CONFIG_KEYS == keys


@pytest.mark.parametrize("product_cap", [25, 17, 60])
def test_unforced_product_caps_observed_and_readback_strict(tmp_path, monkeypatch, product_cap):
    import sys
    tools, recall = ModuleType("hermes_lcm.tools"), ModuleType("hermes_lcm.recall_delivery")
    tools._LCM_RECALL_LIMIT_CAP = product_cap
    tools._LCM_RECALL_ANSWER_READY_PER_SESSION_LIMIT = 3
    tools._LCM_RECALL_ANSWER_READY_EXPANDED_HIT_LIMIT = 7
    recall.deliver_recall = lambda ordered, **kw: kw
    package = ModuleType("hermes_lcm")
    package.tools, package.recall_delivery = tools, recall
    monkeypatch.setitem(sys.modules, "hermes_lcm", package)
    tamper = False

    def dispatch(name, args, engine, offered=None):
        assert args == {"query": "synthetic", "limit": 25, "detail": "answer_ready"}
        candidate_limit, rerank_window = 177, 13  # noqa: F841 - observed product locals
        limit = min(args["limit"], tools._LCM_RECALL_LIMIT_CAP)
        kw = recall.deliver_recall([], limit=limit, requested_limit=args["limit"],
            limit_cap=tools._LCM_RECALL_LIMIT_CAP,
            per_session_limit=tools._LCM_RECALL_ANSWER_READY_PER_SESSION_LIMIT,
            expanded_hit_limit=tools._LCM_RECALL_ANSWER_READY_EXPANDED_HIT_LIMIT)
        result = dict(limit=limit, hits=[], total_results=0, provenance={"answer_ready": dict(
            per_session_limit=kw["per_session_limit"] + int(tamper),
            expanded_hit_limit=kw["expanded_hit_limit"], expanded_hit_count=0)})
        if limit < args["limit"]:
            result["limit_clamped_from"] = args["limit"]
        return json.dumps(result)

    monkeypatch.setattr(delivery, "dispatch", dispatch)
    monkeypatch.setattr(delivery, "question_clock", lambda date: nullcontext())
    monkeypatch.setattr(delivery, "single", lambda q, engine, speakers: ("fixture", delivery.dispatch(
        "lcm_recall", {"query": "synthetic", "limit": 25}, engine)))
    @contextmanager
    def engine(q):
        yield SimpleNamespace(_config=SimpleNamespace(rerank_window_limit=13)), None
    (tmp_path / "speakers.json").write_text("{}")
    store = SimpleNamespace(engine=engine, directory=lambda q: tmp_path)
    value = dev.retrieve(SimpleNamespace(question_date=""), store, {"rerank_window_limit": 13})
    assert value["metadata"]["observed"] == dict(limit=product_cap, candidate_pool=177,
        rerank_window=13, per_session=3, expanded=7)
    assert value["metadata"]["response_limit"] == min(25, product_cap)
    tamper = True
    with pytest.raises(dev.Incomplete, match="readback mismatch"):
        dev.retrieve(SimpleNamespace(question_date=""), store, {})


def test_codex_version_part_of_endpoint_once_per_process(tmp_path, monkeypatch):
    monkeypatch.setattr(dev_cache, "_CODEX_VERSION", None, raising=False)
    calls = []
    version = ["codex-cli synthetic-v1"]
    def read(cmd, **kw):
        assert cmd == ["codex", "--version"]
        calls.append(cmd)
        return version[0] + "\n"
    monkeypatch.setattr(dev_cache.subprocess, "check_output", read)
    backend = Backend()
    cache = dev_cache.Responses(tmp_path, backend, time.monotonic() + 10, {})
    cached_call(cache)
    cached_call(dev_cache.Responses(tmp_path, backend, time.monotonic() + 10, {}))
    first = json.loads(next(tmp_path.glob("*.json")).read_bytes())
    assert "synthetic-v1" in json.dumps(first["identity"]["endpoint"])
    assert len(calls) == 1 and backend.calls == 1
    version[0] = "codex-cli synthetic-v2"
    monkeypatch.setattr(dev_cache, "_CODEX_VERSION", None)
    cached_call(dev_cache.Responses(tmp_path, backend, time.monotonic() + 10, {}))
    assert len(calls) == 2 and backend.calls == 2


def test_unparseable_judge_persisted_as_error_never_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(dev_cache, "_CODEX_VERSION", "fixture", raising=False)
    backend = Backend()  # replies with unparseable synthetic response
    cache = dev_cache.Responses(tmp_path, backend, time.monotonic() + 10, {})
    for _ in range(2):
        with pytest.raises(IncompleteError):
            cached_call(cache, role="judge", validate=lambda r: judges.parse(r["content"], "lme-s") is not None)
    state = json.loads(next(tmp_path.glob("*.json")).read_bytes())
    assert state["status"] == "error" and state["response_sha256"] is None
    assert backend.calls == 1 and cache.counts["judge"]["cached"] == 0


IncompleteError = dev_cache.Incomplete


def test_score_jsonl_unicode_separator_and_blank_lines(tmp_path):
    path = arm(tmp_path, "off", [row(0, hypothesis="synthetic\u2028answer")])
    data = json.loads((path / "judged.jsonl").read_text())
    (path / "judged.jsonl").write_text("\n \n" + json.dumps(data, ensure_ascii=False) + "\n\n")
    result = score.score([path], tmp_path / "report")
    assert next(iter(result["arms"].values()))["scored_n"] == 1
