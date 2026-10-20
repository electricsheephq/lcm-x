"""Sanitized cache/panel/deadline controls for the development instrument."""
import copy
import json
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.fmp import dev, dev_cache, dev_panel, delivery, judges, models
from bench.fmp.common import digest_file, sha
from bench.fmp.datasets import Question
from bench.fmp.dev_cache import Incomplete, Responses, request_identity


def identity():
    return request_identity("codex:fixture@low", [{"role": "user", "content": "synthetic"}],
        endpoint="subscription", effort="low", temperature=0, max_tokens=42, tools=[{"name": "fixture"}],
        tool_choice="none", rubric="r1", renderer={"renderer": "a"}, sources={"store": "b", "retrieval": "c"})


@pytest.mark.parametrize("field", list(identity()))
def test_every_request_identity_field_misses(field, tmp_path, monkeypatch):
    original = identity()
    changed = copy.deepcopy(original)
    changed[field] = {"changed": True}
    assert sha(dev_panel.encoded(original)) != sha(dev_panel.encoded(changed))
    backend = Backend()
    cache = Responses(tmp_path, backend, time.monotonic() + 10, {})
    monkeypatch.setattr(dev_cache, "request_identity", lambda *a, **k: original)
    cached_call(cache)
    monkeypatch.setattr(dev_cache, "request_identity", lambda *a, **k: changed)
    cached_call(cache)
    assert backend.calls == 2 and cache.counts["reader"]["cached"] == 0


class Backend:
    def __init__(self):
        self.calls = 0

    def call(self, spec, messages, records, **kwargs):
        self.calls += 1
        records.append(dict(status="ok", input_tokens=1, requested_model=spec))
        return {"content": "synthetic response"}


def cached_call(cache, **kwargs):
    args = dict(role="reader", rubric="fixture", sources={"store": "a"}, max_tokens=42)
    return cache.call("codex:fixture@low", [{"role": "user", "content": "synthetic"}], [], **(args | kwargs))


@pytest.fixture(autouse=True)
def synthetic_codex_version(monkeypatch):
    monkeypatch.setattr(dev_cache, "_CODEX_VERSION", "codex-cli fixture", raising=False)


def test_identical_request_hits_without_model_call_and_digest_is_checked(tmp_path):
    backend = Backend()
    cache = Responses(tmp_path, backend, time.monotonic() + 10, {"renderer": "a"})
    assert cached_call(cache) == cached_call(cache)
    assert backend.calls == 1
    assert cache.counts["reader"] == {"fresh": 1, "cached": 1}
    assert cached_call(cache, sources={"store": "b"})
    assert backend.calls == 2
    path = next(tmp_path.glob("*.json"))
    data = json.loads(path.read_bytes())
    assert data["status"] == "ok" and data["attempts"][0]["status"] == "ok"
    data["response_sha256"] = "changed"
    path.write_text(json.dumps(data))
    cache.root = tmp_path
    # Read the exact identity represented by this fixture's first cache file.
    store = data["identity"]["retrieved_sources"]
    with pytest.raises(Incomplete, match="digest"):
        cached_call(cache, sources=store)


def test_failed_or_uncertain_send_never_replays(tmp_path):
    backend = Backend()
    cache = Responses(tmp_path, backend, time.monotonic() + 10, {})
    def fail(*args, **kwargs):
        backend.calls += 1
        raise RuntimeError("synthetic failure")
    backend.call = fail
    with pytest.raises(RuntimeError):
        cached_call(cache)
    with pytest.raises(Incomplete):
        cached_call(cache)
    assert backend.calls == 1
    data = json.loads(next(tmp_path.glob("*.json")).read_bytes())
    assert data["status"] == "error" and data["response_sha256"] is None


def test_expired_deadline_makes_no_new_call(tmp_path):
    backend = Backend()
    cache = Responses(tmp_path, backend, time.monotonic() - 1, {})
    with pytest.raises(Incomplete, match="deadline"):
        cached_call(cache)
    assert backend.calls == 0


def test_frozen_panels():
    panel, guard = [json.loads((dev_panel.PANELS / name).read_bytes()) for name in ("dev100.json", "guard.json")]
    assert len(panel["items"]) == 100 and len(guard["items"]) == 76
    assert Counter(e["stratum"] for e in panel["items"]) == {
        "abstention": 25, "multi-hop": 25, "single-hop": 25, "update-temporal": 25}
    assert Counter(e["dataset"] for e in guard["items"]) == {"locomo": 71, "lme-s": 5}
    assert panel["lme_dev_ids"] == guard["lme_dev_ids"] and len(panel["lme_dev_ids"]) == 100
    for e in panel["items"] + guard["items"]:
        assert set(e) == {"id", "dataset", "cluster", "stratum", "sha256"}
        assert len(e["sha256"]) == 64
        assert e["id"] in panel["lme_dev_ids"] or e["id"].startswith(("locomo-1-q", "locomo-2-q"))
    assert digest_file(dev_panel.PANELS / "dev100.json") == "d30030d00f9402e13271dad4a476a6275accd5e22b217b78bc67b42b63257645"
    assert digest_file(dev_panel.PANELS / "guard.json") == "2eb003c22d495486efe640ecae69d1a9ee49a3d2846abe359c81692e7d7709b1"


def test_adversarial_guard_rewards_abstention_without_adversarial_gold():
    q = Question("locomo-1-q0", "locomo", "5", "synthetic question", "adversarial decoy", "", True, [])
    messages, dataset = dev.judge_messages(q, "I do not know")
    text = json.dumps(messages)
    assert dataset == "lme-s" and "unanswerable" in text and "adversarial decoy" not in text
    assert judges.parse("yes", dataset) is True and judges.parse("no", dataset) is False
    assert q.gold == "adversarial decoy"  # no mutation of normalized corpus objects


def test_cluster_bootstrap_and_guard_delta():
    items = [{"id": str(i), "cluster": "same-conversation"} for i in range(4)]
    rows = {(arm, str(i)): {"correct": arm == "candidate"} for arm in ("baseline", "candidate") for i in range(4)}
    result = dev.compare(items, rows)
    assert result["clusters"] == 1 and result["correct_count_difference"] == 4
    assert result["delta"] == 1 and result["ci95"] == [1, 1]


@pytest.mark.parametrize("degraded", [False, True])
def test_d50_changes_actual_delivery_and_observes_caps(tmp_path, monkeypatch, degraded):
    import sys
    from types import ModuleType
    tools, recall_delivery = ModuleType("hermes_lcm.tools"), ModuleType("hermes_lcm.recall_delivery")
    for key in ("_LCM_RECALL_LIMIT_CAP", "_LCM_RECALL_ANSWER_READY_PER_SESSION_LIMIT", "_LCM_RECALL_ANSWER_READY_EXPANDED_HIT_LIMIT"):
        setattr(tools, key, 0)
    recall_delivery.deliver_recall = lambda ordered, **kw: kw
    package = ModuleType("hermes_lcm")
    package.tools, package.recall_delivery = tools, recall_delivery
    monkeypatch.setitem(sys.modules, "hermes_lcm", package)
    monkeypatch.setitem(sys.modules, "hermes_lcm.tools", tools)
    monkeypatch.setitem(sys.modules, "hermes_lcm.recall_delivery", recall_delivery)

    def dispatch(name, args, engine, offered=None):
        candidate_limit, rerank_window = args["limit"] * 4, 50  # noqa: F841 - observed caller locals
        kw = recall_delivery.deliver_recall([], limit=args["limit"], requested_limit=args["limit"],
             limit_cap=tools._LCM_RECALL_LIMIT_CAP,
             per_session_limit=tools._LCM_RECALL_ANSWER_READY_PER_SESSION_LIMIT,
             expanded_hit_limit=tools._LCM_RECALL_ANSWER_READY_EXPANDED_HIT_LIMIT)
        return json.dumps(dict(limit=args["limit"], hits=[], total_results=0, degraded=degraded,
            degraded_reason="summary vectors are unavailable" if degraded else "",
            provenance={"coverage": {"fts": "ok", "summary": "none", "chunk": "full"},
                "answer_ready": dict(per_session_limit=kw["per_session_limit"],
                expanded_hit_limit=kw["expanded_hit_limit"], expanded_hit_count=0)}))
    monkeypatch.setattr(delivery, "dispatch", dispatch)
    monkeypatch.setattr(delivery, "single", lambda q, engine, speakers: ("synthetic evidence", delivery.dispatch("lcm_recall", {}, engine)))
    monkeypatch.setattr(delivery, "question_clock", lambda date: contextlib_null())
    @contextmanager
    def engine(q):
        yield SimpleNamespace(_config=SimpleNamespace()), None
    @contextmanager
    def contextlib_null():
        yield
    (tmp_path / "speakers.json").write_text("{}")
    store = SimpleNamespace(engine=engine, directory=lambda q: tmp_path)
    q = SimpleNamespace(question_date="")
    d25, d50 = [dev.retrieve(q, store, {"recall_depth": n}) for n in (25, 50)]
    assert d25["metadata"]["response_limit"] == 25 and d50["metadata"]["response_limit"] == 50
    assert d50["metadata"]["observed"] == dev.caps(50)
    assert d50["metadata"]["degraded"] == degraded
    assert d50["metadata"]["coverage"]["summary"] == "none"
    assert len(d50["metadata"]["degraded_reason_sha256"]) == 64


def test_source_identity_includes_dirty_product_bytes_but_not_bench(tmp_path, monkeypatch):
    (tmp_path / "product.py").write_text("a")
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench" / "fixture.py").write_text("a")
    monkeypatch.setattr(dev_cache.subprocess, "check_output", lambda *a, **k: b"product.py\0bench/fixture.py\0")
    first = dev_cache.source_identity(tmp_path)
    (tmp_path / "bench" / "fixture.py").write_text("b")
    assert first == dev_cache.source_identity(tmp_path)
    (tmp_path / "product.py").write_text("b")
    assert first != dev_cache.source_identity(tmp_path)


@pytest.mark.parametrize("expired", [False, True])
@pytest.mark.parametrize("no_op", [False, True])
def test_synthetic_end_to_end_cache_and_deadline_receipt(tmp_path, monkeypatch, expired, no_op):
    q = Question("fixture-id", "lme-s", "single-session-user", "synthetic question", "synthetic gold", "", False, [])
    item = dict(id=q.qid, cluster="fixture-cluster")
    monkeypatch.setattr(dev, "load_lcm", lambda repo: None)
    monkeypatch.setattr(dev, "load", lambda *a: ([q], {"items": [item]}, {"items": []}))
    monkeypatch.setattr(dev.shutil, "disk_usage", lambda p: SimpleNamespace(free=200 * 1024 ** 3))
    cache_root = tmp_path / "cache"
    retrieval = cache_root / "retrieval"
    retrieval.mkdir(parents=True)
    (retrieval / "fixture.json").write_text(json.dumps({"value": {"evidence": "synthetic evidence"}}))
    def worker_run(cmd, **kwargs):
        spec = json.loads(Path(cmd[-1]).read_bytes())
        Path(spec["receipt"]).write_text(json.dumps(dict(product="synthetic-sha", rows=[
            dict(id=q.qid, cache_key="fixture", sources={}, retrieval_sha256="synthetic-hash",
                 delivered_sha256="same" if no_op else "candidate" if "candidate" in spec["receipt"] else "baseline")])) )
        if expired:
            time.sleep(.01)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(dev.subprocess, "run", worker_run)
    class SyntheticBackend(Backend):
        def call(self, *args, **kwargs):
            super().call(*args, **kwargs)
            return {"content": "yes"}
        def close(self):
            pass
    backend = SyntheticBackend()
    monkeypatch.setattr(dev, "MemoryCodex", lambda *a: backend)
    class Card:
        def parser(self):
            return SimpleNamespace(parse_args=lambda a: a)
        def execute(self, args):
            assert args[0] in {"init", "headline", "add", "finalize", "check", "status"}
            return 0
    monkeypatch.setattr(dev, "card_tool", lambda: Card())
    baseline = tmp_path / "baseline"
    panel = tmp_path / "panel.json"
    panel.write_text("{}")
    args = dev.parser().parse_args(["--baseline", str(baseline), "--cache-root", str(cache_root),
        "--output", str(tmp_path / "out"), "--panel", str(panel), "--guard", str(panel),
        "--lme-data", str(panel), "--locomo-data", str(panel), "--deadline-min", ".000001" if expired else "1"])
    result = dev.execute(args)
    report = json.loads((args.output / "report.json").read_bytes())
    if expired:
        assert result == 2 and report["verdict"] == "INCOMPLETE"
        assert "primary" not in report and len(report["remaining_ids"]) == 2 and backend.calls == 0
    else:
        assert result == (2 if no_op else 0)
        assert report["verdict"] == ("INCOMPLETE" if no_op else "NO-SIGNAL") and report["remaining_ids"] == []
        if no_op:
            assert report["reason"] == "no-op-arm"
        assert report["calls"] == {"reader": {"fresh": 1, "cached": 1}, "judge": {"fresh": 1, "cached": 1}}
        assert backend.calls == 2
        assert report["guard_margin"] == 2 and "kill/futility" in (args.output / "RESULT.md").read_text()
    for name in ("report.json", "report.md", "rows.json", "RESULT.md"):
        assert "synthetic question" not in (args.output / name).read_text()


def test_codex_jsonl_lf_framing_preserves_unicode_and_rejects_invalid_json(tmp_path, monkeypatch):
    # U+2028 is valid JSON string content, never a JSONL record delimiter.
    text, model = "synthetic\u2028answer", "fixture\u2028model"
    rollout = tmp_path / "synthetic-rollout.jsonl"
    entries = [{"type": "turn_context", "payload": {"model": model}}]
    rollout.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n\n")
    events = [{"type": "thread.started", "thread_id": "synthetic-thread"},
              {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
              {"type": "turn.completed", "usage": {}}]
    output = "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n\n"
    backend = models.Codex(tmp_path / "transport", models.Quota(tmp_path / "quota.json"))
    monkeypatch.setattr(backend, "run", lambda *a: SimpleNamespace(stdout=output, returncode=0))
    monkeypatch.setattr(backend, "rollout", lambda thread: rollout)
    records = []
    try:
        message = backend.call(f"codex:{model}@low", [{"role": "user", "content": "synthetic"}], records)
        assert message["content"].encode() == text.encode()
        assert records[0]["returned_model"].encode() == model.encode()
        output = "{invalid}\n"
        with pytest.raises(json.JSONDecodeError):
            backend.call(f"codex:{model}@low", [], [])
        rollout.write_text("{invalid}\n")
        with pytest.raises(json.JSONDecodeError):
            backend.inspect([], rollout)
    finally:
        backend.close()
