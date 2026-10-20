"""Offline P plumbing probes; no model or corpus calls."""
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import tests.conftest  # noqa: F401 - same package bootstrap as existing FMP tests

from bench.fmp import delivery, dev, mcp_reader, run
from tests import test_lcm_recall

recall_engine = test_lcm_recall.recall_engine


@pytest.mark.parametrize("depth", [1, 2, 25, 50, 100])
@pytest.mark.parametrize("local", [False, True])
def test_dev_p_k_e(recall_engine, tmp_path, monkeypatch, depth, local):
    # RED: base rejects delivery_unit; effective config/readback must retain K/E.
    e = recall_engine
    original_dispatch = delivery.dispatch
    payloads = []
    def checked(*a, **kw):
        raw = original_dispatch(*a, **kw)
        assert "error" not in json.loads(raw), raw
        payloads.append(json.loads(raw))
        return raw
    monkeypatch.setattr(delivery, "dispatch", checked)
    e._config.embeddings_enabled = False
    for i in range(7):
        e._store.append("one", {"role": "user", "content": "needle" if i == 3 else f"context {i}"})
    if local:
        from hermes_lcm import tools
        from tests.test_lcm_recall import MockProvider
        e._config.embeddings_enabled = True
        provider = MockProvider()
        monkeypatch.setattr(tools, "_resolve_recall_provider", lambda *a, **kw: provider)
    @contextmanager
    def engine(q):
        yield e, None
    (tmp_path / "speakers.json").write_text("{}")
    store = SimpleNamespace(engine=engine, directory=lambda q: tmp_path)
    q = SimpleNamespace(question="needle", question_date="")
    hits = dev.retrieve(q, store, {"recall_depth": depth})
    query_calls = len(provider.queries) if local else 0
    pack = dev.retrieve(q, store, {"recall_depth": depth, "delivery_unit": "session"})
    assert pack["metadata"]["delivery_unit"] == "session"
    assert pack["metadata"]["hit_count"] == 7
    assert pack["metadata"]["response_limit"] == depth
    assert hits["source_sha256"] != pack["source_sha256"]
    assert pack["metadata"]["coverage"] == hits["metadata"]["coverage"]
    assert "context 0" in pack["evidence"]
    if local:
        assert len(provider.queries) == 2 * query_calls
    assert payloads[0]["metrics"] == payloads[1]["metrics"]
    assert [h["store_id"] for h in payloads[0]["hits"]] == payloads[1]["provenance"]["delivery"]["sessions"][0]["seed_ids"]


def test_single_does_not_reread(tmp_path, monkeypatch):
    # RED: base calls get_batch and loses supplied content/time.
    payload = {"provenance": {"delivery": {"unit": "session"}}, "hits": [
        {"store_id": 1, "role": "user", "content": "Alice: Alice: image caption"}]}
    monkeypatch.setattr(delivery, "dispatch", lambda *a: json.dumps(payload))
    engine = SimpleNamespace(_store=SimpleNamespace(get_batch=Mock(side_effect=AssertionError("reread"))))
    text, raw = delivery.single(SimpleNamespace(question="needle"), engine, {"1": "Alice"})
    assert text == "[unknown] Alice: Alice: image caption"
    assert json.loads(raw) == payload


def test_mcp_spec_and_reopen(monkeypatch, tmp_path, recall_engine):
    # RED: base drops delivery_unit from server spec/reopen.
    args = run.parser().parse_args(["answer", "--split-label", "fixture", "--store-root", str(tmp_path),
        "--output", str(tmp_path), "--delivery-unit", "session"])
    spec = run.server_spec(args, "fixture", SimpleNamespace(qid="q", dataset="lme-s", store_id="s",
        session_map={}, question_date=""))
    assert spec["delivery_unit"] == "session"
    from io import StringIO
    from bench.fmp import stores
    @contextmanager
    def opened(self, q):
        yield recall_engine, None
    monkeypatch.setattr(stores.Stores, "__init__", lambda *a, **kw: None)
    monkeypatch.setattr(stores.Stores, "engine", opened)
    monkeypatch.setattr(mcp_reader, "load_lcm", lambda *a: None)
    monkeypatch.setattr(delivery, "schemas", lambda e: [])
    def dispatch(name, arguments, engine, offered):
        assert engine._config.recall_delivery_unit == "session"
        return '{"hits":[]}'
    monkeypatch.setattr(delivery, "dispatch", dispatch)
    spec["log"] = str(tmp_path / "metadata.jsonl")
    stdin = StringIO(json.dumps(dict(id=1, method="tools/call", params=dict(name="lcm_recall", arguments={}))) + "\n")
    stdout = StringIO()
    mcp_reader.serve(spec, stdin, stdout)
    assert json.loads(stdout.getvalue())["result"]["content"][0]["text"] == '{"hits":[]}'


def test_agentic_reserve_and_raw_forwarding(monkeypatch, tmp_path):
    # RED: base's 8k reservation admits a tool round where 16k must force final.
    engine = SimpleNamespace(_config=SimpleNamespace(recall_delivery_unit="session"))
    monkeypatch.setattr(delivery, "schemas", lambda e: [{"type": "function"}])
    choices = []
    class Reader:
        def call(self, reader, messages, records, **kw):
            choices.append(kw["tool_choice"])
            records.append({"status": "ok", "input_tokens": 1})
            return {"content": "done"}
    q = SimpleNamespace(qid="q", qtype="x", question="needle", question_date="")
    delivery._answer(q, "agentic", Reader(), "fixture", engine, {}, 12, 20000, [], "speaker", "plain")
    assert choices == ["none"]
    raw = '{"provenance":{"delivery":{"unit":"session"}},"hits":[]}'
    monkeypatch.setattr(delivery, "dispatch", lambda *a: raw)
    spec = {"max_calls": 1, "question": {"question_date": ""}}
    reader = mcp_reader.Reader(spec, engine, [], tmp_path / "log.jsonl")
    assert reader.call("lcm_recall", {}) == (raw, False)
    assert json.loads(reader.log.read_text())["output_bytes"] == len(raw.encode())


@pytest.mark.parametrize("cross", [False, True])
def test_score_separates_p_axis(tmp_path, cross):
    # RED on initial P candidate and original base: P arms collide.
    from bench.fmp import score
    from bench.fmp.tests.test_score_commits import COMMITS, arm
    runs = [arm(tmp_path, f"{i}-{unit}", commit, [True], **(
        dict(delivery_unit=unit, delivery_budget=16000) if unit == "session" else {}))
        for i, commit in enumerate(COMMITS if cross else COMMITS[:1]) for unit in ("hits", "session")]
    result = score.score(runs, tmp_path / "score", pair_across_commits=cross)
    assert len(result["arms"]) == (4 if cross else 2)
    assert len(result["paired"]) == (2 if cross else 1)


def test_score_does_not_pair_changed_p_across_commits(tmp_path):
    # RED: initial candidate pairs hits against session as an equivalent arm.
    from bench.fmp import score
    from bench.fmp.tests.test_score_commits import COMMITS, arm
    runs = [arm(tmp_path, "hits", COMMITS[0], [True]), arm(tmp_path, "session", COMMITS[1], [True],
        delivery_unit="session", delivery_budget=16000)]
    assert not score.score(runs, tmp_path / "score", pair_across_commits=True)["paired"]


def test_judge_default_hits_rejects_session_identity(tmp_path, monkeypatch):
    # RED: initial candidate reaches judging without checking the requested P.
    from bench.fmp import common
    args = run.parser().parse_args(["judge", "--split-label", "synthetic", "--store-root", str(tmp_path),
        "--output", str(tmp_path), "--delivery", "single"])
    monkeypatch.setattr(run, "commit", lambda _: "fixture")
    manifest = dict(dataset_sha256="d", id_list_sha256="i", split_label="synthetic", question_ids=["0"],
        prompts_sha256=run.prompt_hashes(), harness_commit="fixture", lcmx_commit="fixture",
        harness_tree_sha256=common.sha(b"".join(p.read_bytes() for p in sorted(Path(run.__file__).parent.glob("*.py")))),
        store_text="speaker-prefix", hit_label="speaker", delivery_unit="session", delivery_budget=16000)
    (tmp_path / "run-manifest.json").write_text(json.dumps(manifest))
    def admitted(_):
        raise AssertionError("mismatched P identity admitted")
    monkeypatch.setattr(run, "specs", admitted)
    with pytest.raises(ValueError, match="identity mismatch"):
        next(run.run(args, None, [SimpleNamespace(qid="0")], "d", "i", {}, None))
