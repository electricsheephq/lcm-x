"""Network-free acceptance tests using synthetic history only."""
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from bench.fmp import datasets, delivery, judges, score
from bench.fmp.common import load_lcm
from bench.fmp.models import JUDGES, READERS, BudgetStopped, OpenRouter, Spend
from bench.fmp.stores import Stores

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def lme():
    module = load_lcm(REPO)
    from hermes_lcm import tokens
    tokens._encoder_ready = True
    tokens._encoder = None
    return module


@pytest.fixture
def question(lme):
    return datasets.normalize_lme({"question_id": "synthetic_abs", "question_type": "multi-session",
        "question": "Where is the synthetic teal key?", "answer": "unknown", "question_date": "2025/01/03 (Fri) 12:00",
        "haystack_session_ids": ["s1", "s2"], "haystack_dates": ["2025/01/01 12:00", "2025/01/02 12:00"],
        "haystack_sessions": [[{"role": "user", "content": "synthetic teal key " + "x" * 2000}],
                              [{"role": "assistant", "content": "another synthetic teal key"}]]}, "lme-s", lme)


@pytest.fixture
def stores(tmp_path, lme):
    return Stores(tmp_path / "stores", "off", REPO, "synthetic-sha", lme)


def test_datasets(question):
    assert question.is_abstention and question.question_date.startswith("2025-01-03")
    item = {"sample_id": "c0", "conversation": {"speaker_a": "A", "speaker_b": "B",
        "session_1_date_time": "1:30 pm on 2 January, 2025",
        "session_1": [{"speaker": "A", "text": "synthetic hello"}, {"speaker": "B", "text": "synthetic reply"}]},
        "qa": [{"category": 1, "question": "Synthetic?", "answer": "hello"}]}
    q = datasets.normalize_locomo(item)[0]
    assert q.store_id == "c0" and q.sessions[0]["turns"][1]["speaker"] == "B"
    assert q.sessions[0]["date"] == "2025-01-02T13:30:00+00:00"


def test_selection(tmp_path, question, lme):
    raw = {"question_id": "synthetic", "question_type": "multi-session", "question": "Synthetic?",
           "answer": "yes", "haystack_session_ids": [], "haystack_sessions": []}
    data, ids = tmp_path / "data.json", tmp_path / "ids.json"
    data.write_text(json.dumps([raw]))
    ids.write_text('["synthetic"]')
    qs, digest, ids_digest, _ = datasets.load(data, "lme-s", lme, ids)
    assert len(qs) == 1 and len(digest) == len(ids_digest) == 64
    ids.write_text('["missing"]')
    with pytest.raises(ValueError):
        datasets.load(data, "lme-s", lme, ids)


def test_store_reuse_rebuild(stores, question):
    assert stores.build(question) == "built"
    path = stores.directory(question) / "lcm.db"
    stamp = path.stat().st_mtime_ns
    assert stores.build(question) == "reused" and path.stat().st_mtime_ns == stamp
    stores.identity["dataset_sha256"] = "new-synthetic"
    assert stores.build(question) == "built"
    assert len(list(path.parent.parent.glob("*.superseded-*"))) == 1
    with stores.engine(question) as (engine, _):
        assert engine._store.get_batch([1])[1]["observed_at"] > 0
        with pytest.raises(Exception):
            engine._store._conn.execute("DELETE FROM messages")


def test_single_full_content_order_time(stores, question):
    stores.build(question)
    with stores.engine(question) as (engine, _):
        raw = json.dumps({"hits": [{"store_id": 2}, {"store_id": 1}]})
        with patch.object(delivery, "dispatch", return_value=raw):
            evidence, got = delivery.single(question, engine)
        assert got == raw and evidence.index("another synthetic") < evidence.index("x" * 2000)
        assert evidence.startswith("[2025-01-02T12:00:00+00:00] assistant:")


class Scripted:
    metadata = {m: {"context_length": 1_050_000} for m in READERS}

    def __init__(self, messages):
        self.messages, self.requests = iter(messages), []

    def call(self, model, messages, records, tools=None, tool_choice="auto", **kwargs):
        self.requests.append(json.loads(json.dumps({"messages": messages, "tools": tools,
                                                    "tool_choice": tool_choice, **kwargs})))
        records.append({"status": "ok", "input_tokens": 10, "finish_reason": "stop"})
        return next(self.messages)


def tool(name, args, ident="c1"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": ident, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def test_agentic_production_verbatim_and_budget(stores, question):
    from hermes_lcm import tools as production
    stores.build(question)
    fake = Scripted([tool("lcm_recall", {"query": "teal", "limit": 1}),
                     tool("lcm_expand", {"store_id": 1}, "c2"), {"role": "assistant", "content": "unknown"}])
    with stores.engine(question) as (engine, _), patch.object(production, "lcm_recall", wraps=production.lcm_recall) as recall, \
            patch.object(production, "lcm_expand", wraps=production.lcm_expand) as expand:
        result = delivery.answer(question, "agentic", fake, READERS[0], engine, tool_budget=2)
        assert recall.call_count == expand.call_count == 1
    assert result["tool_calls"] == 2 and result["hypothesis"] == "unknown"
    assert result["forced_final"] == "tool_calls"
    assert fake.requests[-1]["tools"] == fake.requests[0]["tools"] and fake.requests[-1]["tool_choice"] == "none"
    assert fake.requests[-1]["max_tokens"] == 32_000
    assert fake.requests[-1]["messages"][-1] == {"role": "user", "content": delivery.FINAL_NUDGE}
    outputs = [m["content"] for m in fake.requests[-1]["messages"] if m["role"] == "tool"]
    assert outputs == [t["result"] for t in result["trace"]]


def test_full_context_fits_or_explicit_skip(question):
    fake = Scripted([{"role": "assistant", "content": "unknown"}])
    result = delivery.answer(question, "full_context", fake, READERS[0])
    assert result["status"] == "ok" and "x" * 2000 in fake.requests[0]["messages"][1]["content"]
    fake.metadata = {READERS[0]: {"context_length": 16_001}}
    skipped = delivery.answer(question, "full_context", fake, READERS[0])
    assert skipped["status"] == "skipped" and skipped["reason"] == "does not fit"


@pytest.mark.parametrize("text,label", [("yes", True), ("NO", False), ("Yes.", True), ("no, it misses", False),
                                      ("**Yes**", True), ("yesterday", None), ("unclear", None), ("", None)])
def test_lme_parse(text, label):
    assert judges.parse(text, "lme-s") is label


@pytest.mark.parametrize("text,label", [('CORRECT', True), ('{"label":"WRONG"}', False),
    ('The answer names the same city.\n```json\n{"label": "CORRECT"}\n```', True),
    ("It refers to another date. WRONG", None), ('{"label":"CORRECT","extra":1}', True),
    ('Explanation mentions nothing.', None), ('{"label": "maybe"}', None)])
def test_locomo_parse(text, label):
    assert judges.parse(text, "locomo") is label


def test_blind_and_retry(question):
    text = judges.prompt(question, "synthetic hypothesis")
    assert all(secret not in text for secret in ("off", "full_context", "agentic", *READERS))
    assert "synthetic hypothesis" in text and "unanswerable" in text
    fake = Scripted([{"content": t} for t in ("unclear", "yes", "maybe", "unclear")])
    result, calls = judges.judge(question, "synthetic hypothesis", fake)
    assert len(calls) == 4 and result[JUDGES[0]]["correct"] is True
    assert result[JUDGES[1]]["status"] == "judge_error" and result[JUDGES[1]]["raw"] == ["maybe", "unclear"]
    assert result[JUDGES[0]]["raw"] == ["unclear", "yes"]


def test_statistics():
    assert score.mcnemar([1, 1, 1, 1], [0, 0, 0, 0])["p"] == .125
    assert score.bootstrap([1] * 5, [0] * 5) == [1, 1]
    assert score.bootstrap([0, 1] * 2, [0, 1] * 2) == [0, 0]
    result = score.agreement([1, 1, 0, 0], [1, 0, 1, 0])
    assert result["percent"] == 50 and result["kappa"] == 0


def test_spend_reservations_and_cap(tmp_path):
    spend = Spend(tmp_path / "spend.json", 1)
    spend.change(reserve=.75)
    with pytest.raises(BudgetStopped):
        spend.change(reserve=.5)
    spend.change(release=.75, cost=1)
    with pytest.raises(BudgetStopped):
        spend.change(reserve=.01)


def metadata():
    return {m: {"pricing": {"prompt": ".000002", "completion": ".00001"}, "context_length": 1_000_000}
            for m in READERS}


def test_openrouter_retry_accounting(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-key")
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"model": READERS[0], "provider": "Synthetic", "usage": {
            "cost": .01, "prompt_tokens": 20, "completion_tokens": 30,
            "completion_tokens_details": {"reasoning_tokens": 10}}, "choices": [{"message": {"content": "yes"}}]})
    client = OpenRouter(Spend(tmp_path / "ledger.json", 1), metadata(), httpx.MockTransport(handler), lambda _: None)
    calls = []
    assert client.call(READERS[0], [{"role": "user", "content": "Synthetic?"}], calls)["content"] == "yes"
    assert len(calls) == 2 and calls[-1]["retries"] == 1 and calls[-1]["reasoning_tokens"] == 10
    assert calls[-1]["finish_reason"] is None and "temperature" not in requests[-1]
    assert "parallel_tool_calls" not in requests[-1]
    assert requests[-1]["reasoning"] == {"effort": "high"} and requests[-1]["usage"]["include"]


def test_judge_one_provider_failure_still_scores_other(question):
    class Partial(Scripted):
        def call(self, model, messages, records, tools=None, **kwargs):
            if model == JUDGES[0]:
                raise RuntimeError("synthetic failure")
            return {"content": "no"}
    result, _ = judges.judge(question, "synthetic", Partial([]))
    assert result[JUDGES[0]]["status"] == "judge_error" and result[JUDGES[1]]["correct"] is False


def test_score_error_denominators_and_pairing(tmp_path):
    runs = []
    for mode in ("single", "agentic"):
        run = tmp_path / mode
        run.mkdir()
        runs.append(run)
        (run / "run-manifest.json").write_text(json.dumps({"config": "off", "delivery": mode,
            "reader": READERS[0], "dataset_sha256": "synthetic", "lcmx_commit": "c", "harness_commit": "c",
            "split_label": "synthetic", "id_list_sha256": "i", "status": "complete", "question_ids": ["0", "1"]}))
        rows = [{"qid": str(i), "qtype": "synthetic", "status": "ok" if i == 0 else "error",
                 "question": "Synthetic?", "gold": "yes", "hypothesis": "no",
                 "judges": {m: {"status": "ok", "correct": i == 0 or m == JUDGES[0]} for m in JUDGES}} for i in range(2)]
        (run / "judged.jsonl").write_text("\n".join(map(json.dumps, rows)))
    result = score.score(runs, tmp_path / "report")
    assert all(s["primary"] == .5 and s["reader_errors"] == 1 for s in result["arms"].values())
    assert all(s["agreement"]["percent"] == 100 for s in result["arms"].values())
    assert len(result["paired"]) == 1
    assert all(line.startswith("|") or not line for line in (tmp_path / "report/report.md").read_text().splitlines())


def test_parallel_batch_cannot_overrun_tool_budget(stores, question):
    stores.build(question)
    message = tool("lcm_describe", {})
    message["tool_calls"].extend(tool("lcm_describe", {}, "c2")["tool_calls"])
    fake = Scripted([message, {"role": "assistant", "content": "unknown"}])
    with stores.engine(question) as (engine, _):
        result = delivery.answer(question, "agentic", fake, READERS[0], engine, tool_budget=1)
    assert result["tool_calls"] == 1
    assert "budget exhausted" in result["trace"][-1]["result"]
    assert fake.requests[-1]["tool_choice"] == "none"


def test_reader_input_budget_forces_final_before_next_tools(stores, question):
    stores.build(question)
    fake = Scripted([{"role": "assistant", "content": "unknown"}])
    with stores.engine(question) as (engine, _):
        floor = delivery.tokens([[{}], delivery.schemas(engine)])
        result = delivery.answer(question, "agentic", fake, READERS[0], engine, input_budget=2 * floor)
    assert result["hypothesis"] == "unknown" and fake.requests[0]["tool_choice"] == "none"
    assert result["forced_final"] == "input_tokens"


def test_locomo_named_speaker_survives_production_tools(stores, question):
    question.sessions[0]["turns"][0]["speaker"] = "SyntheticSpeaker"
    stores.summaries = "deterministic"
    stores.identity["summaries"] = "deterministic"
    stores.build(question)
    with stores.engine(question) as (engine, _):
        raw = delivery.dispatch("lcm_expand", {"store_id": 1}, engine)
        assert "SyntheticSpeaker: synthetic teal key" in raw
        assert engine._dag.get_session_depth_stats("s0001")[0]["count"] == 1


# ---- round 1 (review D1-D13, L2-L5) ----

def leaky(lme):
    return datasets.normalize_lme({"question_id": "synthetic_leak", "question_type": "multi-session",
        "question": "Which synthetic colour did I pick?", "answer": "teal", "question_date": "2025/01/04 (Sat) 12:00",
        "haystack_session_ids": ["answer_synthetic_late", "sharegpt_synthetic_early"],
        "haystack_dates": ["2025/01/03 12:00", "2025/01/01 12:00"],
        "haystack_sessions": [[{"role": "user", "content": "I picked the synthetic colour teal."}],
                              [{"role": "assistant", "content": "Synthetic colours are fun."}]]}, "lme-s", lme)


def test_session_ids_opaque_in_date_order_and_never_reader_visible(stores, lme):
    q = leaky(lme)
    assert [s["session_id"] for s in q.sessions] == ["s0001", "s0002"]
    assert q.session_map == {"s0001": "sharegpt_synthetic_early", "s0002": "answer_synthetic_late"}
    stores.build(q)
    assert json.loads((stores.directory(q) / "build-manifest.json").read_text())["session_map"] == q.session_map
    script = [tool("lcm_recall", {"query": "synthetic colour"}, "c1"), tool("lcm_grep", {"query": "teal"}, "c2"),
              tool("lcm_load_session", {"session_id": "s0002"}, "c3"), tool("lcm_describe", {}, "c4"),
              tool("lcm_recent", {}, "c5"), {"role": "assistant", "content": "teal"}]
    seen = []
    for mode, messages in (("agentic", script), ("single", script[-1:]), ("full_context", script[-1:])):
        fake = Scripted(messages)
        with stores.engine(q) as (engine, _):
            result = delivery.answer(q, mode, fake, READERS[0], engine)
        assert result["status"] == "ok"
        seen.append(json.dumps(fake.requests))
    assert "s0002" in seen[0] and "teal" in seen[1]
    assert not any(original in text for text in seen for original in q.session_map.values())


def test_locomo_adversarial_excluded_and_counted(tmp_path, lme):
    item = {"sample_id": "c0", "conversation": {"speaker_a": "A", "speaker_b": "B",
        "session_1_date_time": "1:30 pm on 2 January, 2025", "session_1": [{"speaker": "A", "text": "synthetic"}]},
        "qa": [{"category": 1, "question": "Synthetic?", "answer": "x"},
               {"category": 5, "question": "Adversarial?", "adversarial_answer": "y"}]}
    data = tmp_path / "locomo.json"
    data.write_text(json.dumps([item]))
    questions, _, _, excluded = datasets.load(data, "locomo", lme)
    assert [q.qtype for q in questions] == ["1"] and excluded == {"locomo_category_5_adversarial": 1}
    assert questions[0].sessions[0]["session_id"] == "s0001" and questions[0].session_map == {"s0001": "session_1"}


def test_token_budget_keeps_tools_until_exhausted_and_holds(stores, question):
    stores.build(question)

    class Reported(Scripted):
        def call(self, model, messages, records, tools=None, tool_choice="auto", **kwargs):
            message = super().call(model, messages, records, tools, tool_choice, **kwargs)
            records[-1]["input_tokens"] = delivery.tokens([messages, tools])
            return {"role": "assistant", "content": "x"} if tool_choice == "none" else message
    with stores.engine(question) as (engine, _):
        floor = delivery.tokens([[{}], delivery.schemas(engine)])
        budget = 5 * floor + 12_000
        fake = Reported([tool("lcm_describe", {}, f"c{i}") for i in range(12)] + [{"role": "assistant", "content": "x"}])
        result = delivery.answer(question, "agentic", fake, READERS[0], engine, input_budget=budget)
    # Byte-based admission (the defect) withdrew tools before the first call; tokens allow rounds.
    assert result["tool_calls"] >= 2 and result["forced_final"] == "input_tokens"
    assert sum(c["input_tokens"] for c in result["calls"]) <= budget
    assert [r["tool_choice"] for r in fake.requests][-1] == "none"


def test_oversized_tool_result_replaced_to_hold_budget(stores, question):
    stores.build(question)
    fake = Scripted([tool("lcm_describe", {}), {"role": "assistant", "content": "x"}])
    with stores.engine(question) as (engine, _), patch.object(delivery, "dispatch", return_value="z" * 1_000_000):
        result = delivery.answer(question, "agentic", fake, READERS[0], engine)
    assert "input token budget exhausted" in result["trace"][0]["result"] and result["tool_calls"] == 0


def test_tool_errors_return_to_reader_and_question_continues(stores, question):
    from hermes_lcm import tools as production
    stores.build(question)
    broken = tool("lcm_recall", {})
    broken["tool_calls"][0]["function"]["arguments"] = "{not json"
    fake = Scripted([broken, tool("lcm_grep", {"query": "teal"}, "c2"), {"role": "assistant", "content": "unknown"}])
    with stores.engine(question) as (engine, _), patch.object(production, "lcm_grep", side_effect=ValueError("boom")):
        result = delivery.answer(question, "agentic", fake, READERS[0], engine)
    errors = [json.loads(t["result"])["error"] for t in result["trace"]]
    assert result["status"] == "ok" and errors[0].startswith("Invalid JSON arguments") and "boom" in errors[1]


def test_toolset_is_production_registration_under_effective_config(stores, question):
    from hermes_lcm.engine import LCMEngine
    stores.build(question)
    with stores.engine(question) as (engine, _):
        names = [s["function"]["name"] for s in delivery.schemas(engine)]
        assert "lcm_expand_query" not in names and not {"lcm_status", "lcm_inspect", "lcm_doctor"} & set(names)
        assert "lcm_query_state" not in names and "lcm_compile_evidence" in names
        engine._config.assertions_enabled = True
        assert "lcm_query_state" in [s["function"]["name"] for s in delivery.schemas(engine)]
    production = [s["name"] for s in LCMEngine.get_tool_schemas(SimpleNamespace(_omitted_tool_names=set))]
    assert set(names) < set(production)
    assert delivery.dispatch("lcm_expand_query", {}, None, delivery.schemas(engine)).startswith('{"error"')


def test_tools_run_at_question_date(stores, question):
    from hermes_lcm import tools as production
    stores.build(question)
    seen = []

    def fake_recall(args, engine):
        seen.append(production.time.time())
        worker = production.threading.Thread(target=lambda: seen.append(production.time.time()))
        worker.start()
        worker.join()
        return "{}"
    fake = Scripted([tool("lcm_recall", {"query": "q"}), {"role": "assistant", "content": "x"}])
    with stores.engine(question) as (engine, _), patch.object(production, "lcm_recall", side_effect=fake_recall):
        delivery.answer(question, "agentic", fake, READERS[0], engine)
    stamp = datetime.fromisoformat(question.question_date).timestamp()
    assert seen == [stamp, stamp] and abs(production.time.time() - time.time()) < 5


def test_agentic_prompt_says_history_is_not_in_prompt(stores, question):
    stores.build(question)
    fake = Scripted([{"role": "assistant", "content": "x"}] * 2)
    with stores.engine(question) as (engine, _):
        delivery.answer(question, "agentic", fake, READERS[0], engine)
        delivery.answer(question, "single", fake, READERS[0], engine)
    agentic, single = fake.requests[0]["messages"], fake.requests[1]["messages"]
    assert "NOT in this prompt" in agentic[0]["content"] and "History evidence" not in agentic[1]["content"]
    assert "History evidence" in single[1]["content"] and "NOT in this prompt" not in single[0]["content"]


def test_store_path_names_build_inputs_and_engine_rechecks(stores, question, tmp_path, lme):
    stores.build(question)
    other = Stores(tmp_path / "stores", "off", REPO, "synthetic-sha", lme, "deterministic")
    assert other.directory(question) != stores.directory(question)
    assert stores.identity["lcmx_commit"][:12] in stores.generation and "off-none-" in stores.generation
    stores.identity["config"] = "changed"
    with pytest.raises(RuntimeError, match="manifest"):
        with stores.engine(question):
            pass


def test_locomo_judge_uses_system_prompt_and_temperature_when_supported():
    q = datasets.Question("c0:0", "locomo", "1", "Synthetic?", "x", "", False, [])
    fake = Scripted([{"content": '{"label": "CORRECT"}'}] * 2)
    result, _ = judges.judge(q, "x", fake)
    assert all(r["correct"] for r in result.values())
    assert fake.requests[0]["messages"][0]["role"] == "system" and fake.requests[0]["temperature"] == 0


def test_payload_error_retried_and_supported_temperature_only(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-key")
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, json={"error": {"code": 502, "message": "synthetic upstream"}})
        return httpx.Response(200, json={"model": READERS[1], "provider": "Synthetic", "usage": {
            "cost": .01, "prompt_tokens": 20, "completion_tokens": 30},
            "choices": [{"message": {"content": "x"}, "finish_reason": "length"}]})
    meta = metadata()
    meta[READERS[1]]["supported_parameters"] = ["tools", "tool_choice", "reasoning"]
    client = OpenRouter(Spend(tmp_path / "ledger.json", 5), meta, httpx.MockTransport(handler), lambda _: None)
    calls = []
    client.call(READERS[1], [{"role": "user", "content": "Synthetic?"}], calls, temperature=0, max_tokens=32_000)
    assert calls[0]["status"] == "error" and calls[0]["error"] == "PayloadError" and calls[-1]["finish_reason"] == "length"
    assert "temperature" not in requests[-1] and requests[-1]["max_tokens"] == 32_000


def test_payment_required_is_a_spend_stop_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-key")
    client = OpenRouter(Spend(tmp_path / "ledger.json", 5), metadata(),
                        httpx.MockTransport(lambda request: httpx.Response(402)), lambda _: None)
    calls = []
    with pytest.raises(BudgetStopped):
        client.call(READERS[0], [{"role": "user", "content": "Synthetic?"}], calls)
    assert len(calls) == 1 and calls[0]["http_status"] == 402 and calls[0]["cost"] == 0


def test_error_and_stopped_rows_not_judged_budget_stop_is_not_run(question):
    class Stopping(Scripted):
        def call(self, *args, **kwargs):
            raise BudgetStopped()
    error_row = {"qid": question.qid, "status": "error", "hypothesis": ""}
    assert judges.judge_row(error_row, question, Stopping([])) is error_row
    stopped = judges.judge_row({"qid": question.qid, "status": "ok", "hypothesis": "x"}, question, Stopping([]))
    assert stopped["status"] == "not_run" and "judges" not in stopped


def write_arm(root, mode, statuses, **manifest):
    run = root / mode
    run.mkdir()
    (run / "run-manifest.json").write_text(json.dumps({"config": "off", "delivery": mode, "reader": READERS[0],
        "dataset_sha256": "d", "lcmx_commit": "c", "harness_commit": "c", "split_label": "s",
        "id_list_sha256": "i", "status": "complete", "question_ids": [str(i) for i in range(len(statuses))], **manifest}))
    (run / "judged.jsonl").write_text("\n".join(json.dumps({"qid": str(i), "qtype": "t", "status": status,
        "question": "Synthetic?", "gold": "g", "hypothesis": "h",
        "judges": {m: {"status": "ok", "correct": m == JUDGES[0]} for m in JUDGES}}) for i, status in enumerate(statuses)))
    return run


def test_score_refuses_partial_and_mismatched_arms(tmp_path):
    full = write_arm(tmp_path, "single", ["ok"] * 60)
    partial = write_arm(tmp_path, "agentic", ["ok"] * 59 + ["not_run"], status="incomplete")
    with pytest.raises(ValueError, match="allow-partial"):
        score.score([full, partial], tmp_path / "r")
    result = score.score([full, partial], tmp_path / "r", allow_partial=True)
    assert result["arms"]["off/agentic/" + READERS[0]]["scored_n"] == 59
    assert result["adjudication"] == {"disagreements": 119, "sampled": 50, "seed": 1189}
    sampled = [json.loads(x)["qid"] for x in (tmp_path / "r/adjudicate.jsonl").read_text().splitlines()]
    assert sampled != [str(i) for i in range(50)]
    other = write_arm(tmp_path, "full_context", ["ok"] * 60, lcmx_commit="other")
    with pytest.raises(ValueError, match="lcmx_commit"):
        score.score([full, other], tmp_path / "r2")


# ---- round 2 (review r1: P1 verdicts, P1 paired errors, P2 budget, recency, 402, reservation) ----

@pytest.mark.parametrize("text,label", [('{"label":"WRONG","reason":"not CORRECT"}', False),
    ("I cannot determine whether this is CORRECT.", None), ("CORRECT or WRONG", None),
    ('The answer is correct.\n**CORRECT**', None), ('Label: WRONG.', None),
    ('{"label": "CORRECT"} {"label": "WRONG"}', None), ("It is not WRONG to say so. CORRECT", None)])
def test_locomo_verdict_never_inverts(text, label):
    assert judges.parse(text, "locomo") is label


def test_locomo_verdict_parse_is_linear():
    started = time.monotonic()
    for text in (": " + " " * 100_000 + "x", "{" * 100_000, ". " * 100_000, "\n" * 100_000):
        assert judges.parse(text, "locomo") is None
    assert time.monotonic() - started < 2


def test_paired_tests_drop_an_error_in_either_arm(tmp_path):
    a = write_arm(tmp_path, "single", ["ok"] * 2 + ["error"] * 8)
    b = write_arm(tmp_path, "agentic", ["ok"] * 10)
    pair = score.score([a, b], tmp_path / "r")["paired"]["off/single/{0} vs off/agentic/{0}".format(READERS[0])]
    assert pair["per_judge"][JUDGES[0]]["p"] == 1.0 and pair["per_judge"][JUDGES[0]]["n"] == 2
    assert pair["paired_n"] == 2 and pair["dropped_for_errors"] == 8 and pair["primary_difference_ci95"] == [0, 0]


def test_reported_input_over_budget_is_not_ok(question):
    class Heavy(Scripted):
        def call(self, *args, **kwargs):
            message = super().call(*args, **kwargs)
            args[2][-1]["input_tokens"] = 206_000
            return message
    result = delivery.answer(question, "full_context", Heavy([{"role": "assistant", "content": "x"}]), READERS[0])
    assert result["status"] == "budget_exceeded" and result["input_tokens"] == 206_000


def test_payment_required_body_or_status_stops_the_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-key")
    for n, response in enumerate((httpx.Response(200, json={"error": {"code": 402, "message": "synthetic"}}),
                                  httpx.Response(402))):
        requests = []
        ledger = Spend(tmp_path / f"ledger{n}.json", 5)

        def handler(request, response=response):
            requests.append(request)
            return response
        client = OpenRouter(ledger, metadata(), httpx.MockTransport(handler), lambda _: None)
        for _ in range(2):
            with pytest.raises(BudgetStopped):
                client.call(READERS[0], [{"role": "user", "content": "Synthetic?"}], [])
        assert len(requests) == 1 and ledger.change()["stopped"] is True and ledger.change()["spent"] == 0


def test_store_ingest_time_is_session_date_for_lme_and_locomo(stores, lme):
    from hermes_lcm import tools as production
    lme_q = leaky(lme)
    locomo_q = datasets.normalize_locomo({"sample_id": "c0", "conversation": {"speaker_a": "A", "speaker_b": "B",
        "session_1_date_time": "1:30 pm on 2 January, 2024", "session_1": [{"speaker": "A", "text": "synthetic old"}],
        "session_2_date_time": "1:30 pm on 2 January, 2025", "session_2": [{"speaker": "B", "text": "synthetic new"}]},
        "qa": [{"category": 1, "question": "Synthetic?", "answer": "x"}]})[0]
    for q in (lme_q, locomo_q):
        stores.build(q)
        with stores.engine(q) as (engine, _):
            rows = engine._store.get_batch([1, 2])
        dates = [datetime.fromisoformat(s["date"]).timestamp() for s in q.sessions]
        assert [rows[i]["timestamp"] for i in (1, 2)] == dates
        now = datetime.fromisoformat(q.question_date).timestamp()
        old, new = (production._lcm_recall_recency_boost(rows[i]["timestamp"], now=now) for i in (1, 2))
        assert old < new
    assert abs(production.time.time() - time.time()) < 5


def test_discarded_tool_result_releases_its_reservation(stores, question):
    stores.build(question)
    fake = Scripted([tool("lcm_describe", {}), tool("lcm_describe", {}, "c2"), {"role": "assistant", "content": "x"}])
    results = iter(["z" * 1_000_000, "{}"])
    with stores.engine(question) as (engine, _), patch.object(delivery, "dispatch", side_effect=lambda *a: next(results)):
        result = delivery.answer(question, "agentic", fake, READERS[0], engine)
    assert "input token budget exhausted" in result["trace"][0]["result"]
    assert [r["tool_choice"] for r in fake.requests] == ["auto", "auto", "auto"] and result["tool_calls"] == 1
    assert result["forced_final"] is None


# ---- subscription transport: codex exec / claude -p, MCP reader, canary ----

import subprocess  # noqa: E402

from bench.fmp import mcp_reader, models, run  # noqa: E402

SOL, SOL_JUDGE, OPUS = "codex:gpt-6.1-sol@high", "codex:gpt-6.1-sol@medium", "claude:claude-opus-5-5"


def test_model_specs_and_openrouter_is_opt_in(tmp_path):
    assert models.parse_spec(SOL) == ("codex", "gpt-6.1-sol", "high")
    assert models.parse_spec(OPUS) == ("claude", "claude-opus-5-5", None)
    assert models.parse_spec(READERS[0]) == ("openrouter", READERS[0], None)
    assert models.parse_spec("openrouter:" + READERS[0]) == ("openrouter", READERS[0], None)
    for bad in ("codex:gpt-6.1-sol", "claude:claude-opus-5-5@high", "codex:@high"):
        with pytest.raises(ValueError):
            models.parse_spec(bad)
    quota = models.Quota(tmp_path / "q.json")
    with patch.object(models, "OpenRouter", side_effect=AssertionError("no metered client")):
        client = models.Backends(tmp_path, quota, [SOL, SOL_JUDGE], codex=SimpleNamespace(close=lambda: None))
        assert client.openrouter is None and client.claude is None
        with pytest.raises(ValueError, match="opt-in"):
            models.Backends(tmp_path, quota, ["openrouter:" + READERS[0]], codex=object())
    args = run.parser().parse_args(["answer", "--split-label", "s", "--store-root", "x", "--output", "y"])
    assert args.reader == SOL and args.judge == [SOL_JUDGE] and args.agreement_judge == OPUS
    assert args.workers == 4 and args.max_spend is None


class FakeCodex:
    """Writes codex exec --json events and a rollout like codex-cli 0.162.1 does."""
    def __init__(self, backend, text="teal", model="gpt-6.1-sol", items=(), calls=(), used=25.0, failure=None):
        self.backend, self.commands, self.envs, self.stdins = backend, [], [], []
        self.text, self.model, self.items, self.calls, self.used, self.failure = text, model, items, calls, used, failure

    def __call__(self, cmd, stdin, capture_output, text, env, cwd, timeout):
        from bench.fmp.tests.test_mcp_readiness import fake_mcp_ready
        fake_mcp_ready(Path(stdin.name).parent)
        self.commands.append(cmd)
        self.envs.append(env)
        self.stdins.append(Path(stdin.name).read_text())
        assert not any(Path(cwd).iterdir())
        thread = "thread-" + str(len(self.commands))
        rollout = Path(env["CODEX_HOME"]) / "sessions" / "2026" / f"rollout-x-{thread}.jsonl"
        rollout.parent.mkdir(parents=True, exist_ok=True)
        lines = [{"type": "turn_context", "payload": {"model": self.model}}]
        lines += [{"type": "response_item", "payload": {"type": t, "name": n}} for t, n in self.calls]
        lines += [{"type": "event_msg", "payload": {"type": "item_completed", "item": i}} for i in self.items]
        lines += [{"type": "event_msg", "payload": {"type": "token_count", "info": {"x": 1}, "rate_limits": {
            "primary": {"used_percent": self.used, "window_minutes": 10080}}}}]
        rollout.write_text("\n".join(map(json.dumps, lines)) + "\n")
        events = [{"type": "thread.started", "thread_id": thread}, {"type": "turn.started"}]
        if self.failure:
            events.append({"type": "turn.failed", "error": {"message": self.failure}})
        else:
            events += [{"type": "item.completed", "item": {"type": "agent_message", "text": self.text}},
                       {"type": "turn.completed", "usage": {"input_tokens": 9_800, "cached_input_tokens": 7_000,
                                                            "output_tokens": 5, "reasoning_output_tokens": 2}}]
        return subprocess.CompletedProcess(cmd, 1 if self.failure else 0, "\n".join(map(json.dumps, events)), "")


def codex_backend(tmp_path, **fake):
    quota = models.Quota(tmp_path / "quota.json")
    backend = models.Codex(tmp_path / "run", quota, runner=lambda *a, **k: None)
    backend.runner = FakeCodex(backend, **fake)
    return backend, quota


def test_codex_backend_isolated_text_call_and_accounting(tmp_path):
    backend, quota = codex_backend(tmp_path)
    records = []
    try:
        message = backend.call(SOL, [{"role": "system", "content": "Sys."}, {"role": "user", "content": "Q?"}], records)
        cmd, env = backend.runner.commands[0], backend.runner.envs[0]
        assert message["content"] == "teal" and backend.runner.stdins[0] == "Sys.\n\nQ?" and cmd[-1] == "-"
        for flag in ("--json", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check"):
            assert flag in cmd
        assert cmd[cmd.index("-m") + 1] == "gpt-6.1-sol" and "model_reasoning_effort=high" in cmd
        assert cmd[cmd.index("-s") + 1] == "read-only" and "web_search=disabled" in cmd
        disabled = {cmd[i + 1] for i, a in enumerate(cmd) if a == "--disable"}
        assert {"shell_tool", "unified_exec", "apps", "plugins", "multi_agent"} <= disabled
        cwd = Path(cmd[cmd.index("--cd") + 1])
        assert cwd.parent.parent == tmp_path / "run" / "harness" and not list(cwd.iterdir())
        assert Path(env["CODEX_HOME"], "auth.json").is_symlink() and env["HOME"] != str(Path.home())
        assert not any(k.startswith(("OPENROUTER", "OPENAI", "CODEX_API")) for k in env)
        assert not list((tmp_path / "run").rglob("prompt.txt"))  # corpus does not stay in the run tree
        record = records[0]
        assert record["status"] == "ok" and record["returned_model"] == "gpt-6.1-sol" and record["input_tokens"] == 9_800
        assert record["wrapper_overhead_tokens"] is None and record["input_tokens_label"] == "total incl. retrieved evidence"
        assert record["weekly_used_percent"] == 25.0 and quota.state["codex"]["weekly_used_percent_first"] == 25.0
    finally:
        backend.close()
    assert not backend.scratch.exists() and (Path.home() / ".codex" / "auth.json").exists()


@pytest.mark.parametrize("fake", [{"calls": [("function_call", "shell")]}, {"calls": [("custom_tool_call", "apply_patch")]},
                                  {"items": [{"type": "FileChange"}]}, {"items": [{"type": "McpToolCall", "server": "other",
                                                                                    "tool": "read"}]}])
def test_codex_any_non_mcp_tool_is_a_violation(tmp_path, fake):
    backend, _ = codex_backend(tmp_path, **fake)
    records = []
    with pytest.raises(models.ToolViolation):
        backend.call(SOL, [{"role": "user", "content": "Q?"}], records)
    assert records[0]["status"] == "tool_violation" and records[0]["violations"]
    backend.close()


def test_codex_mcp_calls_through_code_mode_are_allowed(tmp_path):
    backend, _ = codex_backend(tmp_path, calls=[("custom_tool_call", "exec")], items=[
        {"type": "McpToolCall", "server": "lcmx", "tool": "lcm_grep"},
        {"type": "McpToolCall", "server": "codex", "tool": "list_mcp_resources"}, {"type": "Reasoning"}])
    records = []
    server = {"repo": "r", "question": {"question_date": ""}}
    backend.agentic(SOL, [{"role": "user", "content": "Q?"}], records, server)
    cmd = backend.runner.commands[0]
    assert records[0]["mcp_calls"] == 1 and records[0]["violations"] == []
    assert 'mcp_servers.lcmx.default_tools_approval_mode="approve"' in cmd
    assert any(a.startswith("mcp_servers.lcmx.args=") and "mcp_reader.py" in a for a in cmd)
    spec = json.loads(Path(records[0]["mcp_log"]).with_name("server.json").read_text())
    assert spec["max_calls"] == 30 and spec["log"] == records[0]["mcp_log"]
    backend.close()


def test_codex_model_mismatch_usage_limit_and_weekly_brake(tmp_path):
    backend, _ = codex_backend(tmp_path, model="gpt-other")
    with pytest.raises(RuntimeError, match="model_mismatch"):
        backend.call(SOL, [{"role": "user", "content": "Q?"}], [])
    backend.close()
    backend, quota = codex_backend(tmp_path / "b", failure="You've hit your usage limit.")
    with pytest.raises(BudgetStopped):
        backend.call(SOL, [{"role": "user", "content": "Q?"}], [])
    assert quota.state["codex"]["stopped"]
    backend.close()
    backend, quota = codex_backend(tmp_path / "c", used=95.0)
    backend.call(SOL, [{"role": "user", "content": "Q?"}], [])  # the call that crosses still answers
    with pytest.raises(BudgetStopped):
        backend.call(SOL, [{"role": "user", "content": "Q?"}], [])
    assert len(backend.runner.commands) == 1
    backend.close()


class FakeClaude:
    def __init__(self, models_seen=("claude-opus-5-5",), tools=(), limit=False, deny=False):
        self.models, self.tools, self.limit, self.deny, self.commands, self.envs = models_seen, tools, limit, deny, [], []

    def __call__(self, cmd, stdin, capture_output, text, env, cwd, timeout):
        from bench.fmp.tests.test_mcp_readiness import fake_mcp_ready
        fake_mcp_ready(Path(stdin.name).parent)
        self.commands.append(cmd)
        self.envs.append(dict(env))
        self.cwds = [*getattr(self, "cwds", []), Path(cwd)]
        if self.deny:
            Path(stdin.name).parent.joinpath("denied.jsonl").write_text(json.dumps({"tool": "Bash"}) + "\n")
        events = [{"type": "system", "subtype": "init", "tools": []},
                  {"type": "assistant", "message": {"model": self.models[0], "content": [
                      {"type": "tool_use", "name": t} for t in self.tools] + [{"type": "text", "text": "teal"}]}}]
        if self.limit:
            events.append({"type": "rate_limit_event", "rate_limit_info": {"status": "rejected"}})
        events.append({"type": "result", "subtype": "success", "is_error": False, "result": "teal", "num_turns": 1,
                       "stop_reason": "end_turn", "usage": {"input_tokens": 400, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 0,
                       "output_tokens": 3}, "modelUsage": {m: {} for m in self.models}, "permission_denials": []})
        return subprocess.CompletedProcess(cmd, 0, "\n".join(map(json.dumps, events)), "")


def claude_backend(tmp_path, monkeypatch, **fake):
    for key in ("CLAUDECODE", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SDK_HAS_OAUTH_REFRESH",
                "ANTHROPIC_BASE_URL", "USE_STAGING_OAUTH", "CLAUDE_AGENT_SDK_VERSION"):
        monkeypatch.setenv(key, "1")
    quota = models.Quota(tmp_path / "quota.json")
    backend = models.Claude(tmp_path / "run", quota, runner=FakeClaude(**fake), token=lambda: "synthetic-token")
    return backend, quota


def test_claude_backend_isolated_auth_no_tools_and_model_check(tmp_path, monkeypatch):
    # HOME is tmp_path, so a call cwd regressing under the run directory would sit inside HOME.
    monkeypatch.setenv("HOME", str(tmp_path))
    backend, _ = claude_backend(tmp_path, monkeypatch)
    records = []
    message = backend.call(OPUS, [{"role": "system", "content": "Sys."}, {"role": "user", "content": "Q?"}], records)
    cmd, env = backend.runner.commands[0], backend.runner.envs[0]
    assert message["content"] == "teal" and records[0]["input_tokens"] == 420 and records[0]["status"] == "ok"
    assert cmd[cmd.index("--tools") + 1] == "" and cmd[cmd.index("--system-prompt") + 1] == "Sys."
    assert cmd[cmd.index("--model") + 1] == "claude-opus-5-5" and "--strict-mcp-config" in cmd
    assert cmd[cmd.index("--output-format") + 1] == "stream-json" and "--mcp-config" not in cmd
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "synthetic-token"
    # No ancestor of the working directory may carry CLAUDE.md memory (the run output sits under HOME).
    assert not backend.runner.cwds[0].is_relative_to(Path.home()) and not any(backend.runner.cwds[0].iterdir())
    assert Path(env["CLAUDE_CONFIG_DIR"], "settings.json").read_text().strip() == "{}"
    assert set(env) <= {"HOME", "USER", "LOGNAME", "PATH", "TMPDIR", "SHELL", "LANG", "CLAUDE_CONFIG_DIR",
                        "CLAUDE_CODE_OAUTH_TOKEN"}
    assert "synthetic-token" not in json.dumps(records)
    backend.close()
    backend.runner.models = ("claude-opus-5-5", "claude-haiku-5-5")  # a side call is auxiliary, not the reply
    records = []
    backend.call(OPUS, [{"role": "user", "content": "Q?"}], records)
    assert records[0]["status"] == "ok" and list(records[0]["auxiliary_models"]) == ["claude-haiku-5-5"]
    backend.close()
    backend, _ = claude_backend(tmp_path / "b", monkeypatch, models_seen=("claude-other",))
    with pytest.raises(RuntimeError, match="model_mismatch"):
        backend.call(OPUS, [{"role": "user", "content": "Q?"}], [])
    backend.close()


def test_claude_agentic_hook_violations_and_usage_limit(tmp_path, monkeypatch):
    backend, _ = claude_backend(tmp_path, monkeypatch, tools=("mcp__lcmx__lcm_grep",))
    records = []
    backend.agentic(OPUS, [{"role": "user", "content": "Q?"}], records, {"question": {}})
    cmd = backend.runner.commands[0]
    config = json.loads(cmd[cmd.index("--mcp-config") + 1])["mcpServers"]["lcmx"]
    settings = json.loads(cmd[cmd.index("--settings") + 1])["hooks"]["PreToolUse"][0]
    assert records[0]["mcp_calls"] == 1 and cmd[cmd.index("--allowedTools") + 1] == "mcp__lcmx"
    assert config["args"][0].endswith("mcp_reader.py") and settings["matcher"] == "*"
    assert "deny_hook.py" in settings["hooks"][0]["command"]
    backend.close()
    for fake in ({"tools": ("Read",)}, {"deny": True}):
        backend, _ = claude_backend(tmp_path / str(len(fake)) / list(fake)[0], monkeypatch, **fake)
        with pytest.raises(models.ToolViolation):
            backend.agentic(OPUS, [{"role": "user", "content": "Q?"}], [], {"question": {}})
        backend.close()
    backend, quota = claude_backend(tmp_path / "limit", monkeypatch, limit=True)
    with pytest.raises(BudgetStopped):
        backend.call(OPUS, [{"role": "user", "content": "Q?"}], [])
    assert quota.state["claude"] == {"usage_limit_errors": 1, "stopped": True}
    with pytest.raises(BudgetStopped):
        backend.call(OPUS, [{"role": "user", "content": "Q?"}], [])
    assert len(backend.runner.commands) == 1
    backend.close()


def test_deny_hook_denies_everything_but_lcmx_mcp(tmp_path):
    log = tmp_path / "denied.jsonl"
    for name, denied in (("Bash", True), ("Read", True), ("mcp__other__x", True), ("mcp__lcmx__lcm_grep", False)):
        out = subprocess.run([sys.executable, "-I", str(models.DENY_HOOK), str(log)], input=json.dumps(
            {"tool_name": name}), capture_output=True, text=True, check=True).stdout
        assert ('"deny"' in out) is denied
    assert [json.loads(x)["tool"] for x in log.read_text().splitlines()] == ["Bash", "Read", "mcp__other__x"]


def rpc(spec, *requests):
    lines = "".join(json.dumps(r) + "\n" for r in requests)
    out = subprocess.run([sys.executable, str(Path(mcp_reader.__file__)), str(spec)], input=lines, capture_output=True,
                         text=True, check=True, env={**__import__("os").environ, "HERMES_DISABLE_LAZY_INSTALLS": "1"},
                         timeout=120).stdout
    return [json.loads(line) for line in out.splitlines()]


def test_mcp_reader_serves_production_outputs_unchanged(stores, question, tmp_path):
    stores.build(question)
    spec = tmp_path / "server.json"
    log = tmp_path / "mcp-log.jsonl"
    spec.write_text(json.dumps({**run.server_spec(SimpleNamespace(lcmx_repo=REPO, store_root=stores.root, config="off",
        summaries="none", embed_cache=None), "synthetic-sha", question), "log": str(log), "max_calls": 2}))
    replies = rpc(spec, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
                  {"jsonrpc": "2.0", "method": "notifications/initialized"},
                  {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                  *[{"jsonrpc": "2.0", "id": 3 + i, "method": "tools/call",
                     "params": {"name": "lcm_grep", "arguments": {"query": "teal"}}} for i in range(3)],
                  {"jsonrpc": "2.0", "id": 9, "method": "nope"})
    assert [r["id"] for r in replies] == [1, 2, 3, 4, 5, 9] and replies[-1]["error"]["code"] == -32601
    with stores.engine(question) as (engine, _):
        offered = delivery.schemas(engine)
        with delivery.question_clock(question.question_date):
            local = delivery.dispatch("lcm_grep", {"query": "teal"}, engine, offered)
    assert [t["name"] for t in replies[1]["result"]["tools"]] == [s["function"]["name"] for s in offered]
    assert replies[1]["result"]["tools"][0]["inputSchema"] == offered[0]["function"]["parameters"]
    assert replies[2]["result"]["content"][0]["text"] == local and "teal" in local
    assert json.loads(replies[4]["result"]["content"][0]["text"])["error"].startswith("tool budget exhausted")
    entries = [entry for x in log.read_text().splitlines() if "tool" in (entry := json.loads(x))]
    assert [e["counted"] for e in entries] == [True, True, False]
    assert set(entries[0]) == {"at", "tool", "counted", "error", "args_sha256", "latency_s", "output_bytes"}
    assert "teal" not in log.read_text()


def test_mcp_reader_applies_question_clock(stores, question, tmp_path):
    from hermes_lcm import tools as production
    stores.build(question)
    seen = []
    spec = {**run.server_spec(SimpleNamespace(lcmx_repo=REPO, store_root=stores.root, config="off", summaries="none",
                                              embed_cache=None), "synthetic-sha", question),
            "log": str(tmp_path / "log.jsonl"), "max_calls": 30}
    with stores.engine(question) as (engine, _), patch.object(production, "lcm_recall",
                                                             side_effect=lambda a, engine: seen.append(production.time.time()) or "{}"):
        reader = mcp_reader.Reader(spec, engine, delivery.schemas(engine), spec["log"])
        reader.call("lcm_recall", {"query": "q"})
    assert seen == [datetime.fromisoformat(question.question_date).timestamp()]


def test_harness_agentic_delivery_reads_server_trace(question, tmp_path):
    log = tmp_path / "mcp-log.jsonl"

    class Agent:
        def agentic(self, spec, messages, records, server):
            assert "NOT in this prompt" in messages[0]["content"] and server == {"s": 1}
            log.write_text("".join(json.dumps({"tool": "lcm_grep", "counted": c}) + "\n" for c in [True] * 30 + [False]))
            records.append({"status": "ok", "mcp_log": str(log), "finish_reason": "stop", "input_tokens": 1})
            return {"content": "teal"}
    result = delivery.answer(question, "agentic", Agent(), SOL, server={"s": 1})
    assert result["hypothesis"] == "teal" and result["tool_calls"] == 30 and result["forced_final"] == "tool_calls"


def test_agreement_judge_on_sample_and_secondary_stop_keeps_primary(question):
    assert run.sample([str(i) for i in range(30)], .1, 1189) == run.sample([str(i) for i in range(30)][::-1], .1, 1189)
    assert len(run.sample(range(81), .1, 1189)) == 9

    class Routed(Scripted):
        def call(self, model, messages, records, **kwargs):
            if model == OPUS:
                raise BudgetStopped()
            return {"content": "yes"}
    result, _ = judges.judge(question, "x", Routed([]), models=[SOL_JUDGE], agreement_model=OPUS)
    assert result[SOL_JUDGE]["correct"] is True and result[OPUS]["status"] == "not_run"


def test_score_with_codex_primary_and_sampled_agreement_judge(tmp_path):
    arms = []
    for mode in ("single", "agentic"):
        run_dir = tmp_path / mode
        run_dir.mkdir()
        arms.append(run_dir)
        (run_dir / "run-manifest.json").write_text(json.dumps({"config": "off", "delivery": mode, "reader": SOL,
            "dataset_sha256": "d", "lcmx_commit": "c", "harness_commit": "c", "split_label": "smoke",
            "id_list_sha256": "i", "judges": [SOL_JUDGE], "agreement_judge": OPUS,
            "status": "complete", "question_ids": [str(i) for i in range(10)]}))
        rows = []
        for i in range(10):
            verdicts = {SOL_JUDGE: {"status": "ok", "correct": i % 2 == 0}}
            if i < 4:
                verdicts[OPUS] = {"status": "ok", "correct": i % 4 == 0}
            rows.append({"qid": str(i), "qtype": "t", "status": "ok", "question": "Q?", "gold": "g", "hypothesis": "h",
                         "judges": verdicts, "tool_calls": 2, "calls": [{"status": "ok", "input_tokens": 100,
                         "output_tokens": 5, "wrapper_overhead_tokens": 90, "weekly_used_percent": 25.0 + i, "cost": 0}],
                         "judge_calls": [{"status": "ok", "input_tokens": 50, "output_tokens": 1}]})
        (run_dir / "judged.jsonl").write_text("\n".join(map(json.dumps, rows)))
    report = score.score(arms, tmp_path / "r")
    arm = report["arms"]["off/single/" + SOL]
    assert arm["accuracy"] == {SOL_JUDGE: .5} and arm["primary"] == .5
    assert arm["agreement"]["n"] == 4 and arm["agreement"]["percent"] == 75
    assert arm["agreement_sample"] == {"judge": OPUS, "n": 4, "accuracy": {SOL_JUDGE: .5, OPUS: .25}}
    assert arm["reader_tokens_per_question"]["wrapper_overhead_tokens"] == 90
    assert arm["quota"]["codex_weekly_used_percent"] == [25.0, 34.0]
    assert report["paired"]["off/single/{0} vs off/agentic/{0}".format(SOL)]["both_agree"] is None
    assert "full_run_estimate" not in report and report["tokens_500_per_arm"]["off/single/" + SOL]["input_tokens"] == 75_000
    assert SOL_JUDGE in (tmp_path / "r/report.md").read_text()


def test_canary_fails_on_leak_violation_or_missing_mcp_and_cleans_up(tmp_path, question):
    import re

    class Harnessed:
        def __init__(self, leak=False, violations=(), mcp_calls=1, refuse=False):
            self.leak, self.violations, self.mcp_calls, self.refuse = leak, list(violations), mcp_calls, refuse

        def _run(self, messages, records, mcp):
            transcript = tmp_path / f"t{len(list(tmp_path.glob('t*')))}.jsonl"
            paths = re.findall(r"(/\S+?\.txt)", messages[0]["content"])
            transcript.write_text(Path(paths[0]).read_text() if self.leak else "{}")
            records.append({"transcripts": [str(transcript)], "violations": self.violations,
                            "mcp_calls": self.mcp_calls if mcp else 0, "input_tokens": 1,
                            "terminal_reason": "malformed_tool_use_exhausted" if self.refuse else None})
            if self.refuse and not mcp:
                raise RuntimeError("claude_call_failed")
            return {"content": "x"}

        def call(self, spec, messages, records, **_):
            return self._run(messages, records, False)

        def agentic(self, spec, messages, records, server):
            return self._run(messages, records, True)
    data = tmp_path / "data"
    data.mkdir()
    args = SimpleNamespace(output=tmp_path / "out", data=data / "file.json", reader=OPUS, lcmx_repo=REPO,
                           store_root=tmp_path, config="off", summaries="none", embed_cache=None)
    args.output.mkdir()
    with patch.object(run, "versions", return_value={}):
        run.canary(args, Harnessed(refuse=True), "sha", question)
        assert [r["status"] for r in json.loads((args.output / "canary.json").read_text())["results"]] == \
            ["no_tool_to_call", "ok"]
        for client in (Harnessed(leak=True), Harnessed(violations=["Bash"]), Harnessed(mcp_calls=0)):
            with pytest.raises(RuntimeError, match="canary_failed"):
                run.canary(args, client, "sha", question)
    assert not list(data.iterdir()) and not list(args.output.glob("fmp-canary-*"))
    assert "REF-" not in (args.output / "canary.json").read_text()
