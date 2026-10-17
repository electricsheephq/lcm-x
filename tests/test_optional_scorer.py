import json
import threading
from dataclasses import replace

import pytest

from optional_scorer import (
    BGE_REVISION, Candidate, CandidateScore, JEV_LEVELS, JEV_MODEL, JEV_REQUEST_RESERVE,
    JevBudget, LocalBGEScorer, PublicJevScorer, score_candidates,
)
from benchmarking.portable_recall_eval import (
    FrozenCase, GoldSpan, evaluate, ndcg, read_frozen, synthetic_cases,
    SpanCaptureUnavailable, freeze_production_pool, validate_cases, write_frozen,
)


def candidate(identity, text="fixture", scope="scope"):
    return Candidate(identity, text, scope, f"{scope}:{identity}", 0, len(text))


def test_baseline_identity_stable_ties_and_offsets():
    pool = (candidate("a"), candidate("b"))
    baseline = score_candidates("query", pool)
    assert baseline.candidates is pool
    result = score_candidates("query", pool, lambda *args: [CandidateScore("b", .5), CandidateScore("a", .5)])
    assert result.status == "scored" and result.candidates == pool
    assert all(a is b for a, b in zip(pool, result.candidates))


@pytest.mark.parametrize("entries", [
    [], [CandidateScore("unknown", .2)],
    [CandidateScore("a", .1), CandidateScore("a", .3)],
    [CandidateScore("a", float("nan"))], [CandidateScore("a", float("inf"))],
    [CandidateScore("a", 1.1)], [CandidateScore("a", True)],
])
def test_invalid_scorer_output_falls_back_identically(entries):
    pool = (candidate("a"),)
    result = score_candidates("q", pool, lambda *args: entries)
    assert result.status == "invalid_output" and result.candidates is pool


@pytest.mark.parametrize("scorer,status", [
    (lambda *args: None, "abstention"),
    (lambda *args: (_ for _ in ()).throw(RuntimeError("unavailable")), "unavailable"),
])
def test_optional_failure_falls_back(scorer, status):
    pool = (candidate("a"),)
    result = score_candidates("q", pool, scorer)
    assert result.status == status and result.candidates is pool


def test_timeout_falls_back_and_worker_can_finish():
    started, release = threading.Event(), threading.Event()
    pool = (candidate("a"),)

    def blocked(*args):
        started.set()
        release.wait(1)
        return [CandidateScore("a", 1)]

    try:
        result = score_candidates("q", pool, blocked, .02)
        assert started.is_set() and result.status == "timeout"
        assert result.candidates is pool
    finally:
        release.set()


@pytest.mark.parametrize("pool", [
    (candidate("a"), candidate("a")),
    (candidate("a"), candidate("b", scope="other")),
    tuple(candidate(str(i)) for i in range(31)),
    (replace(candidate("a"), char_end=99),),
])
def test_invalid_input_never_dispatches(pool):
    def forbidden(*args):
        pytest.fail("invalid pool dispatched")
    with pytest.raises(ValueError):
        score_candidates("q", pool, forbidden)


def test_local_model_pin_adapter_without_model_download():
    class Model:
        def predict(self, pairs, **kwargs):
            assert pairs == [("q", "fixture"), ("q", "fixture")]
            assert kwargs["batch_size"] == 8
            return [-2., 2.]
    pool = (candidate("a"), candidate("b"))
    result = score_candidates("q", pool, LocalBGEScorer(model=Model()))
    assert result.status == "scored" and result.candidates[0] is pool[1]



def test_local_model_constructor_uses_registered_identity_module(monkeypatch):
    import sys
    from types import SimpleNamespace

    class Identity:
        pass

    observed = {}

    def cross_encoder(name, **kwargs):
        observed.update(name=name, **kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=cross_encoder))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(nn=SimpleNamespace(Identity=Identity)))
    LocalBGEScorer()
    assert isinstance(observed["activation_fn"], Identity)
    assert observed["revision"] == BGE_REVISION
    assert observed["local_files_only"] is True
    assert observed["trust_remote_code"] is False


def jev_reply(payload, confidence=.9):
    request = json.loads(payload)
    assert set(request["state"]) == {"query", "candidates"}
    assert all(set(c) == {"text"} for c in request["state"]["candidates"])
    return {"model": JEV_MODEL, "usage": {"input_tokens": 100},
            "answers": {key: {"type": "score", "probabilities": {"0": .1, "1": .2, "2": .7},
                              "legend": dict(enumerate(JEV_LEVELS)), "score": 1.6,
                              "confidence": confidence}
                        for key in request["questions"]}}


def transport(payload, timeout):
    response = jev_reply(payload)
    # JSON object keys are strings in actual HTTP response.
    return json.loads(json.dumps(response))


def test_jev_batch_explicit_permission_and_metadata_only_ledger(tmp_path):
    budget = JevBudget(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError):
        PublicJevScorer(input_kind="private", allow_public_egress=True, budget=budget)
    scorer = PublicJevScorer(input_kind="synthetic", allow_public_egress=True,
                            budget=budget, transport=transport)
    result = score_candidates("q", (candidate("a"), candidate("b")), scorer)
    assert result.status == "scored" and all(s.score == pytest.approx(.8) for s in result.scores)
    assert budget.charged_tokens == 100
    records = [json.loads(line) for line in budget.journal.read_text().splitlines()]
    assert len(records) == 2 and all(set(r) <= {"model", "request", "state", "input_tokens"} for r in records)
    assert JevBudget(budget.journal).charged_tokens == 100
    with pytest.raises(ValueError):
        budget.complete(1, 0)


def test_jev_uncertain_send_keeps_reservation_and_blocks_over_budget(tmp_path):
    budget = JevBudget(tmp_path / "ledger.jsonl", max_input_tokens=JEV_REQUEST_RESERVE)
    def failure(*args):
        raise TimeoutError("fixture timeout")
    scorer = PublicJevScorer(input_kind="synthetic", allow_public_egress=True,
                            budget=budget, transport=failure)
    result = score_candidates("q", (candidate("a"),), scorer)
    assert result.status == "unavailable"
    assert JevBudget(budget.journal).charged_tokens == JEV_REQUEST_RESERVE
    with pytest.raises(RuntimeError):
        budget.reserve()



@pytest.mark.parametrize("probabilities, score, expected_status", [
    ({"0": 0.0, "1": .01, "2": .99}, 1.98, "scored"),
    ({"0": .05, "1": .05, "2": .9}, .2, "unavailable"),
])
def test_jev_rounded_wire_preserves_bounded_numeric_validation(tmp_path, probabilities, score, expected_status):
    def rounded(payload, timeout):
        reply = transport(payload, timeout)
        for answer in reply["answers"].values():
            answer.update(probabilities=probabilities, score=score)
        return reply

    scorer = PublicJevScorer(input_kind="synthetic", allow_public_egress=True,
                            budget=JevBudget(tmp_path / "rounded.jsonl"), transport=rounded)
    result = score_candidates("fixture", (candidate("a"),), scorer)
    assert result.status == expected_status
    if expected_status == "scored":
        assert result.scores[0].score == pytest.approx(.995)


def test_jev_confidence_abstention_uses_baseline(tmp_path):
    def low(payload, timeout):
        return json.loads(json.dumps(jev_reply(payload, confidence=.1)))
    scorer = PublicJevScorer(input_kind="synthetic", allow_public_egress=True,
                            budget=JevBudget(tmp_path / "ledger.jsonl"), transport=low,
                            min_confidence=.5)
    pool = (candidate("a"),)
    result = score_candidates("q", pool, scorer)
    assert result.status == "abstention" and result.candidates is pool


def frozen_case(index, *, gold=True):
    pool = tuple(candidate(f"d{index}-{n}") for n in range(5)) + (candidate(f"g{index}", "support"),)
    spans = (GoldSpan(pool[-1].source_ref, 0, 7, "support"),) if gold else None
    return FrozenCase(str(index), str(index), "holdout", "exact-id", "q", pool,
                      (pool[-1].candidate_id,), spans)


def test_synthetic_split_and_frozen_digest(tmp_path):
    fixtures = synthetic_cases()
    assert len(fixtures) == 120 and sum(f["split"] == "dev" for f in fixtures) == 40
    assert len({f["conversation_id"] for f in fixtures}) == 120
    cases = (frozen_case(1), frozen_case(2))
    path = tmp_path / "frozen.jsonl"
    write_frozen(path, cases, input_kind="synthetic", posture="OFF", source_identity="fixture-sha")
    header, restored = read_frozen(path)
    assert restored == cases and header["production_hit_cap"] == 25
    with pytest.raises(FileExistsError):
        write_frozen(path, cases, input_kind="synthetic", posture="LOCAL", source_identity="fixture")
    path.write_text(path.read_text().replace("support", "changed"))
    with pytest.raises(ValueError, match="digest"):
        read_frozen(path)
    with pytest.raises(ValueError, match="crosses"):
        validate_cases((cases[0], replace(cases[1], conversation_id="1", split="dev")))


def test_promotion_exact_coverage_and_missing_annotation_boundary():
    def perfect(query, candidates, timeout):
        return [CandidateScore(c.candidate_id, float(c.text == "support")) for c in candidates]
    cases = tuple(frozen_case(i) for i in range(4))
    report = evaluate(cases, {"fixture-model": perfect})
    assert report["arms"]["current"]["candidate_recall_at_30"] == 1
    assert report["arms"]["current"]["span_recall_at_5"] == 0
    assert report["arms"]["lexical"]["span_recall_at_5"] == 1
    assert report["arms"]["fixture-model"]["promotion"] == "PROMOTE"
    unlabelled = evaluate((frozen_case(1, gold=False),), {"fixture-model": perfect})
    assert unlabelled["arms"]["fixture-model"]["promotion"] == "INSUFFICIENT_EVIDENCE"
    assert unlabelled["answer_correctness"] == "UNMEASURED"
    repeated = (cases[0].candidates[-1], cases[0].candidates[-1])
    assert ndcg(repeated, cases[0].gold) == 1


def test_answer_ready_capture_requires_published_exact_span(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from benchmarking import longmemeval as instrument
    monkeypatch.setattr(instrument, "fts_hits", lambda *args: [])
    source = "xxsupportyy"
    store = SimpleNamespace(get=lambda identity: {"content": source})
    question = SimpleNamespace(question_id="fixture", category="exact-id",
                               question="q", is_abstention=False)
    hit = {"kind": "message_excerpt", "store_id": 1, "snippet": "[support]",
           "content": "support", "content_offset": 2, "content_returned_chars": 7}

    def capture(value):
        return freeze_production_pool(
            question, None, store, None, None, tmp_dir=tmp_path,
            embeddings_enabled=False, provider_name="stub", scope_id="fixture",
            split="holdout", hits=[value], detail="answer_ready", include="verbatim",
        )

    case = capture(hit)
    assert case.candidates[0].text == source[2:9]
    assert case.candidates[0].char_start == 2 and case.candidates[0].char_end == 9
    assert case.delivery_detail == "answer_ready" and case.delivery_include == "verbatim"
    assert evaluate((case,), {})["recall_at_5_gain_possible_by_permutation"] is False
    for bad in ({**hit, "content_offset": None}, {**hit, "content_returned_chars": 8},
                {**hit, "content_offset": 0}):
        with pytest.raises(SpanCaptureUnavailable):
            capture(bad)
