"""Tests for the LongMemEval retrieval harness.

Covers the evidence-matching scorer, the metric math (recall@k / NDCG@10 /
percentiles), CLI argument validation, and an end-to-end offline stub run that
proves the ingest -> retrieve -> score plumbing over a real temp LCM store.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from benchmarking.longmemeval import (
    ARMS,
    DATASET_REVISION,
    ProviderAccounting,
    RERANK_MODE_MIXED,
    RERANK_MODE_PLACEHOLDER,
    RERANK_MODE_VOYAGE,
    Question,
    chunk_sessions,
    deterministic_session_summary,
    embedding_determinism_report,
    evaluate_question,
    evidence_sessions,
    evidence_turns,
    fresh_recall_session_id,
    load_questions,
    ndcg_at_k,
    parse_question,
    percentiles,
    production_recall_hits,
    recall_at_k,
    recall_hit_sessions,
    recall_hit_turn_keys,
    rerank_sessions_voyage,
    resolve_harness_providers,
    rrf_fuse,
    run_harness,
    summary_turn_keys,
    turn_ndcg_at_k,
    turn_recall_at_k,
)
from benchmarking.longmemeval import _typed_provider_degraded_outcomes

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lcm_longmemeval.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("lcm_longmemeval_cli", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_raw(
    question_id: str,
    question_type: str,
    *,
    sessions: dict[str, list[dict]],
    answer_session_ids: list[str],
    question: str = "what did we decide",
) -> dict:
    session_ids = list(sessions)
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": question,
        "answer": "irrelevant",
        "question_date": "2023-01-01",
        "haystack_session_ids": session_ids,
        "haystack_dates": ["2023-01-01"] * len(session_ids),
        "haystack_sessions": [sessions[sid] for sid in session_ids],
        "answer_session_ids": answer_session_ids,
    }


# --------------------------------------------------------------------------- #
# Evidence-matching scorer.
# --------------------------------------------------------------------------- #


def test_evidence_sessions_uses_answer_session_ids():
    raw = _make_raw(
        "q1",
        "multi-session",
        sessions={
            "s1": [{"role": "user", "content": "hi"}],
            "s2": [{"role": "user", "content": "budget is 500", "has_answer": True}],
        },
        answer_session_ids=["s2"],
    )
    question = parse_question(raw)
    assert evidence_sessions(question) == {"s2"}


def test_abstention_questions_have_no_evidence_and_are_flagged():
    raw = _make_raw(
        "q9_abs",
        "single-session-user",
        sessions={"s1": [{"role": "user", "content": "hi"}]},
        answer_session_ids=[],
    )
    question = parse_question(raw)
    assert question.is_abstention is True
    assert evidence_sessions(question) == set()
    assert evidence_turns(question) == set()


def test_evidence_turns_reads_has_answer_markers():
    raw = _make_raw(
        "q2",
        "single-session-assistant",
        sessions={
            "s1": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "the code is X7", "has_answer": True},
            ],
        },
        answer_session_ids=["s1"],
    )
    question = parse_question(raw)
    assert evidence_turns(question) == {("s1", 1)}


def test_category_maps_temporal_reasoning_label():
    raw = _make_raw(
        "q3",
        "temporal-reasoning",
        sessions={"s1": [{"role": "user", "content": "a"}]},
        answer_session_ids=["s1"],
    )
    assert parse_question(raw).category == "temporal"


# --------------------------------------------------------------------------- #
# Metric math.
# --------------------------------------------------------------------------- #


def test_recall_at_k_counts_relevant_within_top_k():
    retrieved = ["s3", "s1", "s2", "s9"]
    assert recall_at_k(retrieved, {"s1", "s2"}, 1) == 0.0
    assert recall_at_k(retrieved, {"s1", "s2"}, 2) == pytest.approx(0.5)
    assert recall_at_k(retrieved, {"s1", "s2"}, 3) == pytest.approx(1.0)


def test_recall_at_k_dedups_and_handles_empty_relevant():
    assert recall_at_k(["s1", "s1", "s1"], set(), 5) == 0.0
    assert recall_at_k(["s1", "s1", "s2"], {"s2"}, 2) == pytest.approx(1.0)


def test_ndcg_at_k_perfect_and_ranked():
    # Single relevant item at rank 1 -> perfect NDCG.
    assert ndcg_at_k(["s1", "s2"], {"s1"}, 10) == pytest.approx(1.0)
    # Relevant item at rank 2: DCG = 1/log2(3); IDCG (1 relevant) = 1/log2(2)=1.
    import math

    assert ndcg_at_k(["s2", "s1"], {"s1"}, 10) == pytest.approx(1.0 / math.log2(3))
    assert ndcg_at_k(["s2", "s3"], {"s1"}, 10) == 0.0


def test_percentiles_nearest_rank():
    values = [10.0, 20.0, 30.0, 40.0, 100.0]
    result = percentiles(values, points=(50, 90, 99))
    assert result["p50"] == 30.0
    assert result["p90"] == 100.0
    assert result["p99"] == 100.0
    assert percentiles([], points=(50,)) == {"p50": 0.0}


def test_rrf_fuse_rewards_agreement_across_arms():
    fts = ["s1", "s2", "s3"]
    vectors = ["s3", "s1", "s9"]
    fused = rrf_fuse(fts, vectors)
    # s1 (ranks 1 and 2) and s3 (ranks 3 and 1) outrank single-arm-only items.
    assert set(fused[:2]) == {"s1", "s3"}
    assert set(fused) == {"s1", "s2", "s3", "s9"}


def test_turn_recall_precise_keys():
    # Two labeled evidence turns; a ranked list of precise (session, turn) keys.
    evidence = {("s1", 1), ("s2", 0)}
    ranked = [("s1", 0), ("s1", 1), ("s3", 2), ("s2", 0)]
    assert turn_recall_at_k(ranked, evidence, 1) == 0.0
    assert turn_recall_at_k(ranked, evidence, 2) == pytest.approx(0.5)
    assert turn_recall_at_k(ranked, evidence, 4) == pytest.approx(1.0)


def test_turn_recall_summary_marker_covers_session_at_granularity():
    # A (session, None) summary marker covers ALL evidence turns of its session in
    # one item — the session-granularity credit an asterisk warns about.
    evidence = {("s1", 3), ("s1", 7), ("s2", 0)}
    # One summary marker for s1 at rank 1 recovers both s1 evidence turns.
    assert turn_recall_at_k([("s1", None)], evidence, 1) == pytest.approx(2 / 3)
    # A marker for a session with no evidence contributes nothing.
    assert turn_recall_at_k([("s9", None)], evidence, 1) == 0.0
    assert turn_recall_at_k([], evidence, 5) == 0.0
    assert turn_recall_at_k([("s1", 3)], set(), 5) == 0.0


def test_hybrid_turn_keys_project_from_fused_ranking_not_raw_key_fusion():
    """C6: a hybrid arm's turn keys are session-granularity markers derived from its
    fused SESSION ranking, NOT an RRF over the raw per-arm turn-key lists.

    Regression for the B5-measured turn-precision collapse: fusing precise (fts /
    chunk) and coarse (summary) turn keys in one ranked list let a flood of precise
    NON-evidence keys consume the fixed top-k coverage budget ahead of the summary
    markers of the high-ranked evidence session, dragging turn recall below every
    input arm. Projecting the (strong) fused session ranking to (session, None)
    markers restores full session-granularity coverage.
    """
    # Evidence lives entirely in session "sE" (3 labeled turns).
    evidence = {("sE", 0), ("sE", 1), ("sE", 2)}
    # The fused SESSION ranking puts the evidence session first (its strong signal).
    fused_ranking = ["sE", "sA", "sB", "sC", "sD", "sF"]
    # The precise arms AGREE on five NON-evidence turns, so each of those keys earns
    # two RRF terms and outscores the evidence session's single-arm summary marker.
    noise = [("sA", 0), ("sB", 0), ("sC", 0), ("sD", 0), ("sF", 0)]
    fts_turns = list(noise)
    chunk_turns = list(noise)
    # In the summary arm the evidence session ranks LAST, so its marker lands at
    # rank 6 — pushed out of the top-5 budget by the agreed-upon noise.
    summary_turns = summary_turn_keys(["sA", "sB", "sC", "sD", "sF", "sE"])

    # Old behavior: raw-key RRF buries ("sE", None) below five non-evidence keys.
    diluted = rrf_fuse(fts_turns, summary_turns, chunk_turns)
    # New behavior: project the fused session ranking to session-granularity markers.
    projected = summary_turn_keys(fused_ranking)

    assert all(key[1] is None for key in projected)
    # The evidence session's marker sits at rank 1 and covers all its turns.
    assert turn_recall_at_k(projected, evidence, 5) == pytest.approx(1.0)
    # The diluted fusion recovers nothing in the top-5 (the collapse being fixed).
    assert turn_recall_at_k(diluted, evidence, 5) == pytest.approx(0.0)


def test_turn_ndcg_rewards_ranking_and_credits_summary_markers():
    evidence = {("s1", 2)}
    # Precise relevant turn at rank 1 -> perfect NDCG.
    assert turn_ndcg_at_k([("s1", 2), ("s4", 0)], evidence, 10) == pytest.approx(1.0)
    # Summary marker for the evidence session counts as relevant at session grain.
    assert turn_ndcg_at_k([("s1", None)], evidence, 10) == pytest.approx(1.0)
    # Irrelevant-only ranking scores zero.
    assert turn_ndcg_at_k([("s7", 0), ("s8", 1)], evidence, 10) == 0.0


def test_deterministic_summary_is_stable_and_content_bearing():
    turns = [{"role": "user", "content": "the vault code is 4417"}]
    first = deterministic_session_summary(turns)
    second = deterministic_session_summary(turns)
    assert first == second
    assert "4417" in first


# --------------------------------------------------------------------------- #
# CLI argument validation.
# --------------------------------------------------------------------------- #


def test_cli_run_requires_model_for_non_stub_provider():
    cli = _load_cli()
    args = cli._parse_args(
        ["run", "--dataset", "x.json", "--output", "out", "--provider", "fastembed"]
    )
    with pytest.raises(SystemExit):
        cli._cmd_run(args)


def test_cli_run_rejects_nonpositive_limit(tmp_path):
    cli = _load_cli()
    dataset = tmp_path / "d.json"
    dataset.write_text("[]", encoding="utf-8")
    args = cli._parse_args(
        ["run", "--dataset", str(dataset), "--output", str(tmp_path / "o"), "--limit", "0"]
    )
    with pytest.raises(SystemExit):
        cli._cmd_run(args)


def test_cli_run_rejects_missing_dataset(tmp_path):
    cli = _load_cli()
    args = cli._parse_args(
        ["run", "--dataset", str(tmp_path / "missing.json"), "--output", str(tmp_path / "o")]
    )
    with pytest.raises(SystemExit):
        cli._cmd_run(args)


def test_cli_rejects_unknown_provider():
    cli = _load_cli()
    with pytest.raises(SystemExit):
        cli._parse_args(["run", "--dataset", "x", "--output", "o", "--provider", "nope"])


def test_cli_parses_dump_candidates_path():
    cli = _load_cli()
    args = cli._parse_args(
        [
            "run",
            "--dataset",
            "x.json",
            "--output",
            "out",
            "--dump-candidates",
            "evidence/candidates.jsonl",
        ]
    )
    assert args.dump_candidates == "evidence/candidates.jsonl"


def test_cli_parses_recall_rerank_flag():
    cli = _load_cli()
    args = cli._parse_args(
        [
            "run",
            "--dataset",
            "x.json",
            "--output",
            "out",
            "--recall-rerank",
        ]
    )
    assert args.recall_rerank is True
    assert args.recall_rerank_window == 0
    assert args.recall_rerank_margin == 0.0


def test_cli_parses_recall_rerank_margin():
    cli = _load_cli()
    args = cli._parse_args(
        [
            "run",
            "--dataset",
            "x.json",
            "--output",
            "out",
            "--recall-rerank",
            "--recall-rerank-margin",
            "0.25",
        ]
    )
    assert args.recall_rerank_margin == 0.25


def test_cli_recall_rerank_margin_requires_recall_rerank(tmp_path):
    cli = _load_cli()
    args = cli._parse_args(
        [
            "run",
            "--dataset",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "out"),
            "--recall-rerank-margin",
            "0",
        ]
    )
    with pytest.raises(SystemExit, match="requires --recall-rerank"):
        cli._cmd_run(args)


def test_cli_recall_rerank_window_requires_recall_rerank(tmp_path):
    cli = _load_cli()
    args = cli._parse_args(
        [
            "run",
            "--dataset",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "out"),
            "--recall-rerank-window",
            "10",
        ]
    )
    assert args.recall_rerank_window == 10
    with pytest.raises(SystemExit, match="requires --recall-rerank"):
        cli._cmd_run(args)


def test_cli_recall_rerank_window_zero_is_still_bound_to_recall_rerank(tmp_path):
    cli = _load_cli()
    args = cli._parse_args(
        [
            "run",
            "--dataset",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "out"),
            "--recall-rerank-window",
            "0",
        ]
    )
    with pytest.raises(SystemExit, match="requires --recall-rerank"):
        cli._cmd_run(args)


def test_load_questions_limit_validation(tmp_path):
    dataset = tmp_path / "d.json"
    dataset.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_questions(dataset, limit=-1)


# --------------------------------------------------------------------------- #
# End-to-end offline stub run (plumbing proof).
# --------------------------------------------------------------------------- #


def _synthetic_dataset() -> list[Question]:
    questions: list[Question] = []
    for index in range(3):
        evidence_id = f"q{index}-s-evidence"
        sessions = {
            f"q{index}-s-noise": [
                {"role": "user", "content": "unrelated small talk about the weather"},
                {"role": "assistant", "content": "yes it is sunny today"},
            ],
            evidence_id: [
                {"role": "user", "content": f"remember my locker passcode is ZEBRA{index}"},
                {
                    "role": "assistant",
                    "content": f"noted, locker passcode ZEBRA{index}",
                    "has_answer": True,
                },
            ],
        }
        questions.append(
            parse_question(
                _make_raw(
                    f"q{index}",
                    "single-session-user",
                    sessions=sessions,
                    answer_session_ids=[evidence_id],
                    question=f"what is my locker passcode ZEBRA{index}",
                )
            )
        )
    return questions


class _FakeChunkStore:
    """Minimal stand-in exposing only ``knn_chunks`` for the chunk arm."""

    def __init__(self, hits):
        self._hits = hits

    def knn_chunks(self, query_vec, k, model, provider):
        return list(self._hits)


def test_chunk_sessions_maps_hits_to_sessions_and_dedups():
    # Chunk ids are ``store_id:chunk_index``; each votes for its owning session,
    # first-seen order wins, and an unmapped store_id is dropped.
    hits = [("10:0", 0.9, "chunk"), ("11:2", 0.8, "chunk"),
            ("10:1", 0.7, "chunk"), ("99:0", 0.6, "chunk")]
    store_id_to_session = {10: "sess-a", 11: "sess-b"}
    ranked = chunk_sessions(
        _FakeChunkStore(hits), [1.0, 0.0], "model", "provider", 10, store_id_to_session
    )
    assert ranked == ["sess-a", "sess-b"]


class _KeyedEmbedder:
    """Deterministic embedder: text mentioning the passcode maps to one axis.

    This makes the chunk KNN arm resolvable offline — the evidence chunk and the
    query share the ``passcode`` axis, so the evidence session ranks first.
    """

    model_id = "keyed"
    dim = 2

    def _vec(self, text: str) -> list[float]:
        return [1.0, 0.0] if "passcode" in text.lower() else [0.0, 1.0]

    def embed_documents(self, texts):
        return [self._vec(str(text)) for text in texts]

    def embed_query(self, text):
        return self._vec(str(text))


def test_evaluate_question_chunk_arm_recovers_evidence(tmp_path):
    evidence_id = "s-evidence"
    long_evidence = (
        "please remember for later reference that my personal locker passcode "
        "phrase is the northern lighthouse keeper, and that this detail matters "
        "quite a lot to me because I keep forgetting it every single time I try "
        "to open the locker at the gym after my afternoon workout session"
    )
    sessions = {
        "s-noise": [
            {"role": "user", "content": "unrelated small talk about the sunny weather today"},
            {"role": "assistant", "content": "yes it certainly is a pleasant afternoon outside"},
        ],
        evidence_id: [
            # The passcode cue and the has_answer marker are the same (user) turn,
            # so the chunk arm's retrieved turn matches the labeled evidence turn.
            {"role": "user", "content": long_evidence, "has_answer": True},
            {"role": "assistant", "content": "noted, I will keep that safe"},
        ],
    }
    question = parse_question(
        _make_raw(
            "q-chunk", "single-session-user", sessions=sessions,
            answer_session_ids=[evidence_id],
            question="what is my locker passcode phrase",
        )
    )

    scored = evaluate_question(
        question, _KeyedEmbedder(), provider_name="stub",
        tmp_dir=tmp_path, embeddings_enabled=True,
    )

    assert "chunk_vectors" in scored and "hybrid_rrf3" in scored
    # The chunk arm recovers the evidence session via the shared passcode axis.
    assert scored["chunk_vectors"]["recall@1"] == pytest.approx(1.0)
    # The three-arm fusion keeps the evidence session in its top-k.
    assert scored["hybrid_rrf3"]["recall@10"] == pytest.approx(1.0)
    for arm in ARMS:
        assert scored[arm]["latency_ms"] >= 0.0
        # Every arm now reports a turn-level block alongside the session metrics.
        turn = scored[arm]["turn"]
        assert set(turn) >= {"recall@1", "recall@5", "recall@10", "ndcg@10", "session_granularity"}
    # The chunk arm localizes to the exact evidence turn (store_id -> turn), so its
    # turn-level recall is exact, not session-granularity.
    assert scored["chunk_vectors"]["turn"]["recall@1"] == pytest.approx(1.0)
    assert scored["chunk_vectors"]["turn"]["session_granularity"] is False
    # Summary-based arms carry the session-granularity asterisk.
    assert scored["summary_vectors"]["turn"]["session_granularity"] is True
    assert scored["hybrid_rerank"]["rerank_mode"] == RERANK_MODE_PLACEHOLDER


class _FakeReranker:
    """Records the rerank call and returns a fixed reordering, or raises."""

    def __init__(self, order=None, raise_error=False):
        self._order = order
        self._raise = raise_error
        self.calls = 0
        self.payloads = []

    def rerank(self, query, documents, *, top_k=None, timeout):
        self.calls += 1
        self.payloads.append((query, list(documents)))
        if self._raise:
            raise RuntimeError("provider down")
        order = self._order if self._order is not None else list(range(len(documents)))
        return [(index, 1.0 - position * 0.1) for position, index in enumerate(order)]


def test_rerank_sessions_voyage_reorders_window_and_appends_tail():
    reranker = _FakeReranker(order=[2, 0, 1])
    sessions = ["a", "b", "c", "d", "e"]
    summaries = {s: f"summary {s}" for s in sessions}
    out = rerank_sessions_voyage(reranker, "q", sessions, summaries, window=3)
    assert reranker.calls == 1
    # Window [a,b,c] reordered to [c,a,b]; tail [d,e] preserved.
    assert out == ["c", "a", "b", "d", "e"]


def test_rerank_sessions_voyage_keeps_raw_production_parity_payloads():
    reranker = _FakeReranker()
    query = "password=raw-query-secret"
    summaries = {"a": "password=raw-candidate-secret"}

    assert rerank_sessions_voyage(reranker, query, ["a"], summaries) == ["a"]
    assert reranker.payloads == [(query, [summaries["a"]])]


def test_rerank_sessions_voyage_signals_fallback_on_error_and_empty():
    reranker = _FakeReranker(raise_error=True)
    assert rerank_sessions_voyage(reranker, "q", ["a", "b"], {"a": "x", "b": "y"}) is None
    assert rerank_sessions_voyage(reranker, "q", [], {}) is None


def test_rerank_sessions_voyage_empty_response_signals_fallback():
    """FIX-3: a non-exception empty response (e.g. ``data: []``) is degenerate --
    it must signal the placeholder fallback, not be accepted as a real rerank."""
    reranker = _FakeReranker(order=[])  # provider returns [] without raising
    out = rerank_sessions_voyage(reranker, "q", ["a", "b", "c"], {"a": "x", "b": "y", "c": "z"})
    assert out is None
    assert reranker.calls == 1  # the provider WAS called; its response was degenerate


def test_rerank_sessions_voyage_partial_coverage_signals_fallback():
    """FIX-3: a response scoring only some candidates does not cover the input
    set, so it is degenerate and must fall back rather than count as real."""
    reranker = _FakeReranker(order=[0])  # only 1 of 3 candidates scored
    out = rerank_sessions_voyage(reranker, "q", ["a", "b", "c"], {"a": "x", "b": "y", "c": "z"})
    assert out is None


class _RerankingEmbedder(_KeyedEmbedder):
    """A voyage-shaped embedder that also exposes a fake ``rerank``."""

    def rerank(self, query, documents, *, top_k=None, timeout):
        # Identity order is enough: we only assert the mode label, not the ordering.
        return [(index, 1.0) for index in range(len(documents))]


def test_evaluate_question_real_rerank_path_labels_voyage_mode(tmp_path):
    # use_rerank + provider voyage + a reranker-bearing embedder takes the real
    # cross-encoder path and labels it, instead of the placeholder.
    evidence_id = "s-evidence"
    sessions = {
        "s-noise": [{"role": "user", "content": "chatter about the weather"}],
        evidence_id: [
            {"role": "user", "content": "my passcode phrase", "has_answer": True},
        ],
    }
    question = parse_question(
        _make_raw(
            "q-rr", "single-session-user", sessions=sessions,
            answer_session_ids=[evidence_id], question="what is my passcode phrase",
        )
    )
    scored = evaluate_question(
        question, _RerankingEmbedder(), provider_name="voyage",
        tmp_dir=tmp_path, embeddings_enabled=True, use_rerank=True,
    )
    assert scored["hybrid_rerank"]["rerank_mode"] == RERANK_MODE_VOYAGE


def test_run_harness_mixed_rerank_reports_mixed_not_real(tmp_path, monkeypatch):
    """FIX-2: when some questions use the real reranker and others silently fall
    back, the run-level mode is ``mixed`` (with counts), never mislabeled ``real``
    from whatever the final question happened to use."""
    import benchmarking.longmemeval as lme

    monkeypatch.setattr(lme, "resolve_harness_provider", lambda *a, **k: _RerankingEmbedder())
    calls = {"n": 0}

    def _fake_rerank(reranker, query, sessions, summaries, **kwargs):
        calls["n"] += 1
        # First scored question gets a real rerank; the rest fall back silently.
        return list(sessions) if calls["n"] == 1 else None

    monkeypatch.setattr(lme, "rerank_sessions_voyage", _fake_rerank)

    report = run_harness(
        _synthetic_dataset(), provider_name="voyage", model="voyage-3",
        tmp_dir=tmp_path, use_rerank=True,
    )
    assert report["rerank"]["mode"] == RERANK_MODE_MIXED
    assert report["rerank"]["real_count"] == 1
    assert report["rerank"]["placeholder_count"] == 2
    assert report["rerank"]["counts"][RERANK_MODE_VOYAGE] == 1


def test_run_harness_all_real_rerank_reports_voyage(tmp_path, monkeypatch):
    """FIX-2: a run where every question used the real reranker is labeled real."""
    import benchmarking.longmemeval as lme

    monkeypatch.setattr(lme, "resolve_harness_provider", lambda *a, **k: _RerankingEmbedder())
    monkeypatch.setattr(
        lme, "rerank_sessions_voyage",
        lambda reranker, query, sessions, summaries, **kwargs: list(sessions),
    )
    report = run_harness(
        _synthetic_dataset(), provider_name="voyage", model="voyage-3",
        tmp_dir=tmp_path, use_rerank=True,
    )
    assert report["rerank"]["mode"] == RERANK_MODE_VOYAGE
    assert report["rerank"]["real_count"] == 3
    assert report["rerank"]["placeholder_count"] == 0


def test_stub_run_end_to_end_produces_report_and_fts_recovers_evidence(tmp_path):
    report = run_harness(
        _synthetic_dataset(),
        provider_name="stub",
        model="",
        tmp_dir=tmp_path,
    )
    assert report["scored_count"] == 3
    assert report["dataset"]["revision"] == DATASET_REVISION
    assert report["transcript_contents_included"] is False
    assert set(report["arms"]) == set(ARMS)
    # FTS is lexical and provider-independent: the passcode query must recover
    # its evidence session even under the meaningless stub embedder.
    assert report["arms"]["fts"]["recall@10"] == pytest.approx(1.0)
    for arm in ARMS:
        assert report["arms"][arm]["latency_ms"]["p50"] >= 0.0
        assert "turn" in report["arms"][arm]
    # F7 provenance is recorded and the placeholder rerank is labeled.
    assert report["rerank"]["mode"] == RERANK_MODE_PLACEHOLDER
    assert report["ingest"]["reuse_db_template"] is True
    assert "per_question_ms" in report["ingest"]
    assert report["ingest"]["privacy"] == {
        "documents": 0,
        "changed": 0,
        "blocked": 0,
        "queries": 0,
        "queries_changed": 0,
        "queries_blocked": 0,
    }


def test_dump_candidates_off_leaves_checkpoint_byte_identical(tmp_path, monkeypatch):
    """The sidecar is opt-in and must not alter the checkpoint serialization."""
    import benchmarking.longmemeval as lme

    def _fixed_evaluate(question, _embedder, **kwargs):
        scored = {"ingest_ms": 1.25}
        for arm in ARMS:
            scored[arm] = {
                "recall@1": 0.0,
                "recall@5": 0.5,
                "recall@10": 1.0,
                "ndcg@10": 0.75,
                "latency_ms": 2.5,
                "turn": {
                    "recall@1": 0.0,
                    "recall@5": 0.5,
                    "recall@10": 1.0,
                    "ndcg@10": 0.75,
                    "session_granularity": arm != "fts",
                },
            }
        scored["hybrid_rerank"]["rerank_mode"] = RERANK_MODE_PLACEHOLDER
        if kwargs.get("include_rankings"):
            scored["_candidate_rankings"] = {
                arm: {
                    "sessions": [question.haystack_session_ids[-1]],
                    "turns": [(question.haystack_session_ids[-1], None)],
                }
                for arm in ARMS
            }
        return scored

    monkeypatch.setattr(lme, "evaluate_question", _fixed_evaluate)
    questions = _synthetic_dataset()[:1]
    off_checkpoint = tmp_path / "off" / "per_question_checkpoint.jsonl"
    on_checkpoint = tmp_path / "on" / "per_question_checkpoint.jsonl"
    dump_path = tmp_path / "on" / "candidates.jsonl"
    run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "off" / "tmp",
        embeddings_enabled=False,
        checkpoint_path=off_checkpoint,
    )
    run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "on" / "tmp",
        embeddings_enabled=False,
        checkpoint_path=on_checkpoint,
        dump_candidates_path=dump_path,
    )
    assert off_checkpoint.read_bytes() == on_checkpoint.read_bytes()
    assert dump_path.is_file()
    assert "_candidate_rankings" not in on_checkpoint.read_text(encoding="utf-8")


def test_recall_rerank_off_leaves_checkpoint_byte_identical(tmp_path, monkeypatch):
    """The default and explicit-off flag paths serialize the same checkpoint."""
    import benchmarking.longmemeval as lme

    def _fixed_evaluate(question, _embedder, **kwargs):
        scored = {"ingest_ms": 1.25}
        for arm in ARMS:
            scored[arm] = {
                "recall@1": 0.0,
                "recall@5": 0.5,
                "recall@10": 1.0,
                "ndcg@10": 0.75,
                "latency_ms": 2.5,
                "turn": {
                    "recall@1": 0.0,
                    "recall@5": 0.5,
                    "recall@10": 1.0,
                    "ndcg@10": 0.75,
                    "session_granularity": arm != "fts",
                },
            }
        scored["hybrid_rerank"]["rerank_mode"] = RERANK_MODE_PLACEHOLDER
        if kwargs.get("recall_rerank"):
            scored["lcm_recall"]["recall_rerank_status"] = "skipped: test provider"
        return scored

    monkeypatch.setattr(lme, "evaluate_question", _fixed_evaluate)
    question = _synthetic_dataset()[:1]
    default_checkpoint = tmp_path / "default" / "per_question_checkpoint.jsonl"
    explicit_off_checkpoint = tmp_path / "explicit-off" / "per_question_checkpoint.jsonl"
    default_report = run_harness(
        question,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "default" / "tmp",
        embeddings_enabled=False,
        checkpoint_path=default_checkpoint,
    )
    explicit_off_report = run_harness(
        question,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "explicit-off" / "tmp",
        embeddings_enabled=False,
        recall_rerank=False,
        recall_rerank_window=0,
        checkpoint_path=explicit_off_checkpoint,
    )
    assert default_checkpoint.read_bytes() == explicit_off_checkpoint.read_bytes()
    assert "recall_rerank_modes" not in default_report
    assert "recall_rerank_modes" not in explicit_off_report
    row = json.loads(default_checkpoint.read_text(encoding="utf-8").splitlines()[1])
    assert "recall_rerank_status" not in row["arms"]["lcm_recall"]


def test_recall_rerank_on_records_status_and_mode_counts(tmp_path):
    questions = _synthetic_dataset()
    checkpoint_path = tmp_path / "run" / "per_question_checkpoint.jsonl"
    dump_path = tmp_path / "run" / "candidates.jsonl"
    report = run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "tmp",
        embeddings_enabled=True,
        recall_rerank=True,
        recall_rerank_window=10,
        checkpoint_path=checkpoint_path,
        dump_candidates_path=dump_path,
    )

    expected_status = "skipped: rerank requires the voyage provider"
    assert report["recall_rerank_modes"] == {expected_status: len(questions)}
    rows = checkpoint_path.read_text(encoding="utf-8").splitlines()
    header = json.loads(rows[0])
    assert header["__checkpoint_header__"]["recall_rerank"] is True
    assert header["__checkpoint_header__"]["recall_rerank_window"] == 10
    assert header["__checkpoint_header__"]["recall_rerank_margin"] == 0.0
    dump_header = json.loads(dump_path.read_text(encoding="utf-8").splitlines()[0])
    assert dump_header["__dump_header__"]["recall_rerank_window"] == 10
    assert dump_header["__dump_header__"]["recall_rerank_margin"] == 0.0
    for row in map(json.loads, rows[1:]):
        assert row["arms"]["lcm_recall"]["recall_rerank_status"] == expected_status
        assert "rerank_scores" not in row["arms"]["lcm_recall"]
    dump_rows = list(map(json.loads, dump_path.read_text(encoding="utf-8").splitlines()[1:]))
    for row in dump_rows:
        assert row["arms"]["lcm_recall"]["rerank_scores"] == []


def test_recall_rerank_resume_rejects_flag_off_checkpoint(tmp_path):
    questions = _synthetic_dataset()[:1]
    checkpoint_path = tmp_path / "run" / "per_question_checkpoint.jsonl"
    run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "off-tmp",
        embeddings_enabled=True,
        checkpoint_path=checkpoint_path,
        recall_rerank=False,
    )
    with pytest.raises(ValueError, match="recall_rerank"):
        run_harness(
            questions,
            provider_name="stub",
            model="",
            tmp_dir=tmp_path / "on-tmp",
            embeddings_enabled=True,
            checkpoint_path=checkpoint_path,
            recall_rerank=True,
            resume=True,
        )


def test_dump_candidates_stub_has_header_rows_gold_and_null_markers(tmp_path):
    questions = _synthetic_dataset()
    questions.append(
        parse_question(
            _make_raw(
                "q_abs",
                "single-session-user",
                sessions={"q-abs-s": [{"role": "user", "content": "hello"}]},
                answer_session_ids=[],
            )
        )
    )
    dump_path = tmp_path / "nested" / "candidates.jsonl"
    run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "run",
        checkpoint_path=tmp_path / "run" / "per_question_checkpoint.jsonl",
        dump_candidates_path=dump_path,
    )

    lines = dump_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(questions) + 1
    assert "ZEBRA" not in dump_path.read_text(encoding="utf-8")
    header = json.loads(lines[0])
    assert header == {
        "__dump_header__": {
            "provider": "stub",
            "model": "",
            "chunk_provider": "stub",
            "chunk_model": "",
            "dataset_label": "s",
            "source_sha256": None,
            "manifest_sha256": None,
            "embeddings_enabled": True,
            "embedding_privacy_revision": None,
            "rerank": False,
            "recall_rerank": False,
            "recall_rerank_window": 0,
            "top_k": 10,
        }
    }

    rows = {row["question_id"]: row for row in map(json.loads, lines[1:])}
    assert set(rows) == {question.question_id for question in questions}
    found_null_marker = False
    for question in questions:
        row = rows[question.question_id]
        assert row["category"] == question.category
        assert row["abstention"] is question.is_abstention
        assert row["gold_sessions"] == sorted(evidence_sessions(question))
        assert row["gold_turns"] == [
            list(turn_key)
            for turn_key in sorted(evidence_turns(question))
        ]
        assert set(row["arms"]) == (set() if question.is_abstention else set(ARMS))
        for arm in row["arms"]:
            payload = row["arms"][arm]
            assert set(payload) == {"sessions_top10", "turns_top10"}
            if payload["sessions_top10"] or payload["turns_top10"]:
                assert payload["sessions_top10"]
                assert payload["turns_top10"]
            if payload["sessions_top10"]:
                assert len(payload["sessions_top10"]) <= 10
            if payload["turns_top10"]:
                assert len(payload["turns_top10"]) <= 10
                found_null_marker |= any(turn_key[1] is None for turn_key in payload["turns_top10"])
    assert found_null_marker


def test_dump_candidates_recall_matches_checkpoint_input(tmp_path):
    questions = _synthetic_dataset()
    checkpoint_path = tmp_path / "run" / "per_question_checkpoint.jsonl"
    dump_path = tmp_path / "run" / "candidates.jsonl"
    run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "tmp",
        checkpoint_path=checkpoint_path,
        dump_candidates_path=dump_path,
    )
    checkpoint_rows = {
        row["question_id"]: row
        for row in map(json.loads, checkpoint_path.read_text(encoding="utf-8").splitlines())
        if "question_id" in row
    }
    dump_rows = {
        row["question_id"]: row
        for row in map(json.loads, dump_path.read_text(encoding="utf-8").splitlines()[1:])
    }
    for question_id, dump_row in dump_rows.items():
        if dump_row["abstention"]:
            continue
        checkpoint_row = checkpoint_rows[question_id]
        gold_sessions = set(dump_row["gold_sessions"])
        for arm in ARMS:
            dumped_recall = recall_at_k(
                dump_row["arms"][arm]["sessions_top10"], gold_sessions, 10
            )
            assert dumped_recall == pytest.approx(
                checkpoint_row["arms"][arm]["recall@10"]
            )


def test_db_template_reuse_matches_from_scratch_bootstrap(tmp_path):
    # Cloning a pre-migrated template must not change any scored output vs a
    # from-scratch bootstrap per question (F7 is a speed optimization only).
    dataset = _synthetic_dataset()
    (tmp_path / "templated").mkdir()
    templated = run_harness(
        dataset, provider_name="stub", model="",
        tmp_dir=tmp_path / "templated", reuse_db_template=True,
    )
    (tmp_path / "scratch").mkdir()
    from_scratch = run_harness(
        dataset, provider_name="stub", model="",
        tmp_dir=tmp_path / "scratch", reuse_db_template=False,
    )
    for arm in ARMS:
        for metric in ("recall@1", "recall@5", "recall@10", "ndcg@10"):
            assert templated["arms"][arm][metric] == from_scratch["arms"][arm][metric]
            assert templated["arms"][arm]["turn"][metric] == from_scratch["arms"][arm]["turn"][metric]
    assert templated["ingest"]["reuse_db_template"] is True
    assert from_scratch["ingest"]["reuse_db_template"] is False


# --------------------------------------------------------------------------- #
# Production lcm_recall arm (the tool users actually call).
# --------------------------------------------------------------------------- #


def test_lcm_recall_arm_is_registered_and_scored(tmp_path):
    """The production arm is a first-class arm: registered and scored end-to-end."""
    assert "lcm_recall" in ARMS
    report = run_harness(
        _synthetic_dataset(), provider_name="stub", model="", tmp_dir=tmp_path
    )
    assert "lcm_recall" in report["arms"]
    recall = report["arms"]["lcm_recall"]
    assert set(recall) >= {"recall@1", "recall@5", "recall@10", "ndcg@10", "turn"}
    # Non-degenerate: the production path recovers the evidence session (its ZEBRA
    # cue is lexically distinctive, so at minimum the FTS arm inside recall fires).
    assert recall["recall@10"] > 0.0


def test_fresh_recall_session_is_disjoint_from_haystack_and_neutralizes_scope(tmp_path):
    """The probe's current-session id sits OUTSIDE the dataset, so the scope prior
    never boosts a dataset session (recency still applies — honest production)."""
    evidence_id = "s-evidence"
    sessions = {
        "s-noise": [{"role": "user", "content": "chatter about the weather"}],
        evidence_id: [
            {"role": "user", "content": "the locker passcode is ZEBRA0", "has_answer": True},
        ],
    }
    question = parse_question(
        _make_raw(
            "q-scope", "single-session-user", sessions=sessions,
            answer_session_ids=[evidence_id], question="what is my locker passcode ZEBRA0",
        )
    )
    fresh = fresh_recall_session_id(question)
    assert fresh not in question.haystack_session_ids
    assert fresh == fresh_recall_session_id(question)  # deterministic

    # Use the real harness StubEmbedder (it carries provider_id="stub", which the
    # production KNN arms match against the recorded profile) so the vector arms
    # return hits, exercising the true scope-prior path rather than a degraded one.
    from benchmarking.longmemeval import StubEmbedder
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.dag import SummaryDAG
    from hermes_lcm.store import MessageStore

    embedder = StubEmbedder()
    config = LCMConfig(
        database_path=str(tmp_path / f"{question.question_id}.db"), embeddings_enabled=True,
        embedding_provider="stub", embedding_model=embedder.model_id,
    )
    # Reuse the harness ingest to seed the store, then invoke the production tool.
    evaluate_question(
        question, embedder, provider_name="stub",
        tmp_dir=tmp_path, embeddings_enabled=True,
    )
    store = MessageStore(config.database_path, ingest_protection_config=config)
    dag = SummaryDAG(config.database_path)
    try:
        hits = production_recall_hits(
            question, config, store, dag, embedder,
            provider_name="stub", tmp_dir=tmp_path, embeddings_enabled=True, limit=25,
        )
    finally:
        dag.close()
        store.close()
    assert hits, "production recall returned no hits"
    # No hit belongs to the (fresh) current conversation, so the scope boost is inert.
    assert all(hit.get("from_current_session") is False for hit in hits)


def test_fresh_recall_session_avoids_haystack_collision():
    """If the sentinel id already exists in the haystack, a unique variant is used."""
    from benchmarking.longmemeval import _LCM_RECALL_FRESH_SESSION

    collide = f"{_LCM_RECALL_FRESH_SESSION}q-collide"
    raw = _make_raw(
        "q-collide", "single-session-user",
        sessions={collide: [{"role": "user", "content": "x"}]},
        answer_session_ids=[collide],
    )
    question = parse_question(raw)
    fresh = fresh_recall_session_id(question)
    assert fresh not in question.haystack_session_ids
    assert fresh.startswith(collide)


def test_recall_hit_projection_sessions_and_turns():
    """store_id -> (session, turn) projection: verbatim hits localize precisely,
    summary hits become (session, None) markers, unmapped/missing ids drop."""
    store_id_to_turn = {10: ("s1", 0), 11: ("s2", 3)}
    hits = [
        {"kind": "message_excerpt", "store_id": 10, "session_id": "s1"},
        {"kind": "summary", "node_id": 7, "session_id": "s2"},
        {"kind": "message_excerpt", "store_id": 11, "session_id": "s2"},
        {"kind": "message_excerpt", "store_id": 99, "session_id": "s9"},  # unmapped -> drop
        {"kind": "summary", "session_id": None},  # no session -> drop
    ]
    # Session ranking dedups in hit order (s2 first seen via the summary hit).
    assert recall_hit_sessions(hits) == ["s1", "s2", "s9"]
    # Turn projection: precise keys for verbatim, (session, None) for summary.
    assert recall_hit_turn_keys(hits, store_id_to_turn) == [
        ("s1", 0), ("s2", None), ("s2", 3),
    ]


def test_rrf_fuse_turn_keys_with_none_turn_tie_is_total_ordered():
    """A summary turn key (session, None) tying a localized (session, int) key
    on score and best rank must not crash the sort (None < int TypeError)."""
    from benchmarking.longmemeval import rrf_fuse

    fused = rrf_fuse([("s1", None), ("s2", 3)], [("s2", 3), ("s1", None)])
    assert set(fused) == {("s1", None), ("s2", 3)}
    # And a same-session tie between None-turn and int-turn keys:
    fused2 = rrf_fuse([("s1", None)], [("s1", 4)])
    assert set(fused2) == {("s1", None), ("s1", 4)}


def test_deterministic_summary_of_empty_session_is_non_empty():
    """Empty haystack sessions must yield embeddable (non-empty) summary text —
    cloud endpoints reject empty inputs with HTTP 400 (regression: voyage 500q
    run died on LongMemEval_S's empty sessions)."""
    from benchmarking.longmemeval import deterministic_session_summary

    assert deterministic_session_summary([]) == "(empty session)"
    assert deterministic_session_summary([{"role": "user", "content": "  "}]).strip()


def test_dump_candidates_header_binds_direct_source_sha(tmp_path):
    questions = _synthetic_dataset()
    dump_path = tmp_path / "candidates.jsonl"
    run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "run-a",
        embeddings_enabled=True,
        direct_source_sha256="a" * 64,
        dump_candidates_path=dump_path,
    )
    header = json.loads(dump_path.read_text(encoding="utf-8").splitlines()[0])
    assert header["__dump_header__"]["source_sha256"] == "a" * 64

    # A dump produced from a different direct corpus must be rejected, not
    # silently appended to.
    with pytest.raises(ValueError, match="configuration mismatch"):
        run_harness(
            questions,
            provider_name="stub",
            model="",
            tmp_dir=tmp_path / "run-b",
            embeddings_enabled=True,
            direct_source_sha256="b" * 64,
            dump_candidates_path=dump_path,
        )


def test_dump_candidates_truncates_torn_tail_and_appends(tmp_path):
    questions = _synthetic_dataset()
    dump_path = tmp_path / "candidates.jsonl"
    run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "run-a",
        embeddings_enabled=True,
        dump_candidates_path=dump_path,
    )
    complete_lines = dump_path.read_text(encoding="utf-8").splitlines()
    with dump_path.open("a", encoding="utf-8") as handle:
        handle.write('{"question_id": "torn-row-never-fini')
    run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "run-b",
        embeddings_enabled=True,
        dump_candidates_path=dump_path,
    )
    lines = dump_path.read_text(encoding="utf-8").splitlines()
    # Torn tail gone, every line parses, and the second run's rows appended.
    assert len(lines) == 2 * len(complete_lines) - 1  # header written once
    for line in lines:
        json.loads(line)
    assert not any("torn-row-never-fini" in line for line in lines)


def test_dump_candidates_record_rejects_incomplete_rankings():
    from benchmarking.longmemeval import _candidate_dump_record

    question = _synthetic_dataset()[0]
    complete = {arm: {"sessions": [], "turns": []} for arm in ARMS}
    record = _candidate_dump_record(question, complete)
    assert set(record["arms"]) == set(ARMS)

    missing_arm = dict(complete)
    missing_arm.pop(ARMS[0])
    with pytest.raises(RuntimeError, match="incomplete"):
        _candidate_dump_record(question, missing_arm)

    malformed = {arm: dict(entry) for arm, entry in complete.items()}
    malformed[ARMS[0]] = {"sessions": []}
    with pytest.raises(RuntimeError, match="malformed"):
        _candidate_dump_record(question, malformed)


def test_dump_candidates_refuses_to_mutate_foreign_file(tmp_path):
    questions = _synthetic_dataset()
    # Three foreign shapes, all rejected byte-intact: (a) full JSON with no
    # trailing newline, (b) a torn prefix like '{"' that also prefixes our own
    # header serialization (uncertain ownership fails closed), (c) a
    # newline-terminated file with a wrong header.
    cases = [
        ("no-newline.json", '{"some": "corpus file without trailing newline"}'),
        ("torn-prefix.json", '{"'),
        ("wrong-header.jsonl", '{"not_our_header": true}\n{"row": 1}\n'),
    ]
    for index, (name, content) in enumerate(cases):
        foreign = tmp_path / name
        foreign.write_text(content, encoding="utf-8")
        before = foreign.read_bytes()
        with pytest.raises(
            ValueError,
            match="configuration mismatch|invalid candidate dump header|cannot prove ownership",
        ):
            run_harness(
                questions,
                provider_name="stub",
                model="",
                tmp_dir=tmp_path / f"run-{index}",
                embeddings_enabled=True,
                dump_candidates_path=foreign,
            )
        assert foreign.read_bytes() == before, name


def test_cli_rejects_dump_candidates_aliasing_run_files(tmp_path):
    cli = _load_cli()
    output_dir = tmp_path / "out"
    dataset = tmp_path / "corpus.json"
    for alias in (
        output_dir / "per_question_checkpoint.jsonl",
        output_dir / "longmemeval_metrics.json",
        output_dir / "longmemeval_metrics.md",
        dataset,
    ):
        with pytest.raises(ValueError, match="must not alias"):
            cli._validated_dump_candidates_path(
                str(alias), output_dir=output_dir, dataset=str(dataset)
            )
    ok = cli._validated_dump_candidates_path(
        str(output_dir / "candidates.jsonl"), output_dir=output_dir, dataset=str(dataset)
    )
    assert ok == (output_dir / "candidates.jsonl").resolve()
    assert cli._validated_dump_candidates_path(None, output_dir=output_dir, dataset=None) is None


# --------------------------------------------------------------------------- #
# #345 provider identity, accounting, and provider-free resume alignment.
# --------------------------------------------------------------------------- #


class _IdentityEmbedder:
    provider_id = "voyage"
    dim = 4

    def __init__(self, model_id):
        self.model_id = model_id
        self.document_calls = 0
        self.document_batch_sizes = []
        self.query_calls = 0

    @staticmethod
    def _vector(text):
        value = float((sum(str(text).encode("utf-8")) % 7) + 1)
        return [value, 1.0, 0.5, 0.25]

    def embed_documents(self, texts):
        self.document_calls += 1
        self.document_batch_sizes.append(len(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return self._vector(text)


class _ContextualIdentityEmbedder(_IdentityEmbedder):
    supports_contextualized_grouping = True

    def __init__(self, model_id):
        super().__init__(model_id)
        self.contextual_groups = []

    def embed_chunk_group_batches(self, groups, *, before_dispatch=None):
        for group in groups:
            indexes = tuple(index for index, _text in group)
            self.contextual_groups.append(indexes)
            if before_dispatch is not None:
                before_dispatch(indexes)
            yield indexes, [self._vector(text) for _index, text in group]


class _AdvertisedContextualIdentityEmbedder(_IdentityEmbedder):
    supports_contextualized_grouping = True


class _ProductionContextualIdentityEmbedder(_ContextualIdentityEmbedder):
    def embed_chunk_group_batches(self, groups, *, before_dispatch=None):
        from hermes_lcm.embedding_provider import EmbeddedDocumentBatch

        for group in groups:
            indexes = tuple(index for index, _text in group)
            self.contextual_groups.append(indexes)
            if before_dispatch is not None:
                before_dispatch(indexes)
            yield EmbeddedDocumentBatch(
                indexes,
                tuple(tuple(self._vector(text)) for _index, text in group),
            )


class _FlatOnlyContextualIdentityEmbedder(_ContextualIdentityEmbedder):
    def embed_chunk_group_batches(self, groups, *, before_dispatch=None):
        raise AssertionError("grouped embedding must not be called in flat mode")


def _chunk_mode_question(question_id="q-chunk-mode"):
    return parse_question(
        _make_raw(
            question_id,
            "single-session-user",
            sessions={
                "s-evidence": [
                    {
                        "role": "user",
                        "content": "chunk embedding mode evidence " * 40,
                        "has_answer": True,
                    }
                ]
            },
            answer_session_ids=["s-evidence"],
            question="where is the chunk embedding mode evidence",
        )
    )


def _install_provider_set(monkeypatch, summary, chunk):
    import benchmarking.longmemeval as lme

    provider_set = lme.HarnessProviderSet(
        summary=summary,
        chunk=chunk,
        summary_binding=("voyage", summary.model_id),
        chunk_binding=("voyage", chunk.model_id),
    )
    monkeypatch.setattr(
        lme, "resolve_harness_providers", lambda *_args, **_kwargs: provider_set
    )


def test_flat_mode_forces_grouping_provider_through_flat_cacheable_path(
    tmp_path, monkeypatch
):
    import benchmarking.longmemeval as lme

    monkeypatch.setenv(lme.CHUNK_EMBEDDING_MODE_ENV, "flat")
    monkeypatch.setattr(lme, "production_recall_hits", lambda *_args, **_kwargs: [])
    summary = _IdentityEmbedder("voyage-4-large")
    chunk = _FlatOnlyContextualIdentityEmbedder("voyage-context-4")
    _install_provider_set(monkeypatch, summary, chunk)

    report = run_harness(
        [_chunk_mode_question()],
        provider_name="voyage",
        model="voyage-4-large",
        tmp_dir=tmp_path,
        reuse_db_template=False,
    )

    assert chunk.document_calls > 0
    assert report["ingest"]["chunk_embedding_mode"] == "flat"


def test_auto_mode_uses_contextual_grouping_without_cache(tmp_path, monkeypatch):
    import benchmarking.longmemeval as lme

    monkeypatch.setenv(lme.CHUNK_EMBEDDING_MODE_ENV, "auto")
    monkeypatch.setattr(lme, "production_recall_hits", lambda *_args, **_kwargs: [])
    summary = _IdentityEmbedder("voyage-4-large")
    chunk = _ContextualIdentityEmbedder("voyage-context-4")
    _install_provider_set(monkeypatch, summary, chunk)

    report = run_harness(
        [_chunk_mode_question()],
        provider_name="voyage",
        model="voyage-4-large",
        tmp_dir=tmp_path,
        reuse_db_template=False,
    )

    assert chunk.contextual_groups
    assert chunk.document_calls == 0
    assert report["ingest"]["chunk_embedding_mode"] == "contextual"


@pytest.mark.parametrize("mode", ["auto", "contextual"])
def test_contextual_cache_conflict_fails_loud(tmp_path, monkeypatch, mode):
    import benchmarking.longmemeval as lme

    monkeypatch.setenv(lme.CHUNK_EMBEDDING_MODE_ENV, mode)
    raw = _FlatOnlyContextualIdentityEmbedder("voyage-context-4")
    cached = lme.ContentHashEmbeddingCache(raw, tmp_path / f"{mode}.sqlite3")

    with pytest.raises(
        ValueError,
        match=(
            "contextual chunk grouping is not cache-backed: set "
            "LCM_LONGMEMEVAL_CHUNK_EMBEDDING_MODE=flat or unset "
            "LCM_LONGMEMEVAL_EMBED_CACHE"
        ),
    ):
        evaluate_question(
            _chunk_mode_question(f"q-{mode}-cache"),
            _IdentityEmbedder("voyage-4-large"),
            chunk_provider=cached,
            provider_name="voyage",
            tmp_dir=tmp_path,
            embeddings_enabled=True,
        )

    assert raw.document_calls == 0
    assert cached.hits == 0
    assert cached.misses == 0


def test_flat_mode_uses_wrapped_contextual_provider_cache(tmp_path, monkeypatch):
    import benchmarking.longmemeval as lme

    cache_path = tmp_path / "flat.sqlite3"
    monkeypatch.setenv(lme.CHUNK_EMBEDDING_MODE_ENV, "flat")
    monkeypatch.setenv(lme.EMBED_CACHE_ENV, str(cache_path))
    monkeypatch.setattr(lme, "production_recall_hits", lambda *_args, **_kwargs: [])
    summary = _IdentityEmbedder("voyage-4-large")
    raw = _FlatOnlyContextualIdentityEmbedder("voyage-context-4")
    cached = lme.ContentHashEmbeddingCache(raw, cache_path)
    _install_provider_set(monkeypatch, summary, cached)

    report = run_harness(
        [_chunk_mode_question("q-flat-cache")],
        provider_name="voyage",
        model="voyage-4-large",
        tmp_dir=tmp_path,
        reuse_db_template=False,
    )

    assert raw.document_calls > 0
    assert cached.misses > 0
    assert report["ingest"]["embed_cache"]["misses"] == cached.misses
    assert report["ingest"]["chunk_embedding_mode"] == "flat"


def test_contextual_mode_requires_grouping_capability(tmp_path, monkeypatch):
    import benchmarking.longmemeval as lme

    monkeypatch.setenv(lme.CHUNK_EMBEDDING_MODE_ENV, "contextual")
    with pytest.raises(
        ValueError,
        match=(
            "contextual chunk embedding requested but the chunk provider does not "
            "support contextualized grouping"
        ),
    ):
        evaluate_question(
            _chunk_mode_question("q-contextual-unsupported"),
            _IdentityEmbedder("voyage-4-large"),
            provider_name="voyage",
            tmp_dir=tmp_path,
            embeddings_enabled=True,
        )


def test_bad_chunk_embedding_mode_fails_loud(monkeypatch):
    import benchmarking.longmemeval as lme

    monkeypatch.setenv(lme.CHUNK_EMBEDDING_MODE_ENV, "surprise")
    with pytest.raises(ValueError, match="auto, flat, contextual"):
        lme._chunk_embedding_mode()


def test_prewarm_rejects_contextual_mode(tmp_path, monkeypatch):
    import benchmarking.longmemeval as lme

    monkeypatch.setenv(lme.CHUNK_EMBEDDING_MODE_ENV, "contextual")
    cached = lme.ContentHashEmbeddingCache(
        _ContextualIdentityEmbedder("voyage-context-4"),
        tmp_path / "prewarm-contextual.sqlite3",
    )
    with pytest.raises(
        ValueError,
        match="prewarm-cache populates flat chunk units; contextual mode is not cache-backed",
    ):
        lme.prewarm_embedding_cache([_chunk_mode_question()], cached)


class _CapturingIdentityEmbedder(_IdentityEmbedder):
    def __init__(self, model_id, *, provider_id="voyage"):
        super().__init__(model_id)
        self.provider_id = provider_id
        self.captured_documents = []
        self.captured_queries = []

    def embed_documents(self, texts):
        current = [str(text) for text in texts]
        self.captured_documents.append(current)
        return super().embed_documents(current)

    def embed_query(self, text):
        current = str(text)
        self.captured_queries.append(current)
        return super().embed_query(current)


class _CapturingContextualIdentityEmbedder(_CapturingIdentityEmbedder):
    supports_contextualized_grouping = True

    def __init__(self, model_id, *, provider_id="voyage"):
        super().__init__(model_id, provider_id=provider_id)
        self.captured_contextual_groups = []

    def embed_chunk_group_batches(self, groups, *, before_dispatch=None):
        for group in groups:
            current = [(int(index), str(text)) for index, text in group]
            self.captured_contextual_groups.append(current)
            indexes = tuple(index for index, _text in current)
            if before_dispatch is not None:
                before_dispatch(indexes)
            yield indexes, [self._vector(text) for _index, text in current]


_PRIVACY_SECRET = "super-secret-370"
_PRIVACY_TEXT = f"password={_PRIVACY_SECRET} " + ("privacy context " * 40)
_PRIVACY_PLACEHOLDER = "[LCM embedding privacy: name=password_assignment]"


def _privacy_question(question_id, *, question="what credential was saved"):
    return parse_question(
        _make_raw(
            question_id,
            "single-session-user",
            sessions={
                "s-private": [
                    {"role": "user", "content": _PRIVACY_TEXT, "has_answer": True}
                ]
            },
            answer_session_ids=["s-private"],
            question=question,
        )
    )


def _flatten_document_calls(provider):
    return [text for call in provider.captured_documents for text in call]


def test_cloud_summary_document_is_protected_before_provider_dispatch(tmp_path):
    summary = _CapturingIdentityEmbedder("voyage-4-large")
    chunk = _IdentityEmbedder("voyage-context-4")

    evaluate_question(
        _privacy_question("q-private-summary"),
        summary,
        chunk_provider=chunk,
        provider_name="voyage",
        tmp_dir=tmp_path,
        embeddings_enabled=True,
    )

    outbound = _flatten_document_calls(summary)
    assert outbound
    assert any(_PRIVACY_PLACEHOLDER in text for text in outbound)
    assert all(_PRIVACY_SECRET not in text for text in outbound)


def test_cloud_flat_chunk_document_is_protected_before_provider_dispatch(tmp_path):
    summary = _CapturingIdentityEmbedder("voyage-4-large")
    chunk = _CapturingIdentityEmbedder("voyage-context-4")

    evaluate_question(
        _privacy_question("q-private-flat-chunk"),
        summary,
        chunk_provider=chunk,
        provider_name="voyage",
        tmp_dir=tmp_path,
        embeddings_enabled=True,
    )

    outbound = _flatten_document_calls(chunk)
    assert outbound
    assert any(_PRIVACY_PLACEHOLDER in text for text in outbound)
    assert all(_PRIVACY_SECRET not in text for text in outbound)


def test_cloud_contextual_chunk_group_is_protected_before_provider_dispatch(tmp_path):
    summary = _CapturingIdentityEmbedder("voyage-4-large")
    chunk = _CapturingContextualIdentityEmbedder("voyage-context-4")

    evaluate_question(
        _privacy_question("q-private-contextual-chunk"),
        summary,
        chunk_provider=chunk,
        provider_name="voyage",
        tmp_dir=tmp_path,
        embeddings_enabled=True,
    )

    outbound = [
        text
        for group in chunk.captured_contextual_groups
        for _index, text in group
    ]
    assert outbound
    assert any(_PRIVACY_PLACEHOLDER in text for text in outbound)
    assert all(_PRIVACY_SECRET not in text for text in outbound)


def test_cloud_embeddings_keep_lossless_raw_corpus_in_fts(tmp_path):
    summary = _CapturingIdentityEmbedder("voyage-4-large")
    chunk = _CapturingIdentityEmbedder("voyage-context-4")
    question = _privacy_question("q-private-fts-cloud")

    evaluate_question(
        question,
        summary,
        chunk_provider=chunk,
        provider_name="voyage",
        tmp_dir=tmp_path,
        embeddings_enabled=True,
    )

    with sqlite3.connect(tmp_path / "q-private-fts-cloud.db") as connection:
        fts_text = "\n".join(
            str(row[0])
            for row in connection.execute(
                "SELECT content FROM messages_fts ORDER BY rowid"
            )
        )
    assert _PRIVACY_SECRET in fts_text
    assert _PRIVACY_PLACEHOLDER not in fts_text


def test_cloud_query_is_protected_before_provider_dispatch(tmp_path, monkeypatch):
    import benchmarking.longmemeval as lme

    monkeypatch.setattr(lme, "production_recall_hits", lambda *_args, **_kwargs: [])
    summary = _CapturingIdentityEmbedder("voyage-4-large")
    chunk = _CapturingIdentityEmbedder("voyage-context-4")

    scored = evaluate_question(
        _privacy_question("q-private-query", question=_PRIVACY_TEXT),
        summary,
        chunk_provider=chunk,
        provider_name="voyage",
        tmp_dir=tmp_path,
        embeddings_enabled=True,
    )

    outbound = summary.captured_queries + chunk.captured_queries
    assert outbound
    assert all(_PRIVACY_PLACEHOLDER in text for text in outbound)
    assert all(_PRIVACY_SECRET not in text for text in outbound)
    assert scored["privacy"]["queries"] == 2
    assert scored["privacy"]["queries_changed"] == 2


def test_query_dispatch_validator_block_increments_query_counter(tmp_path, monkeypatch):
    import benchmarking.longmemeval as lme
    from hermes_lcm import ingest_protection
    from hermes_lcm.ingest_protection import EmbeddingPrivacyPolicyError

    query = "query validator refusal token"

    def _validator(texts, *_args, **_kwargs):
        if list(texts) == [query]:
            raise EmbeddingPrivacyPolicyError("privacy policy blocked query dispatch")

    monkeypatch.setattr(
        ingest_protection, "validate_embedding_privacy_dispatch", _validator
    )
    provider = _CapturingIdentityEmbedder("voyage-4-large")

    with pytest.raises(EmbeddingPrivacyPolicyError) as raised:
        evaluate_question(
            _privacy_question("q-query-validator-block", question=query),
            provider,
            chunk_provider=provider,
            provider_name="voyage",
            tmp_dir=tmp_path,
            embeddings_enabled=True,
        )

    assert raised.value.privacy_counts["queries_blocked"] == 1
    assert raised.value.privacy_counts["blocked"] == 0
    assert provider.captured_queries == []


def test_stub_query_reaches_provider_byte_identical(tmp_path, monkeypatch):
    import benchmarking.longmemeval as lme

    monkeypatch.setattr(lme, "production_recall_hits", lambda *_args, **_kwargs: [])
    provider = _CapturingIdentityEmbedder("stub", provider_id="stub")
    question = _privacy_question("q-private-stub-query", question=_PRIVACY_TEXT)

    evaluate_question(
        question,
        provider,
        chunk_provider=provider,
        provider_name="stub",
        tmp_dir=tmp_path,
        embeddings_enabled=True,
    )

    assert provider.captured_queries == [_PRIVACY_TEXT]


def test_prewarm_cloud_documents_are_protected_before_dispatch(tmp_path):
    import benchmarking.longmemeval as lme

    raw = _CapturingIdentityEmbedder("voyage-context-4")
    cached = lme.ContentHashEmbeddingCache(raw, tmp_path / "prewarm.sqlite3")

    lme.prewarm_embedding_cache(
        [_privacy_question("q-private-prewarm")], cached, progress_every=100
    )

    outbound = _flatten_document_calls(raw)
    assert outbound
    assert any(_PRIVACY_PLACEHOLDER in text for text in outbound)
    assert all(_PRIVACY_SECRET not in text for text in outbound)


def test_prewarm_reports_privacy_transform_counts(tmp_path, monkeypatch):
    import benchmarking.longmemeval as lme

    documents = [
        "ordinary document one",
        "password: hunter2000abc",
        "ordinary document two",
    ]
    monkeypatch.setattr(
        lme,
        "iter_ingest_embedding_request_units",
        lambda _question: iter(documents),
    )
    raw = _CapturingIdentityEmbedder("voyage-context-4")
    cached = lme.ContentHashEmbeddingCache(raw, tmp_path / "prewarm.sqlite3")

    report = lme.prewarm_embedding_cache(
        [_privacy_question("q-private-prewarm-counts")], cached, progress_every=100
    )

    assert report["privacy"] == {
        "documents": 3,
        "changed": 1,
        "blocked": 0,
        "queries": 0,
        "queries_changed": 0,
        "queries_blocked": 0,
    }
    outbound = _flatten_document_calls(raw)
    assert _PRIVACY_PLACEHOLDER in outbound[1]
    assert _PRIVACY_SECRET not in outbound[1]


@pytest.mark.parametrize("dry_run", [False, True])
def test_prewarm_changed_manifest_records_exact_transformed_unit(
    tmp_path, monkeypatch, dry_run
):
    import benchmarking.longmemeval as lme
    from hermes_lcm.ingest_protection import protect_embedding_text

    documents = ["ordinary document", "password: hunter2000abc"]
    monkeypatch.setattr(
        lme,
        "iter_ingest_embedding_request_units",
        lambda _question: iter(documents),
    )
    raw = _CapturingIdentityEmbedder("voyage-context-4")
    cached = lme.ContentHashEmbeddingCache(raw, tmp_path / "manifest-cache.sqlite3")
    manifest = tmp_path / "nested" / "changed.jsonl"

    report = lme.prewarm_embedding_cache(
        [_privacy_question("q-changed-manifest")],
        cached,
        dry_run=dry_run,
        changed_manifest=manifest,
    )

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    privacy_config, privacy_revision = lme._embedding_privacy_context(
        "voyage", "voyage-context-4"
    )
    protected, _revision, changed = protect_embedding_text(
        documents[1],
        privacy_config,
        expected_revision=privacy_revision,
    )
    assert changed is True
    assert rows == [
        {
            "question_id": "q-changed-manifest",
            "unit_index": 1,
            "raw_sha256": cached.content_sha256(documents[1]),
            "protected_sha256": cached.content_sha256(protected),
        }
    ]
    assert report["changed_manifest"] == str(manifest)
    assert report["changed_units"] == report["privacy"]["changed"] == 1
    assert report["chunk_embedding_mode"] == "flat"


def test_prewarm_changed_manifest_stays_empty_without_transform(tmp_path):
    import benchmarking.longmemeval as lme

    manifest = tmp_path / "unchanged.jsonl"
    cached = lme.ContentHashEmbeddingCache(
        lme.StubEmbedder(),
        tmp_path / "unchanged-cache.sqlite3",
        provider_id="stub",
        model_id="stub-hash-64",
    )

    report = lme.prewarm_embedding_cache(
        [_chunk_mode_question("q-unchanged-manifest")],
        cached,
        changed_manifest=manifest,
    )

    assert report["changed_units"] == report["privacy"]["changed"] == 0
    assert manifest.read_text(encoding="utf-8") == ""


def test_prewarm_privacy_block_is_counted_and_reraised(tmp_path, monkeypatch):
    import benchmarking.longmemeval as lme
    from hermes_lcm.ingest_protection import EmbeddingPrivacyPolicyError

    orphan_pem = (
        "trunc -----BEGIN PRIVATE KEY-----\n"
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj\n"
        "ok.\n"
        "MHcCAQEEIQD1eJ7yhkG0987xyzABCDEFghijkLMNOPqrstuvwxyz0987654321pq"
    )
    monkeypatch.setattr(
        lme,
        "iter_ingest_embedding_request_units",
        lambda _question: iter([orphan_pem]),
    )
    raw = _CapturingIdentityEmbedder("voyage-context-4")
    cached = lme.ContentHashEmbeddingCache(raw, tmp_path / "prewarm.sqlite3")

    with pytest.raises(EmbeddingPrivacyPolicyError):
        lme.prewarm_embedding_cache(
            [_privacy_question("q-private-prewarm-blocked")], cached
        )

    assert lme._PRIVACY_COUNTS == {
        "documents": 1,
        "changed": 0,
        "blocked": 1,
        "queries": 0,
        "queries_changed": 0,
        "queries_blocked": 0,
    }
    assert raw.captured_documents == []


def test_determinism_probe_cloud_documents_are_protected_before_dispatch(tmp_path):
    import benchmarking.longmemeval as lme

    provider = _CapturingIdentityEmbedder("voyage-4-large")

    report = lme.embedding_determinism_report(
        [_privacy_question("q-private-determinism")],
        provider,
        sample_size=1,
    )

    outbound = _flatten_document_calls(provider)
    assert report["sample_size"] == 1
    assert report["privacy"] == {
        "documents": 1,
        "changed": 1,
        "blocked": 0,
        "queries": 0,
        "queries_changed": 0,
        "queries_blocked": 0,
    }
    assert outbound
    assert all(_PRIVACY_PLACEHOLDER in text for text in outbound)
    assert all(_PRIVACY_SECRET not in text for text in outbound)


def test_determinism_probe_privacy_counts_are_sample_scoped(tmp_path):
    question = parse_question(
        _make_raw(
            "q-private-determinism-sample",
            "single-session-user",
            sessions={
                "s0": [{"role": "user", "content": "password: alpha-secret"}],
                "s1": [{"role": "user", "content": "password: bravo-secret"}],
                "s2": [{"role": "user", "content": "password: charlie-secret"}],
            },
            answer_session_ids=["s0"],
        )
    )
    provider = _CapturingIdentityEmbedder("voyage-4-large")

    report = embedding_determinism_report(
        [question], provider, sample_size=1, seed=0
    )

    assert report["privacy_scope"] == "sample"
    assert report["privacy"]["documents"] == 1
    assert report["privacy"]["changed"] == 1


def test_determinism_validator_block_counts_sample_batch(tmp_path, monkeypatch):
    import benchmarking.longmemeval as lme
    from hermes_lcm import ingest_protection
    from hermes_lcm.ingest_protection import EmbeddingPrivacyPolicyError

    question = parse_question(
        _make_raw(
            "q-determinism-validator-block",
            "single-session-user",
            sessions={
                f"s{index}": [
                    {"role": "user", "content": f"unique session {index}"}
                ]
                for index in range(3)
            },
            answer_session_ids=["s0"],
        )
    )

    def _blocked(*_args, **_kwargs):
        raise EmbeddingPrivacyPolicyError("privacy policy blocked probe dispatch")

    monkeypatch.setattr(
        ingest_protection, "validate_embedding_privacy_dispatch", _blocked
    )
    with pytest.raises(EmbeddingPrivacyPolicyError) as raised:
        lme.embedding_determinism_report(
            [question], _CapturingIdentityEmbedder("voyage-4-large"), sample_size=3
        )

    assert raised.value.privacy_counts["blocked"] == 3
    assert raised.value.privacy_counts["documents"] == 3


def test_determinism_probe_block_counts_even_during_uncounted_scan(tmp_path, monkeypatch):
    import benchmarking.longmemeval as lme
    from hermes_lcm import ingest_protection
    from hermes_lcm.ingest_protection import EmbeddingPrivacyPolicyError

    question = parse_question(
        _make_raw(
            "q-private-determinism-blocked-scan",
            "single-session-user",
            sessions={
                "s0": [{"role": "user", "content": "first scanned session"}],
                "s1": [{"role": "user", "content": "second scanned session"}],
            },
            answer_session_ids=["s0"],
        )
    )

    def _blocked(*_args, **_kwargs):
        raise EmbeddingPrivacyPolicyError("privacy policy blocked scan")

    monkeypatch.setattr(ingest_protection, "protect_embedding_text", _blocked)
    with pytest.raises(EmbeddingPrivacyPolicyError) as raised:
        lme.embedding_determinism_report(
            [question], _CapturingIdentityEmbedder("voyage-4-large"), sample_size=1
        )

    assert raised.value.privacy_counts["blocked"] == 1
    assert raised.value.privacy_counts["documents"] == 0


def test_per_question_privacy_and_corpus_counts_resume_without_header_change(tmp_path):
    question = parse_question(
        _make_raw(
            "q-instrument-counts",
            "single-session-user",
            sessions={
                "s-counts": [
                    {"role": "user", "content": "first toy message"},
                    {"role": "assistant", "content": "second toy message", "has_answer": True},
                ]
            },
            answer_session_ids=["s-counts"],
            question="what was in the toy message",
        )
    )
    checkpoint = tmp_path / "run" / "per_question_checkpoint.jsonl"

    run_harness(
        [question],
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "initial",
        checkpoint_path=checkpoint,
    )

    lines = checkpoint.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header_line = lines[0]
    row = json.loads(lines[1])
    assert set(header) == {"__checkpoint_header__"}
    assert row["privacy"] == {
        "documents": 0,
        "changed": 0,
        "blocked": 0,
        "queries": 0,
        "queries_changed": 0,
        "queries_blocked": 0,
    }
    assert row["corpus_counts"] == {"messages": 2, "summary_nodes": 1, "chunks": 0}
    assert row["embed_cache"] == {"hits": 0, "misses": 0}
    assert "corpus_counts" not in row["arms"]
    assert "embed_cache" not in row["arms"]
    assert set(row["arms"]) == set(ARMS)

    # Simulate a previously completed privacy-accounted question (all six keys
    # present). The legacy three-key row shape is covered by the next test.
    row["privacy"]["documents"] = 3
    row["privacy"]["changed"] = 1
    checkpoint.write_text(
        header_line + "\n" + json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    resumed = run_harness(
        [question],
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "resume",
        checkpoint_path=checkpoint,
        resume=True,
        selected_question_ids=[question.question_id],
    )
    assert resumed["question_count"] == 1
    assert resumed["ingest"]["privacy"] == {
        "documents": 3,
        "changed": 1,
        "blocked": 0,
        "queries": 0,
        "queries_changed": 0,
        "queries_blocked": 0,
    }
    resumed_header_line = checkpoint.read_text(encoding="utf-8").splitlines()[0]
    assert resumed_header_line == header_line
    assert set(json.loads(resumed_header_line)) == {"__checkpoint_header__"}


@pytest.mark.parametrize(
    ("error_message", "should_raise"),
    [("no such table: lcm_chunk_meta", False), ("database is locked", True)],
)
def test_chunk_count_only_swallows_missing_table_operational_error(
    tmp_path, monkeypatch, error_message, should_raise
):
    import benchmarking.longmemeval as lme
    import hermes_lcm.vector_store as vector_store_module

    real_vector_store = vector_store_module.VectorStore

    class _ConnectionProxy:
        def __init__(self, connection):
            self._connection = connection

        def execute(self, sql, *args, **kwargs):
            if "lcm_chunk_meta" in str(sql):
                raise sqlite3.OperationalError(error_message)
            return self._connection.execute(sql, *args, **kwargs)

    class _VectorStoreProxy:
        def __init__(self, *args, **kwargs):
            self._inner = real_vector_store(*args, **kwargs)
            self.connection = _ConnectionProxy(self._inner.connection)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def close(self):
            self._inner.close()

    monkeypatch.setattr(vector_store_module, "VectorStore", _VectorStoreProxy)
    question = _chunk_mode_question(f"q-operational-{should_raise}")
    kwargs = {
        "provider_name": "stub",
        "tmp_dir": tmp_path,
        "embeddings_enabled": False,
    }

    if should_raise:
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            evaluate_question(question, lme.StubEmbedder(), **kwargs)
    else:
        scored = evaluate_question(question, lme.StubEmbedder(), **kwargs)
        assert scored["corpus_counts"]["chunks"] == 0


def test_resume_restores_legacy_three_key_privacy_rows(tmp_path):
    question = parse_question(
        _make_raw(
            "q-legacy-privacy-row",
            "single-session-user",
            sessions={
                "s-legacy": [
                    {"role": "user", "content": "first toy message"},
                    {"role": "assistant", "content": "second toy message", "has_answer": True},
                ]
            },
            answer_session_ids=["s-legacy"],
            question="what was in the toy message",
        )
    )
    checkpoint = tmp_path / "run" / "per_question_checkpoint.jsonl"
    run_harness(
        [question],
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "initial",
        checkpoint_path=checkpoint,
    )
    lines = checkpoint.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    # A pre-r2 row carried only the three document counters.
    row["privacy"] = {"documents": 5, "changed": 2, "blocked": 0}
    checkpoint.write_text(
        lines[0] + "\n" + json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    resumed = run_harness(
        [question],
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "resume",
        checkpoint_path=checkpoint,
        resume=True,
        selected_question_ids=[question.question_id],
    )
    assert resumed["question_count"] == 1
    assert resumed["ingest"]["privacy"] == {
        "documents": 5,
        "changed": 2,
        "blocked": 0,
        "queries": 0,
        "queries_changed": 0,
        "queries_blocked": 0,
    }


def test_embeddings_disabled_preserves_raw_corpus_text_in_fts(tmp_path):
    provider = _IdentityEmbedder("voyage-4-large")
    question = _privacy_question("q-private-fts-off")

    evaluate_question(
        question,
        provider,
        provider_name="voyage",
        tmp_dir=tmp_path,
        embeddings_enabled=False,
    )

    db_path = tmp_path / "q-private-fts-off.db"
    with sqlite3.connect(db_path) as connection:
        fts_text = "\n".join(
            str(row[0])
            for row in connection.execute(
                "SELECT content FROM messages_fts ORDER BY rowid"
            )
        )
    assert _PRIVACY_SECRET in fts_text
    assert _PRIVACY_PLACEHOLDER not in fts_text


def test_stub_summary_document_reaches_provider_byte_identical(
    tmp_path, monkeypatch
):
    import hermes_lcm.ingest_protection as ingest_protection

    def unexpected_privacy_call(*_args, **_kwargs):
        raise AssertionError("local provider must not call cloud privacy helpers")

    monkeypatch.setattr(
        ingest_protection, "protect_embedding_text", unexpected_privacy_call
    )
    monkeypatch.setattr(
        ingest_protection,
        "validate_embedding_privacy_dispatch",
        unexpected_privacy_call,
    )
    summary = _CapturingIdentityEmbedder("stub", provider_id="stub")
    chunk = _CapturingIdentityEmbedder("stub-chunk", provider_id="stub")
    question = _privacy_question("q-private-stub")
    expected = deterministic_session_summary(question.haystack_sessions[0])

    evaluate_question(
        question,
        summary,
        chunk_provider=chunk,
        provider_name="stub",
        tmp_dir=tmp_path,
        embeddings_enabled=True,
    )

    assert summary.captured_documents == [[expected]]


def test_resolve_harness_providers_uses_production_chunk_identity_and_pair_cache(
    monkeypatch,
):
    import benchmarking.longmemeval as lme

    resolutions = []
    roles = []

    def resolve(provider, model, **kwargs):
        resolutions.append((provider, model))
        roles.append((kwargs.get("accounting"), kwargs.get("accounting_role")))
        return _IdentityEmbedder(model)

    monkeypatch.setattr(lme, "resolve_harness_provider", resolve)
    accounting = ProviderAccounting()
    providers = resolve_harness_providers(
        "voyage", "voyage-4-large", accounting=accounting
    )

    assert providers.summary_binding == ("voyage", "voyage-4-large")
    assert providers.chunk_binding == ("voyage", "voyage-context-4")
    assert resolutions == [
        ("voyage", "voyage-4-large"),
        ("voyage", "voyage-context-4"),
    ]
    assert providers.summary is not providers.chunk
    assert roles == [
        (accounting, "summary_documents"),
        (accounting, "chunk_documents"),
    ]
    resolutions.clear()
    roles.clear()
    contextual_provider = resolve_harness_providers("voyage", "voyage-context-4")
    assert resolutions == [("voyage", "voyage-context-4")]
    assert contextual_provider.summary is contextual_provider.chunk
    header = lme._checkpoint_header(
        provider="voyage",
        model="voyage-4-large",
        rerank=False,
        embeddings_enabled=True,
        dataset_label="m",
        direct_source_sha256=None,
        manifest_sha256=None,
        reuse_db_template=True,
        embedding_batch_size=64,
    )["__checkpoint_header__"]
    assert (header["chunk_provider"], header["chunk_model"]) == (
        "voyage",
        "voyage-context-4",
    )


def test_embedding_disabled_run_never_resolves_distinct_chunk_provider(
    tmp_path, monkeypatch
):
    import benchmarking.longmemeval as lme

    calls = []

    def resolve_summary(provider, model, **kwargs):
        calls.append((provider, model, kwargs.get("accounting_role")))
        return _IdentityEmbedder(model)

    def forbidden_pair_resolution(*_args, **_kwargs):
        raise AssertionError("embedding-disabled run must not resolve a provider pair")

    monkeypatch.setattr(lme, "resolve_harness_provider", resolve_summary)
    monkeypatch.setattr(lme, "resolve_harness_providers", forbidden_pair_resolution)
    question = parse_question(
        _make_raw(
            "q-no-embedding_abs",
            "single-session-user",
            sessions={"s": [{"role": "user", "content": "lexical only"}]},
            answer_session_ids=[],
        )
    )

    report = run_harness(
        [question],
        provider_name="voyage",
        model="voyage-4-large",
        tmp_dir=tmp_path,
        embeddings_enabled=False,
        reuse_db_template=False,
    )

    assert report["abstention_excluded"] == 1
    assert calls == [("voyage", "voyage-4-large", "summary_documents")]


@pytest.mark.parametrize(
    "contextual_shape", [None, "advertise-only", "tuple", "production"]
)
def test_evaluate_question_separates_provider_phases_and_chunk_grouping(
    tmp_path, contextual_shape
):
    summary = _IdentityEmbedder("voyage-4-large")
    chunk_class = {
        None: _IdentityEmbedder,
        "advertise-only": _AdvertisedContextualIdentityEmbedder,
        "tuple": _ContextualIdentityEmbedder,
        "production": _ProductionContextualIdentityEmbedder,
    }[contextual_shape]
    chunk = chunk_class("voyage-context-4")
    accounting = ProviderAccounting()
    question = parse_question(
        _make_raw(
            "q-accounting",
            "single-session-user",
            sessions={
                "s-evidence": [
                    {
                        "role": "user",
                        "content": "accounting chunk evidence " * 40,
                    },
                    {
                        "role": "assistant",
                        "content": "second accounting chunk evidence " * 40,
                        "has_answer": True,
                    }
                ]
            },
            answer_session_ids=["s-evidence"],
            question="where is the accounting chunk evidence",
        )
    )

    evaluate_question(
        question,
        summary,
        chunk_provider=chunk,
        accounting=accounting,
        provider_name="voyage",
        tmp_dir=tmp_path,
        embeddings_enabled=True,
    )

    snapshot = accounting.snapshot()
    assert snapshot["summary_documents"]["requests"] == 1
    assert snapshot["chunk_documents"]["requests"] >= 1
    assert snapshot["harness_queries"]["requests"] == 2
    assert snapshot["harness_queries"]["documents"] == 2
    # Production recall embeds its query on both arms (summary + chunk) before
    # candidate scoping, so two provider requests are the truthful count. The
    # previous ==0 expectation was an artifact of the pre-#370 silent privacy
    # degrade: the cloud-privacy gate raised before any provider call and
    # lcm_recall swallowed it into an FTS fallback, so no query ever dispatched.
    assert snapshot["production_lcm_recall_queries"]["requests"] == 2
    assert snapshot["production_lcm_recall_queries"]["usage_tokens"] is None
    for role in ("summary_documents", "chunk_documents", "harness_queries"):
        assert snapshot[role]["usage_tokens_complete"] is False
        assert snapshot[role]["usage_tokens"] is None
        assert snapshot[role]["known_usage_tokens"] == 0
    if contextual_shape in {"tuple", "production"}:
        assert len(chunk.contextual_groups) >= 2
        assert snapshot["chunk_documents"]["requests"] == 1
        assert snapshot["chunk_documents"]["provider_dispatches"] == len(
            chunk.contextual_groups
        )
        assert chunk.document_calls == 0
    else:
        assert chunk.document_calls == snapshot["chunk_documents"]["requests"]
        assert max(chunk.document_batch_sizes) == 1


def test_evaluate_question_reuses_one_query_for_same_provider_identity(tmp_path):
    provider = _IdentityEmbedder("voyage-context-4")
    accounting = ProviderAccounting()
    question = parse_question(
        _make_raw(
            "q-query-reuse",
            "single-session-user",
            sessions={"s": [{"role": "user", "content": "same identity evidence"}]},
            answer_session_ids=["s"],
            question="same identity question",
        )
    )

    evaluate_question(
        question,
        provider,
        chunk_provider=provider,
        accounting=accounting,
        provider_name="voyage",
        tmp_dir=tmp_path,
        embeddings_enabled=True,
    )

    assert accounting.snapshot()["harness_queries"]["requests"] == 1


def test_production_recall_accounts_separate_summary_and_chunk_queries(
    tmp_path, monkeypatch
):
    import benchmarking.longmemeval as lme

    lme._ensure_hermes_lcm_package()
    import hermes_lcm.tools as lcm_tools

    accounting = ProviderAccounting()
    summary = lme._AccountingProvider(
        _IdentityEmbedder("voyage-4-large"),
        accounting,
        "production_lcm_recall_queries",
    )
    chunk = lme._AccountingProvider(
        _IdentityEmbedder("voyage-context-4"),
        accounting,
        "production_lcm_recall_queries",
    )

    def fake_recall(args, *, engine):
        engine._lcm_embedding_provider_cache[1].embed_query(args["query"])
        engine._lcm_chunk_provider_cache[1].embed_query(args["query"])
        return json.dumps({"hits": [], "degraded": False})

    monkeypatch.setattr(lcm_tools, "lcm_recall", fake_recall)
    production_recall_hits(
        _synthetic_dataset()[0],
        type("Config", (), {})(),
        None,
        None,
        summary,
        chunk_provider_embedder=chunk,
        provider_name="voyage",
        chunk_provider_name="voyage",
        tmp_dir=tmp_path,
        embeddings_enabled=True,
        limit=10,
        accounting=accounting,
    )

    row = accounting.snapshot()["production_lcm_recall_queries"]
    assert row["requests"] == 2
    assert row["documents"] == 2
    assert row["provider_dispatches"] == 2
    assert row["usage_tokens_complete"] is False
    assert row["usage_tokens"] is None


def test_typed_provider_degraded_outcomes_ignore_general_and_reject_unknown_provider():
    assert _typed_provider_degraded_outcomes(
        {
            "degraded": True,
            "degraded_reason": (
                "embedding provider is not configured; "
                "full-text arm unavailable; "
                "chunk embedding provider unavailable: offline; "
                "summary candidate coverage bounded at 10"
            ),
        }
    ) == ("summary_provider_disabled", "chunk_provider_unavailable")
    assert _typed_provider_degraded_outcomes(
        {
            "degraded": True,
            "degraded_reason": "session exclusion scope resolution timed out",
        }
    ) == ()
    with pytest.raises(ValueError, match="unknown provider degraded reason"):
        _typed_provider_degraded_outcomes(
            {
                "degraded": True,
                "degraded_reason": "embedding provider entered surprising mode",
            }
        )


def test_content_cache_accounts_dispatches_by_pair_and_query(tmp_path):
    from benchmarking.longmemeval import ContentHashEmbeddingCache, _AccountingProvider

    class UsageEmbedder(_IdentityEmbedder):
        def __init__(self, model_id):
            super().__init__(model_id)
            self.last_usage_tokens = None
            self._usage_serial = 0

        def _next_usage(self):
            self._usage_serial += 1
            self.last_usage_tokens = self._usage_serial

        def embed_documents(self, texts):
            self._next_usage()
            return super().embed_documents(texts)

        def embed_query(self, text):
            self._next_usage()
            return super().embed_query(text)

    first = UsageEmbedder("voyage-4-large")
    second = UsageEmbedder("voyage-context-4")
    cache_path = tmp_path / "embeddings.sqlite3"
    summary_cache = ContentHashEmbeddingCache(first, cache_path)
    chunk_cache = ContentHashEmbeddingCache(second, cache_path)
    accounting = ProviderAccounting()
    summary_documents = _AccountingProvider(
        summary_cache, accounting, "summary_documents"
    )
    harness_queries = _AccountingProvider(
        summary_cache, accounting, "harness_queries"
    )

    summary_documents.embed_documents(["same text"])
    summary_documents.embed_documents(["same text"])
    chunk_cache.embed_documents(["same text"])
    harness_queries.embed_query("question")

    snapshot = accounting.snapshot()
    assert snapshot["summary_documents"]["requests"] == 2
    assert snapshot["summary_documents"]["provider_dispatches"] == 1
    assert snapshot["summary_documents"]["usage_tokens_complete"] is True
    assert snapshot["harness_queries"]["provider_dispatches"] == 1
    assert first.document_calls == 1
    assert second.document_calls == 1


def test_accounting_accepts_repeated_usage_and_records_failed_dispatch():
    import benchmarking.longmemeval as lme

    class RepeatUsage(_IdentityEmbedder):
        provider_dispatches = 0
        last_usage_tokens = 7

        def embed_query(self, text):
            self.provider_dispatches += 1
            self.last_usage_tokens = 7
            return super().embed_query(text)

    provider = RepeatUsage("voyage-4-large")
    accounting = ProviderAccounting()
    wrapped = lme._AccountingProvider(provider, accounting, "harness_queries")
    wrapped.embed_query("same token total")
    wrapped.embed_query("same token total")
    row = accounting.snapshot()["harness_queries"]
    assert row["provider_dispatches"] == 2
    assert row["usage_tokens"] == 14
    assert row["usage_tokens_complete"] is True

    class FailingProvider(RepeatUsage):
        def embed_query(self, text):
            self.provider_dispatches += 1
            raise RuntimeError("transport failed")

    failed_accounting = ProviderAccounting()
    failed = lme._AccountingProvider(
        FailingProvider("voyage-4-large"), failed_accounting, "harness_queries"
    )
    with pytest.raises(RuntimeError, match="transport failed"):
        failed.embed_query("failed query")
    failed_row = failed_accounting.snapshot()["harness_queries"]
    assert failed_row["requests"] == 1
    assert failed_row["provider_dispatches"] == 1
    assert failed_row["usage_tokens_complete"] is False
    assert failed_row["usage_tokens"] is None


def test_accounted_warmup_attempt_is_not_omitted():
    import benchmarking.longmemeval as lme

    class WarmupProvider(_IdentityEmbedder):
        provider_dispatches = 0
        last_usage_tokens = None

        def warmup(self):
            self.provider_dispatches += 1
            self.last_usage_tokens = 3
            return self._vector("warmup")

    provider = WarmupProvider("voyage-context-4")
    accounting = ProviderAccounting()
    assert lme._accounted_provider_attempt(
        provider, accounting, "chunk_documents", provider.warmup
    )
    row = accounting.snapshot()["chunk_documents"]
    assert row["requests"] == 1
    assert row["provider_dispatches"] == 1
    assert row["usage_tokens"] == 3


@pytest.mark.parametrize("path", ["flat", "cached", "contextual"])
def test_predispatch_refusal_records_no_provider_transport(tmp_path, path):
    import benchmarking.longmemeval as lme

    lme._ensure_hermes_lcm_package()
    from hermes_lcm.embedding_provider import ProviderPreDispatchError

    class RefusingProvider(_IdentityEmbedder):
        transport_calls = 0

        def embed_document_batches(self, texts, *, before_dispatch=None):
            if before_dispatch is not None:
                before_dispatch((0,))
            raise ProviderPreDispatchError("refused before document transport")
            yield  # pragma: no cover - keep this an iterator

        def embed_chunk_group_batches(self, groups, *, before_dispatch=None):
            if before_dispatch is not None:
                before_dispatch((0,))
            raise ProviderPreDispatchError("refused before contextual transport")
            yield  # pragma: no cover - keep this an iterator

    raw = RefusingProvider("voyage-context-4")
    provider = (
        lme.ContentHashEmbeddingCache(raw, tmp_path / "cache.sqlite3")
        if path == "cached"
        else raw
    )
    accounting = ProviderAccounting()
    wrapped = lme._AccountingProvider(provider, accounting, "chunk_documents")

    with pytest.raises(ProviderPreDispatchError):
        if path == "contextual":
            list(wrapped.embed_chunk_group_batches([[(0, "chunk")]]))
        else:
            wrapped.embed_documents(["chunk"])

    row = accounting.snapshot()["chunk_documents"]
    assert row["requests"] == 1
    assert row["provider_dispatches"] == 0
    assert row["usage_tokens"] == 0
    assert row["usage_tokens_complete"] is True
    assert raw.transport_calls == 0
    if path == "cached":
        assert provider.provider_dispatches == 0


def test_content_cache_exposes_internal_split_dispatches_and_usage(tmp_path):
    import benchmarking.longmemeval as lme

    lme._ensure_hermes_lcm_package()
    from hermes_lcm.embedding_provider import EmbeddedDocumentBatch

    class SplitProvider(_IdentityEmbedder):
        last_usage_tokens = None

        def embed_document_batches(self, texts, *, before_dispatch=None):
            for indexes in ((0, 1), (2,)):
                if before_dispatch is not None:
                    before_dispatch(indexes)
                self.last_usage_tokens = 5
                yield EmbeddedDocumentBatch(
                    indexes,
                    tuple(tuple(self._vector(texts[index])) for index in indexes),
                )

    cached = lme.ContentHashEmbeddingCache(
        SplitProvider("voyage-4-large"), tmp_path / "split.sqlite3"
    )
    accounting = ProviderAccounting()
    wrapped = lme._AccountingProvider(cached, accounting, "summary_documents")
    assert len(wrapped.embed_documents(["a", "b", "c"])) == 3
    row = accounting.snapshot()["summary_documents"]
    assert row["requests"] == 1
    assert row["provider_dispatches"] == 2
    assert row["usage_tokens"] == 10
    assert row["usage_tokens_complete"] is True


def test_completed_resume_and_binding_mismatch_do_not_resolve_provider(
    tmp_path, monkeypatch
):
    import benchmarking.longmemeval as lme

    questions = _synthetic_dataset()
    selected_ids = [question.question_id for question in questions]
    checkpoint = tmp_path / "run" / "per_question_checkpoint.jsonl"
    accounting = ProviderAccounting()
    initial_report = run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "initial",
        checkpoint_path=checkpoint,
        accounting=accounting,
    )
    assert accounting.snapshot()["summary_documents"]["requests"] > 0
    assert "accounting" not in initial_report
    assert "accounting" not in checkpoint.read_text(encoding="utf-8")

    def forbidden_resolution(*_args, **_kwargs):
        raise AssertionError("provider resolution must not occur")

    monkeypatch.setattr(lme, "resolve_harness_providers", forbidden_resolution)
    resumed = run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "resume",
        checkpoint_path=checkpoint,
        resume=True,
        selected_question_ids=selected_ids,
    )
    assert resumed["question_count"] == len(questions)

    lines = checkpoint.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["__checkpoint_header__"]["chunk_model"] = "wrong-chunk-binding"
    checkpoint.write_text(
        "\n".join([json.dumps(header, sort_keys=True), *lines[1:]]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checkpoint configuration mismatch"):
        run_harness(
            questions,
            provider_name="stub",
            model="",
            tmp_dir=tmp_path / "mismatch",
            checkpoint_path=checkpoint,
            resume=True,
            selected_question_ids=selected_ids,
        )


def test_resume_binds_active_embedding_privacy_revision(tmp_path, monkeypatch):
    import benchmarking.longmemeval as lme

    question = parse_question(
        _make_raw(
            "q-private-revision_abs",
            "single-session-user",
            sessions={"s": [{"role": "user", "content": "no scored dispatch"}]},
            answer_session_ids=[],
        )
    )
    provider = _IdentityEmbedder("voyage-4-large")
    provider_set = lme.HarnessProviderSet(
        summary=provider,
        chunk=provider,
        summary_binding=("voyage", "voyage-4-large"),
        chunk_binding=("voyage", "voyage-context-4"),
    )
    monkeypatch.setattr(
        lme, "resolve_harness_providers", lambda *_args, **_kwargs: provider_set
    )
    checkpoint = tmp_path / "run" / "per_question_checkpoint.jsonl"
    dump_path = tmp_path / "run" / "candidates.jsonl"

    run_harness(
        [question],
        provider_name="voyage",
        model="voyage-4-large",
        tmp_dir=tmp_path / "initial",
        checkpoint_path=checkpoint,
        dump_candidates_path=dump_path,
        reuse_db_template=False,
    )

    checkpoint_lines = checkpoint.read_text(encoding="utf-8").splitlines()
    checkpoint_header = json.loads(checkpoint_lines[0])
    active_revision = checkpoint_header["__checkpoint_header__"][
        "embedding_privacy_revision"
    ]
    assert isinstance(active_revision, str) and active_revision
    dump_header = json.loads(dump_path.read_text(encoding="utf-8").splitlines()[0])
    assert dump_header["__dump_header__"]["embedding_privacy_revision"] == active_revision

    def forbidden_resolution(*_args, **_kwargs):
        raise AssertionError("completed matching-revision resume must not resolve providers")

    monkeypatch.setattr(lme, "resolve_harness_providers", forbidden_resolution)
    resumed = run_harness(
        [question],
        provider_name="voyage",
        model="voyage-4-large",
        tmp_dir=tmp_path / "matching",
        checkpoint_path=checkpoint,
        resume=True,
        selected_question_ids=[question.question_id],
        reuse_db_template=False,
    )
    assert resumed["question_count"] == 1

    stale_revision = "embedding-privacy-stale:test"
    checkpoint_header["__checkpoint_header__"][
        "embedding_privacy_revision"
    ] = stale_revision
    checkpoint.write_text(
        "\n".join(
            [json.dumps(checkpoint_header, sort_keys=True), *checkpoint_lines[1:]]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc_info:
        run_harness(
            [question],
            provider_name="voyage",
            model="voyage-4-large",
            tmp_dir=tmp_path / "mismatch",
            checkpoint_path=checkpoint,
            resume=True,
            selected_question_ids=[question.question_id],
            reuse_db_template=False,
        )
    message = str(exc_info.value)
    assert "embedding_privacy_revision" in message
    assert stale_revision in message
    assert active_revision in message


def test_provider_aliases_require_privacy_binding():
    # R5 review: the resolver maps "openai"/"siliconflow" to the
    # openai-compatible cloud provider, so the privacy predicate must hold
    # for the alias spellings too — a config-string alias must never reach
    # the cloud with raw payloads.
    import benchmarking.longmemeval as lme

    lme._ensure_hermes_lcm_package()
    from hermes_lcm.ingest_protection import embedding_provider_requires_privacy

    for alias in ("openai", "siliconflow", "openai-compatible", "voyage"):
        assert embedding_provider_requires_privacy(alias), alias
    for local in ("stub", "fastembed", "ollama"):
        assert not embedding_provider_requires_privacy(local), local
    _config, revision = lme._embedding_privacy_context(
        "openai", "text-embedding-x", embeddings_enabled=True
    )
    assert revision is not None
