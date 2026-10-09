"""Frozen public/synthetic retrieval evaluation; gold never reaches scorers.

Production capture is a callback inside the existing LongMemEval instrument,
while its per-question store is still open. Its ID-only dump is not this format.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from optional_scorer import Candidate, score_candidates, validate_candidates

SCHEMA = "portable-recall-v1"
DATASET_REVISION = "2ec2a557f339b6c0369619b1ed5793734cc87533"
CATEGORIES = ("exact-id", "paraphrase", "correction", "negation", "date",
              "contradiction", "noanswer", "fork", "scope", "tool-argument")


class SpanCaptureUnavailable(ValueError):
    """Production text cannot be assigned an exact, unchanged source interval."""


@dataclass(frozen=True)
class GoldSpan:
    source_ref: str
    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True)
class FrozenCase:
    case_id: str
    conversation_id: str
    split: str
    category: str
    query: str
    candidates: tuple[Candidate, ...]
    lexical_ids: tuple[str, ...]
    gold: tuple[GoldSpan, ...] | None  # None means exact gold is unavailable
    noanswer: bool = False
    capture_ms: float = 0.0
    delivery_detail: str = "snippets"
    delivery_include: str = "all"


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def validate_cases(cases):
    identities, conversations = set(), {}
    for case in cases:
        validate_candidates(case.candidates)
        if case.delivery_detail not in {"snippets", "answer_ready"} or case.delivery_include not in {"all", "summaries", "verbatim"}:
            raise ValueError("invalid frozen delivery profile")
        if case.case_id in identities or case.split not in {"dev", "holdout"}:
            raise ValueError("duplicate case or invalid split")
        identities.add(case.case_id)
        prior = conversations.setdefault(case.conversation_id, case.split)
        if prior != case.split:
            raise ValueError("conversation crosses evaluation split")
        pool_ids = {c.candidate_id for c in case.candidates}
        if len(set(case.lexical_ids)) != len(case.lexical_ids) or not set(case.lexical_ids) <= pool_ids:
            raise ValueError("lexical ranking is not within the frozen pool")
        if case.noanswer and case.gold:
            raise ValueError("no-answer case has positive gold")
        for span in case.gold or ():
            if not span.text or span.char_start < 0 or span.char_end - span.char_start != len(span.text):
                raise ValueError("invalid gold span")
        if len(set(case.gold or ())) != len(case.gold or ()):
            raise ValueError("duplicate gold span")


def write_frozen(path: Path, cases, *, input_kind: str, posture: str, source_identity: str,
                 provenance: dict | None = None):
    """Create once; never replace a previously frozen corpus."""
    if input_kind not in {"synthetic", "public"} or posture not in {"OFF", "LOCAL"}:
        raise ValueError("invalid corpus permission or posture")
    cases = tuple(cases)
    validate_cases(cases)
    records = [asdict(case) for case in cases]
    header = {"schema": SCHEMA, "input_kind": input_kind, "posture": posture,
              "source_identity": source_identity, "case_count": len(records),
              "corpus_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
              "candidate_cap": 30, "production_hit_cap": 25}
    header.update(delivery_configuration(cases))
    if input_kind == "synthetic" and all(
        (c.delivery_detail, c.delivery_include) == ("answer_ready", "verbatim") for c in cases
    ):
        header["synthetic_single_session_delivery_cap"] = 5
    if provenance is not None:
        header["provenance"] = provenance
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"header": header}, sort_keys=True) + "\n")
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    return header


def read_frozen(path: Path):
    with path.open(encoding="utf-8") as stream:
        header = json.loads(next(stream))["header"]
        records = [json.loads(line) for line in stream]
    if header["schema"] != SCHEMA or header["input_kind"] not in {"synthetic", "public"}:
        raise ValueError("unsupported frozen input")
    if header["posture"] not in {"OFF", "LOCAL"} or header["case_count"] != len(records):
        raise ValueError("invalid frozen provenance")
    if hashlib.sha256(canonical_bytes(records)).hexdigest() != header["corpus_sha256"]:
        raise ValueError("frozen corpus digest mismatch")
    cases = []
    for record in records:
        record["candidates"] = tuple(Candidate(**{**c, "metadata": tuple(tuple(p) for p in c["metadata"])})
                                     for c in record["candidates"])
        record["lexical_ids"] = tuple(record["lexical_ids"])
        record["gold"] = None if record["gold"] is None else tuple(GoldSpan(**s) for s in record["gold"])
        cases.append(FrozenCase(**record))
    validate_cases(cases)
    return header, tuple(cases)


def synthetic_cases():
    """120 deterministically identified cases, 40 dev / 80 held-out.

Returns messages and gold separately; capture uses the normal production path.
Different conversations never cross splits. Out-of-scope fixture events must
remain outside the selected corpus; they are supplied separately for isolation
checks by the owning portable-host harness.
"""
    cases = []
    for category in CATEGORIES:
        for index in range(12):
            identity = f"synthetic-{category}-{index:02}"
            tag = f"CEDAR-{index:03}"
            scenarios = {
                "exact-id": (f"What is the delivery code for {tag}?", f"Delivery code for {tag} is violet-{index}.", []),
                "paraphrase": (f"How will {tag} be sent?", f"We chose rail freight for transporting {tag}.", []),
                "correction": (f"What is the corrected color for {tag}?", f"Correction: {tag} is amber, replacing the earlier blue choice.", [f"Earlier choice: {tag} is blue."]),
                "negation": (f"Which mode was rejected for {tag}?", f"We rejected air transport for {tag}; rail was selected.", []),
                "date": (f"On what date will {tag} launch?", f"{tag} launches on 2027-03-{index + 1:02}.", []),
                "contradiction": (f"Which statement supersedes the old {tag} limit?", f"The latest decision sets {tag} limit to 9 and supersedes the old limit of 4.", [f"Old decision: {tag} limit is 4."]),
                "noanswer": (f"What is the unrecorded access phrase for {tag}?", f"{tag} has no recorded access phrase.", []),
                "fork": (f"Which color was chosen in this fork for {tag}?", f"This fork chose green for {tag}.", []),
                "scope": (f"What is this project's endpoint for {tag}?", f"This project's {tag} endpoint is /fixtures/{index}.", []),
                "tool-argument": (f"What literal --output argument was used for {tag}?", f"The fixture command for {tag} was export --output ./violet-{index}.json.", []),
            }
            query, answer, previous = scenarios[category]
            distractors = [f"{tag} planning note {n}: review unrelated catalog item {n}." for n in range(18)]
            messages = previous + distractors[:9] + [answer] + distractors[9:]
            excluded = [f"Parent or other project {tag}: color red; endpoint /other."] if category in {"fork", "scope"} else []
            cases.append({"case_id": identity, "conversation_id": identity,
                          "split": "dev" if index < 4 else "holdout", "category": category,
                          "query": query, "messages": messages,
                          "gold_message": None if category == "noanswer" else len(previous) + 9,
                          "excluded_sources": excluded})
    return tuple(cases)


def delivery_configuration(cases):
    profiles = sorted({(c.delivery_detail, c.delivery_include) for c in cases})
    maximum = max((len(c.candidates) for c in cases), default=0)
    return {"delivery_profiles": [{"detail": d, "include": i} for d, i in profiles],
            "observed_max_candidate_count": maximum,
            "recall_at_5_gain_possible_by_permutation": maximum > 5}


def capture_synthetic(*, tmp_dir: Path, posture: str, local_model: str = ""):
    """Reuse evaluate_question; require its root-owned candidate_sink seam.

The sink receives question/config/store/dag/provider, tmp_dir,
embeddings_enabled/provider_name/chunk_provider, hits, store_id_to_turn,
and capture_ms as keyword arguments before any store closes. OFF uses the
instrument's stub only with embeddings disabled. LOCAL uses real FastEmbed.
"""
    import inspect
    from dataclasses import replace
    from . import longmemeval as instrument
    if "candidate_sink" not in inspect.signature(instrument.evaluate_question).parameters:
        raise RuntimeError("existing instrument needs the candidate_sink integration")
    if posture not in {"OFF", "LOCAL"} or (posture == "LOCAL" and not local_model):
        raise ValueError("LOCAL requires an explicit local embedding model")
    provider_name = "stub" if posture == "OFF" else "fastembed"
    providers = instrument.resolve_harness_providers(provider_name, local_model)
    captured = []
    for fixture in synthetic_cases():
        identity = fixture["case_id"]
        noanswer = fixture["gold_message"] is None
        question = instrument.Question(
            identity + ("_abs" if noanswer else ""), fixture["category"], fixture["query"],
            [identity], [[{"role": "user", "content": text,
                          "has_answer": index == fixture["gold_message"]}
                         for index, text in enumerate(fixture["messages"])]],
            [] if noanswer else [identity],
        )

        def sink(*, store_id_to_turn, capture_ms=0, **context):
            annotations = []
            if not noanswer:
                selected = [sid for sid, turn in store_id_to_turn.items()
                            if turn == (identity, fixture["gold_message"])]
                if len(selected) != 1:
                    raise ValueError("synthetic gold projection is ambiguous")
                source = context["store"].get(selected[0])["content"]
                if source != fixture["messages"][fixture["gold_message"]]:
                    raise ValueError("synthetic source projection changed")
                annotations = [(selected[0], 0, len(source))]
            case = freeze_production_pool(**context, scope_id=identity,
                                         split=fixture["split"], span_annotations=annotations,
                                         detail="answer_ready", include="verbatim")
            captured.append(replace(case, case_id=identity, capture_ms=capture_ms))

        instrument.evaluate_question(
            question, providers.summary, chunk_provider=providers.chunk,
            provider_name=provider_name, tmp_dir=tmp_dir,
            embeddings_enabled=posture == "LOCAL", top_k=30, candidate_sink=sink,
            recall_detail="answer_ready", recall_include="verbatim",
        )
    if len(captured) != 120:
        raise ValueError("production sink did not capture every synthetic case once")
    if any(len(case.candidates) > 5 for case in captured):
        raise ValueError("single-session answer-ready delivery exceeded its production cap")
    validate_cases(captured)
    return tuple(captured)


def capture_public(dataset: Path, *, expected_sha256: str, tmp_dir: Path,
                   posture: str, local_model: str, dev_header: dict):
    """Full pinned LongMemEval_S holdout; one existing baseline pass per case.

The expected byte checksum comes from the root's pinned download receipt.
No exact character-span annotations exist in these official labels. Failed
source projections are recorded and excluded from the frozen scoring pool.
"""
    import inspect
    from dataclasses import replace
    from . import longmemeval as instrument
    if "candidate_sink" not in inspect.signature(instrument.evaluate_question).parameters:
        raise RuntimeError("existing instrument needs the candidate_sink integration")
    if posture not in {"OFF", "LOCAL"} or (posture == "LOCAL" and not local_model):
        raise ValueError("LOCAL requires an explicit local embedding model")
    if instrument.DATASET_REVISION != DATASET_REVISION:
        raise ValueError("instrument dataset revision differs from the registered pin")
    instrument.validate_dataset_path_label(dataset, "s")
    if len(expected_sha256) != 64 or set(expected_sha256) - set("0123456789abcdef"):
        raise ValueError("expected dataset SHA-256 must come from the pinned download receipt")
    questions, actual_sha256 = instrument.load_questions_with_sha256(dataset)
    if actual_sha256 != expected_sha256 or len(questions) != 500:
        raise ValueError("full LongMemEval_S checksum/count mismatch")
    if len({q.question_id for q in questions}) != len(questions):
        raise ValueError("duplicate public question identity")
    provider_name = "stub" if posture == "OFF" else "fastembed"
    providers = instrument.resolve_harness_providers(provider_name, local_model)
    cases, rows = [], []
    for question in questions:
        capture = {"status": "MISSING_CAPTURE", "exact_span_gold": "UNMEASURED"}
        calls = 0

        def sink(*, store_id_to_turn, capture_ms=0, **context):
            nonlocal calls
            calls += 1
            try:
                case = freeze_production_pool(**context, scope_id=question.question_id,
                                             split="holdout", span_annotations=None)
            except SpanCaptureUnavailable:
                capture["status"] = "UNMEASURED_CAPTURE: source_projection"
                return
            cases.append(replace(case, capture_ms=capture_ms))
            capture["status"] = "CAPTURED"

        scores = instrument.evaluate_question(
            question, providers.summary, chunk_provider=providers.chunk,
            provider_name=provider_name, tmp_dir=tmp_dir,
            embeddings_enabled=posture == "LOCAL", top_k=10, candidate_sink=sink,
        )
        if calls != 1:
            raise ValueError("production sink must capture once per public baseline question")
        rows.append({"question_id": question.question_id, "category": question.category,
                     "capture": capture, "current_instrument_metrics": scores})
    validate_cases(cases)
    provenance = {
        "dataset": {**instrument.dataset_coordinates("s"), "sha256": actual_sha256},
        "posture": posture, "summary_binding": list(providers.summary_binding),
        "chunk_binding": list(providers.chunk_binding), "full_holdout_questions": 500,
        "current_instrument_top_k": 10, "rerank": False, "recall_rerank": False,
        "captured_questions": len(cases), "unmeasured_capture_questions": 500 - len(cases),
        "exact_span_gold": "UNMEASURED", "answer_correctness": "UNMEASURED",
        "dev_registration": {"corpus_sha256": dev_header["corpus_sha256"],
                             "synthetic_dev": 40, "synthetic_holdout": 80},
    }
    return tuple(cases), {"provenance": provenance, "per_question": rows}


def freeze_production_pool(
    question, config, store, dag, provider, *, tmp_dir, embeddings_enabled,
    provider_name, scope_id, split, chunk_provider=None, span_annotations=None,
    hits=None,
    detail="snippets", include="all",
):
    """Callback contract for the existing per-question LongMemEval instrument.

``hits`` may be its already obtained production hits. Otherwise invoke its
production_recall_hits helper at requested limit=30 (the actual tool caps 25).
Gold annotations are (stored message ID, start, end) triples with explicit
source character intervals; absent annotations remain UNMEASURED. No summary credit.
The store must be a fresh public corpus. This function never opens a live DB.
"""
    from . import longmemeval as instrument
    if hits is None:
        hits = instrument.production_recall_hits(
            question, config, store, dag, provider, tmp_dir=tmp_dir,
            embeddings_enabled=embeddings_enabled, provider_name=provider_name,
            chunk_provider_embedder=chunk_provider, limit=30,
            detail=detail, include=include,
        )
    candidates = []
    for hit in hits:
        if hit.get("kind") == "summary":
            node = dag.get_node(int(hit["node_id"]))
            source = node.summary
            ref = f"{scope_id}:summary:{hit['node_id']}"
        else:
            row = store.get(int(hit["store_id"]))
            source = row["content"]
            ref = f"{scope_id}:message:{hit['store_id']}"
        text = hit.get("content") if isinstance(hit.get("content"), str) else hit["snippet"]
        offset = hit.get("content_offset")
        if detail == "answer_ready" and (
            type(offset) is not int or offset < 0
            or type(hit.get("content_returned_chars")) is not int
            or hit["content_returned_chars"] != len(text)
        ):
            raise SpanCaptureUnavailable("citation-bearing delivery lacks an exact span")
        if offset is None:
            offset = source.find(text)
            if offset < 0 or source.find(text, offset + 1) >= 0:
                raise SpanCaptureUnavailable("production span is ambiguous; use exact answer-ready capture")
        if source[offset:offset + len(text)] != text:
            raise SpanCaptureUnavailable("production source round-trip failed")
        candidates.append(Candidate(f"{ref}:{offset}-{offset + len(text)}", text,
                                    scope_id, ref, offset, offset + len(text)))
    # Existing FTS, restricted to this already-frozen pool; no replacement router.
    lexical_store_ids = [sid for _, sid in instrument.fts_hits(store, question.question, 30)]
    lexical = tuple(c.candidate_id for sid in lexical_store_ids for c in candidates
                    if c.source_ref == f"{scope_id}:message:{sid}")
    lexical = tuple(dict.fromkeys(lexical))
    gold = None
    if span_annotations is not None:
        gold_list = []
        # Caller annotations use actual stored source IDs, avoiding inference
        # from summary text or an unverified dataset-turn projection.
        for annotation in span_annotations:
            sid, start, end = annotation
            row = store.get(sid)
            text = row["content"][start:end]
            if not text or start < 0 or end > len(row["content"]):
                raise ValueError("invalid explicit source annotation")
            gold_list.append(GoldSpan(f"{scope_id}:message:{sid}", start, end, text))
        gold = tuple(gold_list)
    return FrozenCase(question.question_id, scope_id, split, question.category,
                      question.question, tuple(candidates), lexical, gold, question.is_abstention,
                      delivery_detail=detail, delivery_include=include)


def covers(candidate, gold):
    return (candidate.source_ref == gold.source_ref and candidate.char_start <= gold.char_start
            and candidate.char_end >= gold.char_end
            and candidate.text[gold.char_start - candidate.char_start:gold.char_end - candidate.char_start] == gold.text)


def span_recall(candidates, gold):
    return sum(any(covers(c, span) for c in candidates) for span in gold) / len(gold) if gold else None


def ndcg(candidates, gold, k=10):
    if not gold:
        return None
    # One exact-support credit per source, avoiding duplicate-hit inflation.
    seen, relevance = set(), []
    for candidate in candidates[:k]:
        supported = {span.source_ref for span in gold if covers(candidate, span)} - seen
        relevance.append(int(bool(supported)))
        seen.update(supported)
    ideal_count = min(k, len({span.source_ref for span in gold}))
    denominator = sum(1 / math.log2(i + 2) for i in range(ideal_count))
    return sum(value / math.log2(i + 2) for i, value in enumerate(relevance)) / denominator


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * q) - 1)] if ordered else None


def paired_interval(deltas):
    if len(deltas) < 2:
        return None
    rng = random.Random(1010)
    samples = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
    return [samples[49], samples[1949]]


def evaluate(cases, scorers, *, split="holdout", timeout_s=1.0):
    validate_cases(cases)
    cases = tuple(c for c in cases if c.split == split)
    if len({c.conversation_id for c in cases}) != len(cases):
        raise ValueError("paired gate requires one case per conversation")
    rows = {"current": [], "lexical": [], **{name: [] for name in scorers}}
    for case in cases:
        base = score_candidates(case.query, case.candidates)
        lexical_start = time.monotonic()
        lexical_map = {c.candidate_id: c for c in case.candidates}
        lexical = tuple(lexical_map[key] for key in case.lexical_ids) + tuple(
            c for c in case.candidates if c.candidate_id not in case.lexical_ids)
        lexical_ms = (time.monotonic() - lexical_start) * 1000
        results = {"current": base, **{name: score_candidates(case.query, case.candidates, scorer, timeout_s)
                                     for name, scorer in scorers.items()}}
        for name in rows:
            result = results.get(name)
            ranked = result.candidates if result else lexical
            rows[name].append({"category": case.category,
                "candidate_recall_at_30": span_recall(case.candidates[:30], case.gold),
                "span_recall_at_5": span_recall(ranked[:5], case.gold),
                "ndcg_at_10": ndcg(ranked, case.gold),
                "noanswer_false_positive": int(bool(ranked)) if case.noanswer else None,
                "added_ms": result.elapsed_s * 1000 if result else lexical_ms,
                "total_ms": case.capture_ms + (result.elapsed_s * 1000 if result else lexical_ms),
                "status": result.status if result else "lexical"})
    report = {"cases": len(cases), "split": split, "answer_correctness": "UNMEASURED",
              "calibration": "UNMEASURED: no calibrated correctness probabilities claimed",
              "promotion_scope": "frozen corpus only; no default activation",
              "candidate_cap": 30, "production_hit_cap": 25, "arms": {}}
    report.update(delivery_configuration(cases))
    for name, samples in rows.items():
        means = {key: statistics.mean(v[key] for v in samples if v[key] is not None)
                 if any(v[key] is not None for v in samples) else None
                 for key in ("candidate_recall_at_30", "span_recall_at_5", "ndcg_at_10")}
        report["arms"][name] = {**means, "noanswer_false_positives": sum(v["noanswer_false_positive"] or 0 for v in samples),
            "added_latency_p95_ms": percentile([v["added_ms"] for v in samples if v["added_ms"] is not None], .95),
            "total_delivery_p95_ms": percentile([v["total_ms"] for v in samples], .95),
            "status_counts": dict(Counter(v["status"] for v in samples)),
            "exact_gold_cases": sum(v["span_recall_at_5"] is not None for v in samples)}
        if name not in scorers:
            continue
        paired = [(b, v) for b, v in zip(rows["current"], samples) if b["span_recall_at_5"] is not None]
        deltas = [v["span_recall_at_5"] - b["span_recall_at_5"] for b, v in paired]
        interval = paired_interval(deltas)
        categories = {c: statistics.mean(v["span_recall_at_5"] - b["span_recall_at_5"]
                                        for b, v in paired if b["category"] == c)
                      for c in {b["category"] for b, _ in paired}}
        complete = (bool(samples) and all(v["status"] == "scored" for v in samples)
                    and interval is not None and all(c.gold is not None for c in cases))
        promoted = (complete and statistics.mean(deltas) >= .03 and interval[0] > 0
                    and all(delta >= -.02 for delta in categories.values())
                    and report["arms"][name]["noanswer_false_positives"] <= report["arms"]["current"]["noanswer_false_positives"]
                    and report["arms"][name]["added_latency_p95_ms"] <= 1000)
        report["arms"][name].update({"paired_delta_95pct": interval, "category_deltas": categories,
            "promotion": "PROMOTE" if promoted else "NO_ADOPTION" if complete else "INSUFFICIENT_EVIDENCE"})
    return report
