"""Evidence and assertion-state handlers behind the public tools facade."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict

# Resolve shared helpers/constants at call time: tools also imports this family.
from . import tools as _tools
from .assertion_state import query_assertion_state
from .assertion_store import ASSERTION_KINDS
from .reasoning import (
    compile_evidence_plan,
    execute_plan,
    ground_evidence,
    question_date_as_of_epoch,
    validate_selector_alignment,
    verify_final_answer,
)


def _shape_assertion_state_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "assertion_id": row["assertion_id"],
        "subject_key": row["subject_key"],
        "predicate_key": row["predicate_key"],
        "object_value": row["object_value"],
        "value_text": row["value_text"],
        "kind": row["kind"],
        "polarity": row["polarity"],
        "strength": row["strength"],
        "scope_key": row["scope_key"],
        "speaker_role": row["speaker_role"],
        "observed_at": row["observed_at"],
        "event_at": row["event_at"],
        "valid_from": row["valid_from"],
        "valid_to": row["valid_to"],
        "confidence": row["confidence"],
        "active": row["active"],
        "lifecycle_status": (
            list(row["lifecycle_status"])
            if row["lifecycle_status"] is not None
            else None
        ),
        "unresolved_conflict": row["unresolved_conflict"],
        "attribution": row["attribution"],
        "semantic_state": row["semantic_state"],
        "source_ref": {
            "store_id": row["source_store_id"],
            "session_id": row["source_session_id"],
            "source": row["source_name"],
            "role": row["source_role"],
            "span_start": row["source_span_start"],
            "span_end": row["source_span_end"],
            "quote": row["source_quote"],
            "content_sha256": row["source_content_sha256"],
        },
    }


def _shape_assertion_state_relation(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "relation_id": row["relation_id"],
        "relation_type": row["relation_type"],
        "from_assertion_id": row["from_assertion_id"],
        "to_assertion_id": row["to_assertion_id"],
        "confidence": row["confidence"],
        "source_ref": {
            "store_id": row["source_store_id"],
            "session_id": row["source_session_id"],
            "span_start": row["source_span_start"],
            "span_end": row["source_span_end"],
            "quote": row["source_quote"],
            "content_sha256": row["source_content_sha256"],
        },
    }


def lcm_query_state(args: Dict[str, Any], **kwargs: Any) -> str:
    """Return bounded typed assertion state with exact source provenance."""
    engine = _tools._require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})
    store = getattr(engine, "_assertions", None)
    if store is None:
        return json.dumps({
            "status": "disabled",
            "error": "V4 assertions are not enabled for this profile",
        })

    subject_key = str(args.get("subject_key") or "").strip()
    if not subject_key:
        return json.dumps({"error": "subject_key is required"})
    predicate_key = str(args.get("predicate_key") or "").strip() or None
    scope_key = None
    if "scope_key" in args:
        scope_key = str(args.get("scope_key") or "").strip()

    raw_kinds = args.get("kinds")
    kinds: list[str] | None = None
    if raw_kinds is not None and raw_kinds != []:  # [] means "not provided" (#1194)
        if not isinstance(raw_kinds, list) or not raw_kinds:
            return json.dumps({"error": "kinds must be a non-empty array"})
        kinds = [str(value or "").strip().lower() for value in raw_kinds]
        if len(kinds) > len(ASSERTION_KINDS) or any(
            value not in ASSERTION_KINDS for value in kinds
        ):
            return json.dumps({"error": "kinds contains an unsupported assertion kind"})

    speaker_role, role_error = _tools._parse_grep_role(args.get("speaker_role"))
    if role_error:
        return json.dumps({"error": role_error.replace("role", "speaker_role", 1)})
    as_of, as_of_error = _tools._parse_optional_timestamp(args.get("as_of"), "as_of")
    if as_of_error:
        return json.dumps({"error": as_of_error})
    parsed_limit, limit_error = _tools._parse_strict_int(
        args.get("limit", _tools._LCM_QUERY_STATE_DEFAULT_LIMIT),
        "limit",
    )
    if limit_error:
        return json.dumps({"error": limit_error})
    if parsed_limit is None or parsed_limit <= 0:
        return json.dumps({"error": "limit must be a positive integer"})
    requested_limit = parsed_limit
    limit = min(requested_limit, _tools._LCM_QUERY_STATE_LIMIT_CAP)

    try:
        result = query_assertion_state(
            store,
            subject_key=subject_key,
            predicate_key=predicate_key,
            kinds=kinds,
            scope_key=scope_key,
            speaker_role=speaker_role,
            as_of=as_of,
            limit=limit,
        )
    except (TypeError, ValueError, sqlite3.Error) as exc:
        return json.dumps({"error": f"state query failed: {exc}"})

    assertions = [_shape_assertion_state_row(row) for row in result.assertions]
    relations = [_shape_assertion_state_relation(row) for row in result.relations]
    response: dict[str, Any] = {
        "status": "ok",
        "query": {
            "subject_key": subject_key,
            "predicate_key": predicate_key,
            "kinds": kinds,
            "scope_key": scope_key,
            "speaker_role": speaker_role,
            "as_of": as_of,
        },
        "limit": limit,
        "assertions": assertions,
        "relations": relations,
        "active_assertion_ids": (
            list(result.active_assertion_ids)
            if result.active_assertion_ids is not None
            else None
        ),
        "conflict_assertion_ids": list(result.conflict_assertion_ids),
        "assertions_truncated": result.assertions_truncated,
        "relations_truncated": result.relations_truncated,
        "response_truncated": False,
        "response_char_cap": _tools._LCM_QUERY_STATE_RESPONSE_CHAR_CAP,
        "provenance": {
            "store": "same_profile_lcm.db",
            "evidence": "exact_source_spans",
            "recency_resolution": "disabled",
        },
    }
    if requested_limit > _tools._LCM_QUERY_STATE_LIMIT_CAP:
        response["limit_clamped_from"] = requested_limit

    omitted = 0
    while assertions:
        encoded = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) <= _tools._LCM_QUERY_STATE_RESPONSE_CHAR_CAP:
            return encoded
        assertions.pop()
        omitted += 1
        retained_ids = {row["assertion_id"] for row in assertions}
        relations[:] = [
            relation
            for relation in relations
            if relation["from_assertion_id"] in retained_ids
            or relation["to_assertion_id"] in retained_ids
        ]
        if response["active_assertion_ids"] is not None:
            response["active_assertion_ids"] = [
                value
                for value in response["active_assertion_ids"]
                if value in retained_ids
            ]
        response["conflict_assertion_ids"] = [
            value for value in response["conflict_assertion_ids"] if value in retained_ids
        ]
        response["response_truncated"] = True
        response["assertions_truncated"] = True
        response["assertions_omitted_by_response_cap"] = omitted
    return json.dumps(response, ensure_ascii=False, separators=(",", ":"))


def _compute_stage(
    transport: str,
    started_at: float,
    *,
    provider: str,
    model: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "transport": transport,
        "provider": provider,
        "model": model,
        "latency_ms": round((time.perf_counter() - started_at) * 1_000.0, 3),
        **details,
    }


def lcm_compute(args: Dict[str, Any], **kwargs: Any) -> str:
    """Run a pure operation over exact, selector-supplied evidence refs."""
    total_started = time.perf_counter()
    engine = _tools._require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    question = str(args.get("question") or "").strip()
    question_date = args.get("question_date")
    stages: dict[str, Any] = {}
    planner_started = time.perf_counter()
    as_of: float | None = None
    if question_date is not None:
        as_of = question_date_as_of_epoch(question_date)
        if as_of is None:
            stages["planner"] = _compute_stage(
                "deterministic_local",
                planner_started,
                provider="none",
                model="none",
                status="fallback",
            )
            return json.dumps({
                "status": "fallback",
                "reason": "question_date must be a valid timezone-unambiguous ISO date",
                "next_path": "evidence_only",
                "provenance": {"stages": stages},
                "metrics": {
                    "total_latency_ms": round(
                        (time.perf_counter() - total_started) * 1_000.0, 3
                    )
                },
            })

    plan_decision = compile_evidence_plan(question, question_date)
    stages["planner"] = _compute_stage(
        "deterministic_local",
        planner_started,
        provider="none",
        model="none",
        status=plan_decision.status,
    )
    if plan_decision.status != "planned" or plan_decision.plan is None:
        return json.dumps({
            "status": plan_decision.status,
            "reason": plan_decision.reason,
            "next_path": "evidence_only",
            "provenance": {"stages": stages},
            "metrics": {
                "total_latency_ms": round(
                    (time.perf_counter() - total_started) * 1_000.0, 3
                )
            },
        })
    plan = plan_decision.plan

    selector_started = time.perf_counter()
    if plan.requires_complete_evidence and args.get("evidence_complete") is not True:
        stages["selector"] = _compute_stage(
            "host_tool_arguments",
            selector_started,
            provider="unknown_to_plugin",
            model="unknown_to_plugin",
            status="fallback",
            evidence_complete=False,
        )
        return json.dumps({
            "status": "fallback",
            "reason": "operation requires explicit evidence_complete=true",
            "plan": plan.as_dict(),
            "next_path": "evidence_only",
            "provenance": {"stages": stages},
            "metrics": {
                "total_latency_ms": round(
                    (time.perf_counter() - total_started) * 1_000.0, 3
                )
            },
        })

    raw_operands = args.get("operands")
    try:
        grounding = ground_evidence(
            raw_operands,
            messages=engine._store,
            assertions=getattr(engine, "_assertions", None),
            as_of=as_of,
            session_dates=getattr(engine, "_session_occurrence_dates", None),
        )
    except (TypeError, ValueError, sqlite3.Error) as exc:
        grounding = None
        grounding_reason = f"selector validation failed: {exc}"
    else:
        grounding_reason = grounding.reason  # type: ignore[union-attr]
    if grounding is None or grounding.status != "grounded":
        stages["selector"] = _compute_stage(
            "host_tool_arguments",
            selector_started,
            provider="unknown_to_plugin",
            model="unknown_to_plugin",
            status="fallback",
            evidence_complete=bool(args.get("evidence_complete") is True),
            operand_count=(len(raw_operands) if isinstance(raw_operands, list) else 0),
        )
        return json.dumps({
            "status": "fallback",
            "reason": grounding_reason,
            "plan": plan.as_dict(),
            "next_path": "evidence_only",
            "provenance": {"stages": stages},
            "metrics": {
                "total_latency_ms": round(
                    (time.perf_counter() - total_started) * 1_000.0, 3
                )
            },
        })
    alignment_error = validate_selector_alignment(question, plan, grounding.operands)
    if alignment_error:
        stages["selector"] = _compute_stage(
            "host_tool_arguments",
            selector_started,
            provider="unknown_to_plugin",
            model="unknown_to_plugin",
            status="fallback",
            evidence_complete=bool(args.get("evidence_complete") is True),
            operand_count=len(grounding.operands),
        )
        return json.dumps({
            "status": "fallback",
            "reason": alignment_error,
            "plan": plan.as_dict(),
            "next_path": "evidence_only",
            "provenance": {"stages": stages},
            "metrics": {
                "total_latency_ms": round(
                    (time.perf_counter() - total_started) * 1_000.0, 3
                )
            },
        })
    stages["selector"] = _compute_stage(
        "host_tool_arguments",
        selector_started,
        provider="unknown_to_plugin",
        model="unknown_to_plugin",
        status="validated",
        evidence_complete=bool(args.get("evidence_complete") is True),
        operand_count=len(grounding.operands),
    )

    executor_started = time.perf_counter()
    computed = execute_plan(plan, grounding.operands)
    stages["executor"] = _compute_stage(
        "deterministic_local",
        executor_started,
        provider="none",
        model="none",
        status=computed.status,
        operation=plan.operation,
    )
    if computed.status != "computed" or computed.trace is None:
        return json.dumps({
            "status": "fallback",
            "reason": computed.reason,
            "plan": plan.as_dict(),
            "next_path": "evidence_only",
            "provenance": {"stages": stages},
            "metrics": {
                "operand_count": len(grounding.operands),
                "total_latency_ms": round(
                    (time.perf_counter() - total_started) * 1_000.0, 3
                ),
            },
        })

    trace = computed.trace
    candidate_present = "candidate_answer" in args
    candidate = str(args.get("candidate_answer") or "")
    verification_payload: dict[str, Any]
    answer = trace.answer
    candidate_used = False
    if candidate_present:
        verifier_started = time.perf_counter()
        verification = verify_final_answer(candidate, trace)
        stages["verifier"] = _compute_stage(
            "deterministic_local",
            verifier_started,
            provider="none",
            model="none",
            status=verification.status,
        )
        verification_payload = {
            "status": verification.status,
            "reason": verification.reason,
        }
        if verification.status == "verified":
            answer = candidate.strip()
            candidate_used = True
    else:
        verification_payload = {"status": "not_requested", "reason": ""}

    final_started = time.perf_counter()
    if candidate_used:
        final_stage = {
            "transport": "host_supplied_candidate",
            "provider": "unknown_to_plugin",
            "model": "unknown_to_plugin",
        }
    else:
        final_stage = {
            "transport": "deterministic_canonical",
            "provider": "none",
            "model": "none",
        }
    stages["final_answerer"] = {
        **final_stage,
        "latency_ms": round((time.perf_counter() - final_started) * 1_000.0, 3),
        "candidate_used": candidate_used,
    }
    response = {
        "status": "computed",
        "plan": plan.as_dict(),
        "trace": trace.as_dict(),
        "answer": answer,
        "candidate_verification": verification_payload,
        "provenance": {
            "runtime_inputs": ["question", "question_date", "exact_retrieved_evidence"],
            "stages": stages,
        },
        "metrics": {
            "operand_count": len(grounding.operands),
            "answer_chars": len(answer),
            "total_latency_ms": round(
                (time.perf_counter() - total_started) * 1_000.0, 3
            ),
        },
        "response_char_cap": _tools._LCM_COMPUTE_RESPONSE_CHAR_CAP,
    }
    encoded = json.dumps(response, ensure_ascii=False)
    if len(encoded) > _tools._LCM_COMPUTE_RESPONSE_CHAR_CAP:
        return json.dumps({
            "status": "fallback",
            "reason": "deterministic response exceeded its bounded response cap",
            "next_path": "evidence_only",
            "provenance": {"stages": stages},
        })
    return encoded


def lcm_evidence_pack(args: Dict[str, Any], **kwargs: Any) -> str:
    """Build a bounded exact-evidence packet and optional canonical trace."""
    engine = _tools._require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})
    # Lazy import preserves the plugin's order-independent module bootstrap.
    from .evidence_pack import build_evidence_pack
    return build_evidence_pack(
        args,
        engine=engine,
        retrieve=lambda recall_args: _tools.lcm_recall(recall_args, engine=engine),
    )


def lcm_compile_evidence(args: Dict[str, Any], **kwargs: Any) -> str:
    """Compile evidence through legacy proposal or deterministic auto mode."""
    engine = _tools._require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})
    # Lazy import preserves the plugin's order-independent module bootstrap.
    from .evidence_compiler import compile_evidence, compile_preanswer_evidence

    mode = str(args.get("mode") or "proposal").strip().casefold()
    if mode not in {"proposal", "auto"}:
        return json.dumps({"error": "mode must be one of: proposal, auto"})
    if mode == "auto":
        result = compile_preanswer_evidence(  # type: ignore[no-untyped-call]
            args.get("question"),
            engine=engine,
            baseline_refs=args.get("baseline_refs") or (),
            question_as_of=args.get("question_date"),
            retrieve=lambda recall_args: _tools.lcm_recall(recall_args, engine=engine),
            enabled=True,
            budgets=args.get("budgets"),
        )
        return json.dumps(result, ensure_ascii=False)

    proposal = args.get("proposal")
    result = compile_evidence(
        args.get("question"),
        engine=engine,
        baseline_refs=args.get("baseline_refs") or (),
        question_date=args.get("question_date"),
        selector=lambda _request: proposal,
        retrieve=lambda recall_args: _tools.lcm_recall(recall_args, engine=engine),
        enabled=True,
        persist_view=args.get("persist_view") is True,
        budgets=args.get("budgets"),
    )
    return json.dumps(result, ensure_ascii=False)


def lcm_retrieve(args: Dict[str, Any], **kwargs: Any) -> str:
    """Drive one bounded retrieval episode inside the answerer's tool turn."""
    engine = _tools._require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})
    controller = getattr(engine, "_adaptive_retrieval", None)
    if controller is None:
        return json.dumps({
            "status": "disabled",
            "reason": "adaptive retrieval is disabled",
            "enable_with": "LCM_ADAPTIVE_RETRIEVAL_ENABLED=true",
            "provenance": {
                "controller": {
                    "transport": "deterministic_local",
                    "provider": "none",
                    "model": "none",
                }
            },
        })
    if not isinstance(args, dict):
        return json.dumps({"status": "error", "error": "arguments must be an object"})
    unknown = set(args) - _tools._LCM_RETRIEVE_ARGUMENTS
    if unknown:
        return json.dumps({
            "status": "error",
            "error": (
                "unsupported lcm_retrieve arguments: "
                + ", ".join(sorted(str(value) for value in unknown))
            ),
        })

    action = str(args.get("action") or "").strip().casefold()
    def dispatch(name: str, payload: dict[str, Any]) -> str:
        return engine.handle_tool_call(name, payload)
    try:
        if action == "start":
            result = controller.start(
                question=args.get("question"),
                question_date=args.get("question_date"),
                identity=args.get("identity"),
                requirements=args.get("requirements"),
                engine=engine,
            )
        elif action == "search":
            result = controller.search(
                retrieval_id=args.get("retrieval_id"),
                missing_slot=args.get("missing_slot"),
                tool=args.get("tool"),
                tool_args=args.get("tool_args"),
                resolved_slots=args.get("resolved_slots"),
                engine=engine,
                dispatch=dispatch,
            )
        elif action == "finish":
            result = controller.finish(
                retrieval_id=args.get("retrieval_id"),
                resolved_slots=args.get("resolved_slots"),
                selected_refs=args.get("selected_refs"),
                computation=args.get("computation"),
                engine=engine,
                dispatch=dispatch,
            )
        elif action == "status":
            result = controller.status(
                retrieval_id=args.get("retrieval_id"), engine=engine
            )
        elif action == "abandon":
            result = controller.abandon(
                retrieval_id=args.get("retrieval_id"), engine=engine
            )
        else:
            raise ValueError(
                "action must be one of: start, search, finish, status, abandon"
            )
    except (TypeError, ValueError, sqlite3.Error) as exc:
        result = {
            "status": "error",
            "error": str(exc)[:1_000],
            "provenance": {
                "controller": {
                    "transport": "deterministic_local",
                    "provider": "none",
                    "model": "none",
                }
            },
        }
    encoded = json.dumps(result, ensure_ascii=False)
    if len(encoded) > _tools._LCM_RETRIEVE_RESPONSE_CHAR_CAP:
        return json.dumps({
            "status": "error",
            "error": "adaptive retrieval response exceeded its bounded response cap",
            "response_char_cap": _tools._LCM_RETRIEVE_RESPONSE_CHAR_CAP,
        })
    return encoded
