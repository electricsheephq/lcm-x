"""C1b deterministic diagnostic verdicts; never replaces the historical scorer.

Rules and limitations: SPLIT-SCORING.md. No model, store, or network access.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scorer"))
from score_s import classify  # noqa: E402

RULE_VERSION = "c1b-terms-v2"
RULES = ("c1b-terms-v1", RULE_VERSION)
NUMERIC_CLASSES = {"limit", "tool_number"}


def contains(text, value):
    """Case-insensitive literal value with alphanumeric boundaries; no fuzzy IDs."""
    return bool(value) and bool(re.search(r"(?<![a-z0-9])" + re.escape(value.casefold()) + r"(?![a-z0-9])",
                                         (text or "").casefold()))


def identifier(fact):
    value, kind = fact["value"], fact["class"]
    patterns = {
        "limit": r"\d+", "tool_number": r"\d+", "error_with_fix": r"E_[a-f0-9]+",
        "pending_task": r"[a-f0-9]{12,}", "early_user_constraint": r"fixtures/[^\s]+",
        "file_change": r"src/[^\s:]+",
    }
    if kind in patterns:
        match = re.search(patterns[kind], value)
        if not match:
            raise ValueError(f"unsupported {kind} value: {fact['id']}")
        return match.group()
    return value.split(" because ", 1)[0]


def qualifiers(fact, answer):
    """Conservative, class-specific AND of documented terms/alias alternatives."""
    text = answer.casefold()
    groups = {
        "limit": [r"\bmib\b"],
        "tool_number": [r"\brows?\b|rows_verified", r"\bverified\b|rows_verified"],
        "error_with_fix": [r"\bcursor\b", r"\bmissing\b", r"\brebuild\w*\b", r"\bindex\w*\b"],
        "pending_task": [r"\bverif\w*\b", r"\brollback\b", r"\bbefore\b", r"\brelease\b"],
        "early_user_constraint": [r"\bnever\s+modify\b|\bdo not\s+modify\b|\bread[- ]only\b|\bpreserve\w*\b.*\bunchanged\b"],
        "file_change": [r"\bpreserv\w*\b", r"\btool_call_id\b", r"\breplay\b"],
    }
    required = groups.get(fact["class"], [])
    if fact["class"] in ("decision", "superseded_value") and " because " in fact["value"]:
        # The two fixture rationales are explicit generator vocabulary, not a judge.
        rationale = fact["value"].split(" because ", 1)[1]
        required = ([r"\bread[- ]only\b", r"\breplay\b", r"\bpreserv\w*\b", r"\bownership\b"]
                    if rationale == "read-only replay preserves ownership" else
                    [r"\b" + re.escape(w) + r"\b" for w in re.findall(r"\w+", rationale.casefold())])
    return all(re.search(pattern, text) for pattern in required)


def fact_verdicts(fact, answer, context, siblings, *, historical=None, rule=RULE_VERSION):
    """Missing evidence -> None; known wrong answer -> False; scalar N/A -> None.

    Binding is the recorded probe-id→fixture mapping. A bare correct identifier
    suffices for association; qualifiers are semantic's separate responsibility.
    Any conflicting sibling/stale identifier or explicit other fixture fails it.
    """
    if rule not in RULES:
        raise ValueError("unknown split scoring rule: " + rule)
    readable = isinstance(answer, str) and bool(answer.strip())
    ident = identifier(fact)
    bound = contains(answer, ident) if readable else None
    if readable:
        for other in siblings:
            if other["id"] != fact["id"]:
                other_ident = identifier(other)
                if (other_ident != ident and contains(answer, other_ident)) or contains(answer, other["fixture"]):
                    bound = False
        obsolete = [fact["stale"]] if fact.get("stale") else []
        if rule == "c1b-terms-v2":
            obsolete += [v["value"] for v in fact.get("versions", [])]
            obsolete += [v["value"] for v in fact.get("stale_history", [])]
            if fact.get("correction_source"):
                obsolete.append(fact["correction_source"]["value"])
        for value in obsolete:
            stale = identifier(dict(fact, value=value))
            if stale != ident and contains(answer, stale):
                bound = False
    return {
        "strict (historical)": historical if historical is not None else (
            classify(answer, fact["value"], False) == "CORRECT" if answer is not None else None),
        "verbatim": fact["value"] in context if context is not None else None,
        "semantic": bool(bound and qualifiers(fact, answer)) if readable else None,
        "scalar": contains(answer, ident) if readable and fact["class"] in NUMERIC_CLASSES else
            False if answer is not None and fact["class"] in NUMERIC_CLASSES else None,
        "association": bound,
    }


def request_ground_truth(rows, specs, policy=None):
    """deep-v1 generator policy: continuations/corrections do not revoke objective.

    Latest means latest *user row*, including corrections, not latest matching
    continuation. Supersession is scoped: each latest user row replaces the prior
    immediate instruction; fact corrections replace only that fact's value.
    An explicit initial.superseded_by row can retire the standing objective.
    """
    initial = next(p for p in specs if p["kind"] == "current_request")
    policy = policy or dict(initial_row_id=initial["row_id"], initial_superseded_by=None)
    if policy["initial_row_id"] != initial["row_id"]:
        raise ValueError("request ground-truth source mismatch")
    return dict(schema="deep-request-ground-truth-v1", probes=specs,
        initial=dict(row_index=initial["row_index"], row_id=initial["row_id"],
                     text=initial["answer"], superseded_by=policy["initial_superseded_by"]),
        latest_rule="latest user row, including fact corrections",
        supersession_policy="latest user instruction replaces prior immediate instruction only; initial objective remains operative",
        user_instructions=[dict(row_index=i, row_id=r["id"], text=r["content"],
                                supersedes_latest=next((j for j in range(i - 1, -1, -1) if rows[j]["role"] == "user"), None))
                           for i, r in enumerate(rows) if r["role"] == "user"])


def request_semantic(answer, expected):
    # Reordering/case/punctuation are allowed; every content term and digit is
    # required. This deliberately does not infer a seed/report from "continue".
    stop = {"the", "and", "a", "an", "of", "to", "in", "only"}
    def words(text):
        return set(re.findall(r"[a-z0-9]+", (text or "").casefold())) - stop
    wanted = words(expected)
    return bool(wanted) and wanted <= words(answer)


def score_event(receipt, context, row_index, truth):
    answers, specs = receipt.get("answers") or {}, truth["probes"]
    def answer_for(kind):
        p = next(p for p in specs if p["kind"] == kind)
        return answers.get(p["id"])
    answer = answer_for("current_request")
    initial = truth["initial"]
    revoked = initial.get("superseded_by")
    operative = row_index >= initial["row_index"] and (revoked is None or row_index < revoked["row_index"])
    present = initial["text"] in context if context is not None else None
    semantic = request_semantic(answer, initial["text"]) if isinstance(answer, str) else None
    retained = True if present is True or semantic is True else False if present is False and semantic is False else None
    latest = next((r for r in reversed(truth["user_instructions"]) if r["row_index"] <= row_index), None)
    result = {"R-initial": dict(source=initial, operative=operative, verbatim=present, semantic=semantic,
                                retained=retained if operative else None, answer_reused=True),
              "R-latest": dict(source=latest, reader_compliance=request_semantic(answer, latest["text"])
                               if latest and isinstance(answer, str) else None, answer_reused=True)}
    for label, kind in (("Q", "constraint_quote"), ("O", "constraint_obey")):
        p = next(p for p in specs if p["kind"] == kind)
        checks = [c for c in receipt.get("checks", []) if c["kind"] == kind]
        result[label] = dict(verbatim=p.get("verbatim", p["answer"]) in context if context is not None else None,
                            reader_compliance=checks[0].get("reader_pass") if len(checks) == 1 else None)
    return result
