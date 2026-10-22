"""Blinded, independent judgments using verbatim upstream templates."""
import copy
import json
import re

from .models import JUDGES, BudgetStopped
from .prompts import JUDGE_PROMPT, JUDGE_SYSTEM_PROMPT, get_anscheck_prompt

# OmniMemEval scripts/utils/nlp_metrics.py extract_label_json (fenced or not): the JSON
# "label" field is authoritative only for valid JSON and an exact label value.
# Otherwise accept an exact final line, with no other label word in capitals.
# Anything else is ambiguous and retried. Starts are disjoint runs, so matching is linear.
JSON_OBJECT = re.compile(r"\{[^{}]*\}")
LABEL_FIELD = re.compile(r'"label"\s*:\s*["\']([^"\']*)["\']')
FINAL_LABEL = re.compile(r"(CORRECT|WRONG)", re.IGNORECASE)
LABEL_WORD = re.compile(r"\b(CORRECT|WRONG)\b")
LEADING_YES_NO = re.compile(r"[\s*_`\"']*(yes|no)\b", re.IGNORECASE)


def prompt(q, hypothesis):
    if q.dataset.startswith("lme"):
        return get_anscheck_prompt(q.qtype, q.question, q.gold, hypothesis, abstention=q.is_abstention)
    return JUDGE_PROMPT.format(question=q.question, golden_answer=q.gold, response=hypothesis)


def messages(q, hypothesis):
    user = [{"role": "user", "content": prompt(q, hypothesis)}]
    return user if q.dataset.startswith("lme") else [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, *user]


def judge_messages(q, hypothesis):
    if q.dataset == "locomo" and getattr(q, "is_abstention", False):
        # Existing FMP/LME abstention rubric; cat5's adversarial_answer is never a gold answer.
        abstention = copy.copy(q)
        abstention.dataset, abstention.qtype, abstention.gold = "lme-s", "single-session-user", ""
        return messages(abstention, hypothesis), "lme-s"
    return messages(q, hypothesis), q.dataset


def parse(text, dataset):
    if dataset.startswith("lme"):
        lines = text.strip().splitlines() or [""]
        match = LEADING_YES_NO.match(lines[0]) or LEADING_YES_NO.fullmatch(lines[-1])
        if not match or {w.lower() for w in re.findall(r"\b(yes|no)\b", text, re.IGNORECASE)} - {match.group(1).lower()}:
            return None
        return match.group(1).lower() == "yes"
    labels = set()
    for block in JSON_OBJECT.findall(text):
        try:
            value = json.loads(block)
        except ValueError:
            return None
        if isinstance(value, dict) and "label" in value:
            labels.add(str(value["label"]).lower())
    if labels:
        return {"correct": True, "wrong": False}.get(labels.pop()) if len(labels) == 1 else None
    match = FINAL_LABEL.fullmatch((text.strip().splitlines() or [""])[-1])
    if not match or set(LABEL_WORD.findall(text)) - {match.group(1).upper()}:
        return None
    return match.group(1).upper() == "CORRECT"


def judge(q, hypothesis, client, records=None, models=JUDGES, agreement_model=None):
    # BudgetStopped propagates: a spend stop is a run state, not a verdict.
    results, records = {}, records if records is not None else []
    jm, dataset = judge_messages(q, hypothesis)
    for model in dict.fromkeys([*models, *([agreement_model] if agreement_model else [])]):
        verdict, raw, error = None, [], None
        for attempt in range(2):
            try:
                message = client.call(model, jm, records, temperature=None if dataset.startswith("lme") else 0)
            except BudgetStopped as exc:
                if model in models:
                    exc.partial = dict(results)
                    raise
                error = "budget_stopped"  # only the optional agreement judge can stop without voiding the row
                break
            except Exception as exc:
                error = type(exc).__name__
                break
            raw.append(message.get("content") or "")
            verdict = parse(raw[-1], dataset)
            if verdict is not None:
                break
        status = "ok" if verdict is not None else "not_run" if error == "budget_stopped" else "judge_error"
        results[model] = {"status": status, "correct": verdict, "verdict": "valid" if verdict is not None else "invalid",
                          "raw": raw, **({"error": error} if error else {})}
    return results, records


def judge_row(row, q, client, models=JUDGES, agreement_model=None):
    # Only answered rows are judged: errors score incorrect unjudged; skips and not_run are excluded.
    if row["status"] != "ok":
        return row
    records = []
    try:
        result, _ = judge(q, row["hypothesis"], client, records, models, agreement_model)
    except BudgetStopped as exc:
        return {**row, "status": "not_run", "reason": "budget_stopped_at_judge", "judge_calls": records,
                "judges_partial": exc.partial}
    return {**row, "judges": result, "judge_calls": records}
