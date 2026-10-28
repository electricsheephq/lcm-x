"""Differential and absolute checks return evidence, not tuned thresholds."""
import re

from .ACCEPTED_DELTAS import ACCEPTED_DELTAS
from .trees import DEFAULT_BASELINE


PROJECTED = re.compile(r"\[LCM survival fit: this user message \((\d+) tokens\) is stored verbatim as store id "
                       r"(\d+); lcm_expand / lcm_grep reach the full text\. \(projection (\d+)/(\d+)\)\]")


def projection_user_errors(case, cand):
    """Match user rows in order; a mark must reconstruct this input row exactly."""
    output = cand.get("out", [])
    cursor, errors, matched = 0, [], set()
    for index, entry in enumerate(case.get("rows", [])):
        source = entry["message"]
        if source.get("role") != "user":
            continue
        evidence = cand.get("user_sources", {}).get(str(index), {})

        def matches(row):
            if row == source:
                return True
            content = row.get("content")
            if row.get("role") != "user" or not isinstance(content, str):
                return False
            for match in PROJECTED.finditer(content):
                n_tokens, store_id, head, tail = map(int, match.groups())
                if store_id != evidence.get("store_id") or n_tokens != evidence.get("tokens"):
                    continue
                text, marker = evidence["text"], match[0]
                if len(text) > 2 * head:
                    projected = f"{text[:head]}\n...\n{marker}\n...\n{text[-tail:]}"
                elif len(text) > head:
                    projected = f"{text[:head]}\n...\n{marker}"
                else:
                    projected = text
                # The historical short-row form is a mark only.
                forms = [projected] + ([marker] if text and len(text) <= 2 * head else [])
                if any(row == {**source, "content": form} for form in forms):
                    return True
            return False

        found = next((i for i in range(cursor, len(output)) if matches(output[i])), None)
        if found is None:
            errors.append(index)
        else:
            matched.add(found)
            cursor = found + 1
    extra = [i for i, row in enumerate(output) if row.get("role") == "user" and i not in matched]
    return dict(missing=errors, extra=extra) if errors or extra else {}


def compare(case, ga, cand, baseline=DEFAULT_BASELINE):
    failures = []

    def check(oracle, condition, detail):
        if not condition:
            failures.append(dict(oracle=oracle, detail=detail, delta=None))

    if "error" in ga or "error" in cand:
        return [dict(oracle="DRIVER", detail={"ga": ga.get("error"), "candidate": cand.get("error")}, delta=None)]
    if case["layer"] == "projection":
        check("R1", not (ga["out_tokens"] <= case["limit"] < cand["out_tokens"]),
              dict(ga=ga["out_tokens"], candidate=cand["out_tokens"], limit=case["limit"]))
        marks = (cand["user_marks"] if baseline == "v0.26.2"
                 else sorted(set(cand["user_marks"]) - set(ga["user_marks"])))
        check("P_USER", not marks, marks)
        invalid_users = projection_user_errors(case, cand)
        check("P_USER_ROWS", not invalid_users, invalid_users)
        check("P_INPUT", not cand["input_mutated"] and not ga["input_mutated"], "input mutated")
    else:
        check("CALLS", len(ga["calls"]) == len(cand["calls"]) == len(case["steps"]), "model call count differs")
        for i, (a, b) in enumerate(zip(ga["calls"], cand["calls"])):
            check("R1", not (a["tokens"] <= case["window"] < b["tokens"]),
                  dict(call=i, ga=a["tokens"], candidate=b["tokens"], window=case["window"]))
            for sentinel in ("⟦objective⟧", "⟦newest⟧"):
                check("R2", sentinel not in a["sentinels"] or sentinel in b["sentinels"], dict(call=i, sentinel=sentinel))
            check("A2", not b["pair_errors"], dict(call=i, errors=b["pair_errors"]))
        for metric in ("compress_runs", "summarizer_calls"):
            a = sum(call[metric] for call in ga["calls"])
            b = sum(call[metric] for call in cand["calls"])
            check("R3", b <= a * 1.10 + 2, dict(metric=metric, ga=a, candidate=b))
        check("C1", cand["uncovered_rows"] <= ga["uncovered_rows"] + 2,
              dict(ga=ga["uncovered_rows"], candidate=cand["uncovered_rows"]))
        lost = sorted(set(cand["produced"]) - set(cand["stored"]))
        check("A1", not lost, lost)
    for failure in failures:
        for delta in ACCEPTED_DELTAS:
            if (baseline in delta.baselines and case["layer"] in delta.layer
                    and failure["oracle"] == delta.oracle and delta.predicate(case, ga, cand, failure)):
                failure["delta"] = delta.id
                break
    return failures
