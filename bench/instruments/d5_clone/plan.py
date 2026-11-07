"""Deterministic, synthetic D5 clone workload; selection reads ids and counts only."""

import hashlib
import json
from collections import Counter
from contextlib import closing

from .capture import _readonly

# Plain shell, so the host's dangerous-command check passes them: a flagged command (such as `python3 -c`) waits for
# an approval no one gives on the loopback platform and is blocked, so the tool steps would produce no output.
LARGE_COMMAND = "head -c 204800 /dev/zero | tr '\\000' d"
SMALL_COMMAND = "head -c 64 /dev/zero | tr '\\000' d"


class PlanError(ValueError):
    """The eligible session counts cannot fill the workload."""


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def choose_sessions(host_db, lcm_db, covered, *, min_large=1000, min_small=20, n_small=2):
    """The largest covered open session and the next *n_small* covered ones, by (-lcm_rows, sha256 of the id).
    covered() is the costly check (one host rebuild each), so sessions are taken in that order and checked only
    until n_small + 1 are covered: the same choice, and the same refusal, as checking every open session."""
    with closing(_readonly(host_db)) as host, closing(_readonly(lcm_db)) as lcm:
        rows = []
        for (sid,) in host.execute("SELECT id FROM sessions WHERE end_reason IS NULL"):
            counts = [db.execute("SELECT count(*) FROM messages WHERE session_id = ?",
                                 (sid,)).fetchone()[0] for db in (lcm, host)]
            rows.append(dict(session_id=sid, lcm_rows=counts[0], host_rows=counts[1]))
    rows.sort(key=lambda row: (-row["lcm_rows"], _digest(row["session_id"])))
    candidates, checked = [], 0
    for row in rows:
        if len(candidates) == n_small + 1:
            break
        checked += 1
        if covered(row["session_id"]):
            candidates.append(row)
    large = sum(row["lcm_rows"] >= min_large for row in candidates[:1])
    small = sum(row["lcm_rows"] >= min_small for row in candidates[1:])
    if not large or small < n_small:
        raise PlanError(f"covered={len(candidates)} of checked={checked} large={large} small={small} needed={n_small}")
    return [dict(row, name=name, chat_id=f"d5-{name}") for row, name in
            zip(candidates[:n_small + 1], ["L"] + [f"S{i + 1}" for i in range(n_small)])]


def build_plan(conversations, *, epochs=4, warm_turns=2, nonce):
    conversations = sorted((dict(c) for c in conversations),
                           key=lambda c: 0 if c["name"] == "L" else int(c["name"][1:]))
    plan = {"schema": "d5-clone-plan/1", "conversations": conversations, "epochs": [],
            "provider": {"steps": {}, "cron": [{"text": f"d5 cron reply {_digest(nonce)[:12]}"}]},
            "cron": {"schedule": "every 2m", "repeat": 1, "created_before_epoch": 2,
                     "prompt": f"d5 cron {_digest(nonce)[:12]}"}}
    for e in range(epochs):
        epoch = dict(epoch=e, restart_before=e > 0, min_seconds=180 if e == 2 else 0, steps=[])
        for conversation in conversations:
            name = conversation["name"]
            for k in range(warm_turns + 1):
                sid = f"e{e}-{name}-{k}"
                digest = _digest(nonce + sid)
                rewrite = "whitespace" if e in (0, 2) and k == 1 else None
                tool = ("large" if k == 1 else "small") if (
                    e in (1, 2) and name == "L" and k in (1, 2)) else None
                epoch["steps"].append(dict(
                    id=sid, conversation=name, kind="cold" if k == 0 else "warm",
                    user_text=f"d5 plan v1 {sid} {digest[:16]}" + ("  \r\n" if rewrite else ""),
                    rewrite=rewrite, tools=tool))
                entries = []
                if tool:
                    entries.append({"tool_calls": [{"id": f"call_{sid}_1", "name": "terminal",
                                    "arguments": {"command": LARGE_COMMAND if tool == "large"
                                                  else SMALL_COMMAND}}]})
                plan["provider"]["steps"][sid] = entries + [{"text": f"d5 reply {sid} {digest[:12]}"}]
        plan["epochs"].append(epoch)
    return plan


def plan_sha256(plan):
    return _digest(json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def declaration(plan):
    steps = [step for epoch in plan["epochs"] for step in epoch["steps"]]
    return {"conversations": [{**{key: c[key] for key in ("name", "lcm_rows", "host_rows")},
                               "flagged_summary_nodes": c.get("flagged_summary_nodes")}
                              for c in plan["conversations"]],
            "steps_by_kind": dict(Counter(s["kind"] for s in steps)),
            "cold_binds": dict(Counter(s["conversation"] for s in steps if s["kind"] == "cold")),
            "rewrite_steps": sum(s["rewrite"] is not None for s in steps),
            "tool_steps": sum(s["tools"] is not None for s in steps),
            "cron": {k: v for k, v in plan["cron"].items() if k != "prompt"},
            "plan_sha256": plan_sha256(plan)}
