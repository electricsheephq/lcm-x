"""Deterministic host lists built from shape facts only (counts, roles, lengths, tool-pair patterns, id presence).

No real content: every text is fixed filler words plus a serial. The same kit (seed and facts) always gives the
same lists, so `input_hash` (sha256 over canonical JSON of the kit and every generated list) pins the input.

A conversation's stored history is a run of turns. A turn is a user row, then 0..k assistant rows with tool calls,
each followed by its tool result, then a final assistant reply. The target conversation's host list keeps every
turn of its newest part and one turn in `hidden_keep_every` of its oldest part: the dropped turns are stored but
hidden from the host, as after host-side compactions and trimmed lists at restarts (#2868, #2869).
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau "
         "upsilon phi chi psi omega").split()
CHARS_PER_TOKEN = 4


def text(rng: random.Random, tokens: int, tag: str) -> str:
    words, size, target = [], 0, max(1, tokens) * CHARS_PER_TOKEN
    while size < target:
        word = WORDS[rng.randrange(len(WORDS))]
        words.append(word)
        size += len(word) + 1
    return f"{tag} " + " ".join(words)


def _length(rng: random.Random, spec: dict[str, Any]) -> int:
    """Tokens from a [low, high] uniform range, with a `large_share` drawn from [large_low, large_high]."""
    if spec.get("large_share") and rng.random() < spec["large_share"]:
        return rng.randint(*spec["large"])
    return rng.randint(*spec["normal"])


def conversation(kit: dict[str, Any], name: str, turns: int, seed: int) -> list[dict[str, Any]]:
    """The full stored history of one conversation, oldest first."""
    rng = random.Random(seed)
    lengths, tools = kit["lengths"], kit["tools"]
    if not name.startswith("target"):
        lengths = {**lengths, **kit["others"].get("lengths", {})}
    rows: list[dict[str, Any]] = []
    clock = kit["start_epoch"] + seed % 997

    def stamp(row: dict[str, Any]) -> dict[str, Any]:
        nonlocal clock
        clock += rng.uniform(1.0, 30.0)
        if rng.random() < kit["stamp_share"]:
            row["timestamp"] = round(clock, 3)
        if rng.random() < kit["uid_share"]:
            row["message_uid"] = hashlib.sha256(f"{name}:{len(rows)}:{seed}".encode()).hexdigest()[:32]
        rows.append(row)
        return row

    for number in range(turns):
        tag = f"{name} t{number}"
        stamp({"role": "user", "content": text(rng, _length(rng, lengths["user"]), tag + " user")})
        for call in range(rng.choices(range(len(tools["calls_per_turn_weights"])),
                                      weights=tools["calls_per_turn_weights"])[0]):
            call_id = f"call_{hashlib.sha256(f'{tag}:{call}'.encode()).hexdigest()[:20]}"
            tool = tools["names"][rng.randrange(len(tools["names"]))]
            stamp({"role": "assistant", "content": text(rng, _length(rng, lengths["assistant_call"]), tag + " plan"),
                   "tool_calls": [{"id": call_id, "type": "function",
                                   "function": {"name": tool, "arguments": json.dumps({"q": f"{tag}:{call}"})}}]})
            result = {"role": "tool", "tool_call_id": call_id,
                      "content": text(rng, _length(rng, lengths["tool_result"]), tag + f" result {call}")}
            # #2886 shape: a share of tool rows carry `name` and no `tool_name`.
            result["name" if rng.random() < tools["name_only_share"] else "tool_name"] = tool
            stamp(result)
        stamp({"role": "assistant", "content": text(rng, _length(rng, lengths["assistant"]), tag + " reply")})
    return rows


def turn_starts(rows: list[dict[str, Any]]) -> list[int]:
    return [i for i, row in enumerate(rows) if row["role"] == "user"]


def host_list(kit: dict[str, Any], history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The target's host list: one turn in `hidden_keep_every` of the oldest `hidden_turn_share`, then every turn."""
    target = kit["target"]
    starts = turn_starts(history) + [len(history)]
    oldest = int((len(starts) - 1) * target["hidden_turn_share"])
    # The store holds the newest `stored_tail_turns` turns, which the host's list ends before (a turn the plugin
    # stored and the host did not keep, as an interrupted turn leaves it): the restart walk finds no match at the end.
    last = len(starts) - 1 - int(target.get("stored_tail_turns", 0))
    kept: list[dict[str, Any]] = []
    for index in range(last):
        if index >= oldest or index % target["hidden_keep_every"] == 0:
            kept.extend(history[starts[index]:starts[index + 1]])
    return kept


def restart_view(kit: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The host list as the host hands it after a restart, rebuilt cold from its own records: assistant and tool
    rows without timestamps (#1674) and tool rows naming their tool in `name`, not `tool_name` (#2886)."""
    form = kit.get("restart_form") or {}
    unstamp = set(form.get("unstamp_roles") or ())
    rng = random.Random(kit["seed"] * 31 + 7)
    out = []
    for row in rows:
        row = dict(row)
        if row["role"] in unstamp:
            row.pop("timestamp", None)
        if row["role"] == "tool" and "tool_name" in row and rng.random() < form.get("tool_name_as_name_share", 0):
            row["name"] = row.pop("tool_name")
        out.append(row)
    return out


def build(kit: dict[str, Any]) -> dict[str, Any]:
    """Every conversation's history, the target's host list and the pinned identity of the input."""
    seed = kit["seed"]
    target = kit["target"]
    history = conversation(kit, "target", target["turns"], seed)
    others = [conversation(kit, f"other{index}", kit["others"]["turns"], seed + 1 + index)
              for index in range(kit["others"]["conversations"])]
    live = host_list(kit, history)
    restart = restart_view(kit, live)
    digest = hashlib.sha256(json.dumps(kit, sort_keys=True).encode())
    for rows in [history, live, restart, *others]:
        digest.update(json.dumps(rows, sort_keys=True).encode())
    return {"target_history": history, "target_list": live, "restart_list": restart, "others": others, "input_hash": digest.hexdigest(),
            "counts": counts(kit, history, live, others)}


def counts(kit: dict[str, Any], history: list[dict[str, Any]], live: list[dict[str, Any]],
           others: list[list[dict[str, Any]]]) -> dict[str, Any]:
    every = [history, *others]
    threshold = kit["externalization_threshold_chars"]
    return {
        "rows": sum(map(len, every)),
        "conversations": len(every),
        "tool_pairs": sum(row["role"] == "tool" for rows in every for row in rows),
        "externalizable_payloads": sum(row["role"] == "tool" and len(row["content"]) > threshold
                                       for rows in every for row in rows),
        "target_history_rows": len(history),
        "target_list_rows": len(live),
        "target_hidden_rows": len(history) - len(live),
        "target_list_tokens": sum(len(row["content"]) for row in live) // CHARS_PER_TOKEN,
    }
