"""Hermes 0.21.x call/result shapes, matching the issue-1220 regression."""

import random


def user(text):
    return {"role": "user", "content": text}


def assistant(text, calls=()):
    row = {"role": "assistant", "content": text}
    if calls:
        row["tool_calls"] = list(calls)
    return row


def call(id, name="probe", arguments="{}"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


def tool(call_id, text, name="probe"):
    return {"role": "tool", "tool_call_id": call_id, "content": text,
            "name": name, "tool_name": name}


def prose(chars, seed=0, *, sentinel=""):
    """Return exactly ``chars`` characters including the optional sentinel prefix."""
    if chars < len(sentinel):
        raise ValueError("chars must fit the sentinel prefix")
    rng = random.Random(seed)
    words = ("inspect", "source", "record", "reply", "context", "preserve", "check", "result")
    chunks = [sentinel]
    length = len(sentinel)
    while length < chars:
        chunk = rng.choice(words) + " "
        chunks.append(chunk)
        length += len(chunk)
    return "".join(chunks)[:chars]


def sequential_pairs(n, chars):
    """One user request followed by n assistant-call/tool-result pairs.

    Both the assistant prose and each tool result contain ``chars`` characters.
    """
    rows = [user("Run the probes sequentially.")]
    for i in range(n):
        rows.extend([assistant(prose(chars, seed=2 * i), calls=[call(f"c{i}")]),
                     tool(f"c{i}", prose(chars, seed=2 * i + 1))])
    return rows


def parallel_batch(n, chars):
    """One user request, one empty assistant call batch, and n sized results."""
    return [user("Run the probes in parallel."),
            assistant("", calls=[call(f"c{i}") for i in range(n)]),
            *[tool(f"c{i}", prose(chars, seed=i)) for i in range(n)]]
