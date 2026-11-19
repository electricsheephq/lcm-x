"""#2896: synthetic no-system-row survival-fit cases shared by the focused cells and the two-tree sweep.

Imports only the public engine surface that main has too, so tests/behaviour/notice_fit_sweep.py can run the same cases
on a copy of main and on this tree and compare them row by row. The shapes, seeds and cap sweeps are the review
probes' (the band sweep and the direct-fit sweep of PR 2961's delta review). Synthetic data only.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import sys
import types

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

OBJ = "[Current user objective preserved from compacted history]"
BLOCK_RE = re.compile(r"\n\n<lcm-survival-notice digest=[0-9a-f]{16}>\n.*?\n</lcm-survival-notice>", re.S)
OPEN = "<lcm-survival-notice digest="
SHAPES = ["carrier", "carrier_sep2", "standalone", "carrier_objective", "two_carriers", "nohead", "twofits"]
BAND_SHAPES = [(2, "\n\n"), (2, "\n\n---\n\n"), (3, "\n\n"), (4, "\n\n---\n\n")]
PASTED = ("\n\n<lcm-survival-notice digest=0123456789abcdef>\n[LCM survival fit: 3 earlier messages (store ids 1..3) are "
          "stored verbatim but not in live context; lcm_grep / lcm_load_session reach them.]\n</lcm-survival-notice>")


OLD_NOTICE = ("[LCM survival fit: 2 earlier messages (store ids 1..2) are stored verbatim but not in live context; "
              "lcm_grep / lcm_load_session reach them.]")


def marked_block(parts: str, node_ids, notice: str) -> str:
    """The fit's marked block for ``notice`` after ``parts``, built here so a tree without #2896 can be handed one."""
    digest = hashlib.sha256(json.dumps([parts, [int(n) for n in node_ids], notice]).encode("utf-8")).hexdigest()[:16]
    return f"\n\n<lcm-survival-notice digest={digest}>\n{notice}\n</lcm-survival-notice>"


def rough(messages) -> int:
    return sum(4 + (len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or ""))) // 4
               for m in messages)


def install_host_estimator() -> None:
    """The host's rough token estimator, as the reference host has it."""
    module = types.ModuleType("agent.model_metadata")
    module.estimate_messages_tokens_rough = rough
    sys.modules["agent.model_metadata"] = module


class Pattern:
    """A stdlib stand-in for a compiled ignore pattern (the optional ``regex`` package takes ``timeout``)."""

    def __init__(self, text):
        self.pattern = re.compile(text)

    def search(self, text, timeout=None):
        return self.pattern.search(text)


def engine(path, patterns=()) -> LCMEngine:
    made = LCMEngine(config=LCMConfig(fresh_tail_count=4, leaf_chunk_tokens=1600, context_threshold=0.75,
                                      database_path=str(path / "lcm.db"), threshold_full_sweep_enabled=True))
    made.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    if patterns:
        made._compiled_ignore_message_patterns = [Pattern(text) for text in patterns]
    return made


def words(rng, n: int) -> str:
    return " ".join(rng.choice(["alpha", "beta", "gamma", "delta", "eps", "zeta"]) for _ in range(n))


def part(e, text: str) -> str:
    node = e._dag.add_node(SummaryNode(session_id="S", summary=text, token_count=1, source_token_count=1,
                                       source_ids=[1, 2], expand_hint="turns"))
    return f"[Recent Summary (d0, node {node})]\n{text}\n[Expand for details: turns]"


def build_case(e, rng, shape: str):
    """Stores the turns and returns the active list (fresh dicts): the review's direct-fit shapes."""
    n = rng.randint(4, 9)
    listy = shape == "standalone"
    turns = []
    for i in range(n):
        text = f"[U{i}] " + words(rng, rng.randint(5, 300))
        user = {"role": "user", "content": [{"type": "text", "text": text}] if listy else text, "timestamp": 10.0 * i}
        reply = {"role": "assistant", "content": f"[A{i}] " + words(rng, rng.randint(5, 600))}
        extra = ([{"role": "user", "content": f"[U{i}b] " + words(rng, 40), "timestamp": 10.0 * i + 1}]
                 if shape == "two_carriers" and i == 0 else [])
        for message in (user, *extra, reply):
            e._store.append("S", dict(message), conversation_id="conv")
        turns += [user, *extra, reply]
    summary = part(e, "summary one " + words(rng, rng.randint(3, 80)))
    summary += "\n\n---\n\n" + part(e, "summary two " + words(rng, 20)) if shape == "carrier_sep2" else ""
    first = turns[0]
    if shape in ("carrier", "carrier_sep2", "twofits"):
        sep = rng.choice(["\n\n", "\n\n---\n\n"])
        return [{"role": "user", "content": f"{summary}{sep}{first['content']}"}, *[dict(m) for m in turns[1:]]]
    if shape == "standalone":
        return [{"role": "user", "content": summary}, *[dict(m) for m in turns]]
    if shape == "carrier_objective":
        objective = {"role": "user", "content": f"{OBJ}\n" + words(rng, 10)}
        return [{"role": "user", "content": f"{summary}\n\n{first['content']}"}, objective, *[dict(m) for m in turns[1:]]]
    if shape == "two_carriers":
        second = part(e, "summary b " + words(rng, 15))
        return [{"role": "user", "content": f"{summary}\n\n{first['content']}"},
                {"role": "user", "content": f"{second}\n\n{turns[1]['content']}"}, *[dict(m) for m in turns[2:]]]
    return [dict(m) for m in turns]


def build_band(e, nturns: int, sep: str):
    """The band sweep's list: [C(summary + U0), A0, U1, A1, ...], every turn stored."""
    turns = []
    for i in range(nturns):
        user = {"role": "user", "content": f"[U{i}] " + "user words here " * (20 + 7 * i), "timestamp": 10.0 * i}
        reply = {"role": "assistant", "content": f"[A{i}] " + "reply words there " * (30 + 5 * i)}
        for message in (user, reply):
            e._store.append("S", dict(message), conversation_id="conv")
        turns += [user, reply]
    node = e._dag.add_node(SummaryNode(session_id="S", summary="summary of older turns " * 15, token_count=1,
                                       source_token_count=1, source_ids=[1, 2], expand_hint="turns"))
    summary = f"[Recent Summary (d0, node {node})]\n" + "summary of older turns " * 15 + "\n[Expand for details: turns]"
    return [{"role": "user", "content": f"{summary}{sep}{turns[0]['content']}"}, *[dict(m) for m in turns[1:]]]


def band_caps(full: int):
    return [cap for cap in range(max(50, full - 1500), full) if not cap % 3]


def d1_caps(seed: int, full: int, n_caps: int = 45):
    rng = random.Random(seed + 7)
    lo = max(50, full // 5)
    caps = sorted(set([rng.randint(lo, full - 1) for _ in range(n_caps - 25)]
                      + list(range(rng.randint(lo, full - 1), full, 9))[:25]))
    return caps[:n_caps]


def merge(messages):
    """The reference host's role-alternation repair: consecutive string user rows joined with a blank line."""
    out = []
    for message in messages:
        if (out and message["role"] == "user" == out[-1]["role"] and isinstance(message.get("content"), str)
                and isinstance(out[-1].get("content"), str)):
            out[-1] = {**out[-1], "content": out[-1]["content"] + "\n\n" + message["content"]}
        else:
            out.append(dict(message))
    return out


def generated_blocks(result, given):
    """The notice blocks the fit wrote: those in ``result`` that no input row holds (a pasted block or an earlier
    fit's block is the row's own bytes, kept as they are)."""
    seen = [str(m.get("content")) for m in given]
    return sorted({block for m in result for block in BLOCK_RE.findall(str(m.get("content")))
                   if not any(block in text for text in seen)})


def without(content, generated):
    if not isinstance(content, str):
        return content
    for block in generated:
        content = content.replace(block, "")
    return content


def rows_of(messages, generated=()):
    """Each row as its raw JSON with only the fit's own blocks left out: the row-by-row comparison against main."""
    return [json.dumps({**m, "content": without(m.get("content"), generated)}, sort_keys=True, default=str)
            for m in messages]


def nodes_shown(e, messages):
    return sorted({n for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)
                   for n in e._verified_lcm_summary_prefix(m["content"])[1]})


VOLATILE = {"store_id", "created_at", "ingested_at", "id", "observed_at", "timestamp"}  # the store's clock


def stored(e, generated=()):
    """The session's stored rows, raw (every field but the store's own clocks and ids), the fit's blocks left out;
    a row that holds a summary (the only kind that can hold the fit's block) also leaves out its token estimate,
    which counts the block, on both trees alike."""
    out = []
    for r in e._store.get_session_messages(e._session_id, limit=100_000):
        marked = isinstance(r.get("content"), str) and r["content"].startswith(("[Recent Summary", "[Session Arc"))
        out.append(json.dumps({k: (without(v, generated) if k == "content" else v) for k, v in r.items()
                               if k not in VOLATILE and not (marked and k == "token_estimate")},
                              sort_keys=True, default=str))
    return out


def store_counts(e):
    rows = e._store.get_session_messages(e._session_id, limit=100_000)
    counts = collections.Counter((r["role"], json.dumps(r["content"])) for r in rows)
    return [len(rows), sum(v - 1 for v in counts.values() if v > 1)]


def fit(e, active, cap: int, after_exception: bool):
    """One survival fit at ``cap``; the record the comparison reads. ``fitted``: the fit returned a new list (a no-op
    returns the list it was given)."""
    if not after_exception:
        e._ingest_cursor, e._ingest_cursor_needs_reconcile = len(active), False
    given = list(active)
    result = e._survival_fit(active, given, 0, "test", after_exception=after_exception, request_cap=cap)
    done = e._last_survival_fit or {}
    emission = getattr(e, "_survival_fit_emission", None) if result is not given else None
    generated = generated_blocks(result, active)
    return result, {
        "fitted": result is not given, "rows": rows_of(result, generated), "nodes": nodes_shown(e, result),
        "generated": generated, "blocks": [sum(block in str(m.get("content")) for block in generated) for m in result],
        "dropped": done.get("dropped_rows") if result is not given else None,
        "notice_in_list": bool(done.get("notice")) and any(str(done["notice"]) in str(m.get("content")) for m in result),
        "emission_row": None if emission is None else next((i for i, m in enumerate(result) if m is emission["row"]), -1),
        "emission_span": None if emission is None else without(emission.get("span") or "", generated),
    }
