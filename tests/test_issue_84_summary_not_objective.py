"""#84: LCM never presents its own summary row as "the current user objective".

One long tool run that crosses several compactions with no new user message:

- compaction 1 carries the real user request in the objective scaffold;
- compaction 2 carries summaries only (pinned in test_lcm_engine.py,
  ``test_compress_does_not_reanchor_preserved_user_request_across_repeated_compaction``);
- compaction 3 used to scan back past the objective-free list, find LCM's own
  summary row (role=user), and re-render it as "[Current user objective ...]",
  followed by the same summaries again.

The anchor scan now stops at a DAG-verified summary row, the way it already stops
at a previous objective scaffold. A real user row found earlier in the scan still wins.
"""

import re

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.reconcile import _PRESERVED_OBJECTIVE_CONTEXT_PREFIX as OBJ

LATEST = "please rename the billing table to invoices_v2"
NEWER = "now add an index on invoices_v2.customer_id"
SUMMARY_HEADER = re.compile(r"\[Recent Summary \(d\d+, node \d+\)\]")


def _tool_turn(i):
    return [
        {
            "role": "assistant",
            "content": f"step {i}",
            "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": f"c{i}", "content": f"ok {i}"},
    ]


def _engine(tmp_path, monkeypatch, name, **config):
    cfg = LCMConfig(leaf_chunk_tokens=1, database_path=str(tmp_path / f"{name}.db"), **config)
    engine = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    engine.on_session_start(name, platform="discord", conversation_id=f"discord:{name}", context_length=200000)
    monkeypatch.setattr(
        lcm_engine,
        "summarize_with_escalation",
        lambda **kw: ("Older work.\nExpand for details about: x", 1),
    )
    return engine


def _opening(with_system):
    head = [{"role": "system", "content": "sys"}] if with_system else []
    return head + [
        {"role": "user", "content": "build the reporting dashboard"},
        {"role": "assistant", "content": "Dashboard built."},
        {"role": "user", "content": LATEST},
    ]


def _summary_as_objective(out):
    return [m for m in out if str(m.get("content") or "").startswith(OBJ + "\n[Recent Summary")]


@pytest.mark.parametrize("with_system", [True, False])
@pytest.mark.parametrize("tail", [4, 32])
def test_no_compaction_labels_a_summary_as_the_user_objective(tmp_path, monkeypatch, with_system, tail):
    engine = _engine(tmp_path, monkeypatch, f"t84a_{int(with_system)}_{tail}", fresh_tail_count=tail)
    per = tail // 2 + 4
    msgs = _opening(with_system)
    k = 0
    for _ in range(per):
        msgs += _tool_turn(k)
        k += 1
    outs = []
    try:
        result = engine.compress(msgs)
        outs.append(result)
        for _ in range(3):
            nxt = list(result)
            for _ in range(per):
                nxt += _tool_turn(k)
                k += 1
            result = engine.compress(nxt)
            outs.append(result)
    finally:
        engine.shutdown()

    for n, out in enumerate(outs, 1):
        assert not _summary_as_objective(out), f"compaction {n} labelled a summary as the user objective"
        headers = [h for m in out for h in SUMMARY_HEADER.findall(str(m.get("content") or ""))]
        assert len(headers) == len(set(headers)), f"compaction {n} repeated a summary: {headers}"

    # Compaction 1 still carries the real request in the objective scaffold.
    first = [m for m in outs[0] if str(m.get("content") or "").startswith(OBJ)]
    assert len(first) == 1 and LATEST in first[0]["content"]


def test_default_settings_first_compaction_keeps_the_real_request(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch, "t84b")
    assert engine._config.fresh_tail_count == 24 and engine._config.max_assembly_tokens == 0
    msgs = _opening(False)
    for i in range(20):
        msgs += _tool_turn(i)
    try:
        result = engine.compress(msgs)
    finally:
        engine.shutdown()
    users = [m for m in result if m.get("role") == "user"]
    assert len(users) == 1
    assert users[0]["content"].startswith(OBJ) and LATEST in users[0]["content"]


def test_a_newer_real_user_request_still_anchors_after_summaries(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch, "t84c", fresh_tail_count=4)
    per = 6
    msgs = _opening(False)
    k = 0
    for _ in range(per):
        msgs += _tool_turn(k)
        k += 1
    try:
        result = engine.compress(msgs)
        nxt = list(result)
        for _ in range(per):
            nxt += _tool_turn(k)
            k += 1
        result = engine.compress(nxt)
        nxt = list(result) + [{"role": "user", "content": NEWER}]
        for _ in range(per):
            nxt += _tool_turn(k)
            k += 1
        result = engine.compress(nxt)
    finally:
        engine.shutdown()
    anchors = [m for m in result if str(m.get("content") or "").startswith(OBJ)]
    assert len(anchors) == 1
    assert NEWER in anchors[0]["content"]
    assert not _summary_as_objective(result)
