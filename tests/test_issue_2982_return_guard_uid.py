"""#2982 return guard with host message uids ON (the reference agent's mode): every row carries its host uid and a host
state.db names the session, so uid recognition runs where the tree has it (on the release line it can drop covered rows
and substitute a placeholder's stored text). The guard must still hold the list, and no stored output may be summarised
twice, by store id or by CONTENT (the summariser prompts are read: an output already summarised must not reappear)."""

from __future__ import annotations

import re
import sqlite3

import pytest

import hermes_lcm.escalation as escalation
import hermes_lcm.host_uid_emit as host_uid_emit
from hermes_lcm.config import LCMConfig, host_message_uid_mode
from hermes_lcm.engine import LCMEngine

from tests.test_issue_2982_pass_never_grows import CLEARED, QUIET, _summary, _turn, host_estimate

pytestmark = pytest.mark.issue(2982)

PROMPTS: list[str] = []
OUT = re.compile(r"\[OUT(\d+)\]")


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    assert host_message_uid_mode() == "on"
    PROMPTS.clear()

    def summarise(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        PROMPTS.append(prompt)
        return _summary(max_tokens)

    monkeypatch.setattr(escalation, "_invoke_summary_llm", summarise)
    with sqlite3.connect(tmp_path / "state.db") as db:
        db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, "
                   "end_reason TEXT, source TEXT, model_config TEXT)")
        db.execute("INSERT OR IGNORE INTO sessions (id) VALUES ('S')")
    instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), fresh_tail_count=4, **QUIET),
                         hermes_home=str(tmp_path))
    instance.on_session_start("S", platform="telegram", context_length=1_000_000, conversation_id="conv")
    yield instance
    instance.shutdown()


def _uid_turn(i, output_tokens=4_000):
    return [{**row, "message_uid": f"u{i}-{k}"} for k, row in enumerate(_turn(i, 15, output_tokens))]


def _shape(turns=40, old=32):
    stored = [{"role": "system", "content": "system prompt"}] + [r for i in range(turns) for r in _uid_turn(i)]
    shown = [r for i in range(turns) if not (i < old and i % 3 != 2) for r in _uid_turn(i)]
    tools = [k for k, r in enumerate(shown) if r["role"] == "tool"]
    for k in tools[:-2]:
        shown[k] = {**shown[k], "content": CLEARED}
    return stored, [stored[0]] + shown + [{"role": "user", "content": "latest ask", "message_uid": "latest"}]


def _outputs(prompts):
    return [int(n) for p in prompts for n in OUT.findall(p)]


def _twice(engine):
    seen, twice = set(), set()
    for node in engine._dag.get_session_nodes("S"):
        if node.depth == 0:
            for sid in node.source_ids:
                (twice if sid in seen else seen).add(sid)
    return sorted(twice)


@pytest.mark.parametrize("output_tokens", [4_000, 150], ids=["full-new-turns", "small-new-turns"])
def test_uid_on_guard_holds_and_never_summarises_an_output_twice(engine, output_tokens):
    stored, view = _shape()
    engine.ingest(stored)
    pressure = engine.threshold_tokens + 20_000
    comparison_branch = []
    guard = getattr(engine, "_guard_uid_covered_drop", None)
    if guard is not None:  # the release line: record whether the stub-free comparison list was used
        def spy(result):
            out = guard(result)
            comparison_branch.append(out is not result)
            return out
        engine._guard_uid_covered_drop = spy
    host, steps, first_smaller, seen = view, [], None, []
    for step in range(12 if output_tokens > 1_000 else 30):
        start = len(PROMPTS)
        result = engine.compress(host, current_tokens=pressure)
        outputs = _outputs(PROMPTS[start:])
        repeats = sorted(set(outputs) & set(seen))
        seen.extend(outputs)
        steps.append([step + 1, host_estimate(host), host_estimate(result), result is host, len(PROMPTS) - start,
                      repeats])
        if first_smaller is None and host_estimate(result) < host_estimate(host):
            first_smaller = step + 1
        host = result
        engine.note_turn_complete()
        host = host + _uid_turn(300 + step, output_tokens) + [{"role": "assistant", "content": f"done {step}",
                                                              "message_uid": f"d{step}"}]
        engine.ingest(host)
    print("REPORT uid-on", output_tokens, {"N": first_smaller, "uid_recognition": bool(
        getattr(engine, "_uid_recognition_enabled", lambda: False)()), "comparison_branch_used": any(comparison_branch),
        "held_back": engine.get_status()["held_back_returns"], "steps": steps})
    assert all(s[2] <= s[1] for s in steps), steps  # never larger by the host's estimator
    assert not any(s[5] for s in steps), steps  # no output text summarised twice
    assert not _twice(engine), steps
