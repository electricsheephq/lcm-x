"""The budget pass reserves the final sanitizer's missing-result stubs (#1048).

A tail whose newest assistant turn has tool calls without results gets one plain stub per call
from the final sanitizer. The forced-recovery path already reserved them; the normal path did
not, so a full summary budget overflowed after sanitizing and the over-cap pass dropped the
preserved user objective first. With the reservation, the selector drops its lowest-priority
summary part instead and the objective stays.
"""

import importlib.util
import logging
import os
from pathlib import Path
import subprocess
import sys

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.dag import SummaryNode
from hermes_lcm.tokens import count_messages_tokens
from tests.test_active_tool_stubbing import make_engine as make_engine  # noqa: F401 (fixture)
from tests.test_active_tool_stubbing import tool_pair


SEPARATOR = "\n\n---\n\n"
OBJECTIVE = "Deliver the current user's requested objective: repair the orchard map."
SYSTEM = {"role": "system", "content": "system"}


@pytest.fixture
def main_engine_module(tmp_path):
    """The exact origin/main implementation, loaded without touching the worktree."""
    source = subprocess.run(
        ["git", "show", "origin/main:engine.py"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    path = tmp_path / "main_engine.py"
    path.write_text(source, encoding="utf-8")
    name = "hermes_lcm._overflow_reserve_main_engine"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def setup_case(engine, *, calls=1, words=100, results=False):
    """Summaries that exactly fill the cap, plus a tail turn whose calls may lack results."""
    engine._pending_context_anchor_messages = [{"role": "user", "content": OBJECTIVE}]
    nodes = []
    for index, (depth, tag) in enumerate(((2, "DEEPEST"), (0, "SHALLOW_OLD"), (0, "SHALLOW_NEW")), 1):
        node = SummaryNode(
            session_id=engine._session_id,
            depth=depth,
            summary=tag + " " + "orchard " * words,
            token_count=words + 1,
            source_token_count=words + 1,
            source_ids=[],
            source_type="messages",
            created_at=float(index),
            earliest_at=float(index),
            latest_at=float(index),
            expand_hint="recover stored context",
        )
        node.node_id = engine._dag.add_node(node)
        nodes.append(node)
    pair = tool_pair("call-0", "result body")
    assistant = pair[0]
    assistant["tool_calls"] = [
        {**assistant["tool_calls"][0], "id": f"call-{index}"} for index in range(calls)
    ]
    tail = [assistant]
    if results:
        tail += [{"role": "tool", "tool_call_id": f"call-{index}", "content": "result body"}
                 for index in range(calls)]
    prefix = SEPARATOR.join([
        engine._build_preserved_objective_summary_part({"role": "user", "content": OBJECTIVE}),
        *(lcm_engine._summary_part_text(node) for node in nodes),
    ])
    # The cap admits every part plus the raw tail, so only unreserved sanitizer stubs overflow.
    cap = count_messages_tokens([SYSTEM, {"role": "user", "content": prefix}, *tail])
    kwargs = dict(assembly_cap_override=cap, include_lcm_note=False)
    return tail, cap, kwargs


def view(result):
    return "\n".join(str(message.get("content", "")) for message in result)


def test_objective_survives_when_calls_lack_results(make_engine, main_engine_module, caplog):  # noqa: F811
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    tail, cap, kwargs = setup_case(engine, calls=3)
    assemble = (main_engine_module.LCMEngine._assemble_context
                if os.environ.get("LCMX_OVERFLOW_ENGINE_BASELINE") == "1"
                else lcm_engine.LCMEngine._assemble_context)
    with caplog.at_level(logging.WARNING):
        result = assemble(engine, SYSTEM, tail, **kwargs)

    assert OBJECTIVE in view(result)
    assert "DEEPEST" in view(result)
    assert count_messages_tokens(result) <= cap
    # The selector, not the over-cap pass, made room: one depth-0 part is gone.
    assert ("SHALLOW_OLD" in view(result)) + ("SHALLOW_NEW" in view(result)) == 1
    assert "dropping the preserved user objective" not in caplog.text


def test_complete_tool_pairs_are_byte_identical_to_main(make_engine, main_engine_module):  # noqa: F811
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    tail, _cap, kwargs = setup_case(engine, calls=2, results=True)
    ours = lcm_engine.LCMEngine._assemble_context(engine, SYSTEM, tail, **kwargs)
    main = main_engine_module.LCMEngine._assemble_context(engine, SYSTEM, tail, **kwargs)
    assert ours == main
    assert OBJECTIVE in view(ours)


def test_last_resort_strip_still_happens_and_warns(make_engine, monkeypatch, caplog):  # noqa: F811
    """An overflow from a source the budget cannot see keeps today's last resort, now logged."""
    engine = make_engine(large_output_active_replay_stubbing_enabled=False)
    tail, cap, kwargs = setup_case(engine, calls=2, results=True)
    original = lcm_engine.LCMEngine._sanitize_active_context_messages

    def inflating(self, messages, **options):
        cleaned = original(self, messages, **options)
        if options.get("insert_missing_tool_stubs", True) and any(
            lcm_engine._PRESERVED_OBJECTIVE_CONTEXT_PREFIX in str(m.get("content", "")) for m in cleaned
        ):
            # Simulate an unbudgeted final-pass addition on top of a full budget.
            cleaned = [*cleaned, {"role": "user", "content": "unbudgeted " * 400}]
        return cleaned

    monkeypatch.setattr(lcm_engine.LCMEngine, "_sanitize_active_context_messages", inflating)
    with caplog.at_level(logging.WARNING):
        result = lcm_engine.LCMEngine._assemble_context(engine, SYSTEM, tail, **kwargs)

    assert OBJECTIVE not in view(result)
    assert "dropping the preserved user objective" in caplog.text
    assert "unbudgeted" not in caplog.text  # the warning carries counts, never content
