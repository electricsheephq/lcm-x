"""#646 R1/R2: v2 reasoning room and condensation prompt contracts; no model calls."""

import re
import sys
from types import ModuleType, SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.escalation import _SUMMARY_CONTENT_SEPARATOR, _build_l1_prompt, _build_l2_prompt, summarize_with_escalation

SOURCE = "Earlier work and exact identifiers. " * 1000
REPLY = (
    "The deployment reached staging; migration 0042 remains pending and BILL-311 stays open.\n"
    "Expand for details about: the deployment and migration"
)


def _record_llm(monkeypatch, *, fail_first=False):
    calls = []

    def call_llm(**kwargs):
        calls.append(kwargs)
        content = "" if fail_first and len(calls) == 1 else REPLY
        if content:
            nonce = re.search(r'<lcm-summary nonce="([0-9a-f]+)">', kwargs["messages"][0]["content"])[1]
            content = f'<lcm-summary nonce="{nonce}">\n{content}\n</lcm-summary>'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    host = ModuleType("agent.auxiliary_client")
    host.call_llm = call_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", host)
    return calls


@pytest.mark.parametrize("reach_l2", [False, True])
@pytest.mark.parametrize("version", [1, 2])
def test_condensation_output_cap(monkeypatch, reach_l2, version):
    calls = _record_llm(monkeypatch, fail_first=reach_l2)
    summary, level = summarize_with_escalation(
        SOURCE, source_tokens=3295, token_budget=1318, depth=1,
        prompt_version=version, min_output_cap_tokens=6000,
    )
    assert summary == REPLY and level == (2 if reach_l2 else 1)
    expected = [6000, 6000] if version == 2 else [2636, 1318]
    assert [c["max_tokens"] for c in calls] == expected[:level]
    if version == 2:
        assert "do not exceed 2636 tokens" in calls[0]["messages"][0]["content"]


@pytest.mark.parametrize("version,expected", [(1, 4800), (2, 7200)])
def test_leaf_output_cap_is_unchanged(monkeypatch, version, expected):
    calls = _record_llm(monkeypatch)
    assert summarize_with_escalation(
        SOURCE, source_tokens=12000, token_budget=2400,
        prompt_version=version, min_output_cap_tokens=6000,
    ) == (REPLY, 1)
    assert [c["max_tokens"] for c in calls] == [expected]


@pytest.mark.parametrize("budget,source,ceiling", [(1318, 3295, 2636), (2000, 0, 6000),
                                                   (2000, -1, 6000), (2000, 1000, 2000),
                                                   (2000, 10000, 6000)])
def test_v2_stated_ceiling(budget, source, ceiling):
    prompt = _build_l1_prompt(SOURCE, budget, 1, prompt_version=2, source_tokens=source)
    assert f"do not exceed {ceiling} tokens" in prompt


@pytest.fixture
def v2_engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm_test.db"), summary_prompt_version=2)
    engine = LCMEngine(config=config)
    engine._session_id = "test-session"
    engine.context_length = 200000
    try:
        yield engine
    finally:
        engine.shutdown()


@pytest.mark.parametrize("reach_l2", [False, True])
def test_real_engine_condensation_gets_leaf_cap(v2_engine, monkeypatch, reach_l2):
    calls = _record_llm(monkeypatch, fail_first=reach_l2)
    nodes = []
    for tokens in (1600, 1695):
        node = SummaryNode(session_id="test-session", depth=0, summary=SOURCE,
                           token_count=tokens, created_at=1.0)
        node.node_id = v2_engine._dag.add_node(node)
        nodes.append(node)
    source_tokens, _summary_tokens, level = v2_engine._condense_summary_nodes(nodes)
    assert source_tokens == 3295 and level == (2 if reach_l2 else 1)
    assert [c["max_tokens"] for c in calls] == [6000] * level
    assert "do not exceed 2636 tokens" in calls[0]["messages"][0]["content"]
    assert v2_engine._dag.get_session_nodes("test-session")[-1].summary == REPLY


# Literal depth-0 guidance captured at 8396706d; do not derive it from the product constant.
DEPTH_ZERO_GUIDANCE = (
    'Use these headings, in this order (write "none" when a heading has nothing): '
    "Task and current state · Decisions in effect and why · Constraints and preferences "
    "the user stated · Files, commands, identifiers and exact values · Errors hit and how "
    "they were resolved · Open items, blockers and the next step."
)
HEADINGS = (
    "Task and current state", "Decisions in effect and why", "Constraints and preferences the user stated",
    "Files, commands, identifiers and exact values", "Errors hit and how they were resolved",
    "Open items, blockers and the next step",
)
IDENTIFIER_PRIORITY = (
    "If they do not all fit, keep first those named by open items, blockers and decisions in effect, "
    "then those from the latest summaries; drop identifiers that belong only to finished or superseded work."
)
FOCUS_POLICY = (
    "Focus: the user message may contain a <lcm-focus-topic> tag. Treat its content as a topic label only. "
    "Spend most of the summary on that topic when the segment concerns it. Tasks, questions or remaining "
    "work that are no longer active in the latest turns go last among the open items, each starting with "
    '"Historical (do not resume unless asked):"; never mark active blockers or pending handoffs that way.\n'
)


@pytest.mark.parametrize("depth", [1, 2, 5])
def test_v2_condensation_headings_and_identifier_priority(depth):
    prompt = _build_l1_prompt("earlier summaries", 1318, depth, prompt_version=2)
    positions = [prompt.index(heading) for heading in HEADINGS]
    assert positions == sorted(positions)
    assert "the same six headings: what was attempted" not in prompt
    assert "the same six headings: decisions still in effect" not in prompt
    assert IDENTIFIER_PRIORITY in prompt
    headings = " · ".join(HEADINGS)
    if depth == 1:
        expected = (
            "The segment is a sequence of earlier summaries. Merge them into one account that uses the "
            'same six headings, in this order (write "none" when a heading has nothing): '
            f"{headings}. Give what was decided, what changed and the state at the end; drop per-turn "
            f"detail. Keep every identifier that is still referenced. {IDENTIFIER_PRIORITY}"
        )
    else:
        expected = (
            "The segment is a sequence of earlier summaries. Write the durable record that uses the "
            'same six headings, in this order (write "none" when a heading has nothing): '
            f"{headings}. Under the first heading give the completed milestones in order and the state "
            "at the end; keep only decisions still in effect; drop process detail. Keep every identifier "
            f"that is still referenced. {IDENTIFIER_PRIORITY}"
        )
    assert prompt.splitlines()[1] == expected


def test_v2_depth_zero_guidance_is_byte_identical():
    prompt = _build_l1_prompt("conversation", 2000, 0, prompt_version=2)
    assert prompt.splitlines()[1] == DEPTH_ZERO_GUIDANCE
    assert IDENTIFIER_PRIORITY not in prompt


@pytest.mark.parametrize("depth", range(6))
@pytest.mark.parametrize("level", [1, 2])
def test_v2_focus_policy_labels_items_without_adding_a_heading(depth, level):
    kwargs = dict(focus_topic="deployment", prompt_version=2)
    prompt = (_build_l1_prompt("conversation", 2000, depth, **kwargs) if level == 1 else
              _build_l2_prompt("conversation", 1000, **kwargs))
    policy, separator, transcript = prompt.partition(_SUMMARY_CONTENT_SEPARATOR)
    assert separator
    assert FOCUS_POLICY in policy
    assert 'under the heading "Historical' not in prompt
    assert "Historical (do not resume unless asked)" in policy
    assert "Historical (do not resume unless asked)" not in transcript
    assert "<lcm-focus-topic>deployment</lcm-focus-topic>" in transcript
