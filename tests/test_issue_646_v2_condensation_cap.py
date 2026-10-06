"""#646 R1/R2: v2 reasoning room and condensation prompt contracts; no model calls."""

import re
import sys
from types import ModuleType, SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.escalation import _build_l1_prompt, summarize_with_escalation

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
